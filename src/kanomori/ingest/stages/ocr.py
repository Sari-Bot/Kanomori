"""Stage: OCR — extract Japanese text visible in sampled frames.

Compute/persist split (Task #12): ``compute`` has no DB connection, so it sources frames from
disk via :func:`frames_on_disk` (the deterministic JPEGs the frames stage wrote), recovering each
frame's ``ts_sec`` from its filename. It runs OCR and returns an :class:`OcrResult` of
:class:`OcrRow` keyed by ``ts_sec``. ``persist`` resolves each row's ``frame_id`` by
``(video_id, ts_sec)`` and INSERTs, deriving ``tsv`` coordinator-side with ``to_tsvector``.
"""

from __future__ import annotations

import json
from pathlib import Path

from kanomori.ingest.artifacts import frames_on_disk
from kanomori.ingest.stage_result import OcrResult as OcrStageResult
from kanomori.ingest.stage_result import OcrRow
from kanomori.ocr import OcrResult, read_image_ocr
from kanomori.text import tokenize_for_fts


def read_frame_ocr(path: Path) -> list[OcrResult]:
    return read_image_ocr(path, scope="ingest")


def run(conn, ctx):
    result = compute(ctx)
    if result == "skipped":
        return "skipped"
    persist(conn, ctx.video_id, result)
    return None


def compute(ctx):
    """OCR every on-disk frame; return OcrResult (rows keyed by ts_sec) or "skipped".

    Frames come from :func:`frames_on_disk` — no DB read — with ts_sec recovered from each JPEG's
    deterministic name (the value the frames stage persisted). Returns the "skipped" sentinel when
    the video has no extracted frames, preserving run()'s contract.
    """
    frames = frames_on_disk(ctx.content_hash)
    if not frames:
        return "skipped"

    rows: list[OcrRow] = []
    for ts_sec, frame_path in frames:
        for result in read_frame_ocr(frame_path):
            rows.append(
                OcrRow(
                    ts_sec=ts_sec,
                    text=result.text,
                    confidence=result.confidence,
                    bbox=result.bbox,
                )
            )
    return OcrStageResult(rows=rows)


def persist(conn, video_id, result: OcrStageResult) -> None:
    """Replace this video's ocr_segments; resolve frame_id by (video_id, ts_sec), derive tsv."""
    conn.execute("DELETE FROM ocr_segments WHERE video_id = %s", (video_id,))
    frame_ids = {
        ts_sec: frame_id
        for frame_id, ts_sec in conn.execute(
            "SELECT id, ts_sec FROM frames WHERE video_id = %s", (video_id,)
        ).fetchall()
    }
    with conn.cursor() as cur:
        for row in result.rows:
            cur.execute(
                """
                INSERT INTO ocr_segments
                    (video_id, frame_id, ts_sec, text, confidence, bbox, tsv)
                VALUES (%s, %s, %s, %s, %s, %s::jsonb, to_tsvector('simple', %s))
                """,
                (
                    video_id,
                    frame_ids.get(row.ts_sec),
                    row.ts_sec,
                    row.text,
                    row.confidence,
                    json.dumps(row.bbox or {}, ensure_ascii=False),
                    tokenize_for_fts(row.text),
                ),
            )
