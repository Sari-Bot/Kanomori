"""Stage: OCR — extract Japanese text visible in sampled frames."""

from __future__ import annotations

import json
from pathlib import Path

from kanomori.ocr import OcrResult, read_image_ocr
from kanomori.text import tokenize_for_fts


def read_frame_ocr(path: Path) -> list[OcrResult]:
    return read_image_ocr(path)


def run(conn, ctx):
    rows = conn.execute(
        "SELECT id, frame_path, ts_sec FROM frames WHERE video_id = %s ORDER BY ts_sec",
        (ctx.video_id,),
    ).fetchall()
    if not rows:
        return "skipped"

    conn.execute("DELETE FROM ocr_segments WHERE video_id = %s", (ctx.video_id,))
    with conn.cursor() as cur:
        for frame_id, frame_path, ts_sec in rows:
            for result in read_frame_ocr(Path(frame_path)):
                cur.execute(
                    """
                    INSERT INTO ocr_segments
                        (video_id, frame_id, ts_sec, text, confidence, bbox, tsv)
                    VALUES (%s, %s, %s, %s, %s, %s::jsonb, to_tsvector('simple', %s))
                    """,
                    (
                        ctx.video_id,
                        frame_id,
                        ts_sec,
                        result.text,
                        result.confidence,
                        json.dumps(result.bbox or {}, ensure_ascii=False),
                        tokenize_for_fts(result.text),
                    ),
                )
    return None
