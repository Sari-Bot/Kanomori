"""Stage: OCR — extract Japanese text visible in sampled frames."""

from __future__ import annotations

import json
from dataclasses import dataclass
from pathlib import Path

from kanomori.text import tokenize_for_fts


@dataclass(frozen=True)
class OcrResult:
    text: str
    confidence: float | None = None
    bbox: dict | list | None = None


_READER = None


def _reader():
    global _READER
    if _READER is None:
        from rapidocr_onnxruntime import RapidOCR

        _READER = RapidOCR()
    return _READER


def read_frame_ocr(path: Path) -> list[OcrResult]:
    raw, _elapsed = _reader()(str(path))
    out: list[OcrResult] = []
    for item in raw or []:
        bbox, text, score = item
        if text:
            out.append(OcrResult(text=text, confidence=float(score), bbox=bbox))
    return out


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
