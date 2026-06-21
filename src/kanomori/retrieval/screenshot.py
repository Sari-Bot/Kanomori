"""Screenshot candidate generation: pHash + DINOv2 + OCR lexical signals."""

from __future__ import annotations

from io import BytesIO
from pathlib import Path
from tempfile import NamedTemporaryFile

import numpy as np

from kanomori.embed.phash import compute_phash
from kanomori.fusion import reciprocal_rank_fusion
from kanomori.models import Candidate, Modality
from kanomori.text import tokenize_for_fts

DEFAULT_PHASH_THRESHOLD = 10


class UploadOcrReader:
    """RapidOCR wrapper for query-time screenshot text extraction."""

    def text_from_image_bytes(self, data: bytes) -> str:
        from kanomori.ingest.stages.ocr import read_frame_ocr

        with NamedTemporaryFile(suffix=".png") as tmp:
            tmp.write(data)
            tmp.flush()
            results = read_frame_ocr(Path(tmp.name))
        return " ".join(result.text for result in results)


def phash_from_bytes(data: bytes) -> int:
    from PIL import Image

    with Image.open(BytesIO(data)) as image:
        return compute_phash(image)


def phash_candidates(
    conn,
    query_hash: int,
    *,
    threshold: int = DEFAULT_PHASH_THRESHOLD,
    k: int = 50,
) -> list[Candidate]:
    rows = conn.execute(
        """
        SELECT id, video_id, ts_sec, bit_count((phash # %s::bigint)::bit(64)) AS distance
        FROM frames
        WHERE phash IS NOT NULL
          AND bit_count((phash # %s::bigint)::bit(64)) <= %s
        ORDER BY distance, id
        LIMIT %s
        """,
        (query_hash, query_hash, threshold, k),
    ).fetchall()
    return [
        Candidate(
            video_id=vid,
            ts_sec=ts,
            modality=Modality.VISUAL,
            rank=i,
            raw_score=float(distance),
            segment_id=sid,
        )
        for i, (sid, vid, ts, distance) in enumerate(rows)
    ]


def dense_candidates(conn, query_vec: np.ndarray, *, k: int = 50) -> list[Candidate]:
    rows = conn.execute(
        """
        SELECT id, video_id, ts_sec, embedding <=> %s AS distance
        FROM frames
        WHERE embedding IS NOT NULL
        ORDER BY distance
        LIMIT %s
        """,
        (query_vec, k),
    ).fetchall()
    return [
        Candidate(
            video_id=vid,
            ts_sec=ts,
            modality=Modality.VISUAL,
            rank=i,
            raw_score=float(distance),
            segment_id=sid,
        )
        for i, (sid, vid, ts, distance) in enumerate(rows)
    ]


def ocr_candidates(conn, text: str, *, k: int = 50) -> list[Candidate]:
    tokens = tokenize_for_fts(text)
    if not tokens:
        return []
    rows = conn.execute(
        """
        SELECT id, video_id, ts_sec, ts_rank(tsv, plainto_tsquery('simple', %s)) AS rank
        FROM ocr_segments
        WHERE tsv @@ plainto_tsquery('simple', %s)
        ORDER BY rank DESC
        LIMIT %s
        """,
        (tokens, tokens, k),
    ).fetchall()
    return [
        Candidate(
            video_id=vid,
            ts_sec=ts,
            modality=Modality.OCR,
            rank=i,
            raw_score=float(rank),
            segment_id=sid,
        )
        for i, (sid, vid, ts, rank) in enumerate(rows)
    ]


def _key(candidate: Candidate) -> tuple[str, int | None]:
    return (candidate.modality.value, candidate.segment_id)


def candidates(
    conn,
    image: bytes,
    embedder,
    *,
    ocr_reader=None,
    phasher=phash_from_bytes,
    k: int = 50,
) -> list[Candidate]:
    """Return screenshot candidates fused across visual hash, dense image, and OCR signals."""
    qhash = phasher(image)
    phash = phash_candidates(conn, qhash, k=k)
    dense = dense_candidates(conn, embedder.embed_image_bytes(image), k=k)
    ocr_text = ocr_reader.text_from_image_bytes(image) if ocr_reader else ""
    ocr = ocr_candidates(conn, ocr_text, k=k)

    by_key: dict[tuple[str, int | None], Candidate] = {}
    for candidate in (*phash, *dense, *ocr):
        by_key.setdefault(_key(candidate), candidate)

    fused = reciprocal_rank_fusion(
        [[_key(c) for c in phash], [_key(c) for c in dense], [_key(c) for c in ocr]], k=60
    )
    out: list[Candidate] = []
    for rank, (key, score) in enumerate(fused):
        base = by_key[key]
        out.append(base.model_copy(update={"rank": rank, "raw_score": score}))
    return out[:k]
