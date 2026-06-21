from __future__ import annotations

import hashlib

import numpy as np
import pytest
from pgvector.psycopg import register_vector

from kanomori.embed.phash import to_signed_bigint
from kanomori.models import Modality
from kanomori.retrieval import screenshot
from kanomori.text import tokenize_for_fts

pytestmark = pytest.mark.requires_db


class FakeImageEmbedder:
    def embed_image_bytes(self, data: bytes) -> np.ndarray:
        seed = int.from_bytes(hashlib.sha256(data).digest()[:8], "big")
        rng = np.random.default_rng(seed)
        v = rng.standard_normal(768).astype(np.float32)
        v /= np.linalg.norm(v) or 1.0
        return v


class FakeOcrReader:
    def text_from_image_bytes(self, data: bytes) -> str:
        return "入口の看板"


def test_screenshot_candidates_fuse_visual_hash_and_ocr_hits(db_conn) -> None:
    register_vector(db_conn)
    embedder = FakeImageEmbedder()
    ocr = FakeOcrReader()
    image = b"known-frame-image"
    phash = to_signed_bigint(0x8000000000000001)
    vid = db_conn.execute(
        "INSERT INTO videos (content_hash, title) VALUES ('screenhash', 'screen') RETURNING id"
    ).fetchone()[0]
    frame_id = db_conn.execute(
        """
        INSERT INTO frames (video_id, ts_sec, frame_path, phash, embedding)
        VALUES (%s, 42.5, 'media/screenhash/frames/frame_000042_500.jpg', %s, %s)
        RETURNING id
        """,
        (vid, phash, embedder.embed_image_bytes(image)),
    ).fetchone()[0]
    db_conn.execute(
        """
        INSERT INTO ocr_segments (video_id, frame_id, ts_sec, text, confidence, bbox, tsv)
        VALUES (%s, %s, 42.5, '入口の看板', 0.95, '{}'::jsonb, to_tsvector('simple', %s))
        """,
        (vid, frame_id, tokenize_for_fts("入口の看板")),
    )

    hits = screenshot.candidates(
        db_conn,
        image,
        embedder,
        ocr_reader=ocr,
        phasher=lambda _: phash,
        k=10,
    )

    assert hits
    assert hits[0].video_id == vid
    assert hits[0].ts_sec == pytest.approx(42.5)
    assert {hit.modality for hit in hits} >= {Modality.VISUAL, Modality.OCR}
