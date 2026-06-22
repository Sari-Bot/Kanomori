"""Integration tests for the FastAPI app (transcript search + ingest enqueue) against real DB.

Uses TestClient. The embedder is overridden with the deterministic fake (no BGE-M3 download),
and transcript rows are seeded directly so /search/transcript exercises the real retrieval
path. /ingest enqueues a job row (the worker, tested separately, would run it).
"""

from __future__ import annotations

import hashlib
from io import BytesIO

import numpy as np
import pytest
from pgvector.psycopg import register_vector

from kanomori.embed.phash import to_signed_bigint
from kanomori.text import tokenize_for_fts

pytestmark = pytest.mark.requires_db


@pytest.fixture
def client(db_conn, fake_embedder, monkeypatch):
    """A TestClient with the app's DB pool and embedder pointed at the test doubles."""
    from fastapi.testclient import TestClient

    from kanomori.api import app as app_module

    # The endpoints fetch a shared embedder via app_module.get_embedder; override it.
    monkeypatch.setattr(app_module, "get_embedder", lambda: fake_embedder)
    app = app_module.create_app()
    with TestClient(app) as c:
        yield c


@pytest.fixture
def seeded(db_conn, fake_embedder):
    """Seed one video with three JP transcript segments via the live connection."""
    from pgvector.psycopg import register_vector

    from kanomori.text import tokenize_for_fts

    register_vector(db_conn)
    vid = db_conn.execute(
        "INSERT INTO videos (content_hash, title, source_url) VALUES "
        "('apihash', 'api test', 'https://example/v') RETURNING id"
    ).fetchone()[0]
    rows = [
        (0, 0.0, 5.0, "今日はマインクラフトを遊びます"),
        (1, 5.0, 10.0, "大学で英語を勉強していました"),
        (2, 10.0, 15.0, "みなさんこんばんは"),
    ]
    for seq, start, end, text in rows:
        db_conn.execute(
            """
            INSERT INTO transcript_segments
                (video_id, seq, start_sec, end_sec, text, text_norm, embedding, tsv)
            VALUES (%s,%s,%s,%s,%s,%s,%s, to_tsvector('simple', %s))
            """,
            (vid, seq, start, end, text, text, fake_embedder.embed_query(text),
             tokenize_for_fts(text)),
        )
    db_conn.commit()
    yield vid
    db_conn.execute("DELETE FROM videos WHERE id = %s", (vid,))
    db_conn.commit()


def test_search_transcript_returns_hit_for_keyword(client, seeded) -> None:
    resp = client.post("/search/transcript", json={"query": "マインクラフト", "k": 5})
    assert resp.status_code == 200
    hits = resp.json()["hits"]
    assert hits
    assert hits[0]["video_id"] == seeded
    assert hits[0]["ts_sec"] == pytest.approx(0.0)


def test_search_transcript_returns_scene_type_from_merge_layer(client, seeded, db_conn) -> None:
    db_conn.execute(
        """
        INSERT INTO scene_segments (video_id, start_sec, end_sec, scene_type, confidence)
        VALUES (%s, 0.0, 8.0, 'chatting', 0.9)
        """,
        (seeded,),
    )
    db_conn.commit()

    resp = client.post("/search/transcript", json={"query": "マインクラフト", "k": 5})

    assert resp.status_code == 200
    hits = resp.json()["hits"]
    assert hits[0]["scene_type"] == "chatting"
    assert "transcript" in hits[0]["why"]


def test_search_transcript_out_of_corpus_query_returns_dense_neighbors(client, seeded) -> None:
    # Hybrid retrieval: a query with no lexical match still returns dense nearest neighbors
    # (cosine distance is always defined). Such hits rank low and the UI shows confidence;
    # the white paper's model is "return Top-5 candidates, users visually verify." The
    # "no lexical match -> empty" guarantee belongs to the lexical path (tested at the
    # retrieval layer), not to the hybrid endpoint.
    resp = client.post("/search/transcript", json={"query": "ドイツ語フランス語", "k": 5})
    assert resp.status_code == 200
    hits = resp.json()["hits"]
    assert len(hits) <= 5
    assert all(h["video_id"] == seeded for h in hits)


def test_ingest_enqueues_job_and_returns_id(client, tmp_path) -> None:
    media = tmp_path / "clip.mp4"
    media.write_bytes(b"api-ingest-bytes")
    resp = client.post("/ingest", json={"media_path": str(media), "title": "x"})
    assert resp.status_code == 200
    body = resp.json()
    assert body["status"] == "queued"
    assert isinstance(body["job_id"], int)


def test_ingest_status_reports_queued(client, tmp_path) -> None:
    media = tmp_path / "clip2.mp4"
    media.write_bytes(b"api-ingest-bytes-2")
    job_id = client.post("/ingest", json={"media_path": str(media)}).json()["job_id"]
    resp = client.get(f"/ingest/{job_id}")
    assert resp.status_code == 200
    assert resp.json()["status"] == "queued"


def test_ingest_status_404_for_unknown_job(client) -> None:
    resp = client.get("/ingest/99999999")
    assert resp.status_code == 404


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


def test_search_screenshot_returns_visual_timestamp(client, db_conn, monkeypatch) -> None:
    from PIL import Image

    from kanomori.api import app as app_module

    buf = BytesIO()
    Image.new("RGB", (8, 8), color=(200, 120, 20)).save(buf, format="PNG")
    image = buf.getvalue()
    image_embedder = FakeImageEmbedder()
    monkeypatch.setattr(app_module, "get_image_embedder", lambda: image_embedder)
    monkeypatch.setattr(app_module, "get_ocr_reader", lambda: FakeOcrReader())
    register_vector(db_conn)
    vid = db_conn.execute(
        "INSERT INTO videos (content_hash, title) VALUES ('api-screenhash', 'screen') RETURNING id"
    ).fetchone()[0]
    frame_id = db_conn.execute(
        """
        INSERT INTO frames (video_id, ts_sec, frame_path, phash, embedding)
        VALUES (%s, 31.0, 'media/api-screenhash/frames/frame_000031_000.jpg', %s, %s)
        RETURNING id
        """,
        (vid, to_signed_bigint(7), image_embedder.embed_image_bytes(image)),
    ).fetchone()[0]
    db_conn.execute(
        """
        INSERT INTO ocr_segments (video_id, frame_id, ts_sec, text, confidence, bbox, tsv)
        VALUES (%s, %s, 31.0, '入口の看板', 0.9, '{}'::jsonb, to_tsvector('simple', %s))
        """,
        (vid, frame_id, tokenize_for_fts("入口の看板")),
    )
    db_conn.commit()

    resp = client.post(
        "/search/screenshot",
        files={"file": ("screen.jpg", image, "image/jpeg")},
        data={"k": "5"},
    )

    assert resp.status_code == 200
    hits = resp.json()["hits"]
    assert hits
    assert hits[0]["video_id"] == vid
    assert hits[0]["ts_sec"] == pytest.approx(31.0)
