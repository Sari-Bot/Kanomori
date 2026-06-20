"""Integration tests for kanomori.retrieval.transcript against real PostgreSQL + pgvector.

These index real transcript rows (with JP tsvector + a deterministic embedding) and verify
the lexical path, the dense path, and the RRF-fused `candidates()` all locate the right
segment. The database performs the actual full-text and cosine math — only the embedding model
is substituted by the deterministic fake_embedder.
"""

from __future__ import annotations

import pytest

from kanomori.models import Modality
from kanomori.retrieval import transcript
from kanomori.text import tokenize_for_fts

pytestmark = pytest.mark.requires_db


def _insert_video(conn, content_hash: str = "vhash1") -> int:
    return conn.execute(
        "INSERT INTO videos (content_hash, title) VALUES (%s, %s) RETURNING id",
        (content_hash, "test stream"),
    ).fetchone()[0]


def _insert_segment(conn, video_id, seq, start, end, text, embedder) -> int:
    from pgvector.psycopg import register_vector

    register_vector(conn)
    vec = embedder.embed_query(text)
    return conn.execute(
        """
        INSERT INTO transcript_segments
            (video_id, seq, start_sec, end_sec, text, text_norm, embedding, tsv)
        VALUES (%s, %s, %s, %s, %s, %s, %s, to_tsvector('simple', %s))
        RETURNING id
        """,
        (video_id, seq, start, end, text, text, vec, tokenize_for_fts(text)),
    ).fetchone()[0]


@pytest.fixture
def indexed_video(db_conn, fake_embedder):
    """A video with three distinct JP transcript segments at different timestamps."""
    vid = _insert_video(db_conn)
    rows = [
        (0, 0.0, 5.0, "今日はマインクラフトを遊びます"),
        (1, 5.0, 10.0, "大学で英語を勉強していました"),
        (2, 10.0, 15.0, "みなさんこんばんは"),
    ]
    ids = {}
    for seq, start, end, text in rows:
        ids[seq] = _insert_segment(db_conn, vid, seq, start, end, text, fake_embedder)
    return vid, ids


def test_lexical_candidates_find_segment_by_keyword(db_conn, indexed_video) -> None:
    vid, ids = indexed_video
    cands = transcript.lexical_candidates(db_conn, "マインクラフト", k=10)
    assert cands, "expected a lexical match for マインクラフト"
    assert cands[0].segment_id == ids[0]
    assert cands[0].modality == Modality.TRANSCRIPT


def test_dense_candidates_find_exact_text_match_first(
    db_conn, indexed_video, fake_embedder
) -> None:
    vid, ids = indexed_video
    qvec = fake_embedder.embed_query("大学で英語を勉強していました")
    cands = transcript.dense_candidates(db_conn, qvec, k=10)
    assert cands[0].segment_id == ids[1]


def test_candidates_fuses_lexical_and_dense(db_conn, indexed_video, fake_embedder) -> None:
    vid, ids = indexed_video
    hits = transcript.candidates(db_conn, "大学", fake_embedder, k=10)
    assert hits
    # The 大学/英語 segment should surface at the top via at least one signal.
    assert hits[0].segment_id == ids[1]


def test_candidates_carry_video_and_timestamp(db_conn, indexed_video, fake_embedder) -> None:
    vid, ids = indexed_video
    hits = transcript.candidates(db_conn, "こんばんは", fake_embedder, k=10)
    top = hits[0]
    assert top.video_id == vid
    assert top.ts_sec == pytest.approx(10.0)  # start_sec of segment 2


def test_no_match_returns_empty(db_conn, indexed_video, fake_embedder) -> None:
    vid, ids = indexed_video
    # A lexical term absent from the corpus and a query whose vector matches nothing closely
    # still returns dense neighbors, so assert the lexical path specifically is empty.
    cands = transcript.lexical_candidates(db_conn, "ドイツ語フランス語ロシア語", k=10)
    assert cands == []
