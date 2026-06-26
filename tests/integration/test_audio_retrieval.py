from __future__ import annotations

from pathlib import Path

import pytest
from pgvector.psycopg import register_vector

from kanomori.retrieval import audio
from kanomori.text import normalize, tokenize_for_fts

pytestmark = pytest.mark.requires_db


class FakeASR:
    def transcribe(self, _path: Path):
        return [
            {"start": 0.0, "end": 2.0, "text": "今日はゲーム配信"},
            {"start": 2.0, "end": 4.0, "text": "雑談タイム"},
        ]


def _insert_video(conn, content_hash: str) -> int:
    return conn.execute(
        "INSERT INTO videos (content_hash, title) VALUES (%s, 'audio retrieval') RETURNING id",
        (content_hash,),
    ).fetchone()[0]


def _insert_segment(conn, video_id: int, seq: int, start: float, text: str, embedder) -> int:
    register_vector(conn)
    norm = normalize(text)
    return conn.execute(
        """
        INSERT INTO transcript_segments
            (video_id, seq, start_sec, end_sec, text, text_norm, embedding, tsv)
        VALUES (%s, %s, %s, %s, %s, %s, %s, to_tsvector('simple', %s))
        RETURNING id
        """,
        (
            video_id,
            seq,
            start,
            start + 5.0,
            text,
            norm,
            embedder.embed_query(norm),
            tokenize_for_fts(norm),
        ),
    ).fetchone()[0]


def test_audio_candidates_uses_fake_asr_and_existing_transcript_index(
    db_conn, fake_embedder
) -> None:
    video_id = _insert_video(db_conn, "audiohash")
    first_id = _insert_segment(db_conn, video_id, 0, 30.0, "今日はゲーム配信", fake_embedder)
    _insert_segment(db_conn, video_id, 1, 34.0, "雑談タイム", fake_embedder)
    _insert_segment(db_conn, video_id, 2, 90.0, "別の話題", fake_embedder)
    db_conn.commit()

    transcript, hits = audio.audio_candidates(
        db_conn,
        Path("query.wav"),
        FakeASR(),
        fake_embedder,
        k=5,
        per_window_k=5,
    )

    assert transcript == "今日はゲーム配信 雑談タイム"
    assert hits
    assert hits[0].video_id == video_id
    assert hits[0].coverage >= 2
    assert any(ev.matched_segment_id == first_id for ev in hits[0].evidence)
