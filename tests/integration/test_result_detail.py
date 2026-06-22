"""Integration tests for result-detail assembly (the moment-detail view backing /result).

Given a (video_id, ts_sec), result_detail gathers everything the UI shows for one moment:
video metadata + source link, nearby transcript (±window), preview frames (±window), OCR
context (±window), and the scene_type at that timestamp. Runs against real PostgreSQL.
"""

from __future__ import annotations

import pytest

from kanomori.retrieval.result import result_detail

pytestmark = pytest.mark.requires_db


@pytest.fixture
def seeded_moment(db_conn):
    """A video with transcript, frames, OCR, and a scene segment around ts=30s."""
    vid = db_conn.execute(
        "INSERT INTO videos (content_hash, title, source_url, duration_sec) "
        "VALUES ('rhash', 'detail test', 'https://example/v', 120.0) RETURNING id"
    ).fetchone()[0]
    # transcript segments at 10/30/50s
    for seq, (s, e, txt) in enumerate(
        [(10.0, 12.0, "遠い昔の話"), (30.0, 33.0, "これが本題です"), (50.0, 52.0, "また今度")]
    ):
        db_conn.execute(
            "INSERT INTO transcript_segments (video_id, seq, start_sec, end_sec, text, text_norm)"
            " VALUES (%s,%s,%s,%s,%s,%s)",
            (vid, seq, s, e, txt, txt),
        )
    # frames at 28/30/32s and a far one at 90s
    for ts in (28.0, 30.0, 32.0, 90.0):
        fid = db_conn.execute(
            "INSERT INTO frames (video_id, ts_sec, frame_path) VALUES (%s,%s,%s) RETURNING id",
            (vid, ts, f"media/rhash/frames/f_{int(ts)}.jpg"),
        ).fetchone()[0]
        if ts == 30.0:
            db_conn.execute(
                "INSERT INTO ocr_segments (video_id, frame_id, ts_sec, text) "
                "VALUES (%s,%s,%s,%s)",
                (vid, fid, ts, "画面のテキスト"),
            )
    db_conn.execute(
        "INSERT INTO scene_segments (video_id, start_sec, end_sec, scene_type, confidence) "
        "VALUES (%s,%s,%s,%s,%s)",
        (vid, 25.0, 40.0, "chatting", 0.9),
    )
    return vid


def test_result_detail_returns_video_metadata_and_source(db_conn, seeded_moment) -> None:
    d = result_detail(db_conn, seeded_moment, 30.0, window=10.0)
    assert d.video_id == seeded_moment
    assert d.title == "detail test"
    assert d.source_url == "https://example/v"
    assert d.ts_sec == pytest.approx(30.0)


def test_result_detail_includes_nearby_transcript_within_window(db_conn, seeded_moment) -> None:
    d = result_detail(db_conn, seeded_moment, 30.0, window=10.0)
    texts = [t.text for t in d.nearby_transcript]
    assert "これが本題です" in texts          # at 30s, inside window
    assert "遠い昔の話" not in texts            # at 10s, outside ±10s
    assert "また今度" not in texts              # at 50s, outside ±10s


def test_result_detail_includes_preview_frames_within_window(db_conn, seeded_moment) -> None:
    d = result_detail(db_conn, seeded_moment, 30.0, window=10.0)
    frame_ts = sorted(f.ts_sec for f in d.preview_frames)
    assert frame_ts == [28.0, 30.0, 32.0]      # 90s frame excluded
    assert all(f.frame_path for f in d.preview_frames)


def test_result_detail_includes_ocr_context(db_conn, seeded_moment) -> None:
    d = result_detail(db_conn, seeded_moment, 30.0, window=10.0)
    assert any("画面のテキスト" in o.text for o in d.ocr_context)


def test_result_detail_reports_scene_type_at_timestamp(db_conn, seeded_moment) -> None:
    d = result_detail(db_conn, seeded_moment, 30.0, window=10.0)
    assert d.scene_type == "chatting"


def test_result_detail_unknown_video_returns_none(db_conn) -> None:
    assert result_detail(db_conn, 999999, 10.0) is None
