"""Integration tests for the resumable ingestion pipeline against real PostgreSQL.

The pipeline is a staged DAG keyed by content_hash with per-stage status in jobs.stage_status,
so a crash resumes at the first non-done stage and re-ingesting the same bytes is a no-op.
These tests inject a fake KITS transcriber (no GPU) and the deterministic fake embedder, and
exercise register -> ... -> parse_transcript against the live pgvector container.
"""

from __future__ import annotations

from pathlib import Path
from types import SimpleNamespace

import pytest

from kanomori.ingest import pipeline
from kanomori.ingest.stages import register

pytestmark = pytest.mark.requires_db


FIXTURE_SRT = Path(__file__).resolve().parents[1] / "fixtures" / "sample.srt"


@pytest.fixture
def media_file(tmp_path: Path) -> Path:
    """A small stand-in media file; register hashes its bytes (content need not be real audio)."""
    p = tmp_path / "clip.mp4"
    p.write_bytes(b"fake-media-bytes-for-hashing")
    return p


@pytest.fixture
def fake_transcribe(monkeypatch):
    """Patch transcribe (drop fixture SRT) and locate_media (skip real ffmpeg) — no GPU, no
    media decoding. These tests exercise pipeline orchestration, not audio extraction (that's
    covered by test_locate_media and the end-to-end proof)."""
    def _fake_transcribe(audio_path, out_srt, **kwargs):
        out_srt = Path(out_srt)
        out_srt.parent.mkdir(parents=True, exist_ok=True)
        out_srt.write_text(FIXTURE_SRT.read_text(encoding="utf-8"), encoding="utf-8")
        return out_srt

    def _fake_locate(conn, ctx):
        ctx.audio_path = ctx.media_path  # pretend extraction happened

    monkeypatch.setattr("kanomori.ingest.stages.transcribe.kits_transcribe", _fake_transcribe)
    monkeypatch.setattr("kanomori.ingest.stages.locate_media.run", _fake_locate)
    return _fake_transcribe


def test_register_creates_video_and_job(db_conn, media_file) -> None:
    ctx = pipeline.IngestContext(media_path=str(media_file), title="t")
    result = pipeline.run_stage(db_conn, "register", ctx)
    assert result.video_id is not None
    row = db_conn.execute(
        "SELECT content_hash FROM videos WHERE id = %s", (result.video_id,)
    ).fetchone()
    assert row is not None and len(row[0]) == 64  # sha256 hex


def test_register_is_idempotent_on_same_bytes(db_conn, media_file) -> None:
    ctx = pipeline.IngestContext(media_path=str(media_file))
    r1 = pipeline.run_stage(db_conn, "register", ctx)
    r2 = pipeline.run_stage(db_conn, "register", ctx)
    assert r1.video_id == r2.video_id  # same hash -> same video row, no duplicate


def test_parse_transcript_inserts_segments_with_embeddings(
    db_conn, media_file, fake_transcribe, fake_embedder, monkeypatch
) -> None:
    monkeypatch.setattr(pipeline, "make_embedder", lambda: fake_embedder)
    ctx = pipeline.IngestContext(media_path=str(media_file))
    pipeline.run_full(db_conn, ctx)

    vid = db_conn.execute(
        "SELECT id FROM videos WHERE content_hash = %s", (ctx.content_hash,)
    ).fetchone()[0]
    rows = db_conn.execute(
        "SELECT count(*), count(embedding), count(tsv) "
        "FROM transcript_segments WHERE video_id = %s",
        (vid,),
    ).fetchone()
    # fixture has 4 cues; all should have embeddings and tsv populated
    assert rows[0] == 4
    assert rows[1] == 4
    assert rows[2] == 4


def test_full_run_marks_job_complete(
    db_conn, media_file, fake_transcribe, fake_embedder, monkeypatch
) -> None:
    monkeypatch.setattr(pipeline, "make_embedder", lambda: fake_embedder)
    ctx = pipeline.IngestContext(media_path=str(media_file))
    pipeline.run_full(db_conn, ctx)
    status = db_conn.execute(
        "SELECT status FROM jobs WHERE content_hash = %s", (ctx.content_hash,)
    ).fetchone()[0]
    assert status == "complete"


def test_rerun_skips_completed_stages(
    db_conn, media_file, fake_transcribe, fake_embedder, monkeypatch
) -> None:
    monkeypatch.setattr(pipeline, "make_embedder", lambda: fake_embedder)
    ctx = pipeline.IngestContext(media_path=str(media_file))
    pipeline.run_full(db_conn, ctx)

    vid = db_conn.execute(
        "SELECT id FROM videos WHERE content_hash = %s", (ctx.content_hash,)
    ).fetchone()[0]
    before = db_conn.execute(
        "SELECT count(*) FROM transcript_segments WHERE video_id = %s", (vid,)
    ).fetchone()[0]

    # Second full run on the same bytes must not duplicate segments (stages already done).
    pipeline.run_full(db_conn, ctx)
    after = db_conn.execute(
        "SELECT count(*) FROM transcript_segments WHERE video_id = %s", (vid,)
    ).fetchone()[0]
    assert after == before == 4


def test_parse_transcript_reingest_replaces_not_appends(
    db_conn, media_file, fake_transcribe, fake_embedder, monkeypatch
) -> None:
    # Force parse_transcript to re-run by clearing its stage status, then confirm it deletes
    # existing rows for the video before inserting (idempotent stage, no duplicate seqs).
    monkeypatch.setattr(pipeline, "make_embedder", lambda: fake_embedder)
    ctx = pipeline.IngestContext(media_path=str(media_file))
    pipeline.run_full(db_conn, ctx)
    vid = db_conn.execute(
        "SELECT id FROM videos WHERE content_hash = %s", (ctx.content_hash,)
    ).fetchone()[0]

    db_conn.execute(
        "UPDATE jobs SET stage_status = stage_status - 'parse_transcript' "
        "WHERE content_hash = %s",
        (ctx.content_hash,),
    )
    pipeline.run_full(db_conn, ctx)
    count = db_conn.execute(
        "SELECT count(*) FROM transcript_segments WHERE video_id = %s", (vid,)
    ).fetchone()[0]
    assert count == 4  # replaced, not 8


def test_resume_skips_done_transcribe_and_runs_pending_later_stage(
    db_conn, media_file, fake_embedder, monkeypatch
) -> None:
    monkeypatch.setattr(pipeline, "make_embedder", lambda: fake_embedder)
    ctx = pipeline.IngestContext(media_path=str(media_file))
    pipeline.run_stage(db_conn, "register", ctx)
    db_conn.execute(
        """
        UPDATE jobs
        SET stage_status = jsonb_build_object(
                'transcribe', jsonb_build_object('state', 'done')
            )
        WHERE content_hash = %s
        """,
        (ctx.content_hash,),
    )
    db_conn.commit()

    calls: list[str] = []

    def fail_transcribe(conn, ctx):
        raise AssertionError("transcribe should be skipped when already marked done")

    def run_frames(conn, ctx):
        calls.append("frames")

    monkeypatch.setattr(
        pipeline,
        "STAGES",
        [
            ("register", register),
            ("transcribe", SimpleNamespace(run=fail_transcribe)),
            ("frames", SimpleNamespace(run=run_frames)),
        ],
    )

    pipeline.run_full(db_conn, pipeline.IngestContext(media_path=str(media_file)))

    assert calls == ["frames"]


def test_full_run_persists_time_costs_in_stage_order(
    db_conn, media_file, fake_transcribe, fake_embedder, monkeypatch
) -> None:
    values = iter(
        [
            float(i)
            for pair in ((n, n + 0.1111) for n in range(len(pipeline.STAGES)))
            for i in pair
        ]
    )
    monkeypatch.setattr(pipeline, "make_embedder", lambda: fake_embedder)
    monkeypatch.setattr(pipeline.time, "perf_counter", lambda: next(values))
    ctx = pipeline.IngestContext(media_path=str(media_file))

    pipeline.run_full(db_conn, ctx)

    time_costs = db_conn.execute(
        "SELECT time_costs FROM jobs WHERE content_hash = %s", (ctx.content_hash,)
    ).fetchone()[0]
    assert [entry["stage"] for entry in time_costs] == [name for name, _ in pipeline.STAGES]
    assert [entry["seconds"] for entry in time_costs] == [0.111] * len(pipeline.STAGES)


def test_rerun_replaces_existing_stage_time_cost_instead_of_duplicating(
    db_conn, media_file, fake_transcribe, fake_embedder, monkeypatch
) -> None:
    first_values = iter(
        [
            float(i)
            for pair in ((n, n + 0.1111) for n in range(len(pipeline.STAGES)))
            for i in pair
        ]
    )
    monkeypatch.setattr(pipeline, "make_embedder", lambda: fake_embedder)
    monkeypatch.setattr(pipeline.time, "perf_counter", lambda: next(first_values))
    ctx = pipeline.IngestContext(media_path=str(media_file))
    pipeline.run_full(db_conn, ctx)

    db_conn.execute(
        "UPDATE jobs SET stage_status = stage_status - 'parse_transcript' WHERE content_hash = %s",
        (ctx.content_hash,),
    )
    db_conn.commit()

    second_values = iter([50.0, 50.4444, 60.0, 60.5555] + [70.0, 70.0] * (len(pipeline.STAGES) - 2))
    monkeypatch.setattr(pipeline.time, "perf_counter", lambda: next(second_values))
    pipeline.run_full(db_conn, ctx)

    time_costs = db_conn.execute(
        "SELECT time_costs FROM jobs WHERE content_hash = %s", (ctx.content_hash,)
    ).fetchone()[0]
    by_stage = {entry["stage"]: entry["seconds"] for entry in time_costs}
    assert len(time_costs) == len(pipeline.STAGES)
    assert by_stage["register"] == 0.444
    assert by_stage["parse_transcript"] == 0.556
