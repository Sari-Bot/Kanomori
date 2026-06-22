"""Integration tests for the ingestion worker's claim-and-run loop against real PostgreSQL.

The worker claims one queued/failed job at a time (FOR UPDATE SKIP LOCKED), runs the pipeline,
and marks the job failed (recording the error, bumping attempts) on exception. These tests
enqueue jobs and drive a single claim cycle with an injected pipeline runner — no GPU.
"""

from __future__ import annotations

from pathlib import Path

import pytest

from kanomori.ingest import worker

pytestmark = pytest.mark.requires_db

FIXTURE_SRT = Path(__file__).resolve().parents[1] / "fixtures" / "sample.srt"


@pytest.fixture
def media_file(tmp_path: Path) -> Path:
    p = tmp_path / "clip.mp4"
    p.write_bytes(b"worker-test-bytes")
    return p


def _enqueue(conn, media_path: str) -> int:
    """Insert a queued job with its request payload (mirrors what the API's /ingest does)."""
    return conn.execute(
        """
        INSERT INTO jobs (content_hash, status, stage_status)
        VALUES (md5(%s), 'queued', jsonb_build_object('request',
                jsonb_build_object('media_path', %s::text)))
        RETURNING id
        """,
        (media_path, media_path),
    ).fetchone()[0]


def test_claim_returns_none_when_no_jobs(db_conn) -> None:
    # Clear any queued/failed jobs in this transaction's view, then claim.
    db_conn.execute("UPDATE jobs SET status = 'complete' WHERE status IN ('queued','failed')")
    assert worker.claim_one(db_conn) is None


def test_claim_picks_a_queued_job(db_conn, media_file) -> None:
    job_id = _enqueue(db_conn, str(media_file))
    claimed = worker.claim_one(db_conn)
    assert claimed is not None
    assert claimed["id"] == job_id
    assert claimed["media_path"] == str(media_file)


def test_claim_marks_job_running(db_conn, media_file) -> None:
    _enqueue(db_conn, str(media_file))
    claimed = worker.claim_one(db_conn)
    status = db_conn.execute(
        "SELECT status FROM jobs WHERE id = %s", (claimed["id"],)
    ).fetchone()[0]
    assert status == "running"


def test_run_one_invokes_pipeline_and_completes(db_conn, media_file, monkeypatch) -> None:
    calls = {}

    def fake_run_full(conn, ctx):
        calls["media_path"] = ctx.media_path
        # Simulate the pipeline marking the job complete.
        conn.execute(
            "UPDATE jobs SET status='complete' WHERE content_hash=md5(%s)",
            (ctx.media_path,),
        )
        return ctx

    monkeypatch.setattr(worker, "run_full", fake_run_full)
    _enqueue(db_conn, str(media_file))
    ran = worker.claim_and_run_one(db_conn)
    assert ran is True
    assert calls["media_path"] == str(media_file)


def test_run_one_marks_failed_and_records_error(db_conn, media_file, monkeypatch) -> None:
    def boom(conn, ctx):
        raise RuntimeError("kits exploded")

    monkeypatch.setattr(worker, "run_full", boom)
    job_id = _enqueue(db_conn, str(media_file))
    worker.claim_and_run_one(db_conn)
    row = db_conn.execute(
        "SELECT status, error, attempts FROM jobs WHERE id = %s", (job_id,)
    ).fetchone()
    assert row[0] == "failed"
    assert "kits exploded" in row[1]
    assert row[2] == 1  # attempts incremented


def test_claim_and_run_returns_false_when_idle(db_conn) -> None:
    db_conn.execute("UPDATE jobs SET status='complete' WHERE status IN ('queued','failed')")
    assert worker.claim_and_run_one(db_conn) is False


def test_claim_skips_failed_job_at_attempts_cap(db_conn, media_file) -> None:
    # A job that has already failed MAX_ATTEMPTS times must not be re-claimed (no infinite
    # retry loop on a permanently-broken job).
    db_conn.execute("UPDATE jobs SET status='complete' WHERE status IN ('queued','failed')")
    job_id = _enqueue(db_conn, str(media_file))
    db_conn.execute(
        "UPDATE jobs SET status='failed', attempts=%s WHERE id=%s",
        (worker.MAX_ATTEMPTS, job_id),
    )
    assert worker.claim_one(db_conn) is None


def test_claim_still_picks_failed_job_below_attempts_cap(db_conn, media_file) -> None:
    # A failed job with attempts remaining is still eligible for retry (resume).
    db_conn.execute("UPDATE jobs SET status='complete' WHERE status IN ('queued','failed')")
    job_id = _enqueue(db_conn, str(media_file))
    db_conn.execute(
        "UPDATE jobs SET status='failed', attempts=%s WHERE id=%s",
        (worker.MAX_ATTEMPTS - 1, job_id),
    )
    claimed = worker.claim_one(db_conn)
    assert claimed is not None and claimed["id"] == job_id

