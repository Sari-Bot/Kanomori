"""Integration tests for the 0003_lease migration and the enqueue->register identity flow.

Two concerns, both against a live PostgreSQL+pgvector (marked requires_db, skipped when no DB
is reachable):

1. The lease/heartbeat/fencing columns added to ``jobs`` by 0003 exist, and ``content_hash``
   was relaxed to nullable (so a job can be enqueued before register computes the real hash).

2. The double-row orphan bug is fixed: enqueue inserts ONE jobs row with a NULL content_hash;
   when register runs with ``ctx.job_id`` set, it UPDATEs that SAME row to the real sha256
   instead of INSERTing a second row. After the flow there is exactly one job and it carries
   the real hash.
"""

from __future__ import annotations

import json
from pathlib import Path

import pytest

from kanomori.ingest import pipeline
from kanomori.ingest.stages import register

pytestmark = pytest.mark.requires_db


@pytest.fixture
def media_file(tmp_path: Path) -> Path:
    p = tmp_path / "clip.mp4"
    p.write_bytes(b"enqueue-register-flow-bytes")
    return p


def test_lease_columns_added_and_content_hash_nullable(db_conn) -> None:
    cols = {
        row[0]: row[1]
        for row in db_conn.execute(
            """
            SELECT column_name, is_nullable
            FROM information_schema.columns
            WHERE table_schema = 'public' AND table_name = 'jobs'
            """
        ).fetchall()
    }
    # New lease/heartbeat/fencing columns exist.
    for col in ("worker_id", "lease_epoch", "lease_expires_at", "heartbeat_at"):
        assert col in cols, f"missing column {col}"
    # content_hash was relaxed to nullable so a job can exist before register hashes the file.
    assert cols["content_hash"] == "YES"


def test_claimable_index_exists(db_conn) -> None:
    idx = db_conn.execute(
        "SELECT indexname FROM pg_indexes WHERE tablename = 'jobs' "
        "AND indexname = 'jobs_claimable_idx'"
    ).fetchone()
    assert idx is not None


def _enqueue(conn, media_path: str) -> int:
    """Mirror the API's /ingest: one queued job, NULL content_hash, request stashed."""
    payload = json.dumps({"media_path": media_path})
    return conn.execute(
        """
        INSERT INTO jobs (content_hash, status, stage_status)
        VALUES (NULL, 'queued', jsonb_build_object('request', %s::jsonb))
        RETURNING id
        """,
        (payload,),
    ).fetchone()[0]


def test_enqueue_creates_single_row_with_null_hash(db_conn, media_file) -> None:
    job_id = _enqueue(db_conn, str(media_file))
    row = db_conn.execute(
        "SELECT content_hash, status FROM jobs WHERE id = %s", (job_id,)
    ).fetchone()
    assert row[0] is None  # not yet hashed
    assert row[1] == "queued"


def test_register_with_job_id_reconciles_same_row_no_orphan(db_conn, media_file) -> None:
    # Enqueue: one job, NULL hash (the API path).
    job_id = _enqueue(db_conn, str(media_file))
    db_conn.commit()

    before = db_conn.execute("SELECT count(*) FROM jobs").fetchone()[0]

    # Worker resolves identity: register runs with the claimed job id on the context.
    ctx = pipeline.IngestContext(media_path=str(media_file), job_id=job_id)
    register.run(db_conn, ctx)
    db_conn.commit()

    after = db_conn.execute("SELECT count(*) FROM jobs").fetchone()[0]
    assert after == before  # no second (orphan) row was inserted

    row = db_conn.execute(
        "SELECT content_hash, video_id FROM jobs WHERE id = %s", (job_id,)
    ).fetchone()
    assert row[0] == ctx.content_hash  # the SAME row now carries the real sha256
    assert len(row[0]) == 64
    assert row[1] == ctx.video_id


def test_register_without_job_id_still_upserts_by_hash(db_conn, media_file) -> None:
    # Single-machine path (no enqueue): register both creates the video and its jobs row.
    ctx = pipeline.IngestContext(media_path=str(media_file))
    register.run(db_conn, ctx)
    db_conn.commit()

    rows = db_conn.execute(
        "SELECT count(*) FROM jobs WHERE content_hash = %s", (ctx.content_hash,)
    ).fetchone()[0]
    assert rows == 1
