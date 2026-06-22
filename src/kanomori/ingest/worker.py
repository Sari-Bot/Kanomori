"""The ingestion worker: a polling loop that claims and runs queued jobs.

One job at a time: ``claim_one`` locks a queued/failed job with ``FOR UPDATE SKIP LOCKED`` (so
multiple workers don't collide) and marks it running; ``claim_and_run_one`` then runs the
pipeline, marking the job failed (recording the error, bumping ``attempts``) on exception.
``main`` loops with a sleep when idle. No broker — Postgres is the queue, which is plenty at
single-machine MVP scale.

The job's ingest request (media_path etc.) is stashed in ``stage_status->'request'`` by the
API when it enqueues, and reconstructed into an ``IngestContext`` here.
"""

from __future__ import annotations

import time

from kanomori.db import connection
from kanomori.ingest.pipeline import IngestContext, run_full

# A job is retried on failure (per-stage status makes the retry resume, not restart), but a
# permanently-broken job must not loop forever. Once a failed job reaches MAX_ATTEMPTS it is
# left in 'failed' and no longer claimed; an operator can reset attempts to requeue it.
MAX_ATTEMPTS = 3


def claim_one(conn) -> dict | None:
    """Lock and claim the oldest eligible queued/failed job; mark it running. None if none.

    Eligible = queued, or failed with attempts below MAX_ATTEMPTS (so a permanently-failing
    job stops being re-claimed once it exhausts its retries). Returns a dict with the job id
    and the request fields needed to build an IngestContext.
    """
    row = conn.execute(
        """
        SELECT id, content_hash, stage_status -> 'request' AS request
        FROM jobs
        WHERE status = 'queued'
           OR (status = 'failed' AND attempts < %s)
        ORDER BY id
        FOR UPDATE SKIP LOCKED
        LIMIT 1
        """,
        (MAX_ATTEMPTS,),
    ).fetchone()
    if row is None:
        return None

    job_id, content_hash, request = row
    request = request or {}
    conn.execute(
        "UPDATE jobs SET status = 'running', updated_at = now() WHERE id = %s", (job_id,)
    )
    conn.commit()
    return {"id": job_id, "content_hash": content_hash, **request}


def _context_from_claim(claim: dict) -> IngestContext:
    return IngestContext(
        media_path=claim["media_path"],
        source_url=claim.get("source_url"),
        source_platform=claim.get("source_platform"),
        title=claim.get("title"),
        stream_type=claim.get("stream_type"),
        separate=claim.get("separate", False),
    )


def claim_and_run_one(conn) -> bool:
    """Claim one job and run the pipeline. Returns True if a job ran, False if idle.

    On failure, marks the job failed, records the error, and bumps attempts — leaving it
    eligible for a later retry (per-stage status means the retry resumes, not restarts).
    """
    claim = claim_one(conn)
    if claim is None:
        return False

    try:
        run_full(conn, _context_from_claim(claim))
    except Exception as exc:  # noqa: BLE001 - record any failure on the job and move on
        conn.rollback()
        conn.execute(
            """
            UPDATE jobs
            SET status = 'failed', error = %s, attempts = attempts + 1, updated_at = now()
            WHERE id = %s
            """,
            (str(exc), claim["id"]),
        )
        conn.commit()
    return True


def main(poll_interval: float = 5.0) -> None:  # pragma: no cover - loop entry point
    """Run the worker loop forever, sleeping ``poll_interval`` seconds when idle."""
    print("kanomori ingestion worker started; polling for jobs...")
    while True:
        with connection() as conn:
            ran = claim_and_run_one(conn)
        if not ran:
            time.sleep(poll_interval)


if __name__ == "__main__":  # pragma: no cover
    main()
