"""Claim / lease / fence for multi-worker job coordination, plus the stage persist registry.

This is the DB half of the coordinator (Task #13). The single-machine worker (#12) ran the
whole pipeline in-process; the distributed model splits it: remote workers do the GPU compute
and POST results to the coordinator, which owns the database. To hand a job to exactly one
worker and survive worker crashes, jobs are leased with a fencing token:

* **claim** atomically picks the oldest eligible job (``FOR UPDATE SKIP LOCKED`` so concurrent
  coordinators/claims never hand the same row to two workers), marks it running, sets a
  time-bounded lease, and bumps ``lease_epoch`` — a monotonic fencing token.
* **heartbeat** lets a long-running stage extend its lease while it makes progress.
* Every mutation that follows (``mark_stage_done`` / ``complete`` / ``fail``) carries the
  ``lease_epoch`` the worker was handed and updates ``WHERE id=%s AND lease_epoch=%s``. A
  resurrected zombie worker holding a stale epoch updates 0 rows and is told (False -> HTTP 409)
  to stop. An expired lease lets another worker re-claim, bumping the epoch and fencing the old
  holder.

``STAGE_SPECS`` is the single source of truth (shared by coordinator and worker, #14) mapping
each pipeline stage to its wire ``StageResult`` model, its persist callable, and which binary
artifact kind (if any) rides alongside as multipart files. ``persist_stage`` dispatches a parsed
result to the right stage's ``persist`` with that stage's specific signature.
"""

from __future__ import annotations

import json
from dataclasses import dataclass
from datetime import UTC, datetime

from kanomori.ingest.job_time_costs import merge_time_costs
from kanomori.ingest.stage_result import (
    ClassifyResult,
    FramesResult,
    ImageEmbedResult,
    OcrResult,
    ParseTranscriptResult,
    RegisterResult,
    TranscribeResult,
)
from kanomori.ingest.stages import (
    classify,
    frames,
    image_embed,
    locate_media,
    ocr,
    parse_transcript,
    register,
    transcribe,
)

# A permanently-broken job must not be re-claimed forever; once a failed job hits this many
# attempts it stays 'failed' and drops out of the eligible set (mirrors worker.MAX_ATTEMPTS).
MAX_ATTEMPTS = 3

# Fallback lease length when a worker doesn't specify one on claim/heartbeat.
DEFAULT_LEASE_SECONDS = 120


# --- stage registry ----------------------------------------------------------------------


@dataclass(frozen=True)
class StageSpec:
    """How one pipeline stage's result crosses the worker->coordinator boundary.

    ``model`` is the :class:`pydantic.BaseModel` the wire ``result`` JSON parses into (None for
    no-DB stages that carry nothing — locate_media). ``module`` exposes ``persist(...)``.
    ``artifact_kind`` names the binary that accompanies this stage as multipart files —
    ``"frame"`` (the JPEGs) for frames, ``"srt"`` for transcribe, None otherwise.
    """

    model: type | None
    module: object
    artifact_kind: str | None = None


# Keyed by the stage names in pipeline.STAGES (order is owned there; this is a lookup table).
STAGE_SPECS: dict[str, StageSpec] = {
    "register": StageSpec(RegisterResult, register),
    "locate_media": StageSpec(None, locate_media),
    "transcribe": StageSpec(TranscribeResult, transcribe, artifact_kind="srt"),
    "parse_transcript": StageSpec(ParseTranscriptResult, parse_transcript),
    "frames": StageSpec(FramesResult, frames, artifact_kind="frame"),
    "ocr": StageSpec(OcrResult, ocr),
    "classify": StageSpec(ClassifyResult, classify),
    "image_embed": StageSpec(ImageEmbedResult, image_embed),
}


def persist_stage(
    conn, stage_name: str, result, *, job_id: int, video_id, content_hash
) -> int | None:
    """Persist one stage's parsed ``result`` via its stage-specific ``persist`` signature.

    Returns the resolved ``video_id`` for the register stage (which *creates* it), else None.
    The router threads register's return value into ``videos.id`` / the job row and passes it
    back here for the downstream stages, which consume it.

    Signatures differ by stage (see the stage modules):
      * register  -> persist(conn, result, *, media_path=None, job_id=...) returns video_id
      * frames    -> persist(conn, video_id, result, *, content_hash=...)
      * everything else -> persist(conn, video_id, result)
    locate_media / transcribe persist are DB no-ops (their output is the on-disk artifact).
    """
    spec = STAGE_SPECS[stage_name]
    if stage_name == "register":
        # Pass job_id so register reconciles THIS job's row in place (UPDATE by id), stamping the
        # real content_hash + video_id — rather than the single-machine ON CONFLICT path that
        # would INSERT a second, orphan jobs row. media_path stays NULL (worker-local source).
        return spec.module.persist(conn, result, media_path=None, job_id=job_id)
    if stage_name == "frames":
        spec.module.persist(conn, video_id, result, content_hash=content_hash)
        return None
    spec.module.persist(conn, video_id, result)
    return None


# --- claim / lease / fence ---------------------------------------------------------------


def claim(conn, worker_id: str, lease_seconds: int = DEFAULT_LEASE_SECONDS) -> dict | None:
    """Atomically lease the oldest eligible job to ``worker_id``; None when nothing is eligible.

    Eligible = queued, OR running with a lapsed lease (the holder crashed), OR failed with
    attempts left. ``FOR UPDATE SKIP LOCKED`` means a row another claimer already locked in an
    open transaction is skipped rather than blocked on — so two coordinators never hand the same
    job out twice. On claim we bump ``lease_epoch`` (fencing token), stamp the worker + a fresh
    lease window, and flip status to running.

    Returns ``{job_id, content_hash, lease_epoch, request, stages_done}``. ``content_hash`` is
    None for a not-yet-registered job (register reconciles it). ``request`` is the enqueued
    ingest payload (from ``stage_status->'request'``); ``stages_done`` lists stages already
    recorded done/skipped so a resumed worker skips them.
    """
    row = conn.execute(
        """
        UPDATE jobs
        SET status = 'running',
            worker_id = %s,
            lease_epoch = lease_epoch + 1,
            lease_expires_at = now() + make_interval(secs => %s),
            heartbeat_at = now(),
            updated_at = now()
        WHERE id = (
            SELECT id FROM jobs
            WHERE status = 'queued'
               OR (status = 'running' AND lease_expires_at < now())
               OR (status = 'failed' AND attempts < %s)
            ORDER BY id
            FOR UPDATE SKIP LOCKED
            LIMIT 1
        )
        RETURNING id, content_hash, lease_epoch, stage_status
        """,
        (worker_id, lease_seconds, MAX_ATTEMPTS),
    ).fetchone()
    if row is None:
        return None

    job_id, content_hash, lease_epoch, stage_status = row
    stage_status = stage_status or {}
    request = stage_status.get("request") or {}
    stages_done = [
        name
        for name, info in stage_status.items()
        if name != "request"
        and isinstance(info, dict)
        and info.get("state") in {"done", "skipped"}
    ]
    return {
        "job_id": job_id,
        "content_hash": content_hash,
        "lease_epoch": lease_epoch,
        "request": request,
        "stages_done": stages_done,
    }


def heartbeat(
    conn, job_id: int, lease_epoch: int, lease_seconds: int = DEFAULT_LEASE_SECONDS
) -> bool:
    """Extend the lease while a worker still holds the current epoch. False ⇒ fenced/stale.

    Fence-checked: only the worker whose ``lease_epoch`` still matches the row can push the
    lease out. A stale epoch (the job was re-claimed) updates 0 rows -> False -> HTTP 409.
    """
    cur = conn.execute(
        """
        UPDATE jobs
        SET lease_expires_at = now() + make_interval(secs => %s),
            heartbeat_at = now(),
            updated_at = now()
        WHERE id = %s AND lease_epoch = %s
        """,
        (lease_seconds, job_id, lease_epoch),
    )
    return cur.rowcount == 1


def mark_stage_done(
    conn,
    job_id: int,
    lease_epoch: int,
    stage_name: str,
    state: str = "done",
    compute_seconds: float | None = None,
) -> bool:
    """Record a stage as done/skipped in ``stage_status`` (fence-checked). False ⇒ stale epoch.

    Merges ``{stage_name: {state, finished}}`` into the jsonb exactly like
    ``pipeline._mark_stage`` does, and advances ``current_stage`` — but gated on ``lease_epoch``
    so only the current lease holder can record progress.
    """
    row = conn.execute(
        "SELECT time_costs FROM jobs WHERE id = %s AND lease_epoch = %s",
        (job_id, lease_epoch),
    ).fetchone()
    if row is None:
        return False
    merged = row[0] if compute_seconds is None else merge_time_costs(row[0], stage_name, compute_seconds)
    cur = conn.execute(
        """
        UPDATE jobs
        SET stage_status = stage_status || jsonb_build_object(
                %s::text, jsonb_build_object('state', %s::text, 'finished', %s::text)
            ),
            time_costs = %s::jsonb,
            current_stage = %s,
            updated_at = now()
        WHERE id = %s AND lease_epoch = %s
        """,
        (
            stage_name,
            state,
            datetime.now(UTC).isoformat(),
            json.dumps(merged),
            stage_name,
            job_id,
            lease_epoch,
        ),
    )
    return cur.rowcount == 1


def complete(conn, job_id: int, lease_epoch: int) -> bool:
    """Mark the job complete (fence-checked). False ⇒ stale epoch (lost the lease)."""
    cur = conn.execute(
        """
        UPDATE jobs
        SET status = 'complete', updated_at = now()
        WHERE id = %s AND lease_epoch = %s
        """,
        (job_id, lease_epoch),
    )
    return cur.rowcount == 1


def fail(conn, job_id: int, lease_epoch: int, error: str) -> bool:
    """Mark the job failed, record the error, bump attempts (fence-checked). False ⇒ stale.

    Bumping attempts is what eventually drops a permanently-broken job out of the eligible set
    (attempts >= MAX_ATTEMPTS), preventing an infinite reclaim/fail loop.
    """
    cur = conn.execute(
        """
        UPDATE jobs
        SET status = 'failed', error = %s, attempts = attempts + 1, updated_at = now()
        WHERE id = %s AND lease_epoch = %s
        """,
        (error, job_id, lease_epoch),
    )
    return cur.rowcount == 1
