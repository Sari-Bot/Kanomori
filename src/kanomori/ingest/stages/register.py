"""Stage: register — compute content_hash and reconcile the videos row + jobs row.

content_hash is sha256 of the media bytes; it is the idempotency key for the whole pipeline.
Re-registering the same bytes returns the existing video_id (no duplicate row).

Compute/persist split (Task #12): ``compute`` is the pure half — it hashes the media bytes and
returns a :class:`RegisterResult` carrying the content_hash + metadata. ``persist`` is the DB
half — it upserts the videos row and reconciles the jobs row, and (unlike every other stage)
*produces* the ``video_id`` rather than consuming one, so its signature differs: it returns the
resolved video_id for the coordinator to thread into the other stages' ``persist`` calls.

``media_path`` is deliberately absent from ``RegisterResult`` (it is a worker-local source path;
the contract carries ``source_url`` as the public link instead — see CLAUDE.md storage rules).
The single-machine wrapper still writes ``videos.media_path`` from ctx so existing behavior is
byte-identical; the distributed coordinator passes ``media_path=None`` and stores a NULL there.

The jobs row is reconciled one of two ways, depending on how this run was started:

- ``job_id is not None`` — the worker enqueued a jobs row (with a NULL content_hash) and is now
  running it. We UPDATE that exact row by id to stamp the real content_hash/video_id, so a single
  row carries the job from enqueue through completion. This kills the old double-row orphan.
- ``job_id is None`` — a single-machine caller invoked the pipeline directly with no prior row
  (and existing tests). We fall back to upserting the jobs row by content_hash, as before.
"""

from __future__ import annotations

import hashlib
from pathlib import Path

from kanomori.ingest.stage_result import RegisterResult


def _sha256_file(path: Path, chunk: int = 1 << 20) -> str:
    h = hashlib.sha256()
    with path.open("rb") as f:
        while block := f.read(chunk):
            h.update(block)
    return h.hexdigest()


def compute(ctx) -> RegisterResult:
    """Hash the media bytes and package identity + metadata. No DB connection."""
    content_hash = _sha256_file(Path(ctx.media_path))
    return RegisterResult(
        content_hash=content_hash,
        source_url=ctx.source_url,
        source_platform=ctx.source_platform,
        title=ctx.title,
        stream_type=ctx.stream_type,
    )


def persist(
    conn,
    result: RegisterResult,
    *,
    media_path: str | None = None,
    job_id: int | None = None,
) -> int:
    """Upsert the videos row + reconcile the jobs row; return the resolved ``video_id``.

    Unlike the other stages, register *creates* the video_id (the idempotency key is
    content_hash), so this returns it rather than taking it as a parameter. ``media_path`` is the
    worker-local source path the single-machine wrapper persists; the coordinator passes None.
    """
    row = conn.execute(
        """
        INSERT INTO videos
            (content_hash, source_url, source_platform, title, media_path, stream_type)
        VALUES (%s, %s, %s, %s, %s, %s)
        ON CONFLICT (content_hash) DO UPDATE SET content_hash = EXCLUDED.content_hash
        RETURNING id
        """,
        (
            result.content_hash, result.source_url, result.source_platform, result.title,
            media_path, result.stream_type,
        ),
    ).fetchone()
    video_id = row[0]

    if job_id is not None:
        # Enqueue-then-run path: reconcile the pre-existing (NULL-hash) jobs row in place by id.
        conn.execute(
            "UPDATE jobs SET content_hash = %s, video_id = %s WHERE id = %s",
            (result.content_hash, video_id, job_id),
        )
    else:
        # Single-machine path: ensure a jobs row exists for this content_hash, linked to video.
        conn.execute(
            """
            INSERT INTO jobs (video_id, content_hash, status)
            VALUES (%s, %s, 'running')
            ON CONFLICT (content_hash) DO UPDATE SET video_id = EXCLUDED.video_id
            """,
            (video_id, result.content_hash),
        )
    return video_id


def run(conn, ctx) -> None:
    """Single-machine wrapper: compute identity, persist it, and stamp ctx (unchanged behavior)."""
    result = compute(ctx)
    ctx.content_hash = result.content_hash
    ctx.video_id = persist(
        conn, result, media_path=ctx.media_path, job_id=getattr(ctx, "job_id", None)
    )
