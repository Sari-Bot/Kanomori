"""The ingestion orchestrator: IngestContext + stage runner + resumable full run.

STAGES lists the stages in execution order. Each stage's completion is recorded in
``jobs.stage_status[<name>] = {"state": "done", ...}`` and committed, so ``run_full`` skips
stages already marked done — a crash resumes at the first non-done stage, and re-ingesting the
same content_hash is a no-op once all stages are done.

``run_stage`` runs a single named stage unconditionally (used by tests and for targeted
re-runs); ``run_full`` is the normal entry point that respects stage status.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from datetime import UTC, datetime

from kanomori.ingest.stages import locate_media, parse_transcript, register, transcribe


@dataclass
class IngestContext:
    """Mutable state threaded through the stages of one ingestion run."""

    media_path: str
    source_url: str | None = None
    source_platform: str | None = None
    title: str | None = None
    stream_type: str | None = None
    separate: bool = False
    language: str = "japanese"

    # Populated by stages as they run.
    content_hash: str | None = None
    video_id: int | None = None
    audio_path: str | None = None
    srt_path: str | None = None
    embedder: object = field(default=None, repr=False)


# Stage name -> module exposing run(conn, ctx). Order matters (resumable DAG).
STAGES: list[tuple[str, object]] = [
    ("register", register),
    ("locate_media", locate_media),
    ("transcribe", transcribe),
    ("parse_transcript", parse_transcript),
]


def make_embedder():
    """Construct the text embedder. Patched in tests to inject a deterministic fake."""
    from kanomori.embed.text_embedder import BGEEmbedder

    return BGEEmbedder()


def run_stage(conn, name: str, ctx: IngestContext):
    """Run one named stage unconditionally and return the context. Does not touch job status."""
    stage = dict(STAGES)[name]
    stage.run(conn, ctx)
    return ctx


def _stage_done(conn, content_hash: str, name: str) -> bool:
    # %s::text disambiguates the overloaded jsonb `->` operator (object-key vs array-index);
    # without the cast Postgres can't infer the param type (IndeterminateDatatype).
    row = conn.execute(
        "SELECT stage_status -> %s::text ->> 'state' FROM jobs WHERE content_hash = %s",
        (name, content_hash),
    ).fetchone()
    return bool(row) and row[0] == "done"


def _mark_stage(conn, content_hash: str, name: str) -> None:
    conn.execute(
        """
        UPDATE jobs
        SET stage_status = stage_status || jsonb_build_object(
                %s::text, jsonb_build_object('state', 'done', 'finished', %s::text)
            ),
            current_stage = %s,
            updated_at = now()
        WHERE content_hash = %s
        """,
        (name, datetime.now(UTC).isoformat(), name, content_hash),
    )


def run_full(conn, ctx: IngestContext) -> IngestContext:
    """Run all stages in order, skipping any already marked done; mark the job complete.

    register must run first to establish content_hash/video_id and the jobs row. After that,
    stages already recorded done in jobs.stage_status are skipped. Each completed stage is
    committed so progress survives a crash.
    """
    if ctx.embedder is None:
        ctx.embedder = make_embedder()

    for name, stage in STAGES:
        # register is cheap + sets identity; always run it to (re)resolve content_hash/video_id.
        if name != "register" and _stage_done(conn, ctx.content_hash, name):
            continue
        stage.run(conn, ctx)
        _mark_stage(conn, ctx.content_hash, name)
        conn.commit()

    conn.execute(
        "UPDATE jobs SET status = 'complete', updated_at = now() WHERE content_hash = %s",
        (ctx.content_hash,),
    )
    conn.commit()
    return ctx
