"""Offline ingestion: a resumable, idempotent staged DAG.

Ingestion is keyed by ``content_hash`` (sha256 of the media). Each stage records its state in
``jobs.stage_status`` and commits independently, so a crash resumes at the first non-done stage
and re-ingesting the same bytes is a no-op. Stages import their heavy dependencies lazily; the
only GPU stage is transcription, which shells out to KITS.
"""
