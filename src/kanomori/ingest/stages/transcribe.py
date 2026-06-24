"""Stage: transcribe — run KITS (subprocess, GPU) to produce an SRT for the located audio.

The actual call is the module-level ``kits_transcribe`` alias so tests can patch it to drop a
fixture SRT without a GPU. The resulting SRT path is recorded on the context for the next stage.
"""

from __future__ import annotations

from pathlib import Path

from kanomori.ingest.artifacts import audio_path_for, srt_path_for
from kanomori.ingest.stage_result import TranscribeResult
from kanomori.kits_client import transcribe as kits_transcribe


def run(conn, ctx) -> None:
    result = compute(ctx)
    persist(conn, ctx.video_id, result)


def compute(ctx) -> TranscribeResult:
    """Run KITS to produce the SRT artifact; record its path on ctx. No DB rows.

    The SRT is written to the deterministic content-hash-keyed location so parse_transcript finds
    it even when transcribe is skipped on a resumed run. ``TranscribeResult`` declares the SRT as
    an artifact the coordinator stores alongside the (empty) DB payload for resume.
    """
    out_srt = srt_path_for(ctx.content_hash)

    # Fall back to the deterministic extracted-audio artifact (16kHz wav) — NOT ctx.media_path
    # (the raw .mp4) — so a resumed run where locate_media was skipped still feeds KITS the
    # decodable audio it produced. Mirrors how the SRT path is derived above.
    audio = ctx.audio_path or str(audio_path_for(ctx.content_hash))
    stage_log = getattr(ctx, "stage_log", None)
    log_output = None
    if stage_log is not None:
        log_output = lambda stream, line: stage_log("transcribe", stream, line)
    ctx.srt_path = str(
        kits_transcribe(
            Path(audio),
            out_srt,
            separate=ctx.separate,
            language=ctx.language,
            log_output=log_output,
        )
    )
    return TranscribeResult()


def persist(conn, video_id, result: TranscribeResult) -> None:
    """No-op for the DB: the SRT is an artifact, persisted out-of-band by the coordinator."""
    return None
