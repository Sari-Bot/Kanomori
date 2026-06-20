"""Stage: transcribe — run KITS (subprocess, GPU) to produce an SRT for the located audio.

The actual call is the module-level ``kits_transcribe`` alias so tests can patch it to drop a
fixture SRT without a GPU. The resulting SRT path is recorded on the context for the next stage.
"""

from __future__ import annotations

from pathlib import Path

from kanomori.ingest.artifacts import srt_path_for
from kanomori.kits_client import transcribe as kits_transcribe


def run(conn, ctx) -> None:
    # Deterministic, content-hash-keyed location so parse_transcript can find this SRT even
    # when transcribe is skipped on a resumed run.
    out_srt = srt_path_for(ctx.content_hash)

    audio = ctx.audio_path or ctx.media_path  # locate_media sets audio_path; fall back to media
    ctx.srt_path = str(
        kits_transcribe(
            Path(audio),
            out_srt,
            separate=ctx.separate,
            language=ctx.language,
        )
    )
