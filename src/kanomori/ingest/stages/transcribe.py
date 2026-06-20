"""Stage: transcribe — run KITS (subprocess, GPU) to produce an SRT for the located audio.

The actual call is the module-level ``kits_transcribe`` alias so tests can patch it to drop a
fixture SRT without a GPU. The resulting SRT path is recorded on the context for the next stage.
"""

from __future__ import annotations

from pathlib import Path

from kanomori.ingest.artifacts import audio_path_for, srt_path_for
from kanomori.kits_client import transcribe as kits_transcribe


def run(conn, ctx) -> None:
    # Deterministic, content-hash-keyed location so parse_transcript can find this SRT even
    # when transcribe is skipped on a resumed run.
    out_srt = srt_path_for(ctx.content_hash)

    # Fall back to the deterministic extracted-audio artifact (16kHz wav) — NOT ctx.media_path
    # (the raw .mp4) — so a resumed run where locate_media was skipped still feeds KITS the
    # decodable audio it produced. Mirrors how the SRT path is derived above.
    audio = ctx.audio_path or str(audio_path_for(ctx.content_hash))
    ctx.srt_path = str(
        kits_transcribe(
            Path(audio),
            out_srt,
            separate=ctx.separate,
            language=ctx.language,
        )
    )
