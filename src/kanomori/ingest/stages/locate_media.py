"""Stage: locate_media — ensure KITS-ready audio exists and decide the karaoke flag.

Extracts a 16 kHz mono WAV from the source media with ffmpeg (Whisper's expected rate) into the
content-hash-keyed artifact dir, and sets ``ctx.audio_path``. Also decides whether to pass
KITS ``--separate`` (vocal isolation): there is no cheap pre-transcription signal for "this is
singing", so we use a title/metadata keyword heuristic plus the caller's explicit override.

``_run`` wraps ``subprocess.run`` and is patched in tests so the stage can be exercised without
a real media file or ffmpeg.
"""

from __future__ import annotations

import subprocess
from pathlib import Path

from kanomori.ingest.artifacts import artifact_dir

# Substrings (matched case-insensitively, after lowering) that mark a karaoke / singing stream.
# Mix of Japanese and romaji terms common in 鹿乃 / VTuber stream titles.
_KARAOKE_KEYWORDS = ("歌枠", "歌って", "karaoke", "弾き語り", "cover", "setlist", "歌")


def _run(argv: list[str], **kwargs) -> subprocess.CompletedProcess:
    """Indirection over subprocess.run so tests can inject a fake runner."""
    return subprocess.run(argv, capture_output=True, text=True, **kwargs)


def decide_separate(title: str | None, *, explicit: bool) -> bool:
    """Decide whether to run KITS --separate. Explicit override wins; else title heuristic."""
    if explicit:
        return True
    if not title:
        return False
    low = title.lower()
    return any(kw.lower() in low for kw in _KARAOKE_KEYWORDS)


def build_ffmpeg_extract_argv(media: Path, out_wav: Path) -> list[str]:
    """ffmpeg argv to extract 16 kHz mono PCM WAV (Whisper's rate), overwriting output."""
    return [
        "ffmpeg", "-i", str(media),
        "-vn", "-ac", "1", "-ar", "16000",
        str(out_wav), "-y",
    ]


def run(conn, ctx) -> None:
    compute(ctx)
    persist(conn, ctx.video_id, None)


def compute(ctx) -> None:
    """Extract KITS-ready audio and decide --separate; sets ctx.audio_path/ctx.separate.

    Returns None: this stage produces only worker-local state (the WAV artifact + the decision
    threaded on ctx), no DB rows. The audio lands at the deterministic content-hash-keyed path so
    transcribe finds it even when locate_media is skipped on a resumed run.
    """
    out_dir = artifact_dir(ctx.content_hash)
    out_dir.mkdir(parents=True, exist_ok=True)
    out_wav = out_dir / "audio.wav"

    argv = build_ffmpeg_extract_argv(Path(ctx.media_path), out_wav)
    result = _run(argv)
    if result.returncode != 0:
        raise RuntimeError(f"ffmpeg audio extraction failed: {(result.stderr or '')[:300]}")

    ctx.audio_path = str(out_wav)
    ctx.separate = decide_separate(ctx.title, explicit=ctx.separate)


def persist(conn, video_id, result) -> None:
    """No-op: locate_media writes no DB rows (its output is the on-disk WAV + ctx state)."""
    return None
