"""Deterministic locations for derived ingestion artifacts.

Artifacts are keyed by ``content_hash`` under ``MEDIA_ROOT`` so a stage can locate a prior
stage's output without relying on in-memory context — essential for resume: when a completed
stage is skipped, the next stage still finds its artifact on disk. Wipe ``MEDIA_ROOT`` and
re-ingest to rebuild; nothing here is source video.
"""

from __future__ import annotations

from pathlib import Path

from kanomori.config import get_settings


def artifact_dir(content_hash: str) -> Path:
    """Directory holding all derived artifacts for one video."""
    return Path(get_settings().media_root) / content_hash


def srt_path_for(content_hash: str) -> Path:
    """Deterministic path of the KITS transcript SRT for a video."""
    return artifact_dir(content_hash) / "transcript.srt"


def audio_path_for(content_hash: str) -> Path:
    """Deterministic path of the extracted 16kHz mono WAV for a video."""
    return artifact_dir(content_hash) / "audio.wav"


def frame_dir_for(content_hash: str) -> Path:
    """Directory holding short-preview frame thumbnails for a video."""
    return artifact_dir(content_hash) / "frames"


def frame_path_for(content_hash: str, ts_sec: float) -> Path:
    """Deterministic JPEG thumbnail path for a timestamp, rounded to milliseconds."""
    total_ms = max(0, round(ts_sec * 1000))
    seconds, millis = divmod(total_ms, 1000)
    return frame_dir_for(content_hash) / f"frame_{seconds:06d}_{millis:03d}.jpg"
