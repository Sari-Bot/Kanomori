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


def ts_from_frame_name(name: str) -> float:
    """Recover a frame's ``ts_sec`` from its filename — the exact inverse of ``frame_path_for``.

    The compute side of the visual stages (ocr/classify/image_embed) has no DB connection, so it
    reads frames off disk and derives each one's timestamp from the deterministic name. For any
    timestamp the frames stage actually persisted (``round(t, 3)``), this returns the identical
    double, so the persist side can resolve ``frame_id`` by ``(video_id, ts_sec)`` equality.
    """
    stem = name.removesuffix(".jpg")
    _prefix, seconds, millis = stem.split("_")
    total_ms = int(seconds) * 1000 + int(millis)
    return total_ms / 1000.0


def frames_on_disk(content_hash: str) -> list[tuple[float, Path]]:
    """List a video's extracted frame JPEGs as ``(ts_sec, path)`` pairs, sorted by ts_sec.

    This is how the visual stages source their frames on the compute side: the frames stage wrote
    deterministically-named JPEGs into ``frame_dir_for(content_hash)``; here we glob them back and
    recover each timestamp via :func:`ts_from_frame_name`. Returns ``[]`` when no frames exist.
    """
    frame_dir = frame_dir_for(content_hash)
    if not frame_dir.is_dir():
        return []
    pairs = [(ts_from_frame_name(p.name), p) for p in frame_dir.glob("frame_*.jpg")]
    pairs.sort(key=lambda pair: pair[0])
    return pairs
