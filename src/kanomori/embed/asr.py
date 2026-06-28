"""Lazy kotoba-whisper ASR wrapper for uploaded audio queries."""

from __future__ import annotations

import subprocess
from pathlib import Path
from typing import TypedDict

from kanomori.config import get_settings
from kanomori.ingest.stages.locate_media import build_ffmpeg_extract_argv


class AsrSegment(TypedDict):
    start: float | None
    end: float | None
    text: str


class AudioDecodeError(RuntimeError):
    """Raised when ffmpeg/ffprobe cannot decode an uploaded audio clip."""


class KotobaWhisperASR:
    """In-process kotoba-whisper ASR, loaded lazily and kept warm per process."""

    def __init__(self, model_name: str | None = None, device: str = "cpu"):
        resolved = model_name or get_settings().audio_asr_model
        if not resolved:
            raise ValueError("KANOMORI_AUDIO_ASR_MODEL must be set for audio search")
        self.model_name = resolved
        self.device = device
        self._pipe = None

    def _load(self):
        if self._pipe is None:
            import torch
            from transformers import pipeline

            self._pipe = pipeline(
                "automatic-speech-recognition",
                model=self.model_name,
                dtype=torch.float32,
                device=self.device,
                chunk_length_s=15,
            )
        return self._pipe

    def warmup(self) -> None:
        """Load the ASR pipeline without transcribing an audio file."""
        self._load()

    def transcribe(self, audio_wav_path: str | Path) -> list[AsrSegment]:
        pipe = self._load()
        result = pipe(
            str(audio_wav_path),
            return_timestamps=True,
            generate_kwargs={
                "language": "ja",
                "task": "transcribe",
                "num_beams": 3,
                "no_repeat_ngram_size": 3,
            },
        )
        return [
            _chunk_to_segment(chunk)
            for chunk in result.get("chunks", [])
            if _chunk_text(chunk)
        ]


def _chunk_text(chunk: dict) -> str:
    return str(chunk.get("text") or "").strip()


def _chunk_to_segment(chunk: dict) -> AsrSegment:
    timestamp = chunk.get("timestamp") or (None, None)
    start, end = timestamp
    return {"start": start, "end": end, "text": _chunk_text(chunk)}


def normalize_clip_to_wav(src: Path, dst_wav: Path) -> None:
    """Convert uploaded media into KITS-compatible 16 kHz mono WAV."""
    argv = build_ffmpeg_extract_argv(src, dst_wav)
    result = subprocess.run(argv, capture_output=True, text=True)
    if result.returncode != 0:
        detail = (result.stderr or result.stdout or "ffmpeg audio extraction failed").strip()
        raise AudioDecodeError(detail[:300])


def probe_duration_sec(path: Path) -> float:
    """Return decoded media duration using ffprobe."""
    argv = [
        "ffprobe",
        "-v",
        "error",
        "-show_entries",
        "format=duration",
        "-of",
        "default=nokey=1:noprint_wrappers=1",
        str(path),
    ]
    result = subprocess.run(argv, capture_output=True, text=True)
    if result.returncode != 0:
        detail = (result.stderr or result.stdout or "ffprobe duration probe failed").strip()
        raise AudioDecodeError(detail[:300])
    try:
        return float(result.stdout.strip())
    except ValueError as exc:
        raise AudioDecodeError(f"invalid ffprobe duration: {result.stdout.strip()!r}") from exc
