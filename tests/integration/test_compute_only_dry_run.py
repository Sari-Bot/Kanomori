"""L0 dry-run: the worker ``--compute-only`` chain over a real clip, heavy models mocked.

Proves the compute() orchestration + StageResult wire-contract round-trip end to end on a real
sample, exercising the genuinely-local work for real — ffmpeg audio/frame extraction, scene
detection, and RapidOCR — while mocking the parts that need GB of weights or a GPU: KITS
transcription, the BGE-M3 text embedder, SigLIP scene classification, and DINOv2 image
embeddings. This is the committed form of ``kanomori-worker --compute-only`` (plan §6, L0).

Skipped unless the ingest stack (ffmpeg + RapidOCR + scenedetect) and the sample clip are
present; it never downloads models and never touches the DB or a coordinator.
"""

from __future__ import annotations

import importlib.util
import shutil
from pathlib import Path

import numpy as np
import pytest

from kanomori.ingest import worker
from kanomori.ingest.stages import classify as classify_stage
from kanomori.ingest.stages import image_embed as image_embed_stage
from kanomori.ingest.stages import transcribe as transcribe_stage
from kanomori.ingest.stages.classify import SceneResult
from kanomori.media_source import LocalDirSource

# Real ffmpeg/OCR/scenedetect, mocked torch models: needs the ingest group, not embed.
REQUIRED_MODULES = ("scenedetect", "rapidocr", "PIL", "imagehash")
SAMPLES_ROOT = Path(__file__).resolve().parents[2] / "samples"
TALK_CLIP = SAMPLES_ROOT / "2024_talk_cut" / "video.mp4"


def _missing() -> list[str]:
    return [m for m in REQUIRED_MODULES if importlib.util.find_spec(m) is None]


pytestmark = pytest.mark.skipif(
    bool(_missing()) or shutil.which("ffmpeg") is None or not TALK_CLIP.exists(),
    reason="requires the ingest stack (ffmpeg + scenedetect + rapidocr) and the talk sample clip",
)

TEXT_DIM = 1024
IMAGE_DIM = 768


class _FakeEmbedder:
    """Stand-in for BGEEmbedder: deterministic unit vectors, no model load."""

    def embed_texts(self, texts: list[str]):
        return [np.full(TEXT_DIM, 1.0 / np.sqrt(TEXT_DIM), dtype=np.float32) for _ in texts]


def _fake_kits(audio, out_srt, **kwargs):
    """Drop a tiny two-cue SRT where KITS would write one — no GPU, no transcription."""
    out = Path(out_srt)
    out.parent.mkdir(parents=True, exist_ok=True)
    out.write_text(
        "1\n00:00:00,000 --> 00:00:02,000\nこんばんは\n\n"
        "2\n00:00:02,000 --> 00:00:05,000\n配信を始めます\n",
        encoding="utf-8",
    )
    return out


def test_compute_only_dry_run_over_talk_clip(monkeypatch, tmp_path):
    # Mock the four heavy paths; keep ffmpeg/frames/scenedetect/OCR real.
    monkeypatch.setattr(worker, "make_embedder", _FakeEmbedder)
    monkeypatch.setattr(transcribe_stage, "kits_transcribe", _fake_kits)
    monkeypatch.setattr(
        classify_stage, "classify_frame", lambda path: SceneResult("chatting", 0.9)
    )
    monkeypatch.setattr(
        image_embed_stage,
        "embed_frame",
        lambda path: np.full(IMAGE_DIM, 1.0 / np.sqrt(IMAGE_DIM), dtype=np.float32),
    )
    # Derived artifacts (audio.wav, frames/, SRT) go to a temp media root, not the repo's.
    monkeypatch.setenv("KANOMORI_MEDIA_ROOT", str(tmp_path / "media"))
    from kanomori.config import get_settings

    get_settings.cache_clear()

    # Drive run_compute_only against the talk clip via the local source store. It asserts each
    # StageResult round-trips internally (model_validate_json == original); a raised exception
    # here (ffmpeg failure, bad wire shape, OCR crash) fails the test.
    source = LocalDirSource(SAMPLES_ROOT)
    worker.run_compute_only(
        source,
        media_path=str(TALK_CLIP),
        cache_dir=tmp_path / "cache",
    )
