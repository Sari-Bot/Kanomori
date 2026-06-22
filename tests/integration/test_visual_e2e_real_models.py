"""Opt-in real-model smoke for the visual ingestion and screenshot retrieval path.

This deliberately avoids KITS/transcription. It proves the post-transcript visual path can run
against a real clip: ffmpeg frame extraction, RapidOCR, SigLIP scene classification, DINOv2
frame embeddings, screenshot candidate generation, and scene-aware merge.
"""

from __future__ import annotations

import importlib.util
import os
import shutil
from pathlib import Path

import pytest

from kanomori.models import Modality

pytestmark = [pytest.mark.requires_db, pytest.mark.requires_models]

SAMPLE_ENV = "KANOMORI_VISUAL_E2E_SAMPLE"
REQUIRED_MODULES = ("torch", "transformers", "PIL", "imagehash", "rapidocr_onnxruntime")


def _missing_modules() -> list[str]:
    return [name for name in REQUIRED_MODULES if importlib.util.find_spec(name) is None]


requires_visual_stack = pytest.mark.skipif(
    bool(_missing_modules()) or shutil.which("ffmpeg") is None,
    reason="requires embed+ingest dependency groups and ffmpeg",
)


def _sample_clip() -> Path:
    if env_path := os.environ.get(SAMPLE_ENV):
        return Path(env_path).expanduser()
    return Path(__file__).resolve().parents[2] / "samples" / "2024_talk_cut.mp4"


def _index_sample(db_conn, monkeypatch, sample: Path):
    from kanomori.ingest.pipeline import IngestContext
    from kanomori.ingest.stages import classify, frames, image_embed, ocr, register

    ctx = IngestContext(media_path=str(sample), title="visual e2e sample")
    register.run(db_conn, ctx)

    monkeypatch.setattr(frames, "detect_scene_timestamps", lambda media: [])
    monkeypatch.setattr(frames, "probe_duration_sec", lambda media: 9.0)

    assert frames.run(db_conn, ctx) is None
    frame_rows = db_conn.execute(
        "SELECT id, frame_path FROM frames WHERE video_id = %s ORDER BY ts_sec",
        (ctx.video_id,),
    ).fetchall()
    assert len(frame_rows) == 2
    frame_paths = [Path(frame_path) for _id, frame_path in frame_rows]
    assert all(path.exists() for path in frame_paths)

    assert ocr.run(db_conn, ctx) is None
    assert classify.run(db_conn, ctx) is None
    assert image_embed.run(db_conn, ctx) is None

    return ctx.video_id, frame_paths, image_embed._embedder(), set(classify.SCENE_PROMPTS)


def _assert_frame_embeddings(db_conn, video_id: int) -> None:
    frame_counts = db_conn.execute(
        """
        SELECT count(*), count(phash), count(embedding)
        FROM frames
        WHERE video_id = %s
        """,
        (video_id,),
    ).fetchone()
    assert frame_counts == (2, 2, 2)


def _assert_scene_segments(db_conn, video_id: int, scene_types: set[str]) -> None:
    scene_rows = db_conn.execute(
        """
        SELECT scene_type, confidence
        FROM scene_segments
        WHERE video_id = %s
        """,
        (video_id,),
    ).fetchall()
    assert scene_rows
    assert {scene for scene, _confidence in scene_rows} <= scene_types
    assert all(0.0 <= float(confidence) <= 1.0 for _scene, confidence in scene_rows)


def _screenshot_hits(db_conn, frame_path: Path, embedder):
    from kanomori.retrieval import screenshot
    from kanomori.retrieval.merge import merge_from_db

    candidates = screenshot.candidates(
        db_conn,
        frame_path.read_bytes(),
        embedder,
        ocr_reader=screenshot.UploadOcrReader(),
        k=5,
    )
    return candidates, merge_from_db(db_conn, candidates, k=5)


@requires_visual_stack
def test_real_visual_stages_index_sample_clip(db_conn, monkeypatch) -> None:
    sample = _sample_clip()
    if not sample.exists():
        pytest.skip(f"sample clip missing; set {SAMPLE_ENV}=/path/to/clip.mp4")

    video_id, frame_paths, embedder, scene_types = _index_sample(db_conn, monkeypatch, sample)
    _assert_frame_embeddings(db_conn, video_id)
    _assert_scene_segments(db_conn, video_id, scene_types)

    candidates, hits = _screenshot_hits(db_conn, frame_paths[0], embedder)

    assert candidates
    assert {candidate.modality for candidate in candidates} >= {Modality.VISUAL}
    assert hits
    assert hits[0].video_id == video_id
