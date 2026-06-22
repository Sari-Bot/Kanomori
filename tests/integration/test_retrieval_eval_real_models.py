"""Opt-in retrieval quality smoke for the local sample clip.

This seeds known transcript segments instead of invoking KITS, then exercises the real
transcript, screenshot, and scene-aware merge paths with real embedding models.
"""

from __future__ import annotations

import importlib.util
import os
import shutil
from pathlib import Path

import pytest
from pgvector.psycopg import register_vector

from kanomori.retrieval import merge, screenshot, transcript
from kanomori.text import normalize, tokenize_for_fts

pytestmark = [pytest.mark.requires_db, pytest.mark.requires_models]

SAMPLE_ENV = "KANOMORI_EVAL_SAMPLE"
FIXTURE_PATH = Path(__file__).resolve().parents[1] / "fixtures" / "eval" / "2024_talk_cut.json"
REQUIRED_MODULES = (
    "torch",
    "transformers",
    "PIL",
    "imagehash",
    "rapidocr_onnxruntime",
    "fugashi",
)


def _missing_modules() -> list[str]:
    return [name for name in REQUIRED_MODULES if importlib.util.find_spec(name) is None]


requires_eval_stack = pytest.mark.skipif(
    bool(_missing_modules()) or shutil.which("ffmpeg") is None,
    reason="requires embed+ingest dependency groups and ffmpeg",
)


def _sample_clip(default_relative_path: str) -> Path:
    if env_path := os.environ.get(SAMPLE_ENV):
        return Path(env_path).expanduser()
    return Path(__file__).resolve().parents[2] / default_relative_path


def _seed_transcripts(conn, video_id: int, segments, embedder) -> None:
    register_vector(conn)
    texts = [normalize(segment.text) for segment in segments]
    vectors = embedder.embed_texts(texts)
    for seq, (segment, text, vector) in enumerate(zip(segments, texts, vectors, strict=True)):
        conn.execute(
            """
            INSERT INTO transcript_segments
                (video_id, seq, start_sec, end_sec, text, text_norm, embedding, tsv)
            VALUES (%s, %s, %s, %s, %s, %s, %s, to_tsvector('simple', %s))
            """,
            (
                video_id,
                seq,
                segment.start_sec,
                segment.end_sec,
                segment.text,
                text,
                vector,
                tokenize_for_fts(text),
            ),
        )


def _index_visuals(conn, monkeypatch, sample: Path):
    from kanomori.ingest.pipeline import IngestContext
    from kanomori.ingest.stages import classify, frames, image_embed, ocr, register

    ctx = IngestContext(media_path=str(sample), title="retrieval eval sample")
    register.run(conn, ctx)
    monkeypatch.setattr(frames, "detect_scene_timestamps", lambda media: [])
    monkeypatch.setattr(frames, "probe_duration_sec", lambda media: 9.0)
    frames.run(conn, ctx)
    ocr.run(conn, ctx)
    classify.run(conn, ctx)
    image_embed.run(conn, ctx)
    rows = conn.execute(
        "SELECT ts_sec, frame_path FROM frames WHERE video_id = %s ORDER BY ts_sec",
        (ctx.video_id,),
    ).fetchall()
    return ctx.video_id, [(float(ts), Path(path)) for ts, path in rows], image_embed._embedder()


@requires_eval_stack
def test_real_retrieval_eval_sample_hits_expected_moments(db_conn, monkeypatch) -> None:
    from kanomori.embed.text_embedder import BGEEmbedder
    from kanomori.evaluation import evaluate_hit, load_eval_suite

    suite = load_eval_suite(FIXTURE_PATH)
    sample = _sample_clip(suite.sample)
    if not sample.exists():
        pytest.skip(f"sample clip missing; set {SAMPLE_ENV}=/path/to/clip.mp4")

    video_id, frames, image_embedder = _index_visuals(db_conn, monkeypatch, sample)
    text_embedder = BGEEmbedder()
    _seed_transcripts(db_conn, video_id, suite.transcript_segments, text_embedder)
    db_conn.commit()

    for case in suite.transcript_queries:
        candidates = transcript.candidates(db_conn, case.query, text_embedder, k=suite.top_k)
        hits = merge.merge_from_db(db_conn, candidates, k=suite.top_k)
        result = evaluate_hit(
            case.name,
            hits,
            expected_ts_sec=case.expected_ts_sec,
            tolerance_sec=case.tolerance_sec,
            top_k=suite.top_k,
        )
        assert result.passed, result.summary()

    for case in suite.screenshot_queries:
        expected_ts, frame_path = frames[case.frame_index]
        candidates = screenshot.candidates(
            db_conn,
            frame_path.read_bytes(),
            image_embedder,
            ocr_reader=screenshot.UploadOcrReader(),
            k=suite.top_k,
        )
        hits = merge.merge_from_db(db_conn, candidates, k=suite.top_k)
        result = evaluate_hit(
            case.name,
            hits,
            expected_ts_sec=expected_ts,
            tolerance_sec=case.tolerance_sec,
            top_k=suite.top_k,
        )
        assert result.passed, result.summary()
