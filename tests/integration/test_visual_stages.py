from __future__ import annotations

import numpy as np
import pytest

from kanomori.ingest import pipeline
from kanomori.ingest.artifacts import frame_path_for
from kanomori.ingest.stages import classify, image_embed, ocr, register

pytestmark = pytest.mark.requires_db


@pytest.fixture(autouse=True)
def _media_root(tmp_path, monkeypatch):
    """Point MEDIA_ROOT at a tmp dir: the visual stages' compute() globs frames off disk now."""
    monkeypatch.setenv("KANOMORI_MEDIA_ROOT", str(tmp_path / "media"))
    from kanomori.config import get_settings

    get_settings.cache_clear()
    yield
    get_settings.cache_clear()


def _write_frames_on_disk(content_hash: str, timestamps) -> None:
    """Drop deterministically-named JPEGs so compute()'s frames_on_disk() finds them."""
    for ts in timestamps:
        p = frame_path_for(content_hash, ts)
        p.parent.mkdir(parents=True, exist_ok=True)
        p.write_bytes(b"jpeg-bytes")


class SkipStage:
    def __init__(self):
        self.calls = 0

    def run(self, conn, ctx):
        self.calls += 1
        return "skipped"


def _video(conn, content_hash: str = "visualstagehash") -> int:
    return conn.execute(
        "INSERT INTO videos (content_hash, title) VALUES (%s, 'visual') RETURNING id",
        (content_hash,),
    ).fetchone()[0]


def test_pipeline_treats_skipped_stage_as_terminal(db_conn, tmp_path, monkeypatch) -> None:
    media = tmp_path / "clip.mp4"
    media.write_bytes(b"visual-pipeline")
    skip = SkipStage()
    monkeypatch.setattr(pipeline, "STAGES", [("register", register), ("visual", skip)])
    monkeypatch.setattr(pipeline, "make_embedder", lambda: object())

    ctx = pipeline.IngestContext(media_path=str(media))
    pipeline.run_full(db_conn, ctx)
    pipeline.run_full(db_conn, pipeline.IngestContext(media_path=str(media)))

    state = db_conn.execute(
        "SELECT stage_status -> 'visual' ->> 'state' FROM jobs WHERE content_hash = %s",
        (ctx.content_hash,),
    ).fetchone()[0]
    assert state == "skipped"
    assert skip.calls == 1


def test_ocr_stage_inserts_tokenized_text(db_conn, tmp_path, monkeypatch) -> None:
    content_hash = "ocrstagehash"
    vid = _video(db_conn, content_hash)
    # compute() globs frames off disk; persist() resolves frame_id by (video_id, ts_sec), so the
    # DB frame row's ts_sec must equal the value the JPEG's deterministic name encodes.
    _write_frames_on_disk(content_hash, [12.0])
    frame_id = db_conn.execute(
        "INSERT INTO frames (video_id, ts_sec, frame_path) VALUES (%s, 12.0, %s) RETURNING id",
        (vid, str(frame_path_for(content_hash, 12.0))),
    ).fetchone()[0]
    monkeypatch.setattr(
        ocr,
        "read_frame_ocr",
        lambda path: [ocr.OcrResult(text="入口の看板", confidence=0.9, bbox={"x": 1})],
    )

    ctx = pipeline.IngestContext(media_path="unused")
    ctx.video_id = vid
    ctx.content_hash = content_hash
    result = ocr.run(db_conn, ctx)

    row = db_conn.execute(
        "SELECT frame_id, text, confidence, tsv::text FROM ocr_segments WHERE video_id = %s",
        (vid,),
    ).fetchone()
    assert result is None
    assert row[0] == frame_id
    assert row[1] == "入口の看板"
    assert row[2] == pytest.approx(0.9)
    assert "入口" in row[3]


def test_classify_stage_collapses_consecutive_scene_labels(db_conn, tmp_path, monkeypatch) -> None:
    content_hash = "classifystagehash"
    vid = _video(db_conn, content_hash)
    _write_frames_on_disk(content_hash, (0.0, 8.0, 16.0))
    for ts in (0.0, 8.0, 16.0):
        db_conn.execute(
            "INSERT INTO frames (video_id, ts_sec, frame_path) VALUES (%s, %s, %s)",
            (vid, ts, str(frame_path_for(content_hash, ts))),
        )
    labels = iter(
        [
            classify.SceneResult("chatting", 0.8),
            classify.SceneResult("chatting", 0.7),
            classify.SceneResult("gaming", 0.9),
        ]
    )
    monkeypatch.setattr(classify, "classify_frame", lambda path: next(labels))

    ctx = pipeline.IngestContext(media_path="unused")
    ctx.video_id = vid
    ctx.content_hash = content_hash
    result = classify.run(db_conn, ctx)

    rows = db_conn.execute(
        "SELECT start_sec, end_sec, scene_type FROM scene_segments WHERE video_id = %s "
        "ORDER BY start_sec",
        (vid,),
    ).fetchall()
    stream_type = db_conn.execute(
        "SELECT stream_type FROM videos WHERE id = %s", (vid,)
    ).fetchone()[0]
    assert result is None
    assert rows == [(0.0, 16.0, "chatting"), (16.0, 24.0, "gaming")]
    assert stream_type == "chatting"


def test_image_embed_stage_updates_phash_and_embedding(db_conn, tmp_path, monkeypatch) -> None:
    content_hash = "imageembedstagehash"
    vid = _video(db_conn, content_hash)
    _write_frames_on_disk(content_hash, [7.0])
    frame_id = db_conn.execute(
        "INSERT INTO frames (video_id, ts_sec, frame_path) VALUES (%s, 7.0, %s) RETURNING id",
        (vid, str(frame_path_for(content_hash, 7.0))),
    ).fetchone()[0]
    vec = np.ones(768, dtype=np.float32)
    vec /= np.linalg.norm(vec)
    monkeypatch.setattr(image_embed, "compute_frame_phash", lambda path: -1)
    monkeypatch.setattr(image_embed, "embed_frame", lambda path: vec)

    ctx = pipeline.IngestContext(media_path="unused")
    ctx.video_id = vid
    ctx.content_hash = content_hash
    result = image_embed.run(db_conn, ctx)

    row = db_conn.execute(
        "SELECT phash, embedding <=> %s AS distance FROM frames WHERE id = %s",
        (vec, frame_id),
    ).fetchone()
    assert result is None
    assert row[0] == -1
    assert row[1] == pytest.approx(0.0, abs=1e-6)


def test_visual_stages_skip_when_no_frames(db_conn) -> None:
    content_hash = "noframeshash"
    vid = _video(db_conn, content_hash)
    ctx = pipeline.IngestContext(media_path="unused")
    ctx.video_id = vid
    ctx.content_hash = content_hash  # no JPEGs on disk -> compute() returns "skipped"

    assert ocr.run(db_conn, ctx) == "skipped"
    assert classify.run(db_conn, ctx) == "skipped"
    assert image_embed.run(db_conn, ctx) == "skipped"

    assert db_conn.execute("SELECT count(*) FROM ocr_segments").fetchone()[0] == 0
    assert db_conn.execute("SELECT count(*) FROM scene_segments").fetchone()[0] == 0
    assert db_conn.execute("SELECT count(*) FROM frames WHERE phash IS NOT NULL").fetchone()[0] == 0
