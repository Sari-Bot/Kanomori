"""Unit tests for the per-stage compute/persist split (Task #12).

``compute`` is the pure/heavy half (no DB connection): these tests monkeypatch the same seams
the existing stage tests use (KITS, embedder, OCR, classifier, frame extraction) and assert
compute returns the right StageResult shape — including the disk-globbing frame source for
ocr/classify/image_embed and the round-trip ts_sec the persist side resolves on.

``persist`` is exercised against a recording fake connection (mirroring test_register_identity)
to assert the SQL branch/shape without a live database. End-to-end SQL semantics stay covered by
the ``requires_db`` integration tests (which call run() and need real Postgres + pgvector).
"""

from __future__ import annotations

from pathlib import Path
from types import SimpleNamespace

import numpy as np
import pytest

from kanomori.ingest import stage_result as sr
from kanomori.ingest.artifacts import frame_path_for
from kanomori.ingest.stages import (
    classify,
    frames,
    image_embed,
    locate_media,
    ocr,
    parse_transcript,
    register,
    transcribe,
)

# --- recording fake connection ----------------------------------------------------------


class _FakeCursor:
    def __init__(self, conn: _FakeConn) -> None:
        self._conn = conn

    def __enter__(self) -> _FakeCursor:
        return self

    def __exit__(self, *exc) -> None:
        return None

    def execute(self, sql, params=None):
        self._conn.calls.append((sql, params))

    def fetchone(self):
        return None

    def fetchall(self):
        return []


class _FakeConn:
    """Records (sql, params); returns canned rows for the frame-id resolution SELECT."""

    def __init__(self, *, frame_rows: list[tuple[int, float]] | None = None) -> None:
        self.calls: list[tuple[str, object]] = []
        self._frame_rows = frame_rows or []

    def execute(self, sql, params=None):
        self.calls.append((sql, params))
        return _FakeResult(self, sql)

    def cursor(self):
        return _FakeCursor(self)

    def _sql(self) -> str:
        return "\n".join(s for s, _ in self.calls)


class _FakeResult:
    def __init__(self, conn: _FakeConn, sql: str) -> None:
        self._conn = conn
        self._sql = sql

    def fetchone(self):
        if "RETURNING id" in self._sql:
            return (42,)
        return None

    def fetchall(self):
        if "SELECT id, ts_sec FROM frames" in self._sql:
            return self._conn._frame_rows
        return []


@pytest.fixture(autouse=True)
def _no_register_vector(monkeypatch):
    # parse_transcript/image_embed persist call register_vector(conn); the fake conn isn't a real
    # psycopg connection, so stub it out for these pure-SQL-shape tests.
    monkeypatch.setattr(parse_transcript, "register_vector", lambda conn: None)
    monkeypatch.setattr(image_embed, "register_vector", lambda conn: None)


@pytest.fixture
def media_root(tmp_path, monkeypatch):
    monkeypatch.setenv("KANOMORI_MEDIA_ROOT", str(tmp_path))
    from kanomori.config import get_settings

    get_settings.cache_clear()
    return tmp_path


def _write_frames(content_hash: str, timestamps: list[float]) -> None:
    for ts in timestamps:
        p = frame_path_for(content_hash, ts)
        p.parent.mkdir(parents=True, exist_ok=True)
        p.write_bytes(b"jpeg")


# --- register ----------------------------------------------------------------------------


def test_register_compute_returns_result_without_touching_db(tmp_path) -> None:
    media = tmp_path / "clip.mp4"
    media.write_bytes(b"abc")
    ctx = SimpleNamespace(
        media_path=str(media), source_url="https://x.test/v", source_platform="youtube",
        title="t", stream_type=None,
    )

    result = register.compute(ctx)

    assert isinstance(result, sr.RegisterResult)
    assert len(result.content_hash) == 64
    assert result.source_url == "https://x.test/v"
    # media_path is deliberately not on the wire contract.
    assert "media_path" not in result.model_dump()


def test_register_persist_returns_video_id_and_threads_media_path(tmp_path) -> None:
    conn = _FakeConn()
    result = sr.RegisterResult(content_hash="a" * 64, title="t")

    video_id = register.persist(conn, result, media_path="/local/clip.mp4", job_id=None)

    assert video_id == 42
    insert = next(p for s, p in conn.calls if "INSERT INTO videos" in s)
    assert "/local/clip.mp4" in insert  # single-machine wrapper persists media_path
    assert "INSERT INTO jobs" in conn._sql()
    assert "UPDATE jobs SET content_hash" not in conn._sql()


def test_register_persist_omits_media_path_on_wire_path() -> None:
    conn = _FakeConn()
    result = sr.RegisterResult(content_hash="b" * 64)

    register.persist(conn, result, media_path=None, job_id=None)

    insert = next(p for s, p in conn.calls if "INSERT INTO videos" in s)
    assert insert[4] is None  # media_path column is NULL on the coordinator path


def test_register_persist_reconciles_job_by_id_when_job_id_given() -> None:
    conn = _FakeConn()
    result = sr.RegisterResult(content_hash="c" * 64)

    register.persist(conn, result, media_path=None, job_id=99)

    update = next((s, p) for s, p in conn.calls if "UPDATE jobs SET content_hash" in s)
    assert update[1] == ("c" * 64, 42, 99)
    assert "INSERT INTO jobs" not in conn._sql()


# --- locate_media ------------------------------------------------------------------------


def test_locate_media_compute_sets_ctx_and_returns_none(media_root, monkeypatch) -> None:
    def fake_runner(argv, **kwargs):
        Path(argv[argv.index("-y") - 1]).write_bytes(b"RIFF")
        return SimpleNamespace(returncode=0, stderr="")

    monkeypatch.setattr(locate_media, "_run", fake_runner)
    ctx = SimpleNamespace(
        media_path="/in/clip.mp4", content_hash="hashloc", title="歌枠", separate=False
    )

    result = locate_media.compute(ctx)

    assert result is None
    assert ctx.audio_path.endswith("hashloc/audio.wav")
    assert ctx.separate is True  # karaoke title -> --separate
    assert locate_media.persist(None, 1, None) is None


# --- transcribe --------------------------------------------------------------------------


def test_transcribe_compute_writes_srt_and_returns_result(media_root, monkeypatch) -> None:
    def fake_kits(audio, out_srt, **kwargs):
        Path(out_srt).parent.mkdir(parents=True, exist_ok=True)
        Path(out_srt).write_text("1\n00:00:00,000 --> 00:00:01,000\nx\n", encoding="utf-8")
        return Path(out_srt)

    monkeypatch.setattr(transcribe, "kits_transcribe", fake_kits)
    ctx = SimpleNamespace(
        content_hash="hashtr", audio_path=None, separate=False, language="japanese"
    )

    result = transcribe.compute(ctx)

    assert isinstance(result, sr.TranscribeResult)
    assert [(a.name, a.kind) for a in result.artifacts()] == [("transcript.srt", "srt")]
    assert ctx.srt_path.endswith("hashtr/transcript.srt")
    assert transcribe.persist(None, 1, result) is None


# --- parse_transcript --------------------------------------------------------------------


class _StubEmbedder:
    def embed_texts(self, texts):
        return [np.arange(4, dtype=np.float32) + i for i, _ in enumerate(texts)]


def test_parse_transcript_compute_builds_rows_with_embeddings(media_root, monkeypatch) -> None:
    monkeypatch.setattr(
        parse_transcript,
        "parse_srt",
        lambda text: [
            {"start": 0.0, "end": 1.0, "text": "こんにちは"},
            {"start": 1.0, "end": 2.0, "text": "またね"},
        ],
    )
    srt = Path(media_root) / "x.srt"
    srt.write_text("dummy", encoding="utf-8")
    ctx = SimpleNamespace(content_hash="h", srt_path=str(srt), embedder=_StubEmbedder())

    result = parse_transcript.compute(ctx)

    assert isinstance(result, sr.ParseTranscriptResult)
    assert [s.seq for s in result.segments] == [0, 1]
    assert [s.text for s in result.segments] == ["こんにちは", "またね"]
    # embeddings survive the codec round-trip
    np.testing.assert_allclose(result.segments[0].vector(), np.arange(4), atol=1e-6)


def test_parse_transcript_persist_deletes_then_inserts_with_tsvector() -> None:
    conn = _FakeConn()
    seg = sr.TranscriptSegmentRow.build(
        seq=0, start_sec=0.0, end_sec=1.0, text="やあ", text_norm="やあ",
        embedding=np.ones(4, dtype=np.float32),
    )
    parse_transcript.persist(conn, 7, sr.ParseTranscriptResult(segments=[seg]))

    sql = conn._sql()
    assert "DELETE FROM transcript_segments WHERE video_id" in sql
    insert = next(p for s, p in conn.calls if "INSERT INTO transcript_segments" in s)
    assert insert[0] == 7  # video_id threaded as a param, not read from ctx
    assert "to_tsvector('simple', %s)" in conn._sql()


# --- frames ------------------------------------------------------------------------------


def test_frames_compute_skips_when_no_duration(monkeypatch) -> None:
    monkeypatch.setattr(frames, "probe_duration_sec", lambda media, **kwargs: None)
    ctx = SimpleNamespace(media_path="audio.mp3", content_hash="h")
    assert frames.compute(ctx) == "skipped"


def test_frames_compute_extracts_and_returns_framesresult(media_root, monkeypatch) -> None:
    monkeypatch.setattr(frames, "probe_duration_sec", lambda media, **kwargs: 20.0)
    monkeypatch.setattr(frames, "detect_scene_timestamps", lambda media: [3.1])
    extracted: list[float] = []

    def fake_extract(media, ts, out_path, **kwargs):
        out_path.parent.mkdir(parents=True, exist_ok=True)
        out_path.write_bytes(b"jpeg")
        extracted.append(ts)

    monkeypatch.setattr(frames, "extract_frame", fake_extract)
    ctx = SimpleNamespace(media_path="clip.mp4", content_hash="hframes")

    result = frames.compute(ctx)

    assert isinstance(result, sr.FramesResult)
    assert result.scene_timestamps == [3.1]
    # ts_sec of every row equals what frame_path_for would assign (persist resolves on this).
    for row in result.frames:
        assert frame_path_for("hframes", row.ts_sec).name == row.artifact
    assert [r.ts_sec for r in result.frames] == [0.0, 3.1, 8.0, 16.0]


def test_frames_persist_rederives_path_from_content_hash() -> None:
    conn = _FakeConn()
    result = sr.FramesResult(
        frames=[sr.FrameRow(ts_sec=8.0, artifact=frame_path_for("h", 8.0).name)],
        scene_timestamps=[],
    )
    frames.persist(conn, 5, result, content_hash="hcontent")

    assert "DELETE FROM frames WHERE video_id" in conn._sql()
    insert = next(p for s, p in conn.calls if "INSERT INTO frames" in s)
    assert insert[0] == 5 and insert[1] == 8.0
    assert insert[2].endswith(frame_path_for("hcontent", 8.0).name)


# --- ocr / classify / image_embed: disk frame sourcing ----------------------------------


def test_ocr_compute_globs_frames_and_keys_rows_by_ts(media_root, monkeypatch) -> None:
    _write_frames("hocr", [0.0, 8.0])
    monkeypatch.setattr(
        ocr, "read_frame_ocr",
        lambda path: [ocr.OcrResult(text="看板", confidence=0.9, bbox={"x": 1})],
    )
    ctx = SimpleNamespace(content_hash="hocr")

    result = ocr.compute(ctx)

    assert isinstance(result, sr.OcrResult)
    assert sorted(r.ts_sec for r in result.rows) == [0.0, 8.0]
    # ts_sec must equal the value frame_path_for assigns, so persist resolves frame_id.
    for r in result.rows:
        assert frame_path_for("hocr", r.ts_sec).name == frame_path_for("hocr", r.ts_sec).name


def test_ocr_compute_skips_when_no_frames(media_root) -> None:
    assert ocr.compute(SimpleNamespace(content_hash="empty")) == "skipped"


def test_ocr_persist_resolves_frame_id_by_ts_sec() -> None:
    conn = _FakeConn(frame_rows=[(101, 0.0), (102, 8.0)])
    result = sr.OcrResult(rows=[sr.OcrRow(ts_sec=8.0, text="x", confidence=0.5, bbox={})])

    ocr.persist(conn, 7, result)

    assert "DELETE FROM ocr_segments WHERE video_id" in conn._sql()
    insert = next(p for s, p in conn.calls if "INSERT INTO ocr_segments" in s)
    assert insert[0] == 7 and insert[1] == 102 and insert[2] == 8.0  # frame_id resolved by ts


def test_classify_compute_builds_segments_from_disk(media_root, monkeypatch) -> None:
    _write_frames("hcls", [0.0, 8.0, 16.0])
    labels = iter(
        [
            classify.SceneResult("chatting", 0.8),
            classify.SceneResult("chatting", 0.7),
            classify.SceneResult("gaming", 0.9),
        ]
    )
    monkeypatch.setattr(classify, "classify_frame", lambda path: next(labels))

    result = classify.compute(SimpleNamespace(content_hash="hcls"))

    assert isinstance(result, sr.ClassifyResult)
    assert result.stream_type == "chatting"
    spans = [(s.start_sec, s.end_sec, s.scene_type) for s in result.segments]
    assert spans == [(0.0, 16.0, "chatting"), (16.0, 24.0, "gaming")]


def test_classify_compute_skips_when_no_frames(media_root) -> None:
    assert classify.compute(SimpleNamespace(content_hash="empty")) == "skipped"


def test_classify_persist_inserts_segments_and_updates_stream_type() -> None:
    conn = _FakeConn()
    result = sr.ClassifyResult(
        segments=[sr.SceneSegmentRow(start_sec=0.0, end_sec=16.0, scene_type="chatting",
                                     confidence=0.75)],
        stream_type="chatting",
    )
    classify.persist(conn, 9, result)

    assert "DELETE FROM scene_segments WHERE video_id" in conn._sql()
    assert "INSERT INTO scene_segments" in conn._sql()
    update = next((s, p) for s, p in conn.calls if "UPDATE videos SET stream_type" in s)
    assert update[1] == ("chatting", 9)


def test_image_embed_compute_builds_rows_from_disk(media_root, monkeypatch) -> None:
    _write_frames("himg", [0.0, 8.0])
    vec = np.ones(768, dtype=np.float32)
    monkeypatch.setattr(image_embed, "compute_frame_phash", lambda path: 123)
    monkeypatch.setattr(image_embed, "embed_frame", lambda path: vec)

    result = image_embed.compute(SimpleNamespace(content_hash="himg"))

    assert isinstance(result, sr.ImageEmbedResult)
    assert sorted(r.ts_sec for r in result.rows) == [0.0, 8.0]
    assert all(r.phash == 123 for r in result.rows)
    np.testing.assert_allclose(result.rows[0].vector(), vec, atol=1e-6)


def test_image_embed_compute_skips_when_no_frames(media_root) -> None:
    assert image_embed.compute(SimpleNamespace(content_hash="empty")) == "skipped"


def test_image_embed_persist_resolves_frame_id_by_ts_sec() -> None:
    conn = _FakeConn(frame_rows=[(201, 0.0), (202, 8.0)])
    result = sr.ImageEmbedResult(
        rows=[sr.ImageEmbedRow.build(ts_sec=8.0, phash=5, embedding=np.ones(768, dtype=np.float32))]
    )
    image_embed.persist(conn, 7, result)

    update = next(p for s, p in conn.calls if "UPDATE frames SET phash" in s)
    assert update[0] == 5 and update[2] == 202  # phash, embedding, frame_id (resolved by ts)
