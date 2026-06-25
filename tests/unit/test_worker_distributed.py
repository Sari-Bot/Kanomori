"""Pure (DB-free, network-free) tests for the distributed ingestion worker.

``run_one_distributed`` claims a job from the coordinator, fetches its source video, runs each
stage's ``compute`` locally, pushes per-stage results + artifacts back, then completes — with
resume (skip stages_done), fencing (a 409 push aborts before complete), and failure reporting
(an exception triggers ``fail``). These tests drive it with a fake CoordinatorClient that records
calls, a fake MediaSource that writes a dummy file, and monkeypatched stage ``compute`` functions
— so no httpx, no ffmpeg, no KITS, no DB. ``--compute-only`` is exercised through ``main``.
"""

from __future__ import annotations

from pathlib import Path

import pytest

from kanomori.ingest import worker
from kanomori.ingest.pipeline import STAGES
from kanomori.ingest.stage_result import (
    ClassifyResult,
    FramesResult,
    ImageEmbedResult,
    OcrResult,
    ParseTranscriptResult,
    RegisterResult,
    TranscribeResult,
)

STAGE_NAMES = [name for name, _mod in STAGES]
CONTENT_HASH = "deadbeef" * 8


class FakeClient:
    """Records coordinator calls; push/heartbeat/complete/fail return True unless told otherwise."""

    def __init__(self, claim_result, *, push_returns=None):
        self._claim_result = claim_result
        self._push_returns = dict(push_returns or {})
        self.pushed: list[tuple[str, int, str, int, float | None]] = []
        self.completed: list[int] = []
        self.failed: list[tuple[int, str]] = []
        self.heartbeats: list[int] = []

    def claim(self, worker_id, lease_seconds):
        return self._claim_result

    def heartbeat(self, job_id, lease_epoch, lease_seconds):
        self.heartbeats.append(job_id)
        return True

    def push_stage(
        self, job_id, stage_name, lease_epoch, result_json, files, compute_seconds=None
    ):
        self.pushed.append((stage_name, lease_epoch, result_json, len(files), compute_seconds))
        return self._push_returns.get(stage_name, True)

    def complete(self, job_id, lease_epoch):
        self.completed.append(job_id)
        return True

    def fail(self, job_id, lease_epoch, error):
        self.failed.append((job_id, error))
        return True


class FakeSource:
    """A MediaSource whose fetch() writes a dummy file to dest and records the path it served."""

    def __init__(self):
        self.fetched: list[str] = []

    def fetch(self, path, dest):
        self.fetched.append(path)
        dest = Path(dest)
        dest.parent.mkdir(parents=True, exist_ok=True)
        dest.write_bytes(b"dummy-source-bytes")
        return dest

    def read_text(self, path):  # pragma: no cover - not used here
        raise NotImplementedError


def _fake_results():
    """A tiny valid StageResult per stage (the shapes the worker serializes/pushes)."""
    return {
        "register": RegisterResult(content_hash=CONTENT_HASH, title="t"),
        "locate_media": None,
        "transcribe": TranscribeResult(),
        "parse_transcript": ParseTranscriptResult(segments=[]),
        "frames": FramesResult(frames=[], scene_timestamps=[]),
        "ocr": OcrResult(rows=[]),
        "classify": ClassifyResult(segments=[], stream_type="chatting"),
        "image_embed": ImageEmbedResult(rows=[]),
    }


@pytest.fixture
def media_root(tmp_path, monkeypatch):
    """Redirect MEDIA_ROOT to a tmp dir so artifact reads/writes never touch the repo."""
    from kanomori.config import get_settings as _gs

    monkeypatch.setenv("KANOMORI_MEDIA_ROOT", str(tmp_path / "media"))
    _gs.cache_clear()
    yield tmp_path / "media"
    _gs.cache_clear()


@pytest.fixture
def patch_computes(monkeypatch, media_root):
    """Replace every stage's compute() with a stub returning the canned result.

    Stubs mimic the real side effects the worker depends on: register does NOT set
    ctx.content_hash (the worker reads it off the RegisterResult), and transcribe writes its SRT
    artifact to the deterministic path (so the worker can read it back for the multipart push).
    """
    from kanomori.ingest.artifacts import srt_path_for

    results = _fake_results()

    def make_stub(name):
        def stub(ctx):
            if name == "transcribe":
                srt = srt_path_for(CONTENT_HASH)
                srt.parent.mkdir(parents=True, exist_ok=True)
                srt.write_text("1\n00:00:00,000 --> 00:00:01,000\nhi\n", encoding="utf-8")
            return results[name]

        return stub

    from kanomori.ingest.pipeline import STAGES as stages

    for name, module in stages:
        monkeypatch.setattr(module, "compute", make_stub(name))
    return results


@pytest.fixture
def cache_dir(tmp_path, monkeypatch):
    """Point the source-fetch cache at a tmp dir so fetch writes nowhere real."""
    d = tmp_path / "cache"
    monkeypatch.setattr(worker, "make_embedder", lambda: object())
    return d


def _claim(stages_done=None, content_hash=None):
    return {
        "job_id": 42,
        "content_hash": content_hash,
        "lease_epoch": 3,
        "request": {"media_path": "kano/clip.mp4", "title": "t", "separate": False},
        "stages_done": stages_done or [],
    }


def test_idle_returns_false_when_no_job(cache_dir):
    client = FakeClient(claim_result=None)
    ran = worker.run_one_distributed(client, FakeSource(), "w1", 120, cache_dir=cache_dir)
    assert ran is False
    assert client.completed == []


def test_main_logs_startup_and_idle(capsys, monkeypatch, tmp_path):
    source = FakeSource()
    monkeypatch.setattr(worker, "get_media_source", lambda: source)
    monkeypatch.setattr(worker, "CoordinatorClient", lambda *a, **k: FakeClient(claim_result=None))
    monkeypatch.setattr(worker, "get_settings", lambda: _Settings(tmp_path))
    monkeypatch.setattr(worker, "_default_worker_id", lambda: "worker-123")

    worker.main(["--once"])

    captured = capsys.readouterr()
    assert "[worker] startup worker=worker-123 coordinator=http://unused" in captured.out
    assert "[worker] idle worker=worker-123" in captured.out
    assert captured.err == ""


def test_runs_all_stages_in_order_then_completes(patch_computes, cache_dir):
    source = FakeSource()
    client = FakeClient(claim_result=_claim())

    ran = worker.run_one_distributed(client, source, "w1", 120, cache_dir=cache_dir)

    assert ran is True
    assert source.fetched == ["kano/clip.mp4"]
    pushed_stages = [stage for stage, _e, _r, _n, _t in client.pushed]
    assert pushed_stages == STAGE_NAMES  # register first, image_embed last, all in order
    assert client.completed == [42]
    assert client.failed == []
    # Every push carried the claim's lease_epoch.
    assert all(epoch == 3 for _s, epoch, _r, _n, _t in client.pushed)


def test_success_logs_progress_without_verbose_details(
    patch_computes, cache_dir, capsys, monkeypatch
):
    monkeypatch.setattr(worker, "make_embedder", lambda: object())
    source = FakeSource()
    client = FakeClient(claim_result=_claim())

    ran = worker.run_one_distributed(client, source, "w1", 120, cache_dir=cache_dir)

    assert ran is True
    captured = capsys.readouterr()
    assert "[worker] claimed job=42 epoch=3 source=kano/clip.mp4" in captured.out
    assert "[worker] source job=42 action=fetch source=kano/clip.mp4" in captured.out
    assert "[worker] stage start job=42 stage=register" in captured.out
    assert "[worker] stage done job=42 stage=image_embed" in captured.out
    assert "[worker] complete job=42" in captured.out
    assert "content_hash=" not in captured.out
    assert "artifacts=" not in captured.out
    assert captured.err == ""


def test_resume_skips_stages_already_done(patch_computes, cache_dir):
    client = FakeClient(
        claim_result=_claim(
            stages_done=["register", "locate_media", "transcribe"], content_hash=CONTENT_HASH
        )
    )
    worker.run_one_distributed(client, FakeSource(), "w1", 120, cache_dir=cache_dir)

    pushed_stages = [stage for stage, _e, _r, _n, _t in client.pushed]
    assert "register" not in pushed_stages
    assert "transcribe" not in pushed_stages
    assert pushed_stages[0] == "parse_transcript"
    assert client.completed == [42]


def test_resume_logs_skipped_stages(patch_computes, cache_dir, capsys):
    client = FakeClient(
        claim_result=_claim(
            stages_done=["register", "locate_media", "transcribe"], content_hash=CONTENT_HASH
        )
    )

    worker.run_one_distributed(client, FakeSource(), "w1", 120, cache_dir=cache_dir)

    captured = capsys.readouterr()
    assert "[worker] stage resume-skip job=42 stage=register" in captured.out
    assert "[worker] stage resume-skip job=42 stage=locate_media" in captured.out
    assert "[worker] stage resume-skip job=42 stage=transcribe" in captured.out


def test_locate_media_pushes_empty_result_string(patch_computes, cache_dir):
    client = FakeClient(claim_result=_claim())
    worker.run_one_distributed(client, FakeSource(), "w1", 120, cache_dir=cache_dir)
    by_stage = {stage: result_json for stage, _e, result_json, _n, _t in client.pushed}
    assert by_stage["locate_media"] == ""
    # A model-bearing stage serializes to JSON (non-empty).
    assert by_stage["register"] and by_stage["register"] != ""


def test_409_push_aborts_before_complete(patch_computes, cache_dir):
    # transcribe push returns False (lease lost) -> stop, do NOT complete, do NOT fail.
    client = FakeClient(claim_result=_claim(), push_returns={"transcribe": False})
    ran = worker.run_one_distributed(client, FakeSource(), "w1", 120, cache_dir=cache_dir)

    assert ran is True
    pushed_stages = [stage for stage, _e, _r, _n, _t in client.pushed]
    assert pushed_stages == ["register", "locate_media", "transcribe"]  # stopped at the 409
    assert client.completed == []
    assert client.failed == []


def test_409_push_logs_lease_lost_and_not_complete(patch_computes, cache_dir, capsys):
    client = FakeClient(claim_result=_claim(), push_returns={"transcribe": False})

    worker.run_one_distributed(client, FakeSource(), "w1", 120, cache_dir=cache_dir)

    captured = capsys.readouterr()
    assert "[worker] lease-lost job=42 epoch=3 stage=transcribe reason=push-409" in captured.err
    assert "[worker] complete job=42" not in captured.out


def test_push_stage_includes_rounded_compute_seconds(
    patch_computes, cache_dir, monkeypatch
):
    ticks = iter(
        [float(i) for pair in ((n, n + 0.1234) for n in range(len(STAGE_NAMES))) for i in pair]
    )
    monkeypatch.setattr(worker.time, "perf_counter", lambda: next(ticks))
    client = FakeClient(claim_result=_claim())

    worker.run_one_distributed(client, FakeSource(), "w1", 120, cache_dir=cache_dir)

    assert [seconds for _s, _e, _r, _n, seconds in client.pushed] == [0.123] * len(STAGE_NAMES)


def test_exception_triggers_fail(patch_computes, cache_dir, monkeypatch):
    from kanomori.ingest.stages import parse_transcript

    def boom(ctx):
        raise RuntimeError("embedder exploded")

    monkeypatch.setattr(parse_transcript, "compute", boom)
    client = FakeClient(claim_result=_claim())
    ran = worker.run_one_distributed(client, FakeSource(), "w1", 120, cache_dir=cache_dir)

    assert ran is True
    assert client.completed == []
    assert len(client.failed) == 1
    job_id, error = client.failed[0]
    assert job_id == 42
    assert "embedder exploded" in error


def test_exception_logs_failure_summary(patch_computes, cache_dir, capsys, monkeypatch):
    from kanomori.ingest.stages import parse_transcript

    def boom(ctx):
        raise RuntimeError("embedder exploded")

    monkeypatch.setattr(parse_transcript, "compute", boom)
    client = FakeClient(claim_result=_claim())

    worker.run_one_distributed(client, FakeSource(), "w1", 120, cache_dir=cache_dir)

    captured = capsys.readouterr()
    assert "[worker] failed job=42 error=embedder exploded" in captured.err


def test_self_reclaim_cache_skips_refetch(patch_computes, cache_dir):
    """A second run for the same source path reuses the cached local file (no re-fetch)."""
    source = FakeSource()
    client = FakeClient(claim_result=_claim())
    worker.run_one_distributed(client, source, "w1", 120, cache_dir=cache_dir)
    assert source.fetched == ["kano/clip.mp4"]

    client2 = FakeClient(claim_result=_claim())
    worker.run_one_distributed(client2, source, "w1", 120, cache_dir=cache_dir)
    # Cache hit: fetch not called a second time.
    assert source.fetched == ["kano/clip.mp4"]


def test_verbose_logs_claim_cache_and_artifact_details(
    patch_computes, cache_dir, capsys, monkeypatch
):
    monkeypatch.setattr(worker, "make_embedder", lambda: object())
    local = worker._cache_dest(cache_dir, "kano/clip.mp4")
    local.parent.mkdir(parents=True, exist_ok=True)
    local.write_bytes(b"cached")
    client = FakeClient(
        claim_result=_claim(stages_done=["register"], content_hash=CONTENT_HASH)
    )

    ran = worker.run_one_distributed(
        client,
        FakeSource(),
        "w1",
        120,
        cache_dir=cache_dir,
        verbose=True,
    )

    assert ran is True
    captured = capsys.readouterr()
    assert (
        "[worker] detail claimed job=42 epoch=3 content_hash="
        f"{CONTENT_HASH} stages_done=register separate=False language=japanese"
        in captured.out
    )
    assert f"cache_path={local}" in captured.out
    assert "[worker] source detail job=42 action=reuse cache_path=" in captured.out
    assert (
        "[worker] stage detail job=42 stage=locate_media "
        "outcome=no-model result_bytes=0 artifacts=0"
        in captured.out
    )
    assert (
        "[worker] stage detail job=42 stage=transcribe outcome=result result_bytes=50 artifacts=1"
        in captured.out
    )
    assert "[worker] push ok job=42 stage=transcribe artifacts=1" in captured.out


def test_verbose_relays_live_stage_output(patch_computes, cache_dir, capsys, monkeypatch):
    from kanomori.ingest.artifacts import srt_path_for
    from kanomori.ingest.stages import transcribe

    def verbose_transcribe(ctx):
        callback = getattr(ctx, "stage_log", None)
        if callback is not None:
            callback("transcribe", "stdout", "segment 1/4")
        srt = srt_path_for(CONTENT_HASH)
        srt.parent.mkdir(parents=True, exist_ok=True)
        srt.write_text("1\n00:00:00,000 --> 00:00:01,000\nhi\n", encoding="utf-8")
        return TranscribeResult()

    monkeypatch.setattr(transcribe, "compute", verbose_transcribe)

    worker.run_one_distributed(
        FakeClient(claim_result=_claim()),
        FakeSource(),
        "w1",
        120,
        cache_dir=cache_dir,
        verbose=True,
    )

    captured = capsys.readouterr()
    assert (
        "[worker] stage log job=42 stage=transcribe stream=stdout line=segment 1/4"
        in captured.out
    )


# --- --compute-only ----------------------------------------------------------------------


def test_compute_only_runs_chain_without_coordinator(patch_computes, monkeypatch, tmp_path):
    """--compute-only runs the compute chain over a manifest sample, round-trips each result,
    and never constructs/uses a CoordinatorClient."""
    source = FakeSource()
    monkeypatch.setattr(worker, "get_media_source", lambda: source)
    monkeypatch.setattr(
        worker, "iter_manifest", lambda src: [{"path": "kano/clip.mp4", "title": "sample"}]
    )
    monkeypatch.setattr(worker, "make_embedder", lambda: object())

    def _no_client(*a, **k):  # pragma: no cover - asserts it's never built
        raise AssertionError("--compute-only must not talk to a coordinator")

    monkeypatch.setattr(worker, "CoordinatorClient", _no_client)
    monkeypatch.setattr(worker, "get_settings", lambda: _Settings(tmp_path))

    # Should run to completion without raising.
    worker.main(["--compute-only"])
    assert source.fetched == ["kano/clip.mp4"]


class _Settings:
    def __init__(self, root):
        self.media_root = root
        self.coordinator_url = "http://unused"
        self.coordinator_token = "unused"
