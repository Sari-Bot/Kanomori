from types import SimpleNamespace

from kanomori.ingest.stages import frames
from kanomori.ingest.stages.frames import plan_sample_timestamps


def test_plan_sample_timestamps_combines_interval_and_scene_points() -> None:
    timestamps = plan_sample_timestamps(
        duration_sec=25.0,
        scene_timestamps=[3.1, 8.0, 8.02],
        interval_sec=8.0,
    )

    assert timestamps == [0.0, 3.1, 8.0, 16.0, 24.0]


def test_frames_stage_skips_when_no_video_stream(monkeypatch) -> None:
    monkeypatch.setattr(frames, "probe_duration_sec", lambda path, **kwargs: None)
    ctx = SimpleNamespace(media_path="audio.mp3", content_hash="abc", video_id=1)

    assert frames.run(object(), ctx) == "skipped"


def test_frames_compute_relays_subprocess_output(monkeypatch, tmp_path) -> None:
    seen: list[tuple[str, str, str]] = []

    def fake_run(argv, **kwargs):
        callback = kwargs["log_output"]
        if argv[0] == "ffprobe" and "stream=index" in argv:
            callback("stderr", "probing stream")
            return SimpleNamespace(returncode=0, stdout="0\n", stderr="")
        if argv[0] == "ffprobe":
            callback("stdout", "12.5")
            return SimpleNamespace(returncode=0, stdout="12.5\n", stderr="")
        callback("stderr", "extracting frame")
        out_path = argv[argv.index("-y") - 1]
        tmp_path.joinpath(out_path).write_bytes(b"jpeg")
        return SimpleNamespace(returncode=0, stdout="", stderr="")

    monkeypatch.setattr(frames, "_run", fake_run)
    monkeypatch.setattr(frames, "detect_scene_timestamps", lambda media: [])
    ctx = SimpleNamespace(
        media_path="clip.mp4",
        content_hash="abc123",
        stage_log=lambda stage, stream, line: seen.append((stage, stream, line)),
    )
    monkeypatch.setenv("KANOMORI_MEDIA_ROOT", str(tmp_path))
    from kanomori.config import get_settings

    get_settings.cache_clear()

    result = frames.compute(ctx)

    assert result.frames
    assert ("frames", "stderr", "probing stream") in seen
    assert ("frames", "stdout", "12.5") in seen
    assert ("frames", "stderr", "extracting frame") in seen
