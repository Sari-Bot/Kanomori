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
    monkeypatch.setattr(frames, "probe_duration_sec", lambda path: None)
    ctx = SimpleNamespace(media_path="audio.mp3", content_hash="abc", video_id=1)

    assert frames.run(object(), ctx) == "skipped"
