"""Round-trip guard: ``ts_from_frame_name`` must invert ``frame_path_for``'s naming.

This is the load-bearing correctness point of the compute/persist split for the visual
stages: ocr/classify/image_embed run ``compute`` with no DB connection, so they recover each
frame's ``ts_sec`` from its deterministic JPEG filename. The persist side then resolves
``frame_id`` by ``(video_id, ts_sec)``, so the ts_sec compute derives MUST equal the value the
frames stage persisted (which is the same value ``frame_path_for`` encoded into the name).
"""

from __future__ import annotations

import pytest

from kanomori.ingest.artifacts import frame_path_for, frames_on_disk, ts_from_frame_name


@pytest.mark.parametrize(
    "ts",
    [0.0, 8.0, 12.5, 3.1, 12.345, 24.0, 16.0, 7.0, 8.027, 1234.999],
)
def test_ts_from_frame_name_inverts_frame_path_for(ts: float) -> None:
    name = frame_path_for("deadbeef", ts).name
    recovered = ts_from_frame_name(name)
    # frame_path_for rounds to milliseconds; the recovered value is that rounded ts.
    assert recovered == round(ts * 1000) / 1000.0


@pytest.mark.parametrize("ts", [0.0, 3.1, 8.0, 12.345, 24.0])
def test_recovered_ts_matches_a_round_t3_planned_value(ts: float) -> None:
    # The frames stage persists ts_sec = round(t, 3); for such values the name round-trip is
    # exact (bit-identical double), so persist's `WHERE ts_sec = %s` resolves the frame.
    planned = round(ts, 3)
    name = frame_path_for("h", planned).name
    assert ts_from_frame_name(name) == planned


def test_frames_on_disk_returns_sorted_ts_path_pairs(tmp_path, monkeypatch) -> None:
    monkeypatch.setenv("KANOMORI_MEDIA_ROOT", str(tmp_path))
    from kanomori.config import get_settings

    get_settings.cache_clear()
    content_hash = "abc123"
    # Write frames out of order; expect them returned sorted by ts_sec.
    for ts in (16.0, 0.0, 8.0):
        p = frame_path_for(content_hash, ts)
        p.parent.mkdir(parents=True, exist_ok=True)
        p.write_bytes(b"jpeg")

    pairs = frames_on_disk(content_hash)

    assert [ts for ts, _ in pairs] == [0.0, 8.0, 16.0]
    assert all(path.name == frame_path_for(content_hash, ts).name for ts, path in pairs)


def test_frames_on_disk_empty_when_no_dir(tmp_path, monkeypatch) -> None:
    monkeypatch.setenv("KANOMORI_MEDIA_ROOT", str(tmp_path))
    from kanomori.config import get_settings

    get_settings.cache_clear()
    assert frames_on_disk("nonexistent") == []
