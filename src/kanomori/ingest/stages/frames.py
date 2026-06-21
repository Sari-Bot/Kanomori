"""Stage: frames — extract short-preview thumbnails for visual retrieval."""

from __future__ import annotations

import subprocess
from pathlib import Path

from kanomori.ingest.artifacts import frame_dir_for, frame_path_for

DEFAULT_INTERVAL_SEC = 8.0
DEDUP_TOLERANCE_SEC = 0.25


def _run(argv: list[str], **kwargs) -> subprocess.CompletedProcess:
    return subprocess.run(argv, capture_output=True, text=True, **kwargs)


def plan_sample_timestamps(
    duration_sec: float,
    scene_timestamps: list[float],
    *,
    interval_sec: float = DEFAULT_INTERVAL_SEC,
) -> list[float]:
    """Combine fixed interval samples with scene-boundary samples and deduplicate."""
    if duration_sec <= 0:
        return []

    samples: list[float] = []
    t = 0.0
    while t < duration_sec:
        samples.append(round(t, 3))
        t += interval_sec
    samples.extend(round(ts, 3) for ts in scene_timestamps if 0 <= ts < duration_sec)
    samples.sort()

    out: list[float] = []
    for ts in samples:
        if out and abs(ts - out[-1]) <= DEDUP_TOLERANCE_SEC:
            continue
        out.append(ts)
    return out


def probe_duration_sec(media_path: Path) -> float | None:
    stream = _run(
        [
            "ffprobe", "-v", "error",
            "-select_streams", "v:0",
            "-show_entries", "stream=index",
            "-of", "csv=p=0",
            str(media_path),
        ]
    )
    if stream.returncode != 0 or not (stream.stdout or "").strip():
        return None

    result = _run(
        [
            "ffprobe", "-v", "error",
            "-show_entries", "format=duration",
            "-of", "default=noprint_wrappers=1:nokey=1",
            str(media_path),
        ]
    )
    if result.returncode != 0:
        return None
    try:
        duration = float((result.stdout or "").strip())
    except ValueError:
        return None
    return duration if duration > 0 else None


def detect_scene_timestamps(media_path: Path) -> list[float]:
    """Return scene start times via PySceneDetect; dependency is optional until this runs."""
    from scenedetect import ContentDetector, SceneManager, open_video

    video = open_video(str(media_path))
    manager = SceneManager()
    manager.add_detector(ContentDetector())
    manager.detect_scenes(video)
    scenes = manager.get_scene_list()
    return [start.get_seconds() for start, _end in scenes[1:]]


def extract_frame(media_path: Path, ts_sec: float, out_path: Path) -> None:
    out_path.parent.mkdir(parents=True, exist_ok=True)
    result = _run(
        [
            "ffmpeg", "-ss", f"{ts_sec:.3f}", "-i", str(media_path),
            "-frames:v", "1", "-vf", "scale='min(640,iw)':-2",
            "-q:v", "4", str(out_path), "-y",
        ]
    )
    if result.returncode != 0:
        raise RuntimeError(f"ffmpeg frame extraction failed: {(result.stderr or '')[:300]}")


def run(conn, ctx):
    media = Path(ctx.media_path)
    duration = probe_duration_sec(media)
    if duration is None:
        return "skipped"

    frame_dir_for(ctx.content_hash).mkdir(parents=True, exist_ok=True)
    scene_times = detect_scene_timestamps(media)
    timestamps = plan_sample_timestamps(duration, scene_times)
    if not timestamps:
        return "skipped"

    conn.execute("DELETE FROM frames WHERE video_id = %s", (ctx.video_id,))
    for ts in timestamps:
        frame_path = frame_path_for(ctx.content_hash, ts)
        extract_frame(media, ts, frame_path)
        conn.execute(
            """
            INSERT INTO frames (video_id, ts_sec, frame_path)
            VALUES (%s, %s, %s)
            ON CONFLICT (video_id, ts_sec) DO UPDATE
            SET frame_path = EXCLUDED.frame_path
            """,
            (ctx.video_id, ts, str(frame_path)),
        )
