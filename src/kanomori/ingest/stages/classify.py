"""Stage: classify — assign coarse scene labels to frame spans."""

from __future__ import annotations

from collections import Counter
from dataclasses import dataclass
from pathlib import Path

SCENE_PROMPTS = {
    "singing": ["VTuber singing karaoke on stream", "歌枠で歌っている配信画面"],
    "chatting": ["VTuber talking with chat", "雑談配信の画面"],
    "gaming": ["VTuber gameplay stream", "ゲーム実況配信の画面"],
    "waiting": ["stream waiting screen", "配信待機画面"],
    "superchat": ["superchat reading stream", "スーパーチャット読みの画面"],
    "announcement": ["announcement slide on stream", "告知画面"],
    "collab": ["multiple VTubers collaboration stream", "コラボ配信の画面"],
}


@dataclass(frozen=True)
class SceneResult:
    scene_type: str
    confidence: float


_CLASSIFIER = None


def _classifier():
    global _CLASSIFIER
    if _CLASSIFIER is None:
        from kanomori.embed.image_embedder import SigLIPClassifier

        _CLASSIFIER = SigLIPClassifier(labels=SCENE_PROMPTS)
    return _CLASSIFIER


def classify_frame(path: Path) -> SceneResult:
    from PIL import Image

    with Image.open(path) as image:
        scores = _classifier().classify_image(image.convert("RGB"))
    scene_type, confidence = max(scores.items(), key=lambda item: item[1])
    return SceneResult(scene_type=scene_type, confidence=confidence)


def _tail_interval(timestamps: list[float]) -> float:
    if len(timestamps) < 2:
        return 8.0
    gaps = [b - a for a, b in zip(timestamps, timestamps[1:], strict=False) if b > a]
    return min(gaps) if gaps else 8.0


def _insert_segment(
    conn,
    video_id: int,
    start: float,
    end: float,
    labels: list[SceneResult],
) -> None:
    avg = sum(r.confidence for r in labels) / len(labels)
    conn.execute(
        """
        INSERT INTO scene_segments (video_id, start_sec, end_sec, scene_type, confidence)
        VALUES (%s, %s, %s, %s, %s)
        """,
        (video_id, start, end, labels[0].scene_type, avg),
    )


def run(conn, ctx):
    rows = conn.execute(
        "SELECT frame_path, ts_sec FROM frames WHERE video_id = %s ORDER BY ts_sec",
        (ctx.video_id,),
    ).fetchall()
    if not rows:
        return "skipped"

    conn.execute("DELETE FROM scene_segments WHERE video_id = %s", (ctx.video_id,))
    timestamps = [float(ts) for _path, ts in rows]
    tail = _tail_interval(timestamps)
    labels = [(ts, classify_frame(Path(path))) for path, ts in rows]
    dominant = Counter(result.scene_type for _ts, result in labels).most_common(1)[0][0]

    segment_start = labels[0][0]
    segment_labels = [labels[0][1]]
    for (ts, result), (_prev_ts, prev_result) in zip(labels[1:], labels, strict=False):
        if result.scene_type != prev_result.scene_type:
            _insert_segment(conn, ctx.video_id, segment_start, ts, segment_labels)
            segment_start = ts
            segment_labels = []
        segment_labels.append(result)
    _insert_segment(conn, ctx.video_id, segment_start, timestamps[-1] + tail, segment_labels)
    conn.execute("UPDATE videos SET stream_type = %s WHERE id = %s", (dominant, ctx.video_id))
    return None
