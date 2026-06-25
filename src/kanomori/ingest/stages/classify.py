"""Stage: classify — assign coarse scene labels to frame spans.

Compute/persist split (Task #12): ``compute`` sources frames from disk (no DB), classifies each,
collapses consecutive same-label frames into spans, and returns a :class:`ClassifyResult` of
:class:`SceneSegmentRow` plus the dominant ``stream_type``. ``persist`` replaces the video's
scene_segments and writes the dominant stream_type onto videos.
"""

from __future__ import annotations

from collections import Counter
from dataclasses import dataclass
from pathlib import Path

from kanomori.ingest.artifacts import frames_on_disk
from kanomori.ingest.stage_result import ClassifyResult, SceneSegmentRow

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
        from kanomori.ingest.stage_device import device_for_stage

        _CLASSIFIER = SigLIPClassifier(
            labels=SCENE_PROMPTS,
            device=device_for_stage("classify"),
        )
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


def build_scene_segments(labels: list[tuple[float, SceneResult]]) -> list[SceneSegmentRow]:
    """Collapse consecutive same-label frames into scene spans (pure; no DB).

    ``labels`` is ``(ts_sec, SceneResult)`` in ascending ts order. Each emitted span runs from its
    first frame's ts to the next differing label's ts (the final span extends one tail-interval
    past the last frame), with confidence averaged over the frames it covers — mirroring the SQL
    the stage wrote before the split.
    """
    timestamps = [ts for ts, _ in labels]
    tail = _tail_interval(timestamps)

    segments: list[SceneSegmentRow] = []

    def _emit(start: float, end: float, span: list[SceneResult]) -> None:
        avg = sum(r.confidence for r in span) / len(span)
        segments.append(
            SceneSegmentRow(
                start_sec=start, end_sec=end, scene_type=span[0].scene_type, confidence=avg
            )
        )

    segment_start = labels[0][0]
    segment_labels = [labels[0][1]]
    for (ts, result), (_prev_ts, prev_result) in zip(labels[1:], labels, strict=False):
        if result.scene_type != prev_result.scene_type:
            _emit(segment_start, ts, segment_labels)
            segment_start = ts
            segment_labels = []
        segment_labels.append(result)
    _emit(segment_start, timestamps[-1] + tail, segment_labels)
    return segments


def run(conn, ctx):
    result = compute(ctx)
    if result == "skipped":
        return "skipped"
    persist(conn, ctx.video_id, result)
    return None


def compute(ctx):
    """Classify on-disk frames into scene spans; return ClassifyResult or "skipped".

    Frames come from :func:`frames_on_disk` (no DB), ts recovered from each JPEG name. Returns the
    "skipped" sentinel when the video has no frames, preserving run()'s contract.
    """
    frames = frames_on_disk(ctx.content_hash)
    if not frames:
        return "skipped"

    labels = [(ts, classify_frame(path)) for ts, path in frames]
    dominant = Counter(result.scene_type for _ts, result in labels).most_common(1)[0][0]
    segments = build_scene_segments(labels)
    return ClassifyResult(segments=segments, stream_type=dominant)


def persist(conn, video_id, result: ClassifyResult) -> None:
    """Replace this video's scene_segments and write the dominant stream_type onto videos."""
    conn.execute("DELETE FROM scene_segments WHERE video_id = %s", (video_id,))
    for seg in result.segments:
        conn.execute(
            """
            INSERT INTO scene_segments (video_id, start_sec, end_sec, scene_type, confidence)
            VALUES (%s, %s, %s, %s, %s)
            """,
            (video_id, seg.start_sec, seg.end_sec, seg.scene_type, seg.confidence),
        )
    conn.execute(
        "UPDATE videos SET stream_type = %s WHERE id = %s", (result.stream_type, video_id)
    )
