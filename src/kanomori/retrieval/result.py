"""Result-detail assembly for the moment-detail view (backs GET /result/{video_id}).

Given a (video_id, ts_sec), gather everything the UI shows for one moment: video metadata +
source link, nearby transcript (±window), preview frames (±window), OCR context (±window), and
the scene_type at that timestamp. All lookups use the `(video_id, *_sec)` indexes from the
migrations. Returns None if the video doesn't exist.
"""

from __future__ import annotations

from dataclasses import dataclass, field

from kanomori.retrieval.merge import scene_at

DEFAULT_WINDOW = 15.0


@dataclass
class TranscriptLine:
    start_sec: float
    end_sec: float
    text: str


@dataclass
class FramePreview:
    ts_sec: float
    frame_path: str


@dataclass
class OcrLine:
    ts_sec: float
    text: str


@dataclass
class ResultDetail:
    video_id: int
    ts_sec: float
    title: str | None
    source_url: str | None
    scene_type: str | None
    nearby_transcript: list[TranscriptLine] = field(default_factory=list)
    preview_frames: list[FramePreview] = field(default_factory=list)
    ocr_context: list[OcrLine] = field(default_factory=list)


def result_detail(conn, video_id: int, ts_sec: float, *, window: float = DEFAULT_WINDOW):
    """Assemble the moment-detail view for (video_id, ts_sec). None if the video is unknown."""
    video = conn.execute(
        "SELECT title, source_url FROM videos WHERE id = %s", (video_id,)
    ).fetchone()
    if video is None:
        return None
    title, source_url = video

    lo, hi = ts_sec - window, ts_sec + window

    transcript_rows = conn.execute(
        "SELECT start_sec, end_sec, text FROM transcript_segments "
        "WHERE video_id = %s AND start_sec BETWEEN %s AND %s ORDER BY start_sec",
        (video_id, lo, hi),
    ).fetchall()
    frame_rows = conn.execute(
        "SELECT ts_sec, frame_path FROM frames "
        "WHERE video_id = %s AND ts_sec BETWEEN %s AND %s ORDER BY ts_sec",
        (video_id, lo, hi),
    ).fetchall()
    ocr_rows = conn.execute(
        "SELECT ts_sec, text FROM ocr_segments "
        "WHERE video_id = %s AND ts_sec BETWEEN %s AND %s ORDER BY ts_sec",
        (video_id, lo, hi),
    ).fetchall()

    return ResultDetail(
        video_id=video_id,
        ts_sec=ts_sec,
        title=title,
        source_url=source_url,
        scene_type=scene_at(conn, video_id, ts_sec),
        nearby_transcript=[TranscriptLine(s, e, t) for s, e, t in transcript_rows],
        preview_frames=[FramePreview(ts, p) for ts, p in frame_rows],
        ocr_context=[OcrLine(ts, t) for ts, t in ocr_rows],
    )
