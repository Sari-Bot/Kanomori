"""Audio-upload retrieval by transcribing then searching the transcript index."""

from __future__ import annotations

from dataclasses import dataclass, field
from pathlib import Path

from kanomori.models import AudioSearchHit, Candidate, Modality, WindowEvidence
from kanomori.retrieval import transcript
from kanomori.retrieval.audio_query import QueryWindow, build_windows
from kanomori.retrieval.merge import DEFAULT_BUCKET_SEC, RRF_K, _bucket_key, scene_at
from kanomori.text import normalize


@dataclass
class _BucketState:
    video_id: int
    best_ts: float
    best_contrib: float = -1.0
    voters: set[int] = field(default_factory=set)
    quality: float = 0.0
    evidence: list[WindowEvidence] = field(default_factory=list)


def audio_candidates(
    conn,
    audio_wav_path: str | Path,
    asr,
    embedder,
    *,
    k: int = 10,
    per_window_k: int = 50,
) -> tuple[str, list[AudioSearchHit]]:
    segments = asr.transcribe(audio_wav_path)
    full_text = normalize(" ".join(str(segment.get("text") or "") for segment in segments))
    windows = build_windows(segments)
    per_window = [
        (window, transcript.candidates(conn, window.norm, embedder, k=per_window_k))
        for window in windows
    ]
    return full_text, aggregate_coverage(conn, per_window, k=k)


def aggregate_coverage(
    conn,
    per_window_candidates: list[tuple[QueryWindow, list[Candidate]]],
    *,
    bucket_sec: float = DEFAULT_BUCKET_SEC,
    k: int = 10,
) -> list[AudioSearchHit]:
    buckets: dict[tuple[int, int], _BucketState] = {}
    for window_id, (window, candidates) in enumerate(per_window_candidates):
        voted: set[tuple[int, int]] = set()
        for candidate in candidates:
            if candidate.modality != Modality.TRANSCRIPT or candidate.segment_id is None:
                continue
            key = _bucket_key(candidate, bucket_sec)
            state = buckets.setdefault(key, _BucketState(candidate.video_id, candidate.ts_sec))
            contribution = 1.0 / (RRF_K + candidate.rank)
            if key not in voted:
                voted.add(key)
                _add_vote(conn, state, window_id, window, candidate, contribution)
            if contribution > state.best_contrib:
                state.best_contrib = contribution
                state.best_ts = candidate.ts_sec

    hits = [_to_hit(conn, state) for state in buckets.values()]
    hits.sort(key=lambda hit: (hit.coverage, hit.quality), reverse=True)
    return hits[:k]


def _add_vote(
    conn,
    state: _BucketState,
    window_id: int,
    window: QueryWindow,
    candidate: Candidate,
    contribution: float,
) -> None:
    state.voters.add(window_id)
    state.quality += contribution
    state.evidence.append(
        WindowEvidence(
            window_text=window.text,
            window_kind=window.kind,
            matched_segment_id=candidate.segment_id,
            matched_text=_segment_text(conn, candidate.segment_id),
            matched_ts_sec=candidate.ts_sec,
            rrf_score=contribution,
        )
    )


def _segment_text(conn, segment_id: int) -> str:
    row = conn.execute(
        "SELECT text FROM transcript_segments WHERE id = %s",
        (segment_id,),
    ).fetchone()
    return row[0] if row else ""


def _to_hit(conn, state: _BucketState) -> AudioSearchHit:
    return AudioSearchHit(
        video_id=state.video_id,
        ts_sec=state.best_ts,
        coverage=len(state.voters),
        quality=state.quality,
        scene_type=scene_at(conn, state.video_id, state.best_ts),
        evidence=state.evidence,
    )
