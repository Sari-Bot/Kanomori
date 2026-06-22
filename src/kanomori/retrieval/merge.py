"""Scene-aware merge layer for per-modality retrieval candidates."""

from __future__ import annotations

from collections.abc import Callable

from kanomori.models import Candidate, Modality, SearchHit

DEFAULT_BUCKET_SEC = 8.0
RRF_K = 60

DEFAULT_PROFILE = {
    Modality.TRANSCRIPT: 1.0,
    Modality.OCR: 1.0,
    Modality.VISUAL: 1.0,
    Modality.SCENE: 1.0,
    Modality.AUDIO: 0.0,
}

SCENE_WEIGHT_PROFILES = {
    "chatting": {
        Modality.TRANSCRIPT: 1.6,
        Modality.OCR: 1.0,
        Modality.VISUAL: 0.6,
        Modality.SCENE: 0.5,
        Modality.AUDIO: 0.0,
    },
    "gaming": {
        Modality.TRANSCRIPT: 0.8,
        Modality.OCR: 1.1,
        Modality.VISUAL: 1.6,
        Modality.SCENE: 0.8,
        Modality.AUDIO: 0.0,
    },
    "singing": {
        Modality.TRANSCRIPT: 1.2,
        Modality.OCR: 1.2,
        Modality.VISUAL: 0.8,
        Modality.SCENE: 0.8,
        Modality.AUDIO: 0.0,
    },
    "waiting": {
        Modality.TRANSCRIPT: 0.6,
        Modality.OCR: 0.8,
        Modality.VISUAL: 1.0,
        Modality.SCENE: 0.8,
        Modality.AUDIO: 0.0,
    },
    "superchat": {
        Modality.TRANSCRIPT: 1.0,
        Modality.OCR: 1.5,
        Modality.VISUAL: 0.8,
        Modality.SCENE: 0.8,
        Modality.AUDIO: 0.0,
    },
    "announcement": {
        Modality.TRANSCRIPT: 0.8,
        Modality.OCR: 1.5,
        Modality.VISUAL: 1.0,
        Modality.SCENE: 0.8,
        Modality.AUDIO: 0.0,
    },
    "collab": {
        Modality.TRANSCRIPT: 1.0,
        Modality.OCR: 1.0,
        Modality.VISUAL: 1.0,
        Modality.SCENE: 0.8,
        Modality.AUDIO: 0.0,
    },
}


def _bucket_key(candidate: Candidate, bucket_sec: float) -> tuple[int, int]:
    return (candidate.video_id, round(candidate.ts_sec / bucket_sec))


def _profile(scene_type: str | None) -> dict[Modality, float]:
    if scene_type is None:
        return DEFAULT_PROFILE
    return SCENE_WEIGHT_PROFILES.get(scene_type, DEFAULT_PROFILE)


def scene_at(conn, video_id: int, ts_sec: float) -> str | None:
    row = conn.execute(
        """
        SELECT scene_type
        FROM scene_segments
        WHERE video_id = %s
          AND start_sec <= %s
          AND end_sec > %s
        ORDER BY confidence DESC NULLS LAST, start_sec DESC
        LIMIT 1
        """,
        (video_id, ts_sec, ts_sec),
    ).fetchone()
    return row[0] if row else None


def merge_candidates(
    candidates: list[Candidate],
    *,
    scene_lookup: Callable[[int, float], str | None],
    bucket_sec: float = DEFAULT_BUCKET_SEC,
    k: int = 10,
) -> list[SearchHit]:
    buckets: dict[tuple[int, int], list[Candidate]] = {}
    for candidate in candidates:
        buckets.setdefault(_bucket_key(candidate, bucket_sec), []).append(candidate)

    hits: list[SearchHit] = []
    for bucket_candidates in buckets.values():
        anchor = bucket_candidates[0]
        scene_type = scene_lookup(anchor.video_id, anchor.ts_sec)
        weights = _profile(scene_type)
        why: dict[str, float] = {}
        best_ts = anchor.ts_sec
        best_contribution = -1.0

        for candidate in bucket_candidates:
            weight = weights.get(candidate.modality, 1.0)
            contribution = weight / (RRF_K + candidate.rank)
            current = why.get(candidate.modality.value, 0.0)
            why[candidate.modality.value] = max(current, contribution)
            if contribution > best_contribution:
                best_contribution = contribution
                best_ts = candidate.ts_sec

        hits.append(
            SearchHit(
                video_id=anchor.video_id,
                ts_sec=best_ts,
                score=sum(why.values()),
                scene_type=scene_type,
                why=why,
            )
        )

    return sorted(hits, key=lambda hit: hit.score, reverse=True)[:k]


def merge_from_db(
    conn,
    candidates: list[Candidate],
    *,
    bucket_sec: float = DEFAULT_BUCKET_SEC,
    k: int = 10,
) -> list[SearchHit]:
    return merge_candidates(
        candidates,
        scene_lookup=lambda video_id, ts_sec: scene_at(conn, video_id, ts_sec),
        bucket_sec=bucket_sec,
        k=k,
    )
