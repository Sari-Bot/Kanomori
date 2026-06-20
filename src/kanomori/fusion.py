"""Rank fusion for hybrid retrieval.

Reciprocal Rank Fusion (RRF) merges several rank-ordered candidate lists into one combined
ranking. It is **scale-free**: each list contributes ``weight / (k + rank)`` per item (rank
0-based), looking only at positions, never raw scores. That is exactly what we want when
fusing dense-vector cosine distances with lexical ``ts_rank`` — their score scales are
incomparable, but their rankings are not.

``k`` (default 60, the value from the original RRF paper) damps the influence of top ranks so
broad agreement across lists can outweigh a single list's #1.
"""

from __future__ import annotations

from collections.abc import Hashable, Sequence


def reciprocal_rank_fusion(
    ranked_lists: Sequence[Sequence[Hashable]],
    *,
    k: int = 60,
    weights: Sequence[float] | None = None,
) -> list[tuple[Hashable, float]]:
    """Fuse rank-ordered lists into one ``(item, score)`` list, sorted by score descending.

    Each list ``i`` contributes ``weights[i] / (k + rank)`` to every item it contains, summed
    across lists. ``weights`` defaults to all-ones. Ties keep first-seen order (stable sort),
    so output is deterministic.
    """
    if weights is None:
        weights = [1.0] * len(ranked_lists)

    scores: dict[Hashable, float] = {}
    for items, weight in zip(ranked_lists, weights, strict=True):
        for rank, item in enumerate(items):
            scores[item] = scores.get(item, 0.0) + weight / (k + rank)

    # dict preserves insertion order; Python's sort is stable, so equal scores keep that order.
    return sorted(scores.items(), key=lambda kv: kv[1], reverse=True)
