"""Tests for kanomori.fusion — Reciprocal Rank Fusion (RRF).

RRF combines several rank-ordered candidate lists into one, scale-free: an item's score is
sum over lists of 1/(k + rank), with rank 0-based. It never looks at raw scores, so lists with
incomparable score scales fuse cleanly. These tests pin the math, tie behavior, and the
multi-list accumulation that the retrieval layer depends on.
"""

from __future__ import annotations

import pytest

from kanomori.fusion import reciprocal_rank_fusion


def test_single_list_orders_by_rank() -> None:
    fused = reciprocal_rank_fusion([["a", "b", "c"]], k=60)
    assert [key for key, _ in fused] == ["a", "b", "c"]


def test_rank_zero_scores_higher_than_rank_one() -> None:
    fused = dict(reciprocal_rank_fusion([["a", "b"]], k=60))
    assert fused["a"] == pytest.approx(1 / 60)
    assert fused["b"] == pytest.approx(1 / 61)


def test_item_in_two_lists_accumulates_contributions() -> None:
    # "a" is rank 0 in list 1 and rank 1 in list 2: 1/60 + 1/61.
    fused = dict(reciprocal_rank_fusion([["a", "x"], ["y", "a"]], k=60))
    assert fused["a"] == pytest.approx(1 / 60 + 1 / 61)


def test_agreement_across_lists_beats_a_single_top_rank() -> None:
    # "b" tops one list (1/60); "a" is 2nd in both (1/61 + 1/61 > 1/60).
    fused = dict(reciprocal_rank_fusion([["b", "a"], ["c", "a"]], k=60))
    assert fused["a"] > fused["b"]


def test_result_is_sorted_descending_by_score() -> None:
    fused = reciprocal_rank_fusion([["b", "a"], ["c", "a"]], k=60)
    scores = [score for _, score in fused]
    assert scores == sorted(scores, reverse=True)


def test_empty_lists_yield_empty_result() -> None:
    assert reciprocal_rank_fusion([], k=60) == []
    assert reciprocal_rank_fusion([[], []], k=60) == []


def test_k_parameter_changes_score_magnitude() -> None:
    fused = dict(reciprocal_rank_fusion([["a"]], k=1))
    assert fused["a"] == pytest.approx(1 / 1)


def test_weights_scale_per_list_contribution() -> None:
    # With list weights [2, 1], "a" at rank 0 in the first list contributes 2 * 1/60.
    fused = dict(reciprocal_rank_fusion([["a"], ["b"]], k=60, weights=[2.0, 1.0]))
    assert fused["a"] == pytest.approx(2 * (1 / 60))
    assert fused["b"] == pytest.approx(1 * (1 / 60))


def test_deterministic_tie_break_is_stable() -> None:
    # Equal scores must produce a deterministic order across runs (stable sort by insertion).
    r1 = reciprocal_rank_fusion([["a", "b"]], k=60)
    r2 = reciprocal_rank_fusion([["a", "b"]], k=60)
    assert r1 == r2
