"""Small retrieval evaluation helpers for opt-in local-sample smoke tests."""

from __future__ import annotations

import json
from collections.abc import Sequence
from pathlib import Path

from pydantic import BaseModel

from kanomori.models import SearchHit


class TranscriptSegmentCase(BaseModel):
    start_sec: float
    end_sec: float
    text: str


class TranscriptQueryCase(BaseModel):
    name: str
    query: str
    expected_ts_sec: float
    tolerance_sec: float


class ScreenshotQueryCase(BaseModel):
    name: str
    frame_index: int
    tolerance_sec: float


class EvalSuite(BaseModel):
    sample: str
    top_k: int = 5
    transcript_segments: list[TranscriptSegmentCase]
    transcript_queries: list[TranscriptQueryCase]
    screenshot_queries: list[ScreenshotQueryCase]


class EvalHitResult(BaseModel):
    name: str
    expected_ts_sec: float
    tolerance_sec: float
    top_k: int
    passed: bool
    hit_rank: int | None = None
    hit_ts_sec: float | None = None
    timestamp_error_sec: float | None = None

    def summary(self) -> str:
        if self.passed:
            return (
                f"{self.name}: pass rank={self.hit_rank} ts={self.hit_ts_sec:.3f}s "
                f"error={self.timestamp_error_sec:.3f}s"
            )
        return (
            f"{self.name}: miss top_k={self.top_k} expected={self.expected_ts_sec:.3f}s "
            f"tolerance={self.tolerance_sec:.3f}s"
        )


def load_eval_suite(path: Path) -> EvalSuite:
    return EvalSuite.model_validate(json.loads(path.read_text(encoding="utf-8")))


def evaluate_hit(
    name: str,
    hits: Sequence[SearchHit],
    *,
    expected_ts_sec: float,
    tolerance_sec: float,
    top_k: int,
) -> EvalHitResult:
    for rank, hit in enumerate(hits[:top_k], start=1):
        error = abs(hit.ts_sec - expected_ts_sec)
        if error <= tolerance_sec:
            return EvalHitResult(
                name=name,
                expected_ts_sec=expected_ts_sec,
                tolerance_sec=tolerance_sec,
                top_k=top_k,
                passed=True,
                hit_rank=rank,
                hit_ts_sec=hit.ts_sec,
                timestamp_error_sec=error,
            )
    return EvalHitResult(
        name=name,
        expected_ts_sec=expected_ts_sec,
        tolerance_sec=tolerance_sec,
        top_k=top_k,
        passed=False,
    )


class EvalMetrics(BaseModel):
    """Aggregate retrieval metrics over a set of EvalHitResults (white paper §13).

    Top-5 accuracy is the headline metric (users visually verify candidates). MRR rewards
    ranking the correct moment higher; mean timestamp error is over hits only (a miss has no
    meaningful timestamp). All accuracies are fractions in [0, 1]."""

    count: int
    top1_accuracy: float
    top5_accuracy: float
    mrr: float
    mean_timestamp_error_sec: float | None

    def summary(self) -> str:
        err = (
            f"{self.mean_timestamp_error_sec:.3f}s"
            if self.mean_timestamp_error_sec is not None
            else "n/a"
        )
        return (
            f"n={self.count} top1={self.top1_accuracy:.3f} top5={self.top5_accuracy:.3f} "
            f"mrr={self.mrr:.3f} ts_err={err}"
        )


def aggregate_metrics(results: Sequence[EvalHitResult]) -> EvalMetrics:
    """Roll per-query EvalHitResults up into Top-1/Top-5/MRR/mean-timestamp-error.

    A result counts toward top-N when it passed with hit_rank <= N. MRR sums 1/hit_rank over
    all queries (misses contribute 0) divided by the query count. Mean timestamp error averages
    timestamp_error_sec over passing results only (None if there are none)."""
    n = len(results)
    if n == 0:
        return EvalMetrics(
            count=0, top1_accuracy=0.0, top5_accuracy=0.0, mrr=0.0,
            mean_timestamp_error_sec=None,
        )

    top1 = sum(1 for r in results if r.passed and r.hit_rank == 1)
    top5 = sum(1 for r in results if r.passed and r.hit_rank is not None and r.hit_rank <= 5)
    mrr = sum(1.0 / r.hit_rank for r in results if r.passed and r.hit_rank) / n

    errors = [
        r.timestamp_error_sec
        for r in results
        if r.passed and r.timestamp_error_sec is not None
    ]
    mean_err = sum(errors) / len(errors) if errors else None

    return EvalMetrics(
        count=n,
        top1_accuracy=top1 / n,
        top5_accuracy=top5 / n,
        mrr=mrr,
        mean_timestamp_error_sec=mean_err,
    )
