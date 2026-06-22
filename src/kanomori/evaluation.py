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
