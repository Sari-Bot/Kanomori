from __future__ import annotations

import json

import pytest

from kanomori.models import SearchHit


def test_load_eval_suite_parses_fixture(tmp_path) -> None:
    from kanomori.evaluation import load_eval_suite

    fixture = tmp_path / "suite.json"
    fixture.write_text(
        json.dumps(
            {
                "sample": "samples/example.mp4",
                "top_k": 3,
                "transcript_segments": [
                    {"start_sec": 1.0, "end_sec": 2.0, "text": "hello"},
                ],
                "transcript_queries": [
                    {
                        "name": "hello query",
                        "query": "hello",
                        "expected_ts_sec": 1.0,
                        "tolerance_sec": 0.5,
                    }
                ],
                "screenshot_queries": [
                    {"name": "frame zero", "frame_index": 0, "tolerance_sec": 1.0}
                ],
            }
        ),
        encoding="utf-8",
    )

    suite = load_eval_suite(fixture)

    assert suite.sample == "samples/example.mp4"
    assert suite.top_k == 3
    assert suite.transcript_segments[0].text == "hello"
    assert suite.transcript_queries[0].expected_ts_sec == 1.0
    assert suite.screenshot_queries[0].frame_index == 0
    assert suite.audio_snippet_queries == []


def test_load_eval_suite_parses_audio_snippet_queries(tmp_path) -> None:
    from kanomori.evaluation import load_eval_suite

    fixture = tmp_path / "suite.json"
    fixture.write_text(
        json.dumps(
            {
                "sample": "samples/example.mp4",
                "transcript_segments": [],
                "transcript_queries": [],
                "screenshot_queries": [],
                "audio_snippet_queries": [
                    {
                        "name": "standalone clip",
                        "clip": "clips/hello.wav",
                        "expected_ts_sec": 10.0,
                        "tolerance_sec": 3.0,
                    },
                    {
                        "name": "source slice",
                        "source_clip": "samples/example.mp4",
                        "start_sec": 5.0,
                        "end_sec": 12.0,
                        "expected_ts_sec": 5.0,
                        "tolerance_sec": 2.0,
                    },
                ],
            }
        ),
        encoding="utf-8",
    )

    suite = load_eval_suite(fixture)

    assert suite.audio_snippet_queries[0].clip == "clips/hello.wav"
    assert suite.audio_snippet_queries[1].source_clip == "samples/example.mp4"


def test_evaluate_hit_accepts_audio_search_hits() -> None:
    from kanomori.evaluation import evaluate_hit
    from kanomori.models import AudioSearchHit

    result = evaluate_hit(
        "audio",
        [AudioSearchHit(video_id=1, ts_sec=12.0, coverage=2, quality=0.2)],
        expected_ts_sec=11.5,
        tolerance_sec=1.0,
        top_k=1,
    )

    assert result.passed is True


def test_evaluate_hit_marks_pass_when_any_topk_hit_is_in_tolerance() -> None:
    from kanomori.evaluation import evaluate_hit

    hits = [
        SearchHit(video_id=1, ts_sec=2.0, score=0.7),
        SearchHit(video_id=1, ts_sec=10.0, score=0.5),
    ]

    result = evaluate_hit("case", hits, expected_ts_sec=9.5, tolerance_sec=1.0, top_k=2)

    assert result.passed is True
    assert result.hit_rank == 2
    assert result.timestamp_error_sec == 0.5


def test_evaluate_hit_reports_failure_summary() -> None:
    from kanomori.evaluation import evaluate_hit

    hits = [SearchHit(video_id=1, ts_sec=2.0, score=0.7)]

    result = evaluate_hit("case", hits, expected_ts_sec=9.5, tolerance_sec=1.0, top_k=1)

    assert result.passed is False
    assert result.hit_rank is None
    assert "case" in result.summary()
    assert "expected=9.500s" in result.summary()


def _hit(name, *, passed, rank=None, error=None):
    from kanomori.evaluation import EvalHitResult

    return EvalHitResult(
        name=name, expected_ts_sec=0.0, tolerance_sec=1.0, top_k=5,
        passed=passed, hit_rank=rank, hit_ts_sec=None, timestamp_error_sec=error,
    )


def test_aggregate_metrics_top1_and_top5_accuracy() -> None:
    from kanomori.evaluation import aggregate_metrics

    results = [
        _hit("a", passed=True, rank=1, error=0.1),   # top-1 and top-5
        _hit("b", passed=True, rank=3, error=0.2),   # top-5 only
        _hit("c", passed=False),                     # miss
        _hit("d", passed=True, rank=5, error=0.5),   # top-5 only
    ]
    m = aggregate_metrics(results)
    assert m.count == 4
    assert m.top1_accuracy == pytest.approx(1 / 4)   # only "a" is rank 1
    assert m.top5_accuracy == pytest.approx(3 / 4)   # a, b, d


def test_aggregate_metrics_mrr_uses_reciprocal_rank() -> None:
    from kanomori.evaluation import aggregate_metrics

    results = [
        _hit("a", passed=True, rank=1, error=0.0),   # 1/1
        _hit("b", passed=True, rank=2, error=0.0),   # 1/2
        _hit("c", passed=False),                     # 0
    ]
    m = aggregate_metrics(results)
    assert m.mrr == pytest.approx((1.0 + 0.5 + 0.0) / 3)


def test_aggregate_metrics_mean_timestamp_error_over_hits_only() -> None:
    from kanomori.evaluation import aggregate_metrics

    results = [
        _hit("a", passed=True, rank=1, error=0.2),
        _hit("b", passed=True, rank=2, error=0.4),
        _hit("c", passed=False),                     # excluded from error mean
    ]
    m = aggregate_metrics(results)
    assert m.mean_timestamp_error_sec == pytest.approx(0.3)


def test_aggregate_metrics_empty_is_safe() -> None:
    from kanomori.evaluation import aggregate_metrics

    m = aggregate_metrics([])
    assert m.count == 0
    assert m.top1_accuracy == 0.0
    assert m.top5_accuracy == 0.0
    assert m.mrr == 0.0
    assert m.mean_timestamp_error_sec is None
