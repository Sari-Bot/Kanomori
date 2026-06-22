from __future__ import annotations

import json

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
