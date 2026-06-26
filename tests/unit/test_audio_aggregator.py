from __future__ import annotations

from kanomori.models import Candidate, Modality
from kanomori.retrieval.audio import aggregate_coverage
from kanomori.retrieval.audio_query import QueryWindow


class _Rows:
    def __init__(self, row):
        self.row = row

    def fetchone(self):
        return self.row


class FakeConn:
    def execute(self, sql, params):
        if "FROM transcript_segments" in sql:
            return _Rows((f"matched segment {params[0]}",))
        if "FROM scene_segments" in sql:
            return _Rows(("chatting",))
        raise AssertionError(sql)


def _window(text: str, kind: str = "segment") -> QueryWindow:
    return QueryWindow(text=text, norm=text, start=0.0, end=1.0, kind=kind)


def _candidate(video_id: int, ts: float, rank: int, segment_id: int) -> Candidate:
    return Candidate(
        video_id=video_id,
        ts_sec=ts,
        modality=Modality.TRANSCRIPT,
        rank=rank,
        segment_id=segment_id,
    )


def test_aggregate_coverage_prioritizes_distinct_window_coverage_over_rrf_quality() -> None:
    shared = [
        (_window("a"), [_candidate(1, 32.0, 20, 10), _candidate(2, 80.0, 0, 20)]),
        (_window("b"), [_candidate(1, 33.0, 21, 11)]),
        (_window("c"), [_candidate(1, 34.0, 22, 12)]),
    ]

    hits = aggregate_coverage(FakeConn(), shared, bucket_sec=10.0, k=2)

    assert hits[0].video_id == 1
    assert hits[0].coverage == 3
    assert hits[1].video_id == 2
    assert hits[1].coverage == 1


def test_aggregate_coverage_uses_quality_as_tiebreak_and_populates_evidence() -> None:
    windows = [
        (_window("first"), [_candidate(1, 10.0, 5, 101)]),
        (_window("second"), [_candidate(2, 50.0, 0, 201)]),
    ]

    hits = aggregate_coverage(FakeConn(), windows, bucket_sec=10.0, k=2)

    assert [hit.video_id for hit in hits] == [2, 1]
    assert hits[0].quality > hits[1].quality
    assert hits[0].scene_type == "chatting"
    assert hits[0].evidence[0].window_text == "second"
    assert hits[0].evidence[0].matched_segment_id == 201
    assert hits[0].evidence[0].matched_text == "matched segment 201"


def test_aggregate_coverage_counts_one_window_once_per_bucket() -> None:
    windows = [
        (
            _window("repeat"),
            [
                _candidate(1, 31.0, 0, 1),
                _candidate(1, 34.0, 1, 2),
            ],
        )
    ]

    hits = aggregate_coverage(FakeConn(), windows, bucket_sec=10.0, k=5)

    assert len(hits) == 1
    assert hits[0].coverage == 1
    assert len(hits[0].evidence) == 1
