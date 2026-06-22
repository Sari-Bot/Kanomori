from kanomori.models import Candidate, Modality
from kanomori.retrieval.merge import merge_candidates


def test_merge_buckets_nearby_modalities_and_uses_scene_weights() -> None:
    candidates = [
        Candidate(video_id=1, ts_sec=31.0, modality=Modality.VISUAL, rank=0, segment_id=10),
        Candidate(video_id=1, ts_sec=34.0, modality=Modality.TRANSCRIPT, rank=2, segment_id=20),
    ]

    hits = merge_candidates(
        candidates,
        scene_lookup=lambda video_id, ts_sec: "chatting",
        bucket_sec=10.0,
        k=5,
    )

    assert len(hits) == 1
    assert hits[0].video_id == 1
    assert hits[0].ts_sec == 34.0
    assert hits[0].scene_type == "chatting"
    assert hits[0].why["transcript"] > hits[0].why["visual"]


def test_merge_keeps_different_time_buckets_separate() -> None:
    candidates = [
        Candidate(video_id=1, ts_sec=3.0, modality=Modality.TRANSCRIPT, rank=0, segment_id=1),
        Candidate(video_id=1, ts_sec=22.0, modality=Modality.TRANSCRIPT, rank=1, segment_id=2),
    ]

    hits = merge_candidates(
        candidates,
        scene_lookup=lambda video_id, ts_sec: None,
        bucket_sec=10.0,
        k=5,
    )

    assert [hit.ts_sec for hit in hits] == [3.0, 22.0]
