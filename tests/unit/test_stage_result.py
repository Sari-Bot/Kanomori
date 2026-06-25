"""Unit tests for the cross-the-wire stage-result contract and vector codec.

All pure: no DB, no models, no filesystem. Exercises the base64 float32 codec round-trip,
each StageResult ``model_validate(model_dump())`` round-trip (embeddings included), the
FramesResult recipe + artifact manifest, and the natural-key invariant (no database ids
leak into the wire contract).
"""

from __future__ import annotations

import inspect

import numpy as np
import pytest
from pydantic import BaseModel

from kanomori.ingest import stage_result as sr


def _rng(seed: int) -> np.random.Generator:
    return np.random.default_rng(seed)


# --- codec ------------------------------------------------------------------------------


@pytest.mark.parametrize("dim", [1024, 768])
def test_vector_codec_round_trip(dim: int) -> None:
    vec = _rng(dim).standard_normal(dim).astype(np.float64)

    encoded = sr.encode_vector(vec)
    decoded = sr.decode_vector(encoded)

    assert isinstance(encoded, str)
    assert encoded.isascii()
    assert isinstance(decoded, list)
    assert len(decoded) == dim
    assert all(isinstance(x, float) for x in decoded)
    # float32 precision: exact-ish, not bit-exact against the float64 source.
    np.testing.assert_allclose(decoded, vec, rtol=0, atol=1e-6)


def test_vector_codec_accepts_plain_list() -> None:
    vec = [0.0, 1.5, -2.25, 3.125]

    decoded = sr.decode_vector(sr.encode_vector(vec))

    np.testing.assert_allclose(decoded, vec, rtol=0, atol=1e-6)


def test_vector_codec_is_deterministic() -> None:
    vec = _rng(7).standard_normal(1024)

    assert sr.encode_vector(vec) == sr.encode_vector(vec)


def test_vector_codec_round_trips_through_float32() -> None:
    # A second encode of the decoded value must be byte-stable (already at float32 precision).
    vec = _rng(11).standard_normal(768)

    once = sr.encode_vector(vec)
    twice = sr.encode_vector(sr.decode_vector(once))

    assert once == twice


# --- per-stage model round-trips --------------------------------------------------------


def test_register_result_round_trip() -> None:
    result = sr.RegisterResult(
        content_hash="a" * 64,
        source_url="https://example.test/watch?v=x",
        source_platform="youtube",
        title="歌枠アーカイブ",
        stream_type="singing",
    )

    restored = sr.RegisterResult.model_validate(result.model_dump())

    assert restored == result
    assert restored.stage == "register"


def test_parse_transcript_result_round_trip_preserves_embeddings() -> None:
    vecs = [_rng(i).standard_normal(1024) for i in range(3)]
    segments = [
        sr.TranscriptSegmentRow.build(
            seq=i,
            start_sec=float(i) * 10.0,
            end_sec=float(i) * 10.0 + 9.5,
            text=f"こんにちは {i}",
            text_norm=f"こんにちは {i}",
            embedding=v,
        )
        for i, v in enumerate(vecs)
    ]
    result = sr.ParseTranscriptResult(segments=segments)

    restored = sr.ParseTranscriptResult.model_validate(result.model_dump())

    assert restored.stage == "parse_transcript"
    assert [s.seq for s in restored.segments] == [0, 1, 2]
    for got, original in zip(restored.to_vectors(), vecs, strict=True):
        np.testing.assert_allclose(got, original, rtol=0, atol=1e-6)


def test_frames_result_carries_recipe_and_artifact_manifest() -> None:
    timestamps = [0.0, 8.0, 12.5]
    rows = [sr.FrameRow(ts_sec=ts, artifact=_frame_name(ts)) for ts in timestamps]
    result = sr.FramesResult(
        frames=rows,
        scene_timestamps=[8.0, 12.5],
    )

    restored = sr.FramesResult.model_validate(result.model_dump())

    assert restored.scene_timestamps == [8.0, 12.5]
    assert [f.ts_sec for f in restored.frames] == timestamps
    # The manifest names exactly the JPEG artifacts, keyed by ts_sec.
    manifest = restored.artifacts()
    assert [a.kind for a in manifest] == ["frame", "frame", "frame"]
    assert [a.name for a in manifest] == [_frame_name(ts) for ts in timestamps]


def _frame_name(ts_sec: float) -> str:
    total_ms = max(0, round(ts_sec * 1000))
    seconds, millis = divmod(total_ms, 1000)
    return f"frame_{seconds:06d}_{millis:03d}.jpg"


def test_ocr_result_round_trip() -> None:
    rows = [
        sr.OcrRow(ts_sec=8.0, text="スパチャ", confidence=0.97, bbox={"x": 1, "y": 2}),
        sr.OcrRow(ts_sec=12.5, text="待機中", confidence=None, bbox=[[0, 0], [1, 1]]),
    ]
    result = sr.OcrResult(rows=rows)

    restored = sr.OcrResult.model_validate(result.model_dump())

    assert restored == result
    assert restored.stage == "ocr"
    assert restored.rows[1].bbox == [[0, 0], [1, 1]]


def test_classify_result_round_trip() -> None:
    rows = [
        sr.SceneSegmentRow(start_sec=0.0, end_sec=30.0, scene_type="chatting", confidence=0.8),
        sr.SceneSegmentRow(start_sec=30.0, end_sec=90.0, scene_type="singing", confidence=0.91),
    ]
    result = sr.ClassifyResult(segments=rows, stream_type="singing")

    restored = sr.ClassifyResult.model_validate(result.model_dump())

    assert restored == result
    assert restored.stream_type == "singing"
    assert restored.stage == "classify"


def test_image_embed_result_round_trip_preserves_embeddings() -> None:
    vecs = [_rng(100 + i).standard_normal(768) for i in range(2)]
    rows = [
        sr.ImageEmbedRow.build(ts_sec=float(i) * 8.0, phash=123456789 + i, embedding=v)
        for i, v in enumerate(vecs)
    ]
    result = sr.ImageEmbedResult(rows=rows)

    restored = sr.ImageEmbedResult.model_validate(result.model_dump())

    assert restored.stage == "image_embed"
    assert [r.phash for r in restored.rows] == [123456789, 123456790]
    for got, original in zip(restored.to_vectors(), vecs, strict=True):
        np.testing.assert_allclose(got, original, rtol=0, atol=1e-6)


def test_transcribe_result_declares_srt_artifact() -> None:
    result = sr.TranscribeResult(artifact="transcript.srt")

    restored = sr.TranscribeResult.model_validate(result.model_dump())

    assert restored.stage == "transcribe"
    manifest = restored.artifacts()
    assert [(a.name, a.kind) for a in manifest] == [("transcript.srt", "srt")]


# --- natural-key invariant --------------------------------------------------------------


def _all_models() -> list[type[BaseModel]]:
    return [
        obj
        for _name, obj in inspect.getmembers(sr, inspect.isclass)
        if issubclass(obj, BaseModel) and obj.__module__ == sr.__name__
    ]


def test_no_database_ids_leak_into_the_wire_contract() -> None:
    models = _all_models()
    assert models  # guard: we actually found the models

    offenders: list[str] = []
    for model in models:
        for field_name in model.model_fields:
            if field_name == "video_id" or field_name.endswith("_id"):
                offenders.append(f"{model.__name__}.{field_name}")

    assert not offenders, f"database ids leaked into wire contract: {offenders}"
