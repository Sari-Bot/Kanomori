"""Tests for kanomori.embed.text_embedder — the BGE-M3 text embedder.

The real model (BGE-M3, ~560M params, pulls torch + a multi-GB download) is heavy, so the
real backing is exercised only under the `requires_models` marker (the end-to-end proof and a
models-present environment). Importing the module must NOT require torch — the model is loaded
lazily — so the contract constants (EMBED_DIM) are always importable and assertable.
"""

from __future__ import annotations

import importlib.util
import sys
import types

import numpy as np
import pytest

from kanomori.embed.text_embedder import EMBED_DIM, BGEEmbedder

_HAS_MODELS = importlib.util.find_spec("FlagEmbedding") is not None
requires_models = pytest.mark.skipif(
    not _HAS_MODELS, reason="BGE-M3 stack only in the 'embed' dep group"
)


def test_embed_dim_is_bge_m3_dense_dimension() -> None:
    # Pinned to match transcript_segments.embedding vector(1024) in 0001_init.sql.
    assert EMBED_DIM == 1024


def test_importing_module_does_not_require_torch() -> None:
    # The module imports cleanly without the embed group; model load is deferred.
    import kanomori.embed.text_embedder as m

    assert hasattr(m, "BGEEmbedder")


def test_load_passes_configured_device_to_flag_model(monkeypatch) -> None:
    calls: dict[str, object] = {}

    class FakeModel:
        def encode(self, texts, **kwargs):
            return {"dense_vecs": np.zeros((len(texts), EMBED_DIM), dtype=np.float32)}

    def fake_bge(model_name, *, use_fp16, devices=None):
        calls["model_name"] = model_name
        calls["use_fp16"] = use_fp16
        calls["devices"] = devices
        return FakeModel()

    monkeypatch.setitem(
        sys.modules,
        "FlagEmbedding",
        types.SimpleNamespace(BGEM3FlagModel=fake_bge),
    )

    embedder = BGEEmbedder(model_name="test-bge", device="cpu")

    embedder.embed_query("device smoke test")

    assert calls == {
        "model_name": "test-bge",
        "use_fp16": False,
        "devices": "cpu",
    }


@pytest.mark.parametrize("device", ["cpu", "gpu"])
def test_pipeline_make_embedder_uses_parse_transcript_stage_device(monkeypatch, device) -> None:
    from kanomori.config import get_settings
    from kanomori.embed import text_embedder
    from kanomori.ingest import pipeline

    created: list[tuple[str | None, str | None]] = []

    class FakeEmbedder:
        def __init__(self, model_name: str | None = None, device: str | None = None):
            created.append((model_name, device))

    monkeypatch.setattr(text_embedder, "BGEEmbedder", FakeEmbedder)
    monkeypatch.setenv("KANOMORI_STAGE_PARSE_TRANSCRIPT_DEVICE", device)
    get_settings.cache_clear()

    try:
        pipeline.make_embedder()
    finally:
        get_settings.cache_clear()

    assert created == [(None, device)]


@requires_models
def test_embed_query_returns_unit_dim_vector() -> None:
    emb = BGEEmbedder()
    vec = emb.embed_query("こんにちは")
    assert isinstance(vec, np.ndarray)
    assert vec.shape == (EMBED_DIM,)


@requires_models
def test_embed_texts_returns_one_vector_per_input() -> None:
    emb = BGEEmbedder()
    vecs = emb.embed_texts(["おはよう", "こんばんは", "またね"])
    assert len(vecs) == 3
    assert all(v.shape == (EMBED_DIM,) for v in vecs)


@requires_models
def test_embed_texts_empty_list_returns_empty() -> None:
    emb = BGEEmbedder()
    assert emb.embed_texts([]) == []
