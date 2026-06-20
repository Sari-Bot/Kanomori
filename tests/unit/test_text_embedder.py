"""Tests for kanomori.embed.text_embedder — the BGE-M3 text embedder.

The real model (BGE-M3, ~560M params, pulls torch + a multi-GB download) is heavy, so the
real backing is exercised only under the `requires_models` marker (the end-to-end proof and a
models-present environment). Importing the module must NOT require torch — the model is loaded
lazily — so the contract constants (EMBED_DIM) are always importable and assertable.
"""

from __future__ import annotations

import importlib.util

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
