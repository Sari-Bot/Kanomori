"""Tests for kanomori.text — Japanese normalization and the tokenization seam.

normalize() is pure stdlib (NFKC + whitespace) and always runs. tokenize_for_fts() wraps
fugashi (an `ingest`-group dep); its tests skip cleanly when fugashi isn't installed (the
lightweight default `uv sync`). Importing kanomori.text must NOT require fugashi — it is
lazily imported inside the tokenizer — so these module-level imports always succeed.
"""

from __future__ import annotations

import importlib.util

import pytest

from kanomori.text import normalize, tokenize_for_fts

_HAS_FUGASHI = importlib.util.find_spec("fugashi") is not None
needs_fugashi = pytest.mark.skipif(
    not _HAS_FUGASHI, reason="fugashi only in the 'ingest' dep group"
)


def test_normalize_nfkc_folds_fullwidth_alnum() -> None:
    # Full-width Ａ１ -> half-width A1 under NFKC.
    assert normalize("Ａ１") == "A1"


def test_normalize_collapses_whitespace() -> None:
    assert normalize("  hello   world \n") == "hello world"


def test_normalize_halfwidth_katakana_to_fullwidth() -> None:
    # NFKC maps half-width katakana to full-width.
    assert normalize("ｶﾀｶﾅ") == "カタカナ"


def test_normalize_empty() -> None:
    assert normalize("") == ""


@needs_fugashi
def test_tokenize_segments_japanese_into_space_separated_tokens() -> None:
    # Japanese has no spaces; fugashi must split into several surface tokens.
    tokens = tokenize_for_fts("今日はいい天気です").split()
    assert len(tokens) >= 3
    assert "今日" in tokens
    assert "天気" in tokens


@needs_fugashi
def test_tokenize_normalizes_first() -> None:
    # Full-width input should be folded before tokenizing.
    assert "ABC" in tokenize_for_fts("ＡＢＣ").replace(" ", "")


@needs_fugashi
def test_tokenize_empty_returns_empty() -> None:
    assert tokenize_for_fts("") == ""
