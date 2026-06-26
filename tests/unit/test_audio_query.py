from __future__ import annotations

import pytest

from kanomori.retrieval.audio_query import build_windows


def test_build_windows_drops_bare_filler_but_keeps_compound_phrase() -> None:
    segments = [
        {"start": 0.0, "end": 1.0, "text": "うん"},
        {"start": 1.0, "end": 2.0, "text": "なんか面白い"},
        {"start": 2.0, "end": 3.0, "text": "今日は ﾏｲﾝｸﾗﾌﾄ"},
    ]

    windows = build_windows(segments)

    assert all(window.text != "うん" for window in windows)
    assert any(window.text == "なんか面白い" for window in windows)
    assert any("マインクラフト" in window.norm for window in windows)


def test_build_windows_adds_adjacent_pairs_and_fulltext_fallback() -> None:
    segments = [
        {"start": 1.0, "end": 2.0, "text": "最初の話"},
        {"start": 2.0, "end": 4.0, "text": "次の話"},
    ]

    windows = build_windows(segments)

    pair = next(window for window in windows if window.kind == "pair")
    assert pair.text == "最初の話 次の話"
    assert pair.start == pytest.approx(1.0)
    assert pair.end == pytest.approx(4.0)

    fulltext = next(window for window in windows if window.kind == "fulltext")
    assert fulltext.text == "最初の話 次の話"
    assert fulltext.start == pytest.approx(1.0)
    assert fulltext.end == pytest.approx(4.0)


def test_build_windows_returns_no_voting_windows_for_filler_only_clip() -> None:
    windows = build_windows(
        [
            {"start": 0.0, "end": 1.0, "text": "はい"},
            {"start": 1.0, "end": 2.0, "text": "そうですね"},
        ]
    )

    assert windows == []
