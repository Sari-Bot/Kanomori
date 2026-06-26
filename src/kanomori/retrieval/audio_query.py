"""Build transcript-search query windows from an uploaded clip transcript."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Literal

from kanomori.embed.asr import AsrSegment
from kanomori.text import normalize

WindowKind = Literal["segment", "pair", "fulltext"]

_FILLERS = {
    "あの",
    "あのー",
    "うん",
    "えー",
    "えっと",
    "そうですね",
    "なんか",
    "はい",
    "まあ",
}
_PUNCT = " \t\r\n、。,.，．!！?？…ー〜~・"


@dataclass(frozen=True)
class QueryWindow:
    text: str
    norm: str
    start: float | None
    end: float | None
    kind: WindowKind
    is_filler: bool = False


@dataclass(frozen=True)
class _CleanSegment:
    index: int
    text: str
    norm: str
    start: float | None
    end: float | None


def build_windows(segments: list[AsrSegment]) -> list[QueryWindow]:
    """Return non-filler single, adjacent-pair, and full-text query windows."""
    clean = [_clean_segment(i, segment) for i, segment in enumerate(segments)]
    clean = [segment for segment in clean if segment is not None]
    if not clean:
        return []

    windows = [_to_window([segment], "segment") for segment in clean]
    windows.extend(
        _to_window([left, right], "pair")
        for left, right in zip(clean, clean[1:], strict=False)
        if right.index == left.index + 1
    )
    windows.append(_to_window(clean, "fulltext"))
    return windows


def _clean_segment(index: int, segment: AsrSegment) -> _CleanSegment | None:
    text = str(segment.get("text") or "").strip()
    norm = normalize(text)
    if not norm or _is_filler(norm):
        return None
    return _CleanSegment(
        index=index,
        text=text,
        norm=norm,
        start=segment.get("start"),
        end=segment.get("end"),
    )


def _is_filler(norm: str) -> bool:
    return norm.strip(_PUNCT) in _FILLERS


def _to_window(segments: list[_CleanSegment], kind: WindowKind) -> QueryWindow:
    text = " ".join(segment.text for segment in segments)
    return QueryWindow(
        text=text,
        norm=normalize(text),
        start=segments[0].start,
        end=segments[-1].end,
        kind=kind,
    )
