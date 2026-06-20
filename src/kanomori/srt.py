"""SRT parsing — Kanomori's own parser, mirroring KITS's lenient parser.

Kanomori consumes the SRT files KITS produces but cannot ``import kits`` (AGPL + GPU stack),
so this is an independent reimplementation of the parse half of ``KITS/src/kits/subtitle.py``.
The leniency must match KITS so transcripts ingest faithfully: blocks split on blank lines,
the first ``HH:MM:SS,mmm --> HH:MM:SS,mmm`` line in a block locates the timecode (a leading
index is tolerated but not required), remaining lines join as the text, and blocks without a
timecode or with empty text are skipped.

A ``Sentence`` here is the same shape KITS emits — Kanomori's transcript retrieval unit.
"""

from __future__ import annotations

import re
from typing import TypedDict


class Sentence(TypedDict):
    """One subtitle cue: a time span and its text. Kanomori's transcript retrieval unit."""

    start: float
    end: float
    text: str


# Matches an SRT timeline line, e.g. "00:00:00,000 --> 00:00:15,000".
_SRT_TIME_LINE = re.compile(
    r"(\d{2}:\d{2}:\d{2},\d{3})\s*-->\s*(\d{2}:\d{2}:\d{2},\d{3})"
)


def srt_time_to_seconds(srt_time: str) -> float:
    """Parse an SRT timestamp ``HH:MM:SS,mmm`` into seconds."""
    hms, _, millis = srt_time.strip().partition(",")
    hours, minutes, secs = (int(part) for part in hms.split(":"))
    return hours * 3600 + minutes * 60 + secs + int(millis or 0) / 1000


def parse_srt(content: str) -> list[Sentence]:
    """Parse SRT text into a list of Sentences. Lenient, mirroring KITS.

    Normalizes CRLF, splits into blocks on blank lines, and for each block finds the first
    timeline line; lines after it join (with newlines) as the text. Blocks without a timeline
    or with empty text are skipped.
    """
    sentences: list[Sentence] = []
    blocks = re.split(r"\n\s*\n", content.replace("\r\n", "\n").strip())
    for block in blocks:
        lines = [ln for ln in block.split("\n") if ln.strip()]
        time_idx = next(
            (i for i, ln in enumerate(lines) if _SRT_TIME_LINE.search(ln)), None
        )
        if time_idx is None:
            continue
        match = _SRT_TIME_LINE.search(lines[time_idx])
        assert match is not None  # guaranteed by the search above
        text = "\n".join(lines[time_idx + 1 :]).strip()
        if not text:
            continue
        sentences.append(
            {
                "start": srt_time_to_seconds(match.group(1)),
                "end": srt_time_to_seconds(match.group(2)),
                "text": text,
            }
        )
    return sentences
