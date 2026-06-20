"""Tests for kanomori.srt — our own SRT parser, mirroring KITS's lenient parser
(KITS/src/kits/subtitle.py) since we cannot import kits. The round-trip and leniency
behaviors must agree with KITS so transcripts ingest faithfully.
"""

from __future__ import annotations

from pathlib import Path

import pytest

from kanomori.srt import Sentence, parse_srt, srt_time_to_seconds

FIXTURE = Path(__file__).resolve().parents[1] / "fixtures" / "sample.srt"


def test_parses_all_blocks_from_real_kits_fixture() -> None:
    sentences = parse_srt(FIXTURE.read_text(encoding="utf-8"))
    assert len(sentences) == 4


def test_first_block_has_correct_times_and_text() -> None:
    sentences = parse_srt(FIXTURE.read_text(encoding="utf-8"))
    first = sentences[0]
    assert first["start"] == 0.0
    assert first["end"] == 15.0
    assert first["text"] == "ごめん、ごちそう、ごめん、ごちそう、ごめん、ごめん、こんばんはあ。"


def test_fractional_timecode_parsed_to_milliseconds() -> None:
    # Block 3: 00:01:49,000 --> 00:01:54,580
    sentences = parse_srt(FIXTURE.read_text(encoding="utf-8"))
    third = sentences[2]
    assert third["start"] == pytest.approx(109.0)
    assert third["end"] == pytest.approx(114.58)


def test_srt_time_to_seconds_handles_hours_minutes_millis() -> None:
    assert srt_time_to_seconds("01:02:03,456") == pytest.approx(3723.456)


def test_crlf_line_endings_tolerated() -> None:
    content = "1\r\n00:00:01,000 --> 00:00:02,000\r\nテスト\r\n"
    sentences = parse_srt(content)
    assert len(sentences) == 1
    assert sentences[0]["text"] == "テスト"


def test_missing_index_tolerated() -> None:
    # KITS's parser finds the first timeline line regardless of a leading index.
    content = "00:00:01,000 --> 00:00:02,000\nこんにちは\n"
    sentences = parse_srt(content)
    assert len(sentences) == 1
    assert sentences[0]["text"] == "こんにちは"


def test_multiline_text_joined() -> None:
    content = "1\n00:00:01,000 --> 00:00:02,000\nline one\nline two\n"
    sentences = parse_srt(content)
    assert sentences[0]["text"] == "line one\nline two"


def test_blocks_without_timeline_skipped() -> None:
    content = "garbage block with no timecode\n\n1\n00:00:01,000 --> 00:00:02,000\nok\n"
    sentences = parse_srt(content)
    assert len(sentences) == 1
    assert sentences[0]["text"] == "ok"


def test_blocks_with_empty_text_skipped() -> None:
    content = "1\n00:00:01,000 --> 00:00:02,000\n\n\n2\n00:00:03,000 --> 00:00:04,000\nreal\n"
    sentences = parse_srt(content)
    assert [s["text"] for s in sentences] == ["real"]


def test_empty_input_returns_empty_list() -> None:
    assert parse_srt("") == []
    assert parse_srt("   \n  \n") == []


def test_sentence_is_typeddict_shape() -> None:
    s: Sentence = {"start": 1.0, "end": 2.0, "text": "x"}
    assert s["start"] == 1.0
