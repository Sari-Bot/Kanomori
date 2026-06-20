"""Tests for kanomori.kits_client — the subprocess boundary to KITS.

KITS transcription requires a GPU, so we never invoke it for real here. transcribe() takes an
injectable runner (defaulting to subprocess.run); these tests inject a fake to verify the argv
contract, cwd, the empty-output guard, and KitsError handling — no GPU, no real KITS.
"""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path

import pytest

from kanomori.kits_client import KitsError, build_kits_argv, transcribe


@dataclass
class FakeCompleted:
    returncode: int
    stdout: str = ""
    stderr: str = ""


def test_build_argv_core_command() -> None:
    argv = build_kits_argv(Path("/audio/in.mp3"), Path("/out/x.srt"))
    assert argv[:3] == ["uv", "run", "kits"]
    assert "subtitle" in argv
    assert "-i" in argv and "/audio/in.mp3" in argv
    assert "-o" in argv and "/out/x.srt" in argv


def test_build_argv_adds_separate_flag_when_requested() -> None:
    argv = build_kits_argv(Path("a.mp3"), Path("a.srt"), separate=True)
    assert "--separate" in argv


def test_build_argv_omits_separate_by_default() -> None:
    argv = build_kits_argv(Path("a.mp3"), Path("a.srt"))
    assert "--separate" not in argv


def test_build_argv_repeats_filter_game_flag() -> None:
    argv = build_kits_argv(Path("a.mp3"), Path("a.srt"), filter_game=["valorant", "valo"])
    # Each game is its own --filter-game occurrence (KITS uses action="append").
    assert argv.count("--filter-game") == 2
    assert "valorant" in argv and "valo" in argv


def test_build_argv_sets_language() -> None:
    argv = build_kits_argv(Path("a.mp3"), Path("a.srt"), language="english")
    assert "--language" in argv
    assert "english" in argv


def test_transcribe_runs_with_cwd_set_to_kits_dir(tmp_path: Path) -> None:
    out = tmp_path / "out.srt"
    captured = {}

    def fake_runner(argv, **kwargs):
        captured["argv"] = argv
        captured["cwd"] = kwargs.get("cwd")
        out.write_text("1\n00:00:00,000 --> 00:00:01,000\nx\n", encoding="utf-8")
        return FakeCompleted(returncode=0)

    transcribe(
        tmp_path / "in.mp3", out, kits_dir=Path("/k"), runner=fake_runner
    )
    assert captured["cwd"] == Path("/k")
    assert captured["argv"][:3] == ["uv", "run", "kits"]


def test_transcribe_returns_output_path_on_success(tmp_path: Path) -> None:
    out = tmp_path / "out.srt"

    def fake_runner(argv, **kwargs):
        out.write_text("1\n00:00:00,000 --> 00:00:01,000\nx\n", encoding="utf-8")
        return FakeCompleted(returncode=0)

    result = transcribe(tmp_path / "in.mp3", out, kits_dir=Path("/k"), runner=fake_runner)
    assert result == out


def test_transcribe_raises_on_nonzero_exit(tmp_path: Path) -> None:
    def fake_runner(argv, **kwargs):
        return FakeCompleted(returncode=1, stderr="no GPU")

    with pytest.raises(KitsError, match="GPU"):
        transcribe(
            tmp_path / "in.mp3", tmp_path / "out.srt", kits_dir=Path("/k"), runner=fake_runner
        )


def test_transcribe_raises_when_output_missing(tmp_path: Path) -> None:
    # Exit 0 but no SRT written — KITS produced nothing.
    def fake_runner(argv, **kwargs):
        return FakeCompleted(returncode=0)

    with pytest.raises(KitsError, match="no output|empty|not.*produced|missing"):
        transcribe(
            tmp_path / "in.mp3", tmp_path / "out.srt", kits_dir=Path("/k"), runner=fake_runner
        )


def test_transcribe_raises_when_output_empty(tmp_path: Path) -> None:
    out = tmp_path / "out.srt"

    def fake_runner(argv, **kwargs):
        out.write_text("   \n", encoding="utf-8")
        return FakeCompleted(returncode=0)

    with pytest.raises(KitsError):
        transcribe(tmp_path / "in.mp3", out, kits_dir=Path("/k"), runner=fake_runner)
