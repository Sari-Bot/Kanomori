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


def test_transcribe_passes_absolute_paths_to_runner(tmp_path, monkeypatch) -> None:
    # Root cause of the e2e FileNotFoundError: KITS runs with cwd=KITS_DIR, so a relative
    # -i/-o path resolves against KITS's cwd (not kanomori's) and KITS can't find/create it.
    # transcribe must hand KITS absolute paths regardless of how it was called.
    monkeypatch.chdir(tmp_path)  # so genuinely relative paths resolve under tmp_path
    captured = {}

    def fake_runner(argv, **kwargs):
        captured["argv"] = argv
        out_abs = Path(argv[argv.index("-o") + 1])
        out_abs.parent.mkdir(parents=True, exist_ok=True)
        out_abs.write_text("1\n00:00:00,000 --> 00:00:01,000\nx\n", encoding="utf-8")
        return FakeCompleted(returncode=0)

    # Genuinely relative paths (no leading slash) — the exact shape media_root="./media" yields.
    rel_in = Path("media/in.wav")
    rel_in.parent.mkdir(parents=True, exist_ok=True)
    rel_in.write_bytes(b"x")
    rel_out = Path("media/hash/out.srt")
    assert not rel_in.is_absolute() and not rel_out.is_absolute()  # precondition

    transcribe(rel_in, rel_out, kits_dir=Path("/k"), runner=fake_runner)

    i_path = Path(captured["argv"][captured["argv"].index("-i") + 1])
    o_path = Path(captured["argv"][captured["argv"].index("-o") + 1])
    assert i_path.is_absolute(), f"-i must be absolute, got {i_path}"
    assert o_path.is_absolute(), f"-o must be absolute, got {o_path}"


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


def test_transcribe_writes_log_artifact_on_success(tmp_path: Path) -> None:
    # The full KITS stdout/stderr must be captured to a log file alongside the SRT, so a
    # crash is diagnosable after the fact (the plan requires a log artifact).
    out = tmp_path / "out.srt"

    def fake_runner(argv, **kwargs):
        out.write_text("1\n00:00:00,000 --> 00:00:01,000\nx\n", encoding="utf-8")
        return FakeCompleted(returncode=0, stdout="loaded model\n23 cues", stderr="a warning")

    transcribe(tmp_path / "in.mp3", out, kits_dir=Path("/k"), runner=fake_runner)
    log = tmp_path / "out.kits.log"
    assert log.is_file()
    content = log.read_text(encoding="utf-8")
    assert "23 cues" in content
    assert "a warning" in content


def test_transcribe_relays_live_output_via_callback(tmp_path: Path) -> None:
    out = tmp_path / "out.srt"
    seen: list[tuple[str, str]] = []

    def fake_runner(argv, **kwargs):
        callback = kwargs["log_output"]
        callback("stdout", "loaded model")
        callback("stderr", "warming up gpu")
        out.write_text("1\n00:00:00,000 --> 00:00:01,000\nx\n", encoding="utf-8")
        return FakeCompleted(returncode=0, stdout="loaded model\n", stderr="warming up gpu\n")

    transcribe(
        tmp_path / "in.mp3",
        out,
        kits_dir=Path("/k"),
        runner=fake_runner,
        log_output=lambda stream, line: seen.append((stream, line)),
    )

    assert seen == [("stdout", "loaded model"), ("stderr", "warming up gpu")]


def test_transcribe_writes_full_untruncated_stderr_to_log_on_failure(tmp_path: Path) -> None:
    # The exception message is bounded, but the LOG must hold the complete stderr — this is
    # exactly the evidence that was lost when the real end-to-end transcribe failed.
    long_trace = "Traceback...\n" + "X" * 5000 + "\nFinalError: boom"

    def fake_runner(argv, **kwargs):
        return FakeCompleted(returncode=1, stderr=long_trace)

    with pytest.raises(KitsError):
        transcribe(
            tmp_path / "in.mp3", tmp_path / "out.srt", kits_dir=Path("/k"), runner=fake_runner
        )

    log = tmp_path / "out.kits.log"
    assert log.is_file()
    content = log.read_text(encoding="utf-8")
    assert "FinalError: boom" in content  # the tail that truncation would have dropped
    assert len(content) >= 5000


def test_transcribe_error_message_references_log_path(tmp_path: Path) -> None:
    def fake_runner(argv, **kwargs):
        return FakeCompleted(returncode=1, stderr="some failure")

    with pytest.raises(KitsError, match="out.kits.log"):
        transcribe(
            tmp_path / "in.mp3", tmp_path / "out.srt", kits_dir=Path("/k"), runner=fake_runner
        )
