"""Tests for the locate_media stage: audio extraction + karaoke --separate heuristic.

The pure-logic parts (the ffmpeg argv builder and the karaoke heuristic) are unit-tested
directly; the stage's run() uses an injectable runner so we verify it invokes ffmpeg and sets
ctx.audio_path without needing a real media file here (real extraction is exercised in the
end-to-end proof).
"""

from __future__ import annotations

from pathlib import Path

import pytest

from kanomori.ingest.stages.locate_media import (
    build_ffmpeg_extract_argv,
    decide_separate,
)


@pytest.mark.parametrize(
    "title",
    ["【歌枠】sing along", "Karaoke night", "弾き語り cover", "今日のsetlist", "歌ってみた"],
)
def test_decide_separate_true_for_karaoke_titles(title) -> None:
    assert decide_separate(title, explicit=False) is True


@pytest.mark.parametrize("title", ["Minecraft 雑談", "VALORANT ランク", "お知らせ", None])
def test_decide_separate_false_for_non_karaoke_titles(title) -> None:
    assert decide_separate(title, explicit=False) is False


def test_decide_separate_respects_explicit_override() -> None:
    # An explicit request forces separation regardless of title.
    assert decide_separate("Minecraft", explicit=True) is True


def test_ffmpeg_argv_extracts_mono_pcm_wav() -> None:
    argv = build_ffmpeg_extract_argv(Path("/in/clip.mp4"), Path("/out/audio.wav"))
    assert argv[0] == "ffmpeg"
    assert "/in/clip.mp4" in argv
    assert "/out/audio.wav" in argv
    assert "-vn" in argv  # drop video
    assert "-ac" in argv and "1" in argv  # mono
    assert "-ar" in argv and "16000" in argv  # 16 kHz (Whisper's rate)
    assert "-y" in argv  # overwrite


def test_run_invokes_ffmpeg_and_sets_audio_path(tmp_path, monkeypatch) -> None:
    from kanomori.ingest import pipeline
    from kanomori.ingest.stages import locate_media

    captured = {}

    def fake_runner(argv, **kwargs):
        captured["argv"] = argv
        # Simulate ffmpeg producing the output file.
        Path(argv[argv.index("-y") - 1]).write_bytes(b"RIFFfake")

        class R:
            returncode = 0
            stderr = ""

        return R()

    monkeypatch.setattr(locate_media, "_run", fake_runner)
    ctx = pipeline.IngestContext(media_path=str(tmp_path / "clip.mp4"), title="雑談")
    ctx.content_hash = "deadbeef"
    monkeypatch.setenv("KANOMORI_MEDIA_ROOT", str(tmp_path / "media"))
    from kanomori.config import get_settings

    get_settings.cache_clear()

    locate_media.run(None, ctx)
    assert ctx.audio_path is not None
    assert Path(ctx.audio_path).exists()
    assert captured["argv"][0] == "ffmpeg"


def test_run_raises_on_ffmpeg_failure(tmp_path, monkeypatch) -> None:
    from kanomori.ingest import pipeline
    from kanomori.ingest.stages import locate_media

    def failing_runner(argv, **kwargs):
        class R:
            returncode = 1
            stderr = "ffmpeg boom"

        return R()

    monkeypatch.setattr(locate_media, "_run", failing_runner)
    ctx = pipeline.IngestContext(media_path=str(tmp_path / "clip.mp4"))
    ctx.content_hash = "deadbeef2"
    with pytest.raises(RuntimeError, match="ffmpeg"):
        locate_media.run(None, ctx)
