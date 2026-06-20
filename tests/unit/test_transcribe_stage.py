"""Tests for the transcribe stage's audio-path resolution.

Regression guard for a resume bug: when locate_media is skipped on a resumed run, ctx.audio_path
is None, and transcribe must fall back to the deterministic content-hash-keyed audio artifact
(the extracted 16kHz wav) — NOT the raw media (.mp4), which KITS's audio reader can't decode.
Mirrors the same deterministic-artifact-path pattern parse_transcript uses for the SRT.
"""

from __future__ import annotations

from pathlib import Path

from kanomori.ingest.pipeline import IngestContext
from kanomori.ingest.stages import transcribe as transcribe_stage


def test_transcribe_uses_deterministic_audio_when_audio_path_unset(monkeypatch, tmp_path) -> None:
    captured = {}

    def fake_kits(audio, out_srt, **kwargs):
        captured["audio"] = str(audio)
        Path(out_srt).parent.mkdir(parents=True, exist_ok=True)
        Path(out_srt).write_text("1\n00:00:00,000 --> 00:00:01,000\nx\n", encoding="utf-8")
        return Path(out_srt)

    monkeypatch.setattr(transcribe_stage, "kits_transcribe", fake_kits)
    monkeypatch.setenv("KANOMORI_MEDIA_ROOT", str(tmp_path))
    from kanomori.config import get_settings

    get_settings.cache_clear()

    ctx = IngestContext(media_path="/some/clip.mp4")
    ctx.content_hash = "abc123"
    ctx.audio_path = None  # locate_media was skipped on a resumed run

    transcribe_stage.run(None, ctx)

    assert captured["audio"].endswith("abc123/audio.wav")
    assert not captured["audio"].endswith(".mp4")


def test_transcribe_prefers_explicit_audio_path_when_set(monkeypatch, tmp_path) -> None:
    captured = {}

    def fake_kits(audio, out_srt, **kwargs):
        captured["audio"] = str(audio)
        Path(out_srt).parent.mkdir(parents=True, exist_ok=True)
        Path(out_srt).write_text("1\n00:00:00,000 --> 00:00:01,000\nx\n", encoding="utf-8")
        return Path(out_srt)

    monkeypatch.setattr(transcribe_stage, "kits_transcribe", fake_kits)
    monkeypatch.setenv("KANOMORI_MEDIA_ROOT", str(tmp_path))
    from kanomori.config import get_settings

    get_settings.cache_clear()

    ctx = IngestContext(media_path="/some/clip.mp4")
    ctx.content_hash = "abc123"
    ctx.audio_path = "/explicit/extracted.wav"  # locate_media ran this session

    transcribe_stage.run(None, ctx)
    assert captured["audio"] == "/explicit/extracted.wav"
