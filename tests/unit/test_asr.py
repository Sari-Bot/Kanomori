from __future__ import annotations

import subprocess
import sys
import types
from pathlib import Path

import pytest

from kanomori.embed import asr


class FakePipe:
    def __init__(self):
        self.calls = []

    def __call__(self, audio_path, *, return_timestamps, generate_kwargs):
        self.calls.append(
            {
                "audio_path": audio_path,
                "return_timestamps": return_timestamps,
                "generate_kwargs": generate_kwargs,
            }
        )
        return {
            "chunks": [
                {"timestamp": (0.25, 1.5), "text": " こんにちは "},
                {"timestamp": (1.5, 2.0), "text": ""},
            ]
        }


def test_kotoba_whisper_transcribe_uses_kits_compatible_decode_params(monkeypatch) -> None:
    fake_pipe = FakePipe()
    pipeline_calls = []

    def fake_pipeline(task, **kwargs):
        pipeline_calls.append((task, kwargs))
        return fake_pipe

    monkeypatch.setitem(sys.modules, "torch", types.SimpleNamespace(float32="float32"))
    monkeypatch.setitem(sys.modules, "transformers", types.SimpleNamespace(pipeline=fake_pipeline))

    wrapper = asr.KotobaWhisperASR(model_name="kotoba-test", device="cpu")

    segments = wrapper.transcribe(Path("clip.wav"))

    assert pipeline_calls == [
        (
            "automatic-speech-recognition",
            {
                "model": "kotoba-test",
                "dtype": "float32",
                "device": "cpu",
                "chunk_length_s": 15,
            },
        )
    ]
    assert fake_pipe.calls == [
        {
            "audio_path": "clip.wav",
            "return_timestamps": True,
            "generate_kwargs": {
                "language": "ja",
                "task": "transcribe",
                "num_beams": 3,
                "no_repeat_ngram_size": 3,
            },
        }
    ]
    assert segments == [{"start": 0.25, "end": 1.5, "text": "こんにちは"}]


def test_kotoba_whisper_warmup_reuses_loaded_pipeline(monkeypatch) -> None:
    fake_pipe = FakePipe()
    pipeline_calls = []

    def fake_pipeline(task, **kwargs):
        pipeline_calls.append((task, kwargs))
        return fake_pipe

    monkeypatch.setitem(sys.modules, "torch", types.SimpleNamespace(float32="float32"))
    monkeypatch.setitem(sys.modules, "transformers", types.SimpleNamespace(pipeline=fake_pipeline))

    wrapper = asr.KotobaWhisperASR(model_name="kotoba-test", device="cpu")

    wrapper.warmup()
    segments = wrapper.transcribe(Path("clip.wav"))

    assert len(pipeline_calls) == 1
    assert segments == [{"start": 0.25, "end": 1.5, "text": "こんにちは"}]


def test_kotoba_whisper_requires_explicit_model(monkeypatch) -> None:
    monkeypatch.setattr(asr, "get_settings", lambda: types.SimpleNamespace(audio_asr_model=None))

    with pytest.raises(ValueError, match="KANOMORI_AUDIO_ASR_MODEL"):
        asr.KotobaWhisperASR()


def test_normalize_clip_to_wav_raises_decode_error_on_ffmpeg_failure(
    monkeypatch, tmp_path
) -> None:
    calls = []

    def fake_run(argv, **kwargs):
        calls.append((argv, kwargs))
        return subprocess.CompletedProcess(argv, 1, stderr="bad media")

    monkeypatch.setattr(asr.subprocess, "run", fake_run)

    with pytest.raises(asr.AudioDecodeError, match="bad media"):
        asr.normalize_clip_to_wav(tmp_path / "bad.mp3", tmp_path / "out.wav")

    assert calls[0][0][0] == "ffmpeg"
    assert calls[0][1]["capture_output"] is True


def test_probe_duration_sec_parses_ffprobe_output(monkeypatch, tmp_path) -> None:
    def fake_run(argv, **kwargs):
        return subprocess.CompletedProcess(argv, 0, stdout="12.345\n")

    monkeypatch.setattr(asr.subprocess, "run", fake_run)

    assert asr.probe_duration_sec(tmp_path / "clip.wav") == pytest.approx(12.345)
