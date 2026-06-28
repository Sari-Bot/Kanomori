from __future__ import annotations

import asyncio

import pytest


class _FakeSettings:
    def __init__(self, *, preload_search_models: bool):
        self.preload_search_models = preload_search_models


class _FakeAsr:
    def __init__(self, calls: list[str], *, fail: bool = False):
        self._calls = calls
        self._fail = fail

    def warmup(self) -> None:
        self._calls.append("asr")
        if self._fail:
            raise RuntimeError("asr unavailable")


def _run(coro):
    return asyncio.run(coro)


def test_lifespan_preloads_all_search_models_when_enabled(monkeypatch) -> None:
    from kanomori.api import app as app_module

    calls: list[str] = []

    monkeypatch.setattr(
        app_module, "get_settings", lambda: _FakeSettings(preload_search_models=True)
    )
    monkeypatch.setattr(app_module, "get_embedder", lambda: calls.append("text"))
    monkeypatch.setattr(app_module, "get_image_embedder", lambda: calls.append("image"))
    monkeypatch.setattr(app_module, "get_ocr_reader", lambda: calls.append("ocr"))
    monkeypatch.setattr(app_module, "get_asr", lambda: _FakeAsr(calls))
    monkeypatch.setattr(app_module, "close_pool", lambda: calls.append("close"))

    async def run_lifespan():
        async with app_module.lifespan(object()):
            calls.append("serving")

    _run(run_lifespan())

    assert calls == ["text", "image", "ocr", "asr", "serving", "close"]


def test_lifespan_skips_search_model_preload_when_disabled(monkeypatch) -> None:
    from kanomori.api import app as app_module

    calls: list[str] = []

    monkeypatch.setattr(
        app_module, "get_settings", lambda: _FakeSettings(preload_search_models=False)
    )
    monkeypatch.setattr(app_module, "get_embedder", lambda: calls.append("text"))
    monkeypatch.setattr(app_module, "get_image_embedder", lambda: calls.append("image"))
    monkeypatch.setattr(app_module, "get_ocr_reader", lambda: calls.append("ocr"))
    monkeypatch.setattr(app_module, "get_asr", lambda: _FakeAsr(calls))
    monkeypatch.setattr(app_module, "close_pool", lambda: calls.append("close"))

    async def run_lifespan():
        async with app_module.lifespan(object()):
            calls.append("serving")

    _run(run_lifespan())

    assert calls == ["serving", "close"]


def test_lifespan_propagates_preload_failure_when_enabled(monkeypatch) -> None:
    from kanomori.api import app as app_module

    calls: list[str] = []

    monkeypatch.setattr(
        app_module, "get_settings", lambda: _FakeSettings(preload_search_models=True)
    )
    monkeypatch.setattr(app_module, "get_embedder", lambda: calls.append("text"))
    monkeypatch.setattr(app_module, "get_image_embedder", lambda: calls.append("image"))
    monkeypatch.setattr(app_module, "get_ocr_reader", lambda: calls.append("ocr"))
    monkeypatch.setattr(app_module, "get_asr", lambda: _FakeAsr(calls, fail=True))
    monkeypatch.setattr(app_module, "close_pool", lambda: calls.append("close"))

    async def run_lifespan():
        async with app_module.lifespan(object()):
            pytest.fail("startup should not reach serving state")

    with pytest.raises(RuntimeError, match="asr unavailable"):
        _run(run_lifespan())

    assert calls == ["text", "image", "ocr", "asr"]


def test_audio_upload_transcribes_before_borrowing_db_connection(monkeypatch) -> None:
    from kanomori.api import app as app_module

    events: list[str] = []

    class FakeUpload:
        filename = "clip.wav"

        async def read(self) -> bytes:
            return b"audio bytes"

    class FakeAsr:
        def transcribe(self, _path):
            events.append("transcribe")
            return [{"start": 0.0, "end": 1.0, "text": "hello"}]

    class FakeConnection:
        def __enter__(self):
            events.append("db_enter")
            return object()

        def __exit__(self, exc_type, exc, tb):
            events.append("db_exit")
            return False

    monkeypatch.setattr(
        app_module, "normalize_clip_to_wav", lambda _src, dst: dst.write_bytes(b"wav")
    )
    monkeypatch.setattr(app_module, "probe_duration_sec", lambda _path: 1.0)
    monkeypatch.setattr(
        app_module,
        "get_settings",
        lambda: type("Settings", (), {"audio_clip_max_sec": 35.0})(),
    )
    monkeypatch.setattr(app_module, "get_asr", lambda: FakeAsr())
    monkeypatch.setattr(app_module, "get_embedder", lambda: object())
    monkeypatch.setattr(app_module, "connection", lambda: FakeConnection())
    monkeypatch.setattr(
        app_module.audio,
        "audio_candidates",
        lambda _conn, segments, _embedder, *, k: (" ".join(s["text"] for s in segments), []),
    )

    response = _run(app_module._search_audio_upload(FakeUpload(), 5))

    assert response.transcript == "hello"
    assert events == ["transcribe", "db_enter", "db_exit"]
