"""Tests for kanomori.media_source — the read-only source-store abstraction.

Workers only READ source video; derived artifacts live under media_root, not here. These tests
are pure: LocalDirSource against tmp_path (and the real samples/ for one happy-path read),
WebDAVSource against an injected fake httpx client (no network), and the settings-driven factory.
"""

from __future__ import annotations

from pathlib import Path

import pytest

from kanomori.config import get_settings
from kanomori.media_source import (
    LocalDirSource,
    MediaSourceError,
    WebDAVSource,
    get_media_source,
    iter_manifest,
)

REPO_ROOT = Path(__file__).resolve().parents[2]
SAMPLES_DIR = REPO_ROOT / "samples"


# --- LocalDirSource ---------------------------------------------------------------------------


def test_local_fetch_copies_file_and_returns_dest(tmp_path: Path) -> None:
    root = tmp_path / "store"
    (root / "clip_2020").mkdir(parents=True)
    src = root / "clip_2020" / "video.mp4"
    src.write_bytes(b"\x00\x01fake-mp4")

    source = LocalDirSource(root)
    dest = tmp_path / "work" / "out.mp4"
    returned = source.fetch("clip_2020/video.mp4", dest)

    assert returned == dest
    assert dest.read_bytes() == b"\x00\x01fake-mp4"


def test_local_read_text_reads_manifest(tmp_path: Path) -> None:
    root = tmp_path / "store"
    root.mkdir()
    (root / "manifest.jsonl").write_text('{"path":"a/video.mp4"}\n', encoding="utf-8")

    source = LocalDirSource(root)
    assert source.read_text("manifest.jsonl") == '{"path":"a/video.mp4"}\n'


def test_local_read_real_samples_manifest_parses_five_records() -> None:
    source = LocalDirSource(SAMPLES_DIR)
    text = source.read_text("manifest.jsonl")
    lines = [ln for ln in text.splitlines() if ln.strip()]
    assert len(lines) == 5


def test_local_fetch_rejects_path_traversal(tmp_path: Path) -> None:
    root = tmp_path / "store"
    root.mkdir()
    source = LocalDirSource(root)
    with pytest.raises((MediaSourceError, FileNotFoundError)):
        source.fetch("../etc/passwd", tmp_path / "out")


def test_local_read_text_rejects_path_traversal(tmp_path: Path) -> None:
    root = tmp_path / "store"
    root.mkdir()
    source = LocalDirSource(root)
    with pytest.raises((MediaSourceError, FileNotFoundError)):
        source.read_text("../../secrets.txt")


def test_local_fetch_missing_file_raises(tmp_path: Path) -> None:
    root = tmp_path / "store"
    root.mkdir()
    source = LocalDirSource(root)
    with pytest.raises((MediaSourceError, FileNotFoundError)):
        source.fetch("nope/video.mp4", tmp_path / "out.mp4")


# --- WebDAVSource (injected fake client, no network) ------------------------------------------


class _FakeResponse:
    def __init__(self, *, status_code: int = 200, content: bytes = b"", text: str = "") -> None:
        self.status_code = status_code
        self.content = content
        self.text = text

    def iter_bytes(self):
        yield self.content


class _FakeStream:
    """Context manager returned by client.stream(...)."""

    def __init__(self, response: _FakeResponse) -> None:
        self._response = response

    def __enter__(self) -> _FakeResponse:
        return self._response

    def __exit__(self, *exc) -> bool:
        return False


class _FakeClient:
    def __init__(self, response: _FakeResponse) -> None:
        self._response = response
        self.get_urls: list[str] = []
        self.stream_urls: list[str] = []

    def get(self, url: str):
        self.get_urls.append(url)
        return self._response

    def stream(self, method: str, url: str):
        self.stream_urls.append(url)
        return _FakeStream(self._response)


def test_webdav_fetch_composes_url_and_streams(tmp_path: Path) -> None:
    client = _FakeClient(_FakeResponse(content=b"video-bytes"))
    source = WebDAVSource("https://dav.example.com/store", client=client)

    dest = tmp_path / "out.mp4"
    returned = source.fetch("clip_2020/video.mp4", dest)

    assert returned == dest
    assert dest.read_bytes() == b"video-bytes"
    assert client.stream_urls == ["https://dav.example.com/store/clip_2020/video.mp4"]


def test_webdav_read_text_composes_url(tmp_path: Path) -> None:
    client = _FakeClient(_FakeResponse(text='{"path":"a/video.mp4"}\n'))
    source = WebDAVSource("https://dav.example.com/store", client=client)

    assert source.read_text("manifest.jsonl") == '{"path":"a/video.mp4"}\n'
    assert client.get_urls == ["https://dav.example.com/store/manifest.jsonl"]


def test_webdav_non_2xx_raises(tmp_path: Path) -> None:
    client = _FakeClient(_FakeResponse(status_code=404))
    source = WebDAVSource("https://dav.example.com/store", client=client)
    with pytest.raises(MediaSourceError):
        source.read_text("missing.jsonl")
    with pytest.raises(MediaSourceError):
        source.fetch("missing.mp4", tmp_path / "out.mp4")


# --- factory ----------------------------------------------------------------------------------


def test_get_media_source_defaults_to_local(monkeypatch) -> None:
    monkeypatch.delenv("KANOMORI_MEDIA_SOURCE", raising=False)
    get_settings.cache_clear()
    try:
        source = get_media_source()
        assert isinstance(source, LocalDirSource)
    finally:
        get_settings.cache_clear()


def test_get_media_source_webdav_when_configured(monkeypatch) -> None:
    monkeypatch.setenv("KANOMORI_MEDIA_SOURCE", "webdav")
    monkeypatch.setenv("KANOMORI_MEDIA_SOURCE_URL", "https://dav.example.com/store")
    get_settings.cache_clear()
    try:
        source = get_media_source()
        assert isinstance(source, WebDAVSource)
    finally:
        get_settings.cache_clear()


# --- iter_manifest ----------------------------------------------------------------------------


def test_iter_manifest_real_samples_returns_five_dicts() -> None:
    source = LocalDirSource(SAMPLES_DIR)
    records = iter_manifest(source)
    assert len(records) == 5
    assert all("path" in rec for rec in records)


def test_iter_manifest_skips_blank_lines(tmp_path: Path) -> None:
    root = tmp_path / "store"
    root.mkdir()
    (root / "manifest.jsonl").write_text(
        '{"path":"a/video.mp4"}\n\n   \n{"path":"b/video.mp4"}\n', encoding="utf-8"
    )
    source = LocalDirSource(root)
    records = iter_manifest(source)
    assert [r["path"] for r in records] == ["a/video.mp4", "b/video.mp4"]


def test_iter_manifest_malformed_raises_with_line_number(tmp_path: Path) -> None:
    root = tmp_path / "store"
    root.mkdir()
    (root / "manifest.jsonl").write_text(
        '{"path":"a/video.mp4"}\nnot-json\n', encoding="utf-8"
    )
    source = LocalDirSource(root)
    with pytest.raises(MediaSourceError, match="line 2"):
        iter_manifest(source)
