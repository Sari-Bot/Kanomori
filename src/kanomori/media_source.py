"""Read-only source-store abstraction for ingestion workers.

A worker only ever *reads* source video and the batch manifest from a store; every derived
artifact (frames, audio, SRT) is written under ``media_root`` instead, so the source store can
stay a pristine, possibly-remote mirror. ``MediaSource`` is that read boundary: ``LocalDirSource``
is the dev/local mirror (``samples/``), ``WebDAVSource`` the production HTTPS store. The layout is
identical in both (see ``samples/README.md``), so a batch validated locally moves to WebDAV by
copying the tree and flipping ``KANOMORI_MEDIA_SOURCE``. Concrete sources are selected by
``get_media_source()`` from settings, so the worker never hard-codes which store it talks to.
"""

from __future__ import annotations

import json
import shutil
from pathlib import Path
from typing import TYPE_CHECKING, Protocol, runtime_checkable

from kanomori.config import get_settings

if TYPE_CHECKING:
    import httpx


class MediaSourceError(Exception):
    """Raised when a source store cannot serve a path (missing, escaping root, or HTTP error)."""


@runtime_checkable
class MediaSource(Protocol):
    """Read-only view of a source store keyed by manifest ``path`` strings."""

    def fetch(self, path: str, dest: Path) -> Path:
        """Copy/download the source file at ``path`` to ``dest`` and return ``dest``."""
        ...

    def read_text(self, path: str) -> str:
        """Read a text file (e.g. ``manifest.jsonl``) from the store and return its contents."""
        ...


class LocalDirSource:
    """Source store backed by a local directory tree (the ``samples/`` mirror)."""

    def __init__(self, root: Path) -> None:
        self._root = Path(root).resolve()

    def _resolve(self, path: str) -> Path:
        """Resolve ``path`` under root, refusing anything that escapes it (path traversal)."""
        target = (self._root / path).resolve()
        if target != self._root and self._root not in target.parents:
            raise MediaSourceError(f"path escapes source root: {path!r}")
        return target

    def fetch(self, path: str, dest: Path) -> Path:
        src = self._resolve(path)
        if not src.is_file():
            raise MediaSourceError(f"source file not found: {path!r}")
        dest = Path(dest)
        dest.parent.mkdir(parents=True, exist_ok=True)
        shutil.copy(src, dest)
        return dest

    def read_text(self, path: str) -> str:
        src = self._resolve(path)
        if not src.is_file():
            raise MediaSourceError(f"source file not found: {path!r}")
        return src.read_text(encoding="utf-8")


class WebDAVSource:
    """Source store served over HTTPS. Plain GET only — no PROPFIND/XML.

    httpx is imported lazily so this module stays importable when httpx isn't installed (it sits
    in the dev group today); only constructing a WebDAVSource pulls it in.
    """

    def __init__(
        self,
        base_url: str,
        *,
        username: str | None = None,
        password: str | None = None,
        client: httpx.Client | None = None,
    ) -> None:
        self._base_url = base_url.rstrip("/")
        if client is None:
            import httpx

            auth = (username, password) if username is not None else None
            client = httpx.Client(auth=auth)
        self._client = client

    def _url(self, path: str) -> str:
        return f"{self._base_url}/{path.lstrip('/')}"

    def fetch(self, path: str, dest: Path) -> Path:
        url = self._url(path)
        dest = Path(dest)
        dest.parent.mkdir(parents=True, exist_ok=True)
        with self._client.stream("GET", url) as response:
            if not 200 <= response.status_code < 300:
                raise MediaSourceError(f"GET {url} failed: HTTP {response.status_code}")
            with dest.open("wb") as fh:
                for chunk in response.iter_bytes():
                    fh.write(chunk)
        return dest

    def read_text(self, path: str) -> str:
        url = self._url(path)
        response = self._client.get(url)
        if not 200 <= response.status_code < 300:
            raise MediaSourceError(f"GET {url} failed: HTTP {response.status_code}")
        return response.text


def get_media_source() -> MediaSource:
    """Construct the configured source store from settings (``local`` default, or ``webdav``)."""
    settings = get_settings()
    kind = settings.media_source
    if kind == "local":
        return LocalDirSource(settings.media_source_root)
    if kind == "webdav":
        if not settings.media_source_url:
            raise MediaSourceError("KANOMORI_MEDIA_SOURCE=webdav requires MEDIA_SOURCE_URL")
        return WebDAVSource(
            settings.media_source_url,
            username=settings.media_source_user,
            password=settings.media_source_password,
        )
    raise MediaSourceError(f"unknown media source kind: {kind!r}")


def iter_manifest(source: MediaSource, manifest_path: str = "manifest.jsonl") -> list[dict]:
    """Parse a JSONL manifest from ``source``, skipping blank lines.

    Raises ``MediaSourceError`` with the 1-based line number on a malformed line, so a bad batch
    file points the operator straight at the offending record.
    """
    text = source.read_text(manifest_path)
    records: list[dict] = []
    for lineno, line in enumerate(text.splitlines(), start=1):
        if not line.strip():
            continue
        try:
            records.append(json.loads(line))
        except json.JSONDecodeError as exc:
            raise MediaSourceError(f"malformed manifest at line {lineno}: {exc}") from exc
    return records
