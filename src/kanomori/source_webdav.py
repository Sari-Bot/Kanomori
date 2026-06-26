"""Small stdlib WebDAV client for source-store organization."""

from __future__ import annotations

import base64
import urllib.error
import urllib.parse
import urllib.request
import xml.etree.ElementTree as ET
from dataclasses import dataclass

DAV_NS = "{DAV:}"


@dataclass(frozen=True)
class DavEntry:
    name: str
    is_collection: bool


class WebDAVError(Exception):
    def __init__(self, status: int, message: str) -> None:
        super().__init__(message)
        self.status = status


class WebDAVClient:
    def __init__(
        self,
        base_url: str,
        *,
        username: str | None = None,
        password: str | None = None,
        timeout: float = 60.0,
    ) -> None:
        self.base_url = base_url.rstrip("/")
        self.username = username
        self.password = password
        self.timeout = timeout
        parsed = urllib.parse.urlparse(self.base_url)
        self._base_path = urllib.parse.unquote(parsed.path)

    def list_root(self) -> list[DavEntry]:
        body = self._request("PROPFIND", "", headers={"Depth": "1"})
        root = ET.fromstring(body)
        return [entry for entry in self._parse_propfind(root) if entry.name]

    def read_text(self, path: str) -> str:
        try:
            return self._request("GET", path).decode()
        except WebDAVError as exc:
            if exc.status == 404:
                return ""
            raise

    def exists(self, path: str) -> bool:
        try:
            self._request("HEAD", path)
            return True
        except WebDAVError as exc:
            if exc.status == 404:
                return False
            if exc.status == 405:
                return self._propfind_exists(path)
            raise

    def mkdir(self, path: str) -> None:
        try:
            self._request("MKCOL", path)
        except WebDAVError as exc:
            if exc.status != 405:
                raise

    def move(self, source_path: str, target_path: str) -> None:
        self._request(
            "MOVE",
            source_path,
            headers={"Destination": self._url(target_path), "Overwrite": "T"},
        )

    def put_text(self, path: str, text: str) -> None:
        self._request(
            "PUT",
            path,
            data=text.encode(),
            headers={"Content-Type": "application/jsonl; charset=utf-8"},
        )

    def _parse_propfind(self, root: ET.Element) -> list[DavEntry]:
        entries = []
        for response in root.findall(f"{DAV_NS}response"):
            href = response.findtext(f"{DAV_NS}href")
            if not href:
                continue
            collection = response.find(f".//{DAV_NS}collection") is not None
            entries.append(DavEntry(name=self._entry_name(href), is_collection=collection))
        return entries

    def _entry_name(self, href: str) -> str:
        path = urllib.parse.unquote(urllib.parse.urlparse(href).path).rstrip("/")
        base = self._base_path.rstrip("/")
        if path == base:
            return ""
        rel = path.removeprefix(base).strip("/")
        return rel if "/" not in rel else ""

    def _request(
        self,
        method: str,
        path: str,
        *,
        data: bytes | None = None,
        headers: dict[str, str] | None = None,
    ) -> bytes:
        request = urllib.request.Request(
            self._url(path),
            data=data,
            headers=self._headers(headers),
            method=method,
        )
        try:
            with urllib.request.urlopen(request, timeout=self.timeout) as response:
                return response.read()
        except urllib.error.HTTPError as exc:
            message = f"{method} {self._url(path)} failed: HTTP {exc.code}"
            raise WebDAVError(exc.code, message) from exc

    def _propfind_exists(self, path: str) -> bool:
        try:
            self._request("PROPFIND", path, headers={"Depth": "0"})
            return True
        except WebDAVError as exc:
            if exc.status == 404:
                return False
            raise

    def _headers(self, extra: dict[str, str] | None = None) -> dict[str, str]:
        headers = dict(extra or {})
        if self.username is not None:
            raw = f"{self.username}:{self.password or ''}".encode()
            headers["Authorization"] = f"Basic {base64.b64encode(raw).decode('ascii')}"
        return headers

    def _url(self, path: str) -> str:
        if not path:
            return self.base_url
        quoted = "/".join(urllib.parse.quote(part) for part in path.strip("/").split("/"))
        return f"{self.base_url}/{quoted}"
