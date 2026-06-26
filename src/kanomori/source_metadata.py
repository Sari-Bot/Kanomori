"""Filename metadata inference for the WebDAV source organizer."""

from __future__ import annotations

import json
import re
import urllib.request
from dataclasses import dataclass
from pathlib import PurePosixPath
from typing import Protocol

DEFAULT_DEEPSEEK_MODEL = "deepseek-v4-flash"
DEEPSEEK_BASE_URL = "https://api.deepseek.com"


@dataclass(frozen=True)
class FilenameMetadata:
    title: str
    streamed_at: str | None = None
    source_platform: str | None = None
    source_url: str | None = None
    stream_type: str | None = None
    separate: bool = False
    confidence: float | None = None
    notes: str | None = None

    def to_record(self, path: str) -> dict:
        record = {"path": path, "title": self.title}
        for key in ("streamed_at", "source_platform", "source_url", "stream_type"):
            value = getattr(self, key)
            if value:
                record[key] = value
        record["separate"] = self.separate
        return record


class MetadataInferer(Protocol):
    def infer(self, filename: str) -> FilenameMetadata: ...


class DeepSeekMetadataClient:
    def __init__(
        self,
        api_key: str,
        *,
        model: str = DEFAULT_DEEPSEEK_MODEL,
        base_url: str = DEEPSEEK_BASE_URL,
        timeout: float = 60.0,
    ) -> None:
        self.api_key = api_key
        self.model = model
        self.base_url = base_url.rstrip("/")
        self.timeout = timeout

    def infer(self, filename: str) -> FilenameMetadata:
        content = self._chat(filename)
        return parse_llm_metadata(content, fallback_title=fallback_title(filename))

    def _chat(self, filename: str) -> str:
        body = {
            "model": self.model,
            "messages": [
                {"role": "system", "content": _deepseek_system_prompt()},
                {"role": "user", "content": f"Filename: {filename}"},
            ],
            "response_format": {"type": "json_object"},
            "max_tokens": 800,
            "temperature": 0,
        }
        request = urllib.request.Request(
            f"{self.base_url}/chat/completions",
            data=json.dumps(body, ensure_ascii=False).encode(),
            headers={
                "Authorization": f"Bearer {self.api_key}",
                "Content-Type": "application/json",
            },
            method="POST",
        )
        with urllib.request.urlopen(request, timeout=self.timeout) as response:
            payload = json.loads(response.read().decode())
        return payload["choices"][0]["message"].get("content") or ""


def parse_llm_metadata(content: str, *, fallback_title: str) -> FilenameMetadata:
    try:
        payload = json.loads(content)
    except json.JSONDecodeError:
        return FilenameMetadata(title=fallback_title)
    if not isinstance(payload, dict):
        return FilenameMetadata(title=fallback_title)
    title = _clean_text(payload.get("title")) or fallback_title
    return FilenameMetadata(
        title=title,
        streamed_at=_date_or_none(payload.get("streamed_at")),
        source_platform=_clean_text(payload.get("source_platform")),
        source_url=_clean_text(payload.get("source_url")),
        stream_type=_clean_text(payload.get("stream_type")),
        separate=_as_bool(payload.get("separate")),
        confidence=_as_float(payload.get("confidence")),
        notes=_clean_text(payload.get("notes")),
    )


def fallback_title(filename: str) -> str:
    stem = PurePosixPath(filename).stem
    cleaned = re.sub(r"[_\-.]+", " ", stem)
    return re.sub(r"\s+", " ", cleaned).strip() or "untitled"


def _date_or_none(value: object) -> str | None:
    text = _clean_text(value)
    if text and re.fullmatch(r"\d{4}(-\d{2}(-\d{2})?)?", text):
        return text
    return None


def _clean_text(value: object) -> str | None:
    if not isinstance(value, str):
        return None
    text = re.sub(r"\s+", " ", value).strip()
    return text or None


def _as_bool(value: object) -> bool:
    if isinstance(value, bool):
        return value
    if isinstance(value, str):
        return value.strip().lower() in {"1", "true", "yes", "y"}
    return False


def _as_float(value: object) -> float | None:
    try:
        return float(value) if value is not None else None
    except (TypeError, ValueError):
        return None


def _deepseek_system_prompt() -> str:
    return """
Infer Kanomori ingest metadata from an irregular video filename. Output strict json only.
Use this json shape:
{"title":"human title","streamed_at":"YYYY-MM-DD or YYYY-MM or YYYY","source_platform":"bilibili",
"source_url":null,"stream_type":null,"separate":false,"confidence":0.7,"notes":"short"}
Set unknown optional fields to null. Set separate=true only for likely singing/karaoke streams.
""".strip()
