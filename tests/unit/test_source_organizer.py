from __future__ import annotations

import json

import pytest

from kanomori import source_organizer
from kanomori.source_organizer import (
    DavEntry,
    FilenameMetadata,
    SkippedItem,
    parse_llm_metadata,
    run_organizer,
)
from kanomori.source_webdav import WebDAVClient, WebDAVError


class FakeLLM:
    def __init__(self, responses: dict[str, FilenameMetadata]) -> None:
        self.responses = responses
        self.calls: list[str] = []

    def infer(self, filename: str) -> FilenameMetadata:
        self.calls.append(filename)
        return self.responses[filename]


class FakeDav:
    def __init__(
        self,
        entries: list[DavEntry],
        *,
        manifest: str = "",
        existing: set[str] | None = None,
    ) -> None:
        self.entries = entries
        self.manifest = manifest
        self.existing = existing or set()
        self.operations: list[tuple] = []

    def list_root(self) -> list[DavEntry]:
        return list(self.entries)

    def read_text(self, path: str) -> str:
        assert path == "manifest.jsonl"
        return self.manifest

    def exists(self, path: str) -> bool:
        return path in self.existing

    def mkdir(self, path: str) -> None:
        self.operations.append(("mkdir", path))
        self.existing.add(path)

    def move(self, source_path: str, target_path: str) -> None:
        self.operations.append(("move", source_path, target_path))
        self.existing.add(target_path)

    def put_text(self, path: str, text: str) -> None:
        self.operations.append(("put", path, text))
        if path == "manifest.jsonl":
            self.manifest = text


def test_parse_llm_metadata_omits_missing_source_url() -> None:
    metadata = parse_llm_metadata(
        json.dumps(
            {
                "title": "鹿乃 歌回",
                "streamed_at": "2024-02-18",
                "source_platform": "bilibili",
                "source_url": None,
                "separate": True,
                "confidence": 0.9,
            }
        ),
        fallback_title="fallback",
    )

    assert metadata.to_record("鹿乃 歌回_2024-02-18/video.mp4") == {
        "path": "鹿乃 歌回_2024-02-18/video.mp4",
        "title": "鹿乃 歌回",
        "streamed_at": "2024-02-18",
        "source_platform": "bilibili",
        "separate": True,
    }


def test_parse_llm_metadata_falls_back_on_invalid_json() -> None:
    metadata = parse_llm_metadata("not json", fallback_title="raw filename")

    assert metadata.title == "raw filename"
    assert metadata.separate is False
    assert metadata.to_record("raw filename/video.mp4") == {
        "path": "raw filename/video.mp4",
        "title": "raw filename",
        "separate": False,
    }


def test_dry_run_plans_webdav_move_without_mutating() -> None:
    dav = FakeDav([DavEntry(name="raw 鹿乃 2024-02-18.mp4", is_collection=False)])
    llm = FakeLLM(
        {
            "raw 鹿乃 2024-02-18.mp4": FilenameMetadata(
                title="鹿乃 歌回",
                streamed_at="2024-02-18",
                source_platform="bilibili",
                separate=True,
            )
        }
    )

    summary = run_organizer(dav, llm, apply=False)

    assert len(summary.planned) == 1
    assert summary.planned[0].source_path == "raw 鹿乃 2024-02-18.mp4"
    assert summary.planned[0].target_path == "鹿乃 歌回_2024-02-18/video.mp4"
    assert summary.manifest_records == [
        {
            "path": "鹿乃 歌回_2024-02-18/video.mp4",
            "title": "鹿乃 歌回",
            "streamed_at": "2024-02-18",
            "source_platform": "bilibili",
            "separate": True,
        }
    ]
    assert dav.operations == []
    assert llm.calls == ["raw 鹿乃 2024-02-18.mp4"]


def test_run_organizer_skips_existing_target() -> None:
    dav = FakeDav(
        [DavEntry(name="raw.mp4", is_collection=False)],
        existing={"鹿乃 歌回_2024-02-18/video.mp4"},
    )
    llm = FakeLLM(
        {
            "raw.mp4": FilenameMetadata(
                title="鹿乃 歌回",
                streamed_at="2024-02-18",
                separate=False,
            )
        }
    )

    summary = run_organizer(dav, llm, apply=True)

    assert summary.planned == []
    assert summary.skipped == [
        SkippedItem(name="raw.mp4", reason="target exists: 鹿乃 歌回_2024-02-18/video.mp4")
    ]
    assert not any(op[0] == "move" for op in dav.operations)


def test_apply_moves_video_and_rewrites_manifest_safely() -> None:
    existing_record = {"path": "old/video.mp4", "title": "old", "separate": False}
    dav = FakeDav(
        [
            DavEntry(name="new archive.mkv", is_collection=False),
            DavEntry(name="organized_2024-01-01", is_collection=True),
            DavEntry(name="notes.txt", is_collection=False),
        ],
        manifest=json.dumps(existing_record, ensure_ascii=False) + "\n",
    )
    llm = FakeLLM(
        {
            "new archive.mkv": FilenameMetadata(
                title="New Stream",
                streamed_at="2026-06-26",
                source_platform="youtube",
                separate=False,
            )
        }
    )

    summary = run_organizer(dav, llm, apply=True)

    assert summary.skipped == [
        SkippedItem(name="organized_2024-01-01", reason="already organized directory"),
        SkippedItem(name="notes.txt", reason="not a supported video file"),
    ]
    assert ("mkdir", "New Stream_2026-06-26/") in dav.operations
    assert ("move", "new archive.mkv", "New Stream_2026-06-26/video.mkv") in dav.operations

    manifest_tmp_writes = [
        op for op in dav.operations if op[0] == "put" and op[1] == "manifest.jsonl.tmp"
    ]
    assert len(manifest_tmp_writes) == 1
    records = [json.loads(line) for line in manifest_tmp_writes[0][2].splitlines()]
    assert records == [
        existing_record,
        {
            "path": "New Stream_2026-06-26/video.mkv",
            "title": "New Stream",
            "streamed_at": "2026-06-26",
            "source_platform": "youtube",
            "separate": False,
        },
    ]
    assert any(op[0] == "put" and op[1].startswith("manifest.jsonl.bak.") for op in dav.operations)
    assert ("move", "manifest.jsonl.tmp", "manifest.jsonl") in dav.operations


def test_parser_accepts_explicit_dry_run_flag() -> None:
    args = source_organizer._build_parser().parse_args(["--dry-run"])

    assert args.apply is False


def test_webdav_exists_falls_back_when_head_is_not_allowed(monkeypatch: pytest.MonkeyPatch) -> None:
    client = WebDAVClient("https://dav.example.com/store")
    calls: list[tuple[str, str]] = []

    def fake_request(method: str, path: str, **kwargs) -> bytes:
        calls.append((method, path))
        if method == "HEAD":
            raise WebDAVError(405, "HEAD not allowed")
        if method == "PROPFIND":
            return b""
        raise AssertionError(f"unexpected method: {method}")

    monkeypatch.setattr(client, "_request", fake_request)

    assert client.exists("target/video.mp4") is True
    assert calls == [("HEAD", "target/video.mp4"), ("PROPFIND", "target/video.mp4")]
