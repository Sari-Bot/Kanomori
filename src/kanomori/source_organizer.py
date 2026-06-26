"""Organize loose WebDAV source videos into Kanomori's manifest-backed layout."""

from __future__ import annotations

import argparse
import json
import os
import re
from dataclasses import dataclass, field
from datetime import UTC, datetime
from pathlib import PurePosixPath
from typing import Protocol

from kanomori.source_metadata import (
    DEFAULT_DEEPSEEK_MODEL,
    DeepSeekMetadataClient,
    FilenameMetadata,
    MetadataInferer,
    parse_llm_metadata,
)
from kanomori.source_webdav import DavEntry, WebDAVClient

__all__ = [
    "DavEntry",
    "FilenameMetadata",
    "OrganizerSummary",
    "PlannedItem",
    "SkippedItem",
    "parse_llm_metadata",
    "run_organizer",
    "target_directory",
]

VIDEO_EXTENSIONS = {".mp4", ".mkv", ".mov", ".m4v", ".webm", ".flv"}
DEFAULT_MANIFEST = "manifest.jsonl"


@dataclass(frozen=True)
class PlannedItem:
    source_path: str
    target_dir: str
    target_path: str
    record: dict


@dataclass(frozen=True)
class SkippedItem:
    name: str
    reason: str


@dataclass
class OrganizerSummary:
    planned: list[PlannedItem] = field(default_factory=list)
    skipped: list[SkippedItem] = field(default_factory=list)
    manifest_records: list[dict] = field(default_factory=list)
    applied: bool = False


class SourceStore(Protocol):
    def list_root(self) -> list[DavEntry]: ...

    def read_text(self, path: str) -> str: ...

    def exists(self, path: str) -> bool: ...

    def mkdir(self, path: str) -> None: ...

    def move(self, source_path: str, target_path: str) -> None: ...

    def put_text(self, path: str, text: str) -> None: ...


def run_organizer(
    dav: SourceStore,
    llm: MetadataInferer,
    *,
    apply: bool = False,
    manifest_path: str = DEFAULT_MANIFEST,
) -> OrganizerSummary:
    existing_records, original_manifest = _read_manifest(dav, manifest_path)
    existing_paths = {record.get("path") for record in existing_records if record.get("path")}
    summary = OrganizerSummary(manifest_records=list(existing_records), applied=apply)
    for entry in dav.list_root():
        _plan_entry(entry, dav, llm, existing_paths, summary)
    if apply and summary.planned:
        _apply_plan(dav, manifest_path, original_manifest, summary)
    return summary


def target_directory(metadata: FilenameMetadata) -> str:
    title = _safe_component(metadata.title)
    if metadata.streamed_at:
        return f"{title}_{metadata.streamed_at}"
    return title


def main(argv: list[str] | None = None) -> None:
    args = _build_parser().parse_args(argv)
    if not args.webdav_url:
        raise SystemExit("missing --webdav-url or KANOMORI_MEDIA_SOURCE_URL")
    if not args.deepseek_api_key:
        raise SystemExit("missing --deepseek-api-key or DEEPSEEK_API_KEY")
    dav = WebDAVClient(args.webdav_url, username=args.webdav_user, password=args.webdav_password)
    llm = DeepSeekMetadataClient(args.deepseek_api_key, model=args.model)
    summary = run_organizer(dav, llm, apply=args.apply, manifest_path=args.manifest)
    _print_summary(summary)


def _plan_entry(
    entry: DavEntry,
    dav: SourceStore,
    llm: MetadataInferer,
    existing_paths: set[str],
    summary: OrganizerSummary,
) -> None:
    if entry.is_collection:
        summary.skipped.append(SkippedItem(entry.name, "already organized directory"))
        return
    ext = PurePosixPath(entry.name).suffix.lower()
    if ext not in VIDEO_EXTENSIONS:
        summary.skipped.append(SkippedItem(entry.name, "not a supported video file"))
        return
    metadata = llm.infer(entry.name)
    target_dir = target_directory(metadata)
    target_path = f"{target_dir}/video{ext}"
    if target_path in existing_paths:
        summary.skipped.append(SkippedItem(entry.name, f"manifest already has: {target_path}"))
        return
    if dav.exists(target_path):
        summary.skipped.append(SkippedItem(entry.name, f"target exists: {target_path}"))
        return
    record = metadata.to_record(target_path)
    summary.planned.append(PlannedItem(entry.name, f"{target_dir}/", target_path, record))
    summary.manifest_records.append(record)
    existing_paths.add(target_path)


def _apply_plan(
    dav: SourceStore,
    manifest_path: str,
    original_manifest: str,
    summary: OrganizerSummary,
) -> None:
    for item in summary.planned:
        dav.mkdir(item.target_dir)
        dav.move(item.source_path, item.target_path)
    timestamp = datetime.now(UTC).strftime("%Y%m%dT%H%M%SZ")
    if original_manifest:
        dav.put_text(f"{manifest_path}.bak.{timestamp}", original_manifest)
    dav.put_text(f"{manifest_path}.tmp", _manifest_text(summary.manifest_records))
    dav.move(f"{manifest_path}.tmp", manifest_path)


def _read_manifest(dav: SourceStore, manifest_path: str) -> tuple[list[dict], str]:
    text = dav.read_text(manifest_path)
    records = []
    for lineno, line in enumerate(text.splitlines(), start=1):
        if not line.strip():
            continue
        try:
            record = json.loads(line)
        except json.JSONDecodeError as exc:
            raise ValueError(f"malformed manifest at line {lineno}: {exc}") from exc
        if isinstance(record, dict):
            records.append(record)
    return records, text


def _manifest_text(records: list[dict]) -> str:
    lines = [json.dumps(record, ensure_ascii=False, separators=(",", ":")) for record in records]
    return "\n".join(lines) + ("\n" if lines else "")


def _safe_component(value: str) -> str:
    cleaned = re.sub(r'[/\\:*?"<>|\x00-\x1f]', " ", value)
    cleaned = re.sub(r"\s+", " ", cleaned).strip(" .")
    return cleaned or "untitled"


def _build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(prog="kanomori-organize-source", description=__doc__)
    parser.add_argument("--webdav-url", default=os.getenv("KANOMORI_MEDIA_SOURCE_URL"))
    parser.add_argument("--webdav-user", default=os.getenv("KANOMORI_MEDIA_SOURCE_USER"))
    parser.add_argument("--webdav-password", default=os.getenv("KANOMORI_MEDIA_SOURCE_PASSWORD"))
    parser.add_argument("--deepseek-api-key", default=os.getenv("DEEPSEEK_API_KEY"))
    parser.add_argument("--model", default=DEFAULT_DEEPSEEK_MODEL)
    parser.add_argument("--manifest", default=DEFAULT_MANIFEST)
    mode = parser.add_mutually_exclusive_group()
    mode.add_argument("--apply", action="store_true", help="perform WebDAV writes")
    mode.add_argument("--dry-run", action="store_false", dest="apply", help="plan only")
    parser.set_defaults(apply=False)
    return parser


def _print_summary(summary: OrganizerSummary) -> None:
    mode = "APPLY" if summary.applied else "DRY-RUN"
    print(f"{mode}: planned={len(summary.planned)} skipped={len(summary.skipped)}")
    for item in summary.planned:
        print(f"plan: {item.source_path} -> {item.target_path}")
    for item in summary.skipped:
        print(f"skip: {item.name}: {item.reason}")
