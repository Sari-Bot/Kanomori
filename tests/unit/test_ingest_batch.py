"""Pure (DB-free) tests for ``POST /ingest/batch``.

These exercise the real FastAPI route with both the DB layer and the MediaSource layer
monkeypatched out, so they cover the manifest->jobs enqueue logic — dedup of already-queued
paths, the per-line INSERT payload (the verbatim record, keyed by ``path``), the response shape,
and the 400 on a malformed manifest — without a live PostgreSQL. The transactional SQL itself is
covered by tests/integration/test_ingest_batch.py.
"""

from __future__ import annotations

import contextlib
import json

from fastapi.testclient import TestClient

from kanomori.media_source import MediaSourceError

RECORDS = [
    {"path": "a/video.mp4", "title": "A", "source_platform": "youtube"},
    {"path": "b/video.mp4", "title": "B", "separate": True},
    {"path": "c/video.mp4", "title": "C"},
]


class _FakeConn:
    """Records INSERTs and reports a configured set of paths as already-queued duplicates."""

    def __init__(self, dup_paths: set[str]) -> None:
        self.dup_paths = dup_paths
        self.inserted: list[dict] = []
        self.committed = False
        self._next_id = 1000

    def execute(self, sql, params=None):
        normalized = " ".join(sql.split())
        outer = self
        if normalized.startswith("SELECT 1 FROM jobs"):
            path = params[0]
            found = path in outer.dup_paths

            class _Cur:
                def fetchone(self):
                    return (1,) if found else None

            return _Cur()
        if normalized.startswith("INSERT INTO jobs"):
            record = json.loads(params[0])
            outer.inserted.append(record)
            outer._next_id += 1
            new_id = outer._next_id

            class _Cur:
                def fetchone(self):
                    return (new_id,)

            return _Cur()
        raise AssertionError(f"unexpected SQL: {normalized}")

    def commit(self):
        self.committed = True

    def rollback(self):
        pass


def _client(
    monkeypatch, *, records, dup_paths=frozenset(), raises=None
) -> tuple[TestClient, _FakeConn]:
    """Build a TestClient with the app's DB and MediaSource seams replaced by fakes."""
    from kanomori.api import app as app_module

    conn = _FakeConn(set(dup_paths))

    @contextlib.contextmanager
    def fake_connection():
        yield conn

    def fake_iter_manifest(source, manifest_path="manifest.jsonl"):
        if raises is not None:
            raise raises
        return records

    monkeypatch.setattr(app_module, "connection", fake_connection)
    monkeypatch.setattr(app_module, "get_media_source", lambda: object())
    monkeypatch.setattr(app_module, "iter_manifest", fake_iter_manifest)

    # No `with` -> lifespan (embedder warmup) does not run; keeps the test pure.
    return TestClient(app_module.create_app()), conn


def test_batch_enqueues_each_fresh_record(monkeypatch) -> None:
    client, conn = _client(monkeypatch, records=RECORDS)
    resp = client.post("/ingest/batch", json={"manifest_path": "manifest.jsonl"})

    assert resp.status_code == 200
    body = resp.json()
    assert body["total"] == 3
    assert body["skipped"] == []
    assert len(body["enqueued"]) == 3
    assert all(isinstance(j, int) for j in body["enqueued"])
    # Every record is inserted verbatim, keyed by `path` (the worker's canonical source key).
    assert [r["path"] for r in conn.inserted] == ["a/video.mp4", "b/video.mp4", "c/video.mp4"]
    assert conn.inserted[0] == RECORDS[0]
    assert conn.committed is True


def test_batch_skips_already_queued_paths(monkeypatch) -> None:
    client, conn = _client(monkeypatch, records=RECORDS, dup_paths={"b/video.mp4"})
    resp = client.post("/ingest/batch", json={})

    assert resp.status_code == 200
    body = resp.json()
    assert body["total"] == 3
    assert body["skipped"] == ["b/video.mp4"]
    assert len(body["enqueued"]) == 2
    # Only the non-dup records get inserted; the dup is never written.
    assert [r["path"] for r in conn.inserted] == ["a/video.mp4", "c/video.mp4"]


def test_batch_defaults_manifest_path(monkeypatch) -> None:
    seen: dict[str, str] = {}

    from kanomori.api import app as app_module

    conn = _FakeConn(set())

    @contextlib.contextmanager
    def fake_connection():
        yield conn

    def fake_iter_manifest(source, manifest_path="manifest.jsonl"):
        seen["manifest_path"] = manifest_path
        return RECORDS[:1]

    monkeypatch.setattr(app_module, "connection", fake_connection)
    monkeypatch.setattr(app_module, "get_media_source", lambda: object())
    monkeypatch.setattr(app_module, "iter_manifest", fake_iter_manifest)

    client = TestClient(app_module.create_app())
    resp = client.post("/ingest/batch", json={})

    assert resp.status_code == 200
    assert seen["manifest_path"] == "manifest.jsonl"


def test_batch_malformed_manifest_returns_400(monkeypatch) -> None:
    client, _conn = _client(
        monkeypatch,
        records=RECORDS,
        raises=MediaSourceError("malformed manifest at line 2: ..."),
    )
    resp = client.post("/ingest/batch", json={"manifest_path": "bad.jsonl"})

    assert resp.status_code == 400
    assert "line 2" in resp.json()["detail"]
