"""Integration test for ``POST /ingest/batch`` against a real DB + the local samples manifest.

Points the configured MediaSource at ``samples/`` (KANOMORI_MEDIA_SOURCE=local), POSTs the
real 5-line manifest, and asserts five jobs are queued with the verbatim request dicts (keyed by
``path``). A second POST of the same manifest must skip all five — proving batch enqueue is
idempotent against queued/running jobs.
"""

from __future__ import annotations

from pathlib import Path

import pytest

from kanomori.config import get_settings

pytestmark = pytest.mark.requires_db

REPO_ROOT = Path(__file__).resolve().parents[2]
SAMPLES_DIR = REPO_ROOT / "samples"


@pytest.fixture
def local_source(monkeypatch):
    """Configure the source store to read the real samples/ manifest, then restore settings."""
    monkeypatch.setenv("KANOMORI_MEDIA_SOURCE", "local")
    monkeypatch.setenv("KANOMORI_MEDIA_SOURCE_ROOT", str(SAMPLES_DIR))
    get_settings.cache_clear()
    yield
    get_settings.cache_clear()


@pytest.fixture
def client(db_conn, fake_embedder, monkeypatch):
    """A TestClient with the embedder faked (no model download); shares the app DB pool."""
    from fastapi.testclient import TestClient

    from kanomori.api import app as app_module

    monkeypatch.setattr(app_module, "get_embedder", lambda: fake_embedder)
    app = app_module.create_app()
    with TestClient(app) as c:
        yield c


def test_batch_enqueues_five_then_skips_on_rerun(client, local_source, db_conn) -> None:
    resp = client.post("/ingest/batch", json={"manifest_path": "manifest.jsonl"})
    assert resp.status_code == 200
    body = resp.json()
    assert body["total"] == 5
    assert len(body["enqueued"]) == 5
    assert body["skipped"] == []

    # Each enqueued job stashed the verbatim manifest record (keyed by `path`) for the worker.
    rows = db_conn.execute(
        "SELECT stage_status->'request'->>'path' FROM jobs "
        "WHERE id = ANY(%s) ORDER BY id",
        (body["enqueued"],),
    ).fetchall()
    paths = [r[0] for r in rows]
    assert paths == [
        "鹿乃的2月18日歌回直播_2024-02-18/video.mp4",
        "鹿乃演唱会50w粉丝纪念_2020-02-22/video.mp4",
        "kano元気_2025-08-04/video.mp4",
        "鹿乃特别直播演唱会いつかの約束を君に_2019-10-19/video.mp4",
        "2024_talk_cut/video.mp4",
    ]

    # Re-POST: every line already has a queued job, so all five are skipped (idempotent).
    resp2 = client.post("/ingest/batch", json={"manifest_path": "manifest.jsonl"})
    assert resp2.status_code == 200
    body2 = resp2.json()
    assert body2["total"] == 5
    assert body2["enqueued"] == []
    assert len(body2["skipped"]) == 5
