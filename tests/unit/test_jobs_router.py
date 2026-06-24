"""Pure (DB-free) tests for the coordinator /jobs router: auth, claim wiring, stage mapping.

These exercise the real FastAPI router with the DB layer monkeypatched out, so they give real
coverage of the bearer-token gate, the 503-when-unconfigured policy, multipart parsing, the
stage_name -> StageResult model mapping, and the HTTP status contract (200/204/401/409/503)
without needing a live PostgreSQL. The transactional SQL itself is covered by the requires_db
suite in tests/integration/test_coordinator.py.
"""

from __future__ import annotations

import contextlib
import json

import pytest
from fastapi.testclient import TestClient

from kanomori.config import get_settings
from kanomori.ingest import lease
from kanomori.ingest.stage_result import (
    ClassifyResult,
    FramesResult,
    ImageEmbedResult,
    OcrResult,
    ParseTranscriptResult,
    RegisterResult,
    TranscribeResult,
)

TOKEN = "test-coordinator-secret"


@pytest.fixture
def set_token(monkeypatch):
    """Configure the shared bearer token via env and rebuild the cached Settings."""
    monkeypatch.setenv("KANOMORI_COORDINATOR_TOKEN", TOKEN)
    get_settings.cache_clear()
    yield TOKEN
    get_settings.cache_clear()


@pytest.fixture
def no_token(monkeypatch):
    """Ensure no coordinator token is configured (auth must fail closed)."""
    monkeypatch.delenv("KANOMORI_COORDINATOR_TOKEN", raising=False)
    get_settings.cache_clear()
    yield
    get_settings.cache_clear()


class _FakeConn:
    """A no-op connection: the lease functions are monkeypatched, so SQL never runs."""

    def execute(self, *a, **k):
        raise AssertionError("DB should not be touched in pure router tests")

    def commit(self):
        pass

    def rollback(self):
        pass


@pytest.fixture
def app_client(monkeypatch):
    """Build the app with the /jobs DB seam (connection) replaced by a fake."""
    from kanomori.api import jobs as jobs_module

    @contextlib.contextmanager
    def fake_connection():
        yield _FakeConn()

    monkeypatch.setattr(jobs_module, "connection", fake_connection)

    from kanomori.api import app as app_module

    app = app_module.create_app()
    # No `with` -> lifespan (embedder warmup) does not run; keeps the pure test light.
    return TestClient(app)


def _auth(token: str = TOKEN) -> dict[str, str]:
    return {"Authorization": f"Bearer {token}"}


# --- auth gate ---------------------------------------------------------------------------


def test_claim_503_when_token_unconfigured(no_token, app_client):
    resp = app_client.post("/jobs/claim", json={"worker_id": "w1"}, headers=_auth("anything"))
    assert resp.status_code == 503


def test_claim_401_when_token_missing(set_token, app_client):
    resp = app_client.post("/jobs/claim", json={"worker_id": "w1"})
    assert resp.status_code == 401


def test_claim_401_when_token_wrong(set_token, app_client):
    resp = app_client.post("/jobs/claim", json={"worker_id": "w1"}, headers=_auth("nope"))
    assert resp.status_code == 401


# --- claim wiring ------------------------------------------------------------------------


def test_claim_returns_job_dict(set_token, app_client, monkeypatch):
    monkeypatch.setattr(
        lease,
        "claim",
        lambda conn, worker_id, lease_seconds: {
            "job_id": 7,
            "content_hash": None,
            "lease_epoch": 1,
            "request": {"media_path": "/x.mp4"},
            "stages_done": [],
        },
    )
    resp = app_client.post(
        "/jobs/claim", json={"worker_id": "w1", "lease_seconds": 30}, headers=_auth()
    )
    assert resp.status_code == 200
    body = resp.json()
    assert body["job_id"] == 7
    assert body["lease_epoch"] == 1
    assert body["content_hash"] is None
    assert body["request"] == {"media_path": "/x.mp4"}
    assert body["stages_done"] == []


def test_claim_204_when_no_job(set_token, app_client, monkeypatch):
    monkeypatch.setattr(lease, "claim", lambda conn, worker_id, lease_seconds: None)
    resp = app_client.post("/jobs/claim", json={"worker_id": "w1"}, headers=_auth())
    assert resp.status_code == 204


# --- heartbeat fencing -------------------------------------------------------------------


def test_heartbeat_409_on_stale_epoch(set_token, app_client, monkeypatch):
    monkeypatch.setattr(lease, "heartbeat", lambda conn, job_id, lease_epoch, lease_seconds: False)
    resp = app_client.post("/jobs/5/heartbeat", json={"lease_epoch": 2}, headers=_auth())
    assert resp.status_code == 409


def test_heartbeat_200_on_fresh_epoch(set_token, app_client, monkeypatch):
    monkeypatch.setattr(lease, "heartbeat", lambda conn, job_id, lease_epoch, lease_seconds: True)
    resp = app_client.post("/jobs/5/heartbeat", json={"lease_epoch": 2}, headers=_auth())
    assert resp.status_code == 200
    assert resp.json()["ok"] is True


# --- stage_name -> model mapping ---------------------------------------------------------


def test_stage_specs_cover_every_pipeline_stage():
    from kanomori.ingest.pipeline import STAGES

    assert set(lease.STAGE_SPECS) == {name for name, _mod in STAGES}


def test_stage_specs_models_and_artifacts():
    expected_model = {
        "register": RegisterResult,
        "locate_media": None,
        "transcribe": TranscribeResult,
        "parse_transcript": ParseTranscriptResult,
        "frames": FramesResult,
        "ocr": OcrResult,
        "classify": ClassifyResult,
        "image_embed": ImageEmbedResult,
    }
    expected_artifact = {
        "register": None,
        "locate_media": None,
        "transcribe": "srt",
        "parse_transcript": None,
        "frames": "frame",
        "ocr": None,
        "classify": None,
        "image_embed": None,
    }
    for name, spec in lease.STAGE_SPECS.items():
        assert spec.model is expected_model[name], name
        assert spec.artifact_kind == expected_artifact[name], name


# --- stage push fencing (DB seam faked) --------------------------------------------------


def test_stage_push_409_on_stale_epoch(set_token, monkeypatch):
    """When the job's current epoch != supplied epoch, the push is rejected 409 and no
    persist/artifact work happens."""
    from kanomori.api import jobs as jobs_module

    class _EpochConn:
        def execute(self, sql, params=None):
            # The router's first statement is the fence SELECT ... FOR UPDATE.
            class _Cur:
                # (content_hash, video_id, lease_epoch) — current epoch 9 != supplied 2.
                def fetchone(self_inner):
                    return (None, None, 9)

            return _Cur()

        def commit(self):
            pass

        def rollback(self):
            pass

    @contextlib.contextmanager
    def fake_connection():
        yield _EpochConn()

    monkeypatch.setattr(jobs_module, "connection", fake_connection)

    def _boom(*a, **k):
        raise AssertionError("persist must not run on a fenced push")

    monkeypatch.setattr(lease, "persist_stage", _boom)

    from kanomori.api import app as app_module

    client = TestClient(app_module.create_app())
    resp = client.post(
        "/jobs/5/stage/parse_transcript",
        data={"lease_epoch": "2"},
        files={"result_file": ("result.json", b"{}", "application/json")},
        headers=_auth(),
    )
    assert resp.status_code == 409


def test_stage_push_404_for_unknown_stage(set_token, app_client):
    resp = app_client.post(
        "/jobs/5/stage/not_a_stage",
        data={"lease_epoch": "1"},
        files={"result_file": ("result.json", b"{}", "application/json")},
        headers=_auth(),
    )
    assert resp.status_code == 404


def test_stage_push_400_when_model_stage_missing_result_file(set_token, monkeypatch):
    from kanomori.api import jobs as jobs_module

    class _FreshConn:
        def execute(self, sql, params=None):
            class _Cur:
                def fetchone(self_inner):
                    return (None, None, 2)

            return _Cur()

        def commit(self):
            pass

        def rollback(self):
            pass

    @contextlib.contextmanager
    def fake_connection():
        yield _FreshConn()

    monkeypatch.setattr(jobs_module, "connection", fake_connection)

    from kanomori.api import app as app_module

    client = TestClient(app_module.create_app())
    resp = client.post("/jobs/5/stage/parse_transcript", data={"lease_epoch": "2"}, headers=_auth())
    assert resp.status_code == 400
    assert resp.json()["detail"] == "missing result payload for this stage"


def test_stage_push_400_for_invalid_result_json_file(set_token, monkeypatch):
    from kanomori.api import jobs as jobs_module

    class _FreshConn:
        def execute(self, sql, params=None):
            class _Cur:
                def fetchone(self_inner):
                    return (None, None, 2)

            return _Cur()

        def commit(self):
            pass

        def rollback(self):
            pass

    @contextlib.contextmanager
    def fake_connection():
        yield _FreshConn()

    monkeypatch.setattr(jobs_module, "connection", fake_connection)

    from kanomori.api import app as app_module

    client = TestClient(app_module.create_app())
    resp = client.post(
        "/jobs/5/stage/parse_transcript",
        data={"lease_epoch": "2"},
        files={"result_file": ("result.json", b"{", "application/json")},
        headers=_auth(),
    )
    assert resp.status_code == 400
    assert "invalid result JSON" in resp.json()["detail"]


def test_stage_push_400_for_invalid_result_utf8(set_token, monkeypatch):
    from kanomori.api import jobs as jobs_module

    class _FreshConn:
        def execute(self, sql, params=None):
            class _Cur:
                def fetchone(self_inner):
                    return (None, None, 2)

            return _Cur()

        def commit(self):
            pass

        def rollback(self):
            pass

    @contextlib.contextmanager
    def fake_connection():
        yield _FreshConn()

    monkeypatch.setattr(jobs_module, "connection", fake_connection)

    from kanomori.api import app as app_module

    client = TestClient(app_module.create_app())
    resp = client.post(
        "/jobs/5/stage/parse_transcript",
        data={"lease_epoch": "2"},
        files={"result_file": ("result.json", b"\xff", "application/json")},
        headers=_auth(),
    )
    assert resp.status_code == 400
    assert "invalid result UTF-8" in resp.json()["detail"]


def test_stage_push_413_when_result_file_exceeds_limit(set_token, monkeypatch):
    from kanomori.api import jobs as jobs_module

    monkeypatch.setenv("KANOMORI_STAGE_RESULT_MAX_BYTES", "32")
    get_settings.cache_clear()

    class _FreshConn:
        def execute(self, sql, params=None):
            class _Cur:
                def fetchone(self_inner):
                    return (None, None, 2)

            return _Cur()

        def commit(self):
            pass

        def rollback(self):
            pass

    @contextlib.contextmanager
    def fake_connection():
        yield _FreshConn()

    monkeypatch.setattr(jobs_module, "connection", fake_connection)

    from kanomori.api import app as app_module

    body = json.dumps({"stage": "parse_transcript", "segments": []}).encode("utf-8")
    client = TestClient(app_module.create_app())
    resp = client.post(
        "/jobs/5/stage/parse_transcript",
        data={"lease_epoch": "2"},
        files={"result_file": ("result.json", body, "application/json")},
        headers=_auth(),
    )
    assert resp.status_code == 413
    assert resp.json()["detail"] == "stage result file exceeded maximum size of 32 bytes"


def test_stage_push_accepts_large_result_file_beyond_old_form_limit(set_token, monkeypatch):
    from kanomori.api import jobs as jobs_module

    class _FreshConn:
        def __init__(self):
            self.persisted = None

        def execute(self, sql, params=None):
            class _Cur:
                def fetchone(self_inner):
                    return (None, None, 2)

            return _Cur()

        def commit(self):
            pass

        def rollback(self):
            pass

    holder = {}

    @contextlib.contextmanager
    def fake_connection():
        conn = _FreshConn()
        holder["conn"] = conn
        yield conn

    monkeypatch.setattr(jobs_module, "connection", fake_connection)

    def fake_persist_stage(conn, stage_name, result, **kwargs):
        conn.persisted = (stage_name, result)
        return None

    monkeypatch.setattr(lease, "persist_stage", fake_persist_stage)
    monkeypatch.setattr(
        lease,
        "mark_stage_done",
        lambda conn, job_id, lease_epoch, stage_name: True,
    )

    from kanomori.api import app as app_module

    payload = {
        "stage": "parse_transcript",
        "segments": [
            {
                "seq": idx,
                "start_sec": float(idx),
                "end_sec": float(idx + 1),
                "text": "a" * 40,
                "text_norm": "a" * 40,
                "embedding": "x" * 5464,
            }
            for idx in range(200)
        ],
    }
    body = json.dumps(payload).encode("utf-8")
    assert len(body) > 1024 * 1024

    client = TestClient(app_module.create_app())
    resp = client.post(
        "/jobs/5/stage/parse_transcript",
        data={"lease_epoch": "2"},
        files={"result_file": ("result.json", body, "application/json")},
        headers=_auth(),
    )
    assert resp.status_code == 200
    assert holder["conn"].persisted[0] == "parse_transcript"
    assert len(holder["conn"].persisted[1].segments) == 200
