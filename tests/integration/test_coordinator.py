"""Integration tests for the coordinator: claim/lease/fence SQL + the /jobs router end-to-end.

All against a live PostgreSQL+pgvector (requires_db; skipped when no DB is reachable). Two
layers are covered:

1. The pure-ish lease functions in ``kanomori.ingest.lease`` — claim leases the oldest eligible
   job and bumps the fencing epoch; SKIP LOCKED prevents a concurrent claimer double-grabbing;
   an expired lease makes a running job re-claimable; heartbeat/mark_stage_done/complete/fail
   are all fence-checked on ``lease_epoch``.

2. The FastAPI ``/jobs/*`` router via TestClient with a bearer token: claim -> push the register
   stage (the videos row appears and the job's content_hash is reconciled) -> push a couple more
   stages -> complete. Auth failures (401) and stale-epoch pushes (409) are asserted too.
"""

from __future__ import annotations

import json
from pathlib import Path

import pytest

from kanomori.config import get_settings
from kanomori.ingest import lease

pytestmark = pytest.mark.requires_db

TOKEN = "integration-coordinator-secret"


def _clear_active(conn) -> None:
    """Mark any in-flight jobs complete so a test sees only the rows it enqueues."""
    conn.execute("UPDATE jobs SET status='complete' WHERE status IN ('queued','failed','running')")


def _enqueue(conn, media_path: str = "/tmp/x.mp4") -> int:
    """Mirror the API's /ingest: one queued job, NULL content_hash, request stashed."""
    payload = json.dumps({"media_path": media_path})
    return conn.execute(
        """
        INSERT INTO jobs (content_hash, status, stage_status)
        VALUES (NULL, 'queued', jsonb_build_object('request', %s::jsonb))
        RETURNING id
        """,
        (payload,),
    ).fetchone()[0]


# --- claim / lease / fence (lease.py) ----------------------------------------------------


def test_claim_leases_oldest_queued_and_bumps_epoch(db_conn):
    job_id = _enqueue(db_conn, "/a.mp4")
    db_conn.commit()

    claimed = lease.claim(db_conn, worker_id="w1", lease_seconds=60)
    assert claimed is not None
    assert claimed["job_id"] == job_id
    assert claimed["content_hash"] is None
    assert claimed["lease_epoch"] == 1  # bumped from the default 0
    assert claimed["request"] == {"media_path": "/a.mp4"}
    assert claimed["stages_done"] == []

    row = db_conn.execute(
        "SELECT status, worker_id, lease_epoch, lease_expires_at > now() FROM jobs WHERE id=%s",
        (job_id,),
    ).fetchone()
    assert row[0] == "running"
    assert row[1] == "w1"
    assert row[2] == 1
    assert row[3] is True  # lease set into the future


def test_claim_returns_none_when_nothing_eligible(db_conn):
    _clear_active(db_conn)
    db_conn.commit()
    assert lease.claim(db_conn, worker_id="w1", lease_seconds=60) is None


def test_concurrent_claim_does_not_double_grab(db_conn):
    """SKIP LOCKED: with exactly one eligible job, a second claimer on another connection that
    runs while the first holds the row lock must get nothing (not the same job)."""
    import psycopg

    _clear_active(db_conn)
    job_id = _enqueue(db_conn, "/only.mp4")
    db_conn.commit()

    # First claimer (db_conn) leaves its transaction open, holding the FOR UPDATE row lock.
    first = lease.claim(db_conn, worker_id="w1", lease_seconds=60)
    assert first is not None and first["job_id"] == job_id

    with psycopg.connect(get_settings().database_url) as other:
        second = lease.claim(other, worker_id="w2", lease_seconds=60)
        other.commit()
    assert second is None  # SKIP LOCKED skipped the locked row; nothing else eligible


def test_expired_lease_is_reclaimable_and_bumps_epoch_again(db_conn):
    job_id = _enqueue(db_conn, "/b.mp4")
    db_conn.commit()
    first = lease.claim(db_conn, worker_id="w1", lease_seconds=60)
    assert first["lease_epoch"] == 1

    # Simulate the lease lapsing (worker died) by backdating it.
    db_conn.execute(
        "UPDATE jobs SET lease_expires_at = now() - interval '1 second' WHERE id=%s", (job_id,)
    )
    db_conn.commit()

    second = lease.claim(db_conn, worker_id="w2", lease_seconds=60)
    assert second is not None
    assert second["job_id"] == job_id
    assert second["lease_epoch"] == 2  # epoch advanced on re-claim (fences the zombie w1)


def test_kill_and_reclaim_resumes_without_redoing_done_stages(db_conn):
    """§7 kill-and-reclaim: worker A runs partway then dies; worker B reclaims and resumes.

    A claims, records register + transcribe done (the expensive GPU SRT), then vanishes. The
    lease lapses. B reclaims at a bumped epoch, is told via ``stages_done`` that register and
    transcribe are already finished (so it skips them — the SRT is NOT recomputed), and the
    zombie A is fenced: its stale-epoch heartbeat / stage-push / complete all fail.
    """
    job_id = _enqueue(db_conn, "/karaoke.mp4")
    db_conn.commit()

    # Worker A claims and completes register + transcribe, then "dies".
    a = lease.claim(db_conn, worker_id="A", lease_seconds=60)
    a_epoch = a["lease_epoch"]
    assert a["stages_done"] == []
    assert lease.mark_stage_done(db_conn, job_id, a_epoch, "register") is True
    assert lease.mark_stage_done(db_conn, job_id, a_epoch, "transcribe") is True
    db_conn.commit()

    # Lease lapses (A died mid-parse_transcript without heartbeating).
    db_conn.execute(
        "UPDATE jobs SET lease_expires_at = now() - interval '1 second' WHERE id=%s", (job_id,)
    )
    db_conn.commit()

    # Worker B reclaims: epoch bumps, and it learns which stages are already done.
    b = lease.claim(db_conn, worker_id="B", lease_seconds=60)
    db_conn.commit()
    assert b["job_id"] == job_id
    assert b["lease_epoch"] == a_epoch + 1
    assert set(b["stages_done"]) == {"register", "transcribe"}  # B skips these → SRT not redone

    # The zombie A is fully fenced at its stale epoch: every mutation fails (0 rows).
    assert lease.heartbeat(db_conn, job_id, a_epoch, 60) is False
    assert lease.mark_stage_done(db_conn, job_id, a_epoch, "parse_transcript") is False
    assert lease.complete(db_conn, job_id, a_epoch) is False
    db_conn.commit()

    # B carries the job to completion on its live epoch.
    assert lease.mark_stage_done(db_conn, job_id, b["lease_epoch"], "parse_transcript") is True
    assert lease.complete(db_conn, job_id, b["lease_epoch"]) is True
    db_conn.commit()
    status = db_conn.execute("SELECT status FROM jobs WHERE id=%s", (job_id,)).fetchone()[0]
    assert status == "complete"



def test_heartbeat_extends_on_fresh_epoch_and_fails_on_stale(db_conn):
    job_id = _enqueue(db_conn)
    db_conn.commit()
    claimed = lease.claim(db_conn, worker_id="w1", lease_seconds=1)
    epoch = claimed["lease_epoch"]

    assert lease.heartbeat(db_conn, job_id, epoch, lease_seconds=120) is True
    expires = db_conn.execute(
        "SELECT lease_expires_at > now() + interval '60 seconds' FROM jobs WHERE id=%s", (job_id,)
    ).fetchone()[0]
    assert expires is True  # lease pushed well into the future

    # A stale epoch (zombie worker) is rejected.
    assert lease.heartbeat(db_conn, job_id, epoch - 1, lease_seconds=120) is False


def test_mark_stage_done_is_fence_checked(db_conn):
    job_id = _enqueue(db_conn)
    db_conn.commit()
    epoch = lease.claim(db_conn, worker_id="w1", lease_seconds=60)["lease_epoch"]

    assert lease.mark_stage_done(db_conn, job_id, epoch, "transcribe") is True
    state = db_conn.execute(
        "SELECT stage_status -> 'transcribe' ->> 'state' FROM jobs WHERE id=%s", (job_id,)
    ).fetchone()[0]
    assert state == "done"

    # Stale epoch can't mark stages.
    assert lease.mark_stage_done(db_conn, job_id, epoch - 1, "frames") is False


def test_complete_and_fail_are_fence_checked(db_conn):
    job_id = _enqueue(db_conn)
    db_conn.commit()
    epoch = lease.claim(db_conn, worker_id="w1", lease_seconds=60)["lease_epoch"]

    assert lease.complete(db_conn, job_id, epoch - 1) is False  # stale
    assert lease.complete(db_conn, job_id, epoch) is True
    status = db_conn.execute("SELECT status FROM jobs WHERE id=%s", (job_id,)).fetchone()[0]
    assert status == "complete"


def test_fail_bumps_attempts_and_records_error(db_conn):
    job_id = _enqueue(db_conn)
    db_conn.commit()
    epoch = lease.claim(db_conn, worker_id="w1", lease_seconds=60)["lease_epoch"]

    assert lease.fail(db_conn, job_id, epoch, "kits exploded") is True
    row = db_conn.execute(
        "SELECT status, error, attempts FROM jobs WHERE id=%s", (job_id,)
    ).fetchone()
    assert row[0] == "failed"
    assert "kits exploded" in row[1]
    assert row[2] == 1


def test_failed_job_below_cap_is_claimable(db_conn):
    _clear_active(db_conn)
    job_id = _enqueue(db_conn)
    db_conn.execute(
        "UPDATE jobs SET status='failed', attempts=%s WHERE id=%s",
        (lease.MAX_ATTEMPTS - 1, job_id),
    )
    db_conn.commit()
    claimed = lease.claim(db_conn, worker_id="w1", lease_seconds=60)
    assert claimed is not None and claimed["job_id"] == job_id


def test_failed_job_at_cap_is_not_claimable(db_conn):
    _clear_active(db_conn)
    job_id = _enqueue(db_conn)
    db_conn.execute(
        "UPDATE jobs SET status='failed', attempts=%s WHERE id=%s",
        (lease.MAX_ATTEMPTS, job_id),
    )
    db_conn.commit()
    assert lease.claim(db_conn, worker_id="w1", lease_seconds=60) is None


# --- /jobs router end-to-end (TestClient + bearer token) ---------------------------------


@pytest.fixture
def media_file(tmp_path: Path) -> Path:
    p = tmp_path / "clip.mp4"
    p.write_bytes(b"coordinator-router-flow-bytes")
    return p


@pytest.fixture
def client(db_conn, monkeypatch):
    """TestClient against the real DB pool, with the coordinator token configured."""
    from fastapi.testclient import TestClient

    monkeypatch.setenv("KANOMORI_COORDINATOR_TOKEN", TOKEN)
    monkeypatch.setenv("KANOMORI_PRELOAD_SEARCH_MODELS", "false")
    get_settings.cache_clear()

    from kanomori.api import app as app_module

    app = app_module.create_app()
    with TestClient(app) as c:
        yield c
    get_settings.cache_clear()


def _auth(token: str = TOKEN) -> dict[str, str]:
    return {"Authorization": f"Bearer {token}"}


def _result_file(payload: dict) -> dict[str, tuple[str, str, str]]:
    return {"result_file": ("result.json", json.dumps(payload), "application/json")}


def test_router_claim_register_complete_flow(client, db_conn, media_file):
    # Enqueue via the real /ingest so the request payload + NULL hash are set the production way.
    job_id = client.post("/ingest", json={"media_path": str(media_file)}).json()["job_id"]

    claim = client.post(
        "/jobs/claim", json={"worker_id": "w1", "lease_seconds": 120}, headers=_auth()
    ).json()
    assert claim["job_id"] == job_id
    epoch = claim["lease_epoch"]

    # Push the register stage: a RegisterResult JSON (register.persist runs plain SQL).
    register_result = {
        "stage": "register",
        "content_hash": "a" * 64,
        "source_url": "https://example/v",
        "title": "router flow",
    }
    resp = client.post(
        f"/jobs/{job_id}/stage/register",
        data={"lease_epoch": str(epoch), "compute_seconds": "1.111"},
        files=_result_file(register_result),
        headers=_auth(),
    )
    assert resp.status_code == 200, resp.text

    # The videos row now exists and the job's content_hash/video_id were reconciled in place.
    vid = db_conn.execute(
        "SELECT id FROM videos WHERE content_hash=%s", ("a" * 64,)
    ).fetchone()
    assert vid is not None
    jobrow = db_conn.execute(
        "SELECT content_hash, video_id FROM jobs WHERE id=%s", (job_id,)
    ).fetchone()
    assert jobrow[0] == "a" * 64
    assert jobrow[1] == vid[0]

    # Push a no-DB stage (locate_media) and a classify stage with one scene segment.
    resp = client.post(
        f"/jobs/{job_id}/stage/locate_media",
        data={"lease_epoch": str(epoch), "compute_seconds": "2.222"},
        headers=_auth(),
    )
    assert resp.status_code == 200, resp.text

    classify_result = {
        "stage": "classify",
        "segments": [
            {"start_sec": 0.0, "end_sec": 8.0, "scene_type": "chatting", "confidence": 0.9}
        ],
        "stream_type": "chatting",
    }
    resp = client.post(
        f"/jobs/{job_id}/stage/classify",
        data={"lease_epoch": str(epoch), "compute_seconds": "3.333"},
        files=_result_file(classify_result),
        headers=_auth(),
    )
    assert resp.status_code == 200, resp.text
    scene = db_conn.execute(
        "SELECT scene_type FROM scene_segments WHERE video_id=%s", (vid[0],)
    ).fetchone()
    assert scene[0] == "chatting"

    # Complete.
    resp = client.post(
        f"/jobs/{job_id}/complete", json={"lease_epoch": epoch}, headers=_auth()
    )
    assert resp.status_code == 200, resp.text
    status = db_conn.execute("SELECT status FROM jobs WHERE id=%s", (job_id,)).fetchone()[0]
    assert status == "complete"
    time_costs = db_conn.execute("SELECT time_costs FROM jobs WHERE id=%s", (job_id,)).fetchone()[0]
    assert time_costs == [
        {"stage": "register", "seconds": 1.111},
        {"stage": "locate_media", "seconds": 2.222},
        {"stage": "classify", "seconds": 3.333},
    ]


def test_router_frames_stage_saves_uploaded_jpeg(client, db_conn, media_file):
    """The frames stage uploads JPEG artifacts; the coordinator stores them under
    frame_dir_for(content_hash) and persists the frames rows."""
    from kanomori.ingest.artifacts import frame_dir_for, frame_path_for

    content_hash = "b" * 64
    job_id = client.post("/ingest", json={"media_path": str(media_file)}).json()["job_id"]
    claim = client.post("/jobs/claim", json={"worker_id": "w1"}, headers=_auth()).json()
    epoch = claim["lease_epoch"]

    # Register first so content_hash + video_id are set on the job.
    client.post(
        f"/jobs/{job_id}/stage/register",
        data={"lease_epoch": str(epoch), "compute_seconds": "0.5"},
        files=_result_file({"stage": "register", "content_hash": content_hash}),
        headers=_auth(),
    )

    frame_name = frame_path_for(content_hash, 8.0).name
    frames_result = {
        "stage": "frames",
        "frames": [{"ts_sec": 8.0, "artifact": frame_name}],
        "scene_timestamps": [],
    }
    resp = client.post(
        f"/jobs/{job_id}/stage/frames",
        data={"lease_epoch": str(epoch), "compute_seconds": "4.444"},
        files=[
            ("result_file", ("result.json", json.dumps(frames_result), "application/json")),
            ("files", (frame_name, b"\xff\xd8\xff-fake-jpeg", "image/jpeg")),
        ],
        headers=_auth(),
    )
    assert resp.status_code == 200, resp.text

    saved = frame_dir_for(content_hash) / frame_name
    assert saved.is_file()
    assert saved.read_bytes() == b"\xff\xd8\xff-fake-jpeg"

    vid = db_conn.execute(
        "SELECT id FROM videos WHERE content_hash=%s", (content_hash,)
    ).fetchone()[0]
    frow = db_conn.execute(
        "SELECT ts_sec FROM frames WHERE video_id=%s", (vid,)
    ).fetchone()
    assert frow[0] == pytest.approx(8.0)

    # Cleanup the artifact we wrote under the real media_root.
    saved.unlink(missing_ok=True)


def test_router_retry_replaces_stage_time_cost_on_success(client, db_conn, media_file):
    job_id = client.post("/ingest", json={"media_path": str(media_file)}).json()["job_id"]
    first_claim = client.post("/jobs/claim", json={"worker_id": "w1"}, headers=_auth()).json()
    first_epoch = first_claim["lease_epoch"]
    register_result = {"stage": "register", "content_hash": "c" * 64}

    resp = client.post(
        f"/jobs/{job_id}/stage/register",
        data={"lease_epoch": str(first_epoch), "compute_seconds": "1.111"},
        files=_result_file(register_result),
        headers=_auth(),
    )
    assert resp.status_code == 200

    db_conn.execute(
        "UPDATE jobs SET lease_expires_at = now() - interval '1 second' WHERE id=%s", (job_id,)
    )
    db_conn.commit()

    second_claim = client.post("/jobs/claim", json={"worker_id": "w2"}, headers=_auth()).json()
    assert second_claim["job_id"] == job_id
    assert second_claim["lease_epoch"] == first_epoch + 1

    resp = client.post(
        f"/jobs/{job_id}/stage/register",
        data={"lease_epoch": str(second_claim['lease_epoch']), "compute_seconds": "9.999"},
        files=_result_file(register_result),
        headers=_auth(),
    )
    assert resp.status_code == 200

    time_costs = db_conn.execute("SELECT time_costs FROM jobs WHERE id=%s", (job_id,)).fetchone()[0]
    assert time_costs == [{"stage": "register", "seconds": 9.999}]


def test_router_401_without_bearer(client):
    resp = client.post("/jobs/claim", json={"worker_id": "w1"})
    assert resp.status_code == 401


def test_router_stage_409_on_stale_epoch(client, media_file):
    job_id = client.post("/ingest", json={"media_path": str(media_file)}).json()["job_id"]
    claim = client.post("/jobs/claim", json={"worker_id": "w1"}, headers=_auth()).json()
    stale = claim["lease_epoch"] - 1
    resp = client.post(
        f"/jobs/{job_id}/stage/locate_media",
        data={"lease_epoch": str(stale), "result": ""},
        headers=_auth(),
    )
    assert resp.status_code == 409
