"""Pure (network-free) tests for the worker's coordinator HTTP client.

``CoordinatorClient`` is a thin httpx wrapper around the coordinator's ``/jobs/*`` contract
(Task #13). These tests inject a fake httpx client that records ``.post`` calls and returns
canned responses, so they pin down exactly what the worker puts on the wire — the bearer header
on every call, the JSON / multipart bodies, and the status-code policy (204 -> None on claim,
409 -> False on the fenced mutations, raise on any other 4xx/5xx) — without a live coordinator.
"""

from __future__ import annotations

import pytest

from kanomori.ingest.coordinator_client import CoordinatorClient

TOKEN = "worker-secret"
BASE = "http://coord.local:8000"


class _FakeResponse:
    """Minimal stand-in for httpx.Response: status, json(), raise_for_status()."""

    def __init__(self, status_code: int, json_data=None) -> None:
        self.status_code = status_code
        self._json_data = json_data

    def json(self):
        return self._json_data

    def raise_for_status(self):
        if self.status_code >= 400:
            raise RuntimeError(f"HTTP {self.status_code}")


class _FakeHttpClient:
    """Records every .post call and pops canned responses off a queue."""

    def __init__(self, responses: list[_FakeResponse]) -> None:
        self._responses = list(responses)
        self.calls: list[dict] = []

    def post(self, url, *, json=None, data=None, files=None, headers=None):
        self.calls.append(
            {"url": url, "json": json, "data": data, "files": files, "headers": headers}
        )
        return self._responses.pop(0)


def _client(responses):
    fake = _FakeHttpClient(responses)
    return CoordinatorClient(BASE, TOKEN, client=fake), fake


# --- claim -------------------------------------------------------------------------------


def test_claim_composes_url_json_and_bearer():
    job = {
        "job_id": 7,
        "content_hash": None,
        "lease_epoch": 1,
        "request": {"media_path": "/x.mp4"},
        "stages_done": [],
    }
    client, fake = _client([_FakeResponse(200, job)])

    result = client.claim("w1", 30)

    assert result == job
    call = fake.calls[0]
    assert call["url"] == f"{BASE}/jobs/claim"
    assert call["json"] == {"worker_id": "w1", "lease_seconds": 30}
    assert call["headers"]["Authorization"] == f"Bearer {TOKEN}"


def test_claim_returns_none_on_204():
    client, _ = _client([_FakeResponse(204)])
    assert client.claim("w1", 30) is None


def test_claim_raises_on_401():
    client, _ = _client([_FakeResponse(401)])
    with pytest.raises(RuntimeError):
        client.claim("w1", 30)


# --- heartbeat ---------------------------------------------------------------------------


def test_heartbeat_true_on_200_with_bearer():
    client, fake = _client([_FakeResponse(200, {"ok": True})])
    assert client.heartbeat(5, 2, 30) is True
    call = fake.calls[0]
    assert call["url"] == f"{BASE}/jobs/5/heartbeat"
    assert call["json"] == {"lease_epoch": 2, "lease_seconds": 30}
    assert call["headers"]["Authorization"] == f"Bearer {TOKEN}"


def test_heartbeat_false_on_409():
    client, _ = _client([_FakeResponse(409)])
    assert client.heartbeat(5, 2, 30) is False


# --- push_stage --------------------------------------------------------------------------


def test_push_stage_sends_multipart_and_bearer():
    client, fake = _client([_FakeResponse(200, {"ok": True})])
    files = [("frame_000000_000.jpg", b"jpegbytes"), ("frame_000008_000.jpg", b"more")]

    ok = client.push_stage(9, "frames", 3, '{"stage":"frames"}', files)

    assert ok is True
    call = fake.calls[0]
    assert call["url"] == f"{BASE}/jobs/9/stage/frames"
    # lease_epoch + result ride as form fields.
    assert call["data"]["lease_epoch"] == "3"
    assert call["data"]["result"] == '{"stage":"frames"}'
    # Each artifact is attached under the multipart field name "files".
    assert len(call["files"]) == 2
    for (field, payload), (name, content) in zip(call["files"], files, strict=True):
        assert field == "files"
        assert payload[0] == name
        assert payload[1] == content
    assert call["headers"]["Authorization"] == f"Bearer {TOKEN}"


def test_push_stage_empty_result_and_no_files():
    """locate_media carries no model: empty result string, zero files (still authorized)."""
    client, fake = _client([_FakeResponse(200, {"ok": True})])
    ok = client.push_stage(9, "locate_media", 3, "", [])
    assert ok is True
    call = fake.calls[0]
    assert call["data"]["result"] == ""
    assert call["data"]["lease_epoch"] == "3"
    assert call["files"] == []


def test_push_stage_false_on_409():
    client, _ = _client([_FakeResponse(409)])
    assert client.push_stage(9, "frames", 3, "{}", []) is False


def test_push_stage_raises_on_500():
    client, _ = _client([_FakeResponse(500)])
    with pytest.raises(RuntimeError):
        client.push_stage(9, "frames", 3, "{}", [])


# --- complete / fail ---------------------------------------------------------------------


def test_complete_true_on_200_false_on_409():
    client, fake = _client([_FakeResponse(200, {"ok": True})])
    assert client.complete(9, 3) is True
    call = fake.calls[0]
    assert call["url"] == f"{BASE}/jobs/9/complete"
    assert call["json"] == {"lease_epoch": 3}
    assert call["headers"]["Authorization"] == f"Bearer {TOKEN}"

    client2, _ = _client([_FakeResponse(409)])
    assert client2.complete(9, 3) is False


def test_fail_posts_error_and_false_on_409():
    client, fake = _client([_FakeResponse(200, {"ok": True})])
    assert client.fail(9, 3, "kits exploded") is True
    call = fake.calls[0]
    assert call["url"] == f"{BASE}/jobs/9/fail"
    assert call["json"] == {"lease_epoch": 3, "error": "kits exploded"}

    client2, _ = _client([_FakeResponse(409)])
    assert client2.fail(9, 3, "boom") is False


# --- token hygiene -----------------------------------------------------------------------


def test_token_not_in_repr():
    client, _ = _client([])
    assert TOKEN not in repr(client)
