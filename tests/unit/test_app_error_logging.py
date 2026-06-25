"""Focused tests for request-body logging on client request errors."""

from __future__ import annotations

import logging

from fastapi.testclient import TestClient

from kanomori.media_source import MediaSourceError


def test_logs_request_body_for_explicit_400(monkeypatch, caplog) -> None:
    from kanomori.api import app as app_module

    monkeypatch.setattr(app_module, "get_media_source", lambda: object())
    monkeypatch.setattr(
        app_module,
        "iter_manifest",
        lambda source, path: (_ for _ in ()).throw(MediaSourceError("manifest missing")),
    )
    caplog.set_level(logging.WARNING, logger="kanomori.api.app")

    client = TestClient(app_module.create_app())
    resp = client.post("/ingest/batch", json={"manifest_path": "missing.jsonl"})

    assert resp.status_code == 400
    assert any(
        "status=400" in record.message
        and "path=/ingest/batch" in record.message
        and '"manifest_path":"missing.jsonl"' in record.message
        for record in caplog.records
    )


def test_logs_request_body_for_malformed_json(monkeypatch, caplog) -> None:
    from kanomori.api import app as app_module

    caplog.set_level(logging.WARNING, logger="kanomori.api.app")

    client = TestClient(app_module.create_app())
    resp = client.post(
        "/search/transcript",
        content='{"query":',
        headers={"content-type": "application/json"},
    )

    assert resp.status_code == 422
    assert any(
        "status=422" in record.message
        and "path=/search/transcript" in record.message
        and 'body={"query":' in record.message
        for record in caplog.records
    )
