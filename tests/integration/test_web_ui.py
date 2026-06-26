"""Integration tests for the server-rendered web UI routes (Jinja2 + htmx).

The UI is thin: routes render templates around the existing retrieval/result functions. These
tests assert the routes return HTML 200s and embed the expected content (search form, result
cards, moment detail). Template markup itself isn't asserted line-by-line — only that the
route wires data into a rendered page. Uses the same fake embedder as the API tests.
"""

from __future__ import annotations

import pytest

pytestmark = pytest.mark.requires_db


@pytest.fixture
def client(db_conn, fake_embedder, monkeypatch):
    from fastapi.testclient import TestClient

    from kanomori.api import app as app_module

    monkeypatch.setattr(app_module, "get_embedder", lambda: fake_embedder)
    app = app_module.create_app()
    with TestClient(app) as c:
        yield c


@pytest.fixture
def seeded(db_conn, fake_embedder):
    from kanomori.text import tokenize_for_fts

    vid = db_conn.execute(
        "INSERT INTO videos (content_hash, title, source_url) VALUES "
        "('webhash', 'web ui test', 'https://example/web') RETURNING id"
    ).fetchone()[0]
    for seq, (s, e, text) in enumerate([(0.0, 5.0, "今日はゲーム配信"), (5.0, 10.0, "雑談タイム")]):
        db_conn.execute(
            "INSERT INTO transcript_segments "
            "(video_id, seq, start_sec, end_sec, text, text_norm, embedding, tsv) "
            "VALUES (%s,%s,%s,%s,%s,%s,%s, to_tsvector('simple', %s))",
            (vid, seq, s, e, text, text, fake_embedder.embed_query(text), tokenize_for_fts(text)),
        )
    db_conn.commit()
    return vid


def test_index_page_renders_search_form(client) -> None:
    resp = client.get("/")
    assert resp.status_code == 200
    assert "text/html" in resp.headers["content-type"]
    body = resp.text
    # Has a transcript query input and a screenshot upload control.
    assert 'name="query"' in body
    assert 'type="file"' in body
    assert "/ui/search/audio" in body


def test_search_fragment_returns_result_cards(client, seeded) -> None:
    # htmx posts the query; the route returns an HTML fragment of result cards.
    resp = client.post("/ui/search/transcript", data={"query": "ゲーム", "k": "5"})
    assert resp.status_code == 200
    assert "text/html" in resp.headers["content-type"]
    body = resp.text
    assert "今日はゲーム配信" in body              # the matched transcript snippet
    assert f"/result/{seeded}" in body or "/ui/result" in body  # a link to detail


def test_audio_search_fragment_renders_transcript_and_evidence(
    client, seeded, monkeypatch
) -> None:
    from kanomori.api import app as app_module

    def fake_normalize(_src, dst):
        dst.write_bytes(b"wav")

    class FakeASR:
        def transcribe(self, _path):
            return [{"start": 0.0, "end": 2.0, "text": "今日はゲーム配信"}]

    monkeypatch.setattr(app_module, "normalize_clip_to_wav", fake_normalize)
    monkeypatch.setattr(app_module, "probe_duration_sec", lambda _path: 5.0)
    monkeypatch.setattr(app_module, "get_asr", lambda: FakeASR())

    resp = client.post(
        "/ui/search/audio",
        files={"file": ("clip.wav", b"audio bytes", "audio/wav")},
        data={"k": "5"},
    )

    assert resp.status_code == 200
    assert "今日はゲーム配信" in resp.text
    assert "coverage" in resp.text


def test_search_fragment_empty_query_renders_no_results_message(client, seeded) -> None:
    resp = client.post("/ui/search/transcript", data={"query": "", "k": "5"})
    assert resp.status_code == 200
    # No crash on empty query; renders a page/fragment (possibly an empty-state message).
    assert "text/html" in resp.headers["content-type"]


def test_result_page_renders_moment_detail(client, seeded) -> None:
    resp = client.get(f"/ui/result/{seeded}", params={"ts": 0.0})
    assert resp.status_code == 200
    assert "text/html" in resp.headers["content-type"]
    body = resp.text
    assert "web ui test" in body                  # video title
    assert "https://example/web" in body          # source link
    assert "今日はゲーム配信" in body              # nearby transcript


def test_result_page_404_for_unknown_video(client) -> None:
    resp = client.get("/ui/result/99999999", params={"ts": 0.0})
    assert resp.status_code == 404
