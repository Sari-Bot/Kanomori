"""Pure tests for the register stage's job-row identity resolution.

register computes the sha256 content_hash and reconciles the jobs row. Two code paths:

- ``ctx.job_id is None`` (single-machine callers / existing tests): upsert the jobs row keyed
  by content_hash (ON CONFLICT), as before — preserves backward compatibility.
- ``ctx.job_id is not None`` (enqueue-then-worker path): the job row already exists (enqueued
  with a NULL content_hash), so resolve identity by UPDATING that exact row by id, rather than
  INSERTing a second row and orphaning the enqueued one.

These tests use a fake connection that records the SQL it's handed, so they assert the branch
taken without needing a live database (the SQL semantics are covered by the requires_db tests).
"""

from __future__ import annotations

from pathlib import Path

from kanomori.ingest.pipeline import IngestContext
from kanomori.ingest.stages import register


class _FakeCursor:
    def __init__(self, video_id: int) -> None:
        self._video_id = video_id

    def fetchone(self):
        # Only the videos INSERT ... RETURNING id calls fetchone() in register.
        return (self._video_id,)


class _FakeConn:
    """Records every (sql, params) it executes; returns a video id for RETURNING queries."""

    def __init__(self, video_id: int = 42) -> None:
        self.calls: list[tuple[str, object]] = []
        self._video_id = video_id

    def execute(self, sql, params=None):
        self.calls.append((sql, params))
        return _FakeCursor(self._video_id)

    def _sql_text(self) -> str:
        return "\n".join(sql for sql, _ in self.calls)


def _media(tmp_path: Path) -> Path:
    p = tmp_path / "clip.mp4"
    p.write_bytes(b"register-identity-bytes")
    return p


def test_register_without_job_id_upserts_by_content_hash(tmp_path: Path) -> None:
    conn = _FakeConn()
    ctx = IngestContext(media_path=str(_media(tmp_path)))
    assert ctx.job_id is None  # default

    register.run(conn, ctx)

    sql = conn._sql_text()
    # Back-compat path: insert the jobs row, reconciling by content_hash.
    assert "INSERT INTO jobs" in sql
    assert "ON CONFLICT (content_hash)" in sql
    # Must NOT use the by-id update branch when no job id is known.
    assert "UPDATE jobs SET content_hash" not in sql
    assert len(ctx.content_hash) == 64  # sha256 hex still computed


def test_register_with_job_id_updates_existing_row_by_id(tmp_path: Path) -> None:
    conn = _FakeConn(video_id=7)
    ctx = IngestContext(media_path=str(_media(tmp_path)), job_id=99)

    register.run(conn, ctx)

    # The jobs row is reconciled by id (no second INSERT -> no orphan row).
    update_calls = [
        (sql, params)
        for sql, params in conn.calls
        if "UPDATE jobs SET content_hash" in sql
    ]
    assert len(update_calls) == 1
    _, params = update_calls[0]
    assert params == (ctx.content_hash, ctx.video_id, 99)
    # The enqueue-path must not also INSERT a jobs row.
    assert not any("INSERT INTO jobs" in sql for sql, _ in conn.calls)
    assert ctx.video_id == 7
