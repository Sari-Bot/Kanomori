"""Shared fixtures for integration tests that need a live PostgreSQL + pgvector.

A test DB connection is provided by the `db_conn` fixture, which connects to the configured
database, ensures the schema is migrated, and wraps each test in a transaction that is rolled
back at the end — so tests are isolated and leave no rows behind. Tests using it are marked
`requires_db` and skip cleanly when no database is reachable (lightweight CI / no container).

A deterministic `fake_embedder` is also provided: it maps text to real numpy vectors so tests
exercise real pgvector cosine search and real SQL, without loading BGE-M3. It is NOT a mock of
the search — the database does the actual vector math; only the model is substituted.
"""

from __future__ import annotations

import hashlib

import numpy as np
import pytest

from kanomori.embed.text_embedder import EMBED_DIM


def _db_reachable() -> bool:
    import psycopg

    from kanomori.config import get_settings

    try:
        with psycopg.connect(get_settings().database_url, connect_timeout=2):
            return True
    except Exception:
        return False


@pytest.fixture(scope="session", autouse=True)
def _close_pool_at_session_end():
    """Close the global connection pool when the session ends, so its background threads
    don't block interpreter shutdown (the "couldn't stop thread" warnings)."""
    yield
    from kanomori.db import close_pool

    close_pool()


@pytest.fixture(scope="session")
def _migrated() -> None:
    """Ensure the schema exists once per session (no-op if already migrated)."""
    from kanomori.migrate import run

    run()


@pytest.fixture
def db_conn(_migrated):
    """A pooled, pgvector-aware connection wrapped in a rolled-back transaction."""
    if not _db_reachable():
        pytest.skip("no PostgreSQL+pgvector reachable (start docker compose)")

    from kanomori.db import connection

    with connection() as conn:
        try:
            yield conn
        finally:
            conn.rollback()


class FakeEmbedder:
    """Deterministic text->vector map producing real unit-norm 1024-d vectors.

    Same text -> identical vector (so a query matches its indexed segment); different text ->
    near-orthogonal vector (seeded RNG per text hash). Lets pgvector do real cosine search.
    """

    def embed_texts(self, texts: list[str]) -> list[np.ndarray]:
        return [self._vec(t) for t in texts]

    def embed_query(self, text: str) -> np.ndarray:
        return self._vec(text)

    def _vec(self, text: str) -> np.ndarray:
        seed = int.from_bytes(hashlib.sha256(text.encode("utf-8")).digest()[:8], "big")
        rng = np.random.default_rng(seed)
        v = rng.standard_normal(EMBED_DIM).astype(np.float32)
        v /= np.linalg.norm(v) or 1.0
        return v


@pytest.fixture
def fake_embedder() -> FakeEmbedder:
    return FakeEmbedder()
