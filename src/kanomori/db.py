"""Database access: a process-wide psycopg3 connection pool with pgvector registered.

Why a module-level lazily-opened pool: the API opens it once in the FastAPI lifespan and
shares it across requests; the ingestion worker opens its own. ``connection()`` is a context
manager yielding a pooled connection with the pgvector type adapters already registered, so
callers can pass / read ``numpy`` arrays directly against ``vector`` columns.
"""

from __future__ import annotations

from collections.abc import Iterator
from contextlib import contextmanager

from psycopg import Connection
from psycopg_pool import ConnectionPool

from kanomori.config import get_settings

_pool: ConnectionPool | None = None


def _configure(conn: Connection) -> None:
    """Per-connection setup run by the pool: register pgvector adapters on this connection."""
    # Imported here (not at module top) so importing kanomori.db doesn't hard-require the
    # extension to be installed yet — register_vector needs the live connection anyway.
    from pgvector.psycopg import register_vector

    register_vector(conn)


def get_pool() -> ConnectionPool:
    """Return the lazily-initialized global connection pool."""
    global _pool
    if _pool is None:
        settings = get_settings()
        _pool = ConnectionPool(
            conninfo=settings.database_url,
            min_size=1,
            max_size=8,
            configure=_configure,
            open=True,
        )
    return _pool


@contextmanager
def connection() -> Iterator[Connection]:
    """Borrow a pooled connection (pgvector adapters registered). Commits on clean exit."""
    pool = get_pool()
    with pool.connection() as conn:
        yield conn


def close_pool() -> None:
    """Close the global pool (FastAPI shutdown / worker teardown / test cleanup)."""
    global _pool
    if _pool is not None:
        _pool.close()
        _pool = None
