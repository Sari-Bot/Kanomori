"""Minimal forward-only SQL migration runner.

Applies the ordered ``migrations/NNNN_*.sql`` files that haven't run yet, tracked in a
``schema_migrations`` table. Deliberately tiny — no down-migrations, no DSL. For an MVP this
beats pulling in Alembic: migrations are plain SQL a human can read, and ``uv run
kanomori-migrate`` is the one command to bring a fresh pgvector container up to schema.

Each file runs in its own transaction; a failure stops the run and leaves earlier files
applied (forward-only). Re-running is safe: already-applied files are skipped.

Migrations use a **plain** psycopg connection, not ``kanomori.db``'s pooled connection. The
pool registers pgvector type adapters on connect, which query the database for the ``vector``
type — but that type doesn't exist until ``0001_init.sql`` runs ``CREATE EXTENSION vector``.
Bootstrapping the schema therefore must not go through the pgvector-aware pool.
"""

from __future__ import annotations

from pathlib import Path

import psycopg

from kanomori.config import get_settings

# migrations/ sits at the repo root: src/kanomori/migrate.py -> parents[2]/migrations
MIGRATIONS_DIR = Path(__file__).resolve().parents[2] / "migrations"


def _ensure_table(conn) -> None:
    conn.execute(
        """
        CREATE TABLE IF NOT EXISTS schema_migrations (
            filename   text PRIMARY KEY,
            applied_at timestamptz NOT NULL DEFAULT now()
        )
        """
    )
    conn.commit()


def _applied(conn) -> set[str]:
    rows = conn.execute("SELECT filename FROM schema_migrations").fetchall()
    return {r[0] for r in rows}


def pending(conn) -> list[Path]:
    """Ordered list of migration files not yet recorded as applied."""
    done = _applied(conn)
    files = sorted(MIGRATIONS_DIR.glob("[0-9]*.sql"))
    return [f for f in files if f.name not in done]


def run() -> list[str]:
    """Apply all pending migrations in order. Returns the filenames applied this run."""
    applied: list[str] = []
    # Plain connection (no pgvector adapters) — see module docstring. autocommit so each
    # statement/file is durably applied; we commit explicitly per file below.
    with psycopg.connect(get_settings().database_url) as conn:
        _ensure_table(conn)
        for path in pending(conn):
            sql = path.read_text(encoding="utf-8")
            conn.execute(sql)
            conn.execute(
                "INSERT INTO schema_migrations (filename) VALUES (%s)", (path.name,)
            )
            conn.commit()
            applied.append(path.name)
    return applied


def main() -> None:
    applied = run()
    if applied:
        print(f"Applied {len(applied)} migration(s): {', '.join(applied)}")
    else:
        print("No pending migrations.")


if __name__ == "__main__":
    main()
