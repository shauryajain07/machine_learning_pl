"""Supabase PostgreSQL connection helpers."""

from __future__ import annotations

import os


def resolve_database_url(explicit: str | None = None) -> str | None:
    return explicit or os.getenv("SUPABASE_DB_URL") or os.getenv("DATABASE_URL")


def connect(database_url: str | None = None):
    url = resolve_database_url(database_url)
    if not url:
        raise RuntimeError(
            "No database URL supplied. Set SUPABASE_DB_URL (or DATABASE_URL) "
            "to a Supabase PostgreSQL connection string."
        )
    try:
        import psycopg
    except ImportError as exc:  # pragma: no cover - exercised only without optional dependency
        raise RuntimeError(
            "psycopg is required for Supabase persistence; install with "
            "python -m pip install -e '.[dev]'"
        ) from exc
    connection = psycopg.connect(url)
    connection.execute("set time zone 'UTC'")
    return connection

