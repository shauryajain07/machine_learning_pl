"""Apply versioned SQL migrations to Supabase."""

from __future__ import annotations

from pathlib import Path


MIGRATIONS_DIR = Path(__file__).resolve().parents[2] / "migrations"


def migration_files() -> list[Path]:
    return sorted(MIGRATIONS_DIR.glob("*.sql"))


def apply_migrations(connection) -> list[str]:
    applied: list[str] = []
    with connection.transaction():
        connection.execute(
            """
            create table if not exists schema_migrations (
                version text primary key,
                applied_at timestamptz not null default now()
            )
            """
        )
        for migration in migration_files():
            version = migration.stem
            row = connection.execute(
                "select 1 from schema_migrations where version = %s", (version,)
            ).fetchone()
            if row:
                continue
            connection.execute(migration.read_text(encoding="utf-8"))
            connection.execute(
                "insert into schema_migrations (version) values (%s)", (version,)
            )
            applied.append(version)
    return applied

