"""Apply the project's versioned migrations to Supabase."""

from __future__ import annotations

import argparse
import sys

from src.db.connection import connect
from src.db.schema import apply_migrations


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--db-url", "--db", dest="db_url", help="Supabase PostgreSQL connection string")
    args = parser.parse_args()
    connection = connect(args.db_url)
    try:
        applied = apply_migrations(connection)
        print("Applied migrations: " + (", ".join(applied) if applied else "none"))
    finally:
        connection.close()
    return 0


if __name__ == "__main__":
    try:
        raise SystemExit(main())
    except Exception as exc:
        print(f"ERROR: {exc}", file=sys.stderr)
        raise SystemExit(1)
