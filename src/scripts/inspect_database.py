"""Print Supabase row counts, quality checks, and season coverage."""

from __future__ import annotations

import argparse
import sys

from src.db.connection import connect


TABLES = (
    ("Seasons", "seasons"),
    ("Matches", "matches"),
    ("Teams", "teams"),
    ("Players", "players"),
    ("Appearances", "appearances"),
    ("Goals", "goals"),
    ("Substitutions", "substitutions"),
    ("Yellow cards", "cards where card_type = 'yellow'"),
    ("Red cards", "cards where card_type = 'red'"),
    ("Yellow-red cards", "cards where card_type = 'yellow_red'"),
    ("Referees", "referees"),
)


def _count(connection, relation: str) -> int:
    return int(connection.execute(f"select count(*) from {relation}").fetchone()[0])


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--db-url", "--db", dest="db_url", help="Supabase PostgreSQL connection string")
    args = parser.parse_args()
    connection = connect(args.db_url)
    try:
        print("DATABASE SUMMARY")
        for label, relation in TABLES:
            print(f"{label}: {_count(connection, relation)}")

        print("\nQUALITY")
        quality_queries = (
            ("Clean matches", "select count(*) from matches where parse_status = 'clean'"),
            ("Warnings", "select count(*) from matches where parse_status = 'warning'"),
            ("Failed", "select count(*) from matches where parse_status = 'failed'"),
            (
                "Goal-count mismatches",
                "select count(*) from parsing_issues where category = 'validation' and description like 'Parsed goal totals%'",
            ),
            (
                "Lineup-count mismatches",
                "select count(*) from parsing_issues where category = 'lineup' and description like 'Expected 11%'",
            ),
            (
                "Unresolved team aliases",
                "select count(*) from identity_review_candidates where entity_type = 'team' and status = 'pending'",
            ),
            (
                "Unresolved player aliases",
                "select count(*) from identity_review_candidates where entity_type = 'player' and status = 'pending'",
            ),
            (
                "Duplicate candidates",
                "select count(*) from identity_review_candidates where status = 'pending'",
            ),
            ("Parsing issues", "select count(*) from parsing_issues"),
        )
        for label, query in quality_queries:
            print(f"{label}: {connection.execute(query).fetchone()[0]}")

        print("\nCOVERAGE BY SEASON")
        columns = (
            "season_label",
            "full_file_available",
            "matches",
            "lineups",
            "substitutions",
            "goals",
            "yellow_cards",
            "red_cards",
            "yellow_red_cards",
            "referees",
            "attendance",
        )
        print(" | ".join(columns))
        rows = connection.execute(
            """
            select season_label, full_file_available, matches, lineups,
                   substitutions, goals, yellow_cards, red_cards, yellow_red_cards,
                   referees, attendance
            from season_coverage
            order by season_label
            """
        ).fetchall()
        for row in rows:
            print(" | ".join(str(value) for value in row))
    finally:
        connection.close()
    return 0


if __name__ == "__main__":
    try:
        raise SystemExit(main())
    except Exception as exc:
        print(f"ERROR: {exc}", file=sys.stderr)
        raise SystemExit(1)
