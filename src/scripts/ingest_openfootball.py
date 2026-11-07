"""Ingest local OpenFootball England files into Supabase."""

from __future__ import annotations

import argparse
import json
import re
import sys
from pathlib import Path
from typing import Iterable

from src.db.connection import connect
from src.db.repositories import PostgresRepository
from src.ingestion.football_txt.file_parser import parse_file
from src.ingestion.normalization.aliases import load_alias_maps
from src.ingestion.source import repository_commit_sha
from src.ingestion.validation.season_validator import SeasonReport, validate_season
from src.models import ParseIssue


FULL_FILENAME = "1-premierleague-full.txt"
COMPACT_FILENAME = "1-premierleague.txt"


def discover_source_files(
    root: str | Path,
    *,
    season: str | None = None,
    all_seasons: bool = False,
    full_only: bool = False,
) -> list[Path]:
    root_path = Path(root)
    if root_path.is_file():
        if root_path.name not in {FULL_FILENAME, COMPACT_FILENAME}:
            raise ValueError(f"Expected a Premier League source file, got {root_path.name!r}")
        return [root_path]
    if not root_path.is_dir():
        raise FileNotFoundError(f"Source directory not found: {root_path}")
    if not all_seasons and not season:
        raise ValueError("Choose --season SEASON or --all-seasons")
    if re.fullmatch(r"\d{4}-\d{2}", root_path.name):
        season_dirs = [root_path]
    else:
        season_dirs = sorted(
            path
            for path in root_path.iterdir()
            if path.is_dir() and re.fullmatch(r"\d{4}-\d{2}", path.name)
        )
    if season:
        season_dirs = [path for path in season_dirs if path.name == season]
        if not season_dirs:
            raise FileNotFoundError(f"Season directory not found: {season}")
    files: list[Path] = []
    for season_dir in season_dirs:
        full_path = season_dir / FULL_FILENAME
        compact_path = season_dir / COMPACT_FILENAME
        if full_path.exists():
            files.append(full_path)
        elif compact_path.exists() and not full_only:
            files.append(compact_path)
        elif full_only:
            continue
    if not files:
        qualifier = "full " if full_only else ""
        raise FileNotFoundError(f"No {qualifier}Premier League source files found under {root_path}")
    return files


def _source_path(path: Path, root: Path) -> str:
    try:
        return path.resolve().relative_to(root.resolve()).as_posix()
    except ValueError:
        return path.name


def _print_report(report: SeasonReport, *, verbose: bool = False) -> None:
    print(
        f"{report.season_label}: {report.parsed_matches} matches | "
        f"clean={report.clean_matches} warning={report.warning_matches} failed={report.failed_matches} | "
        f"teams={report.unique_teams} goals={report.goals} "
        f"appearances={report.appearances} substitutions={report.substitutions}"
    )
    if report.file_issues:
        for issue in report.file_issues:
            print(f"  {issue.severity.upper()} {issue.category}: {issue.description}")
    if verbose and report.issue_counts:
        print("  issues: " + ", ".join(f"{key}={value}" for key, value in sorted(report.issue_counts.items())))


def _aggregate_reports(reports: Iterable[SeasonReport]) -> dict[str, int]:
    totals = {
        "seasons": 0,
        "matches": 0,
        "goals": 0,
        "appearances": 0,
        "substitutions": 0,
        "yellow_cards": 0,
        "red_cards": 0,
        "yellow_red_cards": 0,
        "clean_matches": 0,
        "warning_matches": 0,
        "failed_matches": 0,
    }
    for report in reports:
        totals["seasons"] += 1
        totals["matches"] += report.parsed_matches
        totals["goals"] += report.goals
        totals["appearances"] += report.appearances
        totals["substitutions"] += report.substitutions
        totals["yellow_cards"] += report.yellow_cards
        totals["red_cards"] += report.red_cards
        totals["yellow_red_cards"] += report.yellow_red_cards
        totals["clean_matches"] += report.clean_matches
        totals["warning_matches"] += report.warning_matches
        totals["failed_matches"] += report.failed_matches
    return totals


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("repo_or_directory", help="Local clone, season directory, or Premier League file")
    selection = parser.add_mutually_exclusive_group(required=True)
    selection.add_argument("--season", help="Only ingest one season directory, e.g. 2025-26")
    selection.add_argument("--all-seasons", action="store_true", help="Discover all season directories")
    parser.add_argument("--full-only", action="store_true", help="Ignore compact files when no full file exists")
    parser.add_argument("--dry-run", action="store_true", help="Parse and validate without writing Supabase")
    parser.add_argument("--verbose", action="store_true")
    parser.add_argument("--db-url", "--db", dest="db_url", help="Supabase PostgreSQL connection string")
    parser.add_argument(
        "--team-alias-file",
        default=str(Path(__file__).resolve().parents[2] / "config" / "team_aliases.json"),
        help="Explicit JSON team/player/referee alias mapping file",
    )
    parser.add_argument("--source-repository", default="https://github.com/openfootball/england")
    return parser


def run(args: argparse.Namespace) -> int:
    root = Path(args.repo_or_directory)
    files = discover_source_files(
        root,
        season=args.season,
        all_seasons=args.all_seasons,
        full_only=args.full_only,
    )
    root_for_paths = root if root.is_dir() else root.parent
    commit_sha = repository_commit_sha(root_for_paths)
    alias_maps = load_alias_maps(args.team_alias_file)
    parsed_files = []
    reports: list[SeasonReport] = []

    print(f"Source: {args.source_repository}")
    print(f"Commit: {commit_sha or 'unavailable'}")
    print(f"Files: {len(files)}")
    for source in files:
        relative_source = _source_path(source, root_for_paths)
        parsed_file = parse_file(
            source,
            source_file=relative_source,
            source_commit_sha=commit_sha,
        )
        report = validate_season(parsed_file)
        parsed_files.append((parsed_file, report))
        reports.append(report)
        print(f"\n{relative_source} ({parsed_file.source_format})")
        _print_report(report, verbose=args.verbose)

    totals = _aggregate_reports(reports)
    print("\nTotals: " + json.dumps(totals, sort_keys=True))
    if args.dry_run:
        print("Dry run: no database changes made.")
        return 0

    connection = connect(args.db_url)
    repository = PostgresRepository(connection, alias_maps=alias_maps)
    run_id = None
    try:
        run_id = repository.start_ingestion_run(
            source_repository=args.source_repository,
            source_root=str(root_for_paths.resolve()),
            source_commit_sha=commit_sha,
            options={
                "season": args.season,
                "all_seasons": args.all_seasons,
                "full_only": args.full_only,
                "team_alias_file": args.team_alias_file,
            },
        )
        connection.commit()
        competition_ids: dict[tuple[str, str], int] = {}
        for parsed_file, report in parsed_files:
            key = (parsed_file.header.competition_name, parsed_file.header.country)
            competition_id = competition_ids.setdefault(
                key,
                repository.ensure_competition(parsed_file.header.competition_name, parsed_file.header.country),
            )
            season_id = repository.ensure_season(competition_id, parsed_file)
            source_file_id = repository.upsert_source_file(
                parsed_file,
                run_id=run_id,
                source_repository=args.source_repository,
                source_root=str(root_for_paths.resolve()),
                source_path=_source_path(Path(parsed_file.path), root_for_paths),
            )
            # Source metadata and the season row are atomic for this file.
            connection.commit()
            for match in parsed_file.matches:
                try:
                    # Psycopg uses a savepoint for this nested transaction. A
                    # malformed source record therefore cannot roll back the
                    # rest of the season.
                    with connection.transaction():
                        repository.persist_match(
                            season_id=season_id,
                            source_file_id=source_file_id,
                            match=match,
                            ingestion_run_id=run_id,
                        )
                except Exception as exc:  # pragma: no cover - DB-specific path
                    issue = f"Database persistence failed for source match {match.source_match_index}: {exc}"
                    print(f"ERROR: {issue}", file=sys.stderr)
                    repository.record_match_issues(
                        match_id=None,
                        run_id=run_id,
                        source_file=match.source_file,
                        source_match_index=match.source_match_index,
                        issues=[
                            ParseIssue(
                                severity="error",
                                category="validation",
                                raw_fragment=match.raw_match_text,
                                description=issue,
                            )
                        ],
                    )
            repository.record_match_issues(
                match_id=None,
                run_id=run_id,
                source_file=(parsed_file.matches[0].source_file if parsed_file.matches else parsed_file.path),
                source_match_index=None,
                issues=report.file_issues,
            )
            connection.commit()
        repository.finish_ingestion_run(run_id, status="completed", counts=totals)
        connection.commit()
    except Exception as exc:
        if run_id is not None:
            connection.rollback()
            repository.finish_ingestion_run(
                run_id,
                status="failed",
                counts=totals,
                error_text=str(exc),
            )
            connection.commit()
        connection.close()
        raise
    connection.close()
    print(f"Supabase ingestion completed (run {run_id}).")
    return 0


def main() -> int:
    args = build_parser().parse_args()
    try:
        return run(args)
    except Exception as exc:
        print(f"ERROR: {exc}", file=sys.stderr)
        return 1


if __name__ == "__main__":
    raise SystemExit(main())
