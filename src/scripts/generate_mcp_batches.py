"""Generate bounded, idempotent SQL batches for Supabase MCP imports.

The normal importer uses a PostgreSQL connection string.  This companion
utility keeps the same parser and normalization rules but emits SQL files that
can be submitted through the authenticated Supabase MCP ``execute_sql`` tool
when a direct database URL is not available.
"""

from __future__ import annotations

import argparse
import json
import re
from collections import OrderedDict
from datetime import date, datetime, time
from pathlib import Path
from typing import Iterable

from src.ingestion.football_txt.file_parser import parse_file
from src.ingestion.normalization.aliases import AliasMaps, load_alias_maps
from src.ingestion.normalization.names import clean_display_name, normalize_name
from src.ingestion.source import stable_match_fingerprint
from src.ingestion.validation.season_validator import SeasonReport, validate_season
from src.models import ParsedFile, ParsedMatch
from src.scripts.ingest_openfootball import discover_source_files


DEFAULT_REPOSITORY = "https://github.com/openfootball/england"
DEFAULT_CHUNK_BYTES = 120_000


def _sql_string(value: str) -> str:
    """Return a safe PostgreSQL E-string literal for source text."""

    escaped = (
        value.replace("\\", "\\\\")
        .replace("'", "\\'")
        .replace("\r", "\\r")
        .replace("\n", "\\n")
        .replace("\t", "\\t")
    )
    return f"E'{escaped}'"


def _sql_value(value: object) -> str:
    if value is None:
        return "NULL"
    if isinstance(value, bool):
        return "TRUE" if value else "FALSE"
    if isinstance(value, int):
        return str(value)
    if isinstance(value, date) and not isinstance(value, datetime):
        return f"{_sql_string(value.isoformat())}::date"
    if isinstance(value, time):
        return f"{_sql_string(value.isoformat())}::time"
    if isinstance(value, datetime):
        return f"{_sql_string(value.isoformat())}::timestamptz"
    if isinstance(value, (list, dict)):
        return f"{_sql_string(json.dumps(value, ensure_ascii=False, separators=(',', ':')))}::jsonb"
    return _sql_string(str(value))


def _team_canonical(alias_maps: AliasMaps, source_name: str) -> tuple[str, str]:
    canonical = clean_display_name(alias_maps.target("team", source_name))
    return canonical, normalize_name(canonical)


def _player_canonical(alias_maps: AliasMaps, source_name: str) -> tuple[str, str]:
    canonical = clean_display_name(alias_maps.target("player", source_name))
    return canonical, normalize_name(canonical)


def _referee_canonical(alias_maps: AliasMaps, source_name: str) -> tuple[str, str]:
    canonical = clean_display_name(alias_maps.target("referee", source_name))
    return canonical, normalize_name(canonical)


def _source_path(path: Path, root: Path) -> str:
    try:
        return path.resolve().relative_to(root.resolve()).as_posix()
    except ValueError:
        return path.name


def _parsed_source_file(parsed_file: ParsedFile) -> str:
    """Return the relative source path carried by parsed match records."""

    if parsed_file.matches:
        return parsed_file.matches[0].source_file
    return Path(parsed_file.path).name


def _statement_batch(
    output_dir: Path,
    prefix: str,
    statements: Iterable[str],
    *,
    chunk_bytes: int,
) -> list[dict[str, object]]:
    """Write statements without splitting any SQL statement across files."""

    batches: list[dict[str, object]] = []
    current: list[str] = []
    current_bytes = 0
    batch_number = 0

    def flush() -> None:
        nonlocal current, current_bytes, batch_number
        if not current:
            return
        name = f"{prefix}_{batch_number:03d}.sql"
        path = output_dir / name
        path.write_text("\n".join(current) + "\n", encoding="utf-8")
        batches.append({"name": name, "bytes": path.stat().st_size, "statements": len(current)})
        batch_number += 1
        current = []
        current_bytes = 0

    for raw_statement in statements:
        statement = raw_statement.rstrip() + ";"
        statement_bytes = len((statement + "\n").encode("utf-8"))
        if current and current_bytes + statement_bytes > chunk_bytes:
            flush()
        current.append(statement)
        current_bytes += statement_bytes
    flush()
    return batches


def _multi_insert_statements(
    table: str,
    columns: str,
    rows: list[list[str]],
    conflict_clause: str,
    *,
    max_bytes: int,
) -> list[str]:
    """Render multi-row inserts while keeping each statement bounded."""

    if not rows:
        return []
    prefix = f"insert into {table} ({columns}) values "
    suffix = f" {conflict_clause}" if conflict_clause else ""
    statements: list[str] = []
    current: list[str] = []

    def flush() -> None:
        if current:
            statements.append(prefix + ", ".join(current) + suffix)
            current.clear()

    for row in rows:
        rendered = "(" + ", ".join(row) + ")"
        candidate = prefix + ", ".join([*current, rendered]) + suffix
        if current and len(candidate.encode("utf-8")) > max_bytes:
            flush()
        current.append(rendered)
    flush()
    return statements


def _team_id_expr(normalized: str) -> str:
    return f"(select team_id from public.teams where normalized_name = {_sql_value(normalized)})"


def _player_id_expr(normalized: str) -> str:
    return f"(select player_id from public.players where normalized_name = {_sql_value(normalized)})"


def _referee_id_expr(normalized: str) -> str:
    return f"(select referee_id from public.referees where normalized_name = {_sql_value(normalized)})"


def _season_id_expr(parsed_file: ParsedFile) -> str:
    header = parsed_file.header
    return (
        "(select s.season_id from public.seasons s "
        "join public.competitions c on c.competition_id = s.competition_id "
        f"where c.name = {_sql_value(header.competition_name)} "
        f"and c.country = {_sql_value(header.country)} "
        f"and s.season_label = {_sql_value(header.season_label)})"
    )


def _source_file_id_expr(parsed_file: ParsedFile, repository: str) -> str:
    return (
        "(select source_file_id from public.source_files "
        f"where source_repository = {_sql_value(repository)} "
        f"and source_path = {_sql_value(_parsed_source_file(parsed_file))} "
        f"and file_sha256 = {_sql_value(parsed_file.file_sha256)})"
    )


def _match_id_expr(parsed_file: ParsedFile, match: ParsedMatch) -> str:
    return (
        "(select m.match_id from public.matches m "
        f"where m.season_id = {_season_id_expr(parsed_file)} "
        f"and m.source_file = {_sql_value(match.source_file)} "
        f"and m.source_match_index = {match.source_match_index})"
    )


def _collect_entities(
    parsed_files: list[ParsedFile],
    alias_maps: AliasMaps,
) -> tuple[
    OrderedDict[str, str],
    list[tuple[str, str, str, str]],
    OrderedDict[str, str],
    list[tuple[str, str, str, str]],
    OrderedDict[str, str],
]:
    teams: OrderedDict[str, str] = OrderedDict()
    team_aliases: list[tuple[str, str, str, str]] = []
    players: OrderedDict[str, str] = OrderedDict()
    player_aliases: list[tuple[str, str, str, str]] = []
    referees: OrderedDict[str, str] = OrderedDict()

    def add_team(source_name: str, source_file: str) -> None:
        source_display = clean_display_name(source_name)
        canonical, canonical_norm = _team_canonical(alias_maps, source_display)
        teams.setdefault(canonical_norm, canonical)
        source_norm = normalize_name(source_display)
        if source_display != canonical or source_norm != canonical_norm:
            alias_source = alias_maps.mapping_source("team", source_display) or source_file
            value = (canonical_norm, source_display, source_norm, alias_source)
            if value not in team_aliases:
                team_aliases.append(value)

    def add_player(source_name: str, source_file: str) -> None:
        source_display = clean_display_name(source_name)
        canonical, canonical_norm = _player_canonical(alias_maps, source_display)
        players.setdefault(canonical_norm, canonical)
        source_norm = normalize_name(source_display)
        if source_display != canonical or source_norm != canonical_norm:
            alias_source = alias_maps.mapping_source("player", source_display) or source_file
            value = (canonical_norm, source_display, source_norm, alias_source)
            if value not in player_aliases:
                player_aliases.append(value)

    def add_referee(source_name: str) -> None:
        canonical, canonical_norm = _referee_canonical(alias_maps, source_name)
        referees.setdefault(canonical_norm, canonical)

    for parsed_file in parsed_files:
        for match in parsed_file.matches:
            if match.home_source_name:
                add_team(match.home_source_name, match.source_file)
            if match.away_source_name:
                add_team(match.away_source_name, match.source_file)
            # Match persistence resolves identities in this same order:
            # appearances, substitutions, goals, then cards.
            for appearance in match.appearances:
                add_player(appearance.player_name, match.source_file)
            for substitution in match.substitutions:
                add_player(substitution.player_off_name, match.source_file)
                add_player(substitution.player_on_name, match.source_file)
            for goal in match.goals:
                add_player(goal.player_name, match.source_file)
            for card in match.cards:
                add_player(card.player_name, match.source_file)
            for referee in match.referees:
                add_referee(referee)
    return teams, team_aliases, players, player_aliases, referees


def _entity_statements(
    teams: OrderedDict[str, str],
    team_aliases: list[tuple[str, str, str, str]],
    players: OrderedDict[str, str],
    player_aliases: list[tuple[str, str, str, str]],
    referees: OrderedDict[str, str],
) -> list[str]:
    statements: list[str] = [
        "-- Entities are loaded before foreign-key-bearing match/event rows.",
    ]
    for normalized, canonical in teams.items():
        statements.append(
            "insert into public.teams (canonical_name, normalized_name) "
            f"values ({_sql_value(canonical)}, {_sql_value(normalized)}) "
            "on conflict (normalized_name) do nothing"
        )
    for canonical_norm, alias, normalized_alias, source in team_aliases:
        statements.append(
            "insert into public.team_aliases (team_id, alias, normalized_alias, source) "
            f"values ({_team_id_expr(canonical_norm)}, {_sql_value(alias)}, "
            f"{_sql_value(normalized_alias)}, {_sql_value(source)}) on conflict do nothing"
        )
    for normalized, canonical in players.items():
        statements.append(
            "insert into public.players (canonical_name, normalized_name) "
            f"values ({_sql_value(canonical)}, {_sql_value(normalized)}) "
            "on conflict (normalized_name) do nothing"
        )
    for canonical_norm, alias, normalized_alias, source in player_aliases:
        statements.append(
            "insert into public.player_aliases (player_id, alias, normalized_alias, source) "
            f"values ({_player_id_expr(canonical_norm)}, {_sql_value(alias)}, "
            f"{_sql_value(normalized_alias)}, {_sql_value(source)}) on conflict do nothing"
        )
    for normalized, canonical in referees.items():
        statements.append(
            "insert into public.referees (canonical_name, normalized_name) "
            f"values ({_sql_value(canonical)}, {_sql_value(normalized)}) "
            "on conflict (normalized_name) do nothing"
        )
    return statements


def _metadata_statements(
    parsed_file: ParsedFile,
    *,
    repository: str,
    source_root: str,
    run_id_expr: str,
) -> list[str]:
    header = parsed_file.header
    season_expr = _season_id_expr(parsed_file)
    statements = [
        f"-- Source file: {_parsed_source_file(parsed_file)}",
        "insert into public.competitions (name, country) "
        f"values ({_sql_value(header.competition_name)}, {_sql_value(header.country)}) "
        "on conflict (name, country) do update set name = excluded.name",
        "insert into public.seasons "
        "(competition_id, season_label, start_date, end_date, expected_team_count, "
        "expected_match_count, source_header_text) "
        f"values ((select competition_id from public.competitions where name = {_sql_value(header.competition_name)} "
        f"and country = {_sql_value(header.country)}), {_sql_value(header.season_label)}, "
        f"{_sql_value(header.start_date)}, {_sql_value(header.end_date)}, "
        f"{_sql_value(header.expected_team_count)}, {_sql_value(header.expected_match_count)}, "
        f"{_sql_value(header.source_header_text)}) "
        "on conflict (competition_id, season_label) do update set "
        "start_date = excluded.start_date, end_date = excluded.end_date, "
        "expected_team_count = excluded.expected_team_count, "
        "expected_match_count = excluded.expected_match_count, "
        "source_header_text = excluded.source_header_text, updated_at = now()",
        "insert into public.source_files "
        "(ingestion_run_id, source_repository, source_root, season_directory, source_filename, "
        "source_path, source_commit_sha, file_sha256, file_format, file_size_bytes, header_text) "
        f"values ({run_id_expr}, {_sql_value(repository)}, {_sql_value(source_root)}, "
        f"{_sql_value(parsed_file.season_directory)}, {_sql_value(Path(parsed_file.path).name)}, "
        f"{_sql_value(_parsed_source_file(parsed_file))}, {_sql_value(parsed_file.source_commit_sha)}, "
        f"{_sql_value(parsed_file.file_sha256)}, {_sql_value(parsed_file.source_format)}, "
        f"{parsed_file.file_size_bytes}, {_sql_value(header.source_header_text)}) "
        "on conflict (source_repository, source_path, file_sha256) do update set "
        "ingestion_run_id = excluded.ingestion_run_id, source_root = excluded.source_root, "
        "source_commit_sha = excluded.source_commit_sha, file_format = excluded.file_format, "
        "file_size_bytes = excluded.file_size_bytes, header_text = excluded.header_text, "
        "ingested_at = now()",
    ]
    seen_rounds: set[str] = set()
    for match in parsed_file.matches:
        if match.round_info is None or match.round_info.raw_heading in seen_rounds:
            continue
        seen_rounds.add(match.round_info.raw_heading)
        info = match.round_info
        statements.append(
            "insert into public.rounds (season_id, round_name, round_number, raw_heading) "
            f"values ({season_expr}, {_sql_value(info.round_name)}, {_sql_value(info.round_number)}, "
            f"{_sql_value(info.raw_heading)}) on conflict (season_id, raw_heading) do update set "
            "round_name = excluded.round_name, round_number = excluded.round_number"
        )

    season_teams: OrderedDict[str, dict[str, object]] = OrderedDict()
    for match in parsed_file.matches:
        for source_name in (match.home_source_name, match.away_source_name):
            if not source_name:
                continue
            canonical, normalized = _team_canonical(load_alias_maps_for_statement, source_name)
            entry = season_teams.setdefault(normalized, {"first": source_name, "names": []})
            names = entry["names"]
            assert isinstance(names, list)
            if source_name not in names:
                names.append(source_name)
    for normalized, entry in season_teams.items():
        # The canonical lookup is intentionally by normalized name. The
        # source spelling list remains the authoritative season evidence.
        names = entry["names"]
        statements.append(
            "insert into public.season_teams (season_id, team_id, first_source_name, source_names) "
            f"values ({season_expr}, {_team_id_expr(normalized)}, {_sql_value(entry['first'])}, "
            f"{_sql_value(names)}::jsonb) on conflict (season_id, team_id) do update set "
            "first_source_name = coalesce(season_teams.first_source_name, excluded.first_source_name), "
            "source_names = excluded.source_names"
        )
    return statements


def _match_values(
    parsed_file: ParsedFile,
    match: ParsedMatch,
    *,
    repository: str,
) -> list[str]:
    season_expr = _season_id_expr(parsed_file)
    round_expr = "NULL"
    if match.round_info is not None:
        round_expr = (
            f"(select round_id from public.rounds where season_id = {season_expr} "
            f"and raw_heading = {_sql_value(match.round_info.raw_heading)})"
        )
    same_team = bool(
        match.home_source_name
        and match.away_source_name
        and normalize_name(match.home_source_name) == normalize_name(match.away_source_name)
    )
    home_team_expr = "NULL"
    away_team_expr = "NULL"
    if match.home_source_name and not same_team:
        _, normalized = _team_canonical(load_alias_maps_for_statement, match.home_source_name)
        home_team_expr = _team_id_expr(normalized)
    if match.away_source_name and not same_team:
        _, normalized = _team_canonical(load_alias_maps_for_statement, match.away_source_name)
        away_team_expr = _team_id_expr(normalized)
    fingerprint = stable_match_fingerprint(match)
    values = [
        season_expr,
        round_expr,
        _sql_value(match.matchday),
        _sql_value(match.match_date),
        _sql_value(match.source_weekday),
        _sql_value(match.day_of_week),
        _sql_value(match.kickoff_local_time),
        _sql_value(match.utc_offset_minutes),
        _sql_value(match.kickoff_utc),
        _sql_value(match.venue),
        _sql_value(match.location),
        _sql_value(match.attendance),
        home_team_expr,
        away_team_expr,
        _sql_value(match.home_score),
        _sql_value(match.away_score),
        _sql_value(match.home_ht_score),
        _sql_value(match.away_ht_score),
        _source_file_id_expr(parsed_file, repository),
        _sql_value(match.source_file),
        str(match.source_match_index),
        _sql_value(fingerprint),
        _sql_value(match.source_format),
        _sql_value(match.raw_match_text),
        _sql_value(match.raw_goal_block),
        _sql_value(match.raw_home_lineup_text),
        _sql_value(match.raw_away_lineup_text),
        _sql_value(match.parse_status),
    ]
    return values


MATCH_COLUMNS = (
    "season_id, round_id, matchday, match_date, source_weekday, day_of_week, "
    "kickoff_local_time, utc_offset_minutes, kickoff_utc, venue, location, attendance, "
    "home_team_id, away_team_id, home_score, away_score, home_ht_score, away_ht_score, "
    "source_file_id, source_file, source_match_index, source_fingerprint, source_format, "
    "raw_match_text, raw_goal_block, raw_home_lineup_text, raw_away_lineup_text, parse_status"
)
MATCH_UPDATES = (
    "round_id, matchday, match_date, source_weekday, day_of_week, kickoff_local_time, "
    "utc_offset_minutes, kickoff_utc, venue, location, attendance, home_team_id, "
    "away_team_id, home_score, away_score, home_ht_score, away_ht_score, source_file_id, "
    "source_file, source_match_index, source_fingerprint, source_format, raw_match_text, "
    "raw_goal_block, raw_home_lineup_text, raw_away_lineup_text, parse_status"
)
MATCH_CONFLICT = "on conflict (season_id, source_file, source_match_index) do update set " + ", ".join(
    f"{column} = excluded.{column}" for column in MATCH_UPDATES.split(", ")
) + ", updated_at = now()"


def _match_statements(
    parsed_file: ParsedFile,
    *,
    repository: str,
    max_bytes: int,
) -> list[str]:
    rows = [
        _match_values(parsed_file, match, repository=repository)
        for match in parsed_file.matches
    ]
    return _multi_insert_statements(
        "public.matches",
        MATCH_COLUMNS,
        rows,
        MATCH_CONFLICT,
        max_bytes=max_bytes,
    )


def _cleanup_statements(parsed_file: ParsedFile) -> list[str]:
    season_expr = _season_id_expr(parsed_file)
    match_ids = (
        f"select match_id from public.matches where season_id = {season_expr} "
        f"and source_file = {_sql_value(_parsed_source_file(parsed_file))}"
    )
    statements = []
    for table in (
        "match_team_names",
        "appearances",
        "substitutions",
        "goals",
        "cards",
        "match_referees",
        "parsing_issues",
    ):
        statements.append(f"delete from public.{table} where match_id in ({match_ids})")
    statements.append(
        "delete from public.parsing_issues where match_id is null "
        f"and source_file = {_sql_value(_parsed_source_file(parsed_file))} "
        "and source_match_index is null"
    )
    return statements


def _child_statements(parsed_file: ParsedFile, *, max_bytes: int) -> list[str]:
    match_team_rows: list[list[str]] = []
    appearance_rows: list[list[str]] = []
    substitution_rows: list[list[str]] = []
    goal_rows: list[list[str]] = []
    card_rows: list[list[str]] = []
    referee_rows: list[list[str]] = []
    issue_rows: list[list[str]] = []
    for match in parsed_file.matches:
        match_id = _match_id_expr(parsed_file, match)
        side_team_expr: dict[str, str] = {"home": "NULL", "away": "NULL"}
        for side, source_name in (("home", match.home_source_name), ("away", match.away_source_name)):
            if source_name:
                _, normalized = _team_canonical(load_alias_maps_for_statement, source_name)
                side_team_expr[side] = _team_id_expr(normalized)

        for side, source_name in (("home", match.home_source_name), ("away", match.away_source_name)):
            if not source_name:
                continue
            match_team_rows.append([
                match_id,
                side_team_expr[side],
                _sql_value(side),
                _sql_value(source_name),
            ])

        for appearance in match.appearances:
            _, player_norm = _player_canonical(load_alias_maps_for_statement, appearance.player_name)
            entered = appearance.entered
            exited = appearance.exited
            appearance_rows.append([
                match_id,
                side_team_expr[appearance.side],
                _player_id_expr(player_norm),
                _sql_value(appearance.started),
                _sql_value(appearance.captain),
                _sql_value(entered.minute if entered else None),
                _sql_value(entered.stoppage_time if entered else None),
                _sql_value(entered.absolute_minute if entered else None),
                _sql_value(exited.minute if exited else None),
                _sql_value(exited.stoppage_time if exited else None),
                _sql_value(exited.absolute_minute if exited else None),
                str(appearance.source_order),
                _sql_value(appearance.raw_group_index),
                _sql_value(appearance.raw_player_text),
            ])

        for substitution_order, substitution in enumerate(match.substitutions, start=1):
            _, off_norm = _player_canonical(load_alias_maps_for_statement, substitution.player_off_name)
            _, on_norm = _player_canonical(load_alias_maps_for_statement, substitution.player_on_name)
            mark = substitution.minute_mark
            substitution_rows.append([
                match_id,
                side_team_expr[substitution.side],
                _player_id_expr(off_norm),
                _player_id_expr(on_norm),
                _sql_value(mark.minute if mark else None),
                _sql_value(mark.stoppage_time if mark else None),
                _sql_value(mark.absolute_minute if mark else None),
                _sql_value(mark.raw if mark else None),
                str(substitution_order),
                _sql_value(substitution.raw_event_text),
            ])

        for goal in match.goals:
            _, player_norm = _player_canonical(load_alias_maps_for_statement, goal.player_name)
            mark = goal.minute_mark
            goal_rows.append([
                match_id,
                side_team_expr[goal.side],
                _player_id_expr(player_norm),
                _sql_value(mark.minute if mark else None),
                _sql_value(mark.stoppage_time if mark else None),
                _sql_value(mark.absolute_minute if mark else None),
                _sql_value(goal.penalty),
                _sql_value(goal.own_goal),
                str(goal.goal_order),
                _sql_value(mark.raw if mark else None),
                _sql_value(goal.annotation),
                _sql_value(goal.raw_event_text),
            ])

        card_orders: dict[str, int] = {}
        for card in match.cards:
            _, player_norm = _player_canonical(load_alias_maps_for_statement, card.player_name)
            mark = card.minute_mark
            card_orders[card.card_type] = card_orders.get(card.card_type, 0) + 1
            card_rows.append([
                match_id,
                side_team_expr[card.side],
                _player_id_expr(player_norm),
                _sql_value(card.card_type),
                _sql_value(mark.minute if mark else None),
                _sql_value(mark.stoppage_time if mark else None),
                _sql_value(mark.absolute_minute if mark else None),
                _sql_value(mark.raw if mark else None),
                _sql_value(card.raw_event_text),
                str(card_orders[card.card_type]),
            ])

        for referee_order, referee in enumerate(match.referees, start=1):
            _, referee_norm = _referee_canonical(load_alias_maps_for_statement, referee)
            referee_rows.append([
                match_id,
                _referee_id_expr(referee_norm),
                str(referee_order),
                "NULL",
            ])

        for issue in match.issues:
            issue_rows.append([
                f"(select max(ingestion_run_id) from public.ingestion_runs where options->>'run_token' = {run_token_sql_literal})",
                match_id,
                _sql_value(match.source_file),
                str(match.source_match_index),
                _sql_value(issue.severity),
                _sql_value(issue.category),
                _sql_value(issue.raw_fragment),
                _sql_value(issue.description),
            ])

    for issue in current_file_issues:
        issue_rows.append([
            f"(select max(ingestion_run_id) from public.ingestion_runs where options->>'run_token' = {run_token_sql_literal})",
            "NULL",
            _sql_value(_parsed_source_file(parsed_file)),
            "NULL",
            _sql_value(issue.severity),
            _sql_value(issue.category),
            _sql_value(issue.raw_fragment),
            _sql_value(issue.description),
        ])

    statements: list[str] = []
    statements.extend(_multi_insert_statements(
        "public.match_team_names",
        "match_id, team_id, side, source_name",
        match_team_rows,
        "on conflict (match_id, side) do update set team_id = excluded.team_id, source_name = excluded.source_name",
        max_bytes=max_bytes,
    ))
    statements.extend(_multi_insert_statements(
        "public.appearances",
        "match_id, team_id, player_id, started, captain, entered_minute, entered_stoppage_time, "
        "entered_absolute_minute, exited_minute, exited_stoppage_time, exited_absolute_minute, "
        "source_order, raw_group_index, raw_player_text",
        appearance_rows,
        "on conflict do nothing",
        max_bytes=max_bytes,
    ))
    statements.extend(_multi_insert_statements(
        "public.substitutions",
        "match_id, team_id, player_off_id, player_on_id, minute, stoppage_time, absolute_minute, "
        "raw_minute, source_order, raw_event_text",
        substitution_rows,
        "on conflict do nothing",
        max_bytes=max_bytes,
    ))
    statements.extend(_multi_insert_statements(
        "public.goals",
        "match_id, scoring_team_id, player_id, minute, stoppage_time, absolute_minute, penalty, "
        "own_goal, goal_order, raw_minute, annotation, raw_event_text",
        goal_rows,
        "on conflict do nothing",
        max_bytes=max_bytes,
    ))
    statements.extend(_multi_insert_statements(
        "public.cards",
        "match_id, team_id, player_id, card_type, minute, stoppage_time, absolute_minute, raw_minute, "
        "raw_event_text, source_order",
        card_rows,
        "on conflict do nothing",
        max_bytes=max_bytes,
    ))
    statements.extend(_multi_insert_statements(
        "public.match_referees",
        "match_id, referee_id, referee_order, role",
        referee_rows,
        "on conflict do nothing",
        max_bytes=max_bytes,
    ))
    statements.extend(_multi_insert_statements(
        "public.parsing_issues",
        "ingestion_run_id, match_id, source_file, source_match_index, severity, category, raw_fragment, description",
        issue_rows,
        "",
        max_bytes=max_bytes,
    ))
    return statements


# These values are bound once by main(). Keeping them module-local avoids
# threading run metadata through every small SQL rendering helper.
load_alias_maps_for_statement = AliasMaps()
run_token_sql_literal = "NULL"
current_file_issues = []


def build_batches(
    *,
    root: Path,
    output_dir: Path,
    alias_maps: AliasMaps,
    source_commit_sha: str | None,
    repository: str,
    run_token: str,
    chunk_bytes: int,
) -> dict[str, object]:
    global load_alias_maps_for_statement, run_token_sql_literal, current_file_issues
    load_alias_maps_for_statement = alias_maps
    run_token_sql_literal = _sql_string(run_token)

    files = discover_source_files(root, all_seasons=True)
    source_root = str(root.resolve())
    parsed_files: list[ParsedFile] = []
    reports: list[SeasonReport] = []
    for source in files:
        relative = _source_path(source, root)
        parsed_file = parse_file(source, source_file=relative, source_commit_sha=source_commit_sha)
        report = validate_season(parsed_file)
        parsed_files.append(parsed_file)
        reports.append(report)

    output_dir.mkdir(parents=True, exist_ok=True)
    batches: list[dict[str, object]] = []
    teams, team_aliases, players, player_aliases, referees = _collect_entities(parsed_files, alias_maps)
    batches.extend(
        {"phase": "entities", **batch}
        for batch in _statement_batch(
            output_dir,
            "00_entities",
            _entity_statements(teams, team_aliases, players, player_aliases, referees),
            chunk_bytes=chunk_bytes,
        )
    )

    for file_index, (parsed_file, report) in enumerate(zip(parsed_files, reports), start=1):
        current_file_issues = report.file_issues
        base = f"{file_index:02d}_{parsed_file.season_directory}_{parsed_file.source_format}"
        batches.extend(
            {"phase": "metadata", "source_file": _parsed_source_file(parsed_file), **batch}
            for batch in _statement_batch(
                output_dir,
                f"{base}_metadata",
                _metadata_statements(
                    parsed_file,
                    repository=repository,
                    source_root=source_root,
                    run_id_expr=(
                        f"(select max(ingestion_run_id) from public.ingestion_runs "
                        f"where options->>'run_token' = {run_token_sql_literal})"
                    ),
                ),
                chunk_bytes=chunk_bytes,
            )
        )
        batches.extend(
            {"phase": "matches", "source_file": _parsed_source_file(parsed_file), **batch}
            for batch in _statement_batch(
                output_dir,
                f"{base}_matches",
                _match_statements(
                    parsed_file,
                    repository=repository,
                    max_bytes=chunk_bytes,
                ),
                chunk_bytes=chunk_bytes,
            )
        )
        batches.extend(
            {"phase": "cleanup", "source_file": _parsed_source_file(parsed_file), **batch}
            for batch in _statement_batch(
                output_dir,
                f"{base}_cleanup",
                _cleanup_statements(parsed_file),
                chunk_bytes=chunk_bytes,
            )
        )
        batches.extend(
            {"phase": "children", "source_file": _parsed_source_file(parsed_file), **batch}
            for batch in _statement_batch(
                output_dir,
                f"{base}_children",
                _child_statements(parsed_file, max_bytes=chunk_bytes),
                chunk_bytes=chunk_bytes,
            )
        )

    manifest = {
        "source_repository": repository,
        "source_root": source_root,
        "source_commit_sha": source_commit_sha,
        "run_token": run_token,
        "files": [
            {
                "source_file": _parsed_source_file(parsed_file),
                "season_label": parsed_file.header.season_label,
                "source_format": parsed_file.source_format,
                "file_sha256": parsed_file.file_sha256,
                "file_size_bytes": parsed_file.file_size_bytes,
                "report": report.to_dict(),
            }
            for parsed_file, report in zip(parsed_files, reports)
        ],
        "totals": {
            "seasons": len(reports),
            "matches": sum(report.parsed_matches for report in reports),
            "goals": sum(report.goals for report in reports),
            "appearances": sum(report.appearances for report in reports),
            "substitutions": sum(report.substitutions for report in reports),
            "yellow_cards": sum(report.yellow_cards for report in reports),
            "red_cards": sum(report.red_cards for report in reports),
            "yellow_red_cards": sum(report.yellow_red_cards for report in reports),
            "clean_matches": sum(report.clean_matches for report in reports),
            "warning_matches": sum(report.warning_matches for report in reports),
            "failed_matches": sum(report.failed_matches for report in reports),
        },
        "batches": batches,
    }
    (output_dir / "manifest.json").write_text(
        json.dumps(manifest, ensure_ascii=False, indent=2, default=str) + "\n",
        encoding="utf-8",
    )
    return manifest


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("root", type=Path, help="Pinned OpenFootball England clone")
    parser.add_argument("output_dir", type=Path, help="Directory for generated SQL batches")
    parser.add_argument("--run-token", required=True)
    parser.add_argument("--source-commit-sha", required=True)
    parser.add_argument("--source-repository", default=DEFAULT_REPOSITORY)
    parser.add_argument(
        "--team-alias-file",
        default=str(Path(__file__).resolve().parents[2] / "config" / "team_aliases.json"),
    )
    parser.add_argument("--chunk-bytes", type=int, default=DEFAULT_CHUNK_BYTES)
    args = parser.parse_args()
    alias_maps = load_alias_maps(Path(args.team_alias_file).resolve())
    manifest = build_batches(
        root=args.root,
        output_dir=args.output_dir,
        alias_maps=alias_maps,
        source_commit_sha=args.source_commit_sha,
        repository=args.source_repository,
        run_token=args.run_token,
        chunk_bytes=args.chunk_bytes,
    )
    print(json.dumps({"output_dir": str(args.output_dir), "batches": len(manifest["batches"]), "totals": manifest["totals"]}))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
