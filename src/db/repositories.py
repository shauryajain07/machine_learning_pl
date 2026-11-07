"""PostgreSQL repositories for normalized OpenFootball records."""

from __future__ import annotations

import json
from collections import defaultdict
from datetime import datetime, timezone
from typing import Any

try:  # Importing the parser remains possible in environments without psycopg.
    from psycopg.types.json import Jsonb
except ImportError:  # pragma: no cover - only used before installing dependencies
    Jsonb = None  # type: ignore[assignment,misc]

from src.ingestion.normalization.aliases import AliasMaps
from src.ingestion.normalization.names import clean_display_name, normalize_name
from src.ingestion.source import stable_match_fingerprint
from src.models import ParsedFile, ParsedMatch, ParseIssue, MinuteMark


def _json(value: Any):
    return Jsonb(value) if Jsonb is not None else value


def _minute_values(mark: MinuteMark | None) -> tuple[int | None, int | None, int | None, str | None]:
    if mark is None:
        return None, None, None, None
    return mark.minute, mark.stoppage_time, mark.absolute_minute, mark.raw


class PostgresRepository:
    """Small query-oriented repository with deterministic upserts.

    A repository instance is intended to handle one ingestion run. Entity IDs
    are cached for that run, while database uniqueness constraints remain the
    final idempotency guard.
    """

    def __init__(self, connection, *, alias_maps: AliasMaps | None = None):
        self.connection = connection
        self.alias_maps = alias_maps or AliasMaps()
        self.team_cache: dict[str, int] = {}
        self.player_cache: dict[str, int] = {}
        self.referee_cache: dict[str, int] = {}
        self.season_cache: dict[tuple[int, str], int] = {}
        self.round_cache: dict[tuple[int, str], int] = {}

    def start_ingestion_run(
        self,
        *,
        source_repository: str,
        source_root: str,
        source_commit_sha: str | None,
        options: dict[str, Any],
    ) -> int:
        row = self.connection.execute(
            """
            insert into ingestion_runs
                (source_repository, source_root, source_commit_sha, options)
            values (%s, %s, %s, %s)
            returning ingestion_run_id
            """,
            (source_repository, source_root, source_commit_sha, _json(options)),
        ).fetchone()
        assert row is not None
        return int(row[0])

    def finish_ingestion_run(
        self,
        run_id: int,
        *,
        status: str,
        counts: dict[str, Any],
        error_text: str | None = None,
    ) -> None:
        self.connection.execute(
            """
            update ingestion_runs
            set completed_at = %s, status = %s, counts = %s, error_text = %s
            where ingestion_run_id = %s
            """,
            (datetime.now(timezone.utc), status, _json(counts), error_text, run_id),
        )

    def upsert_source_file(
        self,
        parsed_file: ParsedFile,
        *,
        run_id: int,
        source_repository: str,
        source_root: str,
        source_path: str,
    ) -> int:
        row = self.connection.execute(
            """
            insert into source_files
                (ingestion_run_id, source_repository, source_root,
                 season_directory, source_filename, source_path,
                 source_commit_sha, file_sha256, file_format,
                 file_size_bytes, header_text)
            values (%s, %s, %s, %s, %s, %s, %s, %s, %s, %s, %s)
            on conflict (source_repository, source_path, file_sha256)
            do update set
                ingestion_run_id = excluded.ingestion_run_id,
                source_root = excluded.source_root,
                source_commit_sha = excluded.source_commit_sha,
                file_format = excluded.file_format,
                file_size_bytes = excluded.file_size_bytes,
                header_text = excluded.header_text,
                ingested_at = now()
            returning source_file_id
            """,
            (
                run_id,
                source_repository,
                source_root,
                parsed_file.season_directory,
                parsed_file.path.split("/")[-1],
                source_path,
                parsed_file.source_commit_sha,
                parsed_file.file_sha256,
                parsed_file.source_format,
                parsed_file.file_size_bytes,
                parsed_file.header.source_header_text,
            ),
        ).fetchone()
        assert row is not None
        return int(row[0])

    def ensure_competition(self, name: str, country: str) -> int:
        row = self.connection.execute(
            """
            insert into competitions (name, country)
            values (%s, %s)
            on conflict (name, country) do update set name = excluded.name
            returning competition_id
            """,
            (name, country),
        ).fetchone()
        assert row is not None
        return int(row[0])

    def ensure_season(self, competition_id: int, parsed_file: ParsedFile) -> int:
        key = (competition_id, parsed_file.header.season_label)
        if key in self.season_cache:
            return self.season_cache[key]
        header = parsed_file.header
        row = self.connection.execute(
            """
            insert into seasons
                (competition_id, season_label, start_date, end_date,
                 expected_team_count, expected_match_count, source_header_text)
            values (%s, %s, %s, %s, %s, %s, %s)
            on conflict (competition_id, season_label)
            do update set
                start_date = excluded.start_date,
                end_date = excluded.end_date,
                expected_team_count = excluded.expected_team_count,
                expected_match_count = excluded.expected_match_count,
                source_header_text = excluded.source_header_text,
                updated_at = now()
            returning season_id
            """,
            (
                competition_id,
                header.season_label,
                header.start_date,
                header.end_date,
                header.expected_team_count,
                header.expected_match_count,
                header.source_header_text,
            ),
        ).fetchone()
        assert row is not None
        season_id = int(row[0])
        self.season_cache[key] = season_id
        return season_id

    def ensure_round(self, season_id: int, parsed_match: ParsedMatch) -> int | None:
        if parsed_match.round_info is None:
            return None
        info = parsed_match.round_info
        key = (season_id, info.raw_heading)
        if key in self.round_cache:
            return self.round_cache[key]
        row = self.connection.execute(
            """
            insert into rounds (season_id, round_name, round_number, raw_heading)
            values (%s, %s, %s, %s)
            on conflict (season_id, raw_heading)
            do update set round_name = excluded.round_name,
                          round_number = excluded.round_number
            returning round_id
            """,
            (season_id, info.round_name, info.round_number, info.raw_heading),
        ).fetchone()
        assert row is not None
        round_id = int(row[0])
        self.round_cache[key] = round_id
        return round_id

    def _record_identity_candidate(
        self,
        *,
        entity_type: str,
        source_value: str,
        normalized_value: str,
        candidate_entity_id: int | None,
        source_file: str,
        evidence: str,
    ) -> None:
        self.connection.execute(
            """
            insert into identity_review_candidates
                (entity_type, source_value, normalized_value,
                 candidate_entity_id, source_file, evidence)
            values (%s, %s, %s, %s, %s, %s)
            on conflict do nothing
            """,
            (entity_type, source_value, normalized_value, candidate_entity_id, source_file, evidence),
        )

    def _possible_team_alias_candidate(self, normalized_value: str) -> tuple[str, int] | None:
        candidates = []
        if normalized_value.endswith(" fc"):
            candidates.append(normalized_value[:-3].rstrip())
        if normalized_value.endswith(" afc"):
            candidates.append(normalized_value[:-4].rstrip())
        if normalized_value.startswith("afc "):
            candidates.append(normalized_value[4:].lstrip())
        if not normalized_value.endswith((" fc", " afc")):
            candidates.extend((normalized_value + " fc", normalized_value + " afc"))
        for candidate in candidates:
            row = self.connection.execute(
                "select team_id from teams where normalized_name = %s",
                (candidate,),
            ).fetchone()
            if row:
                return candidate, int(row[0])
        return None

    def ensure_team(self, source_name: str, *, source_file: str) -> int:
        source_display = clean_display_name(source_name)
        source_norm = normalize_name(source_display)
        explicit_target = self.alias_maps.target("team", source_display)
        target_norm = normalize_name(explicit_target)
        cache_key = target_norm
        if cache_key in self.team_cache:
            team_id = self.team_cache[cache_key]
            self._insert_team_alias(team_id, source_display, source_file, source_norm, explicit_target)
            return team_id

        if self.alias_maps.mapping_source("team", source_display) is None:
            alias_rows = self.connection.execute(
                """
                select team_id from team_aliases
                where normalized_alias = %s
                """,
                (source_norm,),
            ).fetchall()
            distinct_alias_teams = {int(row[0]) for row in alias_rows}
            if len(distinct_alias_teams) == 1:
                team_id = next(iter(distinct_alias_teams))
                self.team_cache[cache_key] = team_id
                self._insert_team_alias(team_id, source_display, source_file, source_norm, explicit_target)
                return team_id
            if len(distinct_alias_teams) > 1:
                for candidate_id in distinct_alias_teams:
                    self._record_identity_candidate(
                        entity_type="team",
                        source_value=source_display,
                        normalized_value=source_norm,
                        candidate_entity_id=candidate_id,
                        source_file=source_file,
                        evidence="Source alias is already mapped to multiple teams; no merge was made.",
                    )

        row = self.connection.execute(
            "select team_id, canonical_name from teams where normalized_name = %s",
            (target_norm,),
        ).fetchone()
        if row:
            team_id = int(row[0])
            canonical_name = str(row[1])
        else:
            candidate = None
            if self.alias_maps.mapping_source("team", source_display) is None:
                candidate = self._possible_team_alias_candidate(source_norm)
            row = self.connection.execute(
                """
                insert into teams (canonical_name, normalized_name)
                values (%s, %s)
                on conflict (normalized_name) do update set normalized_name = excluded.normalized_name
                returning team_id, canonical_name
                """,
                (explicit_target, target_norm),
            ).fetchone()
            assert row is not None
            team_id = int(row[0])
            canonical_name = str(row[1])
            if candidate:
                self._record_identity_candidate(
                    entity_type="team",
                    source_value=source_display,
                    normalized_value=source_norm,
                    candidate_entity_id=candidate[1],
                    source_file=source_file,
                    evidence=(
                        f"Deterministic suffix variant resembles existing team normalized as {candidate[0]!r}; "
                        "the records were kept separate pending explicit review."
                    ),
                )
        self.team_cache[cache_key] = team_id
        self._insert_team_alias(team_id, source_display, source_file, source_norm, canonical_name)
        return team_id

    def _insert_team_alias(
        self,
        team_id: int,
        source_display: str,
        source_file: str,
        source_norm: str,
        canonical_name: str,
    ) -> None:
        if source_display == canonical_name and source_norm == normalize_name(canonical_name):
            return
        alias_source = self.alias_maps.mapping_source("team", source_display) or source_file
        self.connection.execute(
            """
            insert into team_aliases (team_id, alias, normalized_alias, source)
            values (%s, %s, %s, %s)
            on conflict do nothing
            """,
            (team_id, source_display, source_norm, alias_source),
        )

    def ensure_player(self, source_name: str, *, source_file: str) -> int:
        source_display = clean_display_name(source_name)
        source_norm = normalize_name(source_display)
        canonical = self.alias_maps.target("player", source_display)
        target_norm = normalize_name(canonical)
        if target_norm in self.player_cache:
            player_id = self.player_cache[target_norm]
            self._insert_player_alias(player_id, source_display, source_file, source_norm, canonical)
            return player_id
        row = self.connection.execute(
            "select player_id, canonical_name from players where normalized_name = %s",
            (target_norm,),
        ).fetchone()
        if row:
            player_id = int(row[0])
            canonical_name = str(row[1])
        else:
            row = self.connection.execute(
                """
                insert into players (canonical_name, normalized_name)
                values (%s, %s)
                on conflict (normalized_name) do update set normalized_name = excluded.normalized_name
                returning player_id, canonical_name
                """,
                (canonical, target_norm),
            ).fetchone()
            assert row is not None
            player_id = int(row[0])
            canonical_name = str(row[1])
        self.player_cache[target_norm] = player_id
        self._insert_player_alias(player_id, source_display, source_file, source_norm, canonical_name)
        return player_id

    def _insert_player_alias(
        self,
        player_id: int,
        source_display: str,
        source_file: str,
        source_norm: str,
        canonical_name: str,
    ) -> None:
        if source_display == canonical_name and source_norm == normalize_name(canonical_name):
            return
        alias_source = self.alias_maps.mapping_source("player", source_display) or source_file
        self.connection.execute(
            """
            insert into player_aliases (player_id, alias, normalized_alias, source)
            values (%s, %s, %s, %s)
            on conflict do nothing
            """,
            (player_id, source_display, source_norm, alias_source),
        )

    def ensure_referee(self, source_name: str, *, source_file: str) -> int:
        source_display = clean_display_name(source_name)
        key = normalize_name(source_display)
        if key in self.referee_cache:
            return self.referee_cache[key]
        canonical = self.alias_maps.target("referee", source_display)
        canonical_key = normalize_name(canonical)
        row = self.connection.execute(
            "select referee_id from referees where normalized_name = %s",
            (canonical_key,),
        ).fetchone()
        if row:
            referee_id = int(row[0])
        else:
            row = self.connection.execute(
                """
                insert into referees (canonical_name, normalized_name)
                values (%s, %s)
                on conflict (normalized_name) do update set normalized_name = excluded.normalized_name
                returning referee_id
                """,
                (canonical, canonical_key),
            ).fetchone()
            assert row is not None
            referee_id = int(row[0])
        self.referee_cache[key] = referee_id
        return referee_id

    def record_season_team(self, season_id: int, team_id: int, source_name: str) -> None:
        row = self.connection.execute(
            "select first_source_name, source_names from season_teams where season_id = %s and team_id = %s",
            (season_id, team_id),
        ).fetchone()
        if row is None:
            self.connection.execute(
                """
                insert into season_teams (season_id, team_id, first_source_name, source_names)
                values (%s, %s, %s, %s)
                """,
                (season_id, team_id, source_name, _json([source_name])),
            )
            return
        source_names = row[1] if isinstance(row[1], list) else json.loads(row[1])
        if source_name not in source_names:
            source_names.append(source_name)
            self.connection.execute(
                """
                update season_teams set source_names = %s
                where season_id = %s and team_id = %s
                """,
                (_json(source_names), season_id, team_id),
            )

    def _upsert_match_row(
        self,
        *,
        season_id: int,
        round_id: int | None,
        source_file_id: int,
        match: ParsedMatch,
        fingerprint: str,
        home_team_id: int | None,
        away_team_id: int | None,
    ) -> int:
        values = (
            season_id,
            round_id,
            match.matchday,
            match.match_date,
            match.source_weekday,
            match.day_of_week,
            match.kickoff_local_time,
            match.utc_offset_minutes,
            match.kickoff_utc,
            match.venue,
            match.location,
            match.attendance,
            home_team_id,
            away_team_id,
            match.home_score,
            match.away_score,
            match.home_ht_score,
            match.away_ht_score,
            source_file_id,
            match.source_file,
            match.source_match_index,
            fingerprint,
            match.source_format,
            match.raw_match_text,
            match.raw_goal_block,
            match.raw_home_lineup_text,
            match.raw_away_lineup_text,
            match.parse_status,
        )
        existing = self.connection.execute(
            """
            select match_id from matches
            where season_id = %s and source_file = %s and source_match_index = %s
            """,
            (season_id, match.source_file, match.source_match_index),
        ).fetchone()
        if existing is None:
            existing = self.connection.execute(
                """
                select match_id from matches
                where season_id = %s and source_fingerprint = %s
                """,
                (season_id, fingerprint),
            ).fetchone()
        if existing is None:
            row = self.connection.execute(
                """
                insert into matches (
                    season_id, round_id, matchday, match_date, source_weekday,
                    day_of_week, kickoff_local_time, utc_offset_minutes,
                    kickoff_utc, venue, location, attendance, home_team_id,
                    away_team_id, home_score, away_score, home_ht_score,
                    away_ht_score, source_file_id, source_file,
                    source_match_index, source_fingerprint, source_format,
                    raw_match_text, raw_goal_block, raw_home_lineup_text,
                    raw_away_lineup_text, parse_status
                ) values (
                    %s, %s, %s, %s, %s, %s, %s, %s, %s, %s, %s, %s, %s,
                    %s, %s, %s, %s, %s, %s, %s, %s, %s, %s, %s, %s, %s,
                    %s, %s
                ) returning match_id
                """,
                values,
            ).fetchone()
            assert row is not None
            return int(row[0])
        match_id = int(existing[0])
        self.connection.execute(
            """
            update matches set
                round_id = %s, matchday = %s, match_date = %s,
                source_weekday = %s, day_of_week = %s,
                kickoff_local_time = %s, utc_offset_minutes = %s,
                kickoff_utc = %s, venue = %s, location = %s, attendance = %s,
                home_team_id = %s, away_team_id = %s, home_score = %s,
                away_score = %s, home_ht_score = %s, away_ht_score = %s,
                source_file_id = %s, source_file = %s,
                source_match_index = %s, source_fingerprint = %s,
                source_format = %s, raw_match_text = %s,
                raw_goal_block = %s, raw_home_lineup_text = %s,
                raw_away_lineup_text = %s, parse_status = %s, updated_at = now()
            where match_id = %s
            """,
            values[1:] + (match_id,),
        )
        return match_id

    def persist_match(
        self,
        *,
        season_id: int,
        source_file_id: int,
        match: ParsedMatch,
        ingestion_run_id: int,
    ) -> int:
        same_team = bool(
            match.home_source_name
            and match.away_source_name
            and normalize_name(match.home_source_name) == normalize_name(match.away_source_name)
        )
        # Keep a failed raw record persistable even when validation found an
        # invalid home/away pairing; the database check intentionally rejects
        # storing the same team in both FK columns.
        home_team_id = (
            self.ensure_team(match.home_source_name, source_file=match.source_file)
            if match.home_source_name and not same_team
            else None
        )
        away_team_id = (
            self.ensure_team(match.away_source_name, source_file=match.source_file)
            if match.away_source_name and not same_team
            else None
        )
        round_id = self.ensure_round(season_id, match)
        fingerprint = stable_match_fingerprint(match)
        match_id = self._upsert_match_row(
            season_id=season_id,
            round_id=round_id,
            source_file_id=source_file_id,
            match=match,
            fingerprint=fingerprint,
            home_team_id=home_team_id,
            away_team_id=away_team_id,
        )

        for table in (
            "match_team_names",
            "appearances",
            "substitutions",
            "goals",
            "cards",
            "match_referees",
            "parsing_issues",
        ):
            self.connection.execute(f"delete from {table} where match_id = %s", (match_id,))

        side_team_ids = {"home": home_team_id, "away": away_team_id}
        side_team_names = {"home": match.home_source_name, "away": match.away_source_name}
        for side in ("home", "away"):
            source_name = side_team_names[side]
            if source_name:
                team_id = side_team_ids[side]
                self.connection.execute(
                    """
                    insert into match_team_names (match_id, team_id, side, source_name)
                    values (%s, %s, %s, %s)
                    """,
                    (match_id, team_id, side, source_name),
                )
                if team_id is not None:
                    self.record_season_team(season_id, team_id, source_name)

        for appearance in match.appearances:
            team_id = side_team_ids[appearance.side]
            if team_id is None:
                match.issues.append(
                    ParseIssue(
                        severity="warning",
                        category="team_identity",
                        raw_fragment=appearance.raw_player_text,
                        description="Skipped appearance because its team identity was unresolved.",
                    )
                )
                continue
            player_id = self.ensure_player(appearance.player_name, source_file=match.source_file)
            entered = _minute_values(appearance.entered)
            exited = _minute_values(appearance.exited)
            self.connection.execute(
                """
                insert into appearances (
                    match_id, team_id, player_id, started, captain,
                    entered_minute, entered_stoppage_time, entered_absolute_minute,
                    exited_minute, exited_stoppage_time, exited_absolute_minute,
                    source_order, raw_group_index, raw_player_text
                ) values (%s, %s, %s, %s, %s, %s, %s, %s, %s, %s, %s, %s, %s, %s)
                """,
                (
                    match_id,
                    team_id,
                    player_id,
                    appearance.started,
                    appearance.captain,
                    entered[0],
                    entered[1],
                    entered[2],
                    exited[0],
                    exited[1],
                    exited[2],
                    appearance.source_order,
                    appearance.raw_group_index,
                    appearance.raw_player_text,
                ),
            )

        for match_substitution_order, substitution in enumerate(match.substitutions, start=1):
            team_id = side_team_ids[substitution.side]
            player_off_id = self.ensure_player(substitution.player_off_name, source_file=match.source_file)
            player_on_id = self.ensure_player(substitution.player_on_name, source_file=match.source_file)
            minute = _minute_values(substitution.minute_mark)
            self.connection.execute(
                """
                insert into substitutions (
                    match_id, team_id, player_off_id, player_on_id, minute,
                    stoppage_time, absolute_minute, raw_minute, source_order,
                    raw_event_text
                ) values (%s, %s, %s, %s, %s, %s, %s, %s, %s, %s)
                """,
                (
                    match_id,
                    team_id,
                    player_off_id,
                    player_on_id,
                    minute[0],
                    minute[1],
                    minute[2],
                    minute[3],
                    match_substitution_order,
                    substitution.raw_event_text,
                ),
            )

        for goal in match.goals:
            team_id = side_team_ids[goal.side]
            if team_id is None:
                match.issues.append(
                    ParseIssue(
                        severity="warning",
                        category="goal_attribution",
                        raw_fragment=goal.raw_event_text,
                        description="Skipped goal because its scoring team was unresolved.",
                    )
                )
                continue
            player_id = self.ensure_player(goal.player_name, source_file=match.source_file)
            minute = _minute_values(goal.minute_mark)
            self.connection.execute(
                """
                insert into goals (
                    match_id, scoring_team_id, player_id, minute, stoppage_time,
                    absolute_minute, penalty, own_goal, goal_order, raw_minute,
                    annotation, raw_event_text
                ) values (%s, %s, %s, %s, %s, %s, %s, %s, %s, %s, %s, %s)
                """,
                (
                    match_id,
                    team_id,
                    player_id,
                    minute[0],
                    minute[1],
                    minute[2],
                    goal.penalty,
                    goal.own_goal,
                    goal.goal_order,
                    minute[3],
                    goal.annotation,
                    goal.raw_event_text,
                ),
            )

        card_orders: defaultdict[str, int] = defaultdict(int)
        for card in match.cards:
            team_id = side_team_ids[card.side]
            player_id = self.ensure_player(card.player_name, source_file=match.source_file)
            minute = _minute_values(card.minute_mark)
            card_orders[card.card_type] += 1
            self.connection.execute(
                """
                insert into cards (
                    match_id, team_id, player_id, card_type, minute,
                    stoppage_time, absolute_minute, raw_minute, raw_event_text,
                    source_order
                ) values (%s, %s, %s, %s, %s, %s, %s, %s, %s, %s)
                """,
                (
                    match_id,
                    team_id,
                    player_id,
                    card.card_type,
                    minute[0],
                    minute[1],
                    minute[2],
                    minute[3],
                    card.raw_event_text,
                    card_orders[card.card_type],
                ),
            )

        for referee_order, referee in enumerate(match.referees, start=1):
            referee_id = self.ensure_referee(referee, source_file=match.source_file)
            self.connection.execute(
                """
                insert into match_referees (match_id, referee_id, referee_order, role)
                values (%s, %s, %s, %s)
                """,
                (match_id, referee_id, referee_order, None),
            )

        self.record_match_issues(
            match_id=match_id,
            run_id=ingestion_run_id,
            source_file=match.source_file,
            source_match_index=match.source_match_index,
            issues=match.issues,
        )
        return match_id

    def record_match_issues(
        self,
        *,
        match_id: int | None,
        run_id: int,
        source_file: str,
        source_match_index: int | None,
        issues: list[ParseIssue],
    ) -> None:
        for issue in issues:
            self.connection.execute(
                """
                insert into parsing_issues (
                    ingestion_run_id, match_id, source_file,
                    source_match_index, severity, category,
                    raw_fragment, description
                ) values (%s, %s, %s, %s, %s, %s, %s, %s)
                """,
                (
                    run_id,
                    match_id,
                    source_file,
                    source_match_index,
                    issue.severity,
                    issue.category,
                    issue.raw_fragment,
                    issue.description,
                ),
            )
