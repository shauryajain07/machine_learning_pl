"""Reproducible, leakage-safe match-table and baseline evaluation pipeline.

The module deliberately separates three concerns that are easy to conflate in
football experiments:

* source facts are canonicalised and validated before a target is created;
* pre-match features are generated in chronological order and only then split;
* every fitted preprocessing object belongs to the training window it serves.

The public entry points are usable both from the OpenFootball checkout and
from a read-only query of the normalized Supabase ``matches`` table.  No
database writes are performed here.
"""

from __future__ import annotations

import csv
import hashlib
import json
import math
import random
import re
from collections import Counter, defaultdict
from dataclasses import dataclass, replace
from datetime import date, datetime, time, timezone
from pathlib import Path
from typing import Any, Iterable, Mapping, Sequence

import numpy as np
from sklearn.compose import ColumnTransformer
from sklearn.impute import SimpleImputer
from sklearn.linear_model import LogisticRegression
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import OneHotEncoder, StandardScaler

from src.ingestion.football_txt.file_parser import parse_file
from src.ingestion.normalization.aliases import AliasMaps
from src.ingestion.normalization.names import clean_display_name, normalize_name
from src.ingestion.source import repository_commit_sha
from src.ingestion.validation.season_validator import validate_season
from src.ml.football_elo import EloConfig, EloMatch, match_sort_key, season_start_year
from src.ml.football_features import FEATURE_NAMES, PreMatchFeatureRow, build_pre_match_features
from src.models import ParsedMatch
from src.scripts.ingest_openfootball import discover_source_files


OUTCOME_LABELS = ("H", "D", "A")
OUTCOME_NAMES = ("home_win", "draw", "away_win")
PRE_MATCH_FEATURE_NAMES = ("home_team", "away_team", *FEATURE_NAMES)

# A score outside this range is not a plausible completed Premier League
# result.  This is a defensive source-integrity guard, not a modelling rule.
MAX_GOALS_PER_TEAM = 20

_SEASON_LABEL_RE = re.compile(r"^(?P<start>\d{4})[-/](?P<end>\d{2,4})$")


@dataclass(frozen=True)
class QualityIssue:
    """A row or season-level data-quality rejection."""

    code: str
    message: str
    season_label: str | None = None
    source_file: str | None = None
    source_match_index: int | None = None

    def as_dict(self) -> dict[str, object]:
        return {
            "code": self.code,
            "message": self.message,
            "season_label": self.season_label,
            "source_file": self.source_file,
            "source_match_index": self.source_match_index,
        }


class DataQualityError(ValueError):
    """Raised when a selected season contains invalid or duplicate fixtures."""

    def __init__(self, issues: Sequence[QualityIssue], audit: Mapping[str, object]):
        self.issues = tuple(issues)
        self.audit = dict(audit)
        summary = ", ".join(
            f"{code}={count}"
            for code, count in sorted(Counter(issue.code for issue in self.issues).items())
        )
        super().__init__(f"canonical match-table validation failed ({summary or 'unknown issue'})")


@dataclass(frozen=True)
class MatchInput:
    """Source-independent row used by the canonical validator."""

    season_label: object
    match_date: object
    kickoff_utc: object
    kickoff_local_time: object
    home_team: object
    away_team: object
    home_goals: object
    away_goals: object
    source_file: str
    source_match_index: int
    source_format: str = ""
    parse_status: str = "clean"
    validation_errors: tuple[str, ...] = ()
    validation_warnings: tuple[str, ...] = ()
    source_id: str | None = None
    home_ht_goals: object = None
    away_ht_goals: object = None
    matchday: object = None


@dataclass(frozen=True)
class CanonicalMatch:
    """One completed, validated match with a target derived from final score."""

    season_label: str
    match_date: date
    kickoff_utc: datetime | None
    kickoff_local_time: time | None
    home_team: str
    away_team: str
    home_team_key: str
    away_team_key: str
    home_goals: int
    away_goals: int
    source_file: str
    source_match_index: int
    source_format: str
    parse_status: str
    source_id: str | None = None
    matchday: int | None = None

    @property
    def outcome(self) -> str:
        """Derive the target only from the final score."""

        if self.home_goals > self.away_goals:
            return "H"
        if self.home_goals < self.away_goals:
            return "A"
        return "D"

    @property
    def outcome_index(self) -> int:
        return OUTCOME_LABELS.index(self.outcome)

    @property
    def fixture_key(self) -> tuple[str, date, str, str]:
        """Stable duplicate-fixture identity; score is intentionally absent."""

        return self.season_label, self.match_date, self.home_team_key, self.away_team_key

    @property
    def row_key(self) -> str:
        return "|".join(
            (
                self.season_label,
                self.match_date.isoformat(),
                self.kickoff_utc.isoformat() if self.kickoff_utc else "",
                self.home_team_key,
                self.away_team_key,
                self.source_file,
                str(self.source_match_index),
            )
        )

    def to_elo_match(self) -> EloMatch:
        return EloMatch(
            season_label=self.season_label,
            source_match_index=self.source_match_index,
            match_date=self.match_date,
            kickoff_utc=self.kickoff_utc,
            kickoff_local_time=self.kickoff_local_time,
            home_team=self.home_team,
            away_team=self.away_team,
            home_goals=self.home_goals,
            away_goals=self.away_goals,
            source_file=self.source_file,
            source_format=self.source_format,
            parse_status=self.parse_status,
            matchday=self.matchday,
        )

    def as_dict(self) -> dict[str, object]:
        return {
            "row_key": self.row_key,
            "season_label": self.season_label,
            "match_date": self.match_date.isoformat(),
            "kickoff_utc": self.kickoff_utc.isoformat() if self.kickoff_utc else "",
            "kickoff_local_time": self.kickoff_local_time.isoformat() if self.kickoff_local_time else "",
            "home_team": self.home_team,
            "away_team": self.away_team,
            "home_team_key": self.home_team_key,
            "away_team_key": self.away_team_key,
            "home_goals": self.home_goals,
            "away_goals": self.away_goals,
            "outcome": self.outcome,
            "matchday": self.matchday,
            "source_file": self.source_file,
            "source_match_index": self.source_match_index,
            "source_format": self.source_format,
            "parse_status": self.parse_status,
        }


@dataclass(frozen=True)
class FeatureRow:
    """Pre-match feature values attached to one canonical match."""

    match: CanonicalMatch
    values: tuple[float, ...]

    def __post_init__(self) -> None:
        if len(self.values) != len(FEATURE_NAMES):
            raise ValueError(f"expected {len(FEATURE_NAMES)} numeric features")

    @property
    def raw_values(self) -> tuple[object, ...]:
        return (self.match.home_team, self.match.away_team, *self.values)

    def as_dict(self, split: str) -> dict[str, object]:
        return {
            "row_key": self.match.row_key,
            "split": split,
            "season_label": self.match.season_label,
            "match_date": self.match.match_date.isoformat(),
            "kickoff_utc": self.match.kickoff_utc.isoformat() if self.match.kickoff_utc else "",
            "home_team": self.match.home_team,
            "away_team": self.match.away_team,
            "matchday": self.match.matchday,
            **{name: value for name, value in zip(FEATURE_NAMES, self.values)},
            "home_goals": self.match.home_goals,
            "away_goals": self.match.away_goals,
            "outcome": self.match.outcome,
        }


@dataclass(frozen=True)
class DatasetSnapshot:
    matches: tuple[CanonicalMatch, ...]
    audit: dict[str, object]
    source_kind: str
    source_commit_sha: str | None = None


def canonical_season_label(value: object) -> str:
    """Return a stable ``YYYY-YY`` label and reject ambiguous values."""

    if not isinstance(value, str):
        raise ValueError(f"season label must be text, got {value!r}")
    match = _SEASON_LABEL_RE.fullmatch(value.strip())
    if match is None:
        raise ValueError(f"invalid season label: {value!r}")
    start = int(match.group("start"))
    end_raw = match.group("end")
    end = int(end_raw) if len(end_raw) == 4 else (start // 100) * 100 + int(end_raw)
    if end < start:
        end += 100
    if end != start + 1:
        raise ValueError(f"season label must span one football season: {value!r}")
    return f"{start:04d}-{end % 100:02d}"


def _normalise_date(value: object) -> date:
    if isinstance(value, datetime):
        return value.date()
    if isinstance(value, date):
        return value
    if isinstance(value, str):
        return date.fromisoformat(value)
    raise ValueError(f"match date is missing or invalid: {value!r}")


def _normalise_kickoff_utc(value: object) -> datetime | None:
    if value in (None, ""):
        return None
    if isinstance(value, str):
        value = datetime.fromisoformat(value.replace("Z", "+00:00"))
    if not isinstance(value, datetime):
        raise ValueError(f"kickoff_utc is invalid: {value!r}")
    if value.tzinfo is None:
        return value.replace(tzinfo=timezone.utc)
    return value.astimezone(timezone.utc)


def _normalise_local_time(value: object) -> time | None:
    if value in (None, ""):
        return None
    if isinstance(value, str):
        return time.fromisoformat(value)
    if not isinstance(value, time):
        raise ValueError(f"kickoff_local_time is invalid: {value!r}")
    return value.replace(tzinfo=None)


def _normalise_score(value: object, field: str) -> int:
    if isinstance(value, bool):
        raise ValueError(f"{field} must be an integer")
    if isinstance(value, str) and value.strip().isdigit():
        value = int(value.strip())
    if not isinstance(value, (int, np.integer)):
        raise ValueError(f"{field} must be an integer")
    score = int(value)
    if score < 0:
        raise ValueError(f"{field} cannot be negative")
    if score > MAX_GOALS_PER_TEAM:
        raise ValueError(f"{field} exceeds the maximum plausible score ({MAX_GOALS_PER_TEAM})")
    return score


def _canonical_team(alias_maps: AliasMaps, value: object, field: str) -> tuple[str, str]:
    if not isinstance(value, str) or not value.strip():
        raise ValueError(f"{field} is missing")
    display = clean_display_name(alias_maps.target("team", value))
    key = normalize_name(display)
    if not key:
        raise ValueError(f"{field} is empty after normalization")
    return display, key


def _input_from_parsed(match: ParsedMatch) -> MatchInput:
    return MatchInput(
        season_label=match.season_label,
        match_date=match.match_date,
        kickoff_utc=match.kickoff_utc,
        kickoff_local_time=match.kickoff_local_time,
        home_team=match.home_source_name,
        away_team=match.away_source_name,
        home_goals=match.home_score,
        away_goals=match.away_score,
        home_ht_goals=match.home_ht_score,
        away_ht_goals=match.away_ht_score,
        matchday=match.matchday,
        source_file=match.source_file,
        source_match_index=match.source_match_index,
        source_format=match.source_format,
        parse_status=match.parse_status,
        validation_errors=tuple(issue.description for issue in match.issues if issue.severity == "error"),
        validation_warnings=tuple(issue.description for issue in match.issues if issue.severity == "warning"),
    )


def _canonicalize_input(
    row: MatchInput,
    alias_maps: AliasMaps,
) -> tuple[CanonicalMatch | None, list[QualityIssue]]:
    issues: list[QualityIssue] = []

    def reject(code: str, message: str) -> None:
        issues.append(
            QualityIssue(
                code=code,
                message=message,
                season_label=str(row.season_label) if row.season_label is not None else None,
                source_file=row.source_file,
                source_match_index=row.source_match_index,
            )
        )

    try:
        season_label = canonical_season_label(row.season_label)
    except ValueError as exc:
        reject("invalid_season", str(exc))
        season_label = ""

    try:
        match_date = _normalise_date(row.match_date)
    except ValueError as exc:
        reject("missing_or_invalid_date", str(exc))
        match_date = date.min

    if season_label:
        start_year = season_start_year(season_label)
        if not start_year <= match_date.year <= start_year + 1:
            reject("date_outside_season", f"{match_date.isoformat()} is outside {season_label}")

    matchday: int | None = None
    if row.matchday not in (None, ""):
        if isinstance(row.matchday, bool) or not isinstance(row.matchday, (int, np.integer)):
            reject("invalid_matchday", f"matchday must be an integer between 1 and 38: {row.matchday!r}")
        else:
            matchday = int(row.matchday)
            if not 1 <= matchday <= 38:
                reject("invalid_matchday", f"matchday must be between 1 and 38: {matchday}")

    try:
        home_team, home_key = _canonical_team(alias_maps, row.home_team, "home_team")
    except ValueError as exc:
        reject("missing_or_invalid_home_team", str(exc))
        home_team, home_key = "", ""
    try:
        away_team, away_key = _canonical_team(alias_maps, row.away_team, "away_team")
    except ValueError as exc:
        reject("missing_or_invalid_away_team", str(exc))
        away_team, away_key = "", ""
    if home_key and away_key and home_key == away_key:
        reject("same_home_away_team", "home and away teams resolve to the same canonical team")

    try:
        home_goals = _normalise_score(row.home_goals, "home_goals")
    except ValueError as exc:
        reject("missing_or_impossible_score", str(exc))
        home_goals = 0
    try:
        away_goals = _normalise_score(row.away_goals, "away_goals")
    except ValueError as exc:
        reject("missing_or_impossible_score", str(exc))
        away_goals = 0

    for halftime_field, halftime_value, final_value in (
        ("home_ht_goals", row.home_ht_goals, home_goals),
        ("away_ht_goals", row.away_ht_goals, away_goals),
    ):
        if halftime_value in (None, ""):
            continue
        try:
            halftime_score = _normalise_score(halftime_value, halftime_field)
        except ValueError as exc:
            reject("missing_or_impossible_score", str(exc))
            continue
        if halftime_score > final_value:
            reject(
                "impossible_halftime_score",
                f"{halftime_field}={halftime_score} exceeds final score={final_value}",
            )

    for error in row.validation_errors:
        reject("source_validation_error", error)
    if row.parse_status == "failed":
        reject("failed_parse_status", "source parser marked the match as failed")

    try:
        kickoff_utc = _normalise_kickoff_utc(row.kickoff_utc)
        kickoff_local_time = _normalise_local_time(row.kickoff_local_time)
    except ValueError as exc:
        reject("invalid_kickoff", str(exc))
        kickoff_utc = None
        kickoff_local_time = None

    if issues:
        return None, issues
    return (
        CanonicalMatch(
            season_label=season_label,
            match_date=match_date,
            kickoff_utc=kickoff_utc,
            kickoff_local_time=kickoff_local_time,
            home_team=home_team,
            away_team=away_team,
            home_team_key=home_key,
            away_team_key=away_key,
            home_goals=home_goals,
            away_goals=away_goals,
            source_file=row.source_file,
            source_match_index=row.source_match_index,
            source_format=row.source_format,
            parse_status=row.parse_status,
            source_id=row.source_id,
            matchday=matchday,
        ),
        [],
    )


def _empty_season_audit(expected: int | None, rows: Sequence[MatchInput]) -> dict[str, object]:
    return {
        "expected_matches": expected,
        "input_rows": len(rows),
        "accepted_rows": 0,
        "rejected_rows": 0,
        "status": "pending",
        "warnings": sum(len(row.validation_warnings) for row in rows),
    }


def canonicalize_rows(
    rows: Iterable[MatchInput],
    *,
    alias_maps: AliasMaps | None = None,
    expected_matches_by_season: Mapping[str, int | None] | None = None,
    strict: bool = True,
) -> tuple[tuple[CanonicalMatch, ...], dict[str, object]]:
    """Validate source rows and return a duplicate-free canonical match table.

    A season with fewer source rows than its advertised complete-season count
    is excluded as ``incomplete``.  Rows in selected seasons that have a
    missing result, invalid identity/date, impossible score, or duplicate
    fixture cause ``DataQualityError`` in strict mode; no invalid row is
    silently turned into a training example.
    """

    alias_maps = alias_maps or AliasMaps()
    materialized = tuple(rows)
    expected = {
        canonical_season_label(key): value
        for key, value in (expected_matches_by_season or {}).items()
    }
    grouped: dict[str, list[MatchInput]] = defaultdict(list)
    for row in materialized:
        try:
            key = canonical_season_label(row.season_label)
        except ValueError:
            key = str(row.season_label)
        grouped[key].append(row)

    issues: list[QualityIssue] = []
    season_audit: dict[str, dict[str, object]] = {}
    accepted: list[CanonicalMatch] = []
    complete_seasons: list[str] = []
    excluded_incomplete: list[str] = []

    for season in sorted(grouped, key=lambda value: season_start_year(value) if _SEASON_LABEL_RE.fullmatch(value) else 9999):
        season_rows = grouped[season]
        season_expected = expected.get(season)
        audit = _empty_season_audit(season_expected, season_rows)
        season_audit[season] = audit

        canonical_rows: list[CanonicalMatch] = []
        row_issues: list[QualityIssue] = []
        for row in season_rows:
            canonical, canonical_issues = _canonicalize_input(row, alias_maps)
            if canonical is None:
                row_issues.extend(canonical_issues)
            else:
                canonical_rows.append(canonical)

        if season_expected is not None and len(season_rows) < season_expected:
            audit["status"] = "excluded_incomplete"
            audit["accepted_rows"] = 0
            audit["rejected_rows"] = len(row_issues)
            excluded_incomplete.append(season)
            continue
        if season_expected is not None and len(season_rows) != season_expected:
            row_issues.append(
                QualityIssue(
                    code="season_row_count_mismatch",
                    message=f"expected {season_expected} source rows but found {len(season_rows)}",
                    season_label=season,
                )
            )

        issues.extend(row_issues)
        if row_issues:
            audit["status"] = "rejected_quality_errors"
            audit["rejected_rows"] = len(row_issues)
        else:
            audit["status"] = "complete"
            audit["accepted_rows"] = len(canonical_rows)
            complete_seasons.append(season)
            accepted.extend(canonical_rows)

    # Explicit aliases are authoritative.  For otherwise equivalent source
    # spellings (for example case-only changes), choose one deterministic
    # display label so the exported table does not contain multiple labels for
    # the same normalized team key.
    display_counts: dict[str, Counter[str]] = defaultdict(Counter)
    for match in accepted:
        display_counts[match.home_team_key][match.home_team] += 1
        display_counts[match.away_team_key][match.away_team] += 1

    def standard_display(team_key: str) -> str:
        values = display_counts[team_key]
        return min(values, key=lambda value: (-values[value], value.casefold(), value))

    accepted = [
        replace(
            match,
            home_team=standard_display(match.home_team_key),
            away_team=standard_display(match.away_team_key),
        )
        for match in accepted
    ]

    fixture_seen: dict[tuple[str, date, str, str], CanonicalMatch] = {}
    source_seen: dict[tuple[str, str, int], CanonicalMatch] = {}
    unique: list[CanonicalMatch] = []
    duplicate_issues: list[QualityIssue] = []
    for match in sorted(accepted, key=lambda value: match_sort_key(value.to_elo_match())):
        prior_fixture = fixture_seen.get(match.fixture_key)
        prior_source = source_seen.get((match.season_label, match.source_file, match.source_match_index))
        if prior_fixture is not None:
            duplicate_issues.append(
                QualityIssue(
                    code="duplicate_fixture",
                    message=f"fixture duplicates {prior_fixture.row_key}",
                    season_label=match.season_label,
                    source_file=match.source_file,
                    source_match_index=match.source_match_index,
                )
            )
            continue
        if prior_source is not None:
            duplicate_issues.append(
                QualityIssue(
                    code="duplicate_source_row",
                    message=f"source row duplicates {prior_source.row_key}",
                    season_label=match.season_label,
                    source_file=match.source_file,
                    source_match_index=match.source_match_index,
                )
            )
            continue
        fixture_seen[match.fixture_key] = match
        source_seen[(match.season_label, match.source_file, match.source_match_index)] = match
        unique.append(match)

    issues.extend(duplicate_issues)
    if duplicate_issues:
        for issue in duplicate_issues:
            season = issue.season_label or "unknown"
            if season in season_audit:
                season_audit[season]["status"] = "rejected_duplicate_fixture"
                season_audit[season]["rejected_rows"] = int(season_audit[season]["rejected_rows"]) + 1

    by_code = Counter(issue.code for issue in issues)
    audit: dict[str, object] = {
        "input_rows": len(materialized),
        "accepted_rows": len(unique) if not issues else len(unique),
        "rejected_rows": len(issues),
        "complete_seasons": sorted(complete_seasons, key=season_start_year),
        "excluded_incomplete_seasons": sorted(excluded_incomplete, key=season_start_year),
        "canonical_team_labels": {
            key: standard_display(key) for key in sorted(display_counts)
        },
        "issue_counts": dict(sorted(by_code.items())),
        "season_audit": season_audit,
    }
    if strict and issues:
        raise DataQualityError(issues, audit)
    return tuple(unique), audit


def _season_through(season_label: str, through: str | None) -> bool:
    return through is None or season_start_year(season_label) <= season_start_year(canonical_season_label(through))


def load_openfootball_snapshot(
    root: str | Path,
    *,
    alias_maps: AliasMaps | None = None,
    through: str | None = None,
    strict: bool = True,
) -> DatasetSnapshot:
    """Parse the pinned OpenFootball checkout and keep complete seasons only."""

    root_path = Path(root)
    files = discover_source_files(root_path, all_seasons=True)
    commit_sha = repository_commit_sha(root_path)
    rows: list[MatchInput] = []
    expected: dict[str, int | None] = {}
    file_audit: list[dict[str, object]] = []
    for source in files:
        season_from_path = source.parent.name
        if through is not None and not _season_through(season_from_path, through):
            continue
        relative = source.resolve().relative_to(root_path.resolve()).as_posix()
        parsed = parse_file(source, source_file=relative, source_commit_sha=commit_sha)
        report = validate_season(parsed)
        season = parsed.header.season_label
        expected[season] = parsed.header.expected_match_count
        rows.extend(_input_from_parsed(match) for match in parsed.matches)
        file_audit.append(
            {
                "season_label": season,
                "source_file": relative,
                "source_format": parsed.source_format,
                "file_sha256": parsed.file_sha256,
                "file_size_bytes": parsed.file_size_bytes,
                "expected_matches": parsed.header.expected_match_count,
                "parsed_matches": len(parsed.matches),
                "failed_matches": report.failed_matches,
                "warning_matches": report.warning_matches,
                "file_issues": [issue.description for issue in report.file_issues],
            }
        )

    matches, quality = canonicalize_rows(
        rows,
        alias_maps=alias_maps,
        expected_matches_by_season=expected,
        strict=strict,
    )
    if not matches:
        raise ValueError("no complete scored seasons were found in the OpenFootball source")
    audit = {
        "source_kind": "openfootball",
        "source_root": str(root_path.resolve()),
        "source_commit_sha": commit_sha,
        "through": through,
        "source_files": file_audit,
        "quality": quality,
    }
    return DatasetSnapshot(matches=matches, audit=audit, source_kind="openfootball", source_commit_sha=commit_sha)


def load_supabase_snapshot(
    connection: Any,
    *,
    alias_maps: AliasMaps | None = None,
    competition_name: str = "Premier League",
    through: str | None = None,
    strict: bool = True,
) -> DatasetSnapshot:
    """Read normalized match facts from Supabase without changing the database."""

    query = """
        select
            m.match_id::text,
            s.season_label,
            s.expected_match_count,
            m.match_date,
            m.kickoff_utc,
            m.kickoff_local_time,
            m.matchday,
            coalesce(home_team.canonical_name, home_name.source_name),
            coalesce(away_team.canonical_name, away_name.source_name),
            m.home_score,
            m.away_score,
            m.home_ht_score,
            m.away_ht_score,
            m.source_file,
            m.source_match_index,
            m.source_format,
            m.parse_status
        from public.matches as m
        join public.seasons as s on s.season_id = m.season_id
        join public.competitions as c on c.competition_id = s.competition_id
        left join public.teams as home_team on home_team.team_id = m.home_team_id
        left join public.teams as away_team on away_team.team_id = m.away_team_id
        left join public.match_team_names as home_name
            on home_name.match_id = m.match_id and home_name.side = 'home'
        left join public.match_team_names as away_name
            on away_name.match_id = m.match_id and away_name.side = 'away'
        where c.name = %s
        order by s.season_label, m.match_date, m.kickoff_utc, m.source_file, m.source_match_index
    """
    rows: list[MatchInput] = []
    expected: dict[str, int | None] = {}
    for values in connection.execute(query, (competition_name,)):
        (
            source_id,
            season_label,
            season_expected,
            match_date,
            kickoff_utc,
            kickoff_local_time,
            matchday,
            home_team,
            away_team,
            home_goals,
            away_goals,
            home_ht_goals,
            away_ht_goals,
            source_file,
            source_match_index,
            source_format,
            parse_status,
        ) = values
        season = canonical_season_label(season_label)
        if through is not None and not _season_through(season, through):
            continue
        expected[season] = season_expected
        rows.append(
            MatchInput(
                season_label=season,
                match_date=match_date,
                kickoff_utc=kickoff_utc,
                kickoff_local_time=kickoff_local_time,
                home_team=home_team,
                away_team=away_team,
                home_goals=home_goals,
                away_goals=away_goals,
                home_ht_goals=home_ht_goals,
                away_ht_goals=away_ht_goals,
                matchday=matchday,
                source_file=source_file or "",
                source_match_index=int(source_match_index),
                source_format=source_format or "",
                parse_status=parse_status or "",
                source_id=str(source_id) if source_id is not None else None,
            )
        )

    matches, quality = canonicalize_rows(
        rows,
        alias_maps=alias_maps,
        expected_matches_by_season=expected,
        strict=strict,
    )
    if not matches:
        raise ValueError("no complete scored seasons were found in Supabase")
    query_hash = hashlib.sha256(" ".join(query.split()).encode("utf-8")).hexdigest()
    audit = {
        "source_kind": "supabase_postgres",
        "competition_name": competition_name,
        "through": through,
        "query_sha256": query_hash,
        "quality": quality,
    }
    return DatasetSnapshot(matches=matches, audit=audit, source_kind="supabase_postgres")


def load_supabase_export_snapshot(
    path: str | Path,
    *,
    alias_maps: AliasMaps | None = None,
    competition_name: str = "Premier League",
    through: str | None = None,
    strict: bool = True,
) -> DatasetSnapshot:
    """Load a read-only JSON export produced from the Supabase MCP query.

    This is an auditable bridge for environments where the MCP OAuth session
    can query the database but the local terminal does not have a PostgreSQL
    password. The export contains source facts only; canonical validation and
    all chronological feature construction still happen locally.
    """

    export_path = Path(path)
    payload = json.loads(export_path.read_text(encoding="utf-8"))
    if not isinstance(payload, dict) or not isinstance(payload.get("rows"), list):
        raise ValueError("Supabase export must be an object with a rows list")
    rows: list[MatchInput] = []
    expected: dict[str, int | None] = {}
    for value in payload["rows"]:
        if not isinstance(value, dict):
            raise ValueError("Supabase export rows must be JSON objects")
        season = canonical_season_label(value.get("season_label"))
        if through is not None and not _season_through(season, through):
            continue
        expected[season] = value.get("expected_match_count")
        rows.append(
            MatchInput(
                season_label=season,
                match_date=value.get("match_date"),
                kickoff_utc=value.get("kickoff_utc"),
                kickoff_local_time=value.get("kickoff_local_time"),
                matchday=value.get("matchday"),
                home_team=value.get("home_team"),
                away_team=value.get("away_team"),
                home_goals=value.get("home_score"),
                away_goals=value.get("away_score"),
                home_ht_goals=value.get("home_ht_score"),
                away_ht_goals=value.get("away_ht_score"),
                source_file=value.get("source_file") or "",
                source_match_index=int(value.get("source_match_index")),
                source_format=value.get("source_format") or "",
                parse_status=value.get("parse_status") or "",
                source_id=str(value.get("match_id")) if value.get("match_id") is not None else None,
            )
        )

    matches, quality = canonicalize_rows(
        rows,
        alias_maps=alias_maps,
        expected_matches_by_season=expected,
        strict=strict,
    )
    if not matches:
        raise ValueError("no complete scored seasons were found in the Supabase export")
    export_sha256 = hashlib.sha256(export_path.read_bytes()).hexdigest()
    audit = {
        "source_kind": "supabase_mcp_export",
        "competition_name": competition_name,
        "project_ref": payload.get("project_ref"),
        "through": through,
        "export_path": str(export_path.resolve()),
        "export_sha256": export_sha256,
        "query_sha256": payload.get("query_sha256"),
        "quality": quality,
    }
    return DatasetSnapshot(
        matches=matches,
        audit=audit,
        source_kind="supabase_mcp_export",
        source_commit_sha=export_sha256,
    )


def build_feature_rows(
    matches: Sequence[CanonicalMatch],
    *,
    elo_config: EloConfig | None = None,
) -> tuple[FeatureRow, ...]:
    """Build all features before each result is applied to team state."""

    canonical_by_key = {match.row_key: match for match in matches}
    updates: list[PreMatchFeatureRow] = build_pre_match_features(
        [match.to_elo_match() for match in matches],
        elo_config=elo_config or EloConfig(),
    )
    rows: list[FeatureRow] = []
    for update in updates:
        row_key = CanonicalMatch(
            season_label=update.update.match.season_label,
            match_date=update.update.match.match_date,  # type: ignore[arg-type]
            kickoff_utc=update.update.match.kickoff_utc,
            kickoff_local_time=update.update.match.kickoff_local_time,
            home_team=update.update.match.home_team,
            away_team=update.update.match.away_team,
            home_team_key=normalize_name(update.update.match.home_team),
            away_team_key=normalize_name(update.update.match.away_team),
            home_goals=update.update.match.home_goals,
            away_goals=update.update.match.away_goals,
            source_file=update.update.match.source_file,
            source_match_index=update.update.match.source_match_index,
            source_format=update.update.match.source_format,
            parse_status=update.update.match.parse_status,
            matchday=update.update.match.matchday,
        ).row_key
        match = canonical_by_key.get(row_key)
        if match is None:
            raise AssertionError(f"feature row does not map to canonical match: {row_key}")
        rows.append(FeatureRow(match=match, values=tuple(float(value) for value in update.values)))
    return tuple(rows)


def feature_matrix(rows: Sequence[FeatureRow]) -> np.ndarray:
    """Return categorical team identities plus numeric pre-match features."""

    return np.asarray([row.raw_values for row in rows], dtype=object)


def outcome_vector(rows: Sequence[FeatureRow]) -> np.ndarray:
    return np.asarray([row.match.outcome_index for row in rows], dtype=int)


def training_class_frequencies(rows: Sequence[FeatureRow]) -> np.ndarray:
    if not rows:
        raise ValueError("cannot calculate class frequencies from an empty training set")
    counts = np.bincount(outcome_vector(rows), minlength=3).astype(float)
    if not np.all(counts > 0):
        missing = [OUTCOME_LABELS[index] for index, count in enumerate(counts) if count == 0]
        raise ValueError(f"training set is missing outcome classes: {missing}")
    return counts / counts.sum()


def fit_logistic_regression(
    rows: Sequence[FeatureRow],
    *,
    regularization_c: float = 1.0,
    verbose: bool = False,
) -> Pipeline:
    """Fit the multinomial model and all preprocessing on ``rows`` only."""

    if regularization_c <= 0:
        raise ValueError("regularization_c must be positive")
    numeric_columns = list(range(2, 2 + len(FEATURE_NAMES)))
    categorical_columns = [0, 1]
    preprocessor = ColumnTransformer(
        transformers=[
            (
                "numeric",
                Pipeline(
                    steps=[
                        ("imputer", SimpleImputer(strategy="median")),
                        ("scaler", StandardScaler()),
                    ]
                ),
                numeric_columns,
            ),
            (
                "categorical",
                Pipeline(
                    steps=[
                        ("imputer", SimpleImputer(strategy="most_frequent")),
                        ("onehot", OneHotEncoder(handle_unknown="ignore", sparse_output=False)),
                    ]
                ),
                categorical_columns,
            ),
        ],
        remainder="drop",
        verbose_feature_names_out=False,
    )
    classifier = LogisticRegression(
        C=regularization_c,
        max_iter=2000,
        solver="lbfgs",
        # In the pinned scikit-learn 1.9 runtime, lbfgs uses the multinomial
        # objective for multiclass targets and the old multi_class argument is
        # no longer accepted by the constructor.
        random_state=0,
        verbose=1 if verbose else 0,
    )
    model = Pipeline(
        steps=[
            ("preprocessor", preprocessor),
            ("classifier", classifier),
        ]
    )
    if verbose:
        print(
            f"    [logistic] fitting {len(rows):,} rows with lbfgs "
            f"(max_iter={classifier.max_iter}, C={regularization_c})",
            flush=True,
        )
    model.fit(feature_matrix(rows), outcome_vector(rows))
    if verbose:
        iterations = model.named_steps["classifier"].n_iter_.tolist()
        print(f"    [logistic] completed; iterations={iterations}", flush=True)
    return model


def model_probabilities(model: Pipeline, rows: Sequence[FeatureRow]) -> np.ndarray:
    raw = np.asarray(model.predict_proba(feature_matrix(rows)), dtype=float)
    classes = np.asarray(model.named_steps["classifier"].classes_, dtype=int)
    probabilities = np.zeros((len(rows), 3), dtype=float)
    for column, class_index in enumerate(classes):
        probabilities[:, class_index] = raw[:, column]
    return probabilities


def _normalise_probabilities(probabilities: np.ndarray) -> np.ndarray:
    p = np.asarray(probabilities, dtype=float)
    if p.ndim != 2 or p.shape[1] != 3:
        raise ValueError("probabilities must have shape (n, 3)")
    p = np.clip(p, 1e-15, 1.0)
    return p / p.sum(axis=1, keepdims=True)


def _subset_metrics(y: np.ndarray, probabilities: np.ndarray) -> dict[str, float | int | None]:
    if len(y) == 0:
        return {
            "matches": 0,
            "accuracy": None,
            "log_loss": None,
            "brier_score": None,
            "ranked_probability_score": None,
            "class_recall": {name: None for name in OUTCOME_NAMES} | {"macro": None},
        }
    p = _normalise_probabilities(probabilities)
    predictions = np.argmax(p, axis=1)
    one_hot = np.eye(3, dtype=float)[y]
    log_loss = float(-np.log(p[np.arange(len(y)), y]).mean())
    brier = float(np.square(p - one_hot).sum(axis=1).mean())
    rps = float(
        np.square(np.cumsum(p[:, :-1], axis=1) - np.cumsum(one_hot[:, :-1], axis=1))
        .sum(axis=1)
        .mean()
        / 2.0
    )
    recalls: dict[str, float | None] = {}
    for class_index, name in enumerate(OUTCOME_NAMES):
        mask = y == class_index
        recalls[name] = float(np.mean(predictions[mask] == class_index)) if np.any(mask) else None
    observed_recalls = [value for value in recalls.values() if value is not None]
    return {
        "matches": int(len(y)),
        "accuracy": float(np.mean(predictions == y)),
        "log_loss": log_loss,
        "brier_score": brier,
        "ranked_probability_score": rps,
        "class_recall": {
            **recalls,
            "macro": float(np.mean(observed_recalls)) if observed_recalls else None,
        },
        "predicted_class_counts": {
            name: int(np.sum(predictions == index)) for index, name in enumerate(OUTCOME_NAMES)
        },
    }


def evaluate_probabilities(
    rows: Sequence[FeatureRow],
    probabilities: np.ndarray,
) -> dict[str, object]:
    """Calculate proper scores, class recall, and explicitly defined upset stats."""

    y = outcome_vector(rows)
    p = _normalise_probabilities(probabilities)
    result = _subset_metrics(y, p)
    favorite = np.argmax(p, axis=1)
    upset_mask = y != favorite
    upset_rows = [rows[index] for index, is_upset in enumerate(upset_mask) if is_upset]
    upset_base = _subset_metrics(y[upset_mask], p[upset_mask])
    ranked = np.argsort(-p, axis=1)
    top_two = np.any(ranked[:, :2] == y[:, None], axis=1)
    upset_probability = float(np.mean(p[upset_mask, y[upset_mask]])) if np.any(upset_mask) else None
    upset_rank = (
        float(np.mean([int(np.where(ranked[index] == y[index])[0][0]) + 1 for index in np.flatnonzero(upset_mask)]))
        if np.any(upset_mask)
        else None
    )
    result["upset_performance"] = {
        "definition": "an upset is a match whose actual outcome is not the model's highest-probability class",
        "matches": int(np.sum(upset_mask)),
        "rate": float(np.mean(upset_mask)),
        "mean_actual_outcome_probability": upset_probability,
        "mean_actual_outcome_rank": upset_rank,
        "top_two_coverage": float(np.mean(top_two[upset_mask])) if np.any(upset_mask) else None,
        "log_loss": upset_base["log_loss"],
        "brier_score": upset_base["brier_score"],
        "ranked_probability_score": upset_base["ranked_probability_score"],
        "rows": [row.match.row_key for row in upset_rows],
    }
    return result


def chronological_key(match: CanonicalMatch) -> tuple[object, ...]:
    return match_sort_key(match.to_elo_match())


def replace_results_after_cutoff(
    matches: Sequence[CanonicalMatch],
    cutoff: tuple[object, ...],
    *,
    mode: str = "replace",
    seed: int = 0,
) -> tuple[tuple[CanonicalMatch, ...], int]:
    """Return a result-mutated copy for the post-validation leakage test."""

    if mode not in {"replace", "shuffle"}:
        raise ValueError("mode must be 'replace' or 'shuffle'")
    ordered = sorted(matches, key=chronological_key)
    future_indices = [index for index, match in enumerate(ordered) if chronological_key(match) > cutoff]
    if mode == "shuffle":
        rng = random.Random(seed)
        result_pairs = [(ordered[index].home_goals, ordered[index].away_goals) for index in future_indices]
        rng.shuffle(result_pairs)
        replacements = result_pairs
    else:
        replacements = []
        for index in future_indices:
            match = ordered[index]
            if match.home_goals != match.away_goals:
                replacements.append((match.away_goals, match.home_goals))
            else:
                replacements.append((match.home_goals + 1, match.away_goals))

    changed = 0
    result: list[CanonicalMatch] = []
    replacement_cursor = 0
    for match in ordered:
        if chronological_key(match) <= cutoff:
            result.append(match)
            continue
        new_home, new_away = replacements[replacement_cursor]
        replacement_cursor += 1
        if (new_home, new_away) == (match.home_goals, match.away_goals):
            # A shuffled draw can remain a draw with the same score. Force a
            # different valid result so the test actually perturbs every row.
            new_home, new_away = match.home_goals + 1, match.away_goals
        result.append(replace(match, home_goals=new_home, away_goals=new_away))
        changed += int((new_home, new_away) != (match.home_goals, match.away_goals))
    return tuple(result), changed


def compare_pre_match_features(
    before: Sequence[FeatureRow],
    after: Sequence[FeatureRow],
    cutoff: tuple[object, ...],
) -> dict[str, object]:
    """Compare raw feature rows at/before a cutoff using exact equality."""

    before_map = {row.match.row_key: row for row in before if chronological_key(row.match) <= cutoff}
    after_map = {row.match.row_key: row for row in after if chronological_key(row.match) <= cutoff}
    missing = sorted(set(before_map) - set(after_map))
    extra = sorted(set(after_map) - set(before_map))
    changed_rows: list[str] = []
    changed_values = 0
    for row_key in sorted(set(before_map) & set(after_map)):
        left = before_map[row_key].raw_values
        right = after_map[row_key].raw_values
        row_changed = False
        for left_value, right_value in zip(left, right):
            if left_value != right_value:
                changed_values += 1
                row_changed = True
        if row_changed:
            changed_rows.append(row_key)
    return {
        "rows_compared": len(set(before_map) & set(after_map)),
        "missing_rows": missing,
        "extra_rows": extra,
        "changed_rows": changed_rows,
        "changed_values": changed_values,
        "pass": not missing and not extra and not changed_rows,
        "features_checked": list(PRE_MATCH_FEATURE_NAMES),
    }


def split_rows(
    rows: Sequence[FeatureRow],
    *,
    train_through: str | None,
    validation_season: str,
    test_season: str,
) -> dict[str, tuple[FeatureRow, ...]]:
    """Create fixed season splits and reject temporal overlap."""

    validation = canonical_season_label(validation_season)
    test = canonical_season_label(test_season)
    validation_year = season_start_year(validation)
    test_year = season_start_year(test)
    if test_year <= validation_year:
        raise ValueError("test season must follow validation season")
    train_limit = season_start_year(canonical_season_label(train_through)) if train_through else validation_year - 1
    if train_limit >= validation_year:
        raise ValueError("train-through must precede validation season")
    train = tuple(
        row
        for row in rows
        if season_start_year(row.match.season_label) <= train_limit
    )
    validation_rows = tuple(row for row in rows if row.match.season_label == validation)
    test_rows = tuple(row for row in rows if row.match.season_label == test)
    if not train or not validation_rows or not test_rows:
        raise ValueError(
            f"empty split: train={len(train)}, validation={len(validation_rows)}, test={len(test_rows)}"
        )
    if any(season_start_year(row.match.season_label) >= validation_year for row in train):
        raise AssertionError("training rows overlap validation or test time")
    return {
        "train": tuple(sorted(train, key=lambda row: chronological_key(row.match))),
        "validation": tuple(sorted(validation_rows, key=lambda row: chronological_key(row.match))),
        "test": tuple(sorted(test_rows, key=lambda row: chronological_key(row.match))),
    }


def split_name(row: FeatureRow, splits: Mapping[str, Sequence[FeatureRow]]) -> str:
    row_key = row.match.row_key
    for name, values in splits.items():
        if any(candidate.match.row_key == row_key for candidate in values):
            return name
    return "unassigned"


def _json_default(value: object) -> object:
    if isinstance(value, (date, datetime, time)):
        return value.isoformat()
    if isinstance(value, np.generic):
        return value.item()
    raise TypeError(f"not JSON serializable: {value!r}")


def write_json(path: Path, payload: Mapping[str, object]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(payload, indent=2, sort_keys=True, default=_json_default) + "\n", encoding="utf-8")


def write_canonical_matches(path: Path, matches: Sequence[CanonicalMatch]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    fields = list(matches[0].as_dict()) if matches else ["row_key"]
    with path.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=fields)
        writer.writeheader()
        writer.writerows(match.as_dict() for match in matches)


def write_feature_rows(path: Path, rows: Sequence[FeatureRow], splits: Mapping[str, Sequence[FeatureRow]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    fields = list(rows[0].as_dict("unassigned")) if rows else ["row_key"]
    with path.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=fields)
        writer.writeheader()
        writer.writerows(row.as_dict(split_name(row, splits)) for row in rows)
