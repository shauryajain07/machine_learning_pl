"""Typed intermediate records used between parsing and persistence.

The parser produces these objects without knowing anything about SQL. This is
intentional: a match can be tested, inspected, and validated offline before it
is sent to Supabase.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from datetime import date, datetime, time
from typing import Literal


Severity = Literal["info", "warning", "error"]
ParseStatus = Literal["clean", "warning", "failed"]
SourceFormat = Literal["full", "compact"]
Side = Literal["home", "away"]


@dataclass(frozen=True)
class ParseIssue:
    severity: Severity
    category: str
    description: str
    raw_fragment: str | None = None


@dataclass(frozen=True)
class MinuteMark:
    minute: int
    stoppage_time: int
    absolute_minute: int
    raw: str


@dataclass(frozen=True)
class Score:
    home_score: int
    away_score: int
    home_ht_score: int | None
    away_ht_score: int | None
    raw: str


@dataclass(frozen=True)
class SeasonHeader:
    season_label: str
    competition_name: str
    country: str
    start_date: date | None
    end_date: date | None
    expected_team_count: int | None
    expected_match_count: int | None
    source_header_text: str


@dataclass(frozen=True)
class RoundInfo:
    round_name: str
    round_number: int | None
    raw_heading: str


@dataclass
class ParsedAppearance:
    side: Side
    player_name: str
    started: bool
    captain: bool
    entered: MinuteMark | None
    exited: MinuteMark | None
    source_order: int
    raw_group_index: int | None
    raw_player_text: str


@dataclass
class ParsedSubstitution:
    side: Side
    player_off_name: str
    player_on_name: str
    minute_mark: MinuteMark | None
    source_order: int
    raw_event_text: str


@dataclass
class ParsedGoal:
    side: Side
    player_name: str
    minute_mark: MinuteMark | None
    penalty: bool
    own_goal: bool
    annotation: str | None
    goal_order: int
    raw_event_text: str


@dataclass
class ParsedCard:
    side: Side
    player_name: str
    card_type: Literal["yellow", "red", "yellow_red"]
    minute_mark: MinuteMark | None
    source_order: int
    raw_event_text: str


@dataclass
class ParsedMatch:
    season_label: str
    source_file: str
    source_match_index: int
    source_format: SourceFormat
    round_info: RoundInfo | None
    match_date: date | None
    source_weekday: str | None
    kickoff_local_time: time | None
    utc_offset_minutes: int | None
    kickoff_utc: datetime | None
    venue: str | None
    location: str | None
    attendance: int | None
    home_source_name: str | None
    away_source_name: str | None
    score: Score | None
    raw_match_text: str
    raw_goal_block: str | None = None
    raw_home_lineup_text: str | None = None
    raw_away_lineup_text: str | None = None
    appearances: list[ParsedAppearance] = field(default_factory=list)
    substitutions: list[ParsedSubstitution] = field(default_factory=list)
    goals: list[ParsedGoal] = field(default_factory=list)
    cards: list[ParsedCard] = field(default_factory=list)
    referees: list[str] = field(default_factory=list)
    issues: list[ParseIssue] = field(default_factory=list)
    has_goal_block: bool = False
    parse_status: ParseStatus = "clean"

    @property
    def matchday(self) -> int | None:
        return self.round_info.round_number if self.round_info else None

    @property
    def day_of_week(self) -> int | None:
        return self.match_date.isoweekday() if self.match_date else None

    @property
    def home_score(self) -> int | None:
        return self.score.home_score if self.score else None

    @property
    def away_score(self) -> int | None:
        return self.score.away_score if self.score else None

    @property
    def home_ht_score(self) -> int | None:
        return self.score.home_ht_score if self.score else None

    @property
    def away_ht_score(self) -> int | None:
        return self.score.away_ht_score if self.score else None


@dataclass
class ParsedFile:
    path: str
    season_directory: str
    header: SeasonHeader
    source_format: SourceFormat
    matches: list[ParsedMatch]
    file_sha256: str
    file_size_bytes: int
    source_commit_sha: str | None
    file_issues: list[ParseIssue] = field(default_factory=list)
