"""Validated data contracts shared by providers, models and the API."""
from __future__ import annotations

from datetime import datetime
from enum import Enum
from typing import Literal

from pydantic import BaseModel, Field, field_validator, model_validator


class FixtureStatus(str, Enum):
    SCHEDULED = "scheduled"
    LIVE = "live"
    FINISHED = "finished"
    POSTPONED = "postponed"
    CANCELLED = "cancelled"
    UNKNOWN = "unknown"


class Competition(BaseModel):
    code: str  # "PL", "UCL", "FAC", ...
    name: str
    model_supported: bool = False


PREMIER_LEAGUE = Competition(code="PL", name="Premier League", model_supported=True)


class Fixture(BaseModel):
    fixture_id: str
    competition: Competition
    season: str  # "2026-27"
    matchday: int | None = None
    home: str
    away: str
    kickoff_utc: datetime
    status: FixtureStatus
    home_goals: int | None = None
    away_goals: int | None = None
    minute: int | None = None
    venue: str | None = None
    source: str
    source_ids: dict[str, str] = Field(default_factory=dict)
    kickoff_confirmed: bool = True

    @field_validator("kickoff_utc")
    @classmethod
    def _aware(cls, value: datetime) -> datetime:
        if value.tzinfo is None:
            raise ValueError("kickoff_utc must be timezone-aware")
        return value


class Position(str, Enum):
    GK = "GK"
    DEF = "DEF"
    MID = "MID"
    FWD = "FWD"


class SquadPlayer(BaseModel):
    player_id: str  # provider-qualified id, e.g. "fpl:411"
    name: str
    short_name: str
    team: str
    position: Position
    aliases: list[str] = Field(default_factory=list)
    status_code: str | None = None  # FPL a/d/i/s/u/n
    chance_of_playing: int | None = None


class PlayerAppearance(BaseModel):
    """One historical team-match entry for a player (pre-cutoff evidence)."""

    player_id: str
    team: str
    kickoff_utc: datetime
    started: bool
    minutes: int
    goals: int = 0


class Certainty(str, Enum):
    CONFIRMED_FACT = "confirmed_fact"
    REPORTED_EXPECTATION = "reported_expectation"
    SPECULATION = "speculation"
    AI_INFERENCE = "ai_inference"


class Availability(str, Enum):
    AVAILABLE = "available"
    DOUBTFUL = "doubtful"
    UNAVAILABLE = "unavailable"
    UNKNOWN = "unknown"


class EventType(str, Enum):
    INJURY = "injury"
    ILLNESS = "illness"
    SUSPENSION = "suspension"
    POSSIBLE_ABSENCE = "possible_absence"
    FITNESS_DOUBT = "fitness_doubt"
    RETURN_FROM_INJURY = "return_from_injury"
    CONFIRMED_START = "confirmed_start"
    CONFIRMED_BENCH = "confirmed_bench"
    ROTATION_EXPECTED = "rotation_expected"
    TACTICAL_CHANGE = "tactical_change"
    FORMATION_CHANGE = "formation_change"
    PRESS_CONFERENCE = "press_conference"
    FIXTURE_CONGESTION = "fixture_congestion"
    LEFT_CLUB = "left_club"
    OTHER = "other"


class Category(str, Enum):
    PLAYER_AVAILABILITY = "player_availability"
    LINEUP = "lineup"
    TACTICS = "tactics"
    SCHEDULE = "schedule"
    OTHER = "other"


class Relevance(str, Enum):
    HIGH = "high"
    MEDIUM = "medium"
    LOW = "low"


class NewsItem(BaseModel):
    item_id: str
    title: str
    summary: str = ""
    url: str
    source: str  # publisher name
    provider: str  # integration that fetched it
    published_at: datetime
    retrieved_at: datetime
    teams: list[str] = Field(default_factory=list)
    credibility: float = Field(ge=0, le=1)
    is_simulated: bool = False
    cluster_id: str | None = None  # syndication cluster after dedup
    duplicates: list[str] = Field(default_factory=list)


class ExtractedEvent(BaseModel):
    """What an LLM/NLP extractor may return.

    Deliberately has no probability fields: the extractor converts text into
    structured evidence, never into forecast numbers.
    """

    entity: str = Field(description="Player or staff name as written in the text")
    team: str | None = Field(default=None, description="Club the entity belongs to")
    event_type: EventType
    category: Category
    certainty: Certainty
    availability: Availability
    match_relevance: Relevance
    tactical_role: str | None = Field(default=None, description="Role affected, e.g. 'defensive midfield'")
    evidence: str = Field(description="Short phrase from the text supporting the event")
    expected_return: str | None = None


class ExtractionResult(BaseModel):
    events: list[ExtractedEvent] = Field(default_factory=list)


class IntelligenceEvent(ExtractedEvent):
    """An extracted event bound to its source and to a resolved player."""

    event_id: str
    player_id: str | None = None
    player_match_score: float | None = None
    source: str
    provider: str
    url: str
    title: str
    published_at: datetime
    retrieved_at: datetime
    credibility: float
    extractor: str  # "openai:<model>", "rules", "fpl_status", "official_lineup"
    is_simulated: bool = False
    cluster_id: str | None = None


class PlayerAvailability(BaseModel):
    player_id: str
    name: str
    team: str
    p_available: float = Field(ge=0, le=1)
    status: Availability
    basis: Certainty
    conflict: bool = False
    note: str = ""
    evidence_ids: list[str] = Field(default_factory=list)


class LineupSlot(BaseModel):
    player_id: str
    name: str
    short_name: str = ""
    position: Position
    p_start: float = Field(ge=0, le=1)
    p_available: float = Field(ge=0, le=1)
    typical_start_rate: float = Field(ge=0, le=1)
    uncertain: bool
    status: Availability = Availability.UNKNOWN


class TeamLineup(BaseModel):
    team: str
    kind: Literal["predicted", "confirmed", "unavailable"]
    formation: str | None
    starters: list[LineupSlot] = Field(default_factory=list)
    bench: list[LineupSlot] = Field(default_factory=list)
    excluded: list[LineupSlot] = Field(default_factory=list)
    source: str
    note: str = ""

    @model_validator(mode="after")
    def _eleven(self):
        if self.kind != "unavailable" and len(self.starters) != 11:
            raise ValueError(f"{self.team}: lineup must contain exactly 11 starters")
        if len({s.player_id for s in self.starters}) != len(self.starters):
            raise ValueError(f"{self.team}: duplicate starter")
        return self


class OutcomeProbabilities(BaseModel):
    home_win: float
    draw: float
    away_win: float

    @model_validator(mode="after")
    def _sum_to_one(self):
        total = self.home_win + self.draw + self.away_win
        if abs(total - 1) > 1e-6 or min(self.home_win, self.draw, self.away_win) < 0:
            raise ValueError(f"Outcome probabilities must sum to one, got {total}")
        return self


class Scoreline(BaseModel):
    home_goals: int
    away_goals: int
    probability: float


class ForecastBlock(BaseModel):
    label: str
    probabilities: OutcomeProbabilities
    expected_goals_home: float
    expected_goals_away: float
    top_scorelines: list[Scoreline]
    interval_home_win: tuple[float, float] | None = None
    interval_away_win: tuple[float, float] | None = None
    monte_carlo_se: float | None = None


class Contribution(BaseModel):
    factor: str
    team: str
    delta_xg_for_team: float
    delta_xg_against_team: float
    evidence_ids: list[str] = Field(default_factory=list)
    basis: str


class InputStatus(BaseModel):
    name: str
    status: Literal["confirmed", "reported", "inferred", "missing", "stale", "simulated", "ok", "error"]
    detail: str = ""
    as_of: datetime | None = None


class Forecast(BaseModel):
    baseline: ForecastBlock
    adjusted: ForecastBlock | None
    logistic_comparison: OutcomeProbabilities | None = None
    contributions: list[Contribution] = Field(default_factory=list)
    adjustment_method: str
    adjustment_status: str
    explanation: list[str] = Field(default_factory=list)
    # Every Monte Carlo draw (probabilities, goal rates, one sampled scoreline).
    simulations: dict | None = None
    # Model-implied effect of each likely starter's absence, other inputs fixed.
    player_impacts: list[dict] = Field(default_factory=list)


class AskResponse(BaseModel):
    question: str
    status: Literal["forecast", "no_fixture", "live", "finished", "postponed",
                    "unsupported_competition", "unresolved", "error"]
    message: str
    mode: Literal["live", "demo"]
    timezone: str
    subject_team: str | None = None
    # The asked-about team's win/draw/lose chances from the headline forecast.
    subject_probabilities: dict | None = None
    # How the question was understood (LLM interpreter output after validation).
    interpretation: dict | None = None
    fixture: Fixture | None = None
    next_fixture: Fixture | None = None
    window: dict | None = None
    forecast: Forecast | None = None
    lineups: dict[str, TeamLineup] = Field(default_factory=dict)
    news: list[IntelligenceEvent] = Field(default_factory=list)
    availability: list[PlayerAvailability] = Field(default_factory=list)
    data_quality: list[InputStatus] = Field(default_factory=list)
    what_if: dict | None = None
    model_version: str = ""
    prediction_id: str | None = None
    generated_at: datetime
    notes: list[str] = Field(default_factory=list)
