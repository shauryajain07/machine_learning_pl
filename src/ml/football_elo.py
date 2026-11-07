"""A chronological, draw-aware football Elo rating engine.

The engine is deliberately independent of Supabase and the source parser. It
accepts normalized match records, emits pre-match features, and updates team
ratings only after the corresponding result is observed. This makes leakage
checks and chronological backtests straightforward.
"""

from __future__ import annotations

from dataclasses import dataclass
from datetime import date, datetime, time
import math
import re
from typing import Iterable


_SEASON_YEAR_RE = re.compile(r"^(?P<year>\d{4})-")


@dataclass(frozen=True)
class EloConfig:
    """Parameters for the football Elo update and draw model."""

    mean_rating: float = 1500.0
    scale: float = 400.0
    home_advantage: float = 65.0
    draw_parameter: float = 1.0
    k_factor: float = 20.0
    season_regression: float = 0.75
    margin_coefficient: float = 0.4
    max_margin_multiplier: float = 2.0

    def __post_init__(self) -> None:
        if self.scale <= 0:
            raise ValueError("scale must be positive")
        if self.draw_parameter <= 0:
            raise ValueError("draw_parameter must be positive")
        if self.k_factor <= 0:
            raise ValueError("k_factor must be positive")
        if not 0 <= self.season_regression <= 1:
            raise ValueError("season_regression must be between 0 and 1")
        if self.margin_coefficient < 0:
            raise ValueError("margin_coefficient cannot be negative")
        if self.max_margin_multiplier < 1:
            raise ValueError("max_margin_multiplier must be at least 1")


@dataclass(frozen=True)
class EloMatch:
    """The source facts needed by the Elo engine."""

    season_label: str
    source_match_index: int
    match_date: date | None
    kickoff_utc: datetime | None
    kickoff_local_time: time | None
    home_team: str
    away_team: str
    home_goals: int
    away_goals: int
    source_file: str = ""
    source_format: str = ""
    parse_status: str = "clean"
    # The scheduled matchweek is known before kickoff.  It is optional because
    # older source files do not always carry a round heading.
    matchday: int | None = None


@dataclass(frozen=True)
class EloProbabilities:
    """Three-way probabilities from the Davidson-style draw extension."""

    home_win: float
    draw: float
    away_win: float

    @property
    def expected_home_score(self) -> float:
        """Expected Elo score where a draw is worth half a win."""

        return self.home_win + 0.5 * self.draw

    @property
    def as_tuple(self) -> tuple[float, float, float]:
        return self.home_win, self.draw, self.away_win


@dataclass(frozen=True)
class EloUpdate:
    """Pre-match features and post-match rating update for one match."""

    match: EloMatch
    home_rating_before: float
    away_rating_before: float
    probabilities: EloProbabilities
    expected_home_score: float
    actual_home_score: float
    margin_multiplier: float
    home_rating_change: float
    away_rating_change: float
    home_rating_after: float
    away_rating_after: float


def season_start_year(season_label: str) -> int:
    """Return the starting calendar year from a label such as ``2025-26``."""

    match = _SEASON_YEAR_RE.match(season_label)
    if match is None:
        raise ValueError(f"Invalid season label: {season_label!r}")
    return int(match.group("year"))


def _sort_time_key(match: EloMatch) -> tuple[int, object]:
    if match.kickoff_utc is not None:
        return 0, match.kickoff_utc
    if match.kickoff_local_time is not None:
        return 1, match.kickoff_local_time
    return 2, match.source_match_index


def match_sort_key(match: EloMatch) -> tuple[object, ...]:
    """Sort source matches chronologically with a deterministic fallback."""

    date_key = match.match_date.toordinal() if match.match_date is not None else date.max.toordinal()
    return (
        season_start_year(match.season_label),
        date_key,
        _sort_time_key(match),
        match.source_match_index,
    )


def match_batch_key(match: EloMatch) -> tuple[object, ...] | None:
    """Return a key for matches whose chronology cannot be safely separated."""

    if match.match_date is None:
        return None
    if match.kickoff_utc is not None:
        return match.season_label, "utc", match.kickoff_utc
    if match.kickoff_local_time is not None:
        return match.season_label, "local", match.match_date, match.kickoff_local_time
    # With no kickoff information, matches on the same date cannot be ordered
    # safely. Treat the whole date as one pre-match batch so a source-file
    # ordering decision cannot let one result affect another prediction.
    return match.season_label, "date", match.match_date


class FootballElo:
    """Stateful football Elo with season drift and draw-aware probabilities."""

    def __init__(self, config: EloConfig | None = None) -> None:
        self.config = config or EloConfig()
        self.ratings: dict[str, float] = {}
        self._season_label: str | None = None
        self._season_year: int | None = None

    def rating(self, team: str) -> float:
        return self.ratings.get(team, self.config.mean_rating)

    def start_season(self, season_label: str) -> None:
        """Regress existing ratings once for each elapsed season."""

        target_year = season_start_year(season_label)
        if self._season_year is None:
            self._season_year = target_year
            self._season_label = season_label
            return
        if target_year <= self._season_year:
            self._season_label = season_label
            return

        for _ in range(target_year - self._season_year):
            for team, rating in self.ratings.items():
                self.ratings[team] = self.config.mean_rating + self.config.season_regression * (
                    rating - self.config.mean_rating
                )
        self._season_year = target_year
        self._season_label = season_label

    def predict(self, home_team: str, away_team: str) -> EloProbabilities:
        """Return home/draw/away probabilities before a match."""

        home_weight = 10 ** ((self.rating(home_team) + self.config.home_advantage) / self.config.scale)
        away_weight = 10 ** (self.rating(away_team) / self.config.scale)
        draw_weight = self.config.draw_parameter * math.sqrt(home_weight * away_weight)
        total = home_weight + draw_weight + away_weight
        return EloProbabilities(
            home_win=home_weight / total,
            draw=draw_weight / total,
            away_win=away_weight / total,
        )

    def margin_multiplier(self, home_goals: int, away_goals: int) -> float:
        """Reward multi-goal wins modestly while leaving one-goal wins unchanged."""

        margin = abs(home_goals - away_goals)
        if margin <= 1 or self.config.margin_coefficient == 0:
            return 1.0
        multiplier = 1.0 + self.config.margin_coefficient * math.log(margin)
        return min(self.config.max_margin_multiplier, multiplier)

    @staticmethod
    def actual_home_score(home_goals: int, away_goals: int) -> float:
        if home_goals > away_goals:
            return 1.0
        if home_goals < away_goals:
            return 0.0
        return 0.5

    def _calculate_update(self, match: EloMatch) -> EloUpdate:
        home_before = self.rating(match.home_team)
        away_before = self.rating(match.away_team)
        probabilities = self.predict(match.home_team, match.away_team)
        expected_home = probabilities.expected_home_score
        actual_home = self.actual_home_score(match.home_goals, match.away_goals)
        margin_multiplier = self.margin_multiplier(match.home_goals, match.away_goals)
        change = self.config.k_factor * margin_multiplier * (actual_home - expected_home)
        return EloUpdate(
            match=match,
            home_rating_before=home_before,
            away_rating_before=away_before,
            probabilities=probabilities,
            expected_home_score=expected_home,
            actual_home_score=actual_home,
            margin_multiplier=margin_multiplier,
            home_rating_change=change,
            away_rating_change=-change,
            home_rating_after=home_before + change,
            away_rating_after=away_before - change,
        )

    @staticmethod
    def _batch_key(match: EloMatch) -> tuple[object, ...] | None:
        """Group matches with an identical known kickoff for simultaneous updates."""

        return match_batch_key(match)

    def process(self, matches: Iterable[EloMatch]) -> list[EloUpdate]:
        """Process matches and return one update per match in chronological order."""

        ordered = sorted(matches, key=match_sort_key)
        updates: list[EloUpdate] = []
        index = 0
        while index < len(ordered):
            first = ordered[index]
            self.start_season(first.season_label)
            batch_key = self._batch_key(first)
            end = index + 1
            if batch_key is not None:
                while end < len(ordered) and self._batch_key(ordered[end]) == batch_key:
                    end += 1

            pending = [self._calculate_update(match) for match in ordered[index:end]]
            for update in pending:
                self.ratings[update.match.home_team] = update.home_rating_after
                self.ratings[update.match.away_team] = update.away_rating_after
            updates.extend(pending)
            index = end
        return updates
