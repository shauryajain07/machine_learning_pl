"""Leakage-safe chronological football features.

Every value in this module is a snapshot of team state immediately before a
fixture. A match's score is applied only after its feature row has been
created. Matches with the same known kickoff, or with no kickoff information
on the same date, are scored as one batch so source-file order cannot leak one
result into another prediction.
"""

from __future__ import annotations

from collections import deque
from dataclasses import dataclass
from datetime import date

from src.ingestion.normalization.names import normalize_name
from src.ml.football_elo import (
    EloConfig,
    EloMatch,
    EloUpdate,
    FootballElo,
    match_batch_key,
    match_sort_key,
    season_start_year,
)


# Keep the original six features first for backwards-compatible artifacts and
# tests. The remaining columns map to the requested form, venue, schedule,
# matchup, and promoted-team feature groups.
FEATURE_NAMES = (
    "elo_difference_with_home_advantage",
    "recent_points_per_match_difference",
    "recent_goal_difference_per_match_difference",
    "season_points_per_match_difference",
    "season_goal_difference_per_match_difference",
    "rest_days_difference",
    "home_recent_points_per_match",
    "away_recent_points_per_match",
    "home_recent_goals_scored_per_match",
    "away_recent_goals_scored_per_match",
    "recent_goals_scored_per_match_difference",
    "home_recent_goals_conceded_per_match",
    "away_recent_goals_conceded_per_match",
    "recent_goals_conceded_per_match_difference",
    "home_points_per_match",
    "away_points_per_match",
    "home_venue_goal_difference_per_match",
    "away_venue_goal_difference_per_match",
    "home_strength_of_schedule",
    "away_strength_of_schedule",
    "strength_of_schedule_difference",
    "home_opponent_adjusted_form",
    "away_opponent_adjusted_form",
    "opponent_adjusted_form_difference",
    "home_attack_vs_away_defence",
    "away_attack_vs_home_defence",
    "home_rest_days",
    "away_rest_days",
    "matchweek",
    "season_progression",
    "home_promoted_team",
    "away_promoted_team",
    "promoted_team_count",
)


@dataclass(frozen=True)
class PreMatchFeatureRow:
    """One feature vector with the source match and Elo update attached."""

    update: EloUpdate
    values: tuple[float, ...]

    def as_dict(self) -> dict[str, object]:
        match = self.update.match
        return {
            "season_label": match.season_label,
            "source_file": match.source_file,
            "source_match_index": match.source_match_index,
            "source_format": match.source_format,
            "parse_status": match.parse_status,
            "match_date": match.match_date.isoformat() if match.match_date else "",
            "home_team": match.home_team,
            "away_team": match.away_team,
            "matchday": match.matchday,
            "home_goals": match.home_goals,
            "away_goals": match.away_goals,
            "outcome": _outcome_label(match),
            **{name: value for name, value in zip(FEATURE_NAMES, self.values)},
        }


@dataclass
class _TeamHistory:
    """State that is available before the next result for one team."""

    recent_points: deque[float]
    recent_goal_differences: deque[float]
    recent_goals_scored: deque[float]
    recent_goals_conceded: deque[float]
    recent_opponent_ratings: deque[float]
    recent_adjusted_points: deque[float]
    season_points: float = 0.0
    season_goal_difference: float = 0.0
    season_matches: int = 0
    home_points: float = 0.0
    home_goal_difference: float = 0.0
    home_matches: int = 0
    away_points: float = 0.0
    away_goal_difference: float = 0.0
    away_matches: int = 0
    last_match_date: date | None = None

    @classmethod
    def create(cls) -> _TeamHistory:
        return cls(
            recent_points=deque(maxlen=5),
            recent_goal_differences=deque(maxlen=5),
            recent_goals_scored=deque(maxlen=5),
            recent_goals_conceded=deque(maxlen=5),
            recent_opponent_ratings=deque(maxlen=5),
            recent_adjusted_points=deque(maxlen=5),
        )

    def reset_season(self) -> None:
        self.recent_points.clear()
        self.recent_goal_differences.clear()
        self.recent_goals_scored.clear()
        self.recent_goals_conceded.clear()
        self.recent_opponent_ratings.clear()
        self.recent_adjusted_points.clear()
        self.season_points = 0.0
        self.season_goal_difference = 0.0
        self.season_matches = 0
        self.home_points = 0.0
        self.home_goal_difference = 0.0
        self.home_matches = 0
        self.away_points = 0.0
        self.away_goal_difference = 0.0
        self.away_matches = 0
        self.last_match_date = None

    @staticmethod
    def _mean(values: deque[float], default: float) -> float:
        return sum(values) / len(values) if values else default

    @property
    def recent_points_per_match(self) -> float:
        return self._mean(self.recent_points, 1.5)

    @property
    def recent_goal_difference_per_match(self) -> float:
        return self._mean(self.recent_goal_differences, 0.0)

    @property
    def recent_goals_scored_per_match(self) -> float:
        return self._mean(self.recent_goals_scored, 1.0)

    @property
    def recent_goals_conceded_per_match(self) -> float:
        return self._mean(self.recent_goals_conceded, 1.0)

    @property
    def strength_of_schedule(self) -> float:
        return self._mean(self.recent_opponent_ratings, 1500.0)

    @property
    def opponent_adjusted_form(self) -> float:
        return self._mean(self.recent_adjusted_points, 0.0)

    @property
    def season_points_per_match(self) -> float:
        return self.season_points / self.season_matches if self.season_matches else 1.5

    @property
    def season_goal_difference_per_match(self) -> float:
        return self.season_goal_difference / self.season_matches if self.season_matches else 0.0

    @property
    def home_points_per_match(self) -> float:
        return self.home_points / self.home_matches if self.home_matches else 1.5

    @property
    def away_points_per_match(self) -> float:
        return self.away_points / self.away_matches if self.away_matches else 1.5

    @property
    def home_goal_difference_per_match(self) -> float:
        return self.home_goal_difference / self.home_matches if self.home_matches else 0.0

    @property
    def away_goal_difference_per_match(self) -> float:
        return self.away_goal_difference / self.away_matches if self.away_matches else 0.0

    def rest_days(self, match_date: date | None) -> float:
        if match_date is None or self.last_match_date is None:
            return 7.0
        return float(max(0, (match_date - self.last_match_date).days))

    def record(
        self,
        *,
        points: float,
        goals_for: float,
        goals_against: float,
        opponent_rating: float,
        adjusted_points: float,
        match_date: date | None,
        venue: str,
    ) -> None:
        goal_difference = goals_for - goals_against
        self.recent_points.append(points)
        self.recent_goal_differences.append(goal_difference)
        self.recent_goals_scored.append(goals_for)
        self.recent_goals_conceded.append(goals_against)
        self.recent_opponent_ratings.append(opponent_rating)
        self.recent_adjusted_points.append(adjusted_points)
        self.season_points += points
        self.season_goal_difference += goal_difference
        self.season_matches += 1
        if venue == "home":
            self.home_points += points
            self.home_goal_difference += goal_difference
            self.home_matches += 1
        else:
            self.away_points += points
            self.away_goal_difference += goal_difference
            self.away_matches += 1
        self.last_match_date = match_date


def _outcome_label(match: EloMatch) -> str:
    if match.home_goals > match.away_goals:
        return "H"
    if match.home_goals < match.away_goals:
        return "A"
    return "D"


def _points(home_goals: int, away_goals: int) -> tuple[float, float]:
    if home_goals > away_goals:
        return 3.0, 0.0
    if home_goals < away_goals:
        return 0.0, 3.0
    return 1.0, 1.0


def _season_progression(match: EloMatch) -> float:
    """Use known matchweek, with a date-only fallback for older files."""

    if match.matchday is not None and 1 <= match.matchday <= 38:
        return float(max(0.0, min(1.0, (match.matchday - 1) / 37.0)))
    if match.match_date is None:
        return 0.0
    start = date(season_start_year(match.season_label), 8, 1)
    end = date(season_start_year(match.season_label) + 1, 5, 31)
    return float(max(0.0, min(1.0, (match.match_date - start).days / (end - start).days)))


def _promoted_teams(matches: list[EloMatch]) -> dict[str, set[str]]:
    """Identify teams entering a season relative to the prior season roster."""

    teams_by_season: dict[str, set[str]] = {}
    for match in matches:
        teams_by_season.setdefault(match.season_label, set()).update(
            (normalize_name(match.home_team), normalize_name(match.away_team))
        )
    promoted: dict[str, set[str]] = {}
    prior_teams: set[str] = set()
    for season in sorted(teams_by_season, key=season_start_year):
        current_teams = teams_by_season[season]
        promoted[season] = current_teams - prior_teams if prior_teams else set()
        prior_teams = current_teams
    return promoted


def _feature_values(
    update: EloUpdate,
    home: _TeamHistory,
    away: _TeamHistory,
    *,
    home_advantage: float,
    promoted: set[str],
) -> tuple[float, ...]:
    match = update.match
    home_rest = home.rest_days(match.match_date)
    away_rest = away.rest_days(match.match_date)
    home_promoted = float(normalize_name(match.home_team) in promoted)
    away_promoted = float(normalize_name(match.away_team) in promoted)
    return (
        update.home_rating_before + home_advantage - update.away_rating_before,
        home.recent_points_per_match - away.recent_points_per_match,
        home.recent_goal_difference_per_match - away.recent_goal_difference_per_match,
        home.season_points_per_match - away.season_points_per_match,
        home.season_goal_difference_per_match - away.season_goal_difference_per_match,
        home_rest - away_rest,
        home.recent_points_per_match,
        away.recent_points_per_match,
        home.recent_goals_scored_per_match,
        away.recent_goals_scored_per_match,
        home.recent_goals_scored_per_match - away.recent_goals_scored_per_match,
        home.recent_goals_conceded_per_match,
        away.recent_goals_conceded_per_match,
        home.recent_goals_conceded_per_match - away.recent_goals_conceded_per_match,
        home.home_points_per_match,
        away.away_points_per_match,
        home.home_goal_difference_per_match,
        away.away_goal_difference_per_match,
        home.strength_of_schedule,
        away.strength_of_schedule,
        home.strength_of_schedule - away.strength_of_schedule,
        home.opponent_adjusted_form,
        away.opponent_adjusted_form,
        home.opponent_adjusted_form - away.opponent_adjusted_form,
        home.recent_goals_scored_per_match - away.recent_goals_conceded_per_match,
        away.recent_goals_scored_per_match - home.recent_goals_conceded_per_match,
        home_rest,
        away_rest,
        float(match.matchday) if match.matchday is not None else _season_progression(match) * 37.0 + 1.0,
        _season_progression(match),
        home_promoted,
        away_promoted,
        home_promoted + away_promoted,
    )


def build_pre_match_features(
    matches: list[EloMatch],
    *,
    elo_config: EloConfig,
) -> list[PreMatchFeatureRow]:
    """Build the expanded feature vector before each result is applied."""

    ordered = sorted(matches, key=match_sort_key)
    updates = FootballElo(elo_config).process(ordered)
    promoted_by_season = _promoted_teams(ordered)
    histories: dict[str, _TeamHistory] = {}
    current_season: str | None = None
    rows: list[PreMatchFeatureRow] = []
    index = 0

    while index < len(updates):
        first = updates[index]
        if first.match.season_label != current_season:
            for history in histories.values():
                history.reset_season()
            current_season = first.match.season_label

        batch_key = match_batch_key(first.match)
        end = index + 1
        if batch_key is not None:
            while end < len(updates) and match_batch_key(updates[end].match) == batch_key:
                end += 1

        pending: list[tuple[EloUpdate, _TeamHistory, _TeamHistory, tuple[float, float]]] = []
        for update in updates[index:end]:
            match = update.match
            home_history = histories.setdefault(match.home_team, _TeamHistory.create())
            away_history = histories.setdefault(match.away_team, _TeamHistory.create())
            values = _feature_values(
                update,
                home_history,
                away_history,
                home_advantage=elo_config.home_advantage,
                promoted=promoted_by_season.get(match.season_label, set()),
            )
            rows.append(PreMatchFeatureRow(update=update, values=values))
            home_points, away_points = _points(match.home_goals, match.away_goals)
            pending.append((update, home_history, away_history, (home_points, away_points)))

        # Only after every row in the batch has been emitted do we expose its
        # final score to subsequent chronological predictions.
        for update, home_history, away_history, (home_points, away_points) in pending:
            match = update.match
            home_expected_points = 3.0 * update.probabilities.home_win + update.probabilities.draw
            away_expected_points = 3.0 * update.probabilities.away_win + update.probabilities.draw
            home_history.record(
                points=home_points,
                goals_for=float(match.home_goals),
                goals_against=float(match.away_goals),
                opponent_rating=update.away_rating_before,
                adjusted_points=home_points - home_expected_points,
                match_date=match.match_date,
                venue="home",
            )
            away_history.record(
                points=away_points,
                goals_for=float(match.away_goals),
                goals_against=float(match.home_goals),
                opponent_rating=update.home_rating_before,
                adjusted_points=away_points - away_expected_points,
                match_date=match.match_date,
                venue="away",
            )
        index = end

    return rows
