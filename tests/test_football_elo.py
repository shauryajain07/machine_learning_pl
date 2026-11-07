from datetime import date, datetime, time, timezone

import pytest

from src.ml.football_elo import EloConfig, EloMatch, FootballElo
from src.ml.football_features import build_pre_match_features


def match(
    season: str,
    index: int,
    home: str,
    away: str,
    home_goals: int,
    away_goals: int,
    *,
    day: int = 1,
    kickoff: time | None = time(15, 0),
) -> EloMatch:
    return EloMatch(
        season_label=season,
        source_match_index=index,
        match_date=date(2024, 8, day),
        kickoff_utc=(
            datetime(2024, 8, day, kickoff.hour, kickoff.minute, tzinfo=timezone.utc)
            if kickoff is not None
            else None
        ),
        kickoff_local_time=kickoff,
        home_team=home,
        away_team=away,
        home_goals=home_goals,
        away_goals=away_goals,
    )


def test_draw_aware_probabilities_sum_to_one_and_include_home_advantage():
    engine = FootballElo(EloConfig(home_advantage=65, draw_parameter=1.0))

    probabilities = engine.predict("A", "B")

    assert sum(probabilities.as_tuple) == pytest.approx(1.0)
    assert probabilities.home_win > probabilities.away_win
    assert probabilities.draw > 0


def test_draw_moves_a_favored_home_team_down_and_away_team_up():
    engine = FootballElo(EloConfig(home_advantage=65, draw_parameter=1.0, k_factor=20))
    engine.ratings.update({"A": 1600.0, "B": 1500.0})

    update = engine.process([match("2024-25", 1, "A", "B", 0, 0)])[0]

    assert update.expected_home_score > 0.5
    assert update.home_rating_change < 0
    assert update.away_rating_change > 0


def test_margin_multiplier_leaves_one_goal_wins_unchanged():
    engine = FootballElo(EloConfig(margin_coefficient=0.4, max_margin_multiplier=2.0))

    assert engine.margin_multiplier(2, 1) == pytest.approx(1.0)
    assert engine.margin_multiplier(3, 1) > 1.0
    assert engine.margin_multiplier(10, 0) <= 2.0


def test_season_regression_moves_ratings_toward_mean():
    engine = FootballElo(EloConfig(season_regression=0.75))
    engine.ratings["A"] = 1700.0
    engine.start_season("2024-25")
    engine.start_season("2025-26")

    assert engine.rating("A") == pytest.approx(1650.0)


def test_pre_match_rating_does_not_use_current_match_result():
    engine = FootballElo(EloConfig(home_advantage=0, draw_parameter=1.0, k_factor=20))
    updates = engine.process(
        [
            match("2024-25", 1, "A", "B", 5, 0, day=1),
            match("2024-25", 2, "A", "C", 0, 0, day=2),
        ]
    )

    assert updates[0].home_rating_before == pytest.approx(1500.0)
    assert updates[1].home_rating_before == pytest.approx(updates[0].home_rating_after)


def test_identical_kickoffs_share_pre_match_ratings_before_batch_update():
    engine = FootballElo(EloConfig(home_advantage=0, draw_parameter=1.0, k_factor=20))
    updates = engine.process(
        [
            match("2024-25", 1, "A", "B", 3, 0, day=1),
            match("2024-25", 2, "A", "C", 0, 0, day=1),
        ]
    )

    assert updates[0].home_rating_before == pytest.approx(1500.0)
    assert updates[1].home_rating_before == pytest.approx(1500.0)


def test_unknown_kickoffs_on_same_date_share_pre_match_ratings_before_batch_update():
    engine = FootballElo(EloConfig(home_advantage=0, draw_parameter=1.0, k_factor=20))
    updates = engine.process(
        [
            match("2024-25", 1, "A", "B", 3, 0, day=1, kickoff=None),
            match("2024-25", 2, "A", "C", 0, 0, day=1, kickoff=None),
        ]
    )

    assert updates[0].home_rating_before == pytest.approx(1500.0)
    assert updates[1].home_rating_before == pytest.approx(1500.0)


def test_rolling_features_only_reflect_previous_matches():
    rows = build_pre_match_features(
        [
            match("2024-25", 1, "A", "B", 5, 0, day=1),
            match("2024-25", 2, "A", "C", 0, 0, day=2),
        ],
        elo_config=EloConfig(home_advantage=0),
    )

    assert rows[0].values[0] == pytest.approx(0.0)
    assert rows[0].values[1:5] == pytest.approx((0.0, 0.0, 0.0, 0.0))
    assert rows[1].values[1] == pytest.approx(1.5)
    assert rows[1].values[2] == pytest.approx(5.0)
