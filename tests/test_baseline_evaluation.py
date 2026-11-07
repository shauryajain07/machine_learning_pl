from datetime import date

import numpy as np
import pytest

from src.ml.baseline_evaluation import (
    DataQualityError,
    MatchInput,
    build_feature_rows,
    canonicalize_rows,
    compare_pre_match_features,
    chronological_key,
    evaluate_probabilities,
    replace_results_after_cutoff,
)
from src.ml.football_features import FEATURE_NAMES


def source_row(
    season: str,
    index: int,
    day: int,
    home: str,
    away: str,
    home_goals: int | None,
    away_goals: int | None,
) -> MatchInput:
    return MatchInput(
        season_label=season,
        match_date=date(int(season[:4]), 8, day),
        kickoff_utc=None,
        kickoff_local_time=None,
        home_team=home,
        away_team=away,
        home_goals=home_goals,
        away_goals=away_goals,
        source_file=f"{season}/matches.txt",
        source_match_index=index,
    )


def test_canonical_table_derives_target_and_standardizes_season_labels():
    matches, audit = canonicalize_rows(
        [source_row("2024/2025", 1, 1, "  Alpha  ", "Beta", 2, 0)],
        expected_matches_by_season={"2024-25": 1},
    )

    assert len(matches) == 1
    assert matches[0].season_label == "2024-25"
    assert matches[0].home_team == "Alpha"
    assert matches[0].outcome == "H"
    assert audit["issue_counts"] == {}


@pytest.mark.parametrize(
    "row",
    [
        source_row("2024-25", 1, 1, "Alpha", "Beta", None, 0),
        source_row("2024-25", 1, 1, "Alpha", "Beta", -1, 0),
        source_row("2024-25", 1, 1, "Alpha", "Beta", 21, 0),
    ],
)
def test_missing_or_impossible_results_are_rejected(row):
    with pytest.raises(DataQualityError):
        canonicalize_rows([row], expected_matches_by_season={"2024-25": 1})


def test_halftime_score_cannot_exceed_final_score():
    row = source_row("2024-25", 1, 1, "Alpha", "Beta", 1, 0)
    row = MatchInput(**{**row.__dict__, "home_ht_goals": 2, "away_ht_goals": 0})

    with pytest.raises(DataQualityError, match="impossible_halftime_score"):
        canonicalize_rows([row], expected_matches_by_season={"2024-25": 1})


def test_duplicate_fixture_is_rejected_without_using_the_score_as_identity():
    first = source_row("2024-25", 1, 1, "Alpha", "Beta", 1, 0)
    duplicate = source_row("2024-25", 2, 1, "Alpha", "Beta", 0, 2)

    with pytest.raises(DataQualityError) as error:
        canonicalize_rows(
            [first, duplicate],
            expected_matches_by_season={"2024-25": 2},
        )

    assert any(issue.code == "duplicate_fixture" for issue in error.value.issues)


def test_incomplete_season_is_excluded_and_reported():
    matches, audit = canonicalize_rows(
        [source_row("2024-25", 1, 1, "Alpha", "Beta", 1, 0)],
        expected_matches_by_season={"2024-25": 2},
    )

    assert matches == ()
    assert audit["excluded_incomplete_seasons"] == ["2024-25"]


def test_post_cutoff_result_replacement_keeps_prior_feature_rows_identical():
    source = [
        source_row("2024-25", 1, 1, "Alpha", "Beta", 1, 0),
        source_row("2024-25", 2, 2, "Alpha", "Gamma", 0, 0),
        source_row("2025-26", 1, 1, "Alpha", "Delta", 2, 1),
    ]
    matches, _ = canonicalize_rows(source)
    before = build_feature_rows(matches)
    cutoff = chronological_key(before[1].match)
    mutated_matches, changed = replace_results_after_cutoff(matches, cutoff)
    after = build_feature_rows(mutated_matches)

    assert changed == 1
    comparison = compare_pre_match_features(before, after, cutoff)
    assert comparison["pass"] is True
    assert comparison["changed_values"] == 0


def test_probability_metrics_include_ranked_score_and_class_recall():
    source = [
        source_row("2024-25", 1, 1, "A", "B", 1, 0),
        source_row("2024-25", 2, 2, "C", "D", 0, 0),
        source_row("2024-25", 3, 3, "E", "F", 0, 1),
    ]
    matches, _ = canonicalize_rows(source)
    rows = build_feature_rows(matches)
    probabilities = np.asarray(
        [
            [0.8, 0.1, 0.1],
            [0.2, 0.6, 0.2],
            [0.1, 0.2, 0.7],
        ]
    )

    metrics = evaluate_probabilities(rows, probabilities)

    assert metrics["matches"] == 3
    assert metrics["ranked_probability_score"] == pytest.approx(0.03833333333333333)
    assert metrics["class_recall"] == {
        "home_win": 1.0,
        "draw": 1.0,
        "away_win": 1.0,
        "macro": 1.0,
    }
    assert metrics["upset_performance"]["matches"] == 0


def test_expanded_features_include_schedule_and_promoted_team_context():
    first = source_row("2023-24", 1, 1, "Alpha", "Beta", 2, 0)
    second = source_row("2024-25", 1, 1, "Newcomer", "Alpha", 1, 1)
    first = MatchInput(**{**first.__dict__, "matchday": 1})
    second = MatchInput(**{**second.__dict__, "matchday": 1})
    matches, _ = canonicalize_rows([first, second])

    rows = build_feature_rows(matches)
    assert len(rows[0].values) == len(FEATURE_NAMES)
    assert rows[1].values[FEATURE_NAMES.index("matchweek")] == pytest.approx(1.0)
    assert rows[1].values[FEATURE_NAMES.index("season_progression")] == pytest.approx(0.0)
    assert rows[1].values[FEATURE_NAMES.index("home_promoted_team")] == pytest.approx(1.0)
    assert rows[1].values[FEATURE_NAMES.index("away_promoted_team")] == pytest.approx(0.0)


def test_expanded_rolling_goal_features_use_only_previous_results():
    first = source_row("2024-25", 1, 1, "Alpha", "Beta", 4, 1)
    second = source_row("2024-25", 2, 2, "Alpha", "Gamma", 0, 0)
    matches, _ = canonicalize_rows([first, second])
    rows = build_feature_rows(matches)

    scored_index = FEATURE_NAMES.index("home_recent_goals_scored_per_match")
    conceded_index = FEATURE_NAMES.index("home_recent_goals_conceded_per_match")
    assert rows[0].values[scored_index] == pytest.approx(1.0)
    assert rows[1].values[scored_index] == pytest.approx(4.0)
    assert rows[1].values[conceded_index] == pytest.approx(1.0)
