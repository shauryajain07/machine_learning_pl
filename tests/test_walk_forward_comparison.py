from datetime import date

import numpy as np
import pytest

from src.ml.baseline_evaluation import MatchInput, build_feature_rows, canonicalize_rows
from src.ml.walk_forward_comparison import (
    apply_temperature,
    evaluate_predictions,
    fit_temperature,
    select_blend_weight,
    walk_forward_predict_all,
)


def match(season, index, day, home, away, hg, ag, matchday):
    return MatchInput(
        season_label=season,
        match_date=date.fromisoformat(day),
        kickoff_utc=None,
        kickoff_local_time=None,
        home_team=home,
        away_team=away,
        home_goals=hg,
        away_goals=ag,
        source_file=f"{season}/matches.txt",
        source_match_index=index,
        matchday=matchday,
    )


def test_temperature_and_blend_use_valid_probabilities_and_choose_on_loss():
    y = np.asarray([0, 1, 2])
    poisson = np.asarray([[0.5, 0.3, 0.2], [0.3, 0.4, 0.3], [0.2, 0.3, 0.5]])
    logistic = np.asarray([[0.8, 0.1, 0.1], [0.1, 0.8, 0.1], [0.1, 0.1, 0.8]])

    temperature = fit_temperature(y, poisson)
    calibrated = apply_temperature(poisson, temperature)
    weight, scores = select_blend_weight(y, calibrated, logistic, step=0.1)

    assert np.allclose(calibrated.sum(axis=1), 1.0, atol=1e-6)
    assert weight == min(scores, key=lambda item: (item["log_loss"], item["weight"]))["weight"]
    assert len(scores) == 11


def test_walk_forward_training_cutoff_is_strictly_before_every_fixture():
    source = [
        match("2023-24", 1, "2023-08-12", "A", "B", 2, 0, 1),
        match("2023-24", 2, "2023-08-13", "C", "D", 1, 1, 1),
        match("2023-24", 3, "2023-08-14", "E", "F", 0, 2, 1),
        match("2024-25", 1, "2024-08-10", "A", "C", 1, 0, 1),
        match("2024-25", 2, "2024-08-10", "D", "F", 0, 0, 1),
        match("2024-25", 3, "2024-08-17", "B", "E", 2, 1, 2),
    ]
    canonical, _ = canonicalize_rows(source)
    rows = build_feature_rows(canonical)

    poisson, logistic = walk_forward_predict_all(rows, target_seasons={"2024-25"})

    assert len(poisson) == len(logistic) == 3
    assert [record.row.match.row_key for record in poisson] == [record.row.match.row_key for record in logistic]
    for record in poisson + logistic:
        assert record.max_train_date < record.row.match.match_date
        assert record.train_count >= 3
        assert record.probabilities.sum() == pytest.approx(1.0, abs=1e-6)


def test_upset_subset_is_fixed_from_pre_match_elo_feature():
    source = [
        match("2023-24", 1, "2023-08-12", "A", "B", 2, 0, 1),
        match("2023-24", 2, "2023-08-13", "C", "D", 0, 2, 1),
        match("2023-24", 3, "2023-08-14", "E", "F", 1, 1, 1),
    ]
    canonical, _ = canonicalize_rows(source)
    rows = list(build_feature_rows(canonical))
    from src.ml.baseline_evaluation import FEATURE_NAMES

    elo_index = FEATURE_NAMES.index("elo_difference_with_home_advantage")
    for index, row in enumerate(rows):
        values = list(row.values)
        values[elo_index] = -20.0
        rows[index] = type(row)(match=row.match, values=tuple(values))
    probabilities = np.tile(np.asarray([0.4, 0.3, 0.3]), (len(rows), 1))

    metrics = evaluate_predictions(rows, probabilities)

    assert metrics["upset_matches"] == 1
    assert metrics["upset_definition"].startswith("actual home/away win by the side with lower pre-match Elo")
