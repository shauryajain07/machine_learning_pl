"""Walk-forward comparison, calibration, and blending for match outcomes."""

from __future__ import annotations

import csv
import math
from collections import defaultdict
from dataclasses import dataclass
from datetime import date
from pathlib import Path
from typing import Mapping, Sequence

import numpy as np
from scipy.optimize import minimize_scalar

from src.ml.baseline_evaluation import (
    FEATURE_NAMES,
    OUTCOME_LABELS,
    FeatureRow,
    outcome_vector,
)
from src.ml.football_elo import season_start_year
from src.ml.season_data import Match as SeasonMatch
from src.ml.season_features import build_features as build_season_features
from src.ml.season_models import Candidate, FittedModel
from src.ml.season_models import outcomes as score_outcomes
from src.ml.season_models import score_grid


@dataclass(frozen=True)
class FoldPrediction:
    """A prediction and the exact training history available at its cutoff."""

    row: FeatureRow
    fold: str
    probabilities: np.ndarray
    baseline_probabilities: np.ndarray
    train_count: int
    max_train_date: date


def fit_fold_models(
    train_features: np.ndarray,
    train_matches: Sequence[SeasonMatch],
    *,
    logistic_c: float = 0.1,
    poisson_alpha: float = 0.03,
    half_life_days: int = 1460,
) -> tuple[FittedModel, FittedModel]:
    """Fit the current recency-weighted simulator candidates on one fold."""

    if len(train_matches) == 0 or len(train_features) != len(train_matches):
        raise ValueError("cannot fit a fold without training rows")
    if logistic_c <= 0 or poisson_alpha <= 0:
        raise ValueError("regularization parameters must be positive")

    goal_model = FittedModel(Candidate("poisson", half_life_days, poisson_alpha)).fit(train_features, train_matches)
    outcome_model = FittedModel(Candidate("logistic", half_life_days, logistic_c)).fit(train_features, train_matches)
    return goal_model, outcome_model


def predict_fold_models(
    models: tuple[FittedModel, FittedModel],
    target_features: np.ndarray,
) -> tuple[np.ndarray, np.ndarray]:
    goal_model, outcome_model = models
    rates = goal_model.predict(target_features)
    poisson_probabilities = score_outcomes(score_grid(rates, rho=0.0))
    logistic_probabilities = outcome_model.predict(target_features)
    assert_probabilities(logistic_probabilities, "logistic")
    assert_probabilities(poisson_probabilities, "Poisson")
    return poisson_probabilities, logistic_probabilities


def _folds(target_rows: Sequence[FeatureRow]) -> list[tuple[str, list[FeatureRow]]]:
    """Use matchweek folds, with calendar-date folds where the round is absent."""

    groups: dict[tuple[object, ...], list[FeatureRow]] = defaultdict(list)
    for row in target_rows:
        match = row.match
        if match.matchday is not None:
            key: tuple[object, ...] = (match.season_label, "matchweek", match.matchday)
            label = f"{match.season_label}-MW{match.matchday:02d}"
        else:
            key = (match.season_label, "date", match.match_date)
            label = f"{match.season_label}-{match.match_date.isoformat()}"
        groups[key].append(row)

    def start_key(item: tuple[tuple[object, ...], list[FeatureRow]]) -> tuple[object, ...]:
        return min(row.match.match_date for row in item[1]), min(
            row.match.kickoff_utc.isoformat() if row.match.kickoff_utc else ""
            for row in item[1]
        ), item[0]

    by_key = {key: (f"{key[0]}-MW{int(key[2]):02d}" if key[1] == "matchweek" else f"{key[0]}-{key[2].isoformat()}", values)
              for key, values in groups.items()}
    ordered_keys = sorted(groups.items(), key=start_key)
    return [(by_key[key][0], sorted(values, key=lambda row: row.match.row_key)) for key, values in ordered_keys]


def walk_forward_predict_all(
    rows: Sequence[FeatureRow],
    *,
    target_seasons: set[str],
    logistic_c: float = 0.1,
    poisson_alpha: float = 0.03,
    half_life_days: int = 1460,
) -> tuple[list[FoldPrediction], list[FoldPrediction]]:
    """Return matched forecasts from the current season-simulator candidates."""

    chronological_rows = sorted(rows, key=lambda row: (row.match.match_date, row.match.row_key))
    season_matches = [
        SeasonMatch(
            season=row.match.season_label,
            date=row.match.match_date,
            home=row.match.home_team,
            away=row.match.away_team,
            hg=row.match.home_goals,
            ag=row.match.away_goals,
            matchday=row.match.matchday,
            source_id=row.match.row_key,
        )
        for row in chronological_rows
    ]
    season_feature_rows, model_matrix, _ = build_season_features(season_matches)
    features_by_key = {
        match.source_id: model_matrix[index]
        for index, match in enumerate(season_feature_rows)
    }
    model_matches_by_key = {match.source_id: match for match in season_feature_rows}
    poisson_records: list[FoldPrediction] = []
    logistic_records: list[FoldPrediction] = []
    for fold_name, targets in _folds(
        [row for row in chronological_rows if row.match.season_label in target_seasons]
    ):
        earliest_target_date = min(row.match.match_date for row in targets)
        train_rows = [row for row in chronological_rows if row.match.match_date < earliest_target_date]
        if not train_rows:
            raise ValueError(f"fold {fold_name} has no earlier training matches")
        if max(row.match.match_date for row in train_rows) >= earliest_target_date:
            raise AssertionError(f"fold {fold_name} training cutoff is not strictly before its fixtures")

        train_matches = [model_matches_by_key[row.match.row_key] for row in train_rows]
        train_features = np.asarray([features_by_key[row.match.row_key] for row in train_rows], dtype=float)
        target_features = np.asarray([features_by_key[row.match.row_key] for row in targets], dtype=float)
        models = fit_fold_models(
            train_features,
            train_matches,
            logistic_c=logistic_c,
            poisson_alpha=poisson_alpha,
            half_life_days=half_life_days,
        )
        poisson_probabilities, logistic_probabilities = predict_fold_models(
            models,
            target_features,
        )
        prior = np.bincount(outcome_vector(train_rows), minlength=3).astype(float)
        prior /= prior.sum()
        assert_probabilities(np.tile(prior, (len(targets), 1)), "class-frequency")
        latest_train_date = max(row.match.match_date for row in train_rows)
        for index, row in enumerate(targets):
            if not latest_train_date < row.match.match_date:
                raise AssertionError(f"source match date is not before kickoff date for {row.match.row_key}")
            poisson_records.append(
                FoldPrediction(row, fold_name, poisson_probabilities[index], prior.copy(), len(train_rows), latest_train_date)
            )
            logistic_records.append(
                FoldPrediction(row, fold_name, logistic_probabilities[index], prior.copy(), len(train_rows), latest_train_date)
            )
    return poisson_records, logistic_records


def assert_probabilities(probabilities: np.ndarray, name: str, *, tolerance: float = 1e-6) -> None:
    p = np.asarray(probabilities, dtype=float)
    if p.ndim != 2 or p.shape[1] != 3:
        raise AssertionError(f"{name} probabilities must have shape (n, 3)")
    if not np.all(np.isfinite(p)) or np.any(p < 0.0) or np.any(p > 1.0):
        raise AssertionError(f"{name} probabilities must be finite and lie in [0, 1]")
    if not np.allclose(p.sum(axis=1), 1.0, atol=tolerance, rtol=0.0):
        raise AssertionError(f"{name} probabilities do not sum to one within {tolerance}")


def apply_temperature(probabilities: np.ndarray, temperature: float) -> np.ndarray:
    if not np.isfinite(temperature) or temperature <= 0:
        raise ValueError("temperature must be finite and positive")
    p = np.clip(np.asarray(probabilities, dtype=float), 1e-15, 1.0)
    logits = np.log(p) / temperature
    logits -= logits.max(axis=1, keepdims=True)
    calibrated = np.exp(logits)
    calibrated /= calibrated.sum(axis=1, keepdims=True)
    assert_probabilities(calibrated, "temperature-calibrated")
    return calibrated


def fit_temperature(y: np.ndarray, probabilities: np.ndarray) -> float:
    """Fit one multiclass temperature using validation predictions only."""

    y = np.asarray(y, dtype=int)
    p = np.clip(np.asarray(probabilities, dtype=float), 1e-15, 1.0)
    if len(y) != len(p) or len(y) == 0:
        raise ValueError("temperature fitting requires aligned nonempty validation data")
    log_p = np.log(p)

    def loss(log_temperature: float) -> float:
        temperature = math.exp(log_temperature)
        logits = log_p / temperature
        logits -= logits.max(axis=1, keepdims=True)
        log_norm = np.log(np.exp(logits).sum(axis=1))
        return float(np.mean(log_norm - logits[np.arange(len(y)), y]))

    result = minimize_scalar(loss, bounds=(math.log(0.05), math.log(20.0)), method="bounded")
    candidates = [(loss(0.0), 1.0), (loss(result.x), math.exp(float(result.x)))]
    return min(candidates, key=lambda pair: pair[0])[1]


def log_loss(y: np.ndarray, probabilities: np.ndarray) -> float:
    p = np.clip(np.asarray(probabilities, dtype=float), 1e-15, 1.0)
    return float(-np.log(p[np.arange(len(y)), y]).mean())


def select_blend_weight(
    y: np.ndarray,
    poisson_probabilities: np.ndarray,
    logistic_probabilities: np.ndarray,
    *,
    step: float = 0.01,
) -> tuple[float, list[dict[str, float]]]:
    """Select logistic weight by validation log loss; ties favor lower weight."""

    if not 0 < step <= 1:
        raise ValueError("blend grid step must be in (0, 1]")
    weights = np.arange(0.0, 1.0 + step / 2.0, step)
    if weights[-1] < 1.0:
        weights = np.append(weights, 1.0)
    scores = []
    for weight in weights:
        blended = (1.0 - weight) * poisson_probabilities + weight * logistic_probabilities
        assert_probabilities(blended, "validation blend")
        scores.append({"weight": float(weight), "log_loss": log_loss(y, blended)})
    best = min(scores, key=lambda value: (value["log_loss"], value["weight"]))
    return best["weight"], scores


def _binary_upset_mask(rows: Sequence[FeatureRow]) -> np.ndarray:
    """A fixed upset is a win by the lower pre-match Elo-rated side; draws are excluded."""

    elo_index = FEATURE_NAMES.index("elo_difference_with_home_advantage")
    mask = []
    for row in rows:
        difference = row.values[elo_index]
        outcome = row.match.outcome_index
        mask.append((outcome == 0 and difference < 0.0) or (outcome == 2 and difference > 0.0))
    return np.asarray(mask, dtype=bool)


def evaluate_predictions(rows: Sequence[FeatureRow], probabilities: np.ndarray) -> dict[str, object]:
    """Evaluate proper scores, class recall, and the common Elo-defined upset subset."""

    p = np.asarray(probabilities, dtype=float)
    assert_probabilities(p, "evaluation")
    y = outcome_vector(rows)
    if len(rows) != len(p) or not len(rows):
        raise ValueError("evaluation requires aligned nonempty rows and probabilities")
    prediction = np.argmax(p, axis=1)
    one_hot = np.eye(3)[y]
    cumulative_p = np.cumsum(p[:, :2], axis=1)
    cumulative_y = np.cumsum(one_hot[:, :2], axis=1)
    upset_mask = _binary_upset_mask(rows)
    upset_count = int(upset_mask.sum())
    result: dict[str, object] = {
        "matches": int(len(y)),
        "log_loss": log_loss(y, p),
        "ranked_probability_score": float(np.square(cumulative_p - cumulative_y).sum(axis=1).mean() / 2.0),
        "brier_score": float(np.square(p - one_hot).sum(axis=1).mean()),
        "recall_home": float(np.mean(prediction[y == 0] == 0)) if np.any(y == 0) else None,
        "recall_draw": float(np.mean(prediction[y == 1] == 1)) if np.any(y == 1) else None,
        "recall_away": float(np.mean(prediction[y == 2] == 2)) if np.any(y == 2) else None,
        "upset_definition": "actual home/away win by the side with lower pre-match Elo including home advantage; draws and exact Elo ties are excluded",
        "upset_matches": upset_count,
        "upset_log_loss": log_loss(y[upset_mask], p[upset_mask]) if upset_count else None,
        "upset_ranked_probability_score": (
            float(np.square(cumulative_p[upset_mask] - cumulative_y[upset_mask]).sum(axis=1).mean() / 2.0)
            if upset_count else None
        ),
        "upset_brier_score": float(np.square(p[upset_mask] - one_hot[upset_mask]).sum(axis=1).mean()) if upset_count else None,
    }
    return result


def classwise_calibration_error(rows: Sequence[FeatureRow], probabilities: np.ndarray, bins: int = 10) -> dict[str, object]:
    y = outcome_vector(rows)
    p = np.asarray(probabilities, dtype=float)
    per_class: dict[str, float | None] = {}
    for class_index, label in enumerate(OUTCOME_LABELS):
        errors = 0.0
        total = 0
        for low in np.linspace(0.0, 1.0, bins + 1)[:-1]:
            high = min(1.0, low + 1.0 / bins)
            mask = (p[:, class_index] >= low) & (
                (p[:, class_index] < high) if high < 1.0 else (p[:, class_index] <= high)
            )
            count = int(mask.sum())
            if count:
                observed = float(np.mean(y[mask] == class_index))
                confidence = float(np.mean(p[mask, class_index]))
                errors += count * abs(observed - confidence)
                total += count
        per_class[label] = errors / total if total else None
    observed = [value for value in per_class.values() if value is not None]
    return {"bins": bins, "classwise_ece_by_outcome": per_class, "mean_classwise_ece": float(np.mean(observed)) if observed else None}


def write_prediction_csv(
    path: Path,
    records: Sequence[FoldPrediction],
    calibrated: np.ndarray | None = None,
    *,
    split_by_season: Mapping[str, str] | None = None,
) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    fields = [
        "row_key", "split", "fold", "season_label", "match_date", "kickoff_utc", "matchday",
        "home_team", "away_team", "actual_outcome", "train_matches", "max_train_match_date",
        "prob_home_raw", "prob_draw_raw", "prob_away_raw",
        "prob_home_calibrated", "prob_draw_calibrated", "prob_away_calibrated",
    ]
    with path.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=fields)
        writer.writeheader()
        for index, record in enumerate(records):
            match = record.row.match
            calibrated_row = calibrated[index] if calibrated is not None else (None, None, None)
            writer.writerow({
                "row_key": match.row_key,
                    "split": (split_by_season or {}).get(match.season_label, ""),
                "fold": record.fold,
                "season_label": match.season_label,
                "match_date": match.match_date.isoformat(),
                "kickoff_utc": match.kickoff_utc.isoformat() if match.kickoff_utc else "",
                "matchday": match.matchday if match.matchday is not None else "",
                "home_team": match.home_team,
                "away_team": match.away_team,
                "actual_outcome": match.outcome,
                "train_matches": record.train_count,
                "max_train_match_date": record.max_train_date.isoformat(),
                "prob_home_raw": float(record.probabilities[0]),
                "prob_draw_raw": float(record.probabilities[1]),
                "prob_away_raw": float(record.probabilities[2]),
                "prob_home_calibrated": "" if calibrated_row[0] is None else float(calibrated_row[0]),
                "prob_draw_calibrated": "" if calibrated_row[1] is None else float(calibrated_row[1]),
                "prob_away_calibrated": "" if calibrated_row[2] is None else float(calibrated_row[2]),
            })


def comparison_rows(
    rows: Sequence[FeatureRow],
    prior: np.ndarray,
    poisson: np.ndarray,
    logistic: np.ndarray,
    poisson_calibrated: np.ndarray,
    logistic_calibrated: np.ndarray,
    blend: np.ndarray,
    *,
    split: str,
) -> list[dict[str, object]]:
    names_and_probabilities = [
        ("class_frequency", prior),
        ("poisson", poisson),
        ("logistic", logistic),
        ("poisson_calibrated", poisson_calibrated),
        ("logistic_calibrated", logistic_calibrated),
        ("selected_blend", blend),
    ]
    return [{"split": split, "model": name, **evaluate_predictions(rows, p)} for name, p in names_and_probabilities]


def records_by_season(records: Sequence[FoldPrediction], season: str) -> list[FoldPrediction]:
    return sorted(
        [record for record in records if record.row.match.season_label == season],
        key=lambda record: (record.row.match.match_date, record.row.match.row_key),
    )


def records_as_arrays(records: Sequence[FoldPrediction]) -> tuple[list[FeatureRow], np.ndarray, np.ndarray]:
    rows = [record.row for record in records]
    raw = np.asarray([record.probabilities for record in records], dtype=float)
    prior = np.asarray([record.baseline_probabilities for record in records], dtype=float)
    assert_probabilities(raw, "raw walk-forward")
    assert_probabilities(prior, "walk-forward class-frequency")
    return rows, raw, prior


def selected_seasons(complete_seasons: Sequence[str], validation: str | None, test: str | None) -> tuple[str, str]:
    complete = sorted(set(complete_seasons), key=season_start_year)
    if len(complete) < 3:
        raise ValueError("at least three complete seasons are required for train/validation/test")
    test_season = test or complete[-1]
    earlier = [season for season in complete if season_start_year(season) < season_start_year(test_season)]
    validation_season = validation or (earlier[-1] if earlier else None)
    if validation_season is None or validation_season not in complete or test_season not in complete:
        raise ValueError("validation and test seasons must be complete source seasons")
    if season_start_year(validation_season) >= season_start_year(test_season):
        raise ValueError("the test season must follow the validation season")
    return validation_season, test_season
