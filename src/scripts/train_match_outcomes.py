"""Train a leakage-safe Premier League home/draw/away outcome model.

This script uses the football Elo stream plus pre-match rolling features. It
tunes the Elo parameters and logistic-regression regularization on historical
validation seasons, then reports performance on later untouched seasons.
"""

from __future__ import annotations

import argparse
import json
from dataclasses import asdict
from pathlib import Path

import numpy as np
from sklearn.linear_model import LogisticRegression
from sklearn.preprocessing import StandardScaler

from src.ml.football_elo import EloConfig, EloUpdate, FootballElo, season_start_year
from src.ml.football_features import FEATURE_NAMES, PreMatchFeatureRow, build_pre_match_features
from src.scripts.train_football_elo import (
    DEFAULT_REPOSITORY,
    DEFAULT_TEST_SEASONS,
    DEFAULT_TRAIN_THROUGH,
    DEFAULT_VALIDATION_SEASONS,
    Metrics,
    _baseline_metrics,
    _metrics_for_updates,
    _outcome_index,
    load_scored_matches,
    tune_config,
)
from src.ingestion.normalization.aliases import load_alias_maps


DEFAULT_C_VALUES = (0.01, 0.03, 0.1, 0.3, 1.0, 3.0, 10.0)


def _rows_for_seasons(rows: list[PreMatchFeatureRow], seasons: set[str]) -> list[PreMatchFeatureRow]:
    return [row for row in rows if row.update.match.season_label in seasons]


def _metrics_for_probabilities(
    rows: list[PreMatchFeatureRow],
    probabilities: np.ndarray,
) -> dict[str, float | int | None]:
    metrics = Metrics()
    for row, probability in zip(rows, probabilities):
        metrics.add(probability.tolist(), _outcome_index(row.update.match))
    return metrics.as_dict()


def _metrics_by_season(
    rows: list[PreMatchFeatureRow],
    probabilities: np.ndarray,
) -> dict[str, dict[str, float | int | None]]:
    grouped: dict[str, Metrics] = {}
    for row, probability in zip(rows, probabilities):
        grouped.setdefault(row.update.match.season_label, Metrics()).add(
            probability.tolist(), _outcome_index(row.update.match)
        )
    return {season: grouped[season].as_dict() for season in sorted(grouped, key=season_start_year)}


def _fit_model(
    scaler: StandardScaler,
    model: LogisticRegression,
    rows: list[PreMatchFeatureRow],
) -> tuple[np.ndarray, np.ndarray]:
    x = np.asarray([row.values for row in rows], dtype=float)
    y = np.asarray([_outcome_index(row.update.match) for row in rows], dtype=int)
    return scaler.fit_transform(x), y


def _write_feature_rows(path: Path, rows: list[PreMatchFeatureRow]) -> None:
    import csv

    path.parent.mkdir(parents=True, exist_ok=True)
    fields = [
        "season_label",
        "source_file",
        "source_match_index",
        "source_format",
        "parse_status",
        "match_date",
        "home_team",
        "away_team",
        "home_goals",
        "away_goals",
        "outcome",
        *FEATURE_NAMES,
    ]
    with path.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=fields)
        writer.writeheader()
        for row in rows:
            writer.writerow(row.as_dict())


def _write_model_bundle(
    path: Path,
    *,
    scaler: StandardScaler,
    model: LogisticRegression,
    c_value: float,
    train_rows: int,
    validation_rows: int,
) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    bundle = {
        "model_type": "multinomial_logistic_regression",
        "regularization_C": c_value,
        "feature_names": list(FEATURE_NAMES),
        "classes": model.classes_.tolist(),
        "scaler_mean": scaler.mean_.tolist(),
        "scaler_scale": scaler.scale_.tolist(),
        "coefficients": model.coef_.tolist(),
        "intercept": model.intercept_.tolist(),
        "train_rows": train_rows,
        "validation_rows_used_for_deployment_fit": validation_rows,
    }
    path.write_text(json.dumps(bundle, indent=2, sort_keys=True) + "\n", encoding="utf-8")


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("root", type=Path, help="Pinned OpenFootball England clone")
    parser.add_argument("--output-dir", type=Path, default=Path("artifacts/match_outcomes"))
    parser.add_argument("--through", default="2025-26")
    parser.add_argument("--train-through", default=DEFAULT_TRAIN_THROUGH)
    parser.add_argument("--validation-seasons", nargs="+", default=list(DEFAULT_VALIDATION_SEASONS))
    parser.add_argument("--test-seasons", nargs="+", default=list(DEFAULT_TEST_SEASONS))
    parser.add_argument(
        "--team-alias-file",
        default=str(Path(__file__).resolve().parents[2] / "config" / "team_aliases.json"),
    )
    parser.add_argument("--source-repository", default=DEFAULT_REPOSITORY)
    return parser


def run(args: argparse.Namespace) -> int:
    validation_seasons = set(args.validation_seasons)
    test_seasons = set(args.test_seasons)
    if season_start_year(args.train_through) >= min(season_start_year(value) for value in validation_seasons):
        raise ValueError("train-through must precede validation seasons")
    if max(season_start_year(value) for value in validation_seasons) >= min(
        season_start_year(value) for value in test_seasons
    ):
        raise ValueError("test seasons must follow validation seasons")

    alias_maps = load_alias_maps(args.team_alias_file)
    matches, load_stats, commit_sha = load_scored_matches(
        args.root,
        through=args.through,
        alias_maps=alias_maps,
    )
    elo_config, elo_validation_metrics, elo_candidate_count = tune_config(
        matches,
        validation_seasons=validation_seasons,
    )
    rows = build_pre_match_features(matches, elo_config=elo_config)
    train_rows = [row for row in rows if season_start_year(row.update.match.season_label) <= season_start_year(args.train_through)]
    validation_rows = _rows_for_seasons(rows, validation_seasons)
    test_rows = _rows_for_seasons(rows, test_seasons)
    if not train_rows or not validation_rows or not test_rows:
        raise RuntimeError("one or more requested model splits has no rows")

    x_train = np.asarray([row.values for row in train_rows], dtype=float)
    y_train = np.asarray([_outcome_index(row.update.match) for row in train_rows], dtype=int)
    x_validation = np.asarray([row.values for row in validation_rows], dtype=float)
    x_test = np.asarray([row.values for row in test_rows], dtype=float)

    scaler = StandardScaler()
    x_train_scaled = scaler.fit_transform(x_train)
    x_validation_scaled = scaler.transform(x_validation)
    x_test_scaled = scaler.transform(x_test)

    best_c: float | None = None
    best_model: LogisticRegression | None = None
    best_validation_metrics: dict[str, float | int | None] | None = None
    for c_value in DEFAULT_C_VALUES:
        candidate = LogisticRegression(C=c_value, max_iter=2000, solver="lbfgs")
        candidate.fit(x_train_scaled, y_train)
        candidate_metrics = _metrics_for_probabilities(
            validation_rows,
            candidate.predict_proba(x_validation_scaled),
        )
        log_loss = candidate_metrics["log_loss"]
        if isinstance(log_loss, float) and (best_validation_metrics is None or log_loss < best_validation_metrics["log_loss"]):
            best_c = c_value
            best_model = candidate
            best_validation_metrics = candidate_metrics
    if best_model is None or best_c is None or best_validation_metrics is None:
        raise RuntimeError("could not fit a logistic regression model")

    evaluation_validation_probabilities = best_model.predict_proba(x_validation_scaled)
    evaluation_test_probabilities = best_model.predict_proba(x_test_scaled)
    evaluation_validation = _metrics_for_probabilities(validation_rows, evaluation_validation_probabilities)
    evaluation_test = _metrics_for_probabilities(test_rows, evaluation_test_probabilities)

    deployment_rows = train_rows + validation_rows
    deployment_x = np.asarray([row.values for row in deployment_rows], dtype=float)
    deployment_y = np.asarray([_outcome_index(row.update.match) for row in deployment_rows], dtype=int)
    deployment_scaler = StandardScaler()
    deployment_x_scaled = deployment_scaler.fit_transform(deployment_x)
    deployment_model = LogisticRegression(C=best_c, max_iter=2000, solver="lbfgs")
    deployment_model.fit(deployment_x_scaled, deployment_y)

    tuned_updates = [row.update for row in rows]
    all_x_scaled = scaler.transform(np.asarray([row.values for row in rows], dtype=float))
    evaluation_all_probabilities = best_model.predict_proba(all_x_scaled)
    metrics = {
        "source": {
            "repository": args.source_repository,
            "commit_sha": commit_sha,
            "through": args.through,
            "source_files_used": load_stats["source_files_used"],
            "scored_matches": len(matches),
        },
        "splits": {
            "train_through": args.train_through,
            "train_matches": len(train_rows),
            "validation_seasons": sorted(validation_seasons, key=season_start_year),
            "validation_matches": len(validation_rows),
            "test_seasons": sorted(test_seasons, key=season_start_year),
            "test_matches": len(test_rows),
            "current_season_excluded": "2026-27",
        },
        "features": list(FEATURE_NAMES),
        "elo": {
            "candidate_count": elo_candidate_count,
            "selected_config": asdict(elo_config),
            "validation_metrics": elo_validation_metrics,
        },
        "model_tuning": {
            "candidate_C_values": list(DEFAULT_C_VALUES),
            "selected_C": best_c,
            "validation_metrics": best_validation_metrics,
        },
        "metrics": {
            "validation": {
                "historical_prior": _baseline_metrics(matches, target_seasons=validation_seasons),
                "tuned_football_elo": _metrics_for_updates(tuned_updates, validation_seasons),
                "logistic_regression": evaluation_validation,
            },
            "test": {
                "historical_prior": _baseline_metrics(matches, target_seasons=test_seasons),
                "tuned_football_elo": _metrics_for_updates(tuned_updates, test_seasons),
                "logistic_regression": evaluation_test,
            },
            "logistic_regression_by_season": _metrics_by_season(rows, evaluation_all_probabilities),
        },
    }

    output_dir: Path = args.output_dir
    output_dir.mkdir(parents=True, exist_ok=True)
    (output_dir / "metrics.json").write_text(json.dumps(metrics, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    _write_feature_rows(output_dir / "supervised_features.csv", rows)
    _write_model_bundle(
        output_dir / "deployment_model.json",
        scaler=deployment_scaler,
        model=deployment_model,
        c_value=best_c,
        train_rows=len(train_rows),
        validation_rows=len(validation_rows),
    )

    print(f"Source: {args.source_repository}")
    print(f"Commit: {commit_sha or 'unavailable'}")
    print(
        f"Loaded {len(matches)} scored matches; train={len(train_rows)}, "
        f"validation={len(validation_rows)}, test={len(test_rows)}."
    )
    print(f"Elo configuration: {json.dumps(asdict(elo_config), sort_keys=True)}")
    print(f"Logistic regression C: {best_c}")
    print("Validation comparison: " + json.dumps(metrics["metrics"]["validation"], sort_keys=True))
    print("Test comparison: " + json.dumps(metrics["metrics"]["test"], sort_keys=True))
    print(f"Wrote supervised model artifacts to {output_dir.resolve()}")
    return 0


if __name__ == "__main__":
    raise SystemExit(run(build_parser().parse_args()))
