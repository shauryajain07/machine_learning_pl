"""Run the reproducible Supabase/OpenFootball baseline evaluation protocol.

The default protocol uses all complete seasons before ``2024-25`` for
training, ``2024-25`` for validation, and the latest complete season
(``2025-26`` in the current data) as untouched test data. The exact seasons
are inferred from the source and can be overridden explicitly. With
``--supabase-export``, the input is a read-only JSON snapshot obtained from
the project-scoped Supabase MCP query.
"""

from __future__ import annotations

import argparse
import csv
from pathlib import Path
from typing import Mapping, Sequence

import numpy as np

from src.db.connection import connect
from src.ingestion.normalization.aliases import load_alias_maps
from src.ml.baseline_evaluation import (
    FEATURE_NAMES,
    OUTCOME_LABELS,
    PRE_MATCH_FEATURE_NAMES,
    DatasetSnapshot,
    FeatureRow,
    build_feature_rows,
    compare_pre_match_features,
    chronological_key,
    evaluate_probabilities,
    feature_matrix,
    fit_logistic_regression,
    load_openfootball_snapshot,
    load_supabase_export_snapshot,
    load_supabase_snapshot,
    model_probabilities,
    outcome_vector,
    replace_results_after_cutoff,
    split_name,
    split_rows,
    training_class_frequencies,
    write_canonical_matches,
    write_feature_rows,
    write_json,
)
from src.ml.football_elo import season_start_year


DEFAULT_OUTPUT_DIR = Path("artifacts/baseline_evaluation")
DEFAULT_ALIAS_FILE = Path(__file__).resolve().parents[2] / "config" / "team_aliases.json"


def _progress(message: str) -> None:
    print(message, flush=True)


def _quality(snapshot: DatasetSnapshot) -> Mapping[str, object]:
    return snapshot.audit.get("quality", {})  # type: ignore[return-value]


def _choose_seasons(
    snapshot: DatasetSnapshot,
    *,
    validation_season: str | None,
    test_season: str | None,
) -> tuple[str, str]:
    complete = list(_quality(snapshot).get("complete_seasons", []))
    complete.sort(key=season_start_year)
    if len(complete) < 3:
        raise ValueError("at least three complete seasons are required for train/validation/test")
    selected_test = test_season or complete[-1]
    earlier = [season for season in complete if season_start_year(season) < season_start_year(selected_test)]
    if not earlier:
        raise ValueError(f"no complete season precedes test season {selected_test}")
    selected_validation = validation_season or earlier[-1]
    if selected_validation not in complete:
        raise ValueError(f"validation season {selected_validation!r} is not complete in this source")
    if selected_test not in complete:
        raise ValueError(f"test season {selected_test!r} is not complete in this source")
    return selected_validation, selected_test


def _transformed_pre_cutoff_check(
    model,
    before: Sequence[FeatureRow],
    after: Sequence[FeatureRow],
    cutoff: tuple[object, ...],
) -> dict[str, object]:
    before_map = {row.match.row_key: row for row in before if chronological_key(row.match) <= cutoff}
    after_map = {row.match.row_key: row for row in after if chronological_key(row.match) <= cutoff}
    keys = sorted(set(before_map) & set(after_map))
    preprocessor = model.named_steps["preprocessor"]
    before_transformed = preprocessor.transform(feature_matrix([before_map[key] for key in keys]))
    after_transformed = preprocessor.transform(feature_matrix([after_map[key] for key in keys]))
    before_array = np.asarray(before_transformed)
    after_array = np.asarray(after_transformed)
    changed_values = int(np.sum(before_array != after_array))
    return {
        "rows_compared": len(keys),
        "transformed_columns": int(before_array.shape[1]) if before_array.ndim == 2 else 0,
        "changed_values": changed_values,
        "pass": bool(np.array_equal(before_array, after_array, equal_nan=True)),
    }


def _write_predictions(path: Path, rows: Sequence[FeatureRow], probabilities: np.ndarray, split: str) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    fields = [
        "row_key",
        "split",
        "season_label",
        "match_date",
        "home_team",
        "away_team",
        "prob_home_win",
        "prob_draw",
        "prob_away_win",
        "actual_outcome",
    ]
    with path.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=fields)
        writer.writeheader()
        for row, probability in zip(rows, probabilities):
            writer.writerow(
                {
                    "row_key": row.match.row_key,
                    "split": split,
                    "season_label": row.match.season_label,
                    "match_date": row.match.match_date.isoformat(),
                    "home_team": row.match.home_team,
                    "away_team": row.match.away_team,
                    "prob_home_win": float(probability[0]),
                    "prob_draw": float(probability[1]),
                    "prob_away_win": float(probability[2]),
                    "actual_outcome": row.match.outcome,
                }
            )


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "root",
        nargs="?",
        type=Path,
        default=Path("england"),
        help="OpenFootball checkout (used with --source openfootball)",
    )
    parser.add_argument("--source", choices=("openfootball", "supabase"), default="openfootball")
    parser.add_argument("--db-url", help="Supabase PostgreSQL URL when --source supabase is selected")
    parser.add_argument(
        "--supabase-export",
        type=Path,
        help="Read-only JSON export produced from the project-scoped Supabase MCP query",
    )
    parser.add_argument("--competition-name", default="Premier League")
    parser.add_argument("--output-dir", type=Path, default=DEFAULT_OUTPUT_DIR)
    parser.add_argument("--through", help="Exclude source seasons after this label before parsing")
    parser.add_argument("--validation-season")
    parser.add_argument("--test-season")
    parser.add_argument("--train-through", help="Last season included in the training split")
    parser.add_argument("--regularization-c", type=float, default=1.0)
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--leakage-mode", choices=("replace", "shuffle"), default="replace")
    parser.add_argument(
        "--team-alias-file",
        default=str(DEFAULT_ALIAS_FILE),
        help="Explicit team alias mapping used before duplicate checks",
    )
    parser.add_argument(
        "--verbose",
        action="store_true",
        help="Stream the LBFGS iterations from logistic-regression training",
    )
    return parser


def run(args: argparse.Namespace) -> int:
    aliases = load_alias_maps(args.team_alias_file)
    _progress("[1/6] Loading and validating the source snapshot...")
    connection = None
    try:
        if args.source == "supabase":
            if args.supabase_export:
                snapshot = load_supabase_export_snapshot(
                    args.supabase_export,
                    alias_maps=aliases,
                    competition_name=args.competition_name,
                    through=args.through,
                )
            else:
                connection = connect(args.db_url)
                snapshot = load_supabase_snapshot(
                    connection,
                    alias_maps=aliases,
                    competition_name=args.competition_name,
                    through=args.through,
                )
        else:
            snapshot = load_openfootball_snapshot(
                args.root,
                alias_maps=aliases,
                through=args.through,
            )
    finally:
        if connection is not None:
            connection.close()

    validation_season, test_season = _choose_seasons(
        snapshot,
        validation_season=args.validation_season,
        test_season=args.test_season,
    )
    _progress(
        f"    {len(snapshot.matches):,} canonical matches across "
        f"{len(_quality(snapshot).get('complete_seasons', []))} complete seasons; "
        f"validation={validation_season}, test={test_season}."
    )

    _progress("[2/6] Building chronological pre-match features...")
    feature_rows = build_feature_rows(snapshot.matches)
    splits = split_rows(
        feature_rows,
        train_through=args.train_through,
        validation_season=validation_season,
        test_season=test_season,
    )
    train_rows = splits["train"]
    validation_rows = splits["validation"]
    test_rows = splits["test"]
    pretest_rows = tuple(sorted(train_rows + validation_rows, key=lambda row: chronological_key(row.match)))
    _progress(
        f"    train={len(train_rows):,}, validation={len(validation_rows):,}, "
        f"test={len(test_rows):,}."
    )

    _progress("[3/6] Running the post-cutoff result mutation check...")
    cutoff = chronological_key(validation_rows[-1].match)
    mutated_matches, mutated_count = replace_results_after_cutoff(
        snapshot.matches,
        cutoff,
        mode=args.leakage_mode,
        seed=args.seed,
    )
    mutated_feature_rows = build_feature_rows(mutated_matches)
    raw_leakage = compare_pre_match_features(feature_rows, mutated_feature_rows, cutoff)

    _progress("[4/6] Fitting the validation model on training rows only...")
    validation_prior = training_class_frequencies(train_rows)
    validation_prior_probabilities = np.tile(validation_prior, (len(validation_rows), 1))
    validation_model = fit_logistic_regression(
        train_rows,
        regularization_c=args.regularization_c,
        verbose=args.verbose,
    )
    validation_probabilities = model_probabilities(validation_model, validation_rows)

    _progress("[5/6] Refitting on pre-test history and evaluating the untouched test season...")
    test_prior = training_class_frequencies(pretest_rows)
    test_prior_probabilities = np.tile(test_prior, (len(test_rows), 1))
    deployment_model = fit_logistic_regression(
        pretest_rows,
        regularization_c=args.regularization_c,
        verbose=args.verbose,
    )
    test_probabilities = model_probabilities(deployment_model, test_rows)

    transformed_leakage = _transformed_pre_cutoff_check(
        validation_model,
        feature_rows,
        mutated_feature_rows,
        cutoff,
    )
    leakage_report = {
        "cutoff_match": validation_rows[-1].match.row_key,
        "cutoff_date": validation_rows[-1].match.match_date.isoformat(),
        "mutation_mode": args.leakage_mode,
        "mutation_seed": args.seed,
        "post_cutoff_matches_mutated": mutated_count,
        "raw_features": raw_leakage,
        "fitted_preprocessor_features": transformed_leakage,
        "pass": bool(raw_leakage["pass"] and transformed_leakage["pass"]),
    }

    if not leakage_report["pass"]:
        raise RuntimeError("leakage check failed: post-cutoff results changed an earlier feature row")

    validation_metrics = {
        "class_frequency": {
            "training_probabilities": validation_prior.tolist(),
            **evaluate_probabilities(validation_rows, validation_prior_probabilities),
        },
        "multinomial_logistic_regression": {
            "regularization_C": args.regularization_c,
            "preprocessor_fit_rows": len(train_rows),
            **evaluate_probabilities(validation_rows, validation_probabilities),
        },
    }
    test_metrics = {
        "class_frequency": {
            "pretest_training_probabilities": test_prior.tolist(),
            **evaluate_probabilities(test_rows, test_prior_probabilities),
        },
        "multinomial_logistic_regression": {
            "regularization_C": args.regularization_c,
            "preprocessor_fit_rows": len(pretest_rows),
            **evaluate_probabilities(test_rows, test_probabilities),
        },
    }

    _progress("[6/6] Writing auditable artifacts...")
    output_dir: Path = args.output_dir
    output_dir.mkdir(parents=True, exist_ok=True)
    write_canonical_matches(output_dir / "canonical_matches.csv", snapshot.matches)
    write_feature_rows(output_dir / "pre_match_features.csv", feature_rows, splits)
    _write_predictions(output_dir / "validation_predictions.csv", validation_rows, validation_probabilities, "validation")
    _write_predictions(output_dir / "test_predictions.csv", test_rows, test_probabilities, "test")
    write_json(output_dir / "data_quality.json", snapshot.audit)
    write_json(output_dir / "leakage_check.json", leakage_report)
    write_json(
        output_dir / "metrics.json",
        {
            "protocol": {
                "outcome_labels": list(OUTCOME_LABELS),
                "outcome_definition": "H/D/A derived only from final home_score and away_score",
                "pre_match_features": list(PRE_MATCH_FEATURE_NAMES),
                "categorical_features": ["home_team", "away_team"],
                "numeric_features": list(FEATURE_NAMES),
                "logistic_model": "multinomial logistic regression",
                "regularization_C": args.regularization_c,
                "preprocessing": "median imputer and standard scaler for numeric; most-frequent imputer and one-hot encoder for teams; all fit on the corresponding training window only",
            },
            "source": snapshot.audit,
            "splits": {
                "train_seasons": sorted({row.match.season_label for row in train_rows}, key=season_start_year),
                "validation_season": validation_season,
                "test_season": test_season,
                "train_matches": len(train_rows),
                "validation_matches": len(validation_rows),
                "test_matches": len(test_rows),
                "test_model_training_matches": len(pretest_rows),
            },
            "metrics": {
                "validation": validation_metrics,
                "test": test_metrics,
            },
            "leakage_check": leakage_report,
        },
    )
    try:
        import joblib

        joblib.dump(
            {
                "model": deployment_model,
                "feature_names": list(PRE_MATCH_FEATURE_NAMES),
                "outcome_labels": list(OUTCOME_LABELS),
                "train_seasons": sorted({row.match.season_label for row in pretest_rows}, key=season_start_year),
                "test_season": test_season,
            },
            output_dir / "deployment_model.joblib",
        )
    except ImportError:  # pragma: no cover - dependency is pinned in requirements.lock
        pass

    print(
        "Validation log loss: "
        f"prior={validation_metrics['class_frequency']['log_loss']:.6f}, "
        f"logistic={validation_metrics['multinomial_logistic_regression']['log_loss']:.6f}"
    )
    print(
        "Validation RPS: "
        f"prior={validation_metrics['class_frequency']['ranked_probability_score']:.6f}, "
        f"logistic={validation_metrics['multinomial_logistic_regression']['ranked_probability_score']:.6f}"
    )
    print(f"Leakage check: PASS ({mutated_count:,} post-cutoff results replaced; zero earlier feature changes)")
    print(f"Artifacts: {output_dir.resolve()}")
    return 0


def main() -> int:
    try:
        return run(build_parser().parse_args())
    except Exception as exc:
        print(f"ERROR: {exc}")
        return 1


if __name__ == "__main__":
    raise SystemExit(main())
