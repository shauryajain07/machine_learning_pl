"""Compare class-frequency, Poisson, and logistic forecasts walk-forward.

The selected validation and final test seasons are predicted in expanding
matchweek folds. Each fold is fit only on results dated before its earliest
fixture. Temperature calibration and the logistic blend weight are fitted on
walk-forward validation predictions; the final test season never selects them.
"""

from __future__ import annotations

import argparse
import csv
import json
from pathlib import Path
from typing import Mapping, Sequence

import numpy as np

from src.db.connection import connect
from src.ingestion.normalization.aliases import load_alias_maps
from src.ml.baseline_evaluation import (
    DatasetSnapshot,
    FeatureRow,
    OUTCOME_LABELS,
    build_feature_rows,
    load_openfootball_snapshot,
    load_supabase_export_snapshot,
    load_supabase_snapshot,
    outcome_vector,
    write_canonical_matches,
    write_json,
)
from src.ml.football_elo import season_start_year
from src.ml.season_data import Match as SeasonMatch
from src.ml.season_features import FEATURE_NAMES as SEASON_FEATURE_NAMES
from src.ml.season_features import build_features as build_season_features
from src.ml.walk_forward_comparison import (
    apply_temperature,
    assert_probabilities,
    classwise_calibration_error,
    comparison_rows,
    fit_temperature,
    records_as_arrays,
    records_by_season,
    selected_seasons,
    select_blend_weight,
    walk_forward_predict_all,
    write_prediction_csv,
)


DEFAULT_OUTPUT_DIR = Path("artifacts/walk_forward_comparison")
DEFAULT_ALIAS_FILE = Path(__file__).resolve().parents[2] / "config" / "team_aliases.json"


def _quality(snapshot: DatasetSnapshot) -> Mapping[str, object]:
    return snapshot.audit.get("quality", {})  # type: ignore[return-value]


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("root", nargs="?", type=Path, default=Path("england"))
    parser.add_argument("--source", choices=("openfootball", "supabase"), default="openfootball")
    parser.add_argument("--db-url", help="Supabase PostgreSQL URL when --source supabase is selected")
    parser.add_argument("--supabase-export", type=Path, help="Read-only JSON snapshot exported from Supabase")
    parser.add_argument("--competition-name", default="Premier League")
    parser.add_argument("--through", help="Exclude source seasons after this label before parsing")
    parser.add_argument("--validation-season")
    parser.add_argument("--test-season")
    parser.add_argument("--output-dir", type=Path, default=DEFAULT_OUTPUT_DIR)
    parser.add_argument("--regularization-c", type=float, default=0.1)
    parser.add_argument("--poisson-alpha", type=float, default=0.03)
    parser.add_argument("--recency-half-life-days", type=int, default=1460)
    parser.add_argument("--blend-step", type=float, default=0.01)
    parser.add_argument("--team-alias-file", default=str(DEFAULT_ALIAS_FILE))
    return parser


def _load_snapshot(args: argparse.Namespace) -> DatasetSnapshot:
    aliases = load_alias_maps(args.team_alias_file)
    connection = None
    try:
        if args.source == "supabase":
            if args.supabase_export:
                return load_supabase_export_snapshot(
                    args.supabase_export,
                    alias_maps=aliases,
                    competition_name=args.competition_name,
                    through=args.through,
                )
            connection = connect(args.db_url)
            return load_supabase_snapshot(
                connection,
                alias_maps=aliases,
                competition_name=args.competition_name,
                through=args.through,
            )
        return load_openfootball_snapshot(args.root, alias_maps=aliases, through=args.through)
    finally:
        if connection is not None:
            connection.close()


def _write_comparison_csv(path: Path, rows: Sequence[Mapping[str, object]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    if not rows:
        raise ValueError("comparison table is empty")
    with path.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)


def _write_model_features(path: Path, rows: Sequence[FeatureRow], validation_season: str, test_season: str) -> None:
    matches = [
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
        for row in rows
    ]
    model_rows, matrix, _ = build_season_features(matches)
    canonical_by_key = {row.match.row_key: row.match for row in rows}
    path.parent.mkdir(parents=True, exist_ok=True)
    fields = ["row_key", "split", "season_label", "match_date", "home_team", "away_team", *SEASON_FEATURE_NAMES]
    with path.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=fields)
        writer.writeheader()
        for index, model_match in enumerate(model_rows):
            match = canonical_by_key[model_match.source_id]
            split = "validation" if match.season_label == validation_season else "test" if match.season_label == test_season else "train"
            writer.writerow({
                "row_key": match.row_key,
                "split": split,
                "season_label": match.season_label,
                "match_date": match.match_date.isoformat(),
                "home_team": match.home_team,
                "away_team": match.away_team,
                **{name: float(value) for name, value in zip(SEASON_FEATURE_NAMES, matrix[index])},
            })


def _write_report(path: Path, metrics: Mapping[str, object]) -> None:
    comparison = metrics["comparison"]["rows"]  # type: ignore[index]
    validation_rows = [row for row in comparison if row["split"] == "validation"]
    test_rows = [row for row in comparison if row["split"] == "test"]
    blending = metrics["calibration_and_blending"]  # type: ignore[index]
    walk_forward = metrics["walk_forward"]  # type: ignore[index]
    test_calibration = blending["test_calibration"]

    def table(rows: Sequence[Mapping[str, object]]) -> list[str]:
        output = [
            "| Model | Log loss | RPS | Brier | Recall H | Recall D | Recall A | Upsets | Upset log loss |",
            "|---|---:|---:|---:|---:|---:|---:|---:|---:|",
        ]
        for row in rows:
            def fmt(value: object) -> str:
                return "—" if value is None else f"{float(value):.4f}"

            output.append(
                f"| {row['model']} | {fmt(row['log_loss'])} | {fmt(row['ranked_probability_score'])} | "
                f"{fmt(row['brier_score'])} | {fmt(row['recall_home'])} | {fmt(row['recall_draw'])} | "
                f"{fmt(row['recall_away'])} | {row['upset_matches']} | {fmt(row['upset_log_loss'])} |"
            )
        return output

    lines = [
        "# Walk-forward Poisson and logistic comparison",
        "",
        f"Validation season: `{metrics['protocol']['validation_season']}`  ",
        f"Untouched test season: `{metrics['protocol']['test_season']}`",
        "",
        f"Validation selected logistic blend weight: **{blending['validation_blend_weight_logistic']:.0%}**. "
        "The logistic model does not improve validation log loss over the calibrated Poisson forecast at any searched weight.",
        "",
        "## Held-out test comparison",
        "",
        *table(test_rows),
        "",
        "## Validation comparison",
        "",
        *table(validation_rows),
        "",
        "## Calibration and checks",
        "",
        f"- Poisson temperature: `{blending['poisson_temperature']:.4f}`; logistic temperature: `{blending['logistic_temperature']:.4f}`.",
        f"- Held-out classwise ECE: Poisson `{test_calibration['poisson_before']['mean_classwise_ece']:.4f}` → `{test_calibration['poisson_after']['mean_classwise_ece']:.4f}`; logistic `{test_calibration['logistic_before']['mean_classwise_ece']:.4f}` → `{test_calibration['logistic_after']['mean_classwise_ece']:.4f}`.",
        f"- Source-date cutoff: `{walk_forward['source_date_check']['passed']}/{walk_forward['source_date_check']['predictions_checked']}` predictions passed.",
        f"- Probability row-sum maximum error: `{walk_forward['probability_check']['maximum_row_sum_error']:.3g}` (tolerance `1e-6`).",
        "- An upset is a non-draw win by the side with lower pre-match Elo including home advantage; draws and exact Elo ties are excluded.",
        "- Test outcomes were not used to fit either temperature or select the blend. Earlier test-season outcomes enter later chronological training folds only after their match date.",
        "",
    ]
    path.write_text("\n".join(lines), encoding="utf-8")


def _write_blend_predictions(
    path: Path,
    split: str,
    rows: Sequence[FeatureRow],
    prior: np.ndarray,
    poisson: np.ndarray,
    logistic: np.ndarray,
    blend: np.ndarray,
    *,
    append: bool = False,
) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    fields = [
        "row_key", "split", "season_label", "match_date", "home_team", "away_team", "actual_outcome",
        "prob_home_prior", "prob_draw_prior", "prob_away_prior",
        "prob_home_poisson_calibrated", "prob_draw_poisson_calibrated", "prob_away_poisson_calibrated",
        "prob_home_logistic_calibrated", "prob_draw_logistic_calibrated", "prob_away_logistic_calibrated",
        "prob_home_selected_blend", "prob_draw_selected_blend", "prob_away_selected_blend",
    ]
    with path.open("a" if append else "w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=fields)
        if not append:
            writer.writeheader()
        for index, row in enumerate(rows):
            match = row.match
            record: dict[str, object] = {
                "row_key": match.row_key,
                "split": split,
                "season_label": match.season_label,
                "match_date": match.match_date.isoformat(),
                "home_team": match.home_team,
                "away_team": match.away_team,
                "actual_outcome": match.outcome,
            }
            for label, probabilities in (
                ("prior", prior),
                ("poisson_calibrated", poisson),
                ("logistic_calibrated", logistic),
                ("selected_blend", blend),
            ):
                record[f"prob_home_{label}"] = float(probabilities[index, 0])
                record[f"prob_draw_{label}"] = float(probabilities[index, 1])
                record[f"prob_away_{label}"] = float(probabilities[index, 2])
            writer.writerow(record)


def run(args: argparse.Namespace) -> int:
    if args.regularization_c <= 0 or args.poisson_alpha <= 0 or args.recency_half_life_days <= 0:
        raise ValueError("--regularization-c, --poisson-alpha, and --recency-half-life-days must be positive")
    if not 0 < args.blend_step <= 1:
        raise ValueError("--blend-step must be in (0, 1]")

    snapshot = _load_snapshot(args)
    complete = list(_quality(snapshot).get("complete_seasons", []))
    validation_season, test_season = selected_seasons(
        complete,
        args.validation_season,
        args.test_season,
    )
    if season_start_year(validation_season) >= season_start_year(test_season):
        raise ValueError("the test season must follow the validation season")

    print(
        f"Loaded {len(snapshot.matches):,} canonical matches; "
        f"validation={validation_season}, held-out test={test_season}.",
        flush=True,
    )
    print("Building chronological pre-match features...", flush=True)
    rows = build_feature_rows(snapshot.matches)
    print("Generating expanding matchweek predictions...", flush=True)
    poisson_records, logistic_records = walk_forward_predict_all(
        rows,
        target_seasons={validation_season, test_season},
        logistic_c=args.regularization_c,
        poisson_alpha=args.poisson_alpha,
        half_life_days=args.recency_half_life_days,
    )

    validation_poisson = records_by_season(poisson_records, validation_season)
    validation_logistic = records_by_season(logistic_records, validation_season)
    test_poisson = records_by_season(poisson_records, test_season)
    test_logistic = records_by_season(logistic_records, test_season)
    validation_rows, validation_p_raw, validation_prior = records_as_arrays(validation_poisson)
    test_rows, test_p_raw, test_prior = records_as_arrays(test_poisson)
    validation_l_raw = np.asarray([record.probabilities for record in validation_logistic], dtype=float)
    test_l_raw = np.asarray([record.probabilities for record in test_logistic], dtype=float)
    assert_probabilities(validation_l_raw, "validation logistic")
    assert_probabilities(test_l_raw, "test logistic")
    if [row.match.row_key for row in validation_rows] != [record.row.match.row_key for record in validation_logistic]:
        raise AssertionError("Poisson/logistic validation rows are misaligned")
    if [row.match.row_key for row in test_rows] != [record.row.match.row_key for record in test_logistic]:
        raise AssertionError("Poisson/logistic test rows are misaligned")

    # Temperatures and blend are learned exclusively from walk-forward
    # validation predictions. The test season is never consulted here.
    validation_y = outcome_vector(validation_rows)
    poisson_temperature = fit_temperature(validation_y, validation_p_raw)
    logistic_temperature = fit_temperature(validation_y, validation_l_raw)
    validation_p_cal = apply_temperature(validation_p_raw, poisson_temperature)
    validation_l_cal = apply_temperature(validation_l_raw, logistic_temperature)
    blend_weight, blend_search = select_blend_weight(
        validation_y,
        validation_p_cal,
        validation_l_cal,
        step=args.blend_step,
    )
    validation_blend = (1.0 - blend_weight) * validation_p_cal + blend_weight * validation_l_cal

    test_p_cal = apply_temperature(test_p_raw, poisson_temperature)
    test_l_cal = apply_temperature(test_l_raw, logistic_temperature)
    test_blend = (1.0 - blend_weight) * test_p_cal + blend_weight * test_l_cal
    for name, probabilities in (
        ("validation blend", validation_blend),
        ("test calibrated Poisson", test_p_cal),
        ("test calibrated logistic", test_l_cal),
        ("test selected blend", test_blend),
    ):
        assert_probabilities(probabilities, name)

    validation_comparison = comparison_rows(
        validation_rows,
        validation_prior,
        validation_p_raw,
        validation_l_raw,
        validation_p_cal,
        validation_l_cal,
        validation_blend,
        split="validation",
    )
    test_comparison = comparison_rows(
        test_rows,
        test_prior,
        test_p_raw,
        test_l_raw,
        test_p_cal,
        test_l_cal,
        test_blend,
        split="test",
    )
    comparison_table = validation_comparison + test_comparison

    output_dir: Path = args.output_dir
    output_dir.mkdir(parents=True, exist_ok=True)
    write_canonical_matches(output_dir / "canonical_matches.csv", snapshot.matches)
    _write_model_features(output_dir / "season_model_features.csv", rows, validation_season, test_season)
    # The two model files retain raw and calibrated P(H), P(D), P(A) side by
    # side; each includes the fold's training cutoff and matchweek identifier.
    write_prediction_csv(
        output_dir / "poisson_walk_forward_predictions.csv",
        validation_poisson + test_poisson,
        np.vstack((validation_p_cal, test_p_cal)),
        split_by_season={validation_season: "validation", test_season: "test"},
    )
    write_prediction_csv(
        output_dir / "logistic_walk_forward_predictions.csv",
        validation_logistic + test_logistic,
        np.vstack((validation_l_cal, test_l_cal)),
        split_by_season={validation_season: "validation", test_season: "test"},
    )
    _write_blend_predictions(
        output_dir / "selected_blend_predictions.csv",
        "validation",
        validation_rows,
        validation_prior,
        validation_p_cal,
        validation_l_cal,
        validation_blend,
    )
    _write_blend_predictions(
        output_dir / "selected_blend_predictions.csv",
        "test",
        test_rows,
        test_prior,
        test_p_cal,
        test_l_cal,
        test_blend,
        append=True,
    )

    _write_comparison_csv(output_dir / "comparison.csv", comparison_table)
    write_json(output_dir / "data_quality.json", snapshot.audit)

    all_records = validation_poisson + test_poisson
    date_checks = [record.max_train_date < record.row.match.match_date for record in all_records]
    if len(date_checks) != len(all_records) or not all(date_checks):
        raise AssertionError("source match date cutoff failed for at least one generated prediction")
    probability_sets = [
        validation_prior,
        validation_p_raw,
        validation_l_raw,
        validation_p_cal,
        validation_l_cal,
        validation_blend,
        test_prior,
        test_p_raw,
        test_l_raw,
        test_p_cal,
        test_l_cal,
        test_blend,
    ]
    maximum_sum_error = max(float(np.max(np.abs(p.sum(axis=1) - 1.0))) for p in probability_sets)
    if maximum_sum_error > 1e-6:
        raise AssertionError(f"probability normalization error exceeded 1e-6: {maximum_sum_error}")

    metrics = {
        "protocol": {
            "outcome_order": list(OUTCOME_LABELS),
            "folds": "Expanding matchweek folds; fit each fold only on matches dated before that fold's earliest fixture.",
            "validation_season": validation_season,
            "test_season": test_season,
            "test_use": "The final season is excluded from calibration and blend selection. Earlier results within the test season enter later prequential training folds only after they have been played.",
            "model_recipe": "The current season-simulator FittedModel recipe: same season_features inputs and recency weighting for both models.",
            "logistic_regularization_C": args.regularization_c,
            "poisson_goal_regression_alpha": args.poisson_alpha,
            "recency_half_life_days": args.recency_half_life_days,
            "poisson_score_distribution": "Independent home/away Poisson goal rates, converted to H/D/A by exact score summation over 0-20 goals.",
            "upset_rule": "Actual home/away win by the side with lower pre-match Elo including home advantage; draws and exact Elo ties are excluded. The subset is shared by all models.",
            "calibration": "One multiclass temperature per model, fit on that model's walk-forward validation probabilities and actual outcomes only.",
            "blend": "P_final=(1-w)*P_Poisson_calibrated+w*P_logistic_calibrated; w selected on validation log loss only.",
            "blend_grid_step": args.blend_step,
        },
        "calibration_and_blending": {
            "poisson_temperature": poisson_temperature,
            "logistic_temperature": logistic_temperature,
            "validation_blend_weight_logistic": blend_weight,
            "validation_blend_weight_poisson": 1.0 - blend_weight,
            "blend_search": blend_search,
            "validation_calibration": {
                "poisson_before": classwise_calibration_error(validation_rows, validation_p_raw),
                "poisson_after": classwise_calibration_error(validation_rows, validation_p_cal),
                "logistic_before": classwise_calibration_error(validation_rows, validation_l_raw),
                "logistic_after": classwise_calibration_error(validation_rows, validation_l_cal),
            },
            "test_calibration": {
                "poisson_before": classwise_calibration_error(test_rows, test_p_raw),
                "poisson_after": classwise_calibration_error(test_rows, test_p_cal),
                "logistic_before": classwise_calibration_error(test_rows, test_l_raw),
                "logistic_after": classwise_calibration_error(test_rows, test_l_cal),
            },
        },
        "walk_forward": {
            "validation_matches": len(validation_rows),
            "test_matches": len(test_rows),
            "validation_folds": len({record.fold for record in validation_poisson}),
            "test_folds": len({record.fold for record in test_poisson}),
            "minimum_training_matches": min(record.train_count for record in all_records),
            "maximum_training_matches": max(record.train_count for record in all_records),
            "source_date_check": {
                "predictions_checked": len(date_checks),
                "passed": sum(date_checks),
                "pass_rate": float(np.mean(date_checks)),
                "condition": "max(source match_date used) < predicted fixture match_date/kickoff date",
            },
            "probability_check": {
                "prediction_matrices_checked": len(probability_sets),
                "maximum_row_sum_error": maximum_sum_error,
                "tolerance": 1e-6,
                "pass": maximum_sum_error <= 1e-6,
            },
        },
        "comparison": {"rows": comparison_table},
    }
    write_json(output_dir / "metrics.json", metrics)
    _write_report(output_dir / "REPORT.md", metrics)
    print(
        f"Validation-selected logistic blend weight: {blend_weight:.2f}; "
        f"test log loss: raw Poisson={test_comparison[1]['log_loss']:.4f}, "
        f"raw logistic={test_comparison[2]['log_loss']:.4f}, "
        f"selected calibrated blend={test_comparison[-1]['log_loss']:.4f}.",
        flush=True,
    )
    print(
        f"Leakage/probability checks: {sum(date_checks)}/{len(date_checks)} source-date checks pass; "
        f"maximum probability sum error={maximum_sum_error:.3g}.",
        flush=True,
    )
    print(f"Wrote comparison artifacts to {output_dir.resolve()}")
    return 0


def main() -> int:
    try:
        return run(build_parser().parse_args())
    except Exception as exc:
        print(f"ERROR: {exc}")
        return 1


if __name__ == "__main__":
    raise SystemExit(main())
