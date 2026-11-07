"""Train and backtest the Premier League football Elo model offline.

The script parses the pinned local OpenFootball clone, canonicalizes team
names using the project's explicit aliases, and processes scored matches in
time order. It does not use the current/in-progress season by default.
"""

from __future__ import annotations

import argparse
import csv
from dataclasses import asdict
from datetime import date, datetime, time
import json
import math
from pathlib import Path
from typing import Iterable, Sequence

from src.ingestion.football_txt.file_parser import parse_file
from src.ingestion.normalization.aliases import load_alias_maps
from src.ingestion.normalization.names import clean_display_name, normalize_name
from src.ingestion.source import repository_commit_sha
from src.scripts.ingest_openfootball import discover_source_files
from src.ml.football_elo import (
    EloConfig,
    EloMatch,
    EloProbabilities,
    EloUpdate,
    FootballElo,
    match_sort_key,
    season_start_year,
)


DEFAULT_REPOSITORY = "https://github.com/openfootball/england"
DEFAULT_THROUGH = "2025-26"
DEFAULT_TRAIN_THROUGH = "2020-21"
DEFAULT_VALIDATION_SEASONS = ("2021-22", "2022-23", "2023-24")
DEFAULT_TEST_SEASONS = ("2024-25", "2025-26")


def _source_path(path: Path, root: Path) -> str:
    try:
        return path.resolve().relative_to(root.resolve()).as_posix()
    except ValueError:
        return path.name


def _canonical_team(alias_maps, source_name: str) -> tuple[str, str]:
    display = clean_display_name(source_name)
    canonical = clean_display_name(alias_maps.target("team", display))
    return canonical, normalize_name(canonical)


def _season_is_at_most(season_label: str, through: str) -> bool:
    return season_start_year(season_label) <= season_start_year(through)


def load_scored_matches(
    root: Path,
    *,
    through: str,
    alias_maps,
) -> tuple[list[EloMatch], dict[str, int], str | None]:
    """Parse the source and return scored matches through the requested season."""

    files = discover_source_files(root, all_seasons=True)
    commit_sha = repository_commit_sha(root)
    matches: list[EloMatch] = []
    stats = {
        "source_files_discovered": len(files),
        "source_files_used": 0,
        "parsed_matches_through_cutoff": 0,
        "scored_matches": 0,
        "skipped_missing_score_or_teams": 0,
        "skipped_after_cutoff": 0,
        "warning_matches": 0,
        "failed_matches": 0,
    }

    for source in files:
        relative_source = _source_path(source, root)
        # Exclude future files before reading/parsing their contents.
        if not _season_is_at_most(source.parent.name, through):
            stats.setdefault("source_files_excluded_before_read", 0)
            stats["source_files_excluded_before_read"] += 1
            continue
        parsed_file = parse_file(
            source,
            source_file=relative_source,
            source_commit_sha=commit_sha,
        )
        if not _season_is_at_most(parsed_file.season_directory, through):
            stats["skipped_after_cutoff"] += len(parsed_file.matches)
            continue
        stats["source_files_used"] += 1
        for parsed in parsed_file.matches:
            stats["parsed_matches_through_cutoff"] += 1
            if parsed.parse_status == "warning":
                stats["warning_matches"] += 1
            elif parsed.parse_status == "failed":
                stats["failed_matches"] += 1
            if (
                parsed.score is None
                or parsed.home_source_name is None
                or parsed.away_source_name is None
            ):
                stats["skipped_missing_score_or_teams"] += 1
                continue
            home_display, home_key = _canonical_team(alias_maps, parsed.home_source_name)
            away_display, away_key = _canonical_team(alias_maps, parsed.away_source_name)
            if home_key == away_key:
                stats["skipped_missing_score_or_teams"] += 1
                continue
            matches.append(
                EloMatch(
                    season_label=parsed.season_label,
                    source_match_index=parsed.source_match_index,
                    match_date=parsed.match_date,
                    kickoff_utc=parsed.kickoff_utc,
                    kickoff_local_time=parsed.kickoff_local_time,
                    home_team=home_display,
                    away_team=away_display,
                    home_goals=parsed.score.home_score,
                    away_goals=parsed.score.away_score,
                    source_file=parsed.source_file,
                    source_format=parsed.source_format,
                    parse_status=parsed.parse_status,
                )
            )

    matches.sort(key=match_sort_key)
    stats["scored_matches"] = len(matches)
    return matches, stats, commit_sha


class Metrics:
    """Small dependency-free multiclass probability metric accumulator."""

    def __init__(self) -> None:
        self.count = 0
        self.correct = 0
        self.log_loss_total = 0.0
        self.brier_total = 0.0

    def add(self, probabilities: Sequence[float], outcome: int) -> None:
        if len(probabilities) != 3:
            raise ValueError("expected home/draw/away probabilities")
        clipped = [max(1e-15, min(1.0, float(value))) for value in probabilities]
        predicted = max(range(3), key=lambda index: clipped[index])
        self.count += 1
        self.correct += int(predicted == outcome)
        self.log_loss_total -= math.log(clipped[outcome])
        self.brier_total += sum((probability - int(index == outcome)) ** 2 for index, probability in enumerate(clipped))

    def as_dict(self) -> dict[str, float | int]:
        if self.count == 0:
            return {"matches": 0, "accuracy": None, "log_loss": None, "brier_score": None}  # type: ignore[return-value]
        return {
            "matches": self.count,
            "accuracy": self.correct / self.count,
            "log_loss": self.log_loss_total / self.count,
            "brier_score": self.brier_total / self.count,
        }


def _outcome_index(match: EloMatch) -> int:
    if match.home_goals > match.away_goals:
        return 0
    if match.home_goals == match.away_goals:
        return 1
    return 2


def _metrics_for_updates(
    updates: Iterable[EloUpdate],
    seasons: set[str],
) -> dict[str, float | int | None]:
    metrics = Metrics()
    for update in updates:
        if update.match.season_label in seasons:
            metrics.add(update.probabilities.as_tuple, _outcome_index(update.match))
    return metrics.as_dict()


def _metrics_by_season(updates: Iterable[EloUpdate]) -> dict[str, dict[str, float | int | None]]:
    accumulators: dict[str, Metrics] = {}
    for update in updates:
        accumulator = accumulators.setdefault(update.match.season_label, Metrics())
        accumulator.add(update.probabilities.as_tuple, _outcome_index(update.match))
    return {season: accumulators[season].as_dict() for season in sorted(accumulators, key=season_start_year)}


def _outcome_counts(matches: Iterable[EloMatch]) -> list[int]:
    counts = [0, 0, 0]
    for match in matches:
        counts[_outcome_index(match)] += 1
    return counts


def _prior_probabilities(matches: Iterable[EloMatch]) -> tuple[float, float, float]:
    counts = _outcome_counts(matches)
    smoothed = [count + 1 for count in counts]
    total = sum(smoothed)
    return tuple(count / total for count in smoothed)  # type: ignore[return-value]


def _baseline_metrics(
    matches: Sequence[EloMatch],
    *,
    target_seasons: set[str],
) -> dict[str, float | int | None]:
    metrics = Metrics()
    for season in sorted(target_seasons, key=season_start_year):
        prior = _prior_probabilities(
            match for match in matches if season_start_year(match.season_label) < season_start_year(season)
        )
        for match in matches:
            if match.season_label == season:
                metrics.add(prior, _outcome_index(match))
    return metrics.as_dict()


def _candidate_configs() -> Iterable[EloConfig]:
    for k_factor in (12.0, 18.0, 24.0, 30.0):
        for home_advantage in (50.0, 60.0, 70.0, 80.0):
            for draw_parameter in (0.6, 0.8, 1.0, 1.2):
                for season_regression in (0.65, 0.75, 0.85):
                    for margin_coefficient in (0.0, 0.4):
                        yield EloConfig(
                            k_factor=k_factor,
                            home_advantage=home_advantage,
                            draw_parameter=draw_parameter,
                            season_regression=season_regression,
                            margin_coefficient=margin_coefficient,
                        )


def tune_config(
    matches: Sequence[EloMatch],
    *,
    validation_seasons: set[str],
) -> tuple[EloConfig, dict[str, float | int | None], int]:
    """Select parameters by multi-season validation log loss."""

    latest_validation_year = max(season_start_year(season) for season in validation_seasons)
    validation_matches = [
        match for match in matches if season_start_year(match.season_label) <= latest_validation_year
    ]
    best_config: EloConfig | None = None
    best_metrics: dict[str, float | int | None] | None = None
    best_log_loss = float("inf")
    candidates = 0
    for config in _candidate_configs():
        candidates += 1
        updates = FootballElo(config).process(validation_matches)
        metrics = _metrics_for_updates(updates, validation_seasons)
        log_loss = metrics["log_loss"]
        if isinstance(log_loss, float) and log_loss < best_log_loss:
            best_log_loss = log_loss
            best_config = config
            best_metrics = metrics
    if best_config is None or best_metrics is None:
        raise RuntimeError("could not tune an Elo configuration")
    return best_config, best_metrics, candidates


def _format_value(value: object) -> str:
    if value is None:
        return ""
    if isinstance(value, (date, datetime, time)):
        return value.isoformat()
    return str(value)


def write_features(path: Path, updates: Sequence[EloUpdate], config: EloConfig) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    fields = [
        "season_label",
        "source_file",
        "source_match_index",
        "source_format",
        "parse_status",
        "match_date",
        "kickoff_utc",
        "kickoff_local_time",
        "home_team",
        "away_team",
        "home_goals",
        "away_goals",
        "home_elo_pre",
        "away_elo_pre",
        "elo_difference_with_home_advantage",
        "prob_home_win",
        "prob_draw",
        "prob_away_win",
        "expected_home_score",
        "actual_home_score",
        "margin_multiplier",
        "home_elo_change",
        "away_elo_change",
        "home_elo_post",
        "away_elo_post",
    ]
    with path.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=fields)
        writer.writeheader()
        for update in updates:
            match = update.match
            writer.writerow(
                {
                    "season_label": match.season_label,
                    "source_file": match.source_file,
                    "source_match_index": match.source_match_index,
                    "source_format": match.source_format,
                    "parse_status": match.parse_status,
                    "match_date": _format_value(match.match_date),
                    "kickoff_utc": _format_value(match.kickoff_utc),
                    "kickoff_local_time": _format_value(match.kickoff_local_time),
                    "home_team": match.home_team,
                    "away_team": match.away_team,
                    "home_goals": match.home_goals,
                    "away_goals": match.away_goals,
                    "home_elo_pre": round(update.home_rating_before, 6),
                    "away_elo_pre": round(update.away_rating_before, 6),
                    "elo_difference_with_home_advantage": round(
                        update.home_rating_before + config.home_advantage - update.away_rating_before,
                        6,
                    ),
                    "prob_home_win": round(update.probabilities.home_win, 9),
                    "prob_draw": round(update.probabilities.draw, 9),
                    "prob_away_win": round(update.probabilities.away_win, 9),
                    "expected_home_score": round(update.expected_home_score, 9),
                    "actual_home_score": update.actual_home_score,
                    "margin_multiplier": round(update.margin_multiplier, 9),
                    "home_elo_change": round(update.home_rating_change, 6),
                    "away_elo_change": round(update.away_rating_change, 6),
                    "home_elo_post": round(update.home_rating_after, 6),
                    "away_elo_post": round(update.away_rating_after, 6),
                }
            )


def write_ratings(path: Path, engine: FootballElo) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=["team", "rating"])
        writer.writeheader()
        for team, rating in sorted(engine.ratings.items(), key=lambda item: (-item[1], item[0])):
            writer.writerow({"team": team, "rating": round(rating, 6)})


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("root", type=Path, help="Pinned OpenFootball England clone")
    parser.add_argument("--output-dir", type=Path, default=Path("artifacts/football_elo"))
    parser.add_argument("--through", default=DEFAULT_THROUGH)
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
    if not validation_seasons:
        raise ValueError("at least one validation season is required")
    if season_start_year(args.train_through) >= min(season_start_year(season) for season in validation_seasons):
        raise ValueError("train-through must precede validation-season")
    if any(
        season_start_year(season) <= max(season_start_year(value) for value in validation_seasons)
        for season in args.test_seasons
    ):
        raise ValueError("test seasons must follow validation-season")
    if any(season_start_year(season) > season_start_year(args.through) for season in args.test_seasons):
        raise ValueError("--through must include every test season")

    alias_maps = load_alias_maps(args.team_alias_file)
    matches, load_stats, commit_sha = load_scored_matches(
        args.root,
        through=args.through,
        alias_maps=alias_maps,
    )
    print(f"Source: {args.source_repository}")
    print(f"Commit: {commit_sha or 'unavailable'}")
    print(
        f"Loaded {len(matches)} scored matches from {load_stats['source_files_used']} source files "
        f"through {args.through}; skipped {load_stats['skipped_after_cutoff']} current-season records."
    )

    tuned_config, validation_metrics, candidate_count = tune_config(
        matches,
        validation_seasons=validation_seasons,
    )
    print(f"Tuned {candidate_count} configurations on {', '.join(sorted(validation_seasons, key=season_start_year))} by log loss.")
    print("Best configuration: " + json.dumps(asdict(tuned_config), sort_keys=True))
    print("Validation metrics: " + json.dumps(validation_metrics, sort_keys=True))

    default_config = EloConfig()
    default_updates = FootballElo(default_config).process(matches)
    tuned_engine = FootballElo(tuned_config)
    tuned_updates = tuned_engine.process(matches)

    test_seasons = set(args.test_seasons)
    prior_validation = _baseline_metrics(matches, target_seasons=validation_seasons)
    prior_test = _baseline_metrics(matches, target_seasons=test_seasons)
    default_validation = _metrics_for_updates(default_updates, validation_seasons)
    default_test = _metrics_for_updates(default_updates, test_seasons)
    tuned_validation = _metrics_for_updates(tuned_updates, validation_seasons)
    tuned_test = _metrics_for_updates(tuned_updates, test_seasons)

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
            "validation_seasons": sorted(validation_seasons, key=season_start_year),
            "test_seasons": args.test_seasons,
            "current_season_excluded": "2026-27",
        },
        "load_stats": load_stats,
        "tuning": {
            "candidate_count": candidate_count,
            "selected_config": asdict(tuned_config),
            "validation_metrics": validation_metrics,
        },
        "metrics": {
            "validation": {
                "historical_prior": prior_validation,
                "default_football_elo": default_validation,
                "tuned_football_elo": tuned_validation,
            },
            "test": {
                "historical_prior": prior_test,
                "default_football_elo": default_test,
                "tuned_football_elo": tuned_test,
            },
            "tuned_by_season": _metrics_by_season(tuned_updates),
        },
    }

    output_dir: Path = args.output_dir
    output_dir.mkdir(parents=True, exist_ok=True)
    (output_dir / "metrics.json").write_text(json.dumps(metrics, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    write_features(output_dir / "elo_features.csv", tuned_updates, tuned_config)
    write_ratings(output_dir / f"ratings_as_of_{args.through}.csv", tuned_engine)
    print("Validation comparison: " + json.dumps(metrics["metrics"]["validation"], sort_keys=True))
    print("Test comparison: " + json.dumps(metrics["metrics"]["test"], sort_keys=True))
    print(f"Wrote model artifacts to {output_dir.resolve()}")
    return 0


if __name__ == "__main__":
    raise SystemExit(run(build_parser().parse_args()))
