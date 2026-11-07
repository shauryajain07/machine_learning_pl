"""Bridge to the repository's existing, fitted Premier League models.

* Goal model: the deployed ``ScoreModel`` (recency-weighted Poisson regression
  with Dixon–Coles grid) from ``artifacts/season_simulator/model.joblib``.
* State: the bundled ``SeasonState`` at the end of 2025-26, rolled forward
  with completed current-season results via its own ``start_season`` and
  ``update`` transitions (date batches, so no intraday leakage).
* Comparison: the existing multinomial-logistic ``FittedModel`` recipe
  (C=0.1, 1460-day half life, as selected by the walk-forward comparison),
  refitted on the bundled training features. Walk-forward validation chose a
  0% logistic blend, so it is displayed as a comparison only.
"""
from __future__ import annotations

import copy
import csv
import threading
import warnings
from dataclasses import dataclass
from datetime import date, datetime, timedelta
from pathlib import Path

import joblib
import numpy as np

from src.matchsense.schemas import Fixture, FixtureStatus
from src.ml.season_data import Match
from src.ml.season_features import batches
from src.ml.season_models import Candidate, FittedModel, outcomes, reconcile, score_grid


# A result is treated as known only once the match could have finished.
RESULT_DELAY = timedelta(hours=2, minutes=30)


class BaselineUnavailable(RuntimeError):
    pass


@dataclass
class BaselinePrediction:
    rates: np.ndarray  # (home, away) Poisson rates
    grid: np.ndarray  # (21, 21) score probabilities
    probabilities: np.ndarray  # home, draw, away
    features: dict[str, float]
    results_used: int
    latest_result: str | None
    logistic: np.ndarray | None


def _load_bundle(path: Path):
    if not path.exists():
        raise BaselineUnavailable(f"Baseline model bundle not found at {path}. Run "
                                  "`python -m src.scripts.train_season_simulator --source local` first.")
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")  # sklearn minor-version pickle notices
        return joblib.load(path)


class BaselineEngine:
    def __init__(self, bundle_path: Path, training_dir: Path | None = None):
        bundle = _load_bundle(bundle_path)
        self.model = bundle["model"]
        self.state = bundle["state"]
        self.feature_names = list(bundle["feature_names"])
        self.trained_through = self.model.trained_through
        self.training_dir = training_dir or bundle_path.parent
        self._logistic = None
        self._lock = threading.Lock()

    @property
    def version(self) -> str:
        cfg = self.model.goal_spec
        return (f"season_simulator ScoreModel ({cfg.family}, alpha={cfg.complexity:g}, half-life={cfg.half_life}d, "
                f"rho={self.model.rho:g}, blend={self.model.blend:g}) trained through {self.trained_through}")

    def logistic_model(self):
        """Fit (once) the existing logistic FittedModel recipe on bundled training data."""
        with self._lock:
            if self._logistic is not None:
                return self._logistic or None
            features_csv = self.training_dir / "training_features.csv"
            matches_csv = self.training_dir / "historical_matches.csv"
            if not features_csv.exists() or not matches_csv.exists():
                self._logistic = False
                return None
            with matches_csv.open(newline="") as handle:
                outcome = {(r["season"], r["date"], r["home"], r["away"]): (int(r["hg"]), int(r["ag"]))
                           for r in csv.DictReader(handle)}
            rows, x = [], []
            with features_csv.open(newline="") as handle:
                for r in csv.DictReader(handle):
                    key = (r["season"], r["date"], r["home"], r["away"])
                    if key in outcome:
                        hg, ag = outcome[key]
                        rows.append(Match(r["season"], date.fromisoformat(r["date"]), r["home"], r["away"], hg, ag))
                        x.append([float(r[n]) for n in self.feature_names])
            self._logistic = FittedModel(Candidate("logistic", 1460, 0.1)).fit(np.asarray(x), rows)
            return self._logistic

    def state_for(self, season: str, fixtures: list[Fixture], target_date: date, as_of: datetime):
        """Copy the bundled state and apply results final before ``as_of`` and dated before ``target_date``."""
        year = int(season[:4])
        if self.state.year is None or self.state.year >= year:
            raise BaselineUnavailable(f"Bundled state ends in {self.state.year}; cannot roll into {season}")
        teams = {t for f in fixtures if f.season == season for t in (f.home, f.away)}
        unknown = teams - set(self.state.index)
        if unknown:
            raise BaselineUnavailable(f"No historical model state for {sorted(unknown)}")
        state = copy.deepcopy(self.state)
        state.start_season(year, teams)
        completed = [Match(season, f.kickoff_utc.date(), f.home, f.away, f.home_goals, f.away_goals, f.matchday)
                     for f in fixtures
                     if f.season == season and f.status == FixtureStatus.FINISHED and f.home_goals is not None
                     and f.kickoff_utc.date() < target_date and f.kickoff_utc + RESULT_DELAY <= as_of]
        for _, batch in batches(completed):
            batch = list(batch)
            state.update(batch, [[m.hg for m in batch]], [[m.ag for m in batch]])
        return state, completed

    def predict(self, fixture: Fixture, season_fixtures: list[Fixture], as_of: datetime) -> BaselinePrediction:
        if not fixture.competition.model_supported:
            raise BaselineUnavailable(f"No calibrated model for {fixture.competition.name}")
        state, completed = self.state_for(fixture.season, season_fixtures, fixture.kickoff_utc.date(), as_of)
        match = Match(fixture.season, fixture.kickoff_utc.date(), fixture.home, fixture.away, matchday=fixture.matchday)
        x = state.features([match])[0]
        expected = self.model.goal_model.scaler.n_features_in_
        if x.shape[1] < expected:
            raise BaselineUnavailable("Feature schema is narrower than the fitted model")
        x = x[:, :expected]
        rates = self.model.goal_model.predict(x)[0]
        grid = self.model.predict_grid(x)[0]
        logistic = None
        model = self.logistic_model()
        if model is not None:
            logistic = model.predict(x)[0]
        return BaselinePrediction(rates=rates, grid=grid, probabilities=outcomes(grid[None])[0],
                                  features=dict(zip(self.feature_names, x[0].tolist())),
                                  results_used=len(completed),
                                  latest_result=max((str(m.date) for m in completed), default=None),
                                  logistic=logistic)

    def grid_for_rates(self, rates: np.ndarray, x_for_blend: np.ndarray | None = None) -> np.ndarray:
        """Score grid for (adjusted) rates using the fitted rho and blend."""
        grid = score_grid(np.atleast_2d(rates), self.model.rho)
        if getattr(self.model, "outcome_model", None) is not None and self.model.blend > 0 and x_for_blend is not None:
            grid = reconcile(grid, self.model.outcome_model.predict(x_for_blend), self.model.blend)
        return grid
