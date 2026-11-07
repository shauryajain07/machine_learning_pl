"""Monte Carlo over lineup scenarios and lineup-effect parameter uncertainty.

Each draw samples both XIs and the lineup-effect coefficients from their
Laplace posterior, shifts the baseline Poisson rates, and builds the
existing Dixon–Coles score grid. Forecast probabilities are the mean of the
per-draw score grids, so home/draw/away always sum to one.
"""
from __future__ import annotations

from dataclasses import dataclass

import numpy as np

from src.matchsense.impact import LineupImpactModel, normalize_to_eleven
from src.matchsense.lineups import TeamLineupModel
from src.matchsense.schemas import ForecastBlock, OutcomeProbabilities, Scoreline
from src.ml.season_models import GOALS, outcomes


def block_from_grid(label: str, grid: np.ndarray, per_draw: np.ndarray | None = None) -> ForecastBlock:
    grid = grid / grid.sum()
    p = outcomes(grid[None])[0]
    p = p / p.sum()
    flat = np.argsort(grid.ravel())[::-1][:6]
    top = [Scoreline(home_goals=int(i // grid.shape[1]), away_goals=int(i % grid.shape[1]),
                     probability=round(float(grid.ravel()[i]), 4)) for i in flat]
    kwargs = {}
    if per_draw is not None and len(per_draw) > 1:
        kwargs = {"interval_home_win": (float(np.quantile(per_draw[:, 0], .05)), float(np.quantile(per_draw[:, 0], .95))),
                  "interval_away_win": (float(np.quantile(per_draw[:, 2], .05)), float(np.quantile(per_draw[:, 2], .95))),
                  "monte_carlo_se": float(per_draw[:, 0].std(ddof=1) / np.sqrt(len(per_draw)))}
    return ForecastBlock(label=label,
                         probabilities=OutcomeProbabilities(home_win=float(p[0]), draw=float(p[1]),
                                                            away_win=float(1 - p[0] - p[1])),
                         expected_goals_home=float((grid.sum(axis=1) * GOALS).sum()),
                         expected_goals_away=float((grid.sum(axis=0) * GOALS).sum()),
                         top_scorelines=top, **kwargs)


@dataclass
class SimulationResult:
    grid: np.ndarray
    per_draw: np.ndarray
    home_samples: np.ndarray
    away_samples: np.ndarray
    rates: np.ndarray
    scores: np.ndarray  # (n, 2) one scoreline drawn from each draw's grid


def sample_scores(grids: np.ndarray, rng: np.random.Generator) -> np.ndarray:
    """Draw one scoreline per score grid (inverse-CDF on the flattened grid)."""
    flat = grids.reshape(len(grids), -1)
    cdf = np.cumsum(flat, axis=1)
    cdf[:, -1] = 1.0
    idx = (cdf < rng.random((len(grids), 1))).sum(axis=1)
    size = grids.shape[-1]
    return np.column_stack((idx // size, idx % size))


def draws_payload(per_draw: np.ndarray, rates: np.ndarray, scores: np.ndarray, kind: str) -> dict:
    """Every Monte Carlo draw, compactly, for display and download."""
    outcome = np.where(scores[:, 0] > scores[:, 1], "H", np.where(scores[:, 0] == scores[:, 1], "D", "A"))
    return {"kind": kind, "n": int(len(per_draw)),
            "home_win": np.round(per_draw[:, 0], 4).tolist(), "draw": np.round(per_draw[:, 1], 4).tolist(),
            "away_win": np.round(per_draw[:, 2], 4).tolist(),
            "lambda_home": np.round(rates[:, 0], 4).tolist(), "lambda_away": np.round(rates[:, 1], 4).tolist(),
            "score_home": scores[:, 0].astype(int).tolist(), "score_away": scores[:, 1].astype(int).tolist(),
            "simulated_result_share": {k: float(np.mean(outcome == k)) for k in ("H", "D", "A")}}


def baseline_draws(grid: np.ndarray, rates: np.ndarray, n: int, seed: int) -> dict:
    """Without a lineup layer every draw shares one grid; scorelines still vary."""
    rng = np.random.default_rng(seed)
    grids = np.repeat(grid[None], n, axis=0)
    return draws_payload(np.repeat(outcomes(grid[None]), n, axis=0), np.repeat(np.atleast_2d(rates), n, axis=0),
                         sample_scores(grids, rng), "baseline")


def simulate(base_rates: np.ndarray, home: TeamLineupModel, away: TeamLineupModel, impact: LineupImpactModel,
             grid_fn, n: int, seed: int, parameter_uncertainty: bool = True) -> SimulationResult:
    rng = np.random.default_rng(seed)
    hs, as_ = home.sample(n, rng), away.sample(n, rng)
    if parameter_uncertainty and impact.active:
        cov = impact.b_cov + np.eye(len(impact.b)) * 1e-12
        thetas = rng.multivariate_normal(impact.b, cov, size=n, method="eigh")
    else:
        thetas = np.repeat(impact.b[None, :], n, axis=0)
    rates = np.empty((n, 2))
    for i in range(n):
        h_in, a_in = home.impact_inputs(hs[i]), away.impact_inputs(as_[i])
        rates[i, 0] = base_rates[0] * np.exp(impact.delta_eta(h_in, a_in, b=thetas[i]))
        rates[i, 1] = base_rates[1] * np.exp(impact.delta_eta(a_in, h_in, b=thetas[i]))
    rates = np.clip(rates, .08, 5.5)
    grids = grid_fn(rates)
    per_draw = outcomes(grids)
    return SimulationResult(grid=grids.mean(axis=0), per_draw=per_draw, home_samples=hs, away_samples=as_,
                            rates=rates, scores=sample_scores(grids, rng))


def expected_delta(home: TeamLineupModel, away: TeamLineupModel, impact: LineupImpactModel,
                   p_home: np.ndarray, p_away: np.ndarray) -> tuple[float, float]:
    """Log-rate shifts at expected starts (exact for the linear team features)."""
    gk_h, gk_a = home.positions == "GK", away.positions == "GK"
    sh, sa = normalize_to_eleven(p_home, gk_h), normalize_to_eleven(p_away, gk_a)
    h_in, a_in = home.impact_inputs(sh), away.impact_inputs(sa)
    return impact.delta_eta(h_in, a_in), impact.delta_eta(a_in, h_in)
