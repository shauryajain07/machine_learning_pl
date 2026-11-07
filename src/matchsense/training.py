"""Leakage-safe training and walk-forward evaluation of the lineup layer.

Information sets (documented in docs/MODEL.md):

* ``baseline``     – the existing Poisson ScoreModel only (pre-season fit,
                      in-season form features from earlier dates).
* ``confirmed_xi`` – baseline plus the official starting XI, which is public
                      about one hour before kickoff.
* ``predicted_xi`` – baseline plus start probabilities predicted from earlier
                      selections only (what is known ~24h before kickoff,
                      minus news; no historical news archive exists locally).

Historical injury news is not available for 2025-26, so the news layer
itself cannot be backtested; only the lineup channel it feeds can.
"""
from __future__ import annotations

import csv
import warnings
from collections import defaultdict
from pathlib import Path

import joblib
import numpy as np

from src.matchsense.config import PROJECT_ROOT
from src.matchsense.impact import (PROVISIONAL_PRIOR, LineupImpactModel, SelectionModel, TeamObservation,
                                   normalize_to_eleven)
from src.matchsense.lineup_data import SelectionState, TeamMatch
from src.ml.season_data import Match
from src.ml.season_models import metrics, outcomes, score_grid

MIN_PRIOR_MATCHES = 3


def build_observations(team_matches: list[TeamMatch]) -> list[TeamObservation]:
    """Snapshot each side's lineup state *before* applying that match."""
    states: dict[str, SelectionState] = defaultdict(SelectionState)
    by_match: dict[str, list[TeamMatch]] = defaultdict(list)
    for tm in team_matches:
        by_match[tm.match_key].append(tm)
    observations = []
    for key in sorted(by_match, key=lambda k: by_match[k][0].kickoff):
        sides = by_match[key]
        if len(sides) != 2:
            continue
        pair = []
        for tm in sides:
            state = states[tm.team]
            pool = sorted(set(state.players) | set(tm.starters))
            typical = np.array([state.typical_rate(k) for k in pool])
            started = np.array([float(k in tm.starters) for k in pool])
            pair.append(TeamObservation(
                match_key=key, team=tm.team, opponent=tm.opponent, kickoff=tm.kickoff, matchday=tm.matchday,
                is_home=tm.is_home, goals_for=tm.goals_for, pool=pool, typical=typical, started=started,
                goal_share=state.goal_shares(pool), def_mask=state.defensive_mask(pool),
                selection_x=state.features(pool, tm.kickoff, goalkeepers={k for k in pool if state.players.get(k)
                                                                         and state.players[k].is_gk}),
                prior_matches=state.matches))
        pair[0].opponent_obs, pair[1].opponent_obs = pair[1], pair[0]
        observations.extend(pair)
        for tm in sides:  # Update only after both sides were snapshotted.
            states[tm.team].update(tm)
    return observations


def load_baseline_rates(observations: list[TeamObservation],
                        bundle_path: Path = PROJECT_ROOT / "artifacts/season_simulator/backtests/2025-26/model.joblib",
                        features_path: Path = PROJECT_ROOT / "artifacts/season_simulator/training_features.csv"):
    """Pre-season-fitted baseline rates for 2025-26 from the existing backtest model."""
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        bundle = joblib.load(bundle_path)
    model = bundle["model"]
    if model.trained_through >= "2025-08-01":
        raise ValueError("Backtest baseline must be fitted before the 2025-26 season")
    names = list(bundle["feature_names"])
    rows = {}
    with Path(features_path).open(newline="") as handle:
        for row in csv.DictReader(handle):
            if row["season"] == "2025-26":
                rows[f"{row['season']}|{row['date']}|{row['home']}|{row['away']}"] = \
                    np.array([float(row[n]) for n in names])
    keys = sorted({o.match_key for o in observations})
    missing = [k for k in keys if k not in rows]
    if missing:
        raise ValueError(f"No baseline features for {len(missing)} matches, e.g. {missing[:2]}")
    x = np.vstack([rows[k] for k in keys])
    rates = model.goal_model.predict(x)
    lookup = {k: r for k, r in zip(keys, rates)}
    offsets = np.array([np.log(lookup[o.match_key][0 if o.is_home else 1]) for o in observations])
    return offsets, {"baseline_bundle": str(bundle_path.relative_to(PROJECT_ROOT)),
                     "baseline_trained_through": model.trained_through, "rho": float(getattr(model, "rho", 0.0))}


def match_rows(observations, eta):
    """Home-perspective (Match, rates) per match from per-side log rates."""
    by_key = {}
    for obs, e in zip(observations, eta):
        by_key.setdefault(obs.match_key, {})[obs.is_home] = (obs, e)
    result = []
    for key, sides in by_key.items():
        if True in sides and False in sides:
            home, away = sides[True], sides[False]
            date = home[0].kickoff.date()
            match = Match("2025-26", date, home[0].team, away[0].team, home[0].goals_for, away[0].goals_for,
                          home[0].matchday)
            result.append((match, np.exp([home[1], away[1]])))
    return result


def evaluate_rates(pairs, rho=0.0):
    matches = [m for m, _ in pairs]
    rates = np.vstack([r for _, r in pairs])
    grid = score_grid(np.clip(rates, .08, 5.5), rho)
    result = metrics(matches, outcomes(grid), grid)
    per_match = -np.log(np.maximum(outcomes(grid)[np.arange(len(matches)), [m.outcome for m in matches]], 1e-15))
    return result, per_match


# (name, continuity, goal_share, players, alpha_team, alpha_players). Order is
# simplest first; the one-standard-error rule prefers earlier entries.
ROUND1 = [("baseline", False, False, False, 0.0, 0.0)] + \
    [(f"continuity_a{a:g}", True, False, False, a, 0.0) for a in (1, 5, 20, 80)] + \
    [(f"continuity_players_a{a:g}_p{p:g}", True, False, True, a, p) for a in (5, 20) for p in (10, 30, 100, 300)]
# Added after round 1 showed player-level effects overfitting (disclosed in REPORT.md).
ROUND2 = [(f"continuity_goalshare_a{a:g}", True, True, False, a, 0.0) for a in (5, 20, 80)]
# Round 3 (added on request, after rounds 1-2 were seen): opponent goalkeeper/back-line discontinuity.
ROUND3 = [(f"continuity_defence_a{a:g}", True, False, False, a, 0.0, True) for a in (5, 20, 80)] + \
    [(f"continuity_goalshare_defence_a{a:g}", True, True, False, a, 0.0, True) for a in (20, 80)]
CANDIDATES = ROUND1[:5] + ROUND2 + ROUND3 + ROUND1[5:]
# Deployed when no empirical candidate validates. Reported, never "selected".
PROVISIONAL = ("provisional_prior", True, True, False, 0.0, 0.0, True, PROVISIONAL_PRIOR)


def build_model(candidate) -> LineupImpactModel:
    _, cont, goals, players, alpha_team, alpha_players, *rest = candidate
    defence = next((r for r in rest if isinstance(r, bool)), False)
    prior = next((r for r in rest if isinstance(r, dict)), None)
    return LineupImpactModel(cont, goals, players, alpha_team, alpha_players, prior=prior, use_defence=defence)


def complexity(candidate) -> int:
    _, cont, goals, players = candidate[:4]
    defence = any(r is True for r in candidate[6:])
    return 2 * cont + 2 * goals + int(defence) + (1000 if players else 0)


def walk_forward(observations, offsets, eligible, first_fold=8, validation_last=22, rho=0.0, candidates=CANDIDATES):
    """Matchday folds; each fold trains only on matches kicked off earlier."""
    folds = sorted({o.matchday for o in observations if o.matchday and o.matchday >= first_fold})
    store = {name: {"validation": ([], []), "test": ([], [])} for name, *_ in candidates}
    obs_index = {id(o): i for i, o in enumerate(observations)}
    audit = []
    for fold in folds:
        test = [o for o in observations if o.matchday == fold and eligible[obs_index[id(o)]]
                and eligible[obs_index[id(o.opponent_obs)]]]
        if not test:
            continue
        cutoff = min(o.kickoff for o in test)
        train_idx = [i for i, o in enumerate(observations) if o.kickoff < cutoff and eligible[i]]
        train = [observations[i] for i in train_idx]
        if max(o.kickoff for o in train) >= cutoff:
            raise AssertionError("Training fold contains a match at or after the prediction cutoff")
        audit.append({"fold_matchday": fold, "train_observations": len(train), "test_matches": len(test) // 2,
                      "train_last_kickoff": max(o.kickoff for o in train).isoformat(),
                      "fold_first_kickoff": cutoff.isoformat()})
        phase = "validation" if fold <= validation_last else "test"
        test_offsets = offsets[[obs_index[id(o)] for o in test]]
        for candidate in candidates:
            name = candidate[0]
            model = build_model(candidate).fit(train, offsets[train_idx])
            eta = model.predict_offsets(test, test_offsets) if model.active else test_offsets
            for pair in match_rows(test, eta):
                store[name][phase][0].append(pair)
    summary = {}
    for name in store:
        summary[name] = {}
        for phase in ("validation", "test"):
            pairs = store[name][phase][0]
            if pairs:
                result, per_match = evaluate_rates(pairs, rho)
                summary[name][phase] = {k: v for k, v in result.items() if k != "calibration"}
                summary[name][phase]["calibration"] = result["calibration"]
                summary[name][phase]["_per_match"] = per_match
                summary[name][phase]["_keys"] = [m.home + "|" + m.away + "|" + str(m.date) for m, _ in pairs]
    return summary, store, audit


def selection_dataset(observations, eligible):
    X, y, groups, kickoffs = [], [], [], []
    for i, obs in enumerate(observations):
        if not eligible[i]:
            continue
        X.append(obs.selection_x)
        y.append(obs.started)
        groups.extend([i] * len(obs.pool))
        kickoffs.extend([obs.kickoff] * len(obs.pool))
    return np.vstack(X), np.concatenate(y), np.array(groups), np.array(kickoffs)


def predicted_starts(model: SelectionModel, obs: TeamObservation) -> np.ndarray:
    p = model.predict(obs.selection_x)
    gk = obs.selection_x[:, 7] > 0.5
    return normalize_to_eleven(p, gk if gk.sum() >= 1 else None)


def one_se_choice(summary, candidates):
    """Simplest candidate within one paired standard error of the best (validation)."""
    names = {c[0] for c in candidates}
    best = min(names, key=lambda n: summary[n]["validation"]["log_loss"])
    best_loss = summary[best]["validation"]["_per_match"]
    eligible = []
    for cand in candidates:
        name = cand[0]
        diff = summary[name]["validation"]["_per_match"] - best_loss
        se = diff.std(ddof=1) / np.sqrt(len(diff)) if name != best else 0.0
        if diff.mean() <= se:
            eligible.append((complexity(cand), candidates.index(cand), name, float(diff.mean()), float(se)))
    eligible.sort()
    return best, eligible[0][2], [{"candidate": e[2], "mean_diff_vs_best": e[3], "se": e[4]} for e in eligible]


def paired_bootstrap(a: np.ndarray, b: np.ndarray, n=4000, seed=7):
    """95% interval for mean(a - b) over matches (negative => a better)."""
    rng = np.random.default_rng(seed)
    diff = a - b
    samples = rng.choice(diff, size=(n, len(diff)), replace=True).mean(axis=1)
    return float(diff.mean()), float(np.quantile(samples, .025)), float(np.quantile(samples, .975))
