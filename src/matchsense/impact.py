"""Lineup-impact model: how a starting XI shifts the baseline goal rates.

For team t scoring against opponent o::

    goals_t ~ Poisson( lambda_base_t * exp( b1 D_t + b2 D_o + b3 G_t + b4 G_o
                                           + sum_i a_i x_i  + sum_j d_j x_j ) )

* ``lambda_base`` is the existing ScoreModel's Poisson rate (an offset, so the
  fitted Elo/form/rest model is kept intact);
* ``D`` is lineup *discontinuity*: the share of the team's usual starting
  mass (exponentially weighted start rates from earlier matches) that is not
  in this XI;
* ``G`` is the share of the team's recent (decayed) goals scored by players
  who are not in this XI, a pre-match proxy for lost attacking output;
* ``x_i = started_i - typical_rate_i`` for each player of the team (``a_i``
  = attacking effect on own goals) and of the opponent (``d_i`` = effect on
  goals conceded by the player's team, negative is better).

Coefficients are ridge-penalised (Gaussian priors centred on zero), so with
little evidence a player's effect shrinks to nothing and the forecast falls
back to the baseline. Uncertainty uses a Laplace approximation of the
posterior. Hyperparameters are chosen by chronological walk-forward
validation in ``src/scripts/train_lineup_impact.py``.
"""
from __future__ import annotations

import json
from dataclasses import dataclass, field
from pathlib import Path

import numpy as np
from scipy.optimize import minimize
from sklearn.linear_model import LogisticRegression
from sklearn.preprocessing import StandardScaler

from src.matchsense.lineup_data import SELECTION_FEATURES


@dataclass
class TeamObservation:
    """One team's side of one match, with prior-only lineup state."""

    match_key: str
    team: str
    opponent: str
    kickoff: object
    matchday: int | None
    is_home: bool
    goals_for: int
    pool: list[str]
    typical: np.ndarray
    started: np.ndarray
    selection_x: np.ndarray
    goal_share: np.ndarray
    prior_matches: int
    opponent_obs: "TeamObservation | None" = None
    def_mask: np.ndarray | None = None  # Goalkeeper/back-line regulars (prior matches only).


def discontinuity(typical: np.ndarray, starts: np.ndarray) -> float:
    """Share of the usual top-11 starting mass absent from this XI (0..1)."""
    top = np.sort(typical)[::-1][:11].sum()
    if top <= 1e-9:
        return 0.0
    return float(np.clip(1 - (typical * starts).sum() / top, 0, 1))


def normalize_to_eleven(p: np.ndarray, gk_mask: np.ndarray | None = None) -> np.ndarray:
    """Scale start probabilities to sum to 11 (one goalkeeper) with a cap at 1."""
    p = np.clip(np.asarray(p, dtype=float), 0, 1)
    out = np.zeros_like(p)
    groups = [(gk_mask, 1.0), (~gk_mask, 10.0)] if gk_mask is not None and gk_mask.any() else [(np.ones_like(p, bool), 11.0)]
    for mask, target in groups:
        q = p[mask].copy()
        if q.sum() <= 0:
            q = np.ones_like(q)
        for _ in range(50):
            free = q < 1 - 1e-12
            deficit = target - q.sum()
            if abs(deficit) < 1e-9 or not free.any():
                break
            q[free] *= 1 + deficit / max(q[free].sum(), 1e-12)
            q = np.minimum(q, 1)
        out[mask] = q
    return out


def fit_offset_poisson(X: np.ndarray, y: np.ndarray, offset: np.ndarray, penalty: np.ndarray,
                       prior_mean: np.ndarray | None = None):
    """MAP Poisson GLM with offset and Gaussian prior N(prior_mean, 1/penalty).

    Returns (coef, Laplace covariance).
    """
    centre = np.zeros(X.shape[1]) if prior_mean is None else prior_mean

    def objective(theta):
        eta = offset + X @ theta
        mu = np.exp(np.clip(eta, -20, 5))
        value = (mu - y * eta).sum() + 0.5 * (penalty * (theta - centre) ** 2).sum()
        grad = X.T @ (mu - y) + penalty * (theta - centre)
        return value, grad

    theta0 = centre.copy()
    result = minimize(objective, theta0, jac=True, method="L-BFGS-B", options={"maxiter": 2000})
    theta = result.x
    mu = np.exp(np.clip(offset + X @ theta, -20, 5))
    hessian = (X * mu[:, None]).T @ X + np.diag(penalty)
    covariance = np.linalg.inv(hessian)
    return theta, covariance


TEAM_FEATURES = ("own_discontinuity", "opp_discontinuity", "own_goal_share_absent", "opp_goal_share_absent",
                 "opp_defensive_discontinuity")
K = len(TEAM_FEATURES)
DEFENSIVE_UNIT = 5  # Goalkeeper plus a back four.

# Provisional prior (docs/MODEL.md). Conservative domain assumptions, NOT fitted:
# losing all of a team's recent scorers lowers its scoring rate by ~22%
# (replacements recover most output); a fully changed XI lowers it ~14%;
# a rotated opponent concedes slightly more; an opponent missing its whole
# usual goalkeeper + back line concedes ~22% more. Standard deviations are wide.
PROVISIONAL_PRIOR = {"mean": [-0.15, 0.10, -0.25, 0.0, 0.20], "sd": [0.15, 0.15, 0.15, 0.10, 0.15]}


def defensive_discontinuity(typical: np.ndarray, starts: np.ndarray, def_mask: np.ndarray | None) -> float:
    """Share of the usual goalkeeper + back-line starting mass absent from this XI."""
    if def_mask is None or not np.any(def_mask):
        return 0.0
    t, s = typical[def_mask], starts[def_mask]
    top = np.sort(t)[::-1][:DEFENSIVE_UNIT].sum()
    return 0.0 if top <= 1e-9 else float(np.clip(1 - (t * s).sum() / top, 0, 1))


def team_vector(own_typical, own_starts, own_goal_share, opp_typical, opp_starts, opp_goal_share,
                opp_def_mask=None) -> np.ndarray:
    """[D_own, D_opp, G_own, G_opp, Ddef_opp] for one team's scoring rate."""
    return np.array([discontinuity(own_typical, own_starts), discontinuity(opp_typical, opp_starts),
                     float(np.dot(own_goal_share, 1 - own_starts)), float(np.dot(opp_goal_share, 1 - opp_starts)),
                     defensive_discontinuity(opp_typical, opp_starts, opp_def_mask)])


@dataclass
class LineupImpactModel:
    use_continuity: bool
    use_goal_share: bool
    use_players: bool
    alpha_team: float
    alpha_players: float
    b: np.ndarray = field(default_factory=lambda: np.zeros(K))
    b_cov: np.ndarray = field(default_factory=lambda: np.zeros((K, K)))
    attack: dict[str, float] = field(default_factory=dict)
    defence: dict[str, float] = field(default_factory=dict)
    attack_var: dict[str, float] = field(default_factory=dict)
    defence_var: dict[str, float] = field(default_factory=dict)
    meta: dict = field(default_factory=dict)
    prior: dict | None = None  # {"mean": [4], "sd": [4]} => MAP with informative prior.
    # Training means of the team features: effects are deviations from a
    # typical amount of rotation, so an ordinary XI leaves the baseline intact.
    center: np.ndarray = field(default_factory=lambda: np.zeros(K))
    use_defence: bool = False  # Opponent goalkeeper/back-line discontinuity term.

    @property
    def active(self) -> bool:
        return self.use_continuity or self.use_goal_share or self.use_players or self.use_defence

    @property
    def team_mask(self) -> np.ndarray:
        return np.array([self.use_continuity] * 2 + [self.use_goal_share] * 2 + [self.use_defence])

    @property
    def n_parameters(self) -> int:
        return int(self.team_mask.sum()) + (2 * len(self.attack) if self.use_players else 0)

    # -- design ------------------------------------------------------------
    def _rows(self, observations: list[TeamObservation], keys: dict[str, int], starts_override=None):
        n_players = len(keys) if self.use_players else 0
        X = np.zeros((len(observations), K + 2 * n_players))
        for r, obs in enumerate(observations):
            opp = obs.opponent_obs
            s_own = obs.started if starts_override is None else starts_override[id(obs)]
            s_opp = opp.started if starts_override is None else starts_override[id(opp)]
            X[r, :K] = (team_vector(obs.typical, s_own, obs.goal_share, opp.typical, s_opp, opp.goal_share,
                                    opp.def_mask) - self.center) * self.team_mask
            if self.use_players:
                for key, x in zip(obs.pool, s_own - obs.typical):
                    if key in keys:
                        X[r, K + keys[key]] = x
                for key, x in zip(opp.pool, s_opp - opp.typical):
                    if key in keys:
                        X[r, K + n_players + keys[key]] = x
        return X

    def fit(self, observations: list[TeamObservation], offsets: np.ndarray):
        if not self.active:
            return self
        y = np.array([o.goals_for for o in observations], dtype=float)
        players = sorted({k for o in observations for k in o.pool}) if self.use_players else []
        keys = {k: i for i, k in enumerate(players)}
        self.center = np.zeros(K)
        raw = np.array([team_vector(o.typical, o.started, o.goal_share, o.opponent_obs.typical,
                                    o.opponent_obs.started, o.opponent_obs.goal_share, o.opponent_obs.def_mask)
                        for o in observations])
        self.center = raw.mean(axis=0) * self.team_mask
        X = self._rows(observations, keys)
        if self.prior:
            team_penalty = np.where(self.team_mask, 1 / np.square(self.prior["sd"]), 1e8)
            team_mean = np.where(self.team_mask, self.prior["mean"], 0.0)
        else:
            team_penalty, team_mean = np.where(self.team_mask, self.alpha_team, 1e8), np.zeros(K)
        penalty = np.concatenate([team_penalty, np.full(2 * len(players), self.alpha_players)])
        prior_mean = np.concatenate([team_mean, np.zeros(2 * len(players))])
        theta, cov = fit_offset_poisson(X, y, offsets, penalty, prior_mean)
        self.b, self.b_cov = theta[:K] * self.team_mask, cov[:K, :K] * np.outer(self.team_mask, self.team_mask)
        n = len(players)
        self.attack = {k: float(theta[K + i]) for k, i in keys.items()}
        self.defence = {k: float(theta[K + n + i]) for k, i in keys.items()}
        self.attack_var = {k: float(cov[K + i, K + i]) for k, i in keys.items()}
        self.defence_var = {k: float(cov[K + n + i, K + n + i]) for k, i in keys.items()}
        self.meta.update({"training_observations": len(observations), "players": n,
                          **{f"b_{name}": float(v) for name, v in zip(TEAM_FEATURES, self.b)}})
        return self

    # -- prediction ----------------------------------------------------------
    def delta_eta(self, own: dict, opp: dict, b=None, attack=None, defence=None) -> float:
        """Change in log scoring rate for ``own`` given both XIs.

        ``own``/``opp`` hold aligned arrays ``pool``, ``typical``, ``starts``,
        ``goal_share`` and ``def_mask``.
        """
        if not self.active:
            return 0.0
        b = self.b if b is None else b
        value = float(b @ (team_vector(own["typical"], own["starts"], own["goal_share"],
                                       opp["typical"], opp["starts"], opp["goal_share"], opp.get("def_mask"))
                           - self.center))
        if self.use_players:
            attack = self.attack if attack is None else attack
            defence = self.defence if defence is None else defence
            value += sum(attack.get(k, 0.0) * x for k, x in zip(own["pool"], own["starts"] - own["typical"]))
            value += sum(defence.get(k, 0.0) * x for k, x in zip(opp["pool"], opp["starts"] - opp["typical"]))
        return value

    def predict_offsets(self, observations: list[TeamObservation], offsets: np.ndarray, starts_override=None):
        if not self.active:
            return offsets
        players = sorted(self.attack) if self.use_players else []
        keys = {k: i for i, k in enumerate(players)}
        X = self._rows(observations, keys, starts_override)
        theta = np.concatenate([self.b, [self.attack[k] for k in players], [self.defence[k] for k in players]])
        return offsets + X @ theta

    # -- persistence -----------------------------------------------------------
    def to_dict(self) -> dict:
        return {"use_continuity": self.use_continuity, "use_goal_share": self.use_goal_share,
                "use_players": self.use_players, "alpha_team": self.alpha_team, "alpha_players": self.alpha_players,
                "team_features": list(TEAM_FEATURES), "b": self.b.tolist(), "b_cov": self.b_cov.tolist(),
                "attack": self.attack, "defence": self.defence, "attack_var": self.attack_var,
                "defence_var": self.defence_var, "meta": self.meta, "prior": self.prior, "use_defence": self.use_defence,
                "center": self.center.tolist()}

    @classmethod
    def from_dict(cls, data: dict) -> "LineupImpactModel":
        def pad(vector):  # Artifacts trained before the defensive term have 4 features.
            v = np.asarray(vector, dtype=float)
            return np.concatenate([v, np.zeros(K - len(v))]) if v.ndim == 1 else \
                np.pad(v, ((0, K - v.shape[0]), (0, K - v.shape[1])))
        return cls(data["use_continuity"], data["use_goal_share"], data["use_players"], data["alpha_team"],
                   data["alpha_players"], pad(data["b"]), pad(data["b_cov"]), data["attack"],
                   data["defence"], data["attack_var"], data["defence_var"], data.get("meta", {}), data.get("prior"),
                   pad(data.get("center", [0.0] * K)), data.get("use_defence", False))


@dataclass
class SelectionModel:
    """Pooled logistic model of P(start | available) from prior selection history."""

    C: float = 1.0
    scaler: StandardScaler | None = None
    model: LogisticRegression | None = None

    def fit(self, X: np.ndarray, y: np.ndarray):
        self.scaler = StandardScaler().fit(X)
        self.model = LogisticRegression(C=self.C, max_iter=2000).fit(self.scaler.transform(X), y)
        return self

    def predict(self, X: np.ndarray) -> np.ndarray:
        if len(X) == 0:
            return np.zeros(0)
        if self.model is None:  # Unfitted fallback: typical start rate.
            return np.clip(X[:, 0], 0.01, 0.99)
        return self.model.predict_proba(self.scaler.transform(X))[:, 1]

    def to_dict(self) -> dict:
        return {"C": self.C, "features": list(SELECTION_FEATURES),
                "mean": self.scaler.mean_.tolist(), "scale": self.scaler.scale_.tolist(),
                "coef": self.model.coef_[0].tolist(), "intercept": float(self.model.intercept_[0])}

    @classmethod
    def from_dict(cls, data: dict) -> "SelectionModel":
        if list(data["features"]) != list(SELECTION_FEATURES):
            raise ValueError("Selection model feature schema mismatch")
        result = cls(C=data["C"])
        result.scaler = StandardScaler()
        result.scaler.mean_ = np.asarray(data["mean"])
        result.scaler.scale_ = np.asarray(data["scale"])
        result.scaler.var_ = result.scaler.scale_ ** 2
        result.scaler.n_features_in_ = len(data["mean"])
        result.model = LogisticRegression(C=data["C"])
        result.model.coef_ = np.asarray([data["coef"]])
        result.model.intercept_ = np.asarray([data["intercept"]])
        result.model.classes_ = np.array([0, 1])
        result.model.n_features_in_ = len(data["coef"])
        return result


@dataclass
class ImpactArtifact:
    impact: LineupImpactModel
    selection: SelectionModel
    status: str  # "validated", "not_confirmed_out_of_sample", "baseline_only"
    names: dict[str, str]
    report: dict

    def save(self, path: Path):
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_text(json.dumps({"impact": self.impact.to_dict(), "selection": self.selection.to_dict(),
                                    "status": self.status, "names": self.names, "report": self.report},
                                   indent=1, default=float))

    @classmethod
    def load(cls, path: Path) -> "ImpactArtifact":
        data = json.loads(Path(path).read_text())
        return cls(LineupImpactModel.from_dict(data["impact"]), SelectionModel.from_dict(data["selection"]),
                   data["status"], data["names"], data["report"])
