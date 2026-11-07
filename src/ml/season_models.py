"""Recency-weighted models, coherent score distributions, and diagnostics."""
from __future__ import annotations

from dataclasses import asdict, dataclass
import numpy as np
from scipy.stats import poisson
from sklearn.ensemble import (HistGradientBoostingClassifier, HistGradientBoostingRegressor,
                              RandomForestClassifier, ExtraTreesClassifier)
from sklearn.linear_model import LogisticRegression, PoissonRegressor
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import log_loss

MAX_GOALS = 20  # Poisson tail at maximum permitted rate 5.5 is below 4e-7.
GOALS = np.arange(MAX_GOALS + 1)
OUTCOME = np.where(GOALS[:, None] > GOALS[None, :], 0,
                   np.where(GOALS[:, None] == GOALS[None, :], 1, 2))


def recency_weights(matches, cutoff, half_life_days):
    if half_life_days is not None and half_life_days <= 0:
        raise ValueError('Half life must be positive')
    ages = np.array([(cutoff - m.date).days for m in matches])
    if np.any(ages < 0):
        raise ValueError('Training data includes a match after fit cutoff')
    if half_life_days is None:
        return np.ones(len(matches))
    w = np.exp2(-ages / half_life_days)
    return w / w.mean()  # Keep regularization comparable between half lives.


@dataclass(frozen=True)
class Candidate:
    family: str
    half_life: int | None
    complexity: float
    matchday_ratio: float = 1.

    @property
    def name(self):
        return f'{self.family}_half{self.half_life}_size{self.complexity:g}_md{self.matchday_ratio:g}'


@dataclass(frozen=True)
class DecisionPolicy:
    """Top-1 pick policy kept separate from probability generation.

    A score simulator should retain calibrated probabilities, while a match
    picker may need a draw correction for exact home/draw/away accuracy.  The
    multiplier is selected on validation seasons only and does not alter the
    score distribution used by Monte Carlo simulation.
    """

    draw_multiplier: float = 1.0

    def __post_init__(self):
        if not np.isfinite(self.draw_multiplier) or self.draw_multiplier <= 0:
            raise ValueError('draw_multiplier must be finite and positive')

    def predict(self, probabilities):
        p = np.asarray(probabilities, dtype=float)
        if p.ndim != 2 or p.shape[1] != 3:
            raise ValueError('Expected an (n, 3) outcome probability matrix')
        adjusted = p * np.array([1., self.draw_multiplier, 1.])
        return adjusted.argmax(axis=1)

    def config(self):
        return {'draw_multiplier': float(self.draw_multiplier)}


def candidates():
    return [Candidate(family, half, size)
            for half in (365, 730, 1460)
            for family, sizes in [('poisson', (.03, .3)), ('poisson_trees', (7, 15)),
                                  ('logistic', (.1, 1.0)), ('outcome_trees', (7, 15))]
            for size in sizes]


class FittedModel:
    def __init__(self, spec):
        self.spec = spec
        self.scaler = StandardScaler()
        self.models = []

    def fit(self, x, matches):
        w = recency_weights(matches, max(m.date for m in matches), self.spec.half_life)
        ratio = getattr(self.spec, 'matchday_ratio', 1.)
        if ratio < 1:
            raise ValueError('Matchday weight ratio must be at least one')
        if ratio != 1:
            if any(m.matchday is None or not 1 <= m.matchday <= 38 for m in matches):
                raise ValueError('Matchday weighting requires valid scheduled matchdays on all training rows')
            w *= np.array([ratio ** ((m.matchday - 1) / 37) for m in matches])
            w /= w.mean()
        self.effective_sample_size = float(w.sum() ** 2 / (w @ w))
        z = self.scaler.fit_transform(x, sample_weight=w)
        f, c = self.spec.family, self.spec.complexity
        tree = dict(max_iter=140, learning_rate=.045, max_leaf_nodes=int(c),
                    min_samples_leaf=60, l2_regularization=5., early_stopping=False,
                    random_state=202627)
        if f in ('poisson', 'poisson_trees'):
            for y in ([m.hg for m in matches], [m.ag for m in matches]):
                model = (PoissonRegressor(alpha=c, max_iter=500) if f == 'poisson'
                         else HistGradientBoostingRegressor(loss='poisson', **tree))
                model.fit(z, y, sample_weight=w)
                self.models.append(model)
        else:
            if f == 'logistic':
                model = LogisticRegression(C=c, max_iter=1000)
            elif f in ('random_forest', 'extra_trees'):
                cls = RandomForestClassifier if f == 'random_forest' else ExtraTreesClassifier
                model = cls(n_estimators=250, min_samples_leaf=int(c), max_features=.8,
                            max_depth=12, n_jobs=1, random_state=202627)
            else:
                model = HistGradientBoostingClassifier(**tree)
            model.fit(z, [m.outcome for m in matches], sample_weight=w)
            if list(model.classes_) != [0, 1, 2]:
                raise ValueError('Training data must include home, draw and away outcomes')
            self.models.append(model)
        return self

    def predict(self, x):
        z = self.scaler.transform(x)
        if self.spec.family in ('poisson', 'poisson_trees'):
            return np.clip(np.column_stack([m.predict(z) for m in self.models]), .08, 5.5)
        return self.models[0].predict_proba(z)


def score_grid(rates, rho=0.):
    """Poisson scores with a Dixon–Coles low-score correction.

    Project rho into each fixture's admissible interval to preserve positive
    cells; the four corrections preserve Poisson marginals before truncation.
    """
    rates = np.asarray(rates)
    if rates.ndim != 2 or rates.shape[1] != 2 or np.any(~np.isfinite(rates)) or np.any(rates <= 0):
        raise ValueError('Expected finite, positive home/away goal rates')
    h, a = rates.T
    r = np.clip(rho, -.999 / np.maximum(h, a), np.minimum(.999, .999 / (h * a)))
    p = poisson.pmf(GOALS[None, :], h[:, None])[:, :, None] * poisson.pmf(GOALS[None, :], a[:, None])[:, None, :]
    p[:, 0, 0] *= 1 - h * a * r
    p[:, 0, 1] *= 1 + h * r
    p[:, 1, 0] *= 1 + a * r
    p[:, 1, 1] *= 1 - r
    return p / p.sum(axis=(1, 2), keepdims=True)


def outcomes(grid):
    return np.column_stack([grid[:, OUTCOME == k].sum(axis=1) for k in range(3)])


def reconcile(grid, classification, weight):
    base = outcomes(grid)
    target = (1 - weight) * base + weight * classification
    result = grid * (target / np.maximum(base, 1e-15))[:, OUTCOME]
    return result / result.sum(axis=(1, 2), keepdims=True)


class ScoreModel:
    def __init__(self, goal_spec, outcome_spec, rho=0., blend=0., decision_policy=None):
        self.goal_spec, self.outcome_spec = goal_spec, outcome_spec
        self.rho, self.blend = rho, blend
        self.decision_policy = decision_policy or DecisionPolicy()

    def fit(self, x, matches):
        self.goal_model = FittedModel(self.goal_spec).fit(x, matches)
        self.outcome_model = FittedModel(self.outcome_spec).fit(x, matches) if self.blend > 0 else None
        self.trained_through = max(m.date for m in matches).isoformat()
        return self

    def predict_grid(self, x):
        # Backward-compatible append-only feature schema: v1 had 19 features;
        # v2 adds the known scheduled matchday as the last feature.
        count = self.goal_model.scaler.n_features_in_
        if x.shape[1] == count + 1 and count == 19:
            x = x[:, :count]
        grid = score_grid(self.goal_model.predict(x), self.rho)
        if self.outcome_model is not None:
            grid = reconcile(grid, self.outcome_model.predict(x), self.blend)
        return grid

    def predict_outcomes(self, x):
        """Return top-1 picks under the validation-selected decision policy."""
        return self.decision_policy.predict(outcomes(self.predict_grid(x)))

    def config(self):
        return {'goals': asdict(self.goal_spec), 'outcomes': asdict(self.outcome_spec),
                'rho': self.rho, 'blend': self.blend,
                'decision_policy': self.decision_policy.config()}


def metrics(matches, probabilities, grid=None, decision_policy=None):
    y = np.array([m.outcome for m in matches])
    p = np.asarray(probabilities)
    onehot = np.eye(3)[y]
    raw_predictions = p.argmax(axis=1)
    predictions = (raw_predictions if decision_policy is None
                   else decision_policy.predict(p))
    raw_accuracy = float(np.mean(raw_predictions == y))
    accuracy = (raw_accuracy if decision_policy is None else
                float(np.mean(predictions == y)))
    class_f1 = []
    class_recall = []
    for label in range(3):
        true_positive = np.sum((predictions == label) & (y == label))
        precision = true_positive / max(np.sum(predictions == label), 1)
        recall = true_positive / max(np.sum(y == label), 1)
        class_f1.append(2 * precision * recall / max(precision + recall, 1e-15))
        class_recall.append(recall)
    draw_precision = float(np.sum((predictions == 1) & (y == 1)) /
                          max(np.sum(predictions == 1), 1))
    draw_recall = float(np.sum((predictions == 1) & (y == 1)) /
                        max(np.sum(y == 1), 1))
    result = {'matches': len(y), 'log_loss': float(log_loss(y, p, labels=[0, 1, 2])),
              'brier': float(np.mean(np.sum((p - onehot) ** 2, axis=1))),
              'accuracy': accuracy, 'raw_accuracy': raw_accuracy,
              'macro_f1': float(np.mean(class_f1)),
              'balanced_accuracy': float(np.mean(class_recall)),
              'draw_f1': float(class_f1[1]),
              'draw_precision': draw_precision,
              'draw_recall': draw_recall,
              'predicted_draw_rate': float(np.mean(predictions == 1)),
              'actual_draw_rate': float(np.mean(y == 1)),
              'ranked_probability_score': float(np.mean(np.sum((np.cumsum(p, axis=1)[:, :2] - np.cumsum(onehot, axis=1)[:, :2]) ** 2, axis=1) / 2))}
    if decision_policy is not None:
        result['decision_policy'] = decision_policy.config()
    if grid is not None:
        hg, ag = np.array([m.hg for m in matches]), np.array([m.ag for m in matches])
        if np.any(hg > MAX_GOALS) or np.any(ag > MAX_GOALS):
            raise ValueError('Observed score exceeds configured score support')
        result['score_nll'] = float(-np.log(np.maximum(grid[np.arange(len(y)), hg, ag], 1e-15)).mean())
        eh = (grid.sum(axis=2) * GOALS).sum(axis=1)
        ea = (grid.sum(axis=1) * GOALS).sum(axis=1)
        result['goal_mae'] = float(np.abs(np.column_stack((eh - hg, ea - ag))).mean())
    calibration = []
    for k, label in enumerate(('home', 'draw', 'away')):
        for low in np.arange(0, 1, .1):
            mask = (p[:, k] >= low) & (p[:, k] < low + .1 + (1e-10 if low > .89 else 0))
            if mask.any():
                calibration.append({'outcome': label, 'lower': round(float(low), 1), 'count': int(mask.sum()),
                                    'predicted': float(p[mask, k].mean()), 'observed': float((y[mask] == k).mean())})
    result['calibration'] = calibration
    return result
