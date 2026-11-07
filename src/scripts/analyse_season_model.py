"""Validation-only feature ablations, data summaries, and held-out diagnostics.

This never changes the selected model. Run after train_season_simulator.
"""
import argparse
import csv
from dataclasses import replace
from pathlib import Path
import json

import joblib
import numpy as np
from threadpoolctl import threadpool_limits

from src.ml.season_data import load_csv
from src.ml.season_features import FEATURE_NAMES, build_features
from src.ml.season_models import ScoreModel, metrics, outcomes
from src.scripts.train_season_simulator import compact_metrics, say, write_json, write_csv


def analyse(output):
    bundle = joblib.load(output / 'model.joblib')
    report = json.loads((output / 'metrics.json').read_text())
    history, _ = load_csv(output / 'historical_matches.csv')
    rows, x, _ = build_features(history)
    template = bundle['model']
    years = np.array([m.year for m in rows])
    validation_years = report['validation_years']
    val = [m for m in rows if m.year in validation_years]
    groups = {'elo_strength': list(range(3)), 'overall_scoring_form': list(range(3, 7)),
              'venue_scoring_form': list(range(7, 11)), 'season_to_date_form': list(range(11, 17)),
              'promoted_flags': list(range(17, 19))}
    ablations = []
    for name, removed in [('all_features', []), *groups.items(), ('uniform_training_weights', [])]:
        columns = [i for i in range(len(FEATURE_NAMES)) if i not in removed]
        prediction = []
        for year in validation_years:
            train, valid = years < year, years == year
            goal, outcome = template.goal_spec, template.outcome_spec
            if name == 'uniform_training_weights':
                goal, outcome = replace(goal, half_life=None), replace(outcome, half_life=None)
            model = ScoreModel(goal, outcome, template.rho, template.blend).fit(
                x[train][:, columns], [m for m, yes in zip(rows, train) if yes])
            prediction.append(model.predict_grid(x[valid][:, columns]))
        grid = np.concatenate(prediction)
        result = compact_metrics(metrics(val, outcomes(grid), grid))
        ablations.append({'experiment': name, 'removed_features': [FEATURE_NAMES[i] for i in removed], **result})
        say(f'Ablation {name}: score NLL {result["score_nll"]:.5f}; log loss {result["log_loss"]:.5f}')
    for r in ablations:
        r['score_nll_change_vs_full'] = r['score_nll'] - ablations[0]['score_nll']
        r['log_loss_change_vs_full'] = r['log_loss'] - ablations[0]['log_loss']
    eda = []
    for year in sorted(set(years)):
        matches = [m for m in rows if m.year == year]
        eda.append({'season': matches[0].season, 'matches': len(matches),
                    'home_win_rate': float(np.mean([m.outcome == 0 for m in matches])),
                    'draw_rate': float(np.mean([m.outcome == 1 for m in matches])),
                    'away_win_rate': float(np.mean([m.outcome == 2 for m in matches])),
                    'home_goals_per_match': float(np.mean([m.hg for m in matches])),
                    'away_goals_per_match': float(np.mean([m.ag for m in matches]))})
    # Weighted-standardized coefficients are associations, not causal effects.
    coefficients = []
    goal_fit = template.goal_model
    for i, feature in enumerate(FEATURE_NAMES):
        row = {'feature': feature}
        if hasattr(goal_fit.models[0], 'coef_'):
            row['home_log_goal_coefficient_per_weighted_sd'] = float(goal_fit.models[0].coef_[i])
            row['away_log_goal_coefficient_per_weighted_sd'] = float(goal_fit.models[1].coef_[i])
        coefficients.append(row)
    # Paired month-block bootstrap keeps within-month losses together.
    with (output / 'test_predictions.csv').open() as f:
        test = list(csv.DictReader(f))
    feature_index = {(m.season, str(m.date), m.home, m.away): vector for m, vector in zip(rows, x)}
    losses = []
    for r in test:
        y, year = int(r['actual_outcome']), int(r['season'][:4])
        prior_rows = [m for m in rows if m.year < year]
        prior = np.bincount([m.outcome for m in prior_rows], minlength=3) / len(prior_rows)
        vector = feature_index[(r['season'], r['date'], r['home'], r['away'])]
        strength = 10 ** ((vector[2] + 65) / 800)
        ep = np.array([strength ** 2, strength, 1.])
        ep /= ep.sum()
        p = np.array([float(r['p_home']), float(r['p_draw']), float(r['p_away'])])
        losses.append((r['date'][:7], -np.log(p[y]), -np.log(prior[y]), -np.log(ep[y])))
    months = sorted({r[0] for r in losses})
    sums = np.array([[sum(r[i] for r in losses if r[0] == month) for i in range(1, 4)] for month in months])
    counts = np.array([sum(r[0] == month for r in losses) for month in months])
    rng = np.random.default_rng(202627)
    draws = rng.integers(len(months), size=(4000, len(months)))
    means = sums[draws].sum(axis=1) / counts[draws].sum(axis=1)[:, None]
    intervals = {}
    for i, label in ((1, 'historical_prior'), (2, 'fixed_elo')):
        difference = means[:, 0] - means[:, i]
        intervals[label] = {'mean_log_loss_difference': float((sums[:, 0].sum() - sums[:, i].sum()) / counts.sum()),
                            'bootstrap_95pct_interval': np.quantile(difference, [.025, .975]).tolist(),
                            'interpretation': 'Negative favours the selected model; block bootstrap is approximate and does not remove season-level dependence.'}
    analysis = {'feature_ablations': ablations, 'eda_by_season': eda,
                'effective_training_matches_goals': goal_fit.effective_sample_size,
                'effective_training_matches_outcomes': template.outcome_model.effective_sample_size if template.outcome_model else None,
                'coefficient_associations': coefficients,
                'test_month_block_bootstrap': {'blocks': len(months), 'resamples': 4000, 'comparisons': intervals},
                'protocol': 'Post-selection validation diagnostics at fixed hyperparameters; no holdout-driven feature selection or model replacement.'}
    write_json(output / 'analysis.json', analysis)
    write_csv(output / 'feature_ablations.csv', ablations)
    write_csv(output / 'eda_by_season.csv', eda)
    write_csv(output / 'feature_coefficients.csv', coefficients)
    say('Wrote feature ablations and diagnostics')
    return analysis


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--output-dir', type=Path, default=Path('artifacts/season_simulator'))
    args = parser.parse_args()
    with threadpool_limits(limits=1):
        analyse(args.output_dir)
