"""Bounded validation-only search toward the user's 80% all-match target.

Keep the published simulator unchanged; report the selected challenger on the
same already-examined test seasons honestly as a follow-up experiment.
"""
import argparse
from dataclasses import asdict
from pathlib import Path
import json
import joblib
import numpy as np
from threadpoolctl import threadpool_limits

from src.ml.season_data import load_csv
from src.ml.season_features import build_features, FEATURE_NAMES
from src.ml.season_models import Candidate, FittedModel, metrics
from src.scripts.train_season_simulator import compact_metrics, say, write_json, write_csv


def run(output):
    report = json.loads((output / 'metrics.json').read_text())
    history, _ = load_csv(output / 'historical_matches.csv')
    rows, x, _ = build_features(history)
    years = np.array([m.year for m in rows])
    validation = report['validation_years']
    valrows = [m for m in rows if m.year in validation]
    experiments = []
    # Two representations: complete form features, and a lower-variance core.
    for feature_set, indices in [('full', list(range(len(FEATURE_NAMES)))), ('strength_and_scoring', list(range(7)))]:
        for family, sizes in [('random_forest', (20, 60)), ('extra_trees', (20, 60)),
                              ('logistic', (.03, .3)), ('outcome_trees', (4, 7))]:
            for half in (730, 1460):
                for size in sizes:
                    spec = Candidate(family, half, size)
                    p = []
                    for year in validation:
                        train, valid = years < year, years == year
                        fitted = FittedModel(spec).fit(x[train][:, indices], [m for m, yes in zip(rows, train) if yes])
                        p.append(fitted.predict(x[valid][:, indices]))
                    result = compact_metrics(metrics(valrows, np.concatenate(p)))
                    record = {'feature_set': feature_set, 'indices': indices, 'spec': asdict(spec), **result}
                    experiments.append(record)
                    say(f'Accuracy experiment {feature_set}/{spec.name}: {result["accuracy"]:.2%}')
    # Tie-break by log loss; do not select on held-out results.
    best = min(experiments, key=lambda r: (-r['accuracy'], r['log_loss']))
    spec = Candidate(**best['spec'])
    heldout, all_predictions = {}, []
    for year in report['test_years']:
        train, test = years < year, years == year
        targets = [m for m, yes in zip(rows, test) if yes]
        fitted = FittedModel(spec).fit(x[train][:, best['indices']], [m for m, yes in zip(rows, train) if yes])
        p = fitted.predict(x[test][:, best['indices']])
        heldout[str(year)] = metrics(targets, p)
        for m, prob in zip(targets, p):
            all_predictions.append({'season': m.season, 'date': str(m.date), 'home': m.home, 'away': m.away,
                                    'actual': m.outcome, 'predicted': int(prob.argmax()),
                                    'p_home': float(prob[0]), 'p_draw': float(prob[1]), 'p_away': float(prob[2])})
    aggregate = sum(r['accuracy'] * r['matches'] for r in heldout.values()) / sum(r['matches'] for r in heldout.values())
    result = {'target_accuracy': .8, 'metric': 'Argmax home/draw/away accuracy on every held-out match, no abstention',
              'protocol': '32 follow-up candidates x 3 expanding validation folds. Challenger selected by validation accuracy only. Test seasons were already examined in the original study; this is not a fresh independent holdout.',
              'validation_experiments': experiments, 'selected_challenger': best,
              'test': heldout, 'aggregate_test_accuracy': aggregate, 'target_reached': aggregate >= .8,
              'published_simulator_replaced': False,
              'reason': 'This accuracy-focused classifier is a diagnostic challenger; replacing a score simulator also requires validating coherent score probabilities and season forecasts.'}
    write_json(output / 'accuracy_experiments.json', result)
    write_csv(output / 'accuracy_challenger_test_predictions.csv', all_predictions)
    say(f'Best validation accuracy {best["accuracy"]:.2%}; aggregate test accuracy {aggregate:.2%}; 80% reached: {aggregate >= .8}')
    return result


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--output-dir', type=Path, default=Path('artifacts/season_simulator'))
    with threadpool_limits(limits=1):
        run(parser.parse_args().output_dir)
