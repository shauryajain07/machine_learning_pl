"""Test the ML recipe on other seasons without using the 2026 deployment model.

Hyperparameters are selected only on validation years BEFORE every test year.
Every tested season gets a model refit using only earlier historical matches.
"""
import argparse
from pathlib import Path
import numpy as np
from threadpoolctl import threadpool_limits
from src.ml.season_data import load_local, load_csv, audit_history, fingerprint
from src.ml.season_features import build_features
from src.scripts.train_season_simulator import tune, evaluate, split_manifest, write_json, write_csv, say


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--root', type=Path, default=Path('england'))
    parser.add_argument('--snapshot', type=Path, help='Optional historical CSV, including all requested test seasons')
    parser.add_argument('--validation-years', nargs='+', type=int, default=[2021, 2022, 2023])
    parser.add_argument('--test-years', nargs='+', type=int, default=[2024, 2025])
    parser.add_argument('--simulations', type=int, default=2000)
    parser.add_argument('--output-dir', type=Path, default=Path('artifacts/historical_backtests'))
    args = parser.parse_args()
    validation, tests = sorted(set(args.validation_years)), sorted(set(args.test_years))
    if not validation or not tests or max(validation) >= min(tests) or args.simulations < 1:
        raise ValueError('Validation years must precede all test years; simulations must be positive')
    last = max(tests)
    through = f'{last}-{(last + 1) % 100:02d}'
    target = f'{last + 1}-{(last + 2) % 100:02d}'
    raw, source = load_csv(args.snapshot) if args.snapshot else load_local(args.root, through=through)
    history, audit = audit_history(raw, target)
    teams = {t for m in history for t in (m.home, m.away)}
    rows, x, _ = build_features(history)
    output = args.output_dir
    output.mkdir(parents=True, exist_ok=True)
    write_json(output / 'splits.json', split_manifest(rows, validation, tests))
    with threadpool_limits(limits=1):
        template, tuning = tune(rows, x, validation)
        evaluation, predictions, seasons = evaluate(template, rows, x, tests, args.simulations, teams, output)
    write_json(output / 'metrics.json', {'source': source, 'audit': audit, 'history_sha256': fingerprint(history),
                                        'tuning': tuning, 'test': evaluation, 'preseason_backtests': seasons})
    write_csv(output / 'test_predictions.csv', predictions)
    say(f'Historical evaluation complete: {output.resolve()}')


if __name__ == '__main__':
    main()
