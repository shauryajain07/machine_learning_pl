"""Replay a trusted local model bundle without retraining."""
import argparse
from pathlib import Path
import joblib
from threadpoolctl import threadpool_limits
from src.ml.season_data import load_fixtures, load_csv
from src.ml.season_features import FEATURE_NAMES
from src.ml.season_simulation import simulate
from src.scripts.train_season_simulator import save_simulation, say


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--model', type=Path, default=Path('artifacts/season_simulator/model.joblib'))
    parser.add_argument('--fixtures', type=Path, default=Path('artifacts/season_simulator/target_fixtures.csv'))
    parser.add_argument('--simulations', type=int, default=10000)
    parser.add_argument('--seed', type=int, default=202627)
    parser.add_argument('--output-dir', type=Path, default=Path('artifacts/season_replay'))
    args = parser.parse_args()
    bundle = joblib.load(args.model)  # Load only locally generated, trusted bundles.
    bundle_names = tuple(bundle['feature_names'])
    base_count = min(len(bundle_names), len(FEATURE_NAMES))
    state_names = tuple(getattr(bundle.get('state'), 'feature_names', ()))
    if bundle_names[:base_count] != FEATURE_NAMES[:base_count] or (state_names and state_names != bundle_names):
        raise ValueError('Model feature schema differs from installed code')
    fixtures = load_csv(args.fixtures)[0] if args.fixtures.suffix == '.csv' else load_fixtures(args.fixtures)
    args.output_dir.mkdir(parents=True, exist_ok=True)
    with threadpool_limits(limits=1):
        simulation, scores = simulate(bundle['model'], bundle['state'], fixtures,
                                      args.simulations, args.seed, progress=say)
    save_simulation(args.output_dir, simulation, scores)
    say(f'Wrote replay to {args.output_dir.resolve()}')


if __name__ == '__main__':
    main()
