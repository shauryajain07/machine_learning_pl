"""Train, evaluate and simulate the Premier League using chronological splits."""
from __future__ import annotations

import argparse
import csv
from dataclasses import asdict
from datetime import datetime, timezone
import json
from pathlib import Path
import platform

import joblib
import numpy as np
import scipy
import sklearn
from threadpoolctl import threadpool_limits

from src.db.connection import resolve_database_url
from src.ml.season_context import MatchContext
from src.ml.season_data import (Match, audit_history, fingerprint, load_csv, load_database_bundle,
                                load_fixtures, load_local, validate_schedule, write_matches,
                                infer_participants, generated_schedule)
from src.ml.season_features import FEATURE_NAMES, build_features
from src.ml.season_models import (Candidate, DecisionPolicy, FittedModel, ScoreModel, candidates,
                                  metrics, outcomes, reconcile, score_grid)
from src.ml.season_simulation import simulate, summarise_scores


def say(text):
    print(text, flush=True)


def write_json(path, data):
    Path(path).write_text(json.dumps(data, indent=2, default=str, allow_nan=False) + '\n')


def write_csv(path, rows):
    if not rows:
        return
    with Path(path).open('w', newline='') as f:
        writer = csv.DictWriter(f, fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)


def compact_metrics(value):
    return {k: v for k, v in value.items() if k != 'calibration'}


def split_manifest(rows, validation_years, test_years):
    splits = []
    for phase, years in [('validation', validation_years), ('test', test_years)]:
        for year in years:
            train, target = [m for m in rows if m.year < year], [m for m in rows if m.year == year]
            if not train or not target or max(m.date for m in train) >= min(m.date for m in target):
                raise ValueError(f'Empty or chronologically overlapping split: {phase} {year}')
            splits.append({'phase': phase, 'target_season': target[0].season,
                           'train_matches': len(train), 'evaluation_matches': len(target),
                           'train_first_date': str(min(m.date for m in train)),
                           'train_last_date': str(max(m.date for m in train)),
                           'evaluation_first_date': str(min(m.date for m in target)),
                           'evaluation_last_date': str(max(m.date for m in target)),
                           'train_seasons': sorted({m.season for m in train}),
                           'result_used_for_hyperparameter_selection': phase == 'validation'})
    return splits


def tune(rows, x, validation_seasons, draw_focused=False, max_draw_rate=.10):
    years = np.array([m.year for m in rows])
    val_mask = np.isin(years, validation_seasons)
    validation = [m for m, yes in zip(rows, val_mask) if yes]
    goal_predictions, outcome_predictions, leaderboard = {}, {}, []
    for spec in candidates():
        predictions = []
        for year in validation_seasons:
            train = years < year
            valid = years == year
            if not train.any() or not valid.any():
                raise ValueError(f'Empty chronological fold for {year}')
            fitted = FittedModel(spec).fit(x[train], [m for m, yes in zip(rows, train) if yes])
            predictions.append(fitted.predict(x[valid]))
        prediction = np.concatenate(predictions)
        if spec.family in ('poisson', 'poisson_trees'):
            grid = score_grid(prediction)
            result = metrics(validation, outcomes(grid), grid)
            goal_predictions[spec.name] = prediction
        else:
            result = metrics(validation, prediction)
            outcome_predictions[spec.name] = prediction
        leaderboard.append({'name': spec.name, **asdict(spec), **compact_metrics(result)})
        say(f'Validation {spec.name}: log loss {result["log_loss"]:.4f}' +
            (f', score NLL {result["score_nll"]:.4f}' if 'score_nll' in result else ''))
    goal_entry = min((r for r in leaderboard if 'score_nll' in r), key=lambda r: r['score_nll'])
    outcome_entry = min((r for r in leaderboard if 'score_nll' not in r), key=lambda r: r['log_loss'])
    goal_spec = Candidate(goal_entry['family'], goal_entry['half_life'], goal_entry['complexity'])
    outcome_spec = Candidate(outcome_entry['family'], outcome_entry['half_life'], outcome_entry['complexity'])
    best, best_grid, combinations = None, None, []
    for rho in (-.15, -.10, -.05, 0., .05):
        base = score_grid(goal_predictions[goal_spec.name], rho)
        for blend in (0., .25, .5, .75, 1.):
            grid = reconcile(base, outcome_predictions[outcome_spec.name], blend)
            result = metrics(validation, outcomes(grid), grid)
            record = {'rho': rho, 'blend': blend, **compact_metrics(result)}
            combinations.append(record)
            if best is None or result['score_nll'] < best['score_nll']:
                best = record
                best_grid = grid
    if best_grid is None:
        raise RuntimeError('No score ensemble candidate was evaluated')
    # Probability selection remains score-NLL-first for coherent season
    # simulations.  Pick selection is tuned separately so the simulator does
    # not have to sacrifice calibrated probabilities just to force more draws.
    policy_candidates = [DecisionPolicy(round(multiplier, 2))
                         for multiplier in np.arange(.75, 2.51, .05)]
    policy_results = [(candidate, metrics(validation, outcomes(best_grid),
                                          decision_policy=candidate))
                      for candidate in policy_candidates]
    if draw_focused:
        feasible = [(candidate, result) for candidate, result in policy_results
                    if result['predicted_draw_rate'] <= max_draw_rate]
        if not feasible:
            feasible = policy_results
        policy, _ = max(
            feasible,
            key=lambda pair: (pair[1]['macro_f1'], pair[1]['draw_f1'],
                              pair[1]['accuracy'],
                              -abs(pair[0].draw_multiplier - 1.0)))
    else:
        policy, _ = max(
            policy_results,
            key=lambda pair: (pair[1]['accuracy'],
                              -abs(pair[0].draw_multiplier - 1.0)))
    policy_metrics = metrics(validation, outcomes(best_grid), best_grid, decision_policy=policy)
    best = dict(best)
    best['raw_accuracy'] = best['accuracy']
    best['accuracy'] = policy_metrics['accuracy']
    best['decision_policy'] = policy.config()
    model = ScoreModel(goal_spec, outcome_spec, best['rho'], best['blend'], policy)
    return model, {'candidate_leaderboard': leaderboard, 'score_ensemble_search': combinations,
                   'selection_metric': 'Mean out-of-fold exact-score negative log likelihood (lower is better); classifier selected by outcome log loss.',
                   'pick_selection_metric': ('Validation macro-F1 with a predicted-draw-rate cap after '
                                             'draw-multiplier decision policy.' if draw_focused else
                                             'Validation top-1 home/draw/away accuracy after draw-multiplier decision policy.'),
                   'draw_focused': draw_focused,
                   'max_predicted_draw_rate': max_draw_rate if draw_focused else None,
                   'selected': model.config(), 'validation': best,
                   'decision_policy_validation': policy_metrics}


def evaluate(template, rows, x, test_seasons, season_simulations, extra_teams, output=None, context=None):
    years = np.array([m.year for m in rows])
    report, predictions, season_reports = {}, [], {}
    for year in test_seasons:
        train, test = years < year, years == year
        training = [m for m, yes in zip(rows, train) if yes]
        heldout = [m for m, yes in zip(rows, test) if yes]
        if not heldout:
            raise ValueError(f'No test matches for {year}')
        model = ScoreModel(template.goal_spec, template.outcome_spec, template.rho,
                           template.blend, template.decision_policy).fit(x[train], training)
        grid = model.predict_grid(x[test])
        prior = np.bincount([m.outcome for m in training], minlength=3) / len(training)
        elo_delta = x[test, FEATURE_NAMES.index('elo_difference')] + 65
        strength = 10 ** (elo_delta / 800)
        elo = np.column_stack((strength ** 2, strength, np.ones(len(strength))))
        elo /= elo.sum(axis=1, keepdims=True)
        report[str(year)] = {'selected': metrics(heldout, outcomes(grid), grid, model.decision_policy),
                             'historical_outcome_prior': metrics(heldout, np.tile(prior, (len(heldout), 1))),
                             'fixed_elo_baseline': metrics(heldout, elo),
                             'unblended_goal_model': metrics(heldout, outcomes(score_grid(model.goal_model.predict(x[test]))))}
        for m, p in zip(heldout, outcomes(grid)):
            predictions.append({'season': m.season, 'date': str(m.date), 'home': m.home, 'away': m.away,
                                'home_goals': m.hg, 'away_goals': m.ag, 'actual_outcome': m.outcome,
                                'raw_predicted_outcome': int(p.argmax()),
                                'predicted_outcome': int(model.decision_policy.predict(np.asarray([p]))[0]),
                                'p_home': float(p[0]), 'p_draw': float(p[1]), 'p_away': float(p[2])})
        say(f'Test {year}: ' + json.dumps(compact_metrics(report[str(year)]['selected'])))
        if season_simulations:
            _, _, state = build_features(training, extra_teams=extra_teams, context=context)
            fixtures = [Match(m.season, m.date, m.home, m.away, matchday=m.matchday, source_id=m.source_id) for m in heldout]
            say(f'Backtesting full preseason forecast for {year}, {season_simulations:,} seasons')
            simulation, simulated_scores = simulate(model, state, fixtures, n=season_simulations, seed=year, progress=say)
            ordered = sorted(heldout, key=lambda m: (m.date, m.home, m.away))
            actual_scores = np.array([[[m.hg, m.ag] for m in ordered]], dtype=np.int16)
            actual, _, _, _ = summarise_scores(ordered, actual_scores, np.random.default_rng(year))
            actual_points = {r['team']: r['sample_points'] for r in actual}
            actual_ranks = {r['team']: r['sample_rank'] for r in actual}
            comparison = [{'team': r['team'], 'actual_points': actual_points[r['team']],
                           'expected_points': r['expected_points'], 'p05': r['points_p05'], 'p95': r['points_p95'],
                           'actual_rank': actual_ranks[r['team']], 'expected_rank': r['expected_rank'],
                           'title_probability': r['title_probability'], 'relegation_probability': r['relegation_probability']}
                          for r in simulation['table']]
            # Raw match-derived points: sanctions/deductions are intentionally absent.
            season_reports[str(year)] = {'points_mae': float(np.mean([abs(r['actual_points'] - r['expected_points']) for r in comparison])),
                                         'points_90pct_interval_coverage': float(np.mean([r['p05'] <= r['actual_points'] <= r['p95'] for r in comparison])),
                                         'n_simulations': season_simulations, 'clubs': comparison}
            if output:
                directory = output / heldout[0].season
                directory.mkdir(parents=True, exist_ok=True)
                save_simulation(directory, simulation, simulated_scores)
                write_csv(directory / 'predicted_vs_actual.csv', comparison)
                write_json(directory / 'model_config.json', model.config())
                write_matches(directory / 'actual_results.csv', ordered)
                joblib.dump({'model': model, 'state': state, 'feature_names': state.feature_names}, directory / 'model.joblib')
    return report, predictions, season_reports


def save_simulation(output, simulation, scores):
    write_json(output / 'simulation.json', simulation)
    write_csv(output / 'fixture_predictions.csv', simulation['fixtures'])
    write_csv(output / 'season_forecast.csv', [{k: v for k, v in r.items() if k != 'position_probabilities'} for r in simulation['table']])
    np.savez_compressed(output / 'simulated_scores.npz', scores=scores,
                        home=np.array([r['home'] for r in simulation['fixtures']]),
                        away=np.array([r['away'] for r in simulation['fixtures']]))


def build_parser():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--source', choices=['auto', 'supabase', 'local', 'csv'], default='auto')
    parser.add_argument('--snapshot', type=Path, help='CSV exported with the documented projection')
    parser.add_argument('--root', type=Path, default=Path('england'))
    parser.add_argument('--fixtures', type=Path)
    parser.add_argument('--teams', type=Path, help='Optional JSON array of 20 clubs for generated schedule')
    parser.add_argument('--context', type=Path,
                        help='Optional CSV of pre-match numeric context features keyed by fixture')
    parser.add_argument('--target-season', default='2026-27')
    parser.add_argument('--validation-years', nargs='+', type=int, default=[2021, 2022, 2023])
    parser.add_argument('--test-years', nargs='+', type=int, default=[2024, 2025])
    parser.add_argument('--simulations', type=int, default=10000)
    parser.add_argument('--backtest-simulations', type=int, default=2000)
    parser.add_argument('--seed', type=int, default=202627)
    parser.add_argument('--draw-focused', action='store_true',
                        help='Select the top-1 policy with macro-F1 while capping predicted draws')
    parser.add_argument('--max-predicted-draw-rate', type=float, default=.10,
                        help='Maximum validation fraction selected as draws in draw-focused mode')
    parser.add_argument('--output-dir', type=Path, default=Path('artifacts/season_simulator'))
    return parser


def run(args):
    validation, tests = sorted(set(args.validation_years)), sorted(set(args.test_years))
    target = int(args.target_season[:4])
    if not validation or not tests or max(validation) >= min(tests) or max(tests) >= target:
        raise ValueError('Require disjoint validation < test < target seasons')
    if args.simulations < 1 or args.backtest_simulations < 0:
        raise ValueError('Invalid simulation counts')
    if not 0 <= args.max_predicted_draw_rate <= 1:
        raise ValueError('--max-predicted-draw-rate must be between zero and one')
    output = args.output_dir
    output.mkdir(parents=True, exist_ok=True)
    source = args.source
    context = None
    if source == 'auto':
        source = 'supabase' if resolve_database_url() else 'local'
    if source == 'supabase':
        raw, provenance, context = load_database_bundle(target_season=args.target_season)
    elif source == 'csv':
        if args.snapshot is None:
            raise ValueError('--snapshot is required for CSV source')
        raw, provenance = load_csv(args.snapshot)
        context = MatchContext.from_csv(args.context) if args.context else None
    else:
        through = f'{target - 1}-{target % 100:02d}'
        raw, provenance = load_local(args.root, through=through)
        context = MatchContext.from_csv(args.context) if args.context else None
        say('Using pinned LOCAL source. Live Supabase contents have NOT been verified.')
    history, audit = audit_history(raw, args.target_season)
    if args.fixtures:
        fixtures = load_fixtures(args.fixtures)
        fixture_source = {'kind': 'explicit_fixture_snapshot', 'path': str(args.fixtures.resolve()), 'results_discarded': True}
    else:
        if args.teams:
            members = json.loads(args.teams.read_text())
            membership = {'method': 'Explicit club list'}
        else:
            previous = max(m.season for m in history)
            members, membership = infer_participants(history, args.root / previous / '2-championship.txt')
        fixtures = generated_schedule(members, args.target_season, args.seed)
        fixture_source = {'kind': 'generated_double_round_robin', 'dates': 'Illustrative weekly dates, not official fixtures',
                          'target_season_files_read': False, 'membership': membership}
    if any(m.season != args.target_season for m in fixtures):
        raise ValueError('Fixture season does not match target season')
    teams = validate_schedule(fixtures)
    if max(m.date for m in history) >= min(m.date for m in fixtures):
        raise ValueError('Training cutoff overlaps target fixtures')
    provenance.update({'history_sha256': fingerprint(history), 'fixture_sha256': fingerprint(fixtures),
                       'fixture_source': fixture_source, 'audit': audit})
    write_json(output / 'data_audit.json', provenance)
    write_matches(output / 'historical_matches.csv', history)
    write_matches(output / 'target_fixtures.csv', fixtures)
    rows, x, state = build_features(history, extra_teams=teams, context=context)
    splits = split_manifest(rows, validation, tests)
    write_json(output / 'splits.json', splits)
    write_csv(output / 'training_features.csv', [{'season': m.season, 'date': str(m.date), 'home': m.home, 'away': m.away,
                                                 **dict(zip(state.feature_names, vector.tolist()))} for m, vector in zip(rows, x)])
    say(f'Validated {len(history):,} matches across {len(audit["coverage"])} seasons; {len(fixtures)} target fixtures.')
    template, tuning = tune(rows, x, validation, draw_focused=args.draw_focused,
                            max_draw_rate=args.max_predicted_draw_rate)
    say('Selected on validation only: ' + json.dumps(template.config()))
    evaluation, predictions, season_tests = evaluate(template, rows, x, tests, args.backtest_simulations,
                                                      teams, output / 'backtests', context=context)
    write_csv(output / 'test_predictions.csv', predictions)
    report = {'created_at_utc': datetime.now(timezone.utc).isoformat(), 'source': provenance,
              'versions': {'python': platform.python_version(), 'numpy': np.__version__, 'scipy': scipy.__version__, 'sklearn': sklearn.__version__},
              'features': list(state.feature_names), 'validation_years': validation, 'test_years': tests,
              'protocol': 'Expanding-window annual refits; all features use strictly previous dates. Hyperparameters are frozen before test evaluation. Later test years can train on earlier completed test years. Deployment refits on all pre-target matches.',
              'tuning': tuning, 'test': evaluation, 'preseason_backtests': season_tests}
    write_json(output / 'metrics.json', report)
    final_model = template.fit(x, rows)
    joblib.dump({'model': final_model, 'state': state, 'feature_names': state.feature_names,
                 'provenance': provenance, 'versions': report['versions']}, output / 'model.joblib')
    say(f'Fitted deployment model through {final_model.trained_through}; simulating {args.simulations:,} seasons.')
    simulation, scores = simulate(final_model, state, fixtures, n=args.simulations, seed=args.seed, progress=say)
    simulation['season'] = args.target_season
    simulation['promoted_teams'] = sorted(set(teams) - state.current_teams)
    simulation['source_kind'] = provenance['kind']
    simulation['model_config'] = final_model.config()
    save_simulation(output, simulation, scores)
    from src.ml.season_report import write_report
    write_report(output, report, simulation)
    say(f'Complete: {output.resolve() / "report.html"}')
    return 0


if __name__ == '__main__':
    with threadpool_limits(limits=1):
        raise SystemExit(run(build_parser().parse_args()))
