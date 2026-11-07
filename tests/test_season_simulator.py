from dataclasses import replace
from datetime import date, timedelta
import copy

import joblib
import numpy as np
import pytest
from threadpoolctl import threadpool_limits

from src.ml.season_context import MatchContext, fixture_key
from src.ml.season_data import (Match, audit_history, generated_schedule, load_csv,
                                load_fixtures, validate_schedule, write_matches)
from src.ml.season_features import FEATURE_NAMES, SeasonState, build_features
from src.ml.season_models import (Candidate, DecisionPolicy, ScoreModel, GOALS, metrics, outcomes,
                                  recency_weights, reconcile, score_grid)
from src.ml.season_simulation import rank_tables, simulate


def match(day, hg, ag, home='A', away='B'):
    return Match('2024-25', date(2024, 8, day), home, away, hg, ag)


def test_recency_half_life_and_future_rejection():
    latest = match(20, 1, 0)
    old = replace(latest, date=latest.date - timedelta(days=730))
    w = recency_weights([old, latest], latest.date, 730)
    assert w[1] / w[0] == pytest.approx(2)
    assert w.mean() == pytest.approx(1)
    with pytest.raises(ValueError, match='after fit cutoff'):
        recency_weights([latest], old.date, 730)


def test_current_and_future_results_do_not_change_current_features():
    history = [match(1, 5, 0), match(2, 1, 1, 'A', 'C'), match(3, 0, 2, 'B', 'C')]
    _, x, _ = build_features(history)
    changed = [history[0], replace(history[1], hg=8, ag=0), replace(history[2], hg=9)]
    _, y, _ = build_features(changed)
    np.testing.assert_array_equal(x[:2], y[:2])
    assert not np.array_equal(x[2], y[2])


def test_richer_features_are_pre_match_and_have_stable_schema():
    history = [match(1, 2, 0), match(8, 0, 1, 'B', 'A'), match(15, 1, 1)]
    _, x, _ = build_features(history)
    changed = [history[0], replace(history[1], hg=8, ag=0), replace(history[2], hg=4, ag=0)]
    _, y, _ = build_features(changed)
    assert x.shape == (3, len(FEATURE_NAMES))
    np.testing.assert_array_equal(x[:2], y[:2])
    assert x[0, FEATURE_NAMES.index('rest_days_difference')] == 0
    assert x[2, FEATURE_NAMES.index('home_recent_points_per_game')] != x[0, FEATURE_NAMES.index('home_recent_points_per_game')]


def test_optional_context_is_joined_without_changing_base_features():
    history = [match(1, 1, 0), match(8, 0, 1, 'B', 'A')]
    context = MatchContext.from_rows(('market_home_prob',), [
        (fixture_key(history[0]), (.55,)),
        (fixture_key(history[1]), (.25,)),
    ])
    _, x, state = build_features(history, context=context)
    assert state.feature_names[-1] == 'market_home_prob'
    assert x.shape == (2, len(FEATURE_NAMES) + 1)
    assert x[0, -1] == pytest.approx(.55)


def test_legacy_pickled_state_keeps_twenty_feature_schema_after_season_reset():
    state = SeasonState(['A', 'B'])
    del state.feature_names
    del state.legacy_schema
    state.start_season(2024, ['A', 'B'])
    features = state.features([match(1, 1, 0)])
    assert features.shape[-1] == 20


def test_draw_decision_policy_can_change_pick_without_changing_probabilities():
    matches = [match(1, 0, 0), match(2, 2, 0)]
    probabilities = np.array([[.45, .35, .20], [.40, .25, .35]])
    policy = DecisionPolicy(1.5)
    result = metrics(matches, probabilities, decision_policy=policy)
    assert policy.predict(probabilities).tolist() == [1, 0]
    assert result['raw_accuracy'] == pytest.approx(.5)
    assert result['accuracy'] == pytest.approx(1.0)
    assert result['draw_recall'] == pytest.approx(1.0)
    assert result['predicted_draw_rate'] == pytest.approx(.5)
    assert result['macro_f1'] > 0


def test_entire_date_predicted_before_updates():
    # Even malformed overlapping matches must not leak into each other's features.
    _, x, _ = build_features([match(1, 8, 0), match(1, 1, 1, 'A', 'C')])
    assert x[0, 0] == x[1, 0] == 1500


def test_promoted_club_does_not_reuse_stale_rating():
    state = SeasonState(['A', 'B', 'C'])
    state.start_season(2024, ['A', 'B'])
    state.values[:, state.index['C'], 0] = 1900
    state.start_season(2025, ['A', 'C'])
    assert state.values[0, state.index['C'], 0] == 1400
    assert state.values[0, state.index['C'], 10] == 1


def test_vectorized_real_and_simulated_state_transitions_are_identical():
    state = SeasonState(['A', 'B'])
    state.start_season(2024, ['A', 'B'])
    replica = state.replicate(4)
    games = [match(1, 3, 1)]
    state.update(games, [[3]], [[1]])
    replica.update(games, np.full((4, 1), 3), np.full((4, 1), 1))
    np.testing.assert_allclose(replica.values, np.repeat(state.values, 4, axis=0))


@pytest.mark.parametrize('rho', [-.15, 0, .05])
def test_score_distribution_valid_and_preserves_means(rho):
    rates = np.array([[1.5, 1.2], [.08, 5.5], [5.5, 5.5]])
    grid = score_grid(rates, rho)
    assert np.all(np.isfinite(grid)) and np.all(grid >= 0)
    np.testing.assert_allclose(grid.sum(axis=(1, 2)), 1)
    np.testing.assert_allclose((grid.sum(axis=2) * GOALS).sum(axis=1), rates[:, 0], atol=1e-5)
    np.testing.assert_allclose((grid.sum(axis=1) * GOALS).sum(axis=1), rates[:, 1], atol=1e-5)
    np.testing.assert_allclose(outcomes(grid).sum(axis=1), 1)


def test_classifier_reconciliation_preserves_target_outcome_probabilities():
    grid = score_grid(np.array([[1.2, 1.5]]), -.1)
    p = np.array([[.2, .5, .3]])
    np.testing.assert_allclose(outcomes(reconcile(grid, p, 1)), p)
    np.testing.assert_allclose(reconcile(grid, p, 0), grid)


def test_generated_schedule_is_complete_reproducible_and_seeded():
    clubs = [f'T{i}' for i in range(20)]
    fixtures = generated_schedule(clubs, '2026-27', 7)
    assert len(fixtures) == 380
    assert len(validate_schedule(fixtures)) == 20
    assert fixtures == generated_schedule(clubs, '2026-27', 7)
    assert fixtures != generated_schedule(clubs, '2026-27', 8)
    assert all(m.hg is None and m.ag is None for m in fixtures)
    with pytest.raises(ValueError, match='Incomplete/duplicate'):
        validate_schedule(fixtures[:-1])


def test_history_rejects_incomplete_and_duplicate_seasons():
    with pytest.raises(ValueError, match='Duplicate'):
        audit_history([match(1, 1, 0), match(1, 1, 0)], '2026-27')
    with pytest.raises(ValueError, match='Incomplete'):
        audit_history([match(1, 1, 0)], '2026-27')
    with pytest.raises(ValueError, match='known dates'):
        audit_history([replace(match(1, 1, 0), date=None)], '2026-27')


def test_csv_preserves_unplayed_and_played_matches(tmp_path):
    rows = [match(1, 1, 0), replace(match(2, 0, 0), hg=None, ag=None)]
    path = tmp_path / 'snapshot.csv'
    write_matches(path, rows)
    assert load_csv(path)[0] == rows


def test_future_source_file_is_not_read(tmp_path):
    from src.scripts.train_football_elo import load_scored_matches
    from src.ingestion.normalization.aliases import load_alias_maps
    directory = tmp_path / '2026-27'
    directory.mkdir()
    (directory / '1-premierleague.txt').write_text('This file must never reach the parser')
    rows, stats, _ = load_scored_matches(tmp_path, through='2025-26', alias_maps=load_alias_maps('config/team_aliases.json'))
    assert rows == []
    assert stats['source_files_excluded_before_read'] == 1


def test_h2h_breaks_equal_points_goal_difference_and_goals():
    teams = ['A', 'B']
    fixtures = [match(1, 1, 0), match(2, 1, 0, 'B', 'A')]
    # B scored twice away, A once away; equal H2H points, B wins away-goal tiebreak.
    scores = np.array([[[3, 2], [2, 1]]])
    points, gf, ga = np.array([[60, 60]]), np.array([[50, 50]]), np.array([[40, 40]])
    ranks, unresolved = rank_tables(points, gf, ga, fixtures, scores, teams, np.random.default_rng(1))
    assert ranks.tolist() == [[2, 1]]
    assert unresolved == 0


def test_end_to_end_reload_reproducibility_and_league_invariants(tmp_path):
    teams = [f'Club {i:02d}' for i in range(20)]
    rng = np.random.default_rng(1)
    history = [replace(m, hg=int(rng.poisson(1.6)), ag=int(rng.poisson(1.2)))
               for m in generated_schedule(teams, '2024-25')]
    rows, x, state = build_features(history)
    with threadpool_limits(limits=1):
        model = ScoreModel(Candidate('poisson', 730, .3), Candidate('logistic', 730, .1), -.05, .25).fit(x, rows)
        path = tmp_path / 'model.joblib'
        joblib.dump(model, path)
        loaded = joblib.load(path)
        np.testing.assert_allclose(model.predict_grid(x[:3]), loaded.predict_grid(x[:3]))
        fixtures = generated_schedule(teams, '2025-26')
        before = state.values.copy()
        result, scores = simulate(loaded, state, fixtures, n=16, seed=4)
        replay, scores2 = simulate(loaded, state, fixtures, n=16, seed=4)
    np.testing.assert_array_equal(state.values, before)
    np.testing.assert_array_equal(scores, scores2)
    assert result == replay
    assert scores.shape == (16, 380, 2)
    assert all(f['predicted_outcome'] in (0, 1, 2) for f in result['fixtures'])
    assert sum(r['title_probability'] for r in result['table']) == pytest.approx(1)
    assert sum(r['top4_probability'] for r in result['table']) == pytest.approx(4)
    assert sum(r['relegation_probability'] for r in result['table']) == pytest.approx(3)
    assert len(result['matchdays']) == 38
    assert all(r['sample_wins'] + r['sample_draws'] + r['sample_losses'] == 38 for r in result['table'])
    assert sum(r['sample_points'] for r in result['table']) == 1140 - int((scores[0, :, 0] == scores[0, :, 1]).sum())
    with pytest.raises(ValueError, match='before the simulated season'):
        simulate(model, state, generated_schedule(teams, '2024-25'), n=1)
