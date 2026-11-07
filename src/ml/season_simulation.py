"""Reproducible, recursive Monte Carlo seasons with coherent league tables."""
from __future__ import annotations

import numpy as np
from src.ml.season_data import validate_schedule
from src.ml.season_features import batches
from src.ml.season_models import MAX_GOALS, GOALS, outcomes


def rank_tables(points, gf, ga, fixtures, scores, teams, rng):
    """Points, GD, GF, tied-group H2H points, H2H away goals, random playoff.

    Random final tie resolution is an explicit neutral-playoff approximation;
    it prevents alphabetical bias. Return its frequency for the report.
    """
    n, t = points.shape
    order = np.lexsort((rng.random((n, t)), -gf, -(gf - ga), -points), axis=1)
    team_index = {team: i for i, team in enumerate(teams)}
    hp = np.zeros((n, t, t), dtype=np.int16)
    away_goals = np.zeros_like(hp)
    for j, m in enumerate(fixtures):
        h, a = team_index[m.home], team_index[m.away]
        gh, ga_ = scores[:, j, 0], scores[:, j, 1]
        hp[:, h, a] += 3 * (gh > ga_) + (gh == ga_)
        hp[:, a, h] += 3 * (ga_ > gh) + (gh == ga_)
        away_goals[:, a, h] += ga_
    sorted_p = np.take_along_axis(points, order, axis=1)
    sorted_gd = np.take_along_axis(gf - ga, order, axis=1)
    sorted_gf = np.take_along_axis(gf, order, axis=1)
    tied = (np.diff(sorted_p, axis=1) == 0) & (np.diff(sorted_gd, axis=1) == 0) & (np.diff(sorted_gf, axis=1) == 0)
    unresolved = 0
    for sim in np.flatnonzero(tied.any(axis=1)):
        start = 0
        while start < t - 1:
            if not tied[sim, start]:
                start += 1
                continue
            end = start + 1
            while end < t - 1 and tied[sim, end]:
                end += 1
            ids = order[sim, start:end + 1].copy()
            headpoints = hp[sim][np.ix_(ids, ids)].sum(axis=1)
            headaway = away_goals[sim][np.ix_(ids, ids)].sum(axis=1)
            unresolved += int(len(set(zip(headpoints.tolist(), headaway.tolist()))) < len(ids))
            order[sim, start:end + 1] = ids[np.lexsort((rng.random(len(ids)), -headaway, -headpoints))]
            start = end + 1
    ranks = np.empty_like(order)
    np.put_along_axis(ranks, order, np.broadcast_to(np.arange(1, t + 1), order.shape), axis=1)
    return ranks, unresolved


def summarise_scores(fixtures, scores, rng):
    teams = validate_schedule(fixtures)
    index = {t: i for i, t in enumerate(teams)}
    n, t = scores.shape[0], len(teams)
    points = np.zeros((n, t), dtype=np.int16)
    gf, ga, wins, draws, losses = [np.zeros_like(points) for _ in range(5)]
    for j, match in enumerate(fixtures):
        for team, goals, against in ((match.home, scores[:, j, 0], scores[:, j, 1]),
                                     (match.away, scores[:, j, 1], scores[:, j, 0])):
            i = index[team]
            points[:, i] += 3 * (goals > against) + (goals == against)
            gf[:, i] += goals
            ga[:, i] += against
            wins[:, i] += goals > against
            draws[:, i] += goals == against
            losses[:, i] += goals < against
    ranks, unresolved = rank_tables(points, gf, ga, fixtures, scores, teams, rng)
    table = []
    for i, team in enumerate(teams):
        p = points[:, i]
        champion = float(np.mean(ranks[:, i] == 1))
        table.append({'team': team, 'expected_points': float(p.mean()),
                      'points_p05': float(np.quantile(p, .05)), 'points_p95': float(np.quantile(p, .95)),
                      'expected_rank': float(ranks[:, i].mean()),
                      'title_probability': champion,
                      'title_monte_carlo_se': float(np.sqrt(champion * (1 - champion) / n)),
                      'top4_probability': float(np.mean(ranks[:, i] <= 4)),
                      'relegation_probability': float(np.mean(ranks[:, i] > t - 3)),
                      'expected_goals_for': float(gf[:, i].mean()),
                      'expected_goals_against': float(ga[:, i].mean()),
                      'position_probabilities': [float(np.mean(ranks[:, i] == k)) for k in range(1, t + 1)],
                      'sample_points': int(p[0]), 'sample_rank': int(ranks[0, i]),
                      'sample_wins': int(wins[0, i]), 'sample_draws': int(draws[0, i]),
                      'sample_losses': int(losses[0, i]), 'sample_gf': int(gf[0, i]), 'sample_ga': int(ga[0, i])})
    table.sort(key=lambda row: row['expected_rank'])
    if not np.all(wins + draws + losses == 2 * (t - 1)):
        raise AssertionError('Incorrect matches played')
    if not np.array_equal(gf.sum(axis=1), ga.sum(axis=1)):
        raise AssertionError('League goals do not balance')
    return table, points, ranks, unresolved


def simulate(model, historical_state, fixtures, n=10000, seed=202627, progress=None):
    if n < 1:
        raise ValueError('Simulation count must be positive')
    teams = validate_schedule(fixtures)
    fixtures = sorted(fixtures, key=lambda m: (m.date, m.home, m.away))
    if historical_state.year is None or historical_state.year >= fixtures[0].year:
        raise ValueError('Historical state must end before the simulated season')
    if model.trained_through >= str(min(m.date for m in fixtures)):
        raise ValueError('Model was fitted on target-season data')
    rng = np.random.default_rng(seed)
    state = historical_state.replicate(n)
    state.start_season(fixtures[0].year, teams)
    scores = np.zeros((n, len(fixtures), 2), dtype=np.int16)
    forecasts = []
    offset = 0
    for batch_index, (_, group) in enumerate(batches(fixtures)):
        group = list(group)
        features = state.features(group)
        x = features.reshape(-1, features.shape[-1])
        flat_scores = np.empty((len(x), 2), dtype=np.int16)
        prob_sum = np.zeros((len(group), 3))
        goal_sum = np.zeros((len(group), 2))
        # Bound memory independently of the requested simulation count.
        for start in range(0, len(x), 2048):
            grid = model.predict_grid(x[start:start + 2048])
            cdf = np.cumsum(grid.reshape(len(grid), -1), axis=1)
            cdf[:, -1] = 1.
            draw = (cdf < rng.random((len(grid), 1))).sum(axis=1)
            flat_scores[start:start + len(grid)] = np.column_stack((draw // (MAX_GOALS + 1), draw % (MAX_GOALS + 1)))
            probabilities = outcomes(grid)
            expected = np.column_stack(((grid.sum(axis=2) * GOALS).sum(axis=1),
                                        (grid.sum(axis=1) * GOALS).sum(axis=1)))
            which = np.arange(start, start + len(grid)) % len(group)
            np.add.at(prob_sum, which, probabilities)
            np.add.at(goal_sum, which, expected)
        batch_scores = flat_scores.reshape(n, len(group), 2)
        scores[:, offset:offset + len(group)] = batch_scores
        state.update(group, batch_scores[:, :, 0], batch_scores[:, :, 1])
        for j, m in enumerate(group):
            probabilities = prob_sum[j] / n
            raw_prediction = int(probabilities.argmax())
            policy = getattr(model, 'decision_policy', None)
            predicted_outcome = (raw_prediction if policy is None else
                                 int(policy.predict(probabilities[None, :])[0]))
            forecasts.append({'date': str(m.date), 'matchday': m.matchday, 'home': m.home, 'away': m.away,
                              'home_win': float(probabilities[0]), 'draw': float(probabilities[1]),
                              'away_win': float(probabilities[2]),
                              'raw_predicted_outcome': raw_prediction,
                              'predicted_outcome': predicted_outcome,
                              'expected_home_goals': float(goal_sum[j, 0] / n),
                              'expected_away_goals': float(goal_sum[j, 1] / n),
                              'sample_home_goals': int(batch_scores[0, j, 0]),
                              'sample_away_goals': int(batch_scores[0, j, 1])})
        offset += len(group)
        if progress and batch_index % 30 == 0:
            progress(f'Simulated {offset}/{len(fixtures)} fixtures across {n:,} paths')
    table, points, ranks, unresolved = summarise_scores(fixtures, scores, rng)
    matchdays = []
    if all(m.matchday is not None for m in fixtures):
        cum = np.zeros((n, len(teams)), dtype=np.int16)
        gf, ga = np.zeros_like(cum), np.zeros_like(cum)
        index = {t: i for i, t in enumerate(teams)}
        for day in sorted({m.matchday for m in fixtures}):
            for j, m in enumerate(fixtures):
                if m.matchday != day:
                    continue
                for team, g, a in ((m.home, scores[:, j, 0], scores[:, j, 1]), (m.away, scores[:, j, 1], scores[:, j, 0])):
                    i = index[team]
                    cum[:, i] += 3 * (g > a) + (g == a)
                    gf[:, i] += g
                    ga[:, i] += a
            rows = [{'team': team, 'expected_points': float(cum[:, i].mean()),
                     'p05': float(np.quantile(cum[:, i], .05)), 'p95': float(np.quantile(cum[:, i], .95)),
                     'sample_points': int(cum[0, i]), 'sample_gf': int(gf[0, i]), 'sample_ga': int(ga[0, i])}
                    for i, team in enumerate(teams)]
            matchdays.append({'matchday': day, 'table': sorted(rows, key=lambda r: (-r['expected_points'], r['team']))})
    return {'table': table, 'fixtures': forecasts, 'matchdays': matchdays,
            'n_simulations': n, 'seed': seed, 'unresolved_tie_groups': unresolved,
            'mode': 'preseason_recursive_simulation',
            'uncertainty': 'Outcome randomness and recursive form changes conditional on fitted parameters; no parameter, transfer or injury uncertainty.',
            'tie_break': 'Points, goal difference, goals for, tied-group head-to-head points, head-to-head away goals; seeded random neutral-playoff approximation for remaining ties.',
            'fixture_probabilities': 'Preseason marginal probabilities averaged over simulated prior results, not live conditional forecasts.'}, scores
