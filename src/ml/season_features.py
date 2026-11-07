"""Leakage-safe features shared by historical replay and season simulation."""
from __future__ import annotations

import copy
from itertools import groupby

import numpy as np


# Keep the original twenty names first so old local model bundles can still be
# replayed after the richer feature schema is installed.
FEATURE_NAMES = (
    'home_elo', 'away_elo', 'elo_difference',
    'home_goals_for_ewm', 'home_goals_against_ewm',
    'away_goals_for_ewm', 'away_goals_against_ewm',
    'home_venue_goals_for_ewm', 'home_venue_goals_against_ewm',
    'away_venue_goals_for_ewm', 'away_venue_goals_against_ewm',
    'home_points_per_game', 'away_points_per_game',
    'home_goal_difference_per_game', 'away_goal_difference_per_game',
    'home_season_games', 'away_season_games', 'home_promoted', 'away_promoted',
    'scheduled_matchday_fraction',
    'home_recent_points_per_game', 'away_recent_points_per_game',
    'home_recent_goal_difference_per_game', 'away_recent_goal_difference_per_game',
    'home_draw_rate_ewm', 'away_draw_rate_ewm',
    'rest_days_difference', 'absolute_elo_difference',
    'combined_goal_rate', 'recent_points_per_game_difference',
    'recent_goal_difference_per_game_difference', 'draw_rate_average',
)

RECENT_MATCHES = 5
DRAW_STATE_INDEX = 11
STATE_WIDTH = DRAW_STATE_INDEX + 1
BASE_STATE_PRIOR = np.array([1500, 1.4, 1.4, 1.55, 1.25, 1.25, 1.55], dtype=float)
PROMOTED_STATE_PRIOR = np.array([1400, 1.1, 1.75, 1.25, 1.6, .95, 1.9], dtype=float)


class SeasonState:
    """State transition engine for one historical path or many simulations.

    The first seven state values are Elo and exponentially weighted scoring
    rates.  The remaining legacy values are points, goal difference, games
    played, promotion status, and draw tendency.  Exact five-match queues and
    last-match dates support the additional pre-match features.
    """

    def __init__(self, teams, n=1, context=None):
        self.teams = tuple(sorted(teams))
        self.index = {team: i for i, team in enumerate(self.teams)}
        # Elo, GF, GA, home GF, home GA, away GF, away GA, P, GD, GP,
        # promoted, exponentially weighted draw rate.
        self.values = np.zeros((n, len(self.teams), STATE_WIDTH), dtype=float)
        self.values[:, :, :7] = BASE_STATE_PRIOR
        self.values[:, :, DRAW_STATE_INDEX] = .25
        self.recent_points = np.zeros((n, len(self.teams), RECENT_MATCHES), dtype=float)
        self.recent_goal_difference = np.zeros((n, len(self.teams), RECENT_MATCHES), dtype=float)
        self.recent_count = np.zeros((n, len(self.teams)), dtype=np.int16)
        self.last_match_date = np.full((n, len(self.teams)), -1, dtype=np.int32)
        self.context = context
        self.feature_names = tuple(FEATURE_NAMES) + tuple(getattr(context, 'feature_names', ()))
        self.legacy_schema = False
        self.previous_teams = set()
        self.current_teams = set()
        self.year = None

    def _ensure_auxiliary_state(self):
        """Upgrade a v1 pickled state while preserving its twenty-feature output."""
        legacy = getattr(self, 'legacy_schema', not hasattr(self, 'feature_names'))
        if not hasattr(self, 'context'):
            self.context = None
        if legacy:
            self.feature_names = tuple(FEATURE_NAMES[:20])
        self.legacy_schema = legacy
        if self.values.shape[2] < STATE_WIDTH:
            extra = np.full((self.values.shape[0], self.values.shape[1],
                             STATE_WIDTH - self.values.shape[2]), .25, dtype=float)
            self.values = np.concatenate((self.values, extra), axis=2)
        n, team_count = self.values.shape[:2]
        if not hasattr(self, 'recent_points'):
            self.recent_points = np.zeros((n, team_count, RECENT_MATCHES), dtype=float)
        if not hasattr(self, 'recent_goal_difference'):
            self.recent_goal_difference = np.zeros((n, team_count, RECENT_MATCHES), dtype=float)
        if not hasattr(self, 'recent_count'):
            self.recent_count = np.zeros((n, team_count), dtype=np.int16)
        if not hasattr(self, 'last_match_date'):
            self.last_match_date = np.full((n, team_count), -1, dtype=np.int32)
        return legacy

    def replicate(self, n):
        self._ensure_auxiliary_state()
        result = copy.deepcopy(self)
        if self.values.shape[0] != 1:
            raise ValueError('Only a historical state can be replicated')
        result.values = np.repeat(self.values, n, axis=0)
        result.recent_points = np.repeat(self.recent_points, n, axis=0)
        result.recent_goal_difference = np.repeat(self.recent_goal_difference, n, axis=0)
        result.recent_count = np.repeat(self.recent_count, n, axis=0)
        result.last_match_date = np.repeat(self.last_match_date, n, axis=0)
        return result

    def start_season(self, year, teams):
        self._ensure_auxiliary_state()
        if year == self.year:
            return
        if self.year is not None and year < self.year:
            raise ValueError('State cannot travel backwards')
        self.values[:, :, :7] = BASE_STATE_PRIOR + .75 * (self.values[:, :, :7] - BASE_STATE_PRIOR)
        self.values[:, :, 7:11] = 0
        self.values[:, :, DRAW_STATE_INDEX] = .75 * self.values[:, :, DRAW_STATE_INDEX] + .25 * .25
        self.recent_points.fill(0)
        self.recent_goal_difference.fill(0)
        self.recent_count.fill(0)
        self.last_match_date.fill(-1)
        for team in teams:
            if self.year is not None and team not in self.current_teams:
                self.values[:, self.index[team], :7] = PROMOTED_STATE_PRIOR
                self.values[:, self.index[team], 10] = 1
                self.values[:, self.index[team], DRAW_STATE_INDEX] = .25
        self.previous_teams, self.current_teams = self.current_teams, set(teams)
        self.year = year

    def features(self, fixtures):
        legacy = self._ensure_auxiliary_state()
        hi = [self.index[m.home] for m in fixtures]
        ai = [self.index[m.away] for m in fixtures]
        h, a = self.values[:, hi], self.values[:, ai]
        # Four prior games stabilize early-season summaries.
        hppg, appg = (h[..., 7] + 5.2) / (h[..., 9] + 4), (a[..., 7] + 5.2) / (a[..., 9] + 4)
        phase = np.stack([np.full(h.shape[0], (m.matchday - 1) / 37) if m.matchday is not None
                          else np.minimum((h[:, j, 9] + a[:, j, 9]) / 74, 1.)
                          for j, m in enumerate(fixtures)], axis=1)

        hp = self.recent_points[:, hi]
        ap = self.recent_points[:, ai]
        hgd = self.recent_goal_difference[:, hi]
        agd = self.recent_goal_difference[:, ai]
        hc = self.recent_count[:, hi]
        ac = self.recent_count[:, ai]
        h_recent_ppg = (hp.sum(axis=2) + 4 * 1.3) / (hc + 4)
        a_recent_ppg = (ap.sum(axis=2) + 4 * 1.3) / (ac + 4)
        h_recent_gd = hgd.sum(axis=2) / (hc + 4)
        a_recent_gd = agd.sum(axis=2) / (ac + 4)

        fixture_days = np.asarray([m.date.toordinal() for m in fixtures], dtype=np.int32)
        h_last = self.last_match_date[:, hi]
        a_last = self.last_match_date[:, ai]
        h_rest = np.where(h_last >= 0, np.clip(fixture_days[None, :] - h_last, 0, 60), 30)
        a_rest = np.where(a_last >= 0, np.clip(fixture_days[None, :] - a_last, 0, 60), 30)
        combined_goal_rate = (h[..., 1] + h[..., 2] + a[..., 1] + a[..., 2]) / 2

        base = np.stack((h[..., 0], a[..., 0], h[..., 0] - a[..., 0],
                         h[..., 1], h[..., 2], a[..., 1], a[..., 2],
                         h[..., 3], h[..., 4], a[..., 5], a[..., 6],
                         hppg, appg, h[..., 8] / (h[..., 9] + 4),
                         a[..., 8] / (a[..., 9] + 4), h[..., 9], a[..., 9],
                         h[..., 10], a[..., 10], phase,
                         h_recent_ppg, a_recent_ppg, h_recent_gd, a_recent_gd,
                         h[..., DRAW_STATE_INDEX], a[..., DRAW_STATE_INDEX],
                         h_rest - a_rest, np.abs(h[..., 0] - a[..., 0]),
                         combined_goal_rate, h_recent_ppg - a_recent_ppg,
                         h_recent_gd - a_recent_gd,
                         (h[..., DRAW_STATE_INDEX] + a[..., DRAW_STATE_INDEX]) / 2), axis=-1)
        if legacy:
            return base[..., :20]
        if self.context is None:
            return base
        context_values = self.context.values_for(fixtures)
        context_values = np.broadcast_to(context_values[None, ...],
                                         (base.shape[0], *context_values.shape))
        return np.concatenate((base, context_values), axis=-1)

    def update(self, fixtures, hg, ag):
        """Update after the whole date is predicted (no intraday leakage)."""
        self._ensure_auxiliary_state()
        hg, ag = np.asarray(hg), np.asarray(ag)
        for j, m in enumerate(fixtures):
            home = self.values[:, self.index[m.home]]
            away = self.values[:, self.index[m.away]]
            gh, ga = hg[..., j], ag[..., j]
            expected = 1 / (1 + 10 ** (-(home[:, 0] + 65 - away[:, 0]) / 400))
            delta = 20 * ((gh > ga) + .5 * (gh == ga) - expected)
            home[:, 0] += delta
            away[:, 0] -= delta
            for arr, gf, gc, venue in ((home, gh, ga, 3), (away, ga, gh, 5)):
                arr[:, 1] = .94 * arr[:, 1] + .06 * gf
                arr[:, 2] = .94 * arr[:, 2] + .06 * gc
                arr[:, venue] = .90 * arr[:, venue] + .10 * gf
                arr[:, venue + 1] = .90 * arr[:, venue + 1] + .10 * gc
                arr[:, 7] += 3 * (gf > gc) + (gf == gc)
                arr[:, 8] += gf - gc
                arr[:, 9] += 1
                arr[:, DRAW_STATE_INDEX] = .94 * arr[:, DRAW_STATE_INDEX] + .06 * (gf == gc)
            for team_index, points, goal_difference in (
                    (self.index[m.home], 3 * (gh > ga) + (gh == ga), gh - ga),
                    (self.index[m.away], 3 * (ga > gh) + (ga == gh), ga - gh)):
                self.recent_points[:, team_index, :-1] = self.recent_points[:, team_index, 1:]
                self.recent_goal_difference[:, team_index, :-1] = self.recent_goal_difference[:, team_index, 1:]
                self.recent_points[:, team_index, -1] = points
                self.recent_goal_difference[:, team_index, -1] = goal_difference
                self.recent_count[:, team_index] = np.minimum(
                    self.recent_count[:, team_index] + 1, RECENT_MATCHES)
                self.last_match_date[:, team_index] = m.date.toordinal()


def batches(matches):
    return groupby(sorted(matches, key=lambda m: (m.date, m.home, m.away)), key=lambda m: (m.season, m.date))


def build_features(matches, extra_teams=(), context=None):
    state = SeasonState({t for m in matches for t in (m.home, m.away)} | set(extra_teams), context=context)
    participants = {s: {t for m in matches if m.season == s for t in (m.home, m.away)}
                    for s in {m.season for m in matches}}
    rows, features = [], []
    for (season, _), batch in batches(matches):
        batch = list(batch)
        state.start_season(int(season[:4]), participants[season])
        features.extend(state.features(batch)[0])
        rows.extend(batch)
        state.update(batch, [[m.hg for m in batch]], [[m.ag for m in batch]])
    return rows, np.asarray(features), state
