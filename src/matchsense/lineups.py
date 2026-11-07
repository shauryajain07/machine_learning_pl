"""Probabilistic starting-XI prediction for one team.

P(start) = P(start | available) from the fitted selection model (prior
selections, rest) x P(available) from corroborated news. Valid XIs are
then sampled under formation constraints: exactly one goalkeeper and the
team's most frequent recent DEF/MID/FWD split (from FPL position classes,
so formations are approximate). Sampling uses Gumbel top-k on start odds,
an approximation to conditional Bernoulli sampling. Marginal start
probabilities shown in the UI are frequencies over the sampled XIs, so
they always respect the 11-player constraint.

When an official XI is available it replaces the prediction entirely.
"""
from __future__ import annotations

from collections import Counter
from dataclasses import dataclass, field
from datetime import datetime

import numpy as np

from src.matchsense.impact import SelectionModel
from src.matchsense.lineup_data import SelectionState, TeamMatch
from src.matchsense.players import resolve_player
from src.matchsense.schemas import (Availability, LineupSlot, PlayerAppearance, PlayerAvailability, Position,
                                    SquadPlayer, TeamLineup)

DEFAULT_SHAPE = (4, 3, 3)


class LineupError(RuntimeError):
    pass


@dataclass
class TeamLineupModel:
    team: str
    squad: list[SquadPlayer]
    pool: list[str]
    positions: np.ndarray  # Position value per pool entry
    typical: np.ndarray
    goal_share: np.ndarray
    p_selection: np.ndarray
    p_available: np.ndarray
    status: list[Availability]
    shape: tuple[int, int, int]
    shape_basis: str
    confirmed: np.ndarray | None = None
    unresolved_confirmed: list[str] = field(default_factory=list)
    history_matches: int = 0

    @property
    def p_raw(self) -> np.ndarray:
        return np.clip(self.p_selection * self.p_available, 0, 1)

    def sample(self, n: int, rng: np.random.Generator) -> np.ndarray:
        """(n, pool) 0/1 matrix of valid XIs."""
        if self.confirmed is not None:
            return np.repeat(self.confirmed[None, :], n, axis=0)
        return sample_lineups(self.p_raw, self.positions, self.shape, n, rng)

    def impact_inputs(self, starts: np.ndarray) -> dict:
        return {"pool": self.pool, "typical": self.typical, "starts": starts, "goal_share": self.goal_share,
                "def_mask": np.isin(self.positions, ("GK", "DEF"))}


def _shape_from_history(history: list[TeamMatch], positions: dict[str, str]) -> tuple[tuple[int, int, int], str]:
    shapes = []
    for match in history[-6:]:
        counts = Counter(positions.get(p) for p in match.starters)
        if counts.get("GK", 0) == 1 and sum(counts.get(k, 0) for k in ("DEF", "MID", "FWD")) == 10:
            shapes.append((counts["DEF"], counts["MID"], counts["FWD"]))
    if not shapes:
        return DEFAULT_SHAPE, "default 4-3-3 (no usable recent XIs)"
    shape, count = Counter(shapes).most_common(1)[0]
    return shape, f"most common FPL position split in last {len(shapes)} XIs ({count}x)"


def sample_lineups(p: np.ndarray, positions: np.ndarray, shape: tuple[int, int, int], n: int,
                   rng: np.random.Generator) -> np.ndarray:
    """Sample XIs: one GK, then outfielders by Gumbel key with per-line caps.

    Each line may hold at most one more player than the usual shape. The back
    line keeps its usual size (defenders are replaced like-for-like), while a
    missing striker is replaced by whoever is most likely to play.
    """
    p = np.clip(p, 1e-6, 1 - 1e-6)
    logit = np.log(p) - np.log1p(-p)
    out = np.zeros((n, len(p)), dtype=float)
    if (positions == "GK").sum() < 1 or (positions != "GK").sum() < 10:
        raise LineupError("Squad lacks a goalkeeper or ten outfield players")
    for i in range(n):
        out[i, pick_xi(logit + rng.gumbel(size=len(p)), positions, shape)] = 1
    return out


def pick_xi(keys: np.ndarray, positions: np.ndarray, shape: tuple[int, int, int]) -> list[int]:
    """Highest-key valid XI: 1 GK, line caps of shape+1, back line at least its usual size."""
    caps = {"DEF": shape[0] + 1, "MID": shape[1] + 1, "FWD": shape[2] + 1}
    gk_idx = np.flatnonzero(positions == "GK")
    outfield_idx = np.flatnonzero(positions != "GK")
    chosen = [int(gk_idx[np.argmax(keys[gk_idx])])]
    counts = {"DEF": 0, "MID": 0, "FWD": 0}
    ranked = [int(j) for j in outfield_idx[np.argsort(-keys[outfield_idx])]]
    for j in ranked:
        if len(chosen) == 11:
            break
        if counts[positions[j]] < caps[positions[j]]:
            chosen.append(j)
            counts[positions[j]] += 1
    for j in ranked:  # Caps too tight for this squad: fill regardless.
        if len(chosen) == 11:
            break
        if j not in chosen:
            chosen.append(j)
            counts[positions[j]] += 1
    # Defenders are replaced like-for-like: keep the usual back-line size when
    # enough defenders are available (attackers may still cover for attackers).
    while counts["DEF"] < max(3, shape[0]):
        spare = [j for j in ranked if positions[j] == "DEF" and j not in chosen]
        swap = [j for j in reversed(chosen[1:]) if positions[j] != "DEF"]
        if not spare or not swap:
            break
        chosen.remove(swap[0])
        counts[positions[swap[0]]] -= 1
        chosen.append(spare[0])
        counts["DEF"] += 1
    return chosen


def build_team_lineup_model(team: str, squad: list[SquadPlayer], appearances: list[PlayerAppearance],
                            kickoff: datetime, selection: SelectionModel,
                            availability: list[PlayerAvailability], confirmed_names: list[str] | None = None,
                            overrides: dict[str, float] | None = None) -> TeamLineupModel:
    if len(squad) < 11:
        raise LineupError(f"Only {len(squad)} squad players known for {team}")
    positions_by_id = {p.player_id: p.position.value for p in squad}
    history: list[TeamMatch] = []
    by_kickoff: dict[datetime, list[PlayerAppearance]] = {}
    for app in appearances:
        if app.kickoff_utc < kickoff:
            by_kickoff.setdefault(app.kickoff_utc, []).append(app)
    for moment in sorted(by_kickoff):
        apps = by_kickoff[moment]
        starters = [a.player_id for a in apps if a.started]
        gk = next((a.player_id for a in apps if a.started and positions_by_id.get(a.player_id) == "GK"), None)
        history.append(TeamMatch(team, "", moment, True, starters, {a.player_id: a.minutes for a in apps},
                                 goalkeeper=gk, scorers={a.player_id: a.goals for a in apps if a.goals}))
    state = SelectionState()
    for match in history:
        state.update(match)
    pool = [p.player_id for p in squad]
    gks = {p.player_id for p in squad if p.position == Position.GK}
    features = state.features(pool, kickoff, goalkeepers=gks)
    p_sel = selection.predict(features) if history else np.full(len(pool), 11 / len(pool))
    beliefs = {b.player_id: b for b in availability}
    overrides = overrides or {}
    p_avail = np.array([overrides.get(pid, beliefs[pid].p_available if pid in beliefs else 1.0) for pid in pool])
    status = [Availability.UNAVAILABLE if pid in overrides and overrides[pid] < 0.25 else
              beliefs[pid].status if pid in beliefs else Availability.AVAILABLE for pid in pool]
    shape, basis = _shape_from_history(history, positions_by_id)
    model = TeamLineupModel(team=team, squad=squad, pool=pool, positions=np.array([positions_by_id[p] for p in pool]),
                            typical=np.array([state.typical_rate(p) for p in pool]),
                            goal_share=state.goal_shares(pool), p_selection=p_sel, p_available=p_avail,
                            status=status, shape=shape, shape_basis=basis, history_matches=len(history))
    if confirmed_names:
        confirmed = np.zeros(len(pool))
        for name in confirmed_names:
            resolution = resolve_player(name, squad)
            if resolution.player is None:
                model.unresolved_confirmed.append(name)
            else:
                confirmed[pool.index(resolution.player.player_id)] = 1
        if confirmed.sum() + len(model.unresolved_confirmed) == 11 and not model.unresolved_confirmed:
            model.confirmed = confirmed
    return model


def describe(model: TeamLineupModel, samples: np.ndarray, source: str) -> TeamLineup:
    marginal = samples.mean(axis=0)
    by_id = {p.player_id: p for p in model.squad}
    slots = [LineupSlot(player_id=pid, name=by_id[pid].name, short_name=by_id[pid].short_name,
                        position=by_id[pid].position,
                        p_start=round(float(marginal[i]), 4), p_available=round(float(model.p_available[i]), 4),
                        typical_start_rate=round(float(model.typical[i]), 4),
                        uncertain=bool(0.2 < marginal[i] < 0.8) and model.confirmed is None,
                        status=model.status[i]) for i, pid in enumerate(model.pool)]
    # Display XI: the highest-marginal valid lineup under the same line caps.
    modal = set(pick_xi(marginal + 1e-9 * model.typical, model.positions, model.shape))
    xi = sorted((slots[i] for i in modal), key=lambda s: ("GK", "DEF", "MID", "FWD").index(s.position.value))
    rest = sorted((s for i, s in enumerate(slots) if i not in modal), key=lambda s: -s.p_start)
    excluded = [s for s in rest if s.p_available < 0.25]
    bench = [s for s in rest if s.p_available >= 0.25][:9]
    confirmed = model.confirmed is not None
    # Predicted: the team's usual shape; confirmed: the official XI's position classes.
    shape = model.shape if not confirmed else tuple(
        sum(1 for s in xi if s.position.value == g) for g in ("DEF", "MID", "FWD"))
    return TeamLineup(team=model.team, kind="confirmed" if confirmed else "predicted",
                      formation="-".join(map(str, shape)), starters=xi, bench=bench, excluded=excluded,
                      source=source if confirmed else "MatchSense selection model + availability evidence",
                      note=("Official starting XI." if confirmed else
                            f"Predicted, not confirmed. Shape: {model.shape_basis}. Built from {model.history_matches} "
                            "earlier matches this season."))
