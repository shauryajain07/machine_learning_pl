"""Team-match lineup histories and leakage-safe selection features.

The same feature code serves two sources:

* training: ``1-premierleague-full.txt`` lineups parsed by the repository's
  Football.TXT parser (2025-26 is the only season with lineups locally);
* inference: FPL per-fixture ``starts``/``minutes`` for the current season.

Every feature for a target match is computed only from the team's earlier
matches (``kickoff < target kickoff``).
"""
from __future__ import annotations

import math
from dataclasses import dataclass, field
from datetime import datetime, timezone
from pathlib import Path

import numpy as np

from src.ingestion.football_txt.file_parser import parse_file
from src.ingestion.normalization.aliases import load_alias_maps
from src.matchsense.config import PROJECT_ROOT
from src.matchsense.players import name_key

TYPICAL_HALF_LIFE_MATCHES = 4.0
SELECTION_FEATURES = ("ew_start_rate", "started_last", "starts_last3", "ew_minutes_share", "short_rest",
                      "started_last_x_short_rest", "log_appearances", "is_goalkeeper", "never_started")


@dataclass
class TeamMatch:
    team: str
    opponent: str
    kickoff: datetime
    is_home: bool
    starters: list[str]
    minutes: dict[str, int]
    goals_for: int | None = None
    goals_against: int | None = None
    goalkeeper: str | None = None
    match_key: str = ""
    matchday: int | None = None
    scorers: dict[str, int] = field(default_factory=dict)  # Non-own goals by player key.
    defenders: set[str] = field(default_factory=set)  # Starters in goal or the back line.


@dataclass
class PlayerState:
    ew_start: float = 0.0
    ew_minutes: float = 0.0
    ew_goals: float = 0.0
    defensive_starts: int = 0
    weight: float = 0.0  # Total decayed weight of observed team matches.
    last: list[int] = field(default_factory=list)
    appearances: int = 0
    starts: int = 0
    is_gk: bool = False


def _minutes(appearance) -> int:
    end = min(appearance.exited.absolute_minute, 90) if appearance.exited else 90
    start = min(appearance.entered.absolute_minute, 90) if appearance.entered else 0
    return max(end - start, 0) if (appearance.started or appearance.entered) else 0


def load_openfootball_lineups(path: Path | None = None,
                              alias_file: Path = PROJECT_ROOT / "config" / "team_aliases.json") -> list[TeamMatch]:
    path = path or PROJECT_ROOT / "england" / "2025-26" / "1-premierleague-full.txt"
    aliases = load_alias_maps(alias_file)
    parsed = parse_file(path, source_file=str(path.relative_to(path.parents[1])))
    rows = []
    for match in parsed.matches:
        if match.score is None or match.kickoff_utc is None:
            continue
        home, away = aliases.target("team", match.home_source_name), aliases.target("team", match.away_source_name)
        kickoff = match.kickoff_utc if match.kickoff_utc.tzinfo else match.kickoff_utc.replace(tzinfo=timezone.utc)
        for side, team, opp, gf, ga in (("home", home, away, match.score.home_score, match.score.away_score),
                                        ("away", away, home, match.score.away_score, match.score.home_score)):
            apps = [a for a in match.appearances if a.side == side]
            starters = [f"{team}|{name_key(a.player_name)}" for a in apps if a.started]
            minutes = {f"{team}|{name_key(a.player_name)}": _minutes(a) for a in apps}
            gk = next((f"{team}|{name_key(a.player_name)}" for a in apps if a.started and a.raw_group_index == 0), None)
            scorers: dict[str, int] = {}
            for goal in match.goals:
                if goal.side == side and not goal.own_goal:
                    key = f"{team}|{name_key(goal.player_name)}"
                    scorers[key] = scorers.get(key, 0) + 1
            rows.append(TeamMatch(team, opp, kickoff, side == "home", starters, minutes, gf, ga, gk,
                                  match_key=f"{match.season_label}|{match.match_date}|{home}|{away}",
                                  matchday=match.round_info.round_number if match.round_info else None,
                                  scorers=scorers,
                                  # Football.TXT lists GK, then the back line: groups 0 and 1.
                                  defenders={f"{team}|{name_key(a.player_name)}" for a in apps
                                             if a.started and a.raw_group_index in (0, 1)}))
    return sorted(rows, key=lambda r: (r.kickoff, r.team))


def display_names(path: Path | None = None) -> dict[str, str]:
    """Player key -> source spelling (for mapping historical effects to squads)."""
    path = path or PROJECT_ROOT / "england" / "2025-26" / "1-premierleague-full.txt"
    aliases = load_alias_maps(PROJECT_ROOT / "config" / "team_aliases.json")
    names = {}
    for match in parse_file(path, source_file=str(path.relative_to(path.parents[1]))).matches:
        for a in match.appearances:
            team = aliases.target("team", match.home_source_name if a.side == "home" else match.away_source_name)
            names[f"{team}|{name_key(a.player_name)}"] = a.player_name
    return names


class SelectionState:
    """Running per-team player selection state; update only after predicting."""

    def __init__(self):
        self.players: dict[str, PlayerState] = {}
        self.last_kickoff: datetime | None = None
        self.matches = 0

    def update(self, match: TeamMatch):
        decay = math.exp2(-1 / TYPICAL_HALF_LIFE_MATCHES)
        for state in self.players.values():
            state.ew_start *= decay
            state.ew_minutes *= decay
            state.ew_goals *= decay
            state.weight *= decay
        involved = set(match.minutes) | set(match.starters) | set(match.scorers)
        for key in involved:
            self.players.setdefault(key, PlayerState())
        for key, state in self.players.items():
            started = key in match.starters
            state.weight += 1
            state.ew_start += float(started)
            state.ew_minutes += match.minutes.get(key, 0) / 90
            state.ew_goals += match.scorers.get(key, 0)
            state.last = (state.last + [int(started)])[-3:]
            if match.minutes.get(key, 0) > 0 or started:
                state.appearances += 1
            state.starts += int(started)
            state.defensive_starts += int(key in match.defenders)
            if match.goalkeeper == key:
                state.is_gk = True
        self.last_kickoff = match.kickoff
        self.matches += 1

    def typical_rate(self, key: str) -> float:
        state = self.players.get(key)
        return state.ew_start / state.weight if state and state.weight > 0 else 0.0

    def defensive_mask(self, keys: list[str]) -> "np.ndarray":
        """True for players who mostly start in goal or the back line."""
        return np.array([bool(s and s.starts and s.defensive_starts / s.starts >= 0.5)
                         for s in (self.players.get(k) for k in keys)])

    def goal_shares(self, keys: list[str]) -> "np.ndarray":
        """Each player's share of the team's recent (decayed) goals."""
        total = sum(s.ew_goals for s in self.players.values())
        if total <= 1e-9:
            return np.zeros(len(keys))
        return np.array([self.players[k].ew_goals / total if k in self.players else 0.0 for k in keys])

    def features(self, keys: list[str], kickoff: datetime, goalkeepers: set[str] | None = None) -> np.ndarray:
        rest = (kickoff - self.last_kickoff).total_seconds() / 86400 if self.last_kickoff else 7.0
        short = float(rest <= 4.0)
        rows = []
        for key in keys:
            s = self.players.get(key, PlayerState())
            rate = s.ew_start / s.weight if s.weight > 0 else 0.0
            minutes = s.ew_minutes / s.weight if s.weight > 0 else 0.0
            last = s.last[-1] if s.last else 0
            gk = float(s.is_gk or (goalkeepers is not None and key in goalkeepers))
            rows.append((rate, last, sum(s.last) / 3, minutes, short, last * short, math.log1p(s.appearances),
                         gk, float(s.starts == 0)))
        return np.asarray(rows, dtype=float).reshape(len(keys), len(SELECTION_FEATURES))
