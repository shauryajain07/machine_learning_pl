"""Simulate the rest of the current Premier League season from today.

    ./.venv/bin/python -m src.scripts.simulate_rest_of_season
    ./.venv/bin/python -m src.scripts.simulate_rest_of_season --calendar epl.ics --every-game
    ./.venv/bin/python -m src.scripts.simulate_rest_of_season --sims 20000 --seed 7

Real results so far (FPL feed) are locked in; every remaining fixture is
played out N times with the deployed ScoreModel. Each simulated path updates
its own team state (Elo, form) after every matchday, exactly like the
pre-season simulator.

--calendar takes a Premier League .ics export (e.g. the ECAL subscription).
It is cross-checked against FPL; its kickoff times are used for ordering.
--every-game prints all 380 games: actual results with the model's
pre-match forecast for played games, and simulated probabilities plus the
most likely score for every remaining game. --out writes the same as CSV.
"""
from __future__ import annotations

import argparse
import csv
import re
import time
import warnings
from collections import defaultdict
from datetime import datetime, timezone
from pathlib import Path

import numpy as np
from threadpoolctl import threadpool_limits

from src.matchsense.baseline import BaselineEngine
from src.matchsense.config import load_settings
from src.matchsense.http import HttpCache, HttpClient
from src.matchsense.providers.fpl import FPLProvider
from src.matchsense.schemas import FixtureStatus
from src.matchsense.teams import REGISTRY
from src.ml.season_data import Match
from src.ml.season_features import batches
from src.ml.season_models import OUTCOME, outcomes

PRESEASON = "artifacts/season_simulator/season_forecast.csv"
DEFAULT_OUT = "artifacts/season_simulator/every_game_forecast.csv"


def read_calendar(path: Path):
    """Return {(home, away): (kickoff_utc, matchweek)} for upcoming games in an .ics file."""
    text = path.read_text(encoding="utf-8").replace("\r\n ", "").replace("\n ", "")
    games = {}
    for block in text.split("BEGIN:VEVENT")[1:]:
        def field(key):
            m = re.search(rf"^{key}[^:]*:(.*)$", block, re.M)
            return m.group(1).strip() if m else None
        summary = re.sub(r"^[^A-Za-z]+", "", field("SUMMARY") or "")
        m = re.match(r"(.+?) v (.+)$", summary)
        start = field("DTSTART")
        if not m or not start or "T" not in start:
            continue  # welcome message, played games with scores, all-day events
        home, away = REGISTRY.canonical(m.group(1)), REGISTRY.canonical(m.group(2))
        if home is None or away is None:
            raise SystemExit(f"calendar team not recognised: {summary!r}")
        week = re.search(r"Matchweek (\d+)", field("DESCRIPTION") or "")
        games[(home, away)] = (datetime.strptime(start[:15], "%Y%m%dT%H%M%S").replace(tzinfo=timezone.utc),
                               int(week.group(1)) if week else None)
    return games


def likeliest(grid, outcome=None):
    """Most likely scoreline overall, or within one outcome (0 home, 1 draw, 2 away)."""
    g = grid if outcome is None else np.where(OUTCOME == outcome, grid, 0)
    i = int(np.argmax(g))
    return f"{i // grid.shape[1]}-{i % grid.shape[1]}", float(g.ravel()[i])


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--sims", type=int, default=10000, help="number of simulated seasons")
    ap.add_argument("--seed", type=int, default=20261104)
    ap.add_argument("--calendar", type=Path, help="Premier League .ics file to cross-check and order fixtures")
    ap.add_argument("--every-game", action="store_true", help="print all 380 games")
    ap.add_argument("--out", type=Path, default=Path(DEFAULT_OUT), help="CSV with every game's forecast")
    args = ap.parse_args()
    warnings.simplefilter("ignore")

    s = load_settings()
    fpl = FPLProvider(HttpClient(HttpCache(s.cache_path), s.user_agent, s.http_timeout))
    fx = fpl.all_fixtures()
    eng = BaselineEngine(s.baseline_bundle)
    done = [f for f in fx if f.status == FixtureStatus.FINISHED and f.home_goals is not None]
    left = [f for f in fx if f not in done]

    kickoff = {(f.home, f.away): f.kickoff_utc for f in left}
    if args.calendar:
        cal = read_calendar(args.calendar)
        missing = set(kickoff) - set(cal)
        moved = [k for k in set(kickoff) & set(cal) if abs((cal[k][0] - kickoff[k]).total_seconds()) > 60]
        print(f"calendar: {len(cal)} upcoming games listed, {len(set(kickoff) & set(cal))} match FPL's "
              f"{len(left)} remaining, {len(missing)} missing, {len(moved)} with a different kickoff "
              f"(calendar time used)")
        for k in moved:
            kickoff[k] = cal[k][0]

    state, completed = eng.state_for(fx[0].season, fx, max(f.kickoff_utc.date() for f in fx),
                                     datetime.now(timezone.utc))
    teams = sorted({t for f in fx for t in (f.home, f.away)})
    ix = {t: i for i, t in enumerate(teams)}

    pts0, gf0, ga0 = (np.zeros(len(teams)) for _ in range(3))
    for f in done:
        h, a = ix[f.home], ix[f.away]
        gf0[h] += f.home_goals; ga0[h] += f.away_goals; gf0[a] += f.away_goals; ga0[a] += f.home_goals
        pts0[h] += 3 * (f.home_goals > f.away_goals) + (f.home_goals == f.away_goals)
        pts0[a] += 3 * (f.away_goals > f.home_goals) + (f.home_goals == f.away_goals)
    print(f"{fx[0].season}: {len(completed)} real results locked in + {len(left)} fixtures to simulate "
          f"= {len(completed) + len(left)} games, x {args.sims:,} seasons")

    n, rng = args.sims, np.random.default_rng(args.seed)
    st = state.replicate(n)
    P, GF, GA = np.tile(pts0, (n, 1)), np.tile(gf0, (n, 1)), np.tile(ga0, (n, 1))
    by_key = {(f.home, f.away): f for f in left}
    matches = [Match(f.season, kickoff[(f.home, f.away)].date(), f.home, f.away, matchday=f.matchday) for f in left]
    nfeat = eng.model.goal_model.scaler.n_features_in_
    game_grid = {}
    t0 = time.time()
    with threadpool_limits(limits=1):
        for _, grp in batches(matches):
            grp = list(grp)
            x = st.features(grp)
            x = x.reshape(-1, x.shape[-1])[:, :nfeat]
            flat = np.empty((len(x), 2), dtype=np.int16)
            grid_sum = None
            for a in range(0, len(x), 4096):
                g = eng.model.predict_grid(x[a:a + 4096])
                if grid_sum is None:
                    grid_sum = np.zeros((len(grp),) + g.shape[1:])
                np.add.at(grid_sum, np.arange(a, a + len(g)) % len(grp), g)
                cdf = np.cumsum(g.reshape(len(g), -1), axis=1)
                cdf[:, -1] = 1
                d = (cdf < rng.random((len(g), 1))).sum(1)
                flat[a:a + len(g)] = np.column_stack((d // g.shape[2], d % g.shape[2]))
            sc = flat.reshape(n, len(grp), 2)
            for j, m in enumerate(grp):
                h, a = ix[m.home], ix[m.away]
                hg, ag = sc[:, j, 0], sc[:, j, 1]
                GF[:, h] += hg; GA[:, h] += ag; GF[:, a] += ag; GA[:, a] += hg
                P[:, h] += 3 * (hg > ag) + (hg == ag)
                P[:, a] += 3 * (ag > hg) + (hg == ag)
                game_grid[(m.home, m.away)] = grid_sum[j] / n
            st.update(grp, sc[:, :, 0], sc[:, :, 1])

    # Points, then goal difference, then goals scored, then a random draw.
    key = P * 1e6 + (GF - GA) * 1e3 + GF + rng.random(P.shape) * 0.5
    rank = (-key).argsort(1).argsort(1) + 1
    pre = {}
    try:
        pre = {r["team"]: float(r["title_probability"]) for r in csv.DictReader(open(PRESEASON))}
    except FileNotFoundError:
        pass
    rows = []
    for t in teams:
        i = ix[t]
        title = (rank[:, i] == 1).mean()
        rows.append((t, pts0[i], title, (rank[:, i] <= 4).mean(), (rank[:, i] >= 18).mean(), P[:, i].mean(),
                     np.percentile(P[:, i], 5), np.percentile(P[:, i], 95), pre.get(t, float("nan"))))
    rows.sort(key=lambda r: (-r[2], -r[5]))
    print(f"done in {time.time() - t0:.0f}s\n")
    print(f"{'team':24} {'pts':>4} {'TITLE':>7} {'fair odds':>9} {'top 4':>6} {'releg':>6} {'exp pts':>8} "
          f"{'90% range':>10} {'pre-season':>10}")
    for t, now, title, top4, rel, exp, lo, hi, pre_t in rows:
        odds = f"{1 / title:9.1f}" if title > 0 else f"{'>' + str(n):>9}"
        print(f"{t:24} {now:4.0f} {title:7.1%} {odds} {top4:6.1%} {rel:6.1%} {exp:8.1f} {lo:4.0f}-{hi:<5.0f} "
              f"{pre_t:10.1%}")

    # ---- every game: played (actual + pre-match forecast) and remaining (simulated) ----
    games = []
    for f in sorted(done, key=lambda f: f.kickoff_utc):
        pred = eng.predict(f, fx, as_of=f.kickoff_utc)
        p = pred.probabilities / pred.probabilities.sum()
        y = 0 if f.home_goals > f.away_goals else (1 if f.home_goals == f.away_goals else 2)
        games.append(dict(matchweek=f.matchday, kickoff_utc=f.kickoff_utc.isoformat(), home=f.home, away=f.away,
                          status="played", p_home=p[0], p_draw=p[1], p_away=p[2],
                          exp_home_goals=pred.rates[0], exp_away_goals=pred.rates[1],
                          pick="HDA"[int(p.argmax())], likeliest_score=likeliest(pred.grid)[0],
                          pick_score=likeliest(pred.grid, int(p.argmax()))[0],
                          actual=f"{f.home_goals}-{f.away_goals}", pick_correct=int(p.argmax()) == y))
    for k in sorted(game_grid, key=lambda k: kickoff[k]):
        g = game_grid[k] / game_grid[k].sum()
        p = outcomes(g[None])[0]
        pick = int(p.argmax())
        games.append(dict(matchweek=by_key[k].matchday, kickoff_utc=kickoff[k].isoformat(), home=k[0], away=k[1],
                          status="simulated", p_home=p[0], p_draw=p[1], p_away=p[2],
                          exp_home_goals=float((g.sum(1) * np.arange(g.shape[0])).sum()),
                          exp_away_goals=float((g.sum(0) * np.arange(g.shape[1])).sum()),
                          pick="HDA"[pick], likeliest_score=likeliest(g)[0], pick_score=likeliest(g, pick)[0],
                          actual="", pick_correct=""))
    args.out.parent.mkdir(parents=True, exist_ok=True)
    with args.out.open("w", newline="") as handle:
        w = csv.DictWriter(handle, fieldnames=list(games[0]))
        w.writeheader()
        for row in games:
            w.writerow({k: (round(v, 4) if isinstance(v, float) else v) for k, v in row.items()})
    played = [g for g in games if g["status"] == "played"]
    print(f"\n{len(games)} games written to {args.out}  (played: {len(played)}, model pick correct "
          f"{sum(g['pick_correct'] for g in played)}/{len(played)}; simulated: {len(games) - len(played)})")

    if args.every_game:
        by_week = defaultdict(list)
        for g in games:
            by_week[g["matchweek"]].append(g)
        for week in sorted(by_week, key=lambda w: (w is None, w or 0)):
            print(f"\n── Matchweek {week} " + "─" * 80)
            for g in by_week[week]:
                when = datetime.fromisoformat(g["kickoff_utc"]).strftime("%a %d %b %H:%M")
                probs = f"H {g['p_home']:4.0%}  D {g['p_draw']:4.0%}  A {g['p_away']:4.0%}"
                label = {"H": "HOME WIN", "D": "DRAW", "A": "AWAY WIN"}[g["pick"]]
                tail = (f"actual {g['actual']:>5}  {'✔' if g['pick_correct'] else '✘'}" if g["status"] == "played"
                        else f"likeliest score {g['likeliest_score']}")
                print(f"{when}  {g['home']:>24} v {g['away']:<24} {probs}  → {label:8} {g['pick_score']:>5}  {tail}")
    print("\nResults-only model: no injuries, lineups, transfers, odds or news. Fair odds = 1 / probability.")


if __name__ == "__main__":
    main()
