"""Live, streamed simulation of the rest of the current Premier League season.

    ./.venv/bin/python -m src.scripts.live_season
    ./.venv/bin/python -m src.scripts.live_season --seed 7 --delay 0.15
    ./.venv/bin/python -m src.scripts.live_season --calendar data/epl_2026-27.ics

One simulated "universe" is played out game by game from today's real table.
Each game's probabilities come from the deployed model given everything that
has happened so far in this universe (real results + simulated ones); a score
is drawn, the table updates, and after every match date the remaining season
is re-simulated (--projection-sims times) to update title / top-4 /
relegation odds. Different --seed values give different universes.
Nothing is written to disk.
"""
from __future__ import annotations

import argparse
import time
import warnings
from collections import deque
from datetime import datetime, timezone
from pathlib import Path

import numpy as np
import plotext as plt
from rich.console import Console, Group
from rich.layout import Layout
from rich.live import Live
from rich.panel import Panel
from rich.table import Table
from rich.text import Text
from threadpoolctl import threadpool_limits

from src.matchsense.baseline import BaselineEngine
from src.matchsense.config import load_settings
from src.matchsense.http import HttpCache, HttpClient
from src.matchsense.providers.fpl import FPLProvider
from src.matchsense.schemas import FixtureStatus
from src.ml.season_data import Match
from src.ml.season_features import batches
from src.ml.season_models import outcomes
from src.scripts.live_backtest import short
from src.scripts.simulate_rest_of_season import read_calendar

COLOURS = ["cyan", "magenta", "yellow", "green", "red", "blue"]
FORM_STYLE = {"W": "bold black on green", "D": "bold black on yellow", "L": "bold white on red"}


class Universe:
    def __init__(self, args, console: Console):
        self.args, self.c = args, console
        with console.status("[bold cyan]pulling fixtures and results from the Premier League FPL feed...",
                            spinner="dots12"):
            s = load_settings()
            fx = FPLProvider(HttpClient(HttpCache(s.cache_path), s.user_agent, s.http_timeout)).all_fixtures()
        self.eng = BaselineEngine(s.baseline_bundle)
        done = sorted([f for f in fx if f.status == FixtureStatus.FINISHED and f.home_goals is not None],
                      key=lambda f: f.kickoff_utc)
        left = [f for f in fx if f not in done]
        self.real = {(f.home, f.away): (f.home_goals, f.away_goals) for f in done}
        if args.from_start:  # simulate all 380 from the pre-season state; real scores shown for comparison
            done, left = [], list(fx)
        self.kick = {(f.home, f.away): f.kickoff_utc for f in left}
        if args.calendar:
            cal = read_calendar(args.calendar)
            for k in self.kick:
                if k in cal:
                    self.kick[k] = cal[k][0]
            console.log(f"[cyan]calendar[/] {len(set(cal) & set(self.kick))}/{len(left)} remaining games matched")
        with console.status("[bold cyan]rolling the model state through this season's real results...",
                            spinner="dots12"):
            dates = [f.kickoff_utc.date() for f in fx]
            state, completed = self.eng.state_for(fx[0].season, fx, min(dates) if args.from_start else max(dates),
                                                  datetime.now(timezone.utc))
        self.season = fx[0].season
        self.teams = sorted({t for f in fx for t in (f.home, f.away)})
        self.ix = {t: i for i, t in enumerate(self.teams)}
        z = lambda: np.zeros(len(self.teams))
        self.pts, self.gf, self.ga, self.w, self.d, self.l, self.p = (z() for _ in range(7))
        self.form = {t: deque(maxlen=5) for t in self.teams}
        for f in done:
            self.record(f.home, f.away, f.home_goals, f.away_goals)
        self.n_real = len(done)
        self.total = len(done) + len(left)
        self.ordered = sorted([Match(f.season, self.kick[(f.home, f.away)].date(), f.home, f.away, matchday=f.matchday)
                               for f in left], key=lambda m: (m.date, m.home, m.away))
        self.state = state.replicate(1)
        self.nfeat = self.eng.model.goal_model.scaler.n_features_in_
        self.rng = np.random.default_rng(args.seed)
        self.proj_rng = np.random.default_rng(args.seed + 1)
        self.played = 0
        self.current = None
        self.feed = deque(maxlen=60)
        self.points_hist = [(self.n_real, self.pts.copy())]
        self.title_hist = []
        self.proj = None
        self.upsets = []
        self.mode = ("simulated from matchweek 1" if args.from_start
                     else f"{len(done)} real + {len(left)} simulated")
        console.log(f"[cyan]{self.season}[/] {len(done)} real results locked in, {len(left)} games to play, "
                    f"universe seed {args.seed}")

    # ---------- bookkeeping ----------
    def record(self, home, away, hg, ag):
        h, a = self.ix[home], self.ix[away]
        self.gf[h] += hg; self.ga[h] += ag; self.gf[a] += ag; self.ga[a] += hg
        self.p[h] += 1; self.p[a] += 1
        if hg > ag:
            self.pts[h] += 3; self.w[h] += 1; self.l[a] += 1
            self.form[home].append("W"); self.form[away].append("L")
        elif hg < ag:
            self.pts[a] += 3; self.w[a] += 1; self.l[h] += 1
            self.form[home].append("L"); self.form[away].append("W")
        else:
            self.pts[h] += 1; self.pts[a] += 1; self.d[h] += 1; self.d[a] += 1
            self.form[home].append("D"); self.form[away].append("D")

    def order(self):
        key = self.pts * 1e6 + (self.gf - self.ga) * 1e3 + self.gf
        return [self.teams[i] for i in np.argsort(-key, kind="stable")]

    # ---------- simulation ----------
    def project(self, remaining, n):
        """Re-simulate the rest of the season n times from this universe's current state."""
        if not remaining:
            final = {t: r for r, t in enumerate(self.order(), 1)}
            rank = np.array([[final[t] for t in self.teams]])
        else:
            st = self.state.replicate(n)
            P, GF, GA = (np.tile(v, (n, 1)) for v in (self.pts, self.gf, self.ga))
            for _, grp in batches(remaining):
                grp = list(grp)
                x = st.features(grp)
                x = x.reshape(-1, x.shape[-1])[:, :self.nfeat]
                g = self.eng.model.predict_grid(x)
                cdf = np.cumsum(g.reshape(len(g), -1), axis=1)
                cdf[:, -1] = 1
                dr = (cdf < self.proj_rng.random((len(g), 1))).sum(1)
                sc = np.column_stack((dr // g.shape[2], dr % g.shape[2])).reshape(n, len(grp), 2)
                for j, m in enumerate(grp):
                    h, a = self.ix[m.home], self.ix[m.away]
                    hg, ag = sc[:, j, 0], sc[:, j, 1]
                    GF[:, h] += hg; GA[:, h] += ag; GF[:, a] += ag; GA[:, a] += hg
                    P[:, h] += 3 * (hg > ag) + (hg == ag)
                    P[:, a] += 3 * (ag > hg) + (hg == ag)
                st.update(grp, sc[:, :, 0], sc[:, :, 1])
            key = P * 1e6 + (GF - GA) * 1e3 + GF + self.proj_rng.random(P.shape) * .5
            rank = (-key).argsort(1).argsort(1) + 1
        self.proj = {t: ((rank[:, i] == 1).mean(), (rank[:, i] <= 4).mean(), (rank[:, i] >= 18).mean())
                     for t, i in self.ix.items()}
        self.title_hist.append((self.n_real + self.played, {t: v[0] for t, v in self.proj.items()}))

    def run(self):
        """Generator: yields after every simulated game and after every re-projection."""
        self.project(self.ordered, self.args.projection_sims)
        yield "projection"
        consumed = 0
        for _, grp in batches(self.ordered):
            grp = sorted(grp, key=lambda m: (self.kick[(m.home, m.away)], m.home))
            x = self.state.features(grp)[0][:, :self.nfeat]
            grids = self.eng.model.predict_grid(x)
            scores = []
            for m, g in zip(grp, grids):
                p = outcomes(g[None])[0]
                p = p / p.sum()
                cdf = np.cumsum(g.ravel())
                cdf[-1] = 1
                k = int((cdf < self.rng.random()).sum())
                hg, ag = k // g.shape[1], k % g.shape[1]
                y = 0 if hg > ag else (1 if hg == ag else 2)
                self.record(m.home, m.away, hg, ag)
                self.played += 1
                rates = ((g.sum(1) * np.arange(g.shape[0])).sum(), (g.sum(0) * np.arange(g.shape[1])).sum())
                self.current = (m, p, hg, ag, y, rates)
                self.feed.append((m, hg, ag, p, y))
                if p[y] < .2:
                    self.upsets.append((p[y], m, hg, ag))
                self.points_hist.append((self.n_real + self.played, self.pts.copy()))
                scores.append((hg, ag))
                yield "game"
            self.state.update(grp, [[s[0] for s in scores]], [[s[1] for s in scores]])
            consumed += len(grp)
            self.project(self.ordered[consumed:], self.args.projection_sims)
            yield "projection"


class Board:
    def __init__(self, u: Universe, console: Console):
        self.u, self.c = u, console

    def chart(self, w, h, draw, ok=True):
        if not ok:
            return Text("\n" * (h // 2) + "warming up...".center(w), style="dim")
        plt.clear_figure()
        plt.theme("dark")
        plt.plotsize(max(w, 20), max(h, 6))
        draw()
        return Text.from_ansi(plt.build())

    def contenders(self):
        best = {}
        for _, probs in self.u.title_hist:
            for t, p in probs.items():
                best[t] = max(best.get(t, 0), p)
        return [t for t, p in sorted(best.items(), key=lambda kv: -kv[1]) if p >= .02][:6]

    def title_chart(self, w, h):
        teams = self.contenders()
        hist = self.u.title_hist
        def draw():
            xs = [g for g, _ in hist]
            for t, col in zip(teams, COLOURS):
                plt.plot(xs, [100 * probs[t] for _, probs in hist], color=col, marker="braille")
            plt.title("title chance % (re-simulated after every match date)")
            plt.ylim(0, 100)
            plt.xlim(self.u.n_real, self.u.total)
        key = Text.assemble(*[(f"━{short(t)[:10]} ", col) for t, col in zip(teams, COLOURS)])
        return Panel(self.chart(w, h, draw, len(hist) >= 2), subtitle=key, border_style="cyan", padding=0)

    def race_chart(self, w, h):
        top = self.u.order()[:6]
        hist = self.u.points_hist
        def draw():
            xs = [g for g, _ in hist]
            for t, col in zip(top, COLOURS):
                plt.plot(xs, [pts[self.u.ix[t]] for _, pts in hist], color=col, marker="braille")
            plt.title("points race: current top 6")
            plt.xlim(self.u.n_real, self.u.total)
        key = Text.assemble(*[(f"━{short(t)[:10]} ", col) for t, col in zip(top, COLOURS)])
        return Panel(self.chart(w, h, draw, len(hist) >= 2), subtitle=key, border_style="cyan", padding=0)

    def card(self, width):
        u = self.u
        if u.current is None:
            return Panel(Text("kick-off pending...", style="dim"), title="match centre", border_style="cyan")
        m, p, hg, ag, y, rates = u.current
        when = u.kick[(m.home, m.away)].strftime("%a %d %b %H:%M")
        head = Text.assemble((f"MW{m.matchday} ", "bold white"), (f"{when}   ", "dim"),
                             (f"{m.home} ", "bold"), (f" {hg} - {ag} ", "bold black on white"), (f" {m.away}", "bold"))
        barw = max(4, (width - 44) // 3)  # 3 x ("HOME " + bar + " 50%  ") must fit inside the panel
        bars = Text()
        for k, lab in enumerate(("HOME", "DRAW", "AWAY")):
            style = "bold green" if k == y else "grey50"
            n = int(round(p[k] * barw * 1.5))
            bars.append(f"{lab} ", style="bold" if k == y else "dim")
            bars.append("█" * min(n, barw) + " " * max(0, barw - n), style=style)
            bars.append(f" {p[k]:.0%}  ", style=style)
        tag = Text("")
        if p[y] < .2:
            tag = Text(f"  ⚡ UPSET: only a {p[y]:.0%} chance  ", style="bold black on yellow")
        elif p[y] > .6:
            tag = Text(f"  favourite delivered ({p[y]:.0%})  ", style="bold black on green")
        info = Text(f"model expected goals {rates[0]:.2f} - {rates[1]:.2f}", style="dim")
        real = u.real.get((m.home, m.away)) if u.args.from_start else None
        if real:
            same = (real[0] > real[1]) - (real[0] < real[1]) == (hg > ag) - (hg < ag)
            info.append(f"    REAL RESULT {real[0]}-{real[1]} ", style="bold black on white")
            info.append(" same result ✔" if same else " different ✘", style="bold green" if same else "bold red")
        return Panel(Group(head, Text(""), bars, Text(""), Group(tag, info)),
                     title=f"match centre  {u.played}/{len(u.ordered)} simulated", border_style="cyan")

    def table(self):
        u = self.u
        t = Table(box=None, padding=(0, 1), header_style="bold cyan", expand=True)
        for col, j in (("#", "right"), ("team", "left"), ("P", "right"), ("W", "right"), ("D", "right"),
                       ("L", "right"), ("GD", "right"), ("Pts", "right"), ("form", "left"),
                       ("title", "right"), ("top4", "right"), ("rel", "right")):
            t.add_column(col, justify=j, no_wrap=True)
        current = u.current[0] if u.current else None
        for r, team in enumerate(u.order(), 1):
            i = u.ix[team]
            form = Text()
            for ch in u.form[team]:
                form.append(f"{ch}", style=FORM_STYLE[ch])
                form.append(" ")
            ti, t4, rel = u.proj.get(team, (0, 0, 0)) if u.proj else (0, 0, 0)
            zone = "green" if r <= 4 else ("red" if r >= 18 else "")
            name = Text(short(team)[:14], style=("bold reverse " if current and team in (current.home, current.away)
                                                 else "bold ") + zone)
            t.add_row(Text(str(r), style=zone), name, f"{u.p[i]:.0f}", f"{u.w[i]:.0f}", f"{u.d[i]:.0f}",
                      f"{u.l[i]:.0f}", f"{u.gf[i] - u.ga[i]:+.0f}", Text(f"{u.pts[i]:.0f}", style="bold"), form,
                      Text(f"{ti:.0%}" if ti >= .005 else "·", style="bold cyan" if ti >= .2 else "cyan"),
                      Text(f"{t4:.0%}" if t4 >= .005 else "·", style="green"),
                      Text(f"{rel:.0%}" if rel >= .005 else "·", style="bold red" if rel >= .5 else "red"))
        return Panel(t, title=f"LIVE TABLE  {u.season}  (green = top 4, red = relegation)", border_style="green")

    def results(self, height):
        t = Table.grid(padding=(0, 1))
        for _ in range(5):
            t.add_column(no_wrap=True)
        for m, hg, ag, p, y in list(self.u.feed)[-height:]:
            style = "bold yellow" if p[y] < .2 else ""
            real = self.u.real.get((m.home, m.away)) if self.u.args.from_start else None
            t.add_row(Text(f"MW{m.matchday}", style="dim"),
                      Text(f"{short(m.home)[:12]:>12} {hg}-{ag} {short(m.away)[:12]:<12}", style=style),
                      Text(f"{p[y]:.0%}", style="dim"), Text("⚡" if p[y] < .2 else " "),
                      Text(f"real {real[0]}-{real[1]}" if real else "", style="dim cyan"))
        return Panel(t, title="results (chance the model gave that result)", border_style="magenta")

    def render(self):
        u = self.u
        W, H = self.c.size
        done = u.n_real + u.played
        bar = int(30 * done / u.total)
        date = u.current[0].date.strftime("%d %b %Y") if u.current else "today"
        head = Text.assemble((f" MATCHSENSE // LIVE SEASON SIM {u.season} ", "bold black on cyan"), "  ",
                             ("━" * bar, "bold cyan"), ("━" * (30 - bar), "grey30"),
                             (f"  {done}/{u.total} games   ", "bold"), (f"{date}   ", "bold yellow"),
                             (f"universe #{u.args.seed}  ({u.mode})", "dim"))
        root = Layout()
        root.split_column(Layout(Panel(head, border_style="cyan"), size=3), Layout(name="body"))
        body_h = H - 3
        left_w = max(70, int(W * .46))
        root["body"].split_row(Layout(name="left", size=left_w), Layout(name="right"))
        root["left"].split_column(Layout(self.card(left_w), size=9), Layout(self.table()))
        right_w = W - left_w
        ch = max(10, body_h * 3 // 5)
        root["right"].split_column(Layout(name="charts", size=ch), Layout(name="bottom"))
        cw = right_w // 2 - 3
        root["charts"].split_row(Layout(self.title_chart(cw, ch - 3)), Layout(self.race_chart(cw, ch - 3)))
        root["bottom"].update(self.results(max(3, body_h - ch - 2)))
        return root

    def summary(self):
        u, c = self.u, self.c
        order = u.order()
        c.rule(f"[bold cyan]FINAL TABLE: universe #{u.args.seed}")
        champ = order[0]
        start = u.title_hist[0][1]
        c.print(f"🏆 [bold yellow]{champ}[/] are champions on {u.pts[u.ix[champ]]:.0f} points "
                f"(title chance at the start of this run: {start[champ]:.1%})")
        c.print(f"🔵 Top 4: {', '.join(order[:4])}")
        c.print(f"🔻 Relegated: {', '.join(order[-3:])}")
        if u.upsets:
            p, m, hg, ag = min(u.upsets, key=lambda x: x[0])
            c.print(f"⚡ Biggest upset: {m.home} {hg}-{ag} {m.away} (model gave it {p:.0%}); "
                    f"{len(u.upsets)} results under 20% in total")
        c.print("[dim]One possible universe. Run with another --seed for a different season; "
                "simulate_rest_of_season gives the odds across 10,000 universes.[/]")


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--seed", type=int, default=1, help="which simulated universe to play")
    ap.add_argument("--delay", type=float, default=0.25, help="seconds between games")
    ap.add_argument("--projection-sims", type=int, default=400, help="re-simulations per match date for odds")
    ap.add_argument("--calendar", type=Path, help="Premier League .ics file for kickoff times")
    ap.add_argument("--from-start", action="store_true",
                    help="simulate all 380 games from matchweek 1 (real scores shown alongside for played games)")
    args = ap.parse_args()
    warnings.simplefilter("ignore")
    console = Console()
    console.rule("[bold cyan]MatchSense live season simulator")
    with threadpool_limits(limits=1):
        u = Universe(args, console)
        board = Board(u, console)
        with Live(board.render(), console=console, refresh_per_second=12, transient=False) as live:
            for event in u.run():
                live.update(board.render())
                if event == "game":
                    time.sleep(args.delay)
        board.summary()


if __name__ == "__main__":
    main()
