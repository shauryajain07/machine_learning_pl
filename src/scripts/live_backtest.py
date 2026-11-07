"""Live terminal backtest of every 2026-27 Premier League prediction so far.

    ./.venv/bin/python -m src.scripts.live_backtest            # default speed
    ./.venv/bin/python -m src.scripts.live_backtest --delay 0.6

Each model variant is fitted once on matches through 2025-26 with the
deployed recipe (Poisson GLM, 1460-day half-life). Features for every
2026-27 match come from the deployed SeasonState pipeline and use only
results dated before that match, so the stream is a true walk-forward
replay. Only 2026-27 matches are scored. Nothing is written to disk.
"""
from __future__ import annotations

import argparse
import csv
import time
import warnings
from collections import defaultdict
from datetime import date

import numpy as np
import plotext as plt
from rich.align import Align
from rich.console import Console, Group
from rich.layout import Layout
from rich.live import Live
from rich.panel import Panel
from rich.table import Table
from rich.text import Text
from scipy.stats import spearmanr
from threadpoolctl import threadpool_limits

from src.matchsense.config import load_settings
from src.matchsense.http import HttpCache, HttpClient
from src.matchsense.providers.fpl import FPLProvider
from src.matchsense.schemas import FixtureStatus
from src.ml.season_data import Match
from src.ml.season_features import build_features
from src.ml.season_models import Candidate, FittedModel, outcomes, score_grid

HISTORY = "artifacts/season_simulator/historical_matches.csv"
ALL, ELO = list(range(32)), [0, 1, 2]
# name, feature columns, L2 alpha, Dixon-Coles rho (negative = more draws), colour
VARIANTS = [("deployed", ALL, .03, 0., "cyan"),
            ("elo-only", ELO, .03, 0., "magenta"),
            ("all feats a=1", ALL, 1., 0., "yellow"),
            ("deployed+draws", ALL, .03, -.10, "green")]
BASE = np.array([.45, .25, .30])
HDA = "HDA"


def short(team: str) -> str:
    for a, b in (("Manchester", "Man"), (" & Hove Albion", ""), (" Hotspur", ""), ("United", "Utd"),
                 ("Nottingham", "Nott'm"), ("Wolverhampton Wanderers", "Wolves"), ("Crystal Palace", "C Palace")):
        team = team.replace(a, b)
    return team


def load(console: Console):
    with console.status("[bold cyan]loading historical matches...", spinner="dots12"):
        hist = []
        with open(HISTORY) as h:
            for r in csv.DictReader(h):
                hist.append(Match(r["season"], date.fromisoformat(r["date"]), r["home"], r["away"], int(r["hg"]),
                                  int(r["ag"]), int(r["matchday"]) if r["matchday"] else None))
        console.log(f"[cyan]history[/] {len(hist):,} matches, {hist[0].season} to {hist[-1].season}")
    with console.status("[bold cyan]pulling 2026-27 results from the Premier League FPL feed...", spinner="dots12"):
        s = load_settings()
        fpl = FPLProvider(HttpClient(HttpCache(s.cache_path), s.user_agent, s.http_timeout))
        cur = [Match(f.season, f.kickoff_utc.date(), f.home, f.away, f.home_goals, f.away_goals, f.matchday)
               for f in fpl.all_fixtures() if f.status == FixtureStatus.FINISHED and f.home_goals is not None]
        console.log(f"[cyan]2026-27[/] {len(cur)} finished matches, latest {max(m.date for m in cur)}")
    with console.status("[bold cyan]building 32 leakage-safe features (walk-forward state replay)...", spinner="dots12"):
        rows, x, _ = build_features(hist + cur)
        x = x[:, :32]
        console.log(f"[cyan]features[/] matrix {x.shape[0]:,} x {x.shape[1]}")
    years = np.array([m.year for m in rows])
    train, test = years < 2026, years == 2026
    train_rows = [m for m, k in zip(rows, train) if k]
    test_rows = [m for m, k in zip(rows, test) if k]
    preds = {}
    for name, cols, alpha, rho, colour in VARIANTS:
        with console.status(f"[bold {colour}]fitting {name}: Poisson GLM x2, {len(cols)} features, alpha={alpha}...",
                            spinner="bouncingBar"):
            with warnings.catch_warnings():
                warnings.simplefilter("ignore")
                fm = FittedModel(Candidate("poisson", 1460, alpha)).fit(x[train][:, cols], train_rows)
            rates = fm.predict(x[test][:, cols])
            p = outcomes(score_grid(rates, rho))
            preds[name] = (p / p.sum(1, keepdims=True), rates)
            console.log(f"[{colour}]{name:15}[/] trained on {len(train_rows):,} matches "
                        f"(effective n {fm.effective_sample_size:,.0f}), {2 * len(cols)} weights, rho={rho}")
    return test_rows, preds


class Board:
    def __init__(self, matches, preds, console):
        self.m, self.p, self.c = matches, preds, console
        self.y = np.array([m.outcome for m in matches])
        self.n = len(matches)
        self.i = 0

    # ---------- metrics ----------
    def metrics(self, name, upto):
        P, y = self.p[name][0][:upto], self.y[:upto]
        oh = np.eye(3)[y]
        ll = -np.log(P[np.arange(upto), y])
        return dict(ll=ll.mean(), brier=((P - oh) ** 2).sum(1).mean(),
                    rps=(((np.cumsum(P, 1) - np.cumsum(oh, 1))[:, :2]) ** 2).sum(1).mean() / 2,
                    acc=(P.argmax(1) == y).mean(), draw_pred=P[:, 1].mean(), draw_act=(y == 1).mean(), lls=ll)

    def table(self, name, upto):
        pts, xp = defaultdict(float), defaultdict(float)
        P = self.p[name][0]
        for m, p, y in zip(self.m[:upto], P[:upto], self.y[:upto]):
            pts[m.home] += 3 if y == 0 else (1 if y == 1 else 0)
            pts[m.away] += 3 if y == 2 else (1 if y == 1 else 0)
            xp[m.home] += 3 * p[0] + p[1]
            xp[m.away] += 3 * p[2] + p[1]
        return pts, xp

    # ---------- panels ----------
    def chart(self, w, h, draw):
        if self.i < 2:  # plotext cannot lay out a legend for empty series
            return Text("\n" * (h // 2) + "warming up...".center(w), style="dim")
        plt.clear_figure()
        plt.theme("dark")
        plt.plotsize(max(w, 20), max(h, 6))
        draw()
        return Text.from_ansi(plt.build())

    def loss_chart(self, w, h):
        k = self.i
        def draw():
            xs = list(range(1, k + 1))
            for name, *_, colour in VARIANTS:
                lls = -np.log(self.p[name][0][np.arange(k), self.y[:k]])
                plt.plot(xs, list(np.cumsum(lls) / np.arange(1, k + 1)), color=colour, marker="braille")
            base = -np.log(BASE[self.y[:k]])
            plt.plot(xs, list(np.cumsum(base) / np.arange(1, k + 1)), color="red", marker="braille")
            plt.hline(np.log(3), "white")
            plt.title("running log loss (lower = better) | white = coin-flip 1.099")
            plt.ylim(.8, 1.4)
            plt.xlim(1, max(self.n, 2))
        return self.chart(w, h, draw)

    def prob_chart(self, w, h):
        k = self.i
        P = self.p["deployed"][0][:k]
        hit = P.argmax(1) == self.y[:k]
        pa = P[np.arange(k), self.y[:k]]
        def draw():
            xs = np.arange(1, k + 1)
            if hit.any():
                plt.scatter(list(xs[hit]), list(pa[hit]), color="green", marker="dot")
            if (~hit).any():
                plt.scatter(list(xs[~hit]), list(pa[~hit]), color="red", marker="dot")
            plt.hline(1 / 3, "white")
            plt.title("deployed: P(what actually happened) per match")
            plt.ylim(0, .9)
            plt.xlim(1, max(self.n, 2))
        return self.chart(w, h, draw)

    def calib_chart(self, w, h):
        k = self.i
        P = self.p["deployed"][0][:k].ravel()
        O = np.eye(3)[self.y[:k]].ravel()
        def draw():
            plt.plot([0, .8], [0, .8], color="white", marker="braille")
            edges = np.array([0, .15, .25, .35, .45, .55, .65, 1])
            xs, ys = [], []
            for lo, hi in zip(edges[:-1], edges[1:]):
                mask = (P >= lo) & (P < hi)
                if mask.sum() >= 3:
                    xs.append(P[mask].mean()); ys.append(O[mask].mean())
            if xs:
                plt.plot(xs, ys, color="cyan", marker="hd")
            plt.title("calibration: predicted % vs how often it happened")
            plt.xlim(0, .8)
            plt.ylim(0, .9)
        return self.chart(w, h, draw)

    def card(self, width):
        if self.i == 0:
            return Panel(Align.center(Text("warming up...", style="dim")), title="current match")
        j = self.i - 1
        m, y = self.m[j], self.y[j]
        title = Text.assemble((f"R{m.matchday} ", "bold white"), (f"{m.date}  ", "dim"),
                              (f"{m.home} ", "bold"), (f"{m.hg}-{m.ag}", "bold reverse"), (f" {m.away}", "bold"))
        t = Table.grid(padding=(0, 1))
        t.add_column(width=14, no_wrap=True); t.add_column(no_wrap=True); t.add_column(width=1)
        t.add_column(width=5, justify="right")
        barw = max(2, (width - 56) // 3)
        for name, *_, colour in VARIANTS:
            p = self.p[name][0][j]
            bar = Text()
            for k in range(3):
                n = int(round(p[k] * barw * 1.6))
                style = ("bold " + colour) if k == y else "grey42"
                bar.append(f"{HDA[k]} ", style="bold" if k == y else "dim")
                bar.append("█" * min(n, barw) + " " * max(0, barw - n), style=style)
                bar.append(f" {p[k]:.2f} ", style=style)
            ok = p.argmax() == y
            t.add_row(Text(name, style=colour), bar, Text("✔" if ok else "✘", style="bold green" if ok else "bold red"),
                      Text(f"{-np.log(p[y]):.3f}", style="dim"))
        rates = self.p["deployed"][1][j]
        foot = Text(f"result: {['HOME WIN', 'DRAW', 'AWAY WIN'][y]}   model expected goals {rates[0]:.2f} - {rates[1]:.2f}",
                    style="bold yellow")
        return Panel(Group(title, Text(""), t, Text(""), foot), title=f"match {self.i}/{self.n}", border_style="cyan")

    def feed(self, height):
        t = Table.grid(padding=(0, 1))
        for col in ("R", "fixture", "sc", "H", "D", "A", "ok", "ll"):
            if col == "fixture":
                t.add_column(no_wrap=True, overflow="ellipsis", max_width=20)
            else:
                t.add_column(justify="right" if col in ("H", "D", "A", "ll") else "left", no_wrap=True,
                             min_width={"R": 3, "sc": 3, "ok": 1}.get(col, 4))
        P = self.p["deployed"][0]
        for j in range(max(0, self.i - height), self.i):
            m, y, p = self.m[j], self.y[j], P[j]
            ok = p.argmax() == y
            cells = [Text(f"{p[k]:.2f}", style=("bold green" if ok else "bold red") if k == y else "dim") for k in range(3)]
            t.add_row(Text(f"R{m.matchday}", style="dim"), Text(f"{short(m.home)[:11]} v {short(m.away)[:11]}"),
                      Text(f"{m.hg}-{m.ag}", style="bold"), *cells,
                      Text("✔" if ok else "✘", style="green" if ok else "red"),
                      Text(f"{-np.log(p[y]):.2f}", style="yellow" if -np.log(p[y]) > 1.4 else "dim"))
        return Panel(t, title="prediction stream (deployed)", border_style="magenta")

    def league(self):
        pts, xp = self.table("deployed", self.i)
        t = Table(box=None, padding=(0, 1), header_style="bold cyan")
        for col, j in (("#", "right"), ("team", "left"), ("Pts", "right"), ("xPts", "right"), ("luck", "left")):
            t.add_column(col, justify=j)
        order = sorted(pts, key=lambda k: (-pts[k], -xp[k]))
        xrank = {team: r for r, team in enumerate(sorted(xp, key=lambda k: -xp[k]), 1)}
        for r, team in enumerate(order, 1):
            d = pts[team] - xp[team]
            n = min(10, int(round(abs(d))))
            bar = Text(("▶" * n) if d >= 0 else ("◀" * n), style="green" if d >= 0 else "red")
            bar.append(f" {d:+.1f}", style="dim")
            t.add_row(str(r), Text(short(team)[:15], style="bold" if abs(r - xrank[team]) <= 2 else ""),
                      f"{pts[team]:.0f}", f"{xp[team]:.1f}", bar)
        rho = spearmanr([pts[k] for k in order], [xp[k] for k in order])[0] if len(order) > 2 else float("nan")
        return Panel(t, title=f"real table vs model | rank corr {rho:.2f}", border_style="green")

    def scoreboard(self):
        t = Table(expand=True, header_style="bold white", border_style="grey50")
        for col in ("variant", "log loss", "vs deployed", "Brier", "RPS", "accuracy", "draws pred/real", "table rho"):
            t.add_column(col, justify="left" if col == "variant" else "right")
        if self.i < 2:
            return Panel(t, title="scoreboard")
        ms = {name: self.metrics(name, self.i) for name, *_ in VARIANTS}
        best = min(ms, key=lambda k: ms[k]["ll"])
        for name, *_, colour in VARIANTS:
            mt = ms[name]
            pts, xp = self.table(name, self.i)
            teams = sorted(pts)
            rho = spearmanr([pts[k] for k in teams], [xp[k] for k in teams])[0]
            d = mt["ll"] - ms["deployed"]["ll"]
            t.add_row(Text(("★ " if name == best else "  ") + name, style="bold " + colour),
                      f"{mt['ll']:.4f}", Text(f"{d:+.4f}", style="green" if d < 0 else ("red" if d > 0 else "dim")),
                      f"{mt['brier']:.4f}", f"{mt['rps']:.4f}", f"{mt['acc']:.1%}",
                      f"{mt['draw_pred']:.1%} / {mt['draw_act']:.1%}", f"{rho:.2f}")
        b = -np.log(BASE[self.y[:self.i]]).mean()
        t.add_row(Text("  base rates 45/25/30", style="red"), f"{b:.4f}", "", "", "", "", "", "")
        return Panel(t, title="scoreboard (2026-27 only)", border_style="white")

    def render(self):
        W, H = self.c.size
        root = Layout()
        bar = int(30 * self.i / max(self.n, 1))
        head = Text.assemble((" MATCHSENSE // WALK-FORWARD BACKTEST 2026-27 ", "bold black on cyan"), "  ",
                             ("━" * bar, "bold cyan"), ("━" * (30 - bar), "grey30"),
                             (f"  {self.i}/{self.n} matches   ", "bold"),
                             ("features use only results before each matchday", "dim"))
        score_h = len(VARIANTS) + 6
        root.split_column(Layout(Panel(head, border_style="cyan"), size=3), Layout(name="body"),
                          Layout(self.scoreboard(), size=score_h))
        body_h = H - 3 - score_h
        left_w = max(60, int(W * .42))
        right_w = W - left_w
        root["body"].split_row(Layout(name="left", size=left_w), Layout(name="right"))
        card_h = len(VARIANTS) + 7
        root["left"].split_column(Layout(self.card(left_w), size=card_h), Layout(self.league()))
        cw, ch = right_w // 2 - 4, body_h // 2 - 2
        root["right"].split_column(Layout(name="r1"), Layout(name="r2"))
        loss_key = Text.assemble(*[(f"━{name.split()[0][:8]} ", colour) for name, *_, colour in VARIANTS], ("━base", "red"))
        prob_key = Text.assemble(("• pick correct  ", "green"), ("• pick wrong  ", "red"), ("─ 1/3", "white"))
        cal_key = Text.assemble(("━ deployed  ", "cyan"), ("─ perfect", "white"))
        root["r1"].split_row(Layout(Panel(self.loss_chart(cw, ch - 1), subtitle=loss_key, border_style="cyan", padding=0)),
                             Layout(Panel(self.prob_chart(cw, ch - 1), subtitle=prob_key, border_style="cyan", padding=0)))
        root["r2"].split_row(Layout(Panel(self.calib_chart(cw, ch - 1), subtitle=cal_key, border_style="cyan", padding=0)),
                             Layout(self.feed(max(3, body_h // 2 - 3))))
        return root

    def summary(self):
        c = self.c
        ms = {name: self.metrics(name, self.n) for name, *_ in VARIANTS}
        rng = np.random.default_rng(0)
        c.rule("[bold cyan]FINAL VERDICT (2026-27, all matches so far)")
        for name, *_, colour in VARIANTS[1:]:
            d = ms[name]["lls"] - ms["deployed"]["lls"]
            bs = np.array([rng.choice(d, len(d)).mean() for _ in range(4000)])
            lo, hi = np.percentile(bs, [2.5, 97.5])
            verdict = ("clearly better" if hi < 0 else "clearly worse" if lo > 0 else "not distinguishable from deployed")
            c.print(f"[{colour}]{name:15}[/] log loss {d.mean():+.4f} vs deployed  (95% range {lo:+.4f} to {hi:+.4f})  "
                    f"→ [bold]{verdict}[/]")
        c.print(f"[dim]{self.n} matches is a small sample: a variant must beat deployed by more than the range above "
                "before it is worth deploying.[/]")


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--delay", type=float, default=0.35, help="seconds between matches")
    args = ap.parse_args()
    console = Console()
    console.rule("[bold cyan]MatchSense live backtest")
    with threadpool_limits(limits=1):
        matches, preds = load(console)
    board = Board(matches, preds, console)
    time.sleep(.6)
    with Live(board.render(), console=console, screen=False, refresh_per_second=12, transient=False) as live:
        for i in range(1, board.n + 1):
            board.i = i
            live.update(board.render())
            time.sleep(args.delay)
    board.summary()


if __name__ == "__main__":
    main()
