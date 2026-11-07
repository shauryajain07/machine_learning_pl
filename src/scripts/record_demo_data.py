"""Record a trimmed FPL snapshot for offline demo mode.

Usage::

    ./.venv/bin/python -m src.scripts.record_demo_data --teams "Manchester City" "Nottingham Forest"

Only factual FPL fields needed by the providers are kept. The recording time
is written to ``demo_data/manifest.json``; demo responses always label this
data as a recorded snapshot, never as live evidence.
"""
from __future__ import annotations

import argparse
import json
from datetime import datetime, timezone

import httpx

from src.matchsense.providers.demo import DEMO_DIR
from src.matchsense.teams import REGISTRY

BASE = "https://fantasy.premierleague.com/api"
ELEMENT_FIELDS = ("id", "first_name", "second_name", "web_name", "known_name", "team", "element_type", "status",
                  "chance_of_playing_next_round", "news", "news_added", "minutes", "starts")
HISTORY_FIELDS = ("element", "fixture", "kickoff_time", "minutes", "starts", "goals_scored", "was_home",
                  "opponent_team", "round")


def run(args):
    client = httpx.Client(timeout=30, headers={"User-Agent": "MatchSenseAI/0.1 (demo recorder)"})
    boot = client.get(f"{BASE}/bootstrap-static/").json()
    fixtures = client.get(f"{BASE}/fixtures/").json()
    fpl_names = {REGISTRY.get(t).fpl_name for t in args.teams}
    team_ids = {t["id"] for t in boot["teams"] if t["name"] in fpl_names}
    if len(team_ids) != len(args.teams):
        raise SystemExit(f"Could not map all teams to FPL: {args.teams}")
    elements = [{k: e.get(k) for k in ELEMENT_FIELDS} for e in boot["elements"] if e["team"] in team_ids]
    recording = {
        "/bootstrap-static/": {
            "teams": [{k: t[k] for k in ("id", "name", "short_name")} for t in boot["teams"]],
            "events": [{k: e[k] for k in ("id", "deadline_time", "finished", "is_current", "is_next")}
                       for e in boot["events"]],
            "elements": elements},
        "/fixtures/": [{k: f.get(k) for k in ("id", "event", "kickoff_time", "team_h", "team_a", "started",
                                               "finished", "finished_provisional", "minutes", "team_h_score",
                                               "team_a_score", "provisional_start_time")} for f in fixtures],
    }
    for e in elements:
        if e.get("minutes"):
            summary = client.get(f"{BASE}/element-summary/{e['id']}/").json()
            recording[f"/element-summary/{e['id']}/"] = {
                "history": [{k: h.get(k) for k in HISTORY_FIELDS} for h in summary.get("history", [])]}
    (DEMO_DIR / "fpl_recording.json").write_text(json.dumps(recording, separators=(",", ":")))
    path = DEMO_DIR / "manifest.json"
    manifest = json.loads(path.read_text()) if path.exists() else {}
    manifest.update({"fpl_recorded_at": datetime.now(timezone.utc).isoformat(timespec="seconds"),
                     "recorded_teams": args.teams})
    path.write_text(json.dumps(manifest, indent=2) + "\n")
    print(f"Recorded {len(elements)} players, {len(recording) - 2} player histories")


def build_parser():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--teams", nargs="+", default=["Manchester City", "Nottingham Forest", "Arsenal",
                                                        "Hull City", "Crystal Palace", "Liverpool"])
    return parser


if __name__ == "__main__":
    raise SystemExit(run(build_parser().parse_args()))
