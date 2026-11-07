"""Offline demo mode.

* FPL responses are a *recorded* snapshot (see ``demo_data/manifest.json``),
  replayed through the real FPL provider via an httpx mock transport.
* News items and official lineups are *simulated*: fictional publishers on
  ``.invalid`` domains, flagged ``is_simulated`` everywhere and in the UI.
  They are never presented as real-time evidence.
"""
from __future__ import annotations

import json
from datetime import datetime
from pathlib import Path
from urllib.parse import urlsplit

import httpx

from src.matchsense.config import PACKAGE_ROOT
from src.matchsense.providers.base import CREDIBILITY, LineupProvider, NewsProvider, stable_id
from src.matchsense.schemas import NewsItem

DEMO_DIR = PACKAGE_ROOT / "demo_data"


def manifest(directory: Path = DEMO_DIR) -> dict:
    return json.loads((directory / "manifest.json").read_text())


def recorded_transport(directory: Path = DEMO_DIR) -> httpx.MockTransport:
    recording = json.loads((directory / "fpl_recording.json").read_text())

    def handler(request: httpx.Request) -> httpx.Response:
        path = urlsplit(str(request.url)).path.removeprefix("/api")
        if request.url.host == "fantasy.premierleague.com" and path in recording:
            return httpx.Response(200, json=recording[path])
        return httpx.Response(404, json={"detail": "not recorded for demo mode"})

    return httpx.MockTransport(handler)


def _dt(value: str) -> datetime:
    return datetime.fromisoformat(value.replace("Z", "+00:00"))


class DemoNews(NewsProvider):
    name = "demo_simulated_news"

    def __init__(self, http, directory: Path = DEMO_DIR):
        super().__init__(http)
        self.rows = json.loads((directory / "news_simulated.json").read_text())

    def news(self, team, since, until):
        items = []
        for row in self.rows:
            published = _dt(row["published_at"])
            if team in row["teams"] and since <= published <= until:
                items.append(NewsItem(item_id=stable_id("demo", row["id"]), title=row["title"], summary=row["summary"],
                                      url=row["url"], source=row["source"], provider=self.name,
                                      published_at=published, retrieved_at=published,
                                      teams=row["teams"],
                                      credibility=min(row.get("credibility", CREDIBILITY["simulated"]), 0.9),
                                      is_simulated=True))
        return items


class DemoLineups(LineupProvider):
    name = "demo_simulated_lineups"

    def __init__(self, http, directory: Path = DEMO_DIR):
        super().__init__(http)
        self.rows = json.loads((directory / "lineups_simulated.json").read_text())

    def confirmed_lineups(self, fixture, as_of):
        row = self.rows.get(f"{fixture.home}|{fixture.away}")
        if not row or as_of < _dt(row["release_at"]):
            return None
        return {fixture.home: row["home"], fixture.away: row["away"]}
