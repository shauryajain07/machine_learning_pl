"""Provider interfaces and a run-scoped status log for data-quality reporting."""
from __future__ import annotations

import hashlib
from dataclasses import dataclass, field
from datetime import datetime

from src.matchsense.http import HttpClient, MissingCredentials, ProviderError
from src.matchsense.schemas import Fixture, InputStatus, IntelligenceEvent, NewsItem, PlayerAppearance, SquadPlayer

# Publisher credibility tiers used for corroboration weighting. These are
# editorial reliability priors, documented in docs/DATA_SOURCES.md; they
# weight evidence, they never become forecast probabilities.
CREDIBILITY = {
    "official_lineup": 1.0,
    "premier_league_fpl": 0.9,
    "api_football_injuries": 0.85,
    "bbc": 0.85,
    "guardian": 0.85,
    "club_official": 0.95,
    "tier1_press": 0.8,   # Reuters, AP, PA, Sky Sports, The Athletic, ESPN
    "national_press": 0.65,
    "aggregator": 0.5,
    "tabloid": 0.4,
    "unknown": 0.35,
    "simulated": 0.6,
}

TIER1 = ("reuters", "associated press", "ap news", "pa media", "sky sports", "the athletic", "espn",
         "bbc", "guardian", "the times", "telegraph", "independent", "football.london", "manchester evening news")
TABLOIDS = ("the sun", "daily mail", "mirror", "daily star", "express", "metro", "football insider",
            "caught offside", "teamtalk")


def publisher_credibility(publisher: str) -> float:
    name = publisher.lower()
    if any(t in name for t in TABLOIDS):
        return CREDIBILITY["tabloid"]
    if any(t in name for t in TIER1):
        return CREDIBILITY["tier1_press"]
    return CREDIBILITY["national_press"] if name else CREDIBILITY["unknown"]


def stable_id(*parts: str) -> str:
    return hashlib.sha1("|".join(parts).encode()).hexdigest()[:16]


@dataclass
class StatusLog:
    entries: list[InputStatus] = field(default_factory=list)
    understanding: dict | None = None  # Query-interpreter result for this request.

    def add(self, name: str, status: str, detail: str = "", as_of: datetime | None = None):
        if any(e.name == name and e.status == status and e.detail == detail for e in self.entries):
            return
        self.entries.append(InputStatus(name=name, status=status, detail=detail, as_of=as_of))

    def guard(self, name: str, func, *args, default=None, **kwargs):
        """Run a provider call; record (not raise) credential or outage problems."""
        try:
            return func(*args, **kwargs)
        except MissingCredentials as exc:
            self.add(name, "missing", str(exc))
        except ProviderError as exc:
            self.add(name, "error", str(exc))
        return default


class Provider:
    name = "provider"

    def __init__(self, http: HttpClient):
        self.http = http


class FixtureProvider(Provider):
    def fixtures(self, team: str, start: datetime, end: datetime) -> list[Fixture]:
        raise NotImplementedError


class NewsProvider(Provider):
    def news(self, team: str, since: datetime, until: datetime) -> list[NewsItem]:
        raise NotImplementedError


class SquadProvider(Provider):
    def squad(self, team: str) -> list[SquadPlayer]:
        raise NotImplementedError

    def appearances(self, team: str, before: datetime) -> list[PlayerAppearance]:
        raise NotImplementedError

    def status_events(self, team: str, as_of: datetime) -> list[IntelligenceEvent]:
        return []


class LineupProvider(Provider):
    def confirmed_lineups(self, fixture: Fixture, as_of: datetime) -> dict[str, list[str]] | None:
        """Return {team: [11 player names]} only if officially released by ``as_of``."""
        raise NotImplementedError
