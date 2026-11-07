"""Premier League Fantasy (FPL) public endpoints.

These keyless JSON endpoints are published by the Premier League for its
fantasy game. They are not a documented public API, so responses are cached
aggressively and requests are throttled. They provide current-season PL
fixtures/results, squads with position classes, per-fixture starts/minutes
per player, and the league's structured availability flags (status,
chance of playing, news text with its timestamp).
"""
from __future__ import annotations

from datetime import datetime, timezone

from src.matchsense.http import ProviderError
from src.matchsense.providers.base import CREDIBILITY, FixtureProvider, SquadProvider, stable_id
from src.matchsense.schemas import (PREMIER_LEAGUE, Availability, Category, Certainty, EventType, Fixture,
                                    FixtureStatus, IntelligenceEvent, PlayerAppearance, Position, Relevance,
                                    SquadPlayer)
from src.matchsense.teams import REGISTRY

BASE = "https://fantasy.premierleague.com/api"
POSITIONS = {1: Position.GK, 2: Position.DEF, 3: Position.MID, 4: Position.FWD}
ROLE = {Position.GK: "goalkeeper", Position.DEF: "defence", Position.MID: "midfield", Position.FWD: "attack"}


def _dt(value: str | None) -> datetime | None:
    return datetime.fromisoformat(value.replace("Z", "+00:00")) if value else None


class FPLProvider(FixtureProvider, SquadProvider):
    name = "fpl"
    source_label = "Premier League Fantasy (official FPL data)"

    def _get(self, path: str, ttl: float):
        try:
            return self.http.get(f"{BASE}{path}", ttl_seconds=ttl)
        except ProviderError:
            raise
        except ValueError as exc:  # pragma: no cover - malformed JSON
            raise ProviderError(f"FPL returned malformed data for {path}: {exc}") from exc

    def bootstrap(self):
        return self._get("/bootstrap-static/", ttl=1800).json()

    def raw_fixtures(self):
        return self._get("/fixtures/", ttl=300).json()

    def season_label(self) -> str:
        events = self.bootstrap()["events"]
        year = _dt(events[0]["deadline_time"]).year
        return f"{year}-{(year + 1) % 100:02d}"

    def _team_maps(self):
        teams = self.bootstrap()["teams"]
        by_id = {}
        for team in teams:
            canonical = REGISTRY.canonical(team["name"])
            if canonical is None:
                raise ProviderError(f"FPL team {team['name']!r} has no canonical mapping")
            by_id[team["id"]] = canonical
        return by_id, {v: k for k, v in by_id.items()}

    @staticmethod
    def _status(raw) -> FixtureStatus:
        if raw.get("finished") or raw.get("finished_provisional"):
            return FixtureStatus.FINISHED
        if raw.get("started"):
            return FixtureStatus.LIVE
        if raw.get("kickoff_time") is None:
            return FixtureStatus.POSTPONED
        return FixtureStatus.SCHEDULED

    def all_fixtures(self) -> list[Fixture]:
        by_id, _ = self._team_maps()
        season = self.season_label()
        result = []
        for raw in self.raw_fixtures():
            kickoff = _dt(raw.get("kickoff_time"))
            status = self._status(raw)
            if kickoff is None:
                # Postponed/unscheduled fixtures have no time; keep them for
                # status queries with a sentinel far-future kickoff.
                kickoff = datetime(2100, 1, 1, tzinfo=timezone.utc)
            result.append(Fixture(
                fixture_id=f"PL:{season}:{by_id[raw['team_h']]}:{by_id[raw['team_a']]}",
                competition=PREMIER_LEAGUE, season=season, matchday=raw.get("event"),
                home=by_id[raw["team_h"]], away=by_id[raw["team_a"]], kickoff_utc=kickoff, status=status,
                home_goals=raw.get("team_h_score"), away_goals=raw.get("team_a_score"),
                minute=raw.get("minutes") if status == FixtureStatus.LIVE else None,
                source=self.source_label, source_ids={"fpl": str(raw["id"])},
                kickoff_confirmed=not raw.get("provisional_start_time", False)))
        return result

    def fixtures(self, team, start, end):
        return [f for f in self.all_fixtures() if team in (f.home, f.away) and start <= f.kickoff_utc < end
                or (team in (f.home, f.away) and f.status == FixtureStatus.POSTPONED)]

    def _elements(self, team: str):
        _, to_id = self._team_maps()
        if team not in to_id:
            raise ProviderError(f"{team} is not in the current FPL (Premier League) season")
        return [e for e in self.bootstrap()["elements"] if e["team"] == to_id[team]]

    def squad(self, team: str) -> list[SquadPlayer]:
        players = []
        for e in self._elements(team):
            if e["status"] == "u" and not e.get("minutes"):
                continue  # Loaned/transferred out before playing this season.
            full = f"{e['first_name']} {e['second_name']}".strip()
            players.append(SquadPlayer(
                player_id=f"fpl:{e['id']}", name=full, short_name=e["web_name"], team=team,
                position=POSITIONS[e["element_type"]],
                aliases=sorted({full, e["web_name"], e["second_name"], e.get("known_name") or full} - {""}),
                status_code=e["status"], chance_of_playing=e.get("chance_of_playing_next_round")))
        return players

    def appearances(self, team: str, before: datetime) -> list[PlayerAppearance]:
        _, to_id = self._team_maps()
        team_id = to_id[team]
        # A transferred player's history includes his previous club's games.
        own_fixtures = {raw["id"] for raw in self.raw_fixtures() if team_id in (raw["team_h"], raw["team_a"])}
        rows = []
        for e in self._elements(team):
            if not e.get("minutes"):
                continue
            history = self._get(f"/element-summary/{e['id']}/", ttl=6 * 3600).json().get("history", [])
            for h in history:
                kickoff = _dt(h.get("kickoff_time"))
                if kickoff is None or kickoff >= before or h.get("fixture") not in own_fixtures:
                    continue
                rows.append(PlayerAppearance(player_id=f"fpl:{e['id']}", team=team, kickoff_utc=kickoff,
                                             started=bool(h.get("starts")), minutes=int(h.get("minutes", 0)),
                                             goals=int(h.get("goals_scored", 0))))
        return rows

    def status_events(self, team: str, as_of: datetime) -> list[IntelligenceEvent]:
        """Translate FPL availability flags into structured evidence."""
        events = []
        retrieved = self.http._clock()
        for e in self._elements(team):
            status, chance, news = e["status"], e.get("chance_of_playing_next_round"), (e.get("news") or "").strip()
            if status == "a" and not news:
                continue
            if status == "u" and not e.get("minutes"):
                continue
            added = _dt(e.get("news_added")) or retrieved
            if added > as_of:
                continue  # Never use a flag published after the prediction time.
            name = f"{e['first_name']} {e['second_name']}".strip()
            position = POSITIONS[e["element_type"]]
            if status == "s":
                kind, avail, certainty = EventType.SUSPENSION, Availability.UNAVAILABLE, Certainty.CONFIRMED_FACT
            elif status in ("u", "n"):
                kind, avail, certainty = EventType.LEFT_CLUB, Availability.UNAVAILABLE, Certainty.CONFIRMED_FACT
            elif status == "i":
                kind = EventType.ILLNESS if "ill" in news.lower() else EventType.INJURY
                avail = Availability.UNAVAILABLE if not chance else Availability.DOUBTFUL
                certainty = Certainty.REPORTED_EXPECTATION
            elif status == "d":
                kind, avail, certainty = EventType.FITNESS_DOUBT, Availability.DOUBTFUL, Certainty.REPORTED_EXPECTATION
            else:
                kind, avail, certainty = EventType.RETURN_FROM_INJURY, Availability.AVAILABLE, Certainty.REPORTED_EXPECTATION
            chance_text = f" (FPL chance of playing: {chance}%)" if chance is not None else ""
            events.append(IntelligenceEvent(
                event_id=stable_id("fpl", str(e["id"]), news, status), entity=name, team=team,
                event_type=kind, category=Category.PLAYER_AVAILABILITY, certainty=certainty, availability=avail,
                match_relevance=Relevance.HIGH, tactical_role=ROLE[position],
                evidence=(news or f"FPL status '{status}'") + chance_text,
                player_id=f"fpl:{e['id']}", player_match_score=1.0, source=self.source_label, provider=self.name,
                url="https://fantasy.premierleague.com/statistics", title=f"{e['web_name']}: {news or status}",
                published_at=added, retrieved_at=retrieved, credibility=CREDIBILITY["premier_league_fpl"],
                extractor="fpl_status"))
        return events
