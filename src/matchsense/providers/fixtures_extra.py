"""Keyed fixture/lineup/injury providers and the keyless openfootball feed.

* API-Football (api-sports.io): all competitions, injuries, official XIs.
* football-data.org: fixtures across PL/UCL/cups on its free tier.
* openfootball/england on GitHub: public-domain PL fixtures and results,
  parsed with the repository's own Football.TXT header/score parsers.
"""
from __future__ import annotations

import re
from datetime import datetime, time, timedelta, timezone
from zoneinfo import ZoneInfo

from src.ingestion.football_txt.header_parser import parse_date_parts, parse_header, resolve_date
from src.ingestion.football_txt.score_parser import SCORE_RE, parse_leading_time
from src.matchsense.http import MissingCredentials, ProviderError
from src.matchsense.providers.base import CREDIBILITY, FixtureProvider, LineupProvider, stable_id
from src.matchsense.schemas import (PREMIER_LEAGUE, Availability, Category, Certainty, Competition, EventType,
                                    Fixture, FixtureStatus, IntelligenceEvent, Relevance)
from src.matchsense.teams import REGISTRY

LONDON = ZoneInfo("Europe/London")


def season_for(moment: datetime) -> str:
    year = moment.year if moment.month >= 7 else moment.year - 1
    return f"{year}-{(year + 1) % 100:02d}"


# --------------------------------------------------------------------------- API-Football
APIF_LEAGUES = {39: ("PL", "Premier League"), 2: ("UCL", "UEFA Champions League"), 3: ("UEL", "UEFA Europa League"),
                848: ("UECL", "UEFA Conference League"), 45: ("FAC", "FA Cup"), 48: ("ELC", "EFL Cup"),
                528: ("CS", "Community Shield")}
APIF_STATUS = {**{s: FixtureStatus.SCHEDULED for s in ("TBD", "NS")},
               **{s: FixtureStatus.LIVE for s in ("1H", "HT", "2H", "ET", "BT", "P", "SUSP", "INT", "LIVE")},
               **{s: FixtureStatus.FINISHED for s in ("FT", "AET", "PEN")},
               "PST": FixtureStatus.POSTPONED,
               **{s: FixtureStatus.CANCELLED for s in ("CANC", "ABD", "AWD", "WO")}}


class APIFootballProvider(FixtureProvider, LineupProvider):
    name = "api_football"
    base = "https://v3.football.api-sports.io"

    def __init__(self, http, key: str | None):
        super().__init__(http)
        self.key = key
        self.blocked: str | None = None  # Set once the plan rejects the current season.

    @property
    def usable(self) -> bool:
        return bool(self.key) and self.blocked is None

    def _get(self, path, params, ttl):
        if not self.key:
            raise MissingCredentials("API_FOOTBALL_KEY not set: cup/European fixtures, injuries and official XIs unavailable")
        if self.blocked:
            raise ProviderError(self.blocked)
        payload = self.http.get(f"{self.base}{path}", params=params, headers={"x-apisports-key": self.key},
                                ttl_seconds=ttl).json()
        errors = payload.get("errors")
        if errors:
            if isinstance(errors, dict) and "plan" in errors:
                # Free plans exclude the current season: stop spending requests on it.
                self.blocked = f"API-Football plan does not cover this season ({errors['plan']})"
                raise ProviderError(self.blocked)
            raise ProviderError(f"API-Football error: {errors}")
        return payload.get("response", [])

    def team_id(self, team: str) -> int:
        info = REGISTRY.get(team)
        search = (info.api_football_names[0] if info and info.api_football_names else team)
        # The API rejects `country` together with `search`; the name check below disambiguates.
        for row in self._get("/teams", {"search": search}, ttl=30 * 86400):
            if REGISTRY.canonical(row["team"]["name"]) == team:
                return int(row["team"]["id"])
        raise ProviderError(f"API-Football has no team matching {team}")

    def _fixture(self, row) -> Fixture | None:
        league = row["league"]
        code, name = APIF_LEAGUES.get(league["id"], (f"L{league['id']}", league["name"]))
        home, away = REGISTRY.canonical(row["teams"]["home"]["name"]), REGISTRY.canonical(row["teams"]["away"]["name"])
        kickoff = datetime.fromisoformat(row["fixture"]["date"]).astimezone(timezone.utc)
        status = APIF_STATUS.get(row["fixture"]["status"]["short"], FixtureStatus.UNKNOWN)
        competition = PREMIER_LEAGUE if code == "PL" else Competition(code=code, name=name, model_supported=False)
        season = season_for(kickoff)
        if home is None or away is None:
            # Non-English opponents (e.g. UCL): keep the provider spelling.
            home = home or row["teams"]["home"]["name"]
            away = away or row["teams"]["away"]["name"]
        round_text = league.get("round") or ""
        matchday = int(m[1]) if (m := re.search(r"(\d+)$", round_text)) and code == "PL" else None
        return Fixture(fixture_id=f"{code}:{season}:{home}:{away}" if code == "PL" else f"{code}:{row['fixture']['id']}",
                       competition=competition, season=season, matchday=matchday, home=home, away=away,
                       kickoff_utc=kickoff, status=status, home_goals=row["goals"]["home"],
                       away_goals=row["goals"]["away"], minute=row["fixture"]["status"].get("elapsed"),
                       venue=(row["fixture"].get("venue") or {}).get("name"), source="API-Football",
                       source_ids={"api_football": str(row["fixture"]["id"])})

    def fixtures(self, team, start, end):
        team_id = self.team_id(team)
        rows = self._get("/fixtures", {"team": team_id, "season": int(season_for(start)[:4]),
                                       "from": start.date().isoformat(), "to": end.date().isoformat(),
                                       "timezone": "UTC"}, ttl=600)
        result = [self._fixture(r) for r in rows]
        return [f for f in result if f and start <= f.kickoff_utc < end]

    def injuries(self, fixture: Fixture, as_of: datetime) -> list[IntelligenceEvent]:
        fid = fixture.source_ids.get("api_football")
        if not fid:
            return []
        events = []
        retrieved = self.http._clock()
        for row in self._get("/injuries", {"fixture": fid}, ttl=1800):
            team = REGISTRY.canonical(row["team"]["name"])
            missing = row["player"].get("type") == "Missing Fixture"
            reason = row["player"].get("reason") or ""
            events.append(IntelligenceEvent(
                event_id=stable_id("apif", fid, str(row["player"]["id"])), entity=row["player"]["name"], team=team,
                event_type=EventType.SUSPENSION if "suspen" in reason.lower() or "card" in reason.lower() else EventType.INJURY,
                category=Category.PLAYER_AVAILABILITY,
                certainty=Certainty.REPORTED_EXPECTATION,
                availability=Availability.UNAVAILABLE if missing else Availability.DOUBTFUL,
                match_relevance=Relevance.HIGH, evidence=f"{row['player'].get('type')}: {reason}",
                source="API-Football injuries", provider=self.name,
                url="https://www.api-football.com/documentation-v3#tag/Injuries", title=f"{row['player']['name']} – {reason}",
                published_at=min(retrieved, as_of), retrieved_at=retrieved,
                credibility=CREDIBILITY["api_football_injuries"], extractor="api_football_injuries"))
        return events

    def confirmed_lineups(self, fixture, as_of):
        fid = fixture.source_ids.get("api_football")
        if not fid or as_of < fixture.kickoff_utc - timedelta(hours=2):
            return None
        rows = self._get("/fixtures/lineups", {"fixture": fid}, ttl=120)
        lineups = {}
        for row in rows:
            team = REGISTRY.canonical(row["team"]["name"])
            starters = [p["player"]["name"] for p in row.get("startXI", [])]
            if team and len(starters) == 11:
                lineups[team] = starters
        return lineups if len(lineups) == 2 else None


# --------------------------------------------------------------------------- football-data.org
FD_STATUS = {"SCHEDULED": FixtureStatus.SCHEDULED, "TIMED": FixtureStatus.SCHEDULED,
             "IN_PLAY": FixtureStatus.LIVE, "PAUSED": FixtureStatus.LIVE, "LIVE": FixtureStatus.LIVE,
             "FINISHED": FixtureStatus.FINISHED, "AWARDED": FixtureStatus.FINISHED,
             "POSTPONED": FixtureStatus.POSTPONED, "SUSPENDED": FixtureStatus.POSTPONED,
             "CANCELLED": FixtureStatus.CANCELLED}
FD_CODES = {"PL": "PL", "CL": "UCL", "EL": "UEL", "FAC": "FAC", "ELC": "ELC", "ECL": "UECL"}


class FootballDataProvider(FixtureProvider):
    name = "football_data"
    base = "https://api.football-data.org/v4"

    def __init__(self, http, key: str | None):
        super().__init__(http)
        self.key = key

    def _get(self, path, params=None, ttl=600):
        if not self.key:
            raise MissingCredentials("FOOTBALL_DATA_API_KEY not set")
        return self.http.get(f"{self.base}{path}", params=params, headers={"X-Auth-Token": self.key},
                             ttl_seconds=ttl).json()

    def team_id(self, team: str) -> int:
        for row in self._get("/competitions/PL/teams", ttl=7 * 86400).get("teams", []):
            if REGISTRY.canonical(row["name"]) == team or REGISTRY.canonical(row.get("shortName", "")) == team:
                return int(row["id"])
        raise ProviderError(f"football-data.org has no PL team matching {team}")

    def fixtures(self, team, start, end):
        payload = self._get(f"/teams/{self.team_id(team)}/matches",
                            {"dateFrom": start.date().isoformat(), "dateTo": (end + timedelta(days=1)).date().isoformat()})
        result = []
        for row in payload.get("matches", []):
            comp = row["competition"]
            code = FD_CODES.get(comp.get("code"), comp.get("code") or "OTHER")
            competition = PREMIER_LEAGUE if code == "PL" else Competition(code=code, name=comp["name"], model_supported=False)
            kickoff = datetime.fromisoformat(row["utcDate"].replace("Z", "+00:00"))
            home = REGISTRY.canonical(row["homeTeam"]["name"]) or row["homeTeam"]["name"]
            away = REGISTRY.canonical(row["awayTeam"]["name"]) or row["awayTeam"]["name"]
            season = season_for(kickoff)
            full = (row.get("score") or {}).get("fullTime") or {}
            if start <= kickoff < end:
                result.append(Fixture(
                    fixture_id=f"PL:{season}:{home}:{away}" if code == "PL" else f"{code}:fd{row['id']}",
                    competition=competition, season=season, matchday=row.get("matchday") if code == "PL" else None,
                    home=home, away=away, kickoff_utc=kickoff, status=FD_STATUS.get(row["status"], FixtureStatus.UNKNOWN),
                    home_goals=full.get("home"), away_goals=full.get("away"), venue=row.get("venue"),
                    source="football-data.org", source_ids={"football_data": str(row["id"])}))
        return result


# --------------------------------------------------------------------------- openfootball (keyless)
class OpenFootballProvider(FixtureProvider):
    """PL fixtures/results from the community-maintained openfootball repo."""

    name = "openfootball"
    url = "https://raw.githubusercontent.com/openfootball/england/master/{season}/1-premierleague.txt"

    def __init__(self, http, season: str, text: str | None = None):
        super().__init__(http)
        self.season = season
        self._text = text  # Injected for tests/demo; otherwise fetched.

    def text(self) -> str:
        if self._text is None:
            return self.http.get(self.url.format(season=self.season), ttl_seconds=3 * 3600).text
        return self._text

    def all_fixtures(self, now: datetime | None = None) -> list[Fixture]:
        text = self.text()
        header = parse_header(text)
        day, matchday, kickoff_time, result = None, None, None, []
        for line in text.splitlines():
            if m := re.match(r"^\s*▪\s*Matchday\s+(\d+)", line):
                matchday, day = int(m[1]), None
                continue
            if parts := parse_date_parts(line, full_line=True):
                day = resolve_date(parts, header)
                continue
            if not re.search(r"\s+v\s+", line):
                continue
            leading, rest = parse_leading_time(line)
            kickoff_time = leading or kickoff_time
            score = SCORE_RE.search(rest)
            names = rest[:score.start()] if score else rest
            home_raw, away_raw = re.split(r"\s+v\s+", names.strip(), maxsplit=1)
            home, away = REGISTRY.canonical(home_raw.strip()), REGISTRY.canonical(away_raw.strip())
            if day is None or home is None or away is None:
                raise ProviderError(f"openfootball line could not be resolved: {line.strip()}")
            kickoff = datetime.combine(day, kickoff_time or time(15, 0), tzinfo=LONDON).astimezone(timezone.utc)
            if score:
                status = FixtureStatus.FINISHED
            elif now is not None and kickoff + timedelta(hours=2) < now:
                status = FixtureStatus.UNKNOWN  # Kickoff passed but no result published yet.
            else:
                status = FixtureStatus.SCHEDULED
            result.append(Fixture(
                fixture_id=f"PL:{self.season}:{home}:{away}", competition=PREMIER_LEAGUE, season=self.season,
                matchday=matchday, home=home, away=away, kickoff_utc=kickoff, status=status,
                home_goals=int(score["home"]) if score else None, away_goals=int(score["away"]) if score else None,
                source="openfootball/england (community, public domain)",
                kickoff_confirmed=kickoff_time is not None))
        return result

    def fixtures(self, team, start, end):
        return [f for f in self.all_fixtures(self.http._clock()) if team in (f.home, f.away) and start <= f.kickoff_utc < end]
