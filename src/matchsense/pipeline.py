"""End-to-end orchestration: question -> fixture -> evidence -> lineups -> forecast.

Every stage degrades gracefully: a missing key, provider outage or missing
squad data is recorded in ``data_quality`` and the response falls back
towards the unadjusted baseline instead of inventing information.
"""
from __future__ import annotations

import logging
import threading
from datetime import datetime, timedelta

import numpy as np

from src.matchsense import __version__
from src.matchsense.baseline import BaselineEngine, BaselineUnavailable
from src.matchsense.config import Settings, load_settings
from src.matchsense.http import TTL_CAP, HttpCache, HttpClient
from src.matchsense.impact import ImpactArtifact
from src.matchsense.intelligence import ExtractionCache, LLMExtractor, RuleExtractor, bind_events, corroborate, deduplicate
from src.matchsense.interpreter import (CodexInterpreter, InterpretationCache, OpenAIInterpreter,
                                        apply_interpretation)
from src.matchsense.lineups import LineupError, build_team_lineup_model, describe, sample_lineups
from src.matchsense.players import resolve_player
from src.matchsense.providers.base import StatusLog
from src.matchsense.providers.demo import DemoLineups, DemoNews, manifest, recorded_transport
from src.matchsense.providers.fixtures_extra import (APIFootballProvider, FootballDataProvider, OpenFootballProvider,
                                                     season_for)
from src.matchsense.providers.fpl import FPLProvider
from src.matchsense.providers.news import GNews, BBCSportRSS, ExtraRSS, GuardianNews, NewsAPI
from src.matchsense.query import ParsedQuery, parse_query
from src.matchsense.schemas import (AskResponse, Contribution, Fixture, FixtureStatus, Forecast,
                                    OutcomeProbabilities)
from src.matchsense.simulate import baseline_draws, block_from_grid, draws_payload, expected_delta, simulate
from src.matchsense.storage import PredictionStore

log = logging.getLogger(__name__)
LIVE_WINDOW = timedelta(hours=2, minutes=30)
_BASELINES: dict[str, BaselineEngine] = {}
_BASELINE_LOCK = threading.Lock()

IMPACT_STATUS_TEXT = {
    "validated": "Lineup effects validated out of sample on 2025-26 walk-forward folds.",
    "provisional_prior": ("PROVISIONAL: lineup effects could not be identified from one season of lineup data "
                          "(walk-forward confidence intervals include zero). Adjustments use a conservative, "
                          "documented prior updated with 2025-26 data; they are not empirically validated."),
    "missing": "Lineup-impact artifact missing; only the baseline is shown.",
}


def _baseline(settings: Settings) -> BaselineEngine:
    key = str(settings.baseline_bundle)
    with _BASELINE_LOCK:
        if key not in _BASELINES:
            _BASELINES[key] = BaselineEngine(settings.baseline_bundle)
        return _BASELINES[key]


class MatchSense:
    def __init__(self, settings: Settings | None = None, transport=None, extractor=None,
                 store: PredictionStore | None = None, baseline: BaselineEngine | None = None,
                 impact: ImpactArtifact | None = None, openfootball_text: str | None = None):
        self.settings = settings or load_settings()
        s = self.settings
        if s.is_demo:
            if s.demo_now is None:
                s.demo_now = datetime.fromisoformat(manifest()["demo_now"].replace("Z", "+00:00"))
            transport = transport or recorded_transport()
        self.http = HttpClient(HttpCache(s.cache_path), s.user_agent, s.http_timeout,
                               min_interval={"fantasy.premierleague.com": 0.15, "v3.football.api-sports.io": 6.5,
                                             "api.football-data.org": 6.5},
                               transport=transport, clock=s.now)
        self.fpl = FPLProvider(self.http)
        self.api_football = APIFootballProvider(self.http, s.api_football_key)
        self.football_data = FootballDataProvider(self.http, s.football_data_key)
        self.openfootball_text = openfootball_text
        if s.is_demo:
            self.news_providers = [DemoNews(self.http)]
            self.lineup_providers = [DemoLineups(self.http)]
        else:
            self.news_providers = [p for p in (
                BBCSportRSS(self.http) if s.use_bbc else None, GuardianNews(self.http, s.guardian_key),
                NewsAPI(self.http, s.newsapi_key), GNews(self.http, s.gnews_key),
                ExtraRSS(self.http, s.extra_rss) if s.extra_rss else None) if p is not None]
            self.lineup_providers = [self.api_football]
        if extractor is not None:
            self.extractor = extractor
        elif s.openai_key and not s.is_demo:
            self.extractor = LLMExtractor(s.openai_key, s.openai_model,
                                          cache=ExtractionCache(s.data_dir / "llm_extractions.sqlite"))
        else:
            self.extractor = RuleExtractor()
        self.rules = RuleExtractor()
        self.interpreter = self._make_interpreter()
        self.interpretations = InterpretationCache(s.data_dir / "query_interpretations.sqlite")
        self.store = store or PredictionStore(s.db_path)
        self._baseline = baseline
        self._impact = impact
        self._impact_loaded = impact is not None

    def _make_interpreter(self):
        s = self.settings
        choice = s.query_interpreter
        if choice == "auto":
            choice = ("rules" if s.is_demo else "codex" if CodexInterpreter.available()
                      else "openai" if s.openai_key else "rules")
        if choice == "codex" and CodexInterpreter.available():
            return CodexInterpreter(model=s.codex_model)
        if choice in ("openai", "codex") and s.openai_key:
            return OpenAIInterpreter(s.openai_key, s.openai_model)
        return None

    def understand(self, parsed: ParsedQuery, now, tz, log_: StatusLog) -> dict:
        """Ask the LLM interpreter what the question means (cached), then validate and merge."""
        info = {"backend": "rules", "cached": False, "changes": []}
        if self.interpreter is None:
            return info
        backend = self.interpreter.name
        info["backend"] = backend
        interp = self.interpretations.get(backend, parsed.text)
        if interp is not None:
            info["cached"] = True
        else:
            try:
                interp = self.interpreter.interpret(parsed.text)
            except Exception as exc:  # Timeout, auth, malformed output: keep the rule parse.
                log_.add("Query understanding", "error", f"{backend} failed ({str(exc)[:160]}); used rule parser")
                info["backend"] = "rules (fallback)"
                return info
            self.interpretations.put(backend, parsed.text, interp)
        info["raw"] = interp.model_dump()
        info["changes"] = apply_interpretation(parsed, interp, now, tz)
        log_.add("Query understanding", "inferred",
                 f"{backend}{' (cached, no tokens)' if info['cached'] else ''}: intent {interp.intent}"
                 + (f"; adjusted {', '.join(info['changes'])}" if info["changes"] else "; agreed with rule parser"))
        return info

    # ------------------------------------------------------------------ components
    @property
    def baseline(self) -> BaselineEngine:
        if self._baseline is None:
            self._baseline = _baseline(self.settings)
        return self._baseline

    @property
    def impact(self) -> ImpactArtifact | None:
        if not self._impact_loaded:
            path = self.settings.impact_artifact
            self._impact = ImpactArtifact.load(path) if path.exists() else None
            self._impact_loaded = True
        return self._impact

    def _openfootball(self, season: str) -> OpenFootballProvider:
        return OpenFootballProvider(self.http, season, self.openfootball_text)

    def season_fixtures(self, season: str, log_: StatusLog) -> tuple[list[Fixture], str]:
        if self.settings.use_fpl:
            fixtures = log_.guard("FPL fixtures", self.fpl.all_fixtures)
            if fixtures:
                return fixtures, self.fpl.source_label
        if self.settings.use_openfootball_remote and not self.settings.is_demo or self.openfootball_text:
            fixtures = log_.guard("openfootball fixtures", self._openfootball(season).all_fixtures, self.settings.now())
            if fixtures:
                return fixtures, "openfootball/england"
        return [], "none"

    def find_fixtures(self, team: str, start: datetime, end: datetime, log_: StatusLog) -> list[Fixture]:
        found: dict[tuple, Fixture] = {}
        season = season_for(start)
        sources = []
        if self.api_football.usable:
            sources.append(("API-Football", lambda: self.api_football.fixtures(team, start, end)))
        sources.append(("PL season fixtures", lambda: [f for f in self.season_fixtures(season, log_)[0]
                                                       if team in (f.home, f.away)
                                                       and (start <= f.kickoff_utc < end or f.status == FixtureStatus.POSTPONED)]))
        if self.settings.football_data_key:
            sources.append(("football-data.org", lambda: self.football_data.fixtures(team, start, end)))
        for name, call in sources:
            for fixture in log_.guard(name, call, default=[]) or []:
                key = (fixture.competition.code, fixture.home, fixture.away, fixture.kickoff_utc.date()
                       if fixture.status != FixtureStatus.POSTPONED else None)
                if key not in found:  # Earlier sources have priority.
                    found[key] = fixture
        return sorted(found.values(), key=lambda f: f.kickoff_utc)

    # ------------------------------------------------------------------ public API
    def ask(self, question: str, timezone: str | None = None, as_of: datetime | None = None,
            refresh: bool = False) -> AskResponse:
        token = TTL_CAP.set(30.0) if refresh else None
        try:
            return self._ask(question, timezone or self.settings.timezone, as_of or self.settings.now())
        finally:
            if token is not None:
                TTL_CAP.reset(token)

    def _respond(self, parsed, status, message, now, tz, log_, **extra) -> AskResponse:
        notes = list(extra.pop("notes", []))
        if parsed is not None:
            notes = parsed.ambiguities + notes
        if self.settings.is_demo:
            notes.insert(0, "DEMO MODE: FPL data is a recorded snapshot and all news items and official lineups "
                            "are SIMULATED. Nothing here is real-time evidence.")
        if not (self.api_football.usable or self.settings.football_data_key):
            notes.append("Only Premier League fixtures were checked (FPL/openfootball). Cup and European matches "
                         "cannot be ruled out without API_FOOTBALL_KEY or FOOTBALL_DATA_API_KEY.")
        understanding = getattr(log_, "understanding", None) if parsed is not None else None
        if understanding is not None and parsed is not None:
            understanding = {**understanding, "subject": parsed.subject, "opponent": parsed.opponent,
                             "intent": parsed.intent, "window": parsed.window.label,
                             "players": parsed.what_if_players, "players_kind": parsed.what_if_kind}
        extra.setdefault("interpretation", understanding)
        return AskResponse(question=parsed.text if parsed else "", status=status, message=message,
                           mode=self.settings.mode, timezone=tz, generated_at=now, data_quality=log_.entries,
                           notes=notes, window=parsed.window.as_dict() if parsed else None,
                           subject_team=parsed.subject if parsed else None, **extra)

    def _local(self, moment: datetime, tz: str) -> str:
        from zoneinfo import ZoneInfo
        return moment.astimezone(ZoneInfo(tz)).strftime("%a %d %b %H:%M") + f" ({tz})"

    def _ask(self, question: str, tz: str, now: datetime) -> AskResponse:
        log_ = StatusLog()
        try:
            parsed = parse_query(question, now, tz)
        except ValueError as exc:
            return self._respond(None, "unresolved", str(exc), now, tz, log_)
        understanding = self.understand(parsed, now, tz, log_)
        log_.understanding = understanding  # Per-request; the engine is shared across threads.
        if parsed.what_if_players:
            self._subject_from_players(parsed, log_)
        if parsed.subject is None:
            return self._respond(parsed, "unresolved", "I couldn't identify a club in that question. Try e.g. "
                                 "'Will Manchester City win tonight?'", now, tz, log_)
        team, opponent = parsed.subject, parsed.opponent
        horizon = timedelta(days=200 if opponent else 120)
        upcoming = self.find_fixtures(team, now - LIVE_WINDOW, now + horizon, log_)
        if opponent:
            upcoming = [f for f in upcoming if opponent in (f.home, f.away)]
        next_fixture = next((f for f in upcoming if f.status in (FixtureStatus.SCHEDULED, FixtureStatus.LIVE,
                                                                  FixtureStatus.UNKNOWN)
                             and f.kickoff_utc + LIVE_WINDOW > now), None)
        if parsed.window.start_utc is not None:
            in_window = [f for f in self.find_fixtures(team, parsed.window.start_utc, parsed.window.end_utc, log_)
                         if (opponent is None or opponent in (f.home, f.away))
                         and f.status != FixtureStatus.POSTPONED or (
                             f.status == FixtureStatus.POSTPONED and parsed.window.contains(f.kickoff_utc))]
            if not in_window:
                vs = f" against {opponent}" if opponent else ""
                window = parsed.window.as_dict()
                message = (f"{team} have no confirmed fixture{vs} {parsed.window.label} "
                           f"({window['start_local']} – {window['end_local']} {tz}).")
                if next_fixture:
                    message += (f" Their next confirmed match is {next_fixture.home} v {next_fixture.away} "
                                f"({next_fixture.competition.name}), {self._local(next_fixture.kickoff_utc, tz)}.")
                return self._respond(parsed, "no_fixture", message, now, tz, log_, next_fixture=next_fixture)
            fixture = in_window[0]
        else:
            if next_fixture is None:
                vs = f" against {opponent}" if opponent else ""
                return self._respond(parsed, "no_fixture", f"No upcoming {team} fixture{vs} was found in the "
                                     "configured sources.", now, tz, log_)
            fixture = next_fixture
            if not fixture.competition.model_supported and parsed.intent != "next":
                # Undated question: forecast the next match the model covers, and say what was skipped.
                supported = next((f for f in upcoming if f.competition.model_supported
                                  and f.status == FixtureStatus.SCHEDULED and f.kickoff_utc > fixture.kickoff_utc), None)
                if supported:
                    parsed.ambiguities.append(
                        f"{team}'s next match is {fixture.home} v {fixture.away} ({fixture.competition.name}, "
                        f"{self._local(fixture.kickoff_utc, tz)}), which the Premier League model does not cover; "
                        f"this answer is for their next Premier League match instead.")
                    fixture = supported
        return self._fixture_response(parsed, fixture, now, tz, log_, next_fixture)

    def _subject_from_players(self, parsed: ParsedQuery, log_: StatusLog):
        """Make the named player's club the subject ("odds if Khusanov doesn't play")."""
        clubs = log_.guard("Player lookup", lambda: list(self.fpl._team_maps()[1]), default=[]) or []
        found = []
        for team in clubs:
            squad = log_.guard(f"Squad ({team})", self.fpl.squad, team, default=[]) or []
            if resolve_player(parsed.what_if_players[0], squad, threshold=0.85).player:
                found.append(team)
        if len(found) != 1:
            if len(found) > 1:
                parsed.ambiguities.append(f'"{parsed.what_if_players[0]}" matches players at {", ".join(found)}; '
                                          "name the club to choose.")
            return
        club = found[0]
        if parsed.subject != club:
            others = [t for t in parsed.teams if t != club]
            parsed.subject, parsed.opponent = club, (others[0] if others else None)
            parsed.teams = [club] + others

    def _fixture_response(self, parsed: ParsedQuery, fixture: Fixture, now, tz, log_, next_fixture) -> AskResponse:
        label = f"{fixture.home} v {fixture.away} ({fixture.competition.name}), {self._local(fixture.kickoff_utc, tz)}"
        log_.add("Fixture", "confirmed" if fixture.kickoff_confirmed else "reported",
                 f"{label}; source {fixture.source}", now)
        started = fixture.kickoff_utc <= now
        if fixture.status in (FixtureStatus.POSTPONED, FixtureStatus.CANCELLED):
            later = next_fixture if next_fixture and next_fixture.fixture_id != fixture.fixture_id else None
            return self._respond(parsed, "postponed", f"{fixture.home} v {fixture.away} is {fixture.status.value}."
                                 + (f" Next confirmed match: {later.home} v {later.away}, "
                                    f"{self._local(later.kickoff_utc, tz)}." if later else ""),
                                 now, tz, log_, fixture=fixture, next_fixture=later)
        if fixture.status == FixtureStatus.FINISHED:
            self.store.record_outcomes([fixture], fixture.source)
            stored = self.store.latest_for(fixture.fixture_id, before=fixture.kickoff_utc)
            msg = f"{label} has finished: {fixture.home} {fixture.home_goals}–{fixture.away_goals} {fixture.away}."
            if stored:
                msg += (f" MatchSense's last pre-match forecast (adjusted) was H {stored['adjusted_home']:.0%} / "
                        f"D {stored['adjusted_draw']:.0%} / A {stored['adjusted_away']:.0%}.")
            return self._respond(parsed, "finished", msg, now, tz, log_, fixture=fixture, next_fixture=next_fixture,
                                 prediction_id=stored["prediction_id"] if stored else None)
        if fixture.status == FixtureStatus.LIVE or (started and now < fixture.kickoff_utc + LIVE_WINDOW):
            stored = self.store.latest_for(fixture.fixture_id, before=fixture.kickoff_utc)
            score = (f" Current score {fixture.home_goals}–{fixture.away_goals}"
                     + (f" ({fixture.minute}')" if fixture.minute else "") + "." if fixture.home_goals is not None else "")
            msg = (f"{label} is in progress.{score} MatchSense makes pre-match forecasts only; it does not produce "
                   "in-play probabilities.")
            if stored:
                msg += (f" Last pre-match forecast: H {stored['adjusted_home']:.0%} / D {stored['adjusted_draw']:.0%} / "
                        f"A {stored['adjusted_away']:.0%}.")
            return self._respond(parsed, "live", msg, now, tz, log_, fixture=fixture,
                                 prediction_id=stored["prediction_id"] if stored else None)
        if not fixture.competition.model_supported:
            later = [f for f in self.find_fixtures(parsed.subject, fixture.kickoff_utc, now + timedelta(days=120), log_)
                     if f.competition.model_supported and f.kickoff_utc > fixture.kickoff_utc
                     and f.status == FixtureStatus.SCHEDULED]
            next_pl = later[0] if later else None
            msg = (f"{label}: the MatchSense model is calibrated on Premier League matches only. Applying it to "
                   f"{fixture.competition.name} would ignore distribution shift (different opponents, rotation and "
                   "incentives), so no probability is shown.")
            if next_pl:
                msg += (f" Next Premier League match: {next_pl.home} v {next_pl.away}, "
                        f"{self._local(next_pl.kickoff_utc, tz)}.")
            return self._respond(parsed, "unsupported_competition", msg, now, tz, log_, fixture=fixture,
                                 next_fixture=next_pl)
        return self._forecast(parsed, fixture, now, tz, log_, label)

    # ------------------------------------------------------------------ forecasting
    def _evidence(self, fixture: Fixture, squads, now, log_):
        since = now - timedelta(hours=self.settings.news_window_hours)
        items = []
        for provider in self.news_providers:
            for team in (fixture.home, fixture.away):
                got = log_.guard(f"News: {provider.name}", provider.news, team, since, now, default=None)
                if got is not None:
                    items.extend(got)
        for provider in self.news_providers:
            count = sum(1 for i in items if i.provider == provider.name)
            if not any(e.name.startswith(f"News: {provider.name}") for e in log_.entries):
                log_.add(f"News: {provider.name}", "simulated" if self.settings.is_demo else "ok",
                         f"{count} items in the last {self.settings.news_window_hours}h", now)
        unique = deduplicate(items)
        if len(unique) < len(items):
            log_.add("News deduplication", "ok", f"{len(items)} items -> {len(unique)} stories after syndication merge")
        label = f"{fixture.home} v {fixture.away}, {fixture.kickoff_utc.isoformat()}"
        extractor = self.extractor
        try:
            extracted = extractor.extract(unique, squads, label)
        except Exception as exc:  # LLM outage/quota: fall back to deterministic rules.
            log_.add("Semantic extraction", "error", f"{getattr(extractor, 'name', 'extractor')} failed ({exc}); used rules")
            extractor = self.rules
            extracted = extractor.extract(unique, squads, label)
        stats = getattr(extractor, "last_stats", None)
        cache_note = f"; {stats['cached']} from cache, {stats['sent']} sent to the API" if stats else ""
        log_.add("Semantic extraction", "inferred", f"extractor: {extractor.name}; {len(unique)} stories{cache_note}")
        events = bind_events(unique, extracted, squads, extractor.name)
        for team in (fixture.home, fixture.away):
            if team in squads:
                events += log_.guard(f"FPL availability flags ({team})", self.fpl.status_events, team, now,
                                     default=[]) or []
        if self.api_football.usable:
            events += log_.guard("API-Football injuries", self.api_football.injuries, fixture, now, default=[]) or []
        unresolved = [e for e in events if e.category.value == "player_availability" and e.player_id is None]
        if unresolved:
            log_.add("Player identity", "inferred", f"{len(unresolved)} availability mentions could not be matched "
                     "to a squad player and were ignored: " + ", ".join(sorted({e.entity for e in unresolved}))[:300])
        return [e for e in events if e.published_at <= now], unique

    def _forecast(self, parsed: ParsedQuery, fixture: Fixture, now, tz, log_, label) -> AskResponse:
        season_fixtures, results_source = self.season_fixtures(fixture.season, log_)
        try:
            base = self.baseline.predict(fixture, season_fixtures, now)
        except (BaselineUnavailable, ValueError) as exc:
            return self._respond(parsed, "error", f"{label}: the baseline model could not be evaluated ({exc}).",
                                 now, tz, log_, fixture=fixture)
        self.store.record_outcomes(season_fixtures, results_source)
        log_.add("Baseline state", "ok" if base.results_used else "missing",
                 f"{base.results_used} completed {fixture.season} results applied (latest {base.latest_result}) "
                 f"from {results_source}", now)
        baseline_block = block_from_grid("Baseline ML (existing Poisson ScoreModel)", base.grid)

        squads, appearances = {}, {}
        for team in (fixture.home, fixture.away):
            squad = log_.guard(f"Squad ({team})", self.fpl.squad, team)
            apps = log_.guard(f"Selection history ({team})", self.fpl.appearances, team, now) if squad else None
            if squad and apps is not None:
                squads[team], appearances[team] = squad, apps
                log_.add(f"Squad ({team})", "ok", f"{len(squad)} players, {len({a.kickoff_utc for a in apps})} "
                         "earlier matches", now)
        events, stories = self._evidence(fixture, squads, now, log_)
        availability = corroborate(events, squads, now)
        conflicts = [a for a in availability if a.conflict]
        if conflicts:
            log_.add("Conflicting reports", "reported", "; ".join(f"{a.name}: {a.note}" for a in conflicts))

        confirmed = None
        for provider in self.lineup_providers:
            confirmed = log_.guard(f"Official lineups ({provider.name})", provider.confirmed_lineups, fixture, now)
            if confirmed:
                break
        if not confirmed:
            minutes = (fixture.kickoff_utc - now).total_seconds() / 60
            log_.add("Official lineups", "missing", "not yet released" if minutes < 75 else
                     f"normally released ~60 min before kickoff ({minutes / 60:.1f}h from now)")

        impact = self.impact
        what_if = None
        adjusted_block, contributions, lineups, explanation = None, [], {}, []
        lineup_models = {}
        simulations = None
        player_impacts = []
        selection = impact.selection if impact else None
        if impact is None:
            log_.add("Lineup-impact model", "missing", IMPACT_STATUS_TEXT["missing"])
        elif len(squads) == 2:
            try:
                for team in (fixture.home, fixture.away):
                    lineup_models[team] = build_team_lineup_model(
                        team, squads[team], appearances[team], fixture.kickoff_utc, selection,
                        [a for a in availability if a.team == team], (confirmed or {}).get(team))
                    model = lineup_models[team]
                    if model.unresolved_confirmed:
                        log_.add(f"Official XI ({team})", "error", "Could not match: " +
                                 ", ".join(model.unresolved_confirmed) + "; using predicted XI")
            except LineupError as exc:
                log_.add("Lineup prediction", "error", str(exc))
                lineup_models = {}
        else:
            log_.add("Lineup prediction", "missing", "Squad data unavailable for one or both teams; "
                     "showing the baseline only")

        n = self.settings.simulations
        seed = self.settings.seed
        if lineup_models:
            home, away = lineup_models[fixture.home], lineup_models[fixture.away]
            sim = simulate(base.rates, home, away, impact.impact, self.baseline.grid_for_rates, n, seed)
            adjusted_block = block_from_grid("News- and lineup-adjusted (Monte Carlo)", sim.grid, sim.per_draw)
            simulations = draws_payload(sim.per_draw, sim.rates, sim.scores, "lineup_monte_carlo")
            for team, model, samples in ((fixture.home, home, sim.home_samples), (fixture.away, away, sim.away_samples)):
                kind = "confirmed" if model.confirmed is not None else "predicted"
                lineups[team] = describe(model, samples, "Official XI (API provider)" if kind == "confirmed" else "")
                log_.add(f"Lineup ({team})", "confirmed" if kind == "confirmed" else "inferred",
                         lineups[team].note, now)
            contributions, explanation = self._contributions(fixture, base, home, away, impact, availability, events)
            player_impacts = self._player_impacts(fixture, base, home, away, impact.impact, sim)
            if parsed.intent == "what_if":
                what_if = self._what_if(parsed, fixture, base, home, away, impact, squads, availability,
                                        (confirmed or {}), appearances, n, seed)
        else:
            explanation.append("No lineup adjustment was applied; the forecast equals the existing baseline model.")
            simulations = baseline_draws(base.grid, base.rates, n, seed)

        logistic = None
        if base.logistic is not None:
            p = base.logistic / base.logistic.sum()
            logistic = OutcomeProbabilities(home_win=float(p[0]), draw=float(p[1]), away_win=float(1 - p[0] - p[1]))
        status_key = impact.status if impact else "missing"
        forecast = Forecast(baseline=baseline_block, adjusted=adjusted_block, logistic_comparison=logistic,
                            contributions=contributions,
                            adjustment_method=("Baseline Poisson rates x exp(b . lineup features); 4 team-level "
                                               "lineup features; Monte Carlo over sampled XIs and Laplace posterior "
                                               "of b." if impact else "none"),
                            adjustment_status=IMPACT_STATUS_TEXT.get(status_key, status_key),
                            explanation=explanation, simulations=simulations, player_impacts=player_impacts)
        log_.add("Lineup-impact model", "inferred" if impact else "missing", IMPACT_STATUS_TEXT.get(status_key, ""))
        message = self._headline(parsed, fixture, forecast, tz)
        p = (forecast.adjusted or forecast.baseline).probabilities
        home_subject = parsed.subject == fixture.home
        subject_probs = {"team": parsed.subject, "opponent": fixture.away if home_subject else fixture.home,
                         "win": p.home_win if home_subject else p.away_win, "draw": p.draw,
                         "lose": p.away_win if home_subject else p.home_win, "side": "home" if home_subject else "away"}
        version = (f"MatchSense {__version__} | baseline: {self.baseline.version} | lineup layer: "
                   f"{impact.impact.meta.get('candidate') if impact else 'none'} ({status_key})")
        response = self._respond(parsed, "forecast", message, now, tz, log_, fixture=fixture, forecast=forecast,
                                 lineups=lineups, news=sorted(events, key=lambda e: e.published_at, reverse=True),
                                 availability=availability, what_if=what_if, model_version=version,
                                 subject_probabilities=subject_probs)
        response.prediction_id = self.store.save(response, "confirmed" if confirmed else
                                                 "predicted" if lineups else "none")
        return response

    def _headline(self, parsed, fixture, forecast, tz) -> str:
        block = forecast.adjusted or forecast.baseline
        p = block.probabilities
        subject = parsed.subject
        if subject == fixture.home:
            win, lose, venue, opp = p.home_win, p.away_win, "at home to", fixture.away
            base_win = forecast.baseline.probabilities.home_win
        else:
            win, lose, venue, opp = p.away_win, p.home_win, "away at", fixture.home
            base_win = forecast.baseline.probabilities.away_win
        msg = (f"{subject} ({venue} {opp}, {self._local(fixture.kickoff_utc, tz)}): win {win:.0%}, draw {p.draw:.0%}, "
               f"lose {lose:.0%}.")
        if forecast.adjusted is not None:
            msg += f" Baseline model alone: win {base_win:.0%}."
        return msg

    # ------------------------------------------------------------------ explanations
    def _contributions(self, fixture, base, home, away, artifact, availability, events):
        impact = artifact.impact
        names = {p.player_id: p.name for m in (home, away) for p in m.squad}
        evidence_by_player = {}
        for e in events:
            if e.player_id:
                evidence_by_player.setdefault(e.player_id, []).append(e.event_id)
        basis = "provisional lineup prior" if artifact.status != "validated" else "validated lineup model"

        def deltas(p_avail_home, p_avail_away):
            ph = home.p_selection * p_avail_home if home.confirmed is None else home.confirmed
            pa = away.p_selection * p_avail_away if away.confirmed is None else away.confirmed
            return expected_delta(home, away, impact, ph, pa)

        full = deltas(home.p_available, away.p_available)
        no_news = deltas(np.ones_like(home.p_available), np.ones_like(away.p_available))
        lam = base.rates
        contributions = []
        explanation = []

        def xg(d):
            return lam[0] * np.exp(d[0]), lam[1] * np.exp(d[1])

        full_xg = xg(full)
        rotation_xg = xg(no_news)
        for team, model, idx_home in ((fixture.home, home, True), (fixture.away, away, False)):
            if model.confirmed is not None:
                continue
            for i, pid in enumerate(model.pool):
                if model.p_available[i] >= 0.999:
                    continue
                restored_h = home.p_available.copy()
                restored_a = away.p_available.copy()
                (restored_h if idx_home else restored_a)[i] = 1.0
                without = xg(deltas(restored_h, restored_a))
                d_home, d_away = full_xg[0] - without[0], full_xg[1] - without[1]
                if abs(d_home) + abs(d_away) < 0.005:
                    continue
                own, against = (d_home, d_away) if idx_home else (d_away, d_home)
                contributions.append(Contribution(
                    factor=f"{names[pid]} availability {model.p_available[i]:.0%}", team=team,
                    delta_xg_for_team=round(float(own), 3), delta_xg_against_team=round(float(against), 3),
                    evidence_ids=evidence_by_player.get(pid, []), basis=basis))
        for team, model, idx_home in ((fixture.home, home, True), (fixture.away, away, False)):
            own = (rotation_xg[0] - lam[0]) if idx_home else (rotation_xg[1] - lam[1])
            against = (rotation_xg[1] - lam[1]) if idx_home else (rotation_xg[0] - lam[0])
            kind = "Official XI vs usual starters" if model.confirmed is not None else \
                "Predicted selection/rotation vs usual starters (no news)"
            contributions.append(Contribution(factor=kind, team=team, delta_xg_for_team=round(float(own), 3),
                                              delta_xg_against_team=round(float(against), 3), basis=basis))
        contributions.sort(key=lambda c: -(abs(c.delta_xg_for_team) + abs(c.delta_xg_against_team)))
        explanation.append(f"Baseline expected goals {lam[0]:.2f}–{lam[1]:.2f}; after lineup evidence "
                           f"{full_xg[0]:.2f}–{full_xg[1]:.2f} (at expected lineups).")
        explanation.append("Effects are model-based associations estimated with a provisional prior; they "
                           "describe how the forecast responds to the inputs, not proven causal effects."
                           if artifact.status != "validated" else
                           "Effects are model-based associations; they are not proven causal effects.")
        for team, model in ((fixture.home, home), (fixture.away, away)):
            missing = [names[pid] for i, pid in enumerate(model.pool)
                       if model.p_available[i] < 0.25 and model.typical[i] > 0.3]
            if missing:
                explanation.append(f"{team} regulars expected to miss out: {', '.join(missing)}.")
        return contributions, explanation

    def _player_impacts(self, fixture, base, home, away, impact, sim, draws: int = 400) -> list[dict]:
        """Model-implied cost of each likely starter's absence (other inputs held fixed).

        Uses the same position-constrained XI sampler as the forecast, with
        common random numbers so differences reflect the absence, not noise.
        """
        from src.ml.season_models import outcomes

        def expected_starts(model, p_avail):
            if model.confirmed is not None and np.all(p_avail == model.p_available):
                return model.confirmed
            p = np.clip(model.p_selection * p_avail, 0, 1)
            return sample_lineups(p, model.positions, model.shape, draws, np.random.default_rng(self.settings.seed)).mean(axis=0)

        def evaluate(sh, sa):
            h_in, a_in = home.impact_inputs(sh), away.impact_inputs(sa)
            rates = np.array([base.rates[0] * np.exp(impact.delta_eta(h_in, a_in)),
                              base.rates[1] * np.exp(impact.delta_eta(a_in, h_in))])
            return outcomes(self.baseline.grid_for_rates(rates))[0], rates

        ref_h, ref_a = expected_starts(home, home.p_available), expected_starts(away, away.p_available)
        ref, ref_rates = evaluate(ref_h, ref_a)
        rows = []
        for model, samples, is_home in ((home, sim.home_samples, True), (away, sim.away_samples, False)):
            marginal = samples.mean(axis=0)
            names = {p.player_id: p for p in model.squad}
            for i in np.flatnonzero(marginal >= 0.5):
                p_avail = model.p_available.copy()
                p_avail[i] = 0.0
                starts = expected_starts(model, p_avail)
                probs, rates = evaluate(starts, ref_a) if is_home else evaluate(ref_h, starts)
                k, own, opp = (0, 0, 1) if is_home else (2, 1, 0)
                player = names[model.pool[i]]
                rows.append({"player": player.name, "short_name": player.short_name, "team": model.team,
                             "position": player.position.value, "p_start": round(float(marginal[i]), 3),
                             "goal_share": round(float(model.goal_share[i]), 3),
                             "typical_start_rate": round(float(model.typical[i]), 3),
                             "delta_win": round(float(probs[k] - ref[k]), 4),
                             "delta_xg_for": round(float(rates[own] - ref_rates[own]), 3),
                             "delta_xg_against": round(float(rates[opp] - ref_rates[opp]), 3)})
        return sorted(rows, key=lambda r: r["delta_win"])

    def _what_if(self, parsed, fixture, base, home, away, artifact, squads, availability, confirmed, appearances,
                 n, seed):
        resolved, unresolved = [], []
        order = [parsed.subject] + [t for t in (fixture.home, fixture.away) if t != parsed.subject]
        for name in parsed.what_if_players:
            hit = None
            for team in order:
                r = resolve_player(name, squads.get(team, []))
                if r.player:
                    hit = r.player
                    break
            (resolved.append(hit) if hit else unresolved.append(name))
        if not resolved:
            return {"error": f"Could not match {', '.join(unresolved)} to either squad.", "players": []}
        results, marginals = {}, {}
        affected = {p.team for p in resolved}
        for label, value in (("available", 1.0), ("unavailable", 0.0)):
            models = {}
            for team in (fixture.home, fixture.away):
                overrides = {p.player_id: value for p in resolved if p.team == team}
                # A counterfactual cannot keep the official XI of the team it changes.
                official = None if team in affected else confirmed.get(team)
                models[team] = build_team_lineup_model(team, squads[team], appearances[team], fixture.kickoff_utc,
                                                       artifact.selection,
                                                       [a for a in availability if a.team == team],
                                                       official, overrides=overrides)
            sim = simulate(base.rates, models[fixture.home], models[fixture.away], artifact.impact,
                           self.baseline.grid_for_rates, n, seed)
            results[label] = block_from_grid(f"Scenario: {', '.join(p.name for p in resolved)} {label}",
                                             sim.grid, sim.per_draw).model_dump()
            marginals[label] = {team: (models[team], (sim.home_samples if team == fixture.home
                                                      else sim.away_samples).mean(axis=0)) for team in affected}
        replacements = []
        removed = {p.player_id for p in resolved}
        for team in affected:
            model, without = marginals["unavailable"][team]
            _, with_ = marginals["available"][team]
            names = {p.player_id: p.name for p in model.squad}
            gains = [(without[i] - with_[i], pid) for i, pid in enumerate(model.pool) if pid not in removed]
            replacements += [{"player": names[pid], "team": team, "start_gain": round(float(g), 3)}
                             for g, pid in sorted(gains, reverse=True)[:3] if g > 0.05]
        subject_key = "home_win" if parsed.subject == fixture.home else "away_win"
        delta = results["unavailable"]["probabilities"][subject_key] - results["available"]["probabilities"][subject_key]
        note = ("Official XI ignored for " + ", ".join(sorted(t for t in affected if confirmed.get(t)))
                + " in this counterfactual; that side is re-predicted." if any(confirmed.get(t) for t in affected) else "")
        return {"players": [p.name for p in resolved], "unresolved": unresolved, "scenarios": results, "note": note,
                "replacements": replacements,
                "subject_win_change": delta,
                "summary": (f"Under the {('provisional' if artifact.status != 'validated' else 'validated')} lineup "
                            f"model, {parsed.subject}'s win probability changes by {delta * 100:+.1f} points when "
                            f"{', '.join(p.name for p in resolved)} {'is' if len(resolved) == 1 else 'are'} "
                            "unavailable versus available.")}
