"""MatchSense AI tests: recorded/mocked providers and deterministic model outputs.

Integration tests replay the recorded FPL snapshot and simulated demo news
through the real provider code (httpx mock transport); nothing touches the
network.
"""
from __future__ import annotations

import copy
import json
from datetime import datetime, timedelta, timezone
from pathlib import Path
from types import SimpleNamespace

import httpx
import numpy as np
import pytest

from src.matchsense.config import PROJECT_ROOT, load_settings
from src.matchsense.http import HttpCache, HttpClient, MissingCredentials, ProviderError
from src.matchsense.impact import ImpactArtifact, SelectionModel, discontinuity, normalize_to_eleven
from src.matchsense.intelligence import (LLMExtractor, RuleExtractor, bind_events, corroborate, deduplicate,
                                         _BatchExtraction)
from src.matchsense.lineups import LineupError, build_team_lineup_model, describe
from src.matchsense.players import resolve_player
from src.matchsense.providers.base import StatusLog
from src.matchsense.providers.news import GuardianNews
from src.matchsense.query import parse_query
from src.matchsense.schemas import (PREMIER_LEAGUE, Availability, Certainty, EventType, ExtractedEvent, Fixture,
                                    FixtureStatus, NewsItem, OutcomeProbabilities, PlayerAppearance,
                                    PlayerAvailability, Position, SquadPlayer, TeamLineup)
from src.matchsense.teams import REGISTRY

UTC = timezone.utc
BUNDLE = PROJECT_ROOT / "artifacts/season_simulator/model.joblib"
IMPACT = PROJECT_ROOT / "artifacts/matchsense/lineup_impact.json"
needs_models = pytest.mark.skipif(not (BUNDLE.exists() and IMPACT.exists()),
                                  reason="trained artifacts not present (run train_season_simulator / train_lineup_impact)")
DEMO_PRE = datetime(2026, 11, 7, 8, 0, tzinfo=UTC)
DEMO_XI = datetime(2026, 11, 7, 16, 45, tzinfo=UTC)
DEMO_MIDWEEK = datetime(2026, 11, 4, 12, 0, tzinfo=UTC)


# --------------------------------------------------------------------------- helpers

def make_squad(team="Testers FC", n_def=7, n_mid=8, n_fwd=4, n_gk=2):
    squad, i = [], 0
    for pos, count in ((Position.GK, n_gk), (Position.DEF, n_def), (Position.MID, n_mid), (Position.FWD, n_fwd)):
        for k in range(count):
            i += 1
            squad.append(SquadPlayer(player_id=f"t:{i}", name=f"Player{i} {pos.value}{k}", short_name=f"P{i}",
                                     team=team, position=pos))
    return squad


def make_history(squad, matches=8, start=datetime(2026, 8, 20, tzinfo=UTC)):
    """Player 1 (GK) and the first 4 DEF, 4 MID, 2 FWD always start."""
    regulars = [squad[0].player_id] + [p.player_id for p in squad if p.position == Position.DEF][:4] + \
        [p.player_id for p in squad if p.position == Position.MID][:4] + \
        [p.player_id for p in squad if p.position == Position.FWD][:2]
    apps = []
    for m in range(matches):
        kickoff = start + timedelta(days=7 * m)
        for pid in regulars:
            apps.append(PlayerAppearance(player_id=pid, team=squad[0].team, kickoff_utc=kickoff, started=True,
                                         minutes=90, goals=1 if pid == regulars[-1] and m % 2 == 0 else 0))
    return apps, regulars


@pytest.fixture(scope="module")
def selection():
    if IMPACT.exists():
        return ImpactArtifact.load(IMPACT).selection
    return SelectionModel()  # Unfitted fallback uses typical start rates.


@pytest.fixture(scope="module")
def demo(tmp_path_factory):
    if not (BUNDLE.exists() and IMPACT.exists()):
        pytest.skip("trained artifacts not present")
    from src.matchsense.pipeline import MatchSense
    settings = load_settings(mode="demo", simulations=300, data_dir=tmp_path_factory.mktemp("demo"))
    return MatchSense(settings)


# --------------------------------------------------------------------------- 1. name resolution

def test_manchester_city_name_resolution():
    now = datetime(2026, 11, 4, 12, tzinfo=UTC)
    for text in ("Will Man City win tonight?", "Will Manchester City win tonight?", "Can MCFC beat Arsenal?",
                 "Predict City's next game."):
        assert parse_query(text, now).subject == "Manchester City", text
    for spelling in ("Man City", "Manchester City FC", "manchester city"):
        assert REGISTRY.canonical(spelling) == "Manchester City"
    parsed = parse_query("Predict City's next game.", now)
    assert parsed.intent == "next" and parsed.ambiguities  # "City" alone is flagged, not silently assumed.
    assert parse_query("Coventry City vs Hull", now).teams == ["Coventry City", "Hull City"]
    q = parse_query("What are Arsenal's chances against Liverpool?", now)
    assert (q.subject, q.opponent) == ("Arsenal", "Liverpool")


def test_tonight_window_uses_user_timezone():
    now = datetime(2026, 11, 7, 8, tzinfo=UTC)  # 16:00 in Hong Kong
    window = parse_query("Will Man City win tonight?", now, "Asia/Hong_Kong").window
    # A 17:30 UK kickoff is 01:30 HKT: part of a Hong Kong "tonight".
    assert window.contains(datetime(2026, 11, 7, 17, 30, tzinfo=UTC))
    assert not window.contains(datetime(2026, 11, 8, 14, 0, tzinfo=UTC))
    london = parse_query("Will Man City win tonight?", now, "Europe/London").window
    assert london.start_utc != window.start_utc


# --------------------------------------------------------------------------- 2. "tonight" with no fixture

def test_tonight_with_no_scheduled_fixture(demo):
    response = demo.ask("Will Manchester City win tonight?", as_of=DEMO_MIDWEEK)
    assert response.status == "no_fixture"
    assert response.forecast is None
    assert "no confirmed fixture tonight" in response.message
    assert response.next_fixture.home == "Nottingham Forest" and response.next_fixture.away == "Manchester City"


# --------------------------------------------------------------------------- 3. conflicting reports

def _item(item_id, title, summary, source, credibility, published, team="Manchester City"):
    return NewsItem(item_id=item_id, title=title, summary=summary, url=f"https://example.invalid/{item_id}",
                    source=source, provider="test", published_at=published, retrieved_at=published, teams=[team],
                    credibility=credibility)


def test_news_extraction_with_conflicting_reports_and_syndication():
    squad = [SquadPlayer(player_id="fpl:1", name="Rodrigo Hernández", short_name="Rodri", team="Manchester City",
                         position=Position.MID, aliases=["Rodri", "Rodrigo Hernández"]),
             *make_squad("Manchester City")]
    squads = {"Manchester City": squad}
    as_of = datetime(2026, 11, 7, 12, tzinfo=UTC)
    items = [
        _item("a", "Rodri unlikely to feature against Arsenal", "Rodri is unlikely to feature against Arsenal after "
              "missing training.", "BBC Sport", 0.85, as_of - timedelta(hours=20)),
        _item("b", "Rodri set to miss Arsenal clash", "Rodri is set to miss the game, the paper understands.",
              "Daily Example", 0.4, as_of - timedelta(hours=6)),
        _item("c", "Rodri set to miss Arsenal clash", "Rodri is set to miss the game, the paper understands.",
              "Syndicator", 0.5, as_of - timedelta(hours=5)),
        _item("d", "Rodri fit to face Arsenal", "Rodri has recovered and is fit to face Arsenal, according to "
              "sources.", "Another Outlet", 0.65, as_of - timedelta(hours=3)),
    ]
    unique = deduplicate(items)
    assert len(unique) == 3  # b and c are one syndicated story.
    merged = next(i for i in unique if i.duplicates)
    assert len(merged.duplicates) == 1

    extracted = RuleExtractor().extract(unique, squads, "City v Arsenal")
    events = bind_events(unique, extracted, squads, "rules")
    first = next(e for e in events if e.url.endswith("/a"))
    # "Unlikely to feature" is an uncertain absence, not a confirmed one.
    assert first.event_type == EventType.POSSIBLE_ABSENCE
    assert first.availability == Availability.DOUBTFUL
    assert first.certainty == Certainty.REPORTED_EXPECTATION
    assert first.player_id == "fpl:1" and first.tactical_role == "midfield"

    belief = corroborate(events, squads, as_of)[0]
    assert belief.conflict
    assert 0.25 < belief.p_available < 0.9
    # Syndication counted once: duplicating a story further does not move the belief.
    more = deduplicate(items + [_item("e", "Rodri set to miss Arsenal clash", "Rodri is set to miss the game, the "
                                      "paper understands.", "Syndicator 2", 0.5, as_of - timedelta(hours=4))])
    events2 = bind_events(more, RuleExtractor().extract(more, squads, ""), squads, "rules")
    assert corroborate(events2, squads, as_of)[0].p_available == pytest.approx(belief.p_available)


def test_confirmed_statement_overrides_older_reports():
    squad = make_squad("Manchester City")
    target = squad[5]
    as_of = datetime(2026, 11, 7, 12, tzinfo=UTC)
    items = [_item("x", "doubt", f"{target.name} is a doubt with a knock.", "BBC Sport", 0.85, as_of - timedelta(hours=10)),
             _item("y", "update", f"The club confirmed {target.name} has passed a fitness test.", "Club", 0.95,
                   as_of - timedelta(hours=1))]
    squads = {"Manchester City": squad}
    events = bind_events(items, RuleExtractor().extract(items, squads, ""), squads, "rules")
    belief = corroborate(events, squads, as_of)[0]
    assert belief.status == Availability.AVAILABLE and belief.basis == Certainty.CONFIRMED_FACT


def test_llm_extractor_uses_schema_without_probabilities():
    assert not any("prob" in name for name in ExtractedEvent.model_fields)
    parsed = _BatchExtraction.model_validate({"articles": [{"article_index": 0, "events": [{
        "entity": "Rodri", "team": "Manchester City", "event_type": "possible_absence",
        "category": "player_availability", "certainty": "reported_expectation", "availability": "doubtful",
        "match_relevance": "high", "tactical_role": "defensive midfield",
        "evidence": "unlikely to feature after missing training"}]}]})
    calls = []

    class FakeResponses:
        def parse(self, **kwargs):
            calls.append(kwargs)
            return SimpleNamespace(output_parsed=parsed)

    extractor = LLMExtractor("unused", "test-model", client=SimpleNamespace(responses=FakeResponses()))
    item = _item("a", "Rodri doubt", "Rodri is unlikely to feature.", "BBC Sport", .85, datetime(2026, 11, 7, tzinfo=UTC))
    result = extractor.extract([item], {"Manchester City": []}, "fixture")
    assert result["a"][0].tactical_role == "defensive midfield"
    assert calls[0]["text_format"] is _BatchExtraction


# --------------------------------------------------------------------------- 4/5. lineups

def test_injured_player_excluded_from_lineup(selection):
    squad = make_squad()
    apps, regulars = make_history(squad)
    star = regulars[-1]
    kickoff = datetime(2026, 11, 7, 15, tzinfo=UTC)
    injured = [PlayerAvailability(player_id=star, name="star", team=squad[0].team, p_available=0.02,
                                  status=Availability.UNAVAILABLE, basis=Certainty.CONFIRMED_FACT)]
    model = build_team_lineup_model(squad[0].team, squad, apps, kickoff, selection, injured)
    samples = model.sample(2000, np.random.default_rng(1))
    lineup = describe(model, samples, "")
    assert samples[:, model.pool.index(star)].mean() < 0.05
    assert star not in {s.player_id for s in lineup.starters}
    assert star in {s.player_id for s in lineup.excluded}
    healthy = build_team_lineup_model(squad[0].team, squad, apps, kickoff, selection, [])
    assert healthy.sample(500, np.random.default_rng(1))[:, healthy.pool.index(star)].mean() > 0.6


def test_valid_eleven_player_lineups(selection):
    squad = make_squad()
    apps, _ = make_history(squad)
    model = build_team_lineup_model(squad[0].team, squad, apps, datetime(2026, 11, 7, tzinfo=UTC), selection, [])
    samples = model.sample(1000, np.random.default_rng(3))
    assert np.all(samples.sum(axis=1) == 11)
    assert np.all(samples[:, model.positions == "GK"].sum(axis=1) == 1)
    assert np.all(samples[:, model.positions == "DEF"].sum(axis=1) >= 3)
    lineup = describe(model, samples, "")
    assert len(lineup.starters) == 11 and lineup.kind == "predicted"
    with pytest.raises(ValueError):
        TeamLineup(team="x", kind="predicted", formation=None, starters=lineup.starters[:10], source="")
    with pytest.raises(LineupError):
        build_team_lineup_model("x", squad[:9], [], datetime(2026, 11, 7, tzinfo=UTC), selection, [])


def test_normalize_to_eleven_and_discontinuity():
    p = normalize_to_eleven(np.array([.9, .2] + [.5] * 14), np.array([True, True] + [False] * 14))
    assert p.sum() == pytest.approx(11) and p[:2].sum() == pytest.approx(1) and p.max() <= 1
    typical = np.array([1.0] * 11 + [0.0] * 5)
    assert discontinuity(typical, typical) == 0
    assert discontinuity(typical, np.r_[np.zeros(1), np.ones(10), np.ones(1), np.zeros(4)]) == pytest.approx(1 / 11)


# --------------------------------------------------------------------------- 6. official XI replaces prediction

@needs_models
def test_official_xi_replaces_predicted_xi(demo):
    before = demo.ask("Will Man City win tonight?", as_of=DEMO_PRE)
    after = demo.ask("Will Man City win tonight?", as_of=DEMO_XI)
    assert {l.kind for l in before.lineups.values()} == {"predicted"}
    assert {l.kind for l in after.lineups.values()} == {"confirmed"}
    city = after.lineups["Manchester City"]
    # "R. Dias" / "G. Donnarumma" style names were resolved to squad identities.
    assert {"Gianluigi Donnarumma", "Erling Haaland"} <= {s.name for s in city.starters}
    assert all(s.p_start == 1 and not s.uncertain for s in city.starters)
    assert after.forecast.adjusted.probabilities != before.forecast.adjusted.probabilities


def test_player_identity_resolution():
    squad = [SquadPlayer(player_id="1", name="Rúben dos Santos Gato Alves Dias", short_name="Rúben", team="T",
                         position=Position.DEF, aliases=["Rúben", "dos Santos Gato Alves Dias"]),
             SquadPlayer(player_id="2", name="Erling Haaland", short_name="Haaland", team="T", position=Position.FWD),
             SquadPlayer(player_id="3", name="Gabriel Jesus", short_name="G.Jesus", team="T", position=Position.FWD),
             SquadPlayer(player_id="4", name="Gabriel Magalhães", short_name="Gabriel", team="T", position=Position.DEF)]
    assert resolve_player("R. Dias", squad).player.player_id == "1"
    assert resolve_player("Ruben Dias", squad).player.player_id == "1"
    assert resolve_player("E. HAALAND", squad).player.player_id == "2"
    assert resolve_player("Kevin De Bruyne", squad).player is None
    ambiguous = resolve_player("Gabriel", [squad[2], squad[3].model_copy(update={"short_name": "Magalhães"})])
    assert ambiguous.player is None and ambiguous.reason == "ambiguous"


# --------------------------------------------------------------------------- 7/8. forecasts

@needs_models
def test_forecast_probabilities_sum_to_one(demo):
    response = demo.ask("Will Man City win tonight?", as_of=DEMO_PRE)
    for block in (response.forecast.baseline, response.forecast.adjusted):
        p = block.probabilities
        assert p.home_win + p.draw + p.away_win == pytest.approx(1, abs=1e-9)
        assert sum(s.probability for s in block.top_scorelines) < 1
    with pytest.raises(ValueError):
        OutcomeProbabilities(home_win=.5, draw=.3, away_win=.3)
    # Every Monte Carlo draw is returned and averages to the headline forecast.
    sims = response.forecast.simulations
    assert sims["n"] == 300 and len(sims["score_home"]) == 300
    assert np.mean(sims["away_win"]) == pytest.approx(response.forecast.adjusted.probabilities.away_win, abs=1e-3)
    sp = response.subject_probabilities
    assert sp["team"] == "Manchester City" and sp["win"] == response.forecast.adjusted.probabilities.away_win


@needs_models
def test_news_adjusted_prediction_versus_baseline(demo):
    response = demo.ask("How does Haaland being unavailable affect City's chances?", as_of=DEMO_PRE)
    fc = response.forecast
    haaland = next(a for a in response.availability if a.name == "Erling Haaland")
    assert haaland.status == Availability.DOUBTFUL  # Doubt + tabloid "set to miss" (syndicated once).
    assert fc.adjusted.probabilities.away_win < fc.baseline.probabilities.away_win
    assert fc.adjusted.expected_goals_away < fc.baseline.expected_goals_away
    top = fc.contributions[0]
    assert "Haaland" in top.factor and top.delta_xg_for_team < 0 and top.evidence_ids
    assert response.what_if["subject_win_change"] < 0
    assert "PROVISIONAL" in fc.adjustment_status
    # The baseline equals the existing ScoreModel output, untouched by news.
    from src.matchsense.baseline import BaselineEngine
    from src.ml.season_models import outcomes
    engine = BaselineEngine(BUNDLE)
    fixtures = demo.fpl.all_fixtures()
    base = engine.predict(response.fixture, fixtures, DEMO_PRE)
    assert outcomes(base.grid[None])[0][2] == pytest.approx(fc.baseline.probabilities.away_win)


# --------------------------------------------------------------------------- 9. missing credentials

def test_missing_api_credentials_are_reported_not_raised(tmp_path):
    http = HttpClient(HttpCache(tmp_path / "c.sqlite"), "test",
                      transport=httpx.MockTransport(lambda r: httpx.Response(500)))
    with pytest.raises(MissingCredentials):
        GuardianNews(http, None).news("Arsenal", datetime(2026, 11, 1, tzinfo=UTC), datetime(2026, 11, 2, tzinfo=UTC))
    log = StatusLog()
    assert log.guard("Guardian", GuardianNews(http, None).news, "Arsenal", None, None, default=[]) == []
    assert log.entries[0].status == "missing" and "GUARDIAN_API_KEY" in log.entries[0].detail


@needs_models
def test_live_mode_without_any_keys_degrades_gracefully(tmp_path):
    from src.matchsense.pipeline import MatchSense
    from src.matchsense.providers.demo import recorded_transport
    settings = load_settings(mode="live", simulations=200, data_dir=tmp_path, use_bbc=False,
                             use_openfootball_remote=False, api_football_key=None, football_data_key=None,
                             guardian_key=None, newsapi_key=None, gnews_key=None, openai_key=None,
                             query_interpreter="rules")
    ms = MatchSense(settings, transport=recorded_transport())
    response = ms.ask("Will Man City win tonight?", as_of=DEMO_PRE)
    assert response.status == "forecast"
    statuses = {d.name: d.status for d in response.data_quality}
    assert statuses["News: guardian"] == "missing" and statuses["News: newsapi"] == "missing"
    assert any("API_FOOTBALL_KEY" in n for n in response.notes)
    assert ms.extractor.name == "rules"


def test_http_retries_then_serves_stale_cache(tmp_path):
    calls = {"n": 0}

    def handler(request):
        calls["n"] += 1
        return httpx.Response(200, json={"ok": calls["n"]}) if calls["n"] == 1 else httpx.Response(503)

    clock = {"now": datetime(2026, 11, 1, tzinfo=UTC)}
    http = HttpClient(HttpCache(tmp_path / "c.sqlite"), "t", transport=httpx.MockTransport(handler),
                      min_interval={"default": 0}, clock=lambda: clock["now"])
    assert http.get("https://x.test/a", ttl_seconds=60).json() == {"ok": 1}
    assert http.get("https://x.test/a", ttl_seconds=60).from_cache  # Fresh cache hit.
    clock["now"] += timedelta(hours=1)
    stale = http.get("https://x.test/a", ttl_seconds=60, retries=2)
    assert stale.stale and stale.json() == {"ok": 1}
    with pytest.raises(ProviderError):
        http.get("https://x.test/other", retries=1)


# --------------------------------------------------------------------------- 10. leakage

def test_lineup_features_do_not_see_future_matches():
    from src.matchsense.lineup_data import load_openfootball_lineups
    from src.matchsense.training import build_observations
    path = PROJECT_ROOT / "england/2025-26/1-premierleague-full.txt"
    if not path.exists():
        pytest.skip("openfootball checkout not present")
    team_matches = load_openfootball_lineups(path)
    cutoff = sorted({m.kickoff for m in team_matches})[150]
    original = build_observations(team_matches)
    mutated = copy.deepcopy(team_matches)
    for m in mutated:
        if m.kickoff >= cutoff:  # Rewrite every future lineup, goal and result.
            m.starters, m.minutes, m.scorers = m.starters[::-1][:5], {}, {"ghost": 9}
            m.goals_for, m.goals_against = 9, 0
    changed = build_observations(mutated)
    for a, b in zip(original, changed):
        if a.kickoff <= cutoff:
            assert a.pool == b.pool
            assert np.array_equal(a.typical, b.typical) and np.array_equal(a.goal_share, b.goal_share)
            assert np.array_equal(a.selection_x, b.selection_x)


@needs_models
def test_baseline_ignores_results_after_prediction_time():
    from src.matchsense.baseline import BaselineEngine
    engine = BaselineEngine(BUNDLE)

    def fx(home, away, kickoff, hg=None, ag=None, md=1):
        return Fixture(fixture_id=f"PL:2026-27:{home}:{away}", competition=PREMIER_LEAGUE, season="2026-27",
                       matchday=md, home=home, away=away, kickoff_utc=kickoff,
                       status=FixtureStatus.FINISHED if hg is not None else FixtureStatus.SCHEDULED,
                       home_goals=hg, away_goals=ag, source="test")
    t0 = datetime(2026, 8, 22, 14, tzinfo=UTC)
    past = [fx("Arsenal", "Chelsea", t0, 2, 0), fx("Liverpool", "Everton", t0, 1, 1)]
    target = fx("Arsenal", "Liverpool", t0 + timedelta(days=14), md=3)
    as_of = t0 + timedelta(days=13)
    future_a = fx("Chelsea", "Arsenal", t0 + timedelta(days=13, hours=2), 0, 5, md=2)  # Kicks off after as_of.
    future_b = future_a.model_copy(update={"home_goals": 5, "away_goals": 0})
    a = engine.predict(target, past + [future_a], as_of)
    b = engine.predict(target, past + [future_b], as_of)
    assert a.results_used == 2 and np.allclose(a.probabilities, b.probabilities)
    c = engine.predict(target, past + [future_a], as_of + timedelta(hours=6))
    assert c.results_used == 3 and not np.allclose(a.probabilities, c.probabilities)


@needs_models
def test_news_published_after_prediction_time_is_excluded(demo):
    before = demo.ask("Will Man City win tonight?", as_of=DEMO_PRE)
    assert all(e.published_at <= DEMO_PRE for e in before.news)
    haaland = next(a for a in before.availability if a.name == "Erling Haaland")
    assert haaland.status != Availability.AVAILABLE  # The 12:00Z fitness confirmation is not yet known.
    later = demo.ask("Will Man City win tonight?", as_of=DEMO_PRE + timedelta(hours=6))
    assert next(a for a in later.availability if a.name == "Erling Haaland").status == Availability.AVAILABLE


# --------------------------------------------------------------------------- status handling and storage

@needs_models
def test_match_in_progress_and_history(demo):
    live = demo.ask("Will Arsenal win tonight?", as_of=DEMO_XI)
    assert live.status == "live" and live.forecast is None
    history = demo.store.history()
    assert history["predictions"] and all(p["mode"] == "demo" for p in history["predictions"])


def test_unsupported_competition_is_not_forecast(tmp_path):
    from src.matchsense.pipeline import MatchSense
    from src.matchsense.schemas import Competition
    ms = MatchSense(load_settings(mode="live", data_dir=tmp_path),
                    transport=httpx.MockTransport(lambda r: httpx.Response(404)))
    ucl = Fixture(fixture_id="UCL:1", competition=Competition(code="UCL", name="UEFA Champions League"),
                  season="2026-27", home="Manchester City", away="Real Madrid",
                  kickoff_utc=datetime(2026, 11, 25, 20, tzinfo=UTC), status=FixtureStatus.SCHEDULED, source="test")
    parsed = parse_query("Will Man City win?", datetime(2026, 11, 25, 9, tzinfo=UTC))
    response = ms._fixture_response(parsed, ucl, datetime(2026, 11, 25, 9, tzinfo=UTC), "UTC", StatusLog(), None)
    assert response.status == "unsupported_competition" and response.forecast is None


def test_impact_artifact_is_labelled_honestly():
    if not IMPACT.exists():
        pytest.skip("impact artifact not trained")
    data = json.loads(IMPACT.read_text())
    assert data["status"] in ("validated", "provisional_prior")
    if data["status"] == "provisional_prior":
        assert data["impact"]["prior"] is not None and "disclosure" in data["report"]


def test_llm_extractions_are_cached(tmp_path):
    from src.matchsense.intelligence import ExtractionCache
    parsed = _BatchExtraction.model_validate({"articles": [{"article_index": 0, "events": []}]})
    calls = []

    class FakeResponses:
        def parse(self, **kwargs):
            calls.append(kwargs)
            return SimpleNamespace(output_parsed=parsed)

    cache = ExtractionCache(tmp_path / "llm.sqlite")
    extractor = LLMExtractor("unused", "m", client=SimpleNamespace(responses=FakeResponses()), cache=cache)
    item = _item("a", "Rodri doubt", "Rodri is unlikely to feature.", "BBC Sport", .85, datetime(2026, 11, 7, tzinfo=UTC))
    extractor.extract([item], {}, "f")
    extractor.extract([item], {}, "f")
    assert len(calls) == 1 and extractor.last_stats == {"cached": 1, "sent": 0}
    changed = item.model_copy(update={"summary": "Rodri has been ruled out."})
    extractor.extract([changed], {}, "f")
    assert len(calls) == 2  # Edited story text is re-interpreted.


@pytest.mark.parametrize("question", ["what are the odds if khusanov doesnt play", "khusanov out, will city win?",
                                      "City's chances without Khusanov", "if khusanov is injured will city win"])
def test_what_if_phrasings_find_player_and_club(demo, question):
    response = demo.ask(question, as_of=DEMO_PRE)
    assert response.status == "forecast" and response.subject_team == "Manchester City"
    assert response.what_if["players"] == ["Abdukodir Khusanov"]
    assert response.forecast.player_impacts
    assert min(r["delta_win"] for r in response.forecast.player_impacts) < 0


def test_codex_interpreter_understands_then_validates(tmp_path):
    from src.matchsense.interpreter import CodexInterpreter, apply_interpretation

    def fake_codex(cmd, **kwargs):
        assert kwargs["stdin"] is not None  # stdin closed, or codex blocks.
        out = Path(cmd[cmd.index("-o") + 1])
        out.write_text(json.dumps({"intent": "what_if", "subject_team": "Man City", "opponent_team": "Atlantis FC",
                                   "date_phrase": "tonight", "players_out": ["Khusanov"], "players_in": []}))
        return SimpleNamespace(returncode=0, stderr="")

    interp = CodexInterpreter(runner=fake_codex).interpret("what are the odds if khusanov doesnt play tonight")
    now = datetime(2026, 11, 4, 12, tzinfo=UTC)
    parsed = parse_query("what are the odds if khusanov doesnt play tonight", now)
    apply_interpretation(parsed, interp, now, "Asia/Hong_Kong")
    assert parsed.subject == "Manchester City"  # Alias mapped to the canonical club.
    assert parsed.opponent is None  # An unknown club is rejected, not invented.
    assert parsed.intent == "what_if" and parsed.what_if_players == ["Khusanov"]
    assert parsed.window.label == "tonight"  # Date words re-resolved deterministically.


def test_interpreter_failure_falls_back_and_cache_saves_calls(tmp_path):
    from src.matchsense.interpreter import QueryInterpretation
    from src.matchsense.pipeline import MatchSense
    ms = MatchSense(load_settings(mode="live", data_dir=tmp_path, query_interpreter="rules"),
                    transport=httpx.MockTransport(lambda r: httpx.Response(404)))
    calls = []

    class Boom:
        name = "codex-cli"

        def interpret(self, q):
            calls.append(q)
            raise RuntimeError("timeout")

    ms.interpreter = Boom()
    log = StatusLog()
    now = datetime(2026, 11, 4, 12, tzinfo=UTC)
    info = ms.understand(parse_query("will city win tonight", now), now, "UTC", log)
    assert info["backend"] == "rules (fallback)" and log.entries[0].status == "error"

    class Good:
        name = "codex-cli"

        def interpret(self, q):
            calls.append(q)
            return QueryInterpretation(intent="outcome", subject_team="Arsenal", opponent_team=None,
                                       date_phrase=None, players_out=[], players_in=[])

    ms.interpreter = Good()
    for _ in range(3):
        info = ms.understand(parse_query("will arsenal win", now), now, "UTC", StatusLog())
    assert len(calls) == 2 and info["cached"]  # One real call, then cache hits.


def test_defensive_discontinuity_and_like_for_like_defenders(selection):
    from src.matchsense.impact import defensive_discontinuity
    typical = np.array([1.0, 1, 1, 1, 1, 1, 0.2])
    mask = np.array([True, True, True, True, True, False, True])
    assert defensive_discontinuity(typical, np.array([1.0, 1, 1, 1, 1, 1, 0]), mask) == 0
    # Losing one of five usual defenders (replaced by a fringe one) leaves ~16% of the unit's mass missing.
    assert defensive_discontinuity(typical, np.array([1.0, 0, 1, 1, 1, 1, 1]), mask) == pytest.approx(0.16)
    squad = make_squad()
    apps, regulars = make_history(squad)
    centre_back = next(pid for pid in regulars if pid in {p.player_id for p in squad if p.position == Position.DEF})
    out = [PlayerAvailability(player_id=centre_back, name="cb", team=squad[0].team, p_available=0.0,
                              status=Availability.UNAVAILABLE, basis=Certainty.CONFIRMED_FACT)]
    model = build_team_lineup_model(squad[0].team, squad, apps, datetime(2026, 11, 7, tzinfo=UTC), selection, out)
    samples = model.sample(500, np.random.default_rng(5))
    assert np.all(samples[:, model.positions == "DEF"].sum(axis=1) == model.shape[0])  # Like-for-like.


def test_old_four_feature_artifacts_still_load():
    from src.matchsense.impact import K, LineupImpactModel
    legacy = {"use_continuity": True, "use_goal_share": True, "use_players": False, "alpha_team": 0.0,
              "alpha_players": 0.0, "b": [0.1, 0.2, 0.3, 0.4], "b_cov": np.eye(4).tolist(), "attack": {},
              "defence": {}, "attack_var": {}, "defence_var": {}, "center": [0, 0, 0, 0]}
    model = LineupImpactModel.from_dict(legacy)
    assert len(model.b) == K and model.b[-1] == 0 and model.b_cov.shape == (K, K) and not model.use_defence
