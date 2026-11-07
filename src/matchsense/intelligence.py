"""News intelligence: dedup, structured extraction, and corroboration.

Pipeline: NewsItem -> syndication clusters -> ExtractedEvent (LLM or rules)
-> IntelligenceEvent (bound to a source and a resolved squad player) ->
PlayerAvailability (corroborated belief per player).

The extractor output schema has no probability fields. The only numbers
produced here are availability beliefs computed by the fixed, documented
mapping in ``AVAILABILITY_VALUE`` and source weights; they become inputs to
the lineup model, never a win probability.
"""
from __future__ import annotations

import hashlib
import json
import logging
import math
import re
import sqlite3
import threading
from collections import defaultdict
from datetime import datetime
from pathlib import Path

from pydantic import BaseModel, Field

from src.matchsense.players import find_player_mentions, name_key, resolve_player
from src.matchsense.providers.base import stable_id
from src.matchsense.providers.news import canonical_url
from src.matchsense.schemas import (Availability, Category, Certainty, EventType, ExtractedEvent, IntelligenceEvent,
                                    NewsItem, PlayerAvailability, Relevance, SquadPlayer)

log = logging.getLogger(__name__)

# ----------------------------------------------------------------------------- dedup


def _shingles(text: str, n: int = 3) -> set[str]:
    words = [w for w in name_key(text).split() if w not in {"the", "a", "an", "to", "of", "in", "for", "and", "on", "as"}]
    return {" ".join(words[i:i + n]) for i in range(max(len(words) - n + 1, 1))}


def _similar(a: NewsItem, b: NewsItem) -> bool:
    if canonical_url(a.url) == canonical_url(b.url):
        return True
    if abs((a.published_at - b.published_at).total_seconds()) > 48 * 3600:
        return False
    sa, sb = _shingles(a.title + " " + a.summary[:160]), _shingles(b.title + " " + b.summary[:160])
    jaccard = len(sa & sb) / max(len(sa | sb), 1)
    ta, tb = set(name_key(a.title).split()), set(name_key(b.title).split())
    title_overlap = len(ta & tb) / max(min(len(ta), len(tb)), 1)
    return jaccard >= 0.5 or title_overlap >= 0.85


def deduplicate(items: list[NewsItem]) -> list[NewsItem]:
    """Collapse syndicated copies; keep the most credible, earliest copy."""
    clusters: list[list[NewsItem]] = []
    for item in sorted(items, key=lambda i: i.published_at):
        for cluster in clusters:
            if any(_similar(item, other) for other in cluster):
                cluster.append(item)
                break
        else:
            clusters.append([item])
    result = []
    for cluster in clusters:
        lead = max(cluster, key=lambda i: (i.credibility, -i.published_at.timestamp()))
        merged_teams = sorted({t for i in cluster for t in i.teams})
        result.append(lead.model_copy(update={
            "cluster_id": stable_id(*sorted(i.item_id for i in cluster)), "teams": merged_teams,
            "duplicates": [i.url for i in cluster if i is not lead]}))
    return result

# ----------------------------------------------------------------------------- extraction


class _ArticleEvents(BaseModel):
    article_index: int
    events: list[ExtractedEvent] = Field(default_factory=list)


class _BatchExtraction(BaseModel):
    articles: list[_ArticleEvents] = Field(default_factory=list)


SYSTEM_PROMPT = """You are a football intelligence extraction system. Convert news text into
structured, evidence-backed events about the two clubs in an upcoming match.
Rules:
- Only report events explicitly supported by the text. Quote a short supporting phrase in `evidence`.
- certainty: confirmed_fact = officially confirmed or stated by the club/manager as fact;
  reported_expectation = journalist reporting an expected outcome ("set to", "expected", "understood");
  speculation = possibility or rumour ("could", "may", "doubt over"); ai_inference = your own reading
  that the text implies but does not state.
- availability: unavailable only when the text says the player will miss the match or is ruled out;
  doubtful for fitness doubts, missed training, late tests; available for returns or declared fitness.
  "Unlikely to feature" is doubtful with reported_expectation, NOT unavailable.
- Never output probabilities, odds, predictions of the result or scorelines.
- match_relevance: high if the event concerns the specified fixture or the next few days.
- tactical_role: the role affected (e.g. "defensive midfield", "centre-back", "striker") if inferable.
- Use the squad lists to spell player names; skip events about other clubs.
Return JSON matching the schema. Use an empty events list when an article has no relevant events."""


class ExtractionCache:
    """SQLite store of LLM extractions keyed by model, prompt and article text.

    An article already interpreted is never sent to the API again, so repeat
    questions and refreshes cost no tokens unless the story itself changes.
    """

    def __init__(self, path: Path):
        path.parent.mkdir(parents=True, exist_ok=True)
        self.path = path
        self._lock = threading.Lock()
        with sqlite3.connect(path) as conn:
            conn.execute("create table if not exists extractions (key text primary key, model text, "
                         "events text, created_at text)")

    @staticmethod
    def key(model: str, item: NewsItem) -> str:
        prompt = hashlib.sha256(SYSTEM_PROMPT.encode()).hexdigest()[:12]
        text = "\x1f".join((model, prompt, item.source, item.title, item.summary[:1200]))
        return hashlib.sha256(text.encode()).hexdigest()

    def get(self, key: str) -> list[ExtractedEvent] | None:
        with self._lock, sqlite3.connect(self.path) as conn:
            row = conn.execute("select events from extractions where key = ?", (key,)).fetchone()
        return [ExtractedEvent.model_validate(e) for e in json.loads(row[0])] if row else None

    def put(self, key: str, model: str, events: list[ExtractedEvent]):
        with self._lock, sqlite3.connect(self.path) as conn:
            conn.execute("insert or replace into extractions values (?, ?, ?, datetime('now'))",
                         (key, model, json.dumps([e.model_dump(mode="json") for e in events])))


class LLMExtractor:
    """OpenAI structured-output extractor (Responses API + Pydantic schema)."""

    def __init__(self, api_key: str, model: str, client=None, cache: ExtractionCache | None = None):
        self.model = model
        if client is None:
            from openai import OpenAI
            client = OpenAI(api_key=api_key, timeout=60, max_retries=2)
        self.client = client
        self.cache = cache
        self.name = f"openai:{model}"
        self.last_stats = {"cached": 0, "sent": 0}

    def extract(self, items: list[NewsItem], squads: dict[str, list[SquadPlayer]],
                fixture_label: str) -> dict[str, list[ExtractedEvent]]:
        result: dict[str, list[ExtractedEvent]] = {}
        pending = []
        for item in items:
            hit = self.cache.get(ExtractionCache.key(self.model, item)) if self.cache else None
            if hit is None:
                pending.append(item)
            else:
                result[item.item_id] = hit
        self.last_stats = {"cached": len(items) - len(pending), "sent": len(pending)}
        squad_text = {team: [p.name for p in players][:40] for team, players in squads.items()}
        for start in range(0, len(pending), 8):
            batch = pending[start:start + 8]
            articles = [{"article_index": i, "publisher": it.source, "published_at": it.published_at.isoformat(),
                         "title": it.title, "summary": it.summary[:1200]} for i, it in enumerate(batch)]
            payload = json.dumps({"fixture": fixture_label, "squads": squad_text, "articles": articles})
            response = self.client.responses.parse(
                model=self.model,
                input=[{"role": "system", "content": SYSTEM_PROMPT}, {"role": "user", "content": payload}],
                text_format=_BatchExtraction)
            parsed: _BatchExtraction = response.output_parsed
            found = {a.article_index: a.events for a in parsed.articles if 0 <= a.article_index < len(batch)}
            for i, item in enumerate(batch):
                events = found.get(i, [])
                result[item.item_id] = events
                if self.cache:  # Cache empty results too: "nothing relevant" is also an answer.
                    self.cache.put(ExtractionCache.key(self.model, item), self.model, events)
        return result


# Rule-based fallback. Ordered: the first matching pattern wins per sentence.
_RULES: tuple[tuple[str, EventType, Availability], ...] = (
    (r"\bnot (?:been )?ruled out\b", EventType.FITNESS_DOUBT, Availability.DOUBTFUL),
    (r"\b(?:suspended|suspension|banned|ban)\b", EventType.SUSPENSION, Availability.UNAVAILABLE),
    (r"\b(?:back in (?:full )?training|returned to training|return(?:s|ed)? from injury|available again|"
     r"(?:is|are) available|fit to (?:play|face|start)|recovered|in contention|passed (?:a|his|a late) fitness test|"
     r"back in the squad)\b", EventType.RETURN_FROM_INJURY, Availability.AVAILABLE),
    (r"\b(?:ruled out|will miss|to miss|set to miss|out for|sidelined|out injured|misses out|unavailable|"
     r"won'?t play|will not play|absent)\b", EventType.INJURY, Availability.UNAVAILABLE),
    (r"\b(?:unlikely to (?:feature|play|start)|doubt(?:ful)?|fitness test|race against time|missed training|"
     r"miss(?:ed|ing)? (?:training|session|part of)|late test|50-50|touch and go|being assessed|knock|"
     r"minor problem|tightness)\b", EventType.FITNESS_DOUBT, Availability.DOUBTFUL),
    (r"\b(?:rotate|rotation|rest(?:ed|ing)? (?:some|key|several)|fresh legs|make changes|changes to (?:his|the) side)\b",
     EventType.ROTATION_EXPECTED, Availability.UNKNOWN),
    (r"\b(?:back three|back four|three at the back|switch to a|formation)\b", EventType.FORMATION_CHANGE,
     Availability.UNKNOWN),
)
_CERTAINTY_RANK = {Certainty.CONFIRMED_FACT: 3, Certainty.REPORTED_EXPECTATION: 2, Certainty.SPECULATION: 1,
                   Certainty.AI_INFERENCE: 0}
_CONFIRMED = r"\b(?:confirmed|confirms|has been ruled out|announced|club statement|official)\b"
_REPORTED = r"\b(?:expected|set to|understood|reportedly|according to|is likely|are likely|sources|told)\b"
_SPECULATIVE = r"\b(?:could|may|might|possible|possibly|rumou?r|fears?|concern|hope)\b"
_MANAGER = r"\b(?:said|says|told reporters|press conference|insisted|revealed|admitted|confirmed)\b"
_INJURY_WORDS = r"\b(?:hamstring|knee|ankle|groin|calf|thigh|muscle|foot|back|shoulder|concussion|illness|virus|injur\w*)\b"


class RuleExtractor:
    name = "rules"

    def extract(self, items: list[NewsItem], squads: dict[str, list[SquadPlayer]],
                fixture_label: str) -> dict[str, list[ExtractedEvent]]:
        result = {}
        everyone = [(team, p) for team, players in squads.items() for p in players]
        for item in items:
            text = f"{item.title}. {item.summary}"
            events = []
            for sentence in re.split(r"(?<=[.!?])\s+|\s+-\s+", text):
                lower = sentence.lower()
                rule = next(((kind, avail) for pattern, kind, avail in _RULES if re.search(pattern, lower)), None)
                if rule is None:
                    continue
                kind, avail = rule
                mentions = find_player_mentions(sentence, [p for _, p in everyone])
                if not mentions and kind in (EventType.ROTATION_EXPECTED, EventType.FORMATION_CHANGE):
                    mentions = []
                    team = next((t for t in item.teams if t in squads), None)
                    if team:
                        events.append(ExtractedEvent(entity=team, team=team, event_type=kind, category=Category.TACTICS
                                                     if kind == EventType.FORMATION_CHANGE else Category.LINEUP,
                                                     certainty=Certainty.SPECULATION, availability=avail,
                                                     match_relevance=Relevance.MEDIUM, evidence=sentence[:160]))
                    continue
                for player, alias in mentions:
                    team = player.team
                    if kind == EventType.INJURY and re.search(r"\billness|virus|sick\b", lower):
                        kind = EventType.ILLNESS
                    if re.search(_CONFIRMED, lower) or (re.search(_MANAGER, lower) and avail == Availability.UNAVAILABLE
                                                        and not re.search(_SPECULATIVE, lower)):
                        certainty = Certainty.CONFIRMED_FACT
                    elif re.search(_SPECULATIVE, lower):
                        certainty = Certainty.SPECULATION
                    elif re.search(_REPORTED, lower) or avail != Availability.UNKNOWN:
                        certainty = Certainty.REPORTED_EXPECTATION
                    else:
                        certainty = Certainty.AI_INFERENCE
                    if kind == EventType.FITNESS_DOUBT and re.search(r"unlikely to (feature|play|start)", lower):
                        kind = EventType.POSSIBLE_ABSENCE
                    injury = re.search(_INJURY_WORDS, lower)
                    events.append(ExtractedEvent(
                        entity=player.name, team=team, event_type=kind, category=Category.PLAYER_AVAILABILITY,
                        certainty=certainty, availability=avail,
                        match_relevance=Relevance.HIGH if (item.published_at and kind != EventType.ROTATION_EXPECTED)
                        else Relevance.MEDIUM,
                        tactical_role={"GK": "goalkeeper", "DEF": "defence", "MID": "midfield", "FWD": "attack"}[
                            player.position.value],
                        evidence=sentence[:200],
                        expected_return=injury.group(0) if injury else None))
            # One event per entity per story: keep the most certain statement.
            best: dict[str, ExtractedEvent] = {}
            for event in events:
                current = best.get(event.entity)
                if current is None or _CERTAINTY_RANK[event.certainty] > _CERTAINTY_RANK[current.certainty]:
                    best[event.entity] = event
            result[item.item_id] = list(best.values())
        return result


def bind_events(items: list[NewsItem], extracted: dict[str, list[ExtractedEvent]],
                squads: dict[str, list[SquadPlayer]], extractor_name: str) -> list[IntelligenceEvent]:
    by_id = {i.item_id: i for i in items}
    events = []
    for item_id, found in extracted.items():
        item = by_id.get(item_id)
        if item is None:
            continue
        for index, event in enumerate(found):
            team = event.team if event.team in squads else None
            player_id, score = None, None
            if event.category == Category.PLAYER_AVAILABILITY or event.event_type in (
                    EventType.CONFIRMED_START, EventType.CONFIRMED_BENCH):
                pools = [squads[team]] if team else list(squads.values())
                for pool in pools:
                    resolution = resolve_player(event.entity, pool)
                    if resolution.player:
                        player_id, score, team = resolution.player.player_id, resolution.score, resolution.player.team
                        break
            events.append(IntelligenceEvent(
                **event.model_dump(exclude={"team"}), team=team,
                event_id=stable_id(item.item_id, str(index), event.entity), player_id=player_id,
                player_match_score=score, source=item.source, provider=item.provider, url=item.url,
                title=item.title, published_at=item.published_at, retrieved_at=item.retrieved_at,
                credibility=item.credibility, extractor=extractor_name, is_simulated=item.is_simulated,
                cluster_id=item.cluster_id))
    return events

# ----------------------------------------------------------------------------- corroboration

# Fixed, documented mapping from (availability, certainty) to an availability
# belief. These are evidence-to-feature conventions, not fitted parameters.
AVAILABILITY_VALUE = {
    Availability.UNAVAILABLE: {Certainty.CONFIRMED_FACT: 0.02, Certainty.REPORTED_EXPECTATION: 0.12,
                               Certainty.SPECULATION: 0.4, Certainty.AI_INFERENCE: 0.45},
    Availability.DOUBTFUL: {Certainty.CONFIRMED_FACT: 0.5, Certainty.REPORTED_EXPECTATION: 0.45,
                            Certainty.SPECULATION: 0.6, Certainty.AI_INFERENCE: 0.6},
    Availability.AVAILABLE: {Certainty.CONFIRMED_FACT: 0.98, Certainty.REPORTED_EXPECTATION: 0.9,
                             Certainty.SPECULATION: 0.8, Certainty.AI_INFERENCE: 0.8},
}
CERTAINTY_WEIGHT = {Certainty.CONFIRMED_FACT: 1.0, Certainty.REPORTED_EXPECTATION: 0.7,
                    Certainty.SPECULATION: 0.35, Certainty.AI_INFERENCE: 0.25}
FRESHNESS_HALF_LIFE_HOURS = 36.0


def _value(event: IntelligenceEvent, chance: int | None) -> float | None:
    if event.availability not in AVAILABILITY_VALUE:
        return None
    if event.extractor == "fpl_status" and chance is not None:
        return min(max(chance / 100, 0.02), 0.98)  # League's own published percentage.
    return AVAILABILITY_VALUE[event.availability][event.certainty]


def corroborate(events: list[IntelligenceEvent], squads: dict[str, list[SquadPlayer]],
                as_of: datetime) -> list[PlayerAvailability]:
    players = {p.player_id: p for squad in squads.values() for p in squad}
    grouped: dict[str, list[IntelligenceEvent]] = defaultdict(list)
    for event in events:
        if event.player_id in players and event.published_at <= as_of and \
                event.category == Category.PLAYER_AVAILABILITY:
            grouped[event.player_id].append(event)
    beliefs = []
    for player_id, items in grouped.items():
        player = players[player_id]
        # One vote per syndication cluster: keep its highest-weight event.
        per_cluster: dict[str, tuple[float, float, IntelligenceEvent]] = {}
        for event in items:
            value = _value(event, player.chance_of_playing)
            if value is None:
                continue
            age_h = max((as_of - event.published_at).total_seconds() / 3600, 0)
            freshness = 1.0 if event.extractor == "fpl_status" else math.exp2(-age_h / FRESHNESS_HALF_LIFE_HOURS)
            weight = event.credibility * freshness * CERTAINTY_WEIGHT[event.certainty]
            key = event.cluster_id or event.event_id
            if key not in per_cluster or weight > per_cluster[key][0]:
                per_cluster[key] = (weight, value, event)
        if not per_cluster:
            continue
        votes = sorted(per_cluster.values(), key=lambda v: v[2].published_at)
        confirmed = [v for v in votes if v[2].certainty == Certainty.CONFIRMED_FACT]
        conflict = False
        note = ""
        if confirmed:
            # The most recent confirmed statement dominates older reports.
            weight, p, lead = confirmed[-1]
            dissent = [v for v in votes if v[2].published_at > lead.published_at and abs(v[1] - p) > 0.4 and v[0] >= 0.2]
            if dissent:
                conflict = True
                p = (weight * p + sum(v[0] * v[1] for v in dissent)) / (weight + sum(v[0] for v in dissent))
                note = "Later reports contradict an earlier confirmed statement."
            else:
                note = "Confirmed statement takes precedence over reports."
        else:
            total = sum(v[0] for v in votes)
            p = sum(v[0] * v[1] for v in votes) / total if total > 0 else 1.0
            strong = [v for v in votes if v[0] >= 0.15]
            if strong and max(v[1] for v in strong) - min(v[1] for v in strong) > 0.4:
                conflict = True
                note = "Sources disagree; belief is a credibility- and freshness-weighted average."
            lead = max(votes, key=lambda v: v[0])[2]
        status = (Availability.UNAVAILABLE if p < 0.25 else Availability.DOUBTFUL if p < 0.75
                  else Availability.AVAILABLE)
        beliefs.append(PlayerAvailability(player_id=player_id, name=player.name, team=player.team,
                                          p_available=round(float(p), 4), status=status, basis=lead.certainty,
                                          conflict=conflict, note=note,
                                          evidence_ids=[v[2].event_id for v in votes]))
    return sorted(beliefs, key=lambda b: (b.team, b.p_available))
