"""News providers. Every item carries publication and retrieval timestamps.

Only documented feeds/APIs are used: BBC Sport club RSS feeds, The Guardian
Open Platform, NewsAPI.org, GNews and any extra RSS feeds configured by the
user. Items without a parseable publication time are dropped, because the
72-hour evidence window cannot be enforced for them.
"""
from __future__ import annotations

import html
import re
import xml.etree.ElementTree as ET
from datetime import datetime, timezone
from email.utils import parsedate_to_datetime
from urllib.parse import urlsplit, urlunsplit

from src.matchsense.http import MissingCredentials
from src.matchsense.providers.base import CREDIBILITY, NewsProvider, publisher_credibility, stable_id
from src.matchsense.schemas import NewsItem
from src.matchsense.teams import REGISTRY, TeamInfo


def canonical_url(url: str) -> str:
    parts = urlsplit(url)
    return urlunsplit((parts.scheme, parts.netloc.lower(), parts.path.rstrip("/"), "", ""))


def _clean(text: str | None) -> str:
    text = html.unescape(re.sub(r"<[^>]+>", " ", text or ""))
    return re.sub(r"\s+", " ", text).strip()


def _search_terms(info: TeamInfo) -> str:
    return f'"{info.name}"' if " " in info.name else info.name


def parse_rss(xml_text: str, team: str, publisher: str, provider: str, credibility: float,
              retrieved_at: datetime) -> list[NewsItem]:
    root = ET.fromstring(xml_text)
    items = []
    for node in root.iter("item"):
        title, link = _clean(node.findtext("title")), (node.findtext("link") or "").strip()
        pub = node.findtext("pubDate")
        if not title or not link or not pub:
            continue
        try:
            published = parsedate_to_datetime(pub).astimezone(timezone.utc)
        except (TypeError, ValueError):
            continue
        items.append(NewsItem(item_id=stable_id(canonical_url(link)), title=title,
                              summary=_clean(node.findtext("description")), url=link, source=publisher,
                              provider=provider, published_at=published, retrieved_at=retrieved_at,
                              teams=[team], credibility=credibility))
    return items


class BBCSportRSS(NewsProvider):
    name = "bbc_rss"
    url = "https://feeds.bbci.co.uk/sport/football/teams/{slug}/rss.xml"

    def news(self, team, since, until):
        info = REGISTRY.get(team)
        if not info or not info.bbc_slug:
            return []
        response = self.http.get(self.url.format(slug=info.bbc_slug), ttl_seconds=900)
        items = parse_rss(response.text, team, "BBC Sport", self.name, CREDIBILITY["bbc"], response.fetched_at)
        return [i for i in items if since <= i.published_at <= until]


class ExtraRSS(NewsProvider):
    name = "extra_rss"

    def __init__(self, http, feeds: list[tuple[str, str]]):
        super().__init__(http)
        self.feeds = feeds

    def news(self, team, since, until):
        result = []
        for feed_team, url in self.feeds:
            if REGISTRY.canonical(feed_team) != team:
                continue
            response = self.http.get(url, ttl_seconds=900)
            publisher = urlsplit(url).netloc
            items = parse_rss(response.text, team, publisher, self.name, publisher_credibility(publisher),
                              response.fetched_at)
            result.extend(i for i in items if since <= i.published_at <= until)
        return result


class GuardianNews(NewsProvider):
    name = "guardian"
    url = "https://content.guardianapis.com/search"

    def __init__(self, http, key):
        super().__init__(http)
        self.key = key

    def news(self, team, since, until):
        if not self.key:
            raise MissingCredentials("GUARDIAN_API_KEY not set")
        info = REGISTRY.get(team)
        response = self.http.get(self.url, params={
            "q": _search_terms(info) if info else team, "section": "football", "order-by": "newest",
            "from-date": since.date().isoformat(), "page-size": 30, "show-fields": "trailText",
            "api-key": self.key}, ttl_seconds=900)
        results = response.json().get("response", {}).get("results", [])
        items = []
        for row in results:
            published = datetime.fromisoformat(row["webPublicationDate"].replace("Z", "+00:00"))
            if not since <= published <= until:
                continue
            items.append(NewsItem(item_id=stable_id(canonical_url(row["webUrl"])), title=row["webTitle"],
                                  summary=_clean((row.get("fields") or {}).get("trailText")), url=row["webUrl"],
                                  source="The Guardian", provider=self.name, published_at=published,
                                  retrieved_at=response.fetched_at, teams=[team], credibility=CREDIBILITY["guardian"]))
        return items


class NewsAPI(NewsProvider):
    """NewsAPI.org /v2/everything. Note: the free developer plan delays articles."""

    name = "newsapi"
    url = "https://newsapi.org/v2/everything"

    def __init__(self, http, key):
        super().__init__(http)
        self.key = key

    def news(self, team, since, until):
        if not self.key:
            raise MissingCredentials("NEWSAPI_KEY not set")
        info = REGISTRY.get(team)
        response = self.http.get(self.url, params={
            "q": f"{_search_terms(info) if info else team} AND (injury OR lineup OR team news OR press conference)",
            "from": since.isoformat(timespec="seconds"), "to": until.isoformat(timespec="seconds"),
            "language": "en", "sortBy": "publishedAt", "pageSize": 40}, headers={"X-Api-Key": self.key},
            ttl_seconds=900)
        items = []
        for row in response.json().get("articles", []):
            if not row.get("publishedAt") or not row.get("url"):
                continue
            publisher = (row.get("source") or {}).get("name") or ""
            items.append(NewsItem(item_id=stable_id(canonical_url(row["url"])), title=_clean(row.get("title")),
                                  summary=_clean(row.get("description")), url=row["url"], source=publisher or "unknown",
                                  provider=self.name,
                                  published_at=datetime.fromisoformat(row["publishedAt"].replace("Z", "+00:00")),
                                  retrieved_at=response.fetched_at, teams=[team],
                                  credibility=publisher_credibility(publisher)))
        return [i for i in items if since <= i.published_at <= until]


class GNews(NewsProvider):
    name = "gnews"
    url = "https://gnews.io/api/v4/search"

    def __init__(self, http, key):
        super().__init__(http)
        self.key = key

    def news(self, team, since, until):
        if not self.key:
            raise MissingCredentials("GNEWS_API_KEY not set")
        info = REGISTRY.get(team)
        response = self.http.get(self.url, params={
            "q": f"{_search_terms(info) if info else team} (injury OR lineup OR \"team news\")", "lang": "en",
            "from": since.strftime("%Y-%m-%dT%H:%M:%SZ"), "to": until.strftime("%Y-%m-%dT%H:%M:%SZ"),
            "sortby": "publishedAt", "max": 25, "apikey": self.key}, ttl_seconds=900)
        items = []
        for row in response.json().get("articles", []):
            publisher = (row.get("source") or {}).get("name") or ""
            items.append(NewsItem(item_id=stable_id(canonical_url(row["url"])), title=_clean(row.get("title")),
                                  summary=_clean(row.get("description")), url=row["url"], source=publisher or "unknown",
                                  provider=self.name,
                                  published_at=datetime.fromisoformat(row["publishedAt"].replace("Z", "+00:00")),
                                  retrieved_at=response.fetched_at, teams=[team],
                                  credibility=publisher_credibility(publisher)))
        return [i for i in items if since <= i.published_at <= until]
