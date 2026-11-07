"""Shared HTTP layer: SQLite response cache, retries, per-host throttling.

Credentials are passed as headers or query parameters by providers, but the
cache key is computed with secret parameters removed, and secrets are never
written to the cache or logs.
"""
from __future__ import annotations

import contextvars
import hashlib
import json
import sqlite3
import threading
import time
from dataclasses import dataclass
from datetime import datetime, timezone
from pathlib import Path
from urllib.parse import urlparse

import httpx

SECRET_PARAMS = {"api-key", "apikey", "apiKey", "token", "key"}
# Set by a manual refresh: caps cache freshness for the current request only.
TTL_CAP: contextvars.ContextVar[float | None] = contextvars.ContextVar("ttl_cap", default=None)


class ProviderError(RuntimeError):
    """A provider could not return usable data."""


class MissingCredentials(ProviderError):
    """A provider is disabled because its API key is not configured."""


@dataclass
class CachedResponse:
    body: bytes
    fetched_at: datetime
    from_cache: bool
    stale: bool
    status_code: int = 200

    def json(self):
        return json.loads(self.body)

    @property
    def text(self) -> str:
        return self.body.decode("utf-8", errors="replace")


class HttpCache:
    def __init__(self, path: Path):
        path.parent.mkdir(parents=True, exist_ok=True)
        self.path = path
        self._lock = threading.Lock()
        with self._connect() as conn:
            conn.execute("create table if not exists responses (key text primary key, url text, "
                         "body blob, fetched_at text, status integer)")

    def _connect(self):
        return sqlite3.connect(self.path, timeout=30)

    def get(self, key: str) -> tuple[bytes, datetime, int] | None:
        with self._lock, self._connect() as conn:
            row = conn.execute("select body, fetched_at, status from responses where key = ?", (key,)).fetchone()
        if row is None:
            return None
        return row[0], datetime.fromisoformat(row[1]), row[2]

    def put(self, key: str, url: str, body: bytes, fetched_at: datetime, status: int = 200):
        with self._lock, self._connect() as conn:
            conn.execute("insert or replace into responses values (?, ?, ?, ?, ?)",
                         (key, url, body, fetched_at.isoformat(), status))


class HttpClient:
    def __init__(self, cache: HttpCache, user_agent: str, timeout: float = 15.0,
                 min_interval: dict[str, float] | None = None, transport: httpx.BaseTransport | None = None,
                 offline: bool = False, clock=None):
        self.cache = cache
        self.offline = offline  # Demo/tests: never touch the network.
        self.min_interval = {"default": 0.25, **(min_interval or {})}
        self._last_call: dict[str, float] = {}
        self._lock = threading.Lock()
        self._clock = clock or (lambda: datetime.now(timezone.utc))
        self._client = httpx.Client(timeout=timeout, headers={"User-Agent": user_agent},
                                    follow_redirects=True, transport=transport)

    @staticmethod
    def cache_key(url: str, params: dict | None) -> str:
        public = {k: v for k, v in sorted((params or {}).items()) if k not in SECRET_PARAMS}
        return hashlib.sha256(f"{url}?{json.dumps(public, sort_keys=True)}".encode()).hexdigest()

    def _throttle(self, host: str):
        interval = self.min_interval.get(host, self.min_interval["default"])
        with self._lock:
            wait = self._last_call.get(host, 0) + interval - time.monotonic()
            if wait > 0:
                time.sleep(wait)
            self._last_call[host] = time.monotonic()

    def get(self, url: str, params: dict | None = None, headers: dict | None = None,
            ttl_seconds: float = 900, retries: int = 3) -> CachedResponse:
        cap = TTL_CAP.get()
        if cap is not None and not self.offline:
            ttl_seconds = min(ttl_seconds, cap)
        key = self.cache_key(url, params)
        cached = self.cache.get(key)
        now = self._clock()
        if cached and (self.offline or (now - cached[1]).total_seconds() < ttl_seconds):
            return CachedResponse(cached[0], cached[1], True, False, cached[2])
        if self.offline:
            raise ProviderError(f"Offline mode and no recorded response for {urlparse(url).netloc}{urlparse(url).path}")
        host = urlparse(url).netloc
        last_error: Exception | None = None
        for attempt in range(retries):
            self._throttle(host)
            try:
                response = self._client.get(url, params=params, headers=headers)
            except httpx.HTTPError as exc:
                last_error = exc
            else:
                if response.status_code in (401, 403):
                    raise ProviderError(f"{host} rejected the request (HTTP {response.status_code}); check the API key/plan")
                if response.status_code == 429 or response.status_code >= 500:
                    last_error = ProviderError(f"{host} returned HTTP {response.status_code}")
                    retry_after = response.headers.get("Retry-After")
                    if retry_after and retry_after.isdigit():
                        time.sleep(min(float(retry_after), 10))
                elif response.status_code >= 400:
                    raise ProviderError(f"{host} returned HTTP {response.status_code}")
                else:
                    fetched = self._clock()
                    self.cache.put(key, url, response.content, fetched, response.status_code)
                    return CachedResponse(response.content, fetched, False, False, response.status_code)
            time.sleep(min(0.5 * 2 ** attempt, 4))
        if cached:  # Serve stale data rather than nothing, flagged as stale.
            return CachedResponse(cached[0], cached[1], True, True, cached[2])
        raise ProviderError(f"{host} unavailable after {retries} attempts: {last_error}")

    def close(self):
        self._client.close()
