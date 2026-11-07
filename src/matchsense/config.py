"""Runtime settings read from the environment (and an optional ``.env``).

Secrets are only ever read from environment variables. Nothing here has a
default credential; a missing key disables that provider and is reported in
the forecast's data-quality section instead of failing the request.
"""
from __future__ import annotations

import os
from dataclasses import dataclass, field
from datetime import datetime, timezone
from pathlib import Path
from zoneinfo import ZoneInfo

PROJECT_ROOT = Path(__file__).resolve().parents[2]
PACKAGE_ROOT = Path(__file__).resolve().parent

try:  # Optional convenience: load .env without overriding real env vars.
    from dotenv import load_dotenv

    load_dotenv(PROJECT_ROOT / ".env", override=False)
except ImportError:  # pragma: no cover - dotenv is a declared dependency
    pass


def _env(name: str, default: str | None = None) -> str | None:
    value = os.environ.get(name)
    return value.strip() if value and value.strip() else default


def _env_float(name: str, default: float) -> float:
    value = _env(name)
    return float(value) if value is not None else default


def _env_int(name: str, default: int) -> int:
    value = _env(name)
    return int(value) if value is not None else default


@dataclass
class Settings:
    mode: str = "live"  # "live" or "demo"
    timezone: str = "Asia/Hong_Kong"
    demo_now: datetime | None = None
    data_dir: Path = PROJECT_ROOT / "data"
    artifacts_dir: Path = PROJECT_ROOT / "artifacts"
    baseline_bundle: Path = PROJECT_ROOT / "artifacts" / "season_simulator" / "model.joblib"
    impact_artifact: Path = PROJECT_ROOT / "artifacts" / "matchsense" / "lineup_impact.json"
    england_root: Path = PROJECT_ROOT / "england"
    news_window_hours: int = 72
    simulations: int = 2000
    seed: int = 20261104
    http_timeout: float = 15.0
    user_agent: str = "MatchSenseAI/0.1 (+local research tool)"
    api_football_key: str | None = None
    football_data_key: str | None = None
    guardian_key: str | None = None
    newsapi_key: str | None = None
    gnews_key: str | None = None
    openai_key: str | None = None
    openai_model: str = "gpt-5-mini"
    # "auto" = Codex CLI if installed, else OpenAI API if keyed, else rules. Demo mode uses rules.
    query_interpreter: str = "auto"
    codex_model: str | None = None
    use_fpl: bool = True
    use_bbc: bool = True
    use_openfootball_remote: bool = True
    extra_rss: list[tuple[str, str]] = field(default_factory=list)

    @property
    def tz(self) -> ZoneInfo:
        return ZoneInfo(self.timezone)

    @property
    def is_demo(self) -> bool:
        return self.mode == "demo"

    def now(self) -> datetime:
        """Wall clock in UTC; frozen to the recorded instant in demo mode."""
        if self.is_demo and self.demo_now is not None:
            return self.demo_now
        return datetime.now(timezone.utc)

    @property
    def cache_path(self) -> Path:
        return self.data_dir / ("demo_cache.sqlite" if self.is_demo else "http_cache.sqlite")

    @property
    def db_path(self) -> Path:
        return self.data_dir / ("matchsense_demo.sqlite" if self.is_demo else "matchsense.sqlite")

    def credentials_report(self) -> dict[str, bool]:
        return {
            "API_FOOTBALL_KEY": bool(self.api_football_key),
            "FOOTBALL_DATA_API_KEY": bool(self.football_data_key),
            "GUARDIAN_API_KEY": bool(self.guardian_key),
            "NEWSAPI_KEY": bool(self.newsapi_key),
            "GNEWS_API_KEY": bool(self.gnews_key),
            "OPENAI_API_KEY": bool(self.openai_key),
        }


def _parse_extra_rss(raw: str | None) -> list[tuple[str, str]]:
    """``"Manchester City|https://...rss,Arsenal|https://..."`` -> pairs."""
    pairs = []
    for chunk in (raw or "").split(","):
        if "|" in chunk:
            team, url = chunk.split("|", 1)
            if url.strip().startswith("https://"):
                pairs.append((team.strip(), url.strip()))
    return pairs


def load_settings(**overrides) -> Settings:
    demo_now = _env("MATCHSENSE_DEMO_NOW")
    settings = Settings(
        mode=_env("MATCHSENSE_MODE", "live").lower(),
        timezone=_env("MATCHSENSE_TIMEZONE", "Asia/Hong_Kong"),
        demo_now=datetime.fromisoformat(demo_now.replace("Z", "+00:00")) if demo_now else None,
        data_dir=Path(_env("MATCHSENSE_DATA_DIR", str(PROJECT_ROOT / "data"))),
        news_window_hours=_env_int("MATCHSENSE_NEWS_WINDOW_HOURS", 72),
        simulations=_env_int("MATCHSENSE_SIMULATIONS", 2000),
        http_timeout=_env_float("MATCHSENSE_HTTP_TIMEOUT", 15.0),
        api_football_key=_env("API_FOOTBALL_KEY"),
        football_data_key=_env("FOOTBALL_DATA_API_KEY"),
        guardian_key=_env("GUARDIAN_API_KEY"),
        newsapi_key=_env("NEWSAPI_KEY"),
        gnews_key=_env("GNEWS_API_KEY"),
        openai_key=_env("OPENAI_API_KEY"),
        openai_model=_env("OPENAI_MODEL", "gpt-5-mini"),
        query_interpreter=_env("MATCHSENSE_QUERY_INTERPRETER", "auto").lower(),
        codex_model=_env("MATCHSENSE_CODEX_MODEL"),
        use_fpl=_env("MATCHSENSE_USE_FPL", "1") != "0",
        use_bbc=_env("MATCHSENSE_USE_BBC", "1") != "0",
        use_openfootball_remote=_env("MATCHSENSE_USE_OPENFOOTBALL_REMOTE", "1") != "0",
        extra_rss=_parse_extra_rss(_env("MATCHSENSE_EXTRA_RSS")),
    )
    for key, value in overrides.items():
        if not hasattr(settings, key):
            raise AttributeError(f"Unknown setting {key!r}")
        setattr(settings, key, value)
    if settings.mode not in ("live", "demo"):
        raise ValueError("MATCHSENSE_MODE must be 'live' or 'demo'")
    ZoneInfo(settings.timezone)  # Fail fast on an invalid zone name.
    return settings
