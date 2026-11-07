"""FastAPI service and static dashboard.

Run with::

    ./.venv/bin/uvicorn src.matchsense.api:app --port 8000
"""
from __future__ import annotations

import threading
from datetime import datetime
from zoneinfo import ZoneInfo

from fastapi import FastAPI, HTTPException
from fastapi.responses import FileResponse
from fastapi.staticfiles import StaticFiles
from pydantic import BaseModel, Field

from src.matchsense import __version__
from src.matchsense.config import PACKAGE_ROOT, load_settings
from src.matchsense.pipeline import IMPACT_STATUS_TEXT, MatchSense
from src.matchsense.providers.demo import manifest
from src.matchsense.schemas import AskResponse

app = FastAPI(title="MatchSense AI", version=__version__)
app.mount("/static", StaticFiles(directory=PACKAGE_ROOT / "static"), name="static")
_engine: MatchSense | None = None
_lock = threading.Lock()


def engine() -> MatchSense:
    global _engine
    with _lock:
        if _engine is None:
            _engine = MatchSense(load_settings())
        return _engine


class AskRequest(BaseModel):
    question: str = Field(min_length=2, max_length=300)
    timezone: str | None = None
    refresh: bool = False
    as_of: datetime | None = Field(default=None, description="Demo mode only: replay the demo at this instant")


@app.get("/")
def index():
    return FileResponse(PACKAGE_ROOT / "static" / "index.html")


@app.get("/api/health")
def health():
    ms = engine()
    s = ms.settings
    impact = ms.impact
    info = {"version": __version__, "mode": s.mode, "timezone": s.timezone, "now": s.now().isoformat(),
            "credentials": s.credentials_report(),
            "extractor": getattr(ms.extractor, "name", "rules"),
            "impact_status": impact.status if impact else "missing",
            "impact_status_text": IMPACT_STATUS_TEXT.get(impact.status if impact else "missing", ""),
            "keyless_sources": {"FPL": s.use_fpl, "BBC Sport RSS": s.use_bbc,
                                "openfootball": s.use_openfootball_remote}}
    try:
        info["baseline"] = ms.baseline.version
    except Exception as exc:  # Report, don't crash the health check.
        info["baseline"] = f"unavailable: {exc}"
    if s.is_demo:
        info["demo"] = manifest()
    return info


@app.post("/api/ask", response_model=AskResponse)
def ask(request: AskRequest):
    ms = engine()
    if request.timezone:
        try:
            ZoneInfo(request.timezone)
        except Exception:
            raise HTTPException(status_code=422, detail=f"Unknown timezone {request.timezone!r}")
    as_of = None
    if request.as_of is not None:
        if not ms.settings.is_demo:
            raise HTTPException(status_code=422, detail="as_of is only accepted in demo mode")
        as_of = request.as_of
    return ms.ask(request.question, timezone=request.timezone, as_of=as_of, refresh=request.refresh)


@app.get("/api/history")
def history(limit: int = 100):
    return engine().store.history(limit=min(max(limit, 1), 500))


@app.get("/api/prediction/{prediction_id}")
def prediction(prediction_id: str):
    payload = engine().store.payload(prediction_id)
    if payload is None:
        raise HTTPException(status_code=404, detail="Unknown prediction id")
    return payload
