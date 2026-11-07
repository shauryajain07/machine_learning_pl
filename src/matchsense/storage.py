"""SQLite prediction history and outcome tracking."""
from __future__ import annotations

import json
import math
import sqlite3
import threading
import uuid
from datetime import datetime, timezone
from pathlib import Path

from src.matchsense.schemas import Fixture, FixtureStatus

SCHEMA = """
create table if not exists predictions (
    prediction_id text primary key,
    created_at text not null,
    as_of text not null,
    fixture_id text not null,
    competition text not null,
    home text not null,
    away text not null,
    kickoff_utc text not null,
    mode text not null,
    lineup_kind text not null,
    baseline_home real, baseline_draw real, baseline_away real,
    adjusted_home real, adjusted_draw real, adjusted_away real,
    xg_home real, xg_away real,
    model_version text not null,
    adjustment_status text not null,
    question text,
    payload text not null
);
create index if not exists predictions_fixture on predictions(fixture_id, created_at);
create table if not exists outcomes (
    fixture_id text primary key,
    home_goals integer not null,
    away_goals integer not null,
    recorded_at text not null,
    source text not null
);
"""


class PredictionStore:
    def __init__(self, path: Path):
        path.parent.mkdir(parents=True, exist_ok=True)
        self.path = path
        self._lock = threading.Lock()
        with self._connect() as conn:
            conn.executescript(SCHEMA)

    def _connect(self):
        conn = sqlite3.connect(self.path, timeout=30)
        conn.row_factory = sqlite3.Row
        return conn

    def save(self, response, lineup_kind: str) -> str:
        forecast = response.forecast
        fixture = response.fixture
        prediction_id = uuid.uuid4().hex[:12]
        b = forecast.baseline.probabilities
        a = (forecast.adjusted or forecast.baseline).probabilities
        adj = forecast.adjusted or forecast.baseline
        with self._lock, self._connect() as conn:
            conn.execute("insert into predictions values (?,?,?,?,?,?,?,?,?,?,?,?,?,?,?,?,?,?,?,?,?,?)", (
                prediction_id, datetime.now(timezone.utc).isoformat(), response.generated_at.isoformat(),
                fixture.fixture_id, fixture.competition.code, fixture.home, fixture.away,
                fixture.kickoff_utc.isoformat(), response.mode, lineup_kind,
                b.home_win, b.draw, b.away_win, a.home_win, a.draw, a.away_win,
                adj.expected_goals_home, adj.expected_goals_away, response.model_version,
                forecast.adjustment_status, response.question,
                response.model_dump_json(exclude={"news": True, "availability": True,
                                                  "forecast": {"simulations"}})))
        return prediction_id

    def record_outcomes(self, fixtures: list[Fixture], source: str) -> int:
        count = 0
        with self._lock, self._connect() as conn:
            for f in fixtures:
                if f.status == FixtureStatus.FINISHED and f.home_goals is not None and f.away_goals is not None:
                    cursor = conn.execute(
                        "insert or ignore into outcomes values (?,?,?,?,?)",
                        (f.fixture_id, f.home_goals, f.away_goals, datetime.now(timezone.utc).isoformat(), source))
                    count += cursor.rowcount
        return count

    def latest_for(self, fixture_id: str, before: datetime | None = None):
        query = "select * from predictions where fixture_id = ?"
        params: list = [fixture_id]
        if before is not None:
            query += " and kickoff_utc > as_of and as_of <= ?"
            params.append(before.isoformat())
        query += " order by created_at desc limit 1"
        with self._connect() as conn:
            row = conn.execute(query, params).fetchone()
        return dict(row) if row else None

    def history(self, limit: int = 100) -> dict:
        with self._connect() as conn:
            rows = [dict(r) for r in conn.execute(
                "select p.prediction_id, p.created_at, p.as_of, p.fixture_id, p.home, p.away, p.kickoff_utc, p.mode, "
                "p.lineup_kind, p.baseline_home, p.baseline_draw, p.baseline_away, p.adjusted_home, p.adjusted_draw, "
                "p.adjusted_away, p.xg_home, p.xg_away, p.model_version, p.adjustment_status, p.question, "
                "o.home_goals, o.away_goals from predictions p left join outcomes o on o.fixture_id = p.fixture_id "
                "order by p.created_at desc limit ?", (limit,))]
        scored = {"baseline": [], "adjusted": []}
        for r in rows:
            # Only score forecasts issued before kickoff.
            if r["home_goals"] is None or r["as_of"] >= r["kickoff_utc"]:
                continue
            k = 0 if r["home_goals"] > r["away_goals"] else 1 if r["home_goals"] == r["away_goals"] else 2
            for name, cols in (("baseline", ("baseline_home", "baseline_draw", "baseline_away")),
                               ("adjusted", ("adjusted_home", "adjusted_draw", "adjusted_away"))):
                p = [r[c] for c in cols]
                scored[name].append((-math.log(max(p[k], 1e-15)), sum((p[i] - (i == k)) ** 2 for i in range(3))))
        summary = {name: {"n": len(v), "log_loss": sum(x[0] for x in v) / len(v) if v else None,
                          "brier": sum(x[1] for x in v) / len(v) if v else None} for name, v in scored.items()}
        return {"predictions": rows, "scoring": summary}

    def payload(self, prediction_id: str):
        with self._connect() as conn:
            row = conn.execute("select payload from predictions where prediction_id = ?", (prediction_id,)).fetchone()
        return json.loads(row["payload"]) if row else None
