"""LLM query understanding that runs before the deterministic pipeline.

The interpreter (Codex CLI or the OpenAI API) only reads the question and
returns a typed structure: intent, clubs, a verbatim date phrase and players
in/out. Everything it returns is validated: clubs must map to the canonical
registry, the date phrase is re-resolved by ``query.resolve_window`` in the
user's timezone, and player names are later matched against real squads.
Interpretations are cached by question text, so a repeated question costs
no tokens. Any failure falls back to the rule-based parser.
"""
from __future__ import annotations

import hashlib
import json
import shutil
import sqlite3
import subprocess
import tempfile
import threading
from datetime import datetime
from pathlib import Path
from typing import Literal

from pydantic import BaseModel, Field

from src.matchsense.query import ParsedQuery, resolve_window
from src.matchsense.teams import REGISTRY, TEAMS


class QueryInterpretation(BaseModel):
    intent: Literal["outcome", "next", "what_if", "unknown"]
    subject_team: str | None = Field(description="Club whose chances are asked about")
    opponent_team: str | None
    date_phrase: str | None = Field(description="Date words copied verbatim, e.g. 'tonight', 'on Saturday'")
    players_out: list[str]
    players_in: list[str]


PROMPT = """You interpret questions for a Premier League forecasting app. Do not run any commands.
Return only the JSON object required by the schema.
- intent: "what_if" if the question asks how a player being out/in changes the chances; "next" if it asks
  about a club's next game; "outcome" for win/chance/odds questions; "unknown" if it is not about a match.
- subject_team: the club whose chances are asked about, using one of these exact names, else null:
  {teams}
  If only a player is named and you are not certain of his current club, use null.
- opponent_team: the other club if named, same naming rule, else null.
- date_phrase: copy the date/time words exactly as written ("tonight", "tomorrow", "this weekend",
  "on Saturday", "7 November"); never convert them to dates; null if none.
- players_out: players described as not playing (injured, out, suspended, rested, doesn't play).
- players_in: players described as playing or returning.
Question: {question}"""
PROMPT_VERSION = hashlib.sha256(PROMPT.encode()).hexdigest()[:10]


def _schema() -> dict:
    nullable = {"type": ["string", "null"]}
    return {"type": "object", "additionalProperties": False,
            "required": ["intent", "subject_team", "opponent_team", "date_phrase", "players_out", "players_in"],
            "properties": {"intent": {"type": "string", "enum": ["outcome", "next", "what_if", "unknown"]},
                           "subject_team": nullable, "opponent_team": nullable, "date_phrase": nullable,
                           "players_out": {"type": "array", "items": {"type": "string"}},
                           "players_in": {"type": "array", "items": {"type": "string"}}}}


def _prompt(question: str) -> str:
    return PROMPT.format(teams=", ".join(t.name for t in TEAMS), question=question.strip()[:300])


class InterpretationCache:
    def __init__(self, path: Path):
        path.parent.mkdir(parents=True, exist_ok=True)
        self.path = path
        self._lock = threading.Lock()
        with sqlite3.connect(path) as conn:
            conn.execute("create table if not exists interpretations (key text primary key, backend text, "
                         "question text, result text, created_at text)")

    @staticmethod
    def key(backend: str, question: str) -> str:
        normal = " ".join(question.lower().split())
        return hashlib.sha256(f"{backend}|{PROMPT_VERSION}|{normal}".encode()).hexdigest()

    def get(self, backend: str, question: str) -> QueryInterpretation | None:
        with self._lock, sqlite3.connect(self.path) as conn:
            row = conn.execute("select result from interpretations where key = ?",
                               (self.key(backend, question),)).fetchone()
        return QueryInterpretation.model_validate_json(row[0]) if row else None

    def put(self, backend: str, question: str, result: QueryInterpretation):
        with self._lock, sqlite3.connect(self.path) as conn:
            conn.execute("insert or replace into interpretations values (?, ?, ?, ?, datetime('now'))",
                         (self.key(backend, question), backend, question, result.model_dump_json()))


class CodexInterpreter:
    """Runs ``codex exec`` non-interactively with a strict JSON output schema."""

    def __init__(self, binary: str = "codex", model: str | None = None, timeout: float = 90.0, runner=None):
        self.binary = binary
        self.model = model
        self.timeout = timeout
        self.runner = runner or subprocess.run  # Injectable for tests.
        self.name = f"codex-cli{':' + model if model else ''}"

    @staticmethod
    def available(binary: str = "codex") -> bool:
        return shutil.which(binary) is not None

    def interpret(self, question: str) -> QueryInterpretation:
        with tempfile.TemporaryDirectory(prefix="matchsense-codex-") as work:
            schema, output = Path(work) / "schema.json", Path(work) / "out.json"
            schema.write_text(json.dumps(_schema()))
            cmd = [self.binary, "exec", "--ephemeral", "--skip-git-repo-check", "-s", "read-only", "-C", work,
                   "--output-schema", str(schema), "-o", str(output)]
            if self.model:
                cmd += ["-m", self.model]
            cmd.append(_prompt(question))
            # stdin must be closed or codex waits for additional input.
            result = self.runner(cmd, stdin=subprocess.DEVNULL, capture_output=True, text=True, timeout=self.timeout)
            if result.returncode != 0 or not output.exists():
                raise RuntimeError(f"codex exited {result.returncode}: {(result.stderr or '')[-300:]}")
            return QueryInterpretation.model_validate_json(output.read_text())


class OpenAIInterpreter:
    def __init__(self, api_key: str, model: str, client=None):
        if client is None:
            from openai import OpenAI
            client = OpenAI(api_key=api_key, timeout=30, max_retries=1)
        self.client, self.model, self.name = client, model, f"openai:{model}"

    def interpret(self, question: str) -> QueryInterpretation:
        response = self.client.responses.parse(model=self.model, input=_prompt(question),
                                               text_format=QueryInterpretation)
        return response.output_parsed


def apply_interpretation(parsed: ParsedQuery, interp: QueryInterpretation, now: datetime, tz: str) -> list[str]:
    """Merge a validated interpretation into the rule-based parse; returns what changed."""
    changes = []
    subject = REGISTRY.canonical(interp.subject_team) if interp.subject_team else None
    opponent = REGISTRY.canonical(interp.opponent_team) if interp.opponent_team else None
    if subject and subject != parsed.subject:
        parsed.subject = subject
        changes.append(f"subject {subject}")
    if opponent and opponent != subject and opponent != parsed.opponent:
        parsed.opponent = opponent
        changes.append(f"opponent {opponent}")
    if subject or opponent:
        parsed.teams = [t for t in (parsed.subject, parsed.opponent) if t]
    if interp.players_out or interp.players_in:
        players = interp.players_out or interp.players_in
        kind = "unavailable" if interp.players_out else "available"
        if players != parsed.what_if_players or kind != parsed.what_if_kind:
            parsed.what_if_players, parsed.what_if_kind = players, kind
            changes.append(f"players {kind}: {', '.join(players)}")
        parsed.intent = "what_if"
    elif interp.intent in ("outcome", "next") and parsed.intent != interp.intent and not parsed.what_if_players:
        parsed.intent = interp.intent
        changes.append(f"intent {interp.intent}")
    if interp.date_phrase and not parsed.explicit_date:
        window, explicit = resolve_window(interp.date_phrase, now, tz)
        if explicit:  # The phrase is re-resolved deterministically; never trusted as a date.
            parsed.window, parsed.explicit_date = window, True
            changes.append(f"date '{interp.date_phrase}' -> {window.label}")
    return changes
