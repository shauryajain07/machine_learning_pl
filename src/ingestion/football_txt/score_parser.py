"""Parsers for score and result lines."""

from __future__ import annotations

import re
from datetime import time

from src.models import Score


SCORE_RE = re.compile(
    r"(?<!\d)(?P<home>\d{1,2})\s*-\s*(?P<away>\d{1,2})"
    r"(?:\s*\(\s*(?P<home_ht>\d{1,2})\s*-\s*(?P<away_ht>\d{1,2})\s*\))?"
)
TIME_RE = re.compile(r"^\s*(?P<hour>\d{1,2}):(?P<minute>\d{2})(?=\s|$)")


def parse_score(raw: str) -> Score:
    match = SCORE_RE.fullmatch(raw.strip())
    if not match:
        raise ValueError(f"Invalid score: {raw!r}")
    values = match.groupdict()
    return Score(
        home_score=int(values["home"]),
        away_score=int(values["away"]),
        home_ht_score=int(values["home_ht"]) if values["home_ht"] is not None else None,
        away_ht_score=int(values["away_ht"]) if values["away_ht"] is not None else None,
        raw=match.group(0).strip(),
    )


def parse_leading_time(raw: str) -> tuple[time | None, str]:
    """Remove an optional compact-file kickoff time from a result line."""

    match = TIME_RE.match(raw)
    if not match:
        return None, raw
    hour = int(match.group("hour"))
    minute = int(match.group("minute"))
    if hour > 23 or minute > 59:
        raise ValueError(f"Invalid kickoff time: {match.group(0)!r}")
    return time(hour=hour, minute=minute), raw[match.end() :]


def parse_match_result_line(line: str) -> tuple[str, str, Score, time | None] | None:
    """Parse full (`home v away score`) and compact result lines.

    The score is used as the stable delimiter. This avoids a giant team-name
    regex and supports variable spacing in historical files.
    """

    score_match = SCORE_RE.search(line)
    if not score_match:
        return None

    before = line[: score_match.start()]
    after = line[score_match.end() :]
    try:
        kickoff, before_without_time = parse_leading_time(before)
    except ValueError:
        return None

    before_without_time = before_without_time.strip()
    after = after.strip()
    full_separator = re.search(r"\s+v\s+", before_without_time)
    if full_separator:
        home_name = before_without_time[: full_separator.start()].strip()
        away_name = before_without_time[full_separator.end() :].strip()
    else:
        if not after:
            return None
        home_name = before_without_time
        away_name = after

    if not home_name or not away_name:
        return None
    try:
        score = parse_score(score_match.group(0))
    except ValueError:
        return None
    return home_name, away_name, score, kickoff
