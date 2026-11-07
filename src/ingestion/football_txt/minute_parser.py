"""Reusable Football.TXT minute parser."""

from __future__ import annotations

import re

from src.models import MinuteMark


_MINUTE_RE = re.compile(
    r"^\s*(?P<minute>\d{1,3})(?:\s*\+\s*(?P<stoppage>\d{1,2}))?\s*['’]?\s*$"
)


def parse_minute(raw: str) -> MinuteMark:
    """Parse `45'` or `90+4'`, retaining regulation and stoppage parts.

    The parser intentionally allows unusual but syntactically valid values so
    that one bad historical event does not abort a season. Callers can record
    a warning when a source-specific range is suspicious.
    """

    match = _MINUTE_RE.fullmatch(raw)
    if not match:
        raise ValueError(f"Invalid football minute: {raw!r}")
    minute = int(match.group("minute"))
    stoppage = int(match.group("stoppage") or 0)
    return MinuteMark(
        minute=minute,
        stoppage_time=stoppage,
        absolute_minute=minute + stoppage,
        raw=raw.strip(),
    )


def parse_minute_prefix(raw: str) -> tuple[MinuteMark, str] | None:
    """Parse a minute at the start of `raw` and return the remaining text."""

    match = re.match(
        r"^\s*(?P<minute>\d{1,3}(?:\s*\+\s*\d{1,2})?\s*['’])\s*(?P<rest>.*)$",
        raw,
    )
    if not match:
        return None
    mark = parse_minute(match.group("minute"))
    return mark, match.group("rest")

