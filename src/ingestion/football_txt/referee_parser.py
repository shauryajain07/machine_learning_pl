"""Referee list parser."""

from __future__ import annotations

from src.models import ParseIssue


def parse_referee_line(raw_line: str) -> tuple[list[str], list[ParseIssue]]:
    prefix, separator, value = raw_line.partition(":")
    if not separator or prefix.strip().casefold() != "refs":
        return [], [
            ParseIssue(
                severity="warning",
                category="referee",
                raw_fragment=raw_line,
                description="Line was not a recognized Refs: record.",
            )
        ]
    names = [name.strip() for name in value.split(",") if name.strip()]
    if not names:
        return [], [
            ParseIssue(
                severity="warning",
                category="referee",
                raw_fragment=raw_line,
                description="Refs: line contained no referee names.",
            )
        ]
    return names, []

