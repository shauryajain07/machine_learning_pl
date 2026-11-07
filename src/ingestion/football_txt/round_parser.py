"""Round/matchday heading parser."""

from __future__ import annotations

import re

from src.models import RoundInfo


ROUND_RE = re.compile(r"^\s*▪\s*(?P<heading>.*?)\s*$")
NUMBER_RE = re.compile(r"(?:^|\s|-)(?P<number>\d+)\s*$")


def parse_round_heading(line: str) -> RoundInfo | None:
    match = ROUND_RE.fullmatch(line)
    if not match:
        return None
    raw_heading = match.group("heading").strip()
    number_match = NUMBER_RE.search(raw_heading)
    if number_match:
        round_number = int(number_match.group("number"))
        round_name = raw_heading[: number_match.start()].rstrip(" -") or raw_heading
    else:
        round_number = None
        round_name = raw_heading
    return RoundInfo(
        round_name=round_name,
        round_number=round_number,
        raw_heading=raw_heading,
    )

