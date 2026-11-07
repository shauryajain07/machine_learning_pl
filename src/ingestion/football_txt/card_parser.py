"""Yellow/red card line parser."""

from __future__ import annotations

from typing import Literal

from src.ingestion.football_txt.goal_parser import parse_event_list
from src.models import ParsedCard, ParseIssue, Side


def parse_card_line(
    raw_line: str,
    *,
    side: Side,
    card_type: Literal["yellow", "red", "yellow_red"],
) -> tuple[list[ParsedCard], list[ParseIssue]]:
    prefix, _, event_text = raw_line.partition(":")
    if not event_text:
        return [], [
            ParseIssue(
                severity="warning",
                category=f"{card_type}_card",
                raw_fragment=raw_line,
                description=f"{prefix} card line had no event text.",
            )
        ]
    events, issues = parse_event_list(event_text.strip(), category=f"{card_type}_card")
    cards: list[ParsedCard] = []
    for source_order, (player_name, minute_mark, _annotation, raw_event_text) in enumerate(events, start=1):
        cards.append(
            ParsedCard(
                side=side,
                player_name=player_name,
                card_type=card_type,
                minute_mark=minute_mark,
                source_order=source_order,
                raw_event_text=raw_event_text,
            )
        )
    return cards, issues
