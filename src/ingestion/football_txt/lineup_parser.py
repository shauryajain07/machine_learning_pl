"""Lineup, captain, and substitution parsing.

The visual `-` separators in Football.TXT are retained as group indexes only.
They are deliberately not converted into positions.
"""

from __future__ import annotations

from dataclasses import dataclass

from src.ingestion.football_txt.minute_parser import parse_minute_prefix
from src.ingestion.normalization.names import clean_display_name
from src.models import (
    ParsedAppearance,
    ParsedSubstitution,
    ParseIssue,
    Side,
    MinuteMark,
)


@dataclass
class _LineupContext:
    side: Side
    appearances: list[ParsedAppearance]
    substitutions: list[ParsedSubstitution]
    issues: list[ParseIssue]


def split_lineup_entries(raw_text: str) -> list[tuple[str, int]]:
    """Split top-level comma/dash entries and return `(text, group_index)`."""

    entries: list[tuple[str, int]] = []
    buffer: list[str] = []
    depth = 0
    group_index = 0

    def flush() -> None:
        value = "".join(buffer).strip()
        buffer.clear()
        if value:
            entries.append((value, group_index))

    index = 0
    while index < len(raw_text):
        char = raw_text[index]
        if char == "(":
            depth += 1
            buffer.append(char)
        elif char == ")":
            if depth:
                depth -= 1
            buffer.append(char)
        elif char == "," and depth == 0:
            flush()
        elif (
            char == "-"
            and depth == 0
            and index > 0
            and index + 1 < len(raw_text)
            and raw_text[index - 1].isspace()
            and raw_text[index + 1].isspace()
        ):
            flush()
            group_index += 1
        else:
            buffer.append(char)
        index += 1
    flush()
    return entries


def _outer_parenthesis(raw: str) -> tuple[str, str | None, str | None]:
    """Return `(prefix, inside, suffix)` for the first outer parentheses."""

    start = raw.find("(")
    if start < 0:
        return raw, None, None
    depth = 0
    for index in range(start, len(raw)):
        char = raw[index]
        if char == "(":
            depth += 1
        elif char == ")":
            depth -= 1
            if depth == 0:
                return raw[:start], raw[start + 1 : index], raw[index + 1 :]
    return raw[:start], None, raw[start:]


def _source_text(raw: str) -> str:
    return " ".join(raw.split())


def _parse_node(
    raw: str,
    *,
    context: _LineupContext,
    started: bool,
    entered: MinuteMark | None,
    group_index: int,
    source_order: int,
) -> tuple[ParsedAppearance | None, int]:
    source_text = _source_text(raw)
    prefix, inside, suffix = _outer_parenthesis(source_text)
    captain = "[c]" in prefix.casefold()
    player_name = clean_display_name(prefix)
    if not player_name:
        context.issues.append(
            ParseIssue(
                severity="warning",
                category="lineup",
                raw_fragment=source_text,
                description="Lineup entry did not contain a player name.",
            )
        )
        return None, source_order

    appearance = ParsedAppearance(
        side=context.side,
        player_name=player_name,
        started=started,
        captain=captain,
        entered=entered,
        exited=None,
        source_order=source_order,
        raw_group_index=group_index,
        raw_player_text=source_text,
    )
    context.appearances.append(appearance)
    source_order += 1

    if inside is None:
        if suffix:
            context.issues.append(
                ParseIssue(
                    severity="warning",
                    category="lineup",
                    raw_fragment=suffix,
                    description="Unbalanced lineup parentheses followed a player entry.",
                )
            )
        return appearance, source_order

    parsed_prefix = parse_minute_prefix(inside)
    if parsed_prefix is None:
        context.issues.append(
            ParseIssue(
                severity="warning",
                category="substitution",
                raw_fragment=inside,
                description="Could not parse the substitution minute inside a lineup entry.",
            )
        )
        return appearance, source_order

    minute_mark, child_raw = parsed_prefix
    child_raw = child_raw.strip()
    if not child_raw:
        context.issues.append(
            ParseIssue(
                severity="warning",
                category="substitution",
                raw_fragment=inside,
                description="Substitution event had no incoming player.",
            )
        )
        appearance.exited = minute_mark
        return appearance, source_order

    appearance.exited = minute_mark
    child_prefix, _, _ = _outer_parenthesis(_source_text(child_raw))
    child_name = clean_display_name(child_prefix)
    if not child_name:
        context.issues.append(
            ParseIssue(
                severity="warning",
                category="substitution",
                raw_fragment=child_raw,
                description="Substitution incoming player could not be identified.",
            )
        )
        return appearance, source_order
    context.substitutions.append(
        ParsedSubstitution(
            side=context.side,
            player_off_name=appearance.player_name,
            player_on_name=child_name,
            minute_mark=minute_mark,
            source_order=len(context.substitutions) + 1,
            raw_event_text=inside.strip(),
        )
    )
    child, source_order = _parse_node(
        child_raw,
        context=context,
        started=False,
        entered=minute_mark,
        group_index=group_index,
        source_order=source_order,
    )
    if child is None:
        context.substitutions.pop()
        return appearance, source_order
    return appearance, source_order


def parse_lineup(raw_text: str, *, side: Side) -> tuple[list[ParsedAppearance], list[ParsedSubstitution], list[ParseIssue]]:
    context = _LineupContext(side=side, appearances=[], substitutions=[], issues=[])
    source_order = 1
    for entry, group_index in split_lineup_entries(raw_text):
        _, source_order = _parse_node(
            entry,
            context=context,
            started=True,
            entered=None,
            group_index=group_index,
            source_order=source_order,
        )
    return context.appearances, context.substitutions, context.issues
