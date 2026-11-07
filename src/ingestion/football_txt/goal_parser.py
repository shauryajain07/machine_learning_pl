"""Goal and event-list parsing for scorer blocks."""

from __future__ import annotations

import re
from dataclasses import dataclass

from src.ingestion.football_txt.minute_parser import parse_minute
from src.models import MinuteMark, ParsedGoal, ParseIssue, Side


EVENT_RE = re.compile(
    r"^(?P<player>.*?)\s+(?P<minute>\d{1,3}(?:\s*\+\s*\d{1,2})?\s*['’])"
    r"\s*(?:\((?P<annotation>[^()]*)\))?\s*$"
)
MINUTE_ONLY_RE = re.compile(
    r"^\s*(?P<minute>\d{1,3}(?:\s*\+\s*\d{1,2})?\s*['’])"
    r"\s*(?:\((?P<annotation>[^()]*)\))?\s*$"
)


@dataclass(frozen=True)
class EventToken:
    player_name: str | None
    minute_mark: MinuteMark | None
    annotation: str | None


def _split_top_level(raw: str, separator: str) -> list[str]:
    parts: list[str] = []
    start = 0
    depth = 0
    for index, char in enumerate(raw):
        if char == "(":
            depth += 1
        elif char == ")" and depth:
            depth -= 1
        elif char == separator and depth == 0:
            parts.append(raw[start:index])
            start = index + 1
    parts.append(raw[start:])
    return parts


def _clean_block(raw_block: str) -> str:
    value = raw_block.strip()
    if value.startswith("(") and value.endswith(")"):
        return value[1:-1]
    return value


def _tokenize_side(raw_side: str) -> list[str]:
    return [" ".join(token.split()) for token in _split_top_level(raw_side, ",") if token.strip()]


def _parse_event_token(token: str, previous_player: str | None) -> tuple[EventToken | None, ParseIssue | None]:
    match = EVENT_RE.fullmatch(token)
    if match:
        player_name = " ".join(match.group("player").split()) or None
        minute_raw = match.group("minute")
        annotation = match.group("annotation")
    else:
        minute_only = MINUTE_ONLY_RE.fullmatch(token)
        if not minute_only:
            return None, ParseIssue(
                severity="warning",
                category="goal",
                raw_fragment=token,
                description="Could not parse an event token; raw match text was retained.",
            )
        player_name = previous_player
        minute_raw = minute_only.group("minute")
        annotation = minute_only.group("annotation")
        if not player_name:
            return None, ParseIssue(
                severity="warning",
                category="goal_attribution",
                raw_fragment=token,
                description="A repeated goal minute had no preceding scorer.",
            )

    try:
        minute_mark = parse_minute(minute_raw)
    except ValueError:
        return None, ParseIssue(
            severity="warning",
            category="goal",
            raw_fragment=token,
            description=f"Invalid goal minute {minute_raw!r}.",
        )
    return EventToken(player_name, minute_mark, annotation.strip() if annotation else None), None


def _annotation_flags(annotation: str | None) -> tuple[bool, bool, ParseIssue | None]:
    if not annotation:
        return False, False, None
    normalized = annotation.casefold().strip()
    penalty = normalized in {"p", "pen", "penalty"}
    own_goal = normalized in {"og", "own goal", "own-goal"}
    if not penalty and not own_goal:
        return False, False, ParseIssue(
            severity="warning",
            category="goal",
            raw_fragment=annotation,
            description=f"Unknown goal annotation {annotation!r}; raw event was preserved.",
        )
    return penalty, own_goal, None


def _side_for_unseparated_block(
    parsed_count: int,
    home_score: int | None,
    away_score: int | None,
) -> tuple[Side, ParseIssue | None]:
    if home_score == 0 and away_score and parsed_count == away_score:
        return "away", None
    if away_score == 0 and home_score and parsed_count == home_score:
        return "home", None
    if home_score == parsed_count and away_score != parsed_count:
        return "home", None
    if away_score == parsed_count and home_score != parsed_count:
        return "away", None
    return "home", ParseIssue(
        severity="warning",
        category="goal_attribution",
        raw_fragment=None,
        description="Scorer block had no semicolon; home attribution was used because the side was ambiguous.",
    )


def parse_goal_block(
    raw_block: str,
    *,
    home_score: int | None = None,
    away_score: int | None = None,
) -> tuple[list[ParsedGoal], list[ParseIssue]]:
    """Parse one outer scorer block into home/away goal rows."""

    content = _clean_block(raw_block)
    side_groups = _split_top_level(content, ";")
    issues: list[ParseIssue] = []
    if len(side_groups) > 2:
        issues.append(
            ParseIssue(
                severity="warning",
                category="goal",
                raw_fragment=raw_block,
                description="Scorer block contained more than one semicolon; only the first separates sides.",
            )
        )
    if len(side_groups) == 1:
        tokens = _tokenize_side(side_groups[0])
        side, attribution_issue = _side_for_unseparated_block(len(tokens), home_score, away_score)
        if attribution_issue:
            issues.append(attribution_issue)
        side_groups_with_sides: list[tuple[Side, list[str]]] = [(side, tokens)]
    else:
        side_groups_with_sides = [
            ("home", _tokenize_side(side_groups[0])),
            ("away", _tokenize_side(side_groups[1])),
        ]

    goals: list[ParsedGoal] = []
    goal_order = 1
    for side, tokens in side_groups_with_sides:
        previous_player: str | None = None
        for token in tokens:
            event, issue = _parse_event_token(token, previous_player)
            if issue:
                issues.append(issue)
            if event is None or event.player_name is None or event.minute_mark is None:
                continue
            previous_player = event.player_name
            penalty, own_goal, annotation_issue = _annotation_flags(event.annotation)
            if annotation_issue:
                annotation_issue = ParseIssue(
                    severity=annotation_issue.severity,
                    category=annotation_issue.category,
                    raw_fragment=token,
                    description=annotation_issue.description,
                )
                issues.append(annotation_issue)
            goals.append(
                ParsedGoal(
                    side=side,
                    player_name=event.player_name,
                    minute_mark=event.minute_mark,
                    penalty=penalty,
                    own_goal=own_goal,
                    annotation=event.annotation,
                    goal_order=goal_order,
                    raw_event_text=token,
                )
            )
            goal_order += 1
    return goals, issues


def parse_event_list(raw_text: str, *, category: str) -> tuple[list[tuple[str, MinuteMark | None, str | None, str]], list[ParseIssue]]:
    """Parse card-like comma-separated `Player 58'` event lists."""

    events: list[tuple[str, MinuteMark | None, str | None, str]] = []
    issues: list[ParseIssue] = []
    previous_player: str | None = None
    for token in _tokenize_side(raw_text):
        event, issue = _parse_event_token(token, previous_player)
        if issue:
            issues.append(
                ParseIssue(
                    severity=issue.severity,
                    category=category,
                    raw_fragment=issue.raw_fragment,
                    description=issue.description,
                )
            )
        if event is None or event.player_name is None:
            continue
        previous_player = event.player_name
        events.append((event.player_name, event.minute_mark, event.annotation, token))
    return events, issues
