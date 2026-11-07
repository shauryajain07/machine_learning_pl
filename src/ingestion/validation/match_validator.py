"""Conservative post-parse match validation."""

from __future__ import annotations

from collections import defaultdict

from src.ingestion.normalization.names import normalize_name
from src.models import ParsedMatch, ParseIssue


def _issue(
    match: ParsedMatch,
    *,
    severity: str,
    category: str,
    description: str,
    raw_fragment: str | None = None,
) -> None:
    if any(
        issue.severity == severity
        and issue.category == category
        and issue.description == description
        and issue.raw_fragment == raw_fragment
        for issue in match.issues
    ):
        return
    match.issues.append(
        ParseIssue(
            severity=severity,  # type: ignore[arg-type]
            category=category,
            description=description,
            raw_fragment=raw_fragment,
        )
    )


def _validate_substitutions(match: ParsedMatch) -> None:
    if not match.appearances:
        return
    for side in ("home", "away"):
        side_appearances = [appearance for appearance in match.appearances if appearance.side == side]
        active = {
            normalize_name(appearance.player_name)
            for appearance in side_appearances
            if appearance.started
        }
        known = {normalize_name(appearance.player_name) for appearance in side_appearances}
        substitutions = sorted(
            (sub for sub in match.substitutions if sub.side == side),
            key=lambda sub: sub.source_order,
        )
        for substitution in substitutions:
            player_off = normalize_name(substitution.player_off_name)
            player_on = normalize_name(substitution.player_on_name)
            if player_off not in known or player_on not in known:
                _issue(
                    match,
                    severity="warning",
                    category="substitution",
                    raw_fragment=substitution.raw_event_text,
                    description="Substitution player was not present in the parsed lineup set.",
                )
            if player_off not in active:
                _issue(
                    match,
                    severity="warning",
                    category="substitution",
                    raw_fragment=substitution.raw_event_text,
                    description="Substitution player-off was not active at the parsed event point.",
                )
            else:
                active.remove(player_off)
            if player_on in active:
                _issue(
                    match,
                    severity="warning",
                    category="substitution",
                    raw_fragment=substitution.raw_event_text,
                    description="Substitution player-on was already active at the parsed event point.",
                )
            else:
                active.add(player_on)
            if player_off == player_on:
                _issue(
                    match,
                    severity="error",
                    category="substitution",
                    raw_fragment=substitution.raw_event_text,
                    description="Substitution player-off and player-on were identical.",
                )


def validate_match(match: ParsedMatch) -> ParsedMatch:
    """Mutate and return `match` with a clean/warning/failed status."""

    if not match.home_source_name or not match.away_source_name:
        _issue(
            match,
            severity="error",
            category="validation",
            description="Exactly two team names are required for a match.",
        )
    elif normalize_name(match.home_source_name) == normalize_name(match.away_source_name):
        _issue(
            match,
            severity="error",
            category="team_identity",
            raw_fragment=f"{match.home_source_name} v {match.away_source_name}",
            description="Home and away team names resolve to the same normalized value.",
        )

    if match.match_date is None:
        _issue(
            match,
            severity="error",
            category="date",
            description="Match date could not be resolved.",
        )
    elif match.source_weekday and match.match_date.strftime("%a") != match.source_weekday:
        _issue(
            match,
            severity="warning",
            category="date",
            raw_fragment=match.source_weekday,
            description=(
                f"Source weekday {match.source_weekday!r} disagrees with the resolved date "
                f"{match.match_date.isoformat()!r}."
            ),
        )

    if match.kickoff_local_time is None:
        _issue(
            match,
            severity="warning",
            category="match_boundary",
            description="Kickoff local time is missing from the source record.",
        )
    if match.score is None:
        _issue(
            match,
            severity="error",
            category="score",
            description="Final score could not be parsed.",
        )
    else:
        if match.home_ht_score is not None and match.home_ht_score > match.home_score:  # type: ignore[operator]
            _issue(
                match,
                severity="warning",
                category="halftime_score",
                raw_fragment=match.score.raw,
                description="Home halftime score exceeds the final home score.",
            )
        if match.away_ht_score is not None and match.away_ht_score > match.away_score:  # type: ignore[operator]
            _issue(
                match,
                severity="warning",
                category="halftime_score",
                raw_fragment=match.score.raw,
                description="Away halftime score exceeds the final away score.",
            )
        if match.has_goal_block:
            home_goals = sum(goal.side == "home" for goal in match.goals)
            away_goals = sum(goal.side == "away" for goal in match.goals)
            if home_goals != match.home_score or away_goals != match.away_score:
                _issue(
                    match,
                    severity="warning",
                    category="validation",
                    raw_fragment=match.raw_goal_block,
                    description=(
                        "Parsed goal totals do not match the final score: "
                        f"parsed {home_goals}-{away_goals}, score {match.home_score}-{match.away_score}."
                    ),
                )

    starters_by_side = defaultdict(int)
    if match.appearances:
        for appearance in match.appearances:
            if appearance.started:
                starters_by_side[appearance.side] += 1
        for side in ("home", "away"):
            if starters_by_side[side] != 11:
                _issue(
                    match,
                    severity="warning",
                    category="lineup",
                    description=(
                        f"Expected 11 {side} starters but parsed {starters_by_side[side]}."
                    ),
                )
        _validate_substitutions(match)

        known_players = {
            side: {normalize_name(appearance.player_name) for appearance in match.appearances if appearance.side == side}
            for side in ("home", "away")
        }
        for card in match.cards:
            if normalize_name(card.player_name) not in known_players[card.side]:
                _issue(
                    match,
                    severity="warning",
                    category=f"{card.card_type}_card",
                    raw_fragment=card.raw_event_text,
                    description="Carded player was not found in that team's parsed lineup.",
                )

    if any(issue.severity == "error" for issue in match.issues):
        match.parse_status = "failed"
    elif match.issues:
        match.parse_status = "warning"
    else:
        match.parse_status = "clean"
    return match
