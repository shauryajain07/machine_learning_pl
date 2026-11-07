"""Season-level counts and coverage reporting."""

from __future__ import annotations

from collections import Counter
from dataclasses import dataclass
from datetime import date

from src.models import ParsedFile, ParseIssue
from src.ingestion.validation.match_validator import validate_match


@dataclass
class SeasonReport:
    season_label: str
    expected_matches: int | None
    parsed_matches: int
    unique_teams: int
    players: int
    goals: int
    yellow_cards: int
    red_cards: int
    yellow_red_cards: int
    substitutions: int
    appearances: int
    referees: int
    clean_matches: int
    warning_matches: int
    failed_matches: int
    goal_count_mismatches: int
    lineup_count_mismatches: int
    full_file_available: bool
    coverage: dict[str, bool]
    issue_counts: Counter[str]
    file_issues: list[ParseIssue]

    def to_dict(self) -> dict[str, object]:
        return {
            "season_label": self.season_label,
            "expected_matches": self.expected_matches,
            "parsed_matches": self.parsed_matches,
            "unique_teams": self.unique_teams,
            "players": self.players,
            "goals": self.goals,
            "yellow_cards": self.yellow_cards,
            "red_cards": self.red_cards,
            "yellow_red_cards": self.yellow_red_cards,
            "substitutions": self.substitutions,
            "appearances": self.appearances,
            "referees": self.referees,
            "clean_matches": self.clean_matches,
            "warning_matches": self.warning_matches,
            "failed_matches": self.failed_matches,
            "goal_count_mismatches": self.goal_count_mismatches,
            "lineup_count_mismatches": self.lineup_count_mismatches,
            "full_file_available": self.full_file_available,
            "coverage": self.coverage,
            "issue_counts": dict(self.issue_counts),
            "file_issues": [issue.description for issue in self.file_issues],
        }


def validate_season(parsed_file: ParsedFile) -> SeasonReport:
    for match in parsed_file.matches:
        validate_match(match)

    teams = {
        value
        for match in parsed_file.matches
        for value in (match.home_source_name, match.away_source_name)
        if value
    }
    players = {
        value
        for match in parsed_file.matches
        for value in (
            [appearance.player_name for appearance in match.appearances]
            + [goal.player_name for goal in match.goals]
            + [card.player_name for card in match.cards]
            + [sub.player_off_name for sub in match.substitutions]
            + [sub.player_on_name for sub in match.substitutions]
        )
    }
    issue_counts: Counter[str] = Counter(
        issue.category for match in parsed_file.matches for issue in match.issues
    )
    goal_count_mismatches = sum(
        any(issue.category == "validation" and "goal totals" in issue.description for issue in match.issues)
        for match in parsed_file.matches
    )
    lineup_count_mismatches = sum(
        any(issue.category == "lineup" and "Expected 11" in issue.description for issue in match.issues)
        for match in parsed_file.matches
    )
    coverage = {
        "lineups": any(match.appearances for match in parsed_file.matches),
        "substitutions": any(match.substitutions for match in parsed_file.matches),
        "goals": any(match.has_goal_block for match in parsed_file.matches),
        "yellow_cards": any(card.card_type == "yellow" for match in parsed_file.matches for card in match.cards),
        "red_cards": any(card.card_type == "red" for match in parsed_file.matches for card in match.cards),
        "yellow_red_cards": any(
            card.card_type == "yellow_red" for match in parsed_file.matches for card in match.cards
        ),
        "referees": any(match.referees for match in parsed_file.matches),
        "attendance": any(match.attendance is not None for match in parsed_file.matches),
    }
    file_issues: list[ParseIssue] = list(parsed_file.file_issues)
    expected = parsed_file.header.expected_match_count
    if expected is not None and expected != len(parsed_file.matches):
        # A source file whose advertised end date is still in the future is
        # an in-progress season, not a corrupt historical file. Keep the
        # mismatch visible, but classify it as a warning so import reports
        # distinguish partial/current coverage from parser failures.
        partial_current = bool(
            parsed_file.header.end_date
            and parsed_file.header.end_date > date.today()
            and len(parsed_file.matches) < expected
        )
        severity = "warning" if partial_current else "error"
        description = (
            f"Current/in-progress source snapshot contains {len(parsed_file.matches)} "
            f"of {expected} advertised matches; missing matches were not invented."
            if partial_current
            else f"Header expected {expected} matches but parsed {len(parsed_file.matches)}."
        )
        file_issues.append(
            ParseIssue(
                severity=severity,
                category="validation",
                description=description,
                raw_fragment=parsed_file.header.source_header_text,
            )
        )
    return SeasonReport(
        season_label=parsed_file.header.season_label,
        expected_matches=expected,
        parsed_matches=len(parsed_file.matches),
        unique_teams=len(teams),
        players=len(players),
        goals=sum(len(match.goals) for match in parsed_file.matches),
        yellow_cards=sum(
            card.card_type == "yellow" for match in parsed_file.matches for card in match.cards
        ),
        red_cards=sum(card.card_type == "red" for match in parsed_file.matches for card in match.cards),
        yellow_red_cards=sum(
            card.card_type == "yellow_red" for match in parsed_file.matches for card in match.cards
        ),
        substitutions=sum(len(match.substitutions) for match in parsed_file.matches),
        appearances=sum(len(match.appearances) for match in parsed_file.matches),
        referees=sum(len(match.referees) for match in parsed_file.matches),
        clean_matches=sum(match.parse_status == "clean" for match in parsed_file.matches),
        warning_matches=sum(match.parse_status == "warning" for match in parsed_file.matches),
        failed_matches=sum(match.parse_status == "failed" for match in parsed_file.matches),
        goal_count_mismatches=goal_count_mismatches,
        lineup_count_mismatches=lineup_count_mismatches,
        full_file_available=parsed_file.source_format == "full",
        coverage=coverage,
        issue_counts=issue_counts,
        file_issues=file_issues,
    )
