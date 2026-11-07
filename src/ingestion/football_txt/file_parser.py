"""Layered state-machine parser for OpenFootball Premier League files."""

from __future__ import annotations

import hashlib
import re
from dataclasses import dataclass, field
from pathlib import Path

from src.ingestion.football_txt.card_parser import parse_card_line
from src.ingestion.football_txt.goal_parser import parse_goal_block
from src.ingestion.football_txt.header_parser import (
    is_round_heading,
    parse_header,
)
from src.ingestion.football_txt.lineup_parser import parse_lineup
from src.ingestion.football_txt.metadata_parser import (
    FULL_METADATA_RE,
    parse_compact_date_line,
    parse_full_metadata_line,
)
from src.ingestion.football_txt.referee_parser import parse_referee_line
from src.ingestion.football_txt.round_parser import parse_round_heading
from src.ingestion.football_txt.score_parser import parse_match_result_line
from src.models import (
    ParsedFile,
    ParsedMatch,
    ParseIssue,
    RoundInfo,
    SeasonHeader,
)


TEAM_HEADER_RE = re.compile(r"^\s*(?P<team>[^:]+):\s*(?P<body>.*)$")
SPECIAL_PREFIXES = ("yellow:", "red:", "yellow-red:", "refs:")


@dataclass
class _TeamSection:
    source_name: str
    lineup_parts: list[str] = field(default_factory=list)
    raw_lineup_parts: list[str] = field(default_factory=list)


def _is_special_line(line: str) -> bool:
    return line.strip().casefold().startswith(SPECIAL_PREFIXES)


def _is_team_header(line: str) -> bool:
    if _is_special_line(line):
        return False
    return TEAM_HEADER_RE.fullmatch(line.rstrip("\r\n")) is not None


def _collect_parenthesized(lines: list[str], start: int) -> tuple[str, int, bool]:
    """Collect a scorer block, including nested `(p)`/`(og)` annotations."""

    depth = 0
    end = start
    for index in range(start, len(lines)):
        for char in lines[index]:
            if char == "(":
                depth += 1
            elif char == ")" and depth:
                depth -= 1
        end = index + 1
        if depth == 0:
            return "".join(lines[start:end]), end, True
    return "".join(lines[start:end]), end, False


def _trim_full_raw_block(block_lines: list[str]) -> tuple[str, list[str]]:
    """Exclude the following round heading/blank separator from raw match text."""

    cut = len(block_lines)
    for index, line in enumerate(block_lines):
        if index and is_round_heading(line):
            cut = index
            break
    content_lines = block_lines[:cut]
    while content_lines and not content_lines[-1].strip():
        content_lines.pop()
    return "".join(content_lines), content_lines


def _blank_match(
    *,
    header: SeasonHeader,
    source_file: str,
    source_match_index: int,
    source_format: str,
    round_info: RoundInfo | None,
    raw_match_text: str,
    issues: list[ParseIssue],
    metadata: dict[str, object] | None = None,
) -> ParsedMatch:
    metadata = metadata or {}
    return ParsedMatch(
        season_label=header.season_label,
        source_file=source_file,
        source_match_index=source_match_index,
        source_format=source_format,  # type: ignore[arg-type]
        round_info=round_info,
        match_date=metadata.get("match_date"),  # type: ignore[arg-type]
        source_weekday=metadata.get("source_weekday"),  # type: ignore[arg-type]
        kickoff_local_time=metadata.get("kickoff_local_time"),  # type: ignore[arg-type]
        utc_offset_minutes=metadata.get("utc_offset_minutes"),  # type: ignore[arg-type]
        kickoff_utc=metadata.get("kickoff_utc"),  # type: ignore[arg-type]
        venue=metadata.get("venue"),  # type: ignore[arg-type]
        location=metadata.get("location"),  # type: ignore[arg-type]
        attendance=metadata.get("attendance"),  # type: ignore[arg-type]
        home_source_name=None,
        away_source_name=None,
        score=None,
        raw_match_text=raw_match_text,
        issues=issues,
    )


def _parse_full_match(
    block_lines: list[str],
    *,
    header: SeasonHeader,
    source_file: str,
    source_match_index: int,
    round_info: RoundInfo | None,
) -> ParsedMatch:
    raw_match_text, content_lines = _trim_full_raw_block(block_lines)
    issues: list[ParseIssue] = []
    metadata: dict[str, object] | None = None
    try:
        metadata = parse_full_metadata_line(content_lines[0], header)
    except (ValueError, IndexError) as exc:
        issues.append(
            ParseIssue(
                severity="error",
                category="date",
                raw_fragment=content_lines[0] if content_lines else raw_match_text,
                description=str(exc),
            )
        )

    score_index = None
    result = None
    for index, line in enumerate(content_lines[1:], start=1):
        result = parse_match_result_line(line)
        if result is not None:
            score_index = index
            break
    if result is None or score_index is None:
        issues.append(
            ParseIssue(
                severity="error",
                category="score",
                raw_fragment=raw_match_text,
                description="No parseable home/away result line followed match metadata.",
            )
        )
        return _blank_match(
            header=header,
            source_file=source_file,
            source_match_index=source_match_index,
            source_format="full",
            round_info=round_info,
            raw_match_text=raw_match_text,
            issues=issues,
            metadata=metadata,
        )

    home_name, away_name, score, _line_time = result
    match = ParsedMatch(
        season_label=header.season_label,
        source_file=source_file,
        source_match_index=source_match_index,
        source_format="full",
        round_info=round_info,
        match_date=metadata.get("match_date") if metadata else None,  # type: ignore[arg-type]
        source_weekday=metadata.get("source_weekday") if metadata else None,  # type: ignore[arg-type]
        kickoff_local_time=metadata.get("kickoff_local_time") if metadata else None,  # type: ignore[arg-type]
        utc_offset_minutes=metadata.get("utc_offset_minutes") if metadata else None,  # type: ignore[arg-type]
        kickoff_utc=metadata.get("kickoff_utc") if metadata else None,  # type: ignore[arg-type]
        venue=metadata.get("venue") if metadata else None,  # type: ignore[arg-type]
        location=metadata.get("location") if metadata else None,  # type: ignore[arg-type]
        attendance=metadata.get("attendance") if metadata else None,  # type: ignore[arg-type]
        home_source_name=home_name,
        away_source_name=away_name,
        score=score,
        raw_match_text=raw_match_text,
        issues=issues,
    )

    next_index = score_index + 1
    while next_index < len(content_lines) and not content_lines[next_index].strip():
        next_index += 1
    if next_index < len(content_lines) and content_lines[next_index].lstrip().startswith("("):
        raw_goal_block, next_index, balanced = _collect_parenthesized(content_lines, next_index)
        match.raw_goal_block = raw_goal_block.strip()
        match.has_goal_block = True
        if not balanced:
            match.issues.append(
                ParseIssue(
                    severity="warning",
                    category="goal",
                    raw_fragment=raw_goal_block,
                    description="Scorer block parentheses were not balanced before the match ended.",
                )
            )
        goals, goal_issues = parse_goal_block(
            raw_goal_block,
            home_score=score.home_score,
            away_score=score.away_score,
        )
        match.goals.extend(goals)
        match.issues.extend(goal_issues)

    sections: list[_TeamSection] = []
    current_section: _TeamSection | None = None
    current_side = "home"
    index = next_index
    while index < len(content_lines):
        line = content_lines[index]
        stripped = line.strip()
        if not stripped:
            index += 1
            continue
        if stripped.casefold().startswith("refs:"):
            referees, referee_issues = parse_referee_line(stripped)
            match.referees.extend(referees)
            match.issues.extend(referee_issues)
            index += 1
            continue
        if (
            stripped.casefold().startswith("yellow:")
            or stripped.casefold().startswith("yellow-red:")
            or stripped.casefold().startswith("red:")
        ):
            if current_section is None:
                match.issues.append(
                    ParseIssue(
                        severity="warning",
                        category="lineup",
                        raw_fragment=line.rstrip("\r\n"),
                        description="Card line appeared before a team lineup.",
                    )
                )
            else:
                if stripped.casefold().startswith("yellow-red:"):
                    card_type = "yellow_red"
                elif stripped.casefold().startswith("yellow:"):
                    card_type = "yellow"
                else:
                    card_type = "red"
                cards, card_issues = parse_card_line(
                    stripped,
                    side=current_side,  # type: ignore[arg-type]
                    card_type=card_type,  # type: ignore[arg-type]
                )
                match.cards.extend(cards)
                match.issues.extend(card_issues)
            index += 1
            continue
        if _is_team_header(line):
            header_match = TEAM_HEADER_RE.fullmatch(line.rstrip("\r\n"))
            assert header_match is not None
            current_section = _TeamSection(source_name=header_match.group("team").strip())
            current_section.lineup_parts.append(header_match.group("body"))
            current_section.raw_lineup_parts.append(line)
            sections.append(current_section)
            current_side = "home" if len(sections) == 1 else "away"
            if len(sections) > 2:
                match.issues.append(
                    ParseIssue(
                        severity="warning",
                        category="lineup",
                        raw_fragment=line.rstrip("\r\n"),
                        description="More than two team lineup sections were found.",
                    )
                )
            index += 1
            continue
        if current_section is not None:
            current_section.lineup_parts.append(line.strip())
            current_section.raw_lineup_parts.append(line)
        else:
            match.issues.append(
                ParseIssue(
                    severity="warning",
                    category="lineup",
                    raw_fragment=line.rstrip("\r\n"),
                    description="Unrecognized content appeared outside a team section.",
                )
            )
        index += 1

    if len(sections) < 2:
        match.issues.append(
            ParseIssue(
                severity="warning",
                category="lineup",
                raw_fragment=None,
                description=f"Expected two team lineup sections; found {len(sections)}.",
            )
        )
    for section_index, section in enumerate(sections[:2]):
        side = "home" if section_index == 0 else "away"
        lineup_text = " ".join(part for part in section.lineup_parts if part)
        appearances, substitutions, lineup_issues = parse_lineup(lineup_text, side=side)  # type: ignore[arg-type]
        match.appearances.extend(appearances)
        match.substitutions.extend(substitutions)
        match.issues.extend(lineup_issues)
        raw_lineup = "".join(section.raw_lineup_parts).strip()
        if side == "home":
            match.raw_home_lineup_text = raw_lineup
        else:
            match.raw_away_lineup_text = raw_lineup

    return match


def _parse_compact_matches(
    lines: list[str],
    *,
    header: SeasonHeader,
    source_file: str,
) -> list[ParsedMatch]:
    matches: list[ParsedMatch] = []
    current_round: RoundInfo | None = None
    current_date: dict[str, object] | None = None
    current_date_raw: str | None = None
    current_time = None
    index = 0
    while index < len(lines):
        line = lines[index]
        round_info = parse_round_heading(line)
        if round_info:
            current_round = round_info
            index += 1
            continue
        date_info = parse_compact_date_line(line, header)
        if date_info:
            current_date = date_info
            current_date_raw = line
            current_time = None
            index += 1
            continue

        result = parse_match_result_line(line)
        if result is None:
            index += 1
            continue
        home_name, away_name, score, line_time = result
        issues: list[ParseIssue] = []
        if line_time is not None:
            current_time = line_time
        if current_date is None:
            issues.append(
                ParseIssue(
                    severity="error",
                    category="date",
                    raw_fragment=line.rstrip("\r\n"),
                    description="Compact result appeared before a date line.",
                )
            )
        if current_time is None:
            issues.append(
                ParseIssue(
                    severity="warning",
                    category="match_boundary",
                    raw_fragment=line.rstrip("\r\n"),
                    description="Compact result had no explicit or inherited kickoff time.",
                )
            )

        raw_goal_block = None
        has_goal_block = False
        goals = []
        next_index = index + 1
        probe = next_index
        while probe < len(lines) and not lines[probe].strip():
            probe += 1
        if probe < len(lines) and lines[probe].lstrip().startswith("("):
            raw_goal_block, after_goal, balanced = _collect_parenthesized(lines, probe)
            has_goal_block = True
            goals, goal_issues = parse_goal_block(
                raw_goal_block,
                home_score=score.home_score,
                away_score=score.away_score,
            )
            issues.extend(goal_issues)
            if not balanced:
                issues.append(
                    ParseIssue(
                        severity="warning",
                        category="goal",
                        raw_fragment=raw_goal_block,
                        description="Compact scorer block parentheses were not balanced.",
                    )
                )
            next_index = after_goal

        raw_parts = []
        if current_date_raw is not None:
            raw_parts.append(current_date_raw)
        raw_parts.append(line)
        if raw_goal_block is not None:
            raw_parts.append(raw_goal_block)
        match = ParsedMatch(
            season_label=header.season_label,
            source_file=source_file,
            source_match_index=len(matches) + 1,
            source_format="compact",
            round_info=current_round,
            match_date=current_date.get("match_date") if current_date else None,  # type: ignore[arg-type]
            source_weekday=current_date.get("source_weekday") if current_date else None,  # type: ignore[arg-type]
            kickoff_local_time=current_time,
            utc_offset_minutes=None,
            kickoff_utc=None,
            venue=None,
            location=None,
            attendance=None,
            home_source_name=home_name,
            away_source_name=away_name,
            score=score,
            raw_match_text="".join(raw_parts),
            raw_goal_block=raw_goal_block if raw_goal_block else None,
            goals=goals,
            issues=issues,
            has_goal_block=has_goal_block,
        )
        matches.append(match)
        index = next_index
    return matches


def parse_file(
    path: str | Path,
    *,
    source_file: str | None = None,
    source_commit_sha: str | None = None,
) -> ParsedFile:
    """Parse one local raw file and return typed records for validation/storage."""

    file_path = Path(path)
    raw_bytes = file_path.read_bytes()
    text = raw_bytes.decode("utf-8")
    lines = text.splitlines(keepends=True)
    header = parse_header(text)
    source_name = source_file or str(file_path)
    has_full_metadata = any(FULL_METADATA_RE.fullmatch(line.rstrip("\r\n")) for line in lines)
    if has_full_metadata:
        matches: list[ParsedMatch] = []
        metadata_indices = [
            index for index, line in enumerate(lines) if FULL_METADATA_RE.fullmatch(line.rstrip("\r\n"))
        ]
        current_round: RoundInfo | None = None
        metadata_cursor = 0
        for index, line in enumerate(lines):
            round_info = parse_round_heading(line)
            if round_info:
                current_round = round_info
            if metadata_cursor >= len(metadata_indices) or index != metadata_indices[metadata_cursor]:
                continue
            end = metadata_indices[metadata_cursor + 1] if metadata_cursor + 1 < len(metadata_indices) else len(lines)
            matches.append(
                _parse_full_match(
                    lines[index:end],
                    header=header,
                    source_file=source_name,
                    source_match_index=len(matches) + 1,
                    round_info=current_round,
                )
            )
            metadata_cursor += 1
        source_format = "full"
    else:
        matches = _parse_compact_matches(lines, header=header, source_file=source_name)
        source_format = "compact"
    return ParsedFile(
        path=str(file_path),
        season_directory=file_path.parent.name,
        header=header,
        source_format=source_format,  # type: ignore[arg-type]
        matches=matches,
        file_sha256=hashlib.sha256(raw_bytes).hexdigest(),
        file_size_bytes=len(raw_bytes),
        source_commit_sha=source_commit_sha,
    )
