"""Season header and date helpers for Football.TXT."""

from __future__ import annotations

import re
from dataclasses import dataclass
from datetime import date, datetime
from typing import Iterable

from src.models import SeasonHeader


WEEKDAYS = "Mon Tue Wed Thu Fri Sat Sun".split()
MONTHS = "Jan Feb Mar Apr May Jun Jul Aug Sep Oct Nov Dec".split()
_WEEKDAY = r"(?:Mon|Tue|Wed|Thu|Fri|Sat|Sun)"
_MONTH = r"(?:Jan|Feb|Mar|Apr|May|Jun|Jul|Aug|Sep|Oct|Nov|Dec)"
DATE_DESCRIPTOR_RE = re.compile(
    rf"(?P<weekday>{_WEEKDAY})\s+(?P<month>{_MONTH})\s+"
    r"(?P<day>\d{1,2})(?:\s+(?P<year>\d{4}))?"
)
DATE_LINE_RE = re.compile(
    rf"^\s*(?P<weekday>{_WEEKDAY})\s+(?P<month>{_MONTH})\s+"
    r"(?P<day>\d{1,2})(?:\s+(?P<year>\d{4}))?\s*$"
)
SEASON_RE = re.compile(r"(?P<start>\d{4})\s*/\s*(?P<end>\d{2,4})")


@dataclass(frozen=True)
class DateParts:
    weekday: str
    month: int
    day: int
    year: int | None
    raw: str


def _month_number(month_name: str) -> int:
    return MONTHS.index(month_name) + 1


def parse_date_parts(raw: str, *, full_line: bool = False) -> DateParts | None:
    match = DATE_LINE_RE.fullmatch(raw) if full_line else DATE_DESCRIPTOR_RE.search(raw)
    if not match:
        return None
    values = match.groupdict()
    return DateParts(
        weekday=values["weekday"],
        month=_month_number(values["month"]),
        day=int(values["day"]),
        year=int(values["year"]) if values["year"] else None,
        raw=match.group(0).strip(),
    )


def _date_from_parts(parts: DateParts, year: int) -> date:
    try:
        return datetime(year, parts.month, parts.day).date()
    except ValueError as exc:
        raise ValueError(f"Invalid source date: {parts.raw!r} for year {year}") from exc


def season_label_from_years(start_year: int, end_year: int) -> str:
    return f"{start_year:04d}-{end_year % 100:02d}"


def _season_years(season_label: str) -> tuple[int, int]:
    match = re.match(r"^(?P<start>\d{4})-(?P<end>\d{2,4})$", season_label)
    if not match:
        raise ValueError(f"Invalid season label: {season_label!r}")
    start = int(match.group("start"))
    end_raw = match.group("end")
    end = int(end_raw) if len(end_raw) == 4 else (start // 100) * 100 + int(end_raw)
    if end < start:
        end += 100
    return start, end


def resolve_date(parts: DateParts, header: SeasonHeader) -> date:
    """Resolve a yearless match date from the parsed season range.

    The source header is authoritative when available. When a header omits its
    date range, the season label supplies the two candidate years; that is a
    deterministic fallback rather than a hard-coded Premier League calendar.
    """

    if parts.year is not None:
        return _date_from_parts(parts, parts.year)

    label_start, label_end = _season_years(header.season_label)
    start_year = header.start_date.year if header.start_date else label_start
    end_year = header.end_date.year if header.end_date else label_end
    candidates: list[date] = []
    for year in range(start_year, end_year + 1):
        try:
            candidate = _date_from_parts(parts, year)
        except ValueError:
            continue
        if header.start_date and candidate < header.start_date:
            continue
        if header.end_date and candidate > header.end_date:
            continue
        candidates.append(candidate)
    if len(candidates) == 1:
        return candidates[0]

    # If the header range is incomplete, use the conventional two-year
    # ordering derived from the season label and let the caller record any
    # ambiguity as a parsing issue.
    preferred_year = label_start if parts.month >= 7 else label_end
    return _date_from_parts(parts, preferred_year)


def parse_header(text: str) -> SeasonHeader:
    lines = text.splitlines(keepends=True)
    competition_line = next(
        (line for line in lines if line.lstrip().startswith("=")), None
    )
    if not competition_line:
        raise ValueError("Premier League header line not found")

    season_match = SEASON_RE.search(competition_line)
    if not season_match:
        raise ValueError(f"Season label not found in header: {competition_line.strip()!r}")
    start_year = int(season_match.group("start"))
    end_raw = season_match.group("end")
    end_year = int(end_raw) if len(end_raw) == 4 else (start_year // 100) * 100 + int(end_raw)
    if end_year < start_year:
        end_year += 100

    header_value = competition_line.lstrip()[1:].strip()
    if "|" in header_value:
        country, competition_value = [part.strip() for part in header_value.split("|", 1)]
        country = country or "England"
    else:
        country = "England"
        competition_value = header_value
    competition_name = "Premier League" if "Premier League" in competition_value else competition_value

    date_header = next(
        (line for line in lines if re.match(r"^\s*#\s*Dates?\b", line, re.IGNORECASE)),
        None,
    )
    date_parts = list(DATE_DESCRIPTOR_RE.finditer(date_header or ""))
    start_date = end_date = None
    if len(date_parts) >= 2:
        parsed_start = parse_date_parts(date_parts[0].group(0))
        parsed_end = parse_date_parts(date_parts[1].group(0))
        if parsed_start and parsed_end:
            start_date = _date_from_parts(parsed_start, parsed_start.year or start_year)
            end_date = _date_from_parts(parsed_end, parsed_end.year or end_year)

    expected_team_count = _header_int(lines, "Teams")
    expected_match_count = _header_int(lines, "Matches")
    first_round = next(
        (index for index, line in enumerate(lines) if is_round_heading(line)), len(lines)
    )
    return SeasonHeader(
        season_label=season_label_from_years(start_year, end_year),
        competition_name=competition_name,
        country=country,
        start_date=start_date,
        end_date=end_date,
        expected_team_count=expected_team_count,
        expected_match_count=expected_match_count,
        source_header_text="".join(lines[:first_round]),
    )


def _header_int(lines: Iterable[str], field: str) -> int | None:
    pattern = re.compile(rf"^\s*#\s*{re.escape(field)}\s+(\d+)\b", re.IGNORECASE)
    for line in lines:
        match = pattern.match(line)
        if match:
            return int(match.group(1))
    return None


def is_round_heading(line: str) -> bool:
    return bool(re.match(r"^\s*▪\s*.+$", line))
