"""Match metadata parsing, including per-match UTC offsets."""

from __future__ import annotations

import re
from datetime import datetime, time, timezone, timedelta

from src.ingestion.football_txt.header_parser import DateParts, parse_date_parts, resolve_date
from src.models import SeasonHeader


FULL_METADATA_RE = re.compile(
    r"^\s*(?P<date>(?:Mon|Tue|Wed|Thu|Fri|Sat|Sun)\s+"
    r"(?:Jan|Feb|Mar|Apr|May|Jun|Jul|Aug|Sep|Oct|Nov|Dec)\s+\d{1,2}"
    r"(?:\s+\d{4})?)\s+"
    r"(?P<time>\d{1,2}:\d{2})\s+"
    r"(?P<offset>UTC[+-]\d{1,2}(?::?\d{2})?)\s+@\s+"
    r"(?P<venue_location>.*?)\s*$"
)
ATTENDANCE_RE = re.compile(r",\s*Att:\s*(?P<attendance>[\d,]+)\s*$", re.IGNORECASE)
UTC_OFFSET_RE = re.compile(r"^UTC(?P<sign>[+-])(?P<hour>\d{1,2})(?::?(?P<minute>\d{2}))?$", re.IGNORECASE)


def parse_utc_offset(raw: str) -> int:
    match = UTC_OFFSET_RE.fullmatch(raw.strip())
    if not match:
        raise ValueError(f"Invalid UTC offset: {raw!r}")
    minutes = int(match.group("hour")) * 60 + int(match.group("minute") or 0)
    return minutes if match.group("sign") == "+" else -minutes


def parse_clock(raw: str) -> time:
    hour, minute = (int(part) for part in raw.strip().split(":", 1))
    if hour > 23 or minute > 59:
        raise ValueError(f"Invalid kickoff time: {raw!r}")
    return time(hour=hour, minute=minute)


def split_venue_location(raw: str) -> tuple[str | None, str | None, int | None]:
    """Split `Venue, City, Region, Att: 123` without losing location commas."""

    attendance_match = ATTENDANCE_RE.search(raw)
    attendance = None
    without_attendance = raw
    if attendance_match:
        attendance = int(attendance_match.group("attendance").replace(",", ""))
        without_attendance = raw[: attendance_match.start()].rstrip()
    if "," in without_attendance:
        venue, location = without_attendance.split(",", 1)
        return venue.strip() or None, location.strip() or None, attendance
    return without_attendance.strip() or None, None, attendance


def parse_full_metadata_line(
    line: str,
    header: SeasonHeader,
) -> dict[str, object] | None:
    """Parse a rich-file metadata line, or return `None` for other lines."""

    match = FULL_METADATA_RE.fullmatch(line.rstrip("\r\n"))
    if not match:
        return None
    values = match.groupdict()
    date_parts = parse_date_parts(values["date"], full_line=True)
    if date_parts is None:
        raise ValueError(f"Invalid metadata date: {values['date']!r}")
    match_date = resolve_date(date_parts, header)
    kickoff_local_time = parse_clock(values["time"])
    utc_offset_minutes = parse_utc_offset(values["offset"])
    venue, location, attendance = split_venue_location(values["venue_location"])
    local_dt = datetime.combine(match_date, kickoff_local_time)
    kickoff_utc = (local_dt - timedelta(minutes=utc_offset_minutes)).replace(tzinfo=timezone.utc)
    return {
        "match_date": match_date,
        "source_weekday": date_parts.weekday,
        "kickoff_local_time": kickoff_local_time,
        "utc_offset_minutes": utc_offset_minutes,
        "kickoff_utc": kickoff_utc,
        "venue": venue,
        "location": location,
        "attendance": attendance,
        "raw_metadata": line.rstrip("\r\n"),
    }


def parse_compact_date_line(line: str, header: SeasonHeader) -> dict[str, object] | None:
    parts = parse_date_parts(line.rstrip("\r\n"), full_line=True)
    if parts is None:
        return None
    return {
        "match_date": resolve_date(parts, header),
        "source_weekday": parts.weekday,
        "raw_date": line.rstrip("\r\n"),
    }

