"""Natural-language question parsing: teams, intent, relative dates, players.

The parser is deterministic and explainable. Relative dates are resolved in
the user's timezone (default Asia/Hong_Kong). Because English evening kickoffs
fall in the early hours in East Asia, a local "night" runs from 17:00 to
06:00 the next morning; the resolved window is always echoed back.
"""
from __future__ import annotations

import re
from dataclasses import dataclass, field
from datetime import date, datetime, time, timedelta, timezone
from zoneinfo import ZoneInfo

from src.matchsense.teams import REGISTRY, TeamRegistry, fold

NIGHT_START = time(17, 0)
NIGHT_END = time(6, 0)
WEEKDAYS = ("monday", "tuesday", "wednesday", "thursday", "friday", "saturday", "sunday")
MONTHS = {m: i for i, m in enumerate(
    ("january", "february", "march", "april", "may", "june", "july", "august",
     "september", "october", "november", "december"), 1)}
MONTHS.update({m[:3]: i for m, i in list(MONTHS.items())})
MONTHS["sept"] = 9


@dataclass
class TimeWindow:
    label: str
    start_utc: datetime | None  # None => open-ended "next fixture"
    end_utc: datetime | None
    timezone: str

    def contains(self, moment: datetime) -> bool:
        if self.start_utc is None or self.end_utc is None:
            return True
        return self.start_utc <= moment < self.end_utc

    def as_dict(self) -> dict:
        tz = ZoneInfo(self.timezone)
        fmt = "%a %d %b %Y %H:%M"
        return {"label": self.label, "timezone": self.timezone,
                "start_utc": self.start_utc.isoformat() if self.start_utc else None,
                "end_utc": self.end_utc.isoformat() if self.end_utc else None,
                "start_local": self.start_utc.astimezone(tz).strftime(fmt) if self.start_utc else None,
                "end_local": self.end_utc.astimezone(tz).strftime(fmt) if self.end_utc else None}


@dataclass
class ParsedQuery:
    text: str
    teams: list[str]
    subject: str | None
    opponent: str | None
    intent: str  # "outcome" | "next" | "what_if"
    window: TimeWindow
    explicit_date: bool
    what_if_players: list[str] = field(default_factory=list)
    what_if_kind: str | None = None  # "unavailable" | "available"
    ambiguities: list[str] = field(default_factory=list)


def _local(day: date, at: time, tz: ZoneInfo) -> datetime:
    return datetime.combine(day, at, tzinfo=tz).astimezone(timezone.utc)


def _night(day: date, tz: ZoneInfo, label: str) -> TimeWindow:
    return TimeWindow(label, _local(day, NIGHT_START, tz), _local(day + timedelta(days=1), NIGHT_END, tz), tz.key)


def _day(day: date, tz: ZoneInfo, label: str) -> TimeWindow:
    # Includes the following early morning so a 20:00 UK kickoff (04:00 HKT)
    # still counts as "Saturday's" match for an East-Asian user.
    return TimeWindow(label, _local(day, time(0, 0), tz), _local(day + timedelta(days=1), NIGHT_END, tz), tz.key)


def resolve_window(text: str, now_utc: datetime, tz_name: str) -> tuple[TimeWindow, bool]:
    tz = ZoneInfo(tz_name)
    local_now = now_utc.astimezone(tz)
    today = local_now.date()
    q = fold(text)

    match = re.search(r"\b(\d{4})-(\d{2})-(\d{2})\b", text)
    if match:
        day = date(int(match[1]), int(match[2]), int(match[3]))
        return _day(day, tz, day.isoformat()), True
    month_names = "|".join(sorted(MONTHS, key=len, reverse=True))
    match = (re.search(rf"\b(\d{{1,2}})(?:st|nd|rd|th)? (?:of )?({month_names})\b", q)
             or re.search(rf"\b({month_names}) (\d{{1,2}})(?:st|nd|rd|th)?\b", q))
    if match:
        a, b = match.groups()
        day_num, month = (int(a), MONTHS[b]) if a.isdigit() else (int(b), MONTHS[a])
        candidate = date(today.year, month, day_num)
        if candidate < today - timedelta(days=60):
            candidate = date(today.year + 1, month, day_num)
        return _day(candidate, tz, candidate.isoformat()), True
    match = re.search(r"\b(\d{1,2})/(\d{1,2})(?:/(\d{2,4}))?\b", text)
    if match:  # Day-first, as used in Hong Kong and the UK.
        year = int(match[3]) if match[3] else today.year
        year += 2000 if year < 100 else 0
        day = date(year, int(match[2]), int(match[1]))
        return _day(day, tz, day.isoformat()), True

    if re.search(r"\b(tonight|this evening|tonite)\b", q):
        if local_now.time() < NIGHT_END:
            return _night(today - timedelta(days=1), tz, "tonight"), True
        return _night(today, tz, "tonight"), True
    if re.search(r"\btomorrow (night|evening)\b", q):
        return _night(today + timedelta(days=1), tz, "tomorrow night"), True
    if re.search(r"\btomorrow\b", q):
        return _day(today + timedelta(days=1), tz, "tomorrow"), True
    if re.search(r"\btoday\b", q):
        return _day(today, tz, "today"), True
    if re.search(r"\b(this|the) weekend\b", q):
        offset = (4 - today.weekday()) % 7  # Friday
        if today.weekday() in (5, 6):  # Already the weekend.
            offset = -(today.weekday() - 4)
        friday = today + timedelta(days=offset)
        return TimeWindow("this weekend", _local(friday, NIGHT_START, tz),
                          _local(friday + timedelta(days=3), NIGHT_END, tz), tz.key), True
    for index, name in enumerate(WEEKDAYS):
        match = re.search(rf"\b(?:on |this |next )?{name}( night| evening)?\b", q)
        if match:
            offset = (index - today.weekday()) % 7
            if "next " + name in q and offset == 0:
                offset = 7
            day = today + timedelta(days=offset)
            label = name.capitalize() + (" night" if match[1] else "")
            return (_night(day, tz, label) if match[1] else _day(day, tz, label)), True
    return TimeWindow("next fixture", None, None, tz.key), False


_ABSENT = (r"(?:unavailable|out|injured|missing|absent|suspended|banned|ruled out|benched|dropped|rested|"
           r"not playing|not starting|not available|not fit|unfit)")
_NOT_PLAY = r"(?:doesn'?t|does not|won'?t|will not|can'?t|cannot|isn'?t|is not|aren'?t|are not|didn'?t)\s+(?:play|start|feature|make it|available|fit)"
_WHAT_IF_PATTERNS = (
    (rf"(?:how (?:does|would|will)|what if|what happens if|impact of|effect of)\s+(.+?)\s+(?:being|is|was|were|are|gets?)?\s*{_ABSENT}", "unavailable"),
    (rf"\bif\s+(.+?)\s+{_NOT_PLAY}", "unavailable"),
    (rf"\bif\s+(.+?)\s+(?:is|are|was|were|gets?|got)\s+{_ABSENT}", "unavailable"),
    (rf"\b(?:what if|how about|and if)\s+(.+?)\s+{_NOT_PLAY}", "unavailable"),
    (r"\bwithout\s+([a-z][a-z' .-]+?)(?:\s+(?:affect|against|vs|v|for|in|tonight|today|playing|will|would)\b|[?,.]|$)", "unavailable"),
    (rf"^\s*([a-z][a-z' .-]+?)\s+(?:is |are )?{_ABSENT}\b", "unavailable"),
    (r"\b(?:what if|if)\s+(.+?)\s+(?:plays|starts|is fit|returns|is available|is back|comes back)", "available"),
)
_FILLER = r"^(?:the|odds|chances|what|are|is)\s+"


def _what_if(text: str, registry: TeamRegistry) -> tuple[list[str], str | None]:
    q = fold(text)
    for pattern, kind in _WHAT_IF_PATTERNS:
        match = re.search(pattern, q)
        if match:
            names = re.split(r"\s+(?:and|&)\s+|,\s*", match[1])
            players = []
            for name in names:
                name = re.sub(r"^(?:the|city'?s|united'?s)\s+", "", name).strip(" '")
                # Remove a team mention ("City's Haaland").
                for mention in sorted(registry.find_mentions(name), key=lambda m: -m.start):
                    name = (name[:mention.start] + name[mention.end:]).strip(" 's")
                name = re.sub(_FILLER, "", name).strip(" '")
                if name and len(name) > 1 and not re.search(r"\b(?:we|they|he|it|odds|chances)\b", name):
                    players.append(name)
            if players:
                return players, kind
    return [], None


def parse_query(text: str, now_utc: datetime, tz_name: str = "Asia/Hong_Kong",
                registry: TeamRegistry = REGISTRY) -> ParsedQuery:
    if not text or not text.strip():
        raise ValueError("Empty question")
    mentions = registry.find_mentions(text)
    teams: list[str] = []
    ambiguities = []
    for mention in mentions:
        if mention.team not in teams:
            teams.append(mention.team)
        if mention.ambiguous_with:
            ambiguities.append(f'"{mention.alias}" was read as {mention.team}; say the full name if you meant '
                               + " or ".join(mention.ambiguous_with) + ".")
    window, explicit = resolve_window(text, now_utc, tz_name)
    players, kind = _what_if(text, registry)
    q = fold(text)
    intent = "what_if" if players else ("next" if re.search(r"\bnext\b", q) else "outcome")
    if intent == "next" and explicit and window.label.startswith("next"):
        explicit = False
    subject = teams[0] if teams else None
    opponent = teams[1] if len(teams) > 1 else None
    return ParsedQuery(text=text, teams=teams, subject=subject, opponent=opponent, intent=intent,
                       window=window, explicit_date=explicit, what_if_players=players,
                       what_if_kind=kind, ambiguities=ambiguities)
