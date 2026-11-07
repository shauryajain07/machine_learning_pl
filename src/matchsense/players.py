"""Player identity resolution across FPL, openfootball, API-Football and news text.

Names arrive as "Erling HAALAND", "E. Haaland", "Haaland", "Rúben Dias" or
"Ruben". Matching is restricted to a club's squad, scored by explicit rules,
and refuses to choose when two players tie (e.g. two "Gomes" in one squad).
An unresolved mention is kept as unresolved evidence instead of guessed.
"""
from __future__ import annotations

import re
import unicodedata
from dataclasses import dataclass
from difflib import SequenceMatcher

from src.matchsense.schemas import SquadPlayer

PARTICLES = {"de", "da", "dos", "das", "van", "von", "der", "den", "di", "le", "la", "el", "al", "bin", "jr", "junior"}


def name_key(text: str) -> str:
    text = unicodedata.normalize("NFKD", text)
    text = "".join(c for c in text if not unicodedata.combining(c))
    text = text.replace("ø", "o").replace("Ø", "o").replace("ł", "l").replace("ß", "ss").replace("đ", "d")
    text = re.sub(r"\[c\]", " ", text, flags=re.I)
    text = re.sub(r"[^A-Za-z0-9 ]+", " ", text.replace("'", "").replace("’", ""))
    return re.sub(r"\s+", " ", text).strip().lower()


def tokens(text: str) -> list[str]:
    return [t for t in name_key(text).split() if t not in PARTICLES]


@dataclass
class Resolution:
    player: SquadPlayer | None
    score: float
    reason: str
    candidates: tuple[str, ...] = ()


def _score(mention: str, player: SquadPlayer) -> tuple[float, str]:
    m_key, m_tokens = name_key(mention), tokens(mention)
    if not m_tokens:
        return 0.0, "empty"
    best = (0.0, "no match")
    for alias in {player.name, player.short_name, *player.aliases}:
        a_key, a_tokens = name_key(alias), tokens(alias)
        if not a_tokens:
            continue
        if m_key == a_key:
            return 1.0, f"exact '{alias}'"
        # "E. Haaland" / "E Haaland": initial + surname.
        if len(m_tokens) >= 2 and len(m_tokens[0]) == 1 and m_tokens[-1] == a_tokens[-1] \
                and a_tokens[0].startswith(m_tokens[0]):
            best = max(best, (0.92, "initial and surname"))
        # Every mention token appears in the alias (surname-only, reordered names).
        if set(m_tokens) <= set(a_tokens):
            best = max(best, (0.9 if m_tokens[-1] == a_tokens[-1] or len(m_tokens) > 1 else 0.85,
                              f"tokens of '{alias}'"))
        elif set(a_tokens) <= set(m_tokens) and len(a_tokens) >= 1:
            best = max(best, (0.86, f"contains '{alias}'"))
        ratio = SequenceMatcher(None, m_key, a_key).ratio()
        if ratio >= 0.88:
            best = max(best, (0.75 * ratio, f"spelling ~'{alias}'"))
    return best


def resolve_player(mention: str, squad: list[SquadPlayer], threshold: float = 0.8) -> Resolution:
    scored = sorted(((_score(mention, p), p) for p in squad), key=lambda x: -x[0][0])
    if not scored or scored[0][0][0] < threshold:
        return Resolution(None, scored[0][0][0] if scored else 0.0, "no squad player matched")
    top_score = scored[0][0][0]
    tied = [p for (s, _), p in scored if abs(s - top_score) < 1e-9]
    if len(tied) > 1:
        return Resolution(None, top_score, "ambiguous", tuple(p.name for p in tied))
    (score, reason), player = scored[0]
    return Resolution(player, score, reason)


def find_player_mentions(text: str, squad: list[SquadPlayer]) -> list[tuple[SquadPlayer, str]]:
    """Locate squad players named in free text (used by the rule extractor)."""
    key = f" {name_key(text)} "
    found = []
    surname_counts: dict[str, int] = {}
    for p in squad:
        surname_counts[tokens(p.short_name)[-1] if tokens(p.short_name) else ""] = \
            surname_counts.get(tokens(p.short_name)[-1] if tokens(p.short_name) else "", 0) + 1
    for p in squad:
        for alias in sorted({p.name, p.short_name, *p.aliases}, key=len, reverse=True):
            alias_key = name_key(alias)
            if len(alias_key) < 3:
                continue
            if f" {alias_key} " in key:
                alias_tokens = tokens(alias)
                if len(alias_tokens) == 1 and surname_counts.get(alias_tokens[0], 0) > 1:
                    continue  # A bare surname shared by two squad members is ambiguous.
                found.append((p, alias))
                break
    return found
