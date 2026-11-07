"""Conservative, Unicode-preserving source-name normalization."""

from __future__ import annotations

import re
import unicodedata


_CAPTAIN_MARKER_RE = re.compile(r"\s*\[c\]\s*", re.IGNORECASE)
_SPACE_RE = re.compile(r"\s+")


def remove_source_annotations(value: str) -> str:
    """Remove only annotations that the Football.TXT format defines.

    In particular, this removes `[c]` for comparison. It does not strip
    accents, punctuation, apostrophes, hyphens, or other name content.
    """

    return _CAPTAIN_MARKER_RE.sub(" ", value)


def normalize_name(value: str) -> str:
    """Return a stable comparison key without changing stored spellings."""

    normalized = unicodedata.normalize("NFKC", value)
    normalized = remove_source_annotations(normalized)
    normalized = _SPACE_RE.sub(" ", normalized.strip())
    return normalized.casefold()


def clean_display_name(value: str) -> str:
    """Remove format markers while preserving the source's Unicode spelling."""

    normalized = unicodedata.normalize("NFKC", value)
    normalized = remove_source_annotations(normalized)
    return _SPACE_RE.sub(" ", normalized.strip())


def normalized_aliases(values: list[str]) -> dict[str, str]:
    """Build a normalized-key lookup while retaining the configured spelling."""

    return {normalize_name(value): value for value in values}

