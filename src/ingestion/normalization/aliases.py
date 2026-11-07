"""Explicit identity mappings.

No fuzzy matching happens here. A mapping is only applied when it is present
in the supplied JSON file, so historical name decisions remain reviewable.
"""

from __future__ import annotations

import json
from dataclasses import dataclass, field
from pathlib import Path
from typing import Literal

from .names import clean_display_name, normalize_name


EntityType = Literal["team", "player", "referee"]


@dataclass
class AliasMaps:
    teams: dict[str, str] = field(default_factory=dict)
    players: dict[str, str] = field(default_factory=dict)
    referees: dict[str, str] = field(default_factory=dict)
    sources: dict[tuple[EntityType, str], str] = field(default_factory=dict)

    def target(self, entity_type: EntityType, source_value: str) -> str:
        key = normalize_name(source_value)
        mapping = getattr(self, f"{entity_type}s")
        return mapping.get(key, clean_display_name(source_value))

    def mapping_source(self, entity_type: EntityType, source_value: str) -> str | None:
        return self.sources.get((entity_type, normalize_name(source_value)))


def load_alias_maps(path: str | Path | None) -> AliasMaps:
    """Load an explicit alias file.

    Accepted JSON shape::

        {"teams": {"AFC Bournemouth": "Bournemouth"},
         "players": {}, "referees": {}}

    For convenience, a top-level `team`/`player`/`referee` mapping is also
    accepted. Values are canonical display names; keys are source spellings.
    """

    if path is None:
        return AliasMaps()

    alias_path = Path(path)
    if not alias_path.exists():
        raise FileNotFoundError(f"Alias file not found: {alias_path}")

    payload = json.loads(alias_path.read_text(encoding="utf-8"))
    result = AliasMaps()
    for entity_type in ("team", "player", "referee"):
        plural = f"{entity_type}s"
        if plural in payload or entity_type in payload:
            configured = payload.get(plural, payload.get(entity_type, {}))
        elif entity_type == "team":
            # The original project config is a flat object of source team
            # spellings to canonical names. Treat that shape as team aliases
            # while keeping player/referee mappings opt-in and explicit.
            configured = payload
        else:
            configured = {}
        if not isinstance(configured, dict):
            raise ValueError(f"{plural} aliases must be a JSON object")
        mapping: dict[str, str] = getattr(result, plural)
        for source_value, canonical_value in configured.items():
            if not isinstance(source_value, str) or not isinstance(canonical_value, str):
                raise ValueError(f"{plural} alias keys and values must be strings")
            mapping[normalize_name(source_value)] = clean_display_name(canonical_value)
            result.sources[(entity_type, normalize_name(source_value))] = str(alias_path)
    return result
