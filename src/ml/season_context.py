"""Leakage-safe optional pre-match context features.

The historical openfootball snapshot contains results, but not bookmaker
prices, xG providers, availability, or transfer data.  This sidecar format
lets those signals be supplied later without changing the core match schema.
Every row is keyed to a fixture and may include an ``available_date`` column;
when present, it must not be after kickoff.
"""
from __future__ import annotations

import csv
from dataclasses import dataclass
from datetime import date
import math
from pathlib import Path

import numpy as np


KEY_FIELDS = ('season', 'date', 'home', 'away')
META_FIELDS = (*KEY_FIELDS, 'available_date')
FORBIDDEN_FEATURE_NAMES = {
    'actual', 'actual_outcome', 'home_goals', 'away_goals', 'result', 'score',
}


def fixture_key(match) -> tuple[str, str, str, str]:
    return (str(match.season), match.date.isoformat(), str(match.home), str(match.away))


@dataclass(frozen=True)
class MatchContext:
    """Numeric pre-match features keyed by season/date/home/away."""

    feature_names: tuple[str, ...]
    rows: dict[tuple[str, str, str, str], tuple[float, ...]]
    defaults: tuple[float, ...]

    def values_for(self, fixtures) -> np.ndarray:
        """Return one context vector per fixture, using column means if absent."""
        return np.asarray([self.rows.get(fixture_key(m), self.defaults) for m in fixtures], dtype=float)

    @classmethod
    def from_rows(cls, feature_names, rows) -> 'MatchContext':
        """Build context from validated ``(fixture_key, values)`` pairs."""
        names = tuple(feature_names)
        if not names:
            raise ValueError('At least one context feature is required')
        values_by_key = {}
        for key, raw_values in rows:
            values = tuple(float(value) for value in raw_values)
            if len(values) != len(names) or not all(math.isfinite(value) for value in values):
                raise ValueError(f'Invalid context values for fixture {key}')
            if key in values_by_key:
                raise ValueError(f'Duplicate context fixture key: {key}')
            values_by_key[key] = values
        if not values_by_key:
            raise ValueError('Context rows are empty')
        matrix = np.asarray(list(values_by_key.values()), dtype=float)
        return cls(names, values_by_key, tuple(matrix.mean(axis=0).tolist()))

    @classmethod
    def from_csv(cls, path: str | Path) -> 'MatchContext':
        source = Path(path)
        with source.open(newline='') as handle:
            reader = csv.DictReader(handle)
            if not reader.fieldnames:
                raise ValueError(f'Context CSV has no header: {source}')
            missing = set(KEY_FIELDS) - set(reader.fieldnames)
            if missing:
                raise ValueError(f'Context CSV is missing key columns: {sorted(missing)}')
            feature_names = tuple(name for name in reader.fieldnames if name not in META_FIELDS)
            if not feature_names:
                raise ValueError('Context CSV must contain at least one numeric feature column')
            for name in feature_names:
                if name.lower() in FORBIDDEN_FEATURE_NAMES:
                    raise ValueError(f'Context feature {name!r} is a post-match target')
            rows: dict[tuple[str, str, str, str], tuple[float, ...]] = {}
            sums = np.zeros(len(feature_names), dtype=float)
            counts = np.zeros(len(feature_names), dtype=int)
            for line_number, raw in enumerate(reader, 2):
                try:
                    key = (raw['season'], date.fromisoformat(raw['date']).isoformat(), raw['home'], raw['away'])
                    available = raw.get('available_date')
                    if available and date.fromisoformat(available) > date.fromisoformat(raw['date']):
                        raise ValueError('available_date is after fixture date')
                    values = tuple(float(raw[name]) for name in feature_names)
                except (KeyError, TypeError, ValueError) as exc:
                    raise ValueError(f'Invalid context row {line_number} in {source}: {exc}') from exc
                if not all(math.isfinite(value) for value in values):
                    raise ValueError(f'Context row {line_number} contains a non-finite value')
                if key in rows:
                    raise ValueError(f'Duplicate context fixture key at row {line_number}: {key}')
                rows[key] = values
                sums += values
                counts += 1
            if not rows:
                raise ValueError(f'Context CSV has no data rows: {source}')
            defaults = tuple((sums / np.maximum(counts, 1)).tolist())
        return cls(feature_names, rows, defaults)
