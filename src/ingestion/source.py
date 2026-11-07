"""Stable source metadata and match fingerprints."""

from __future__ import annotations

import hashlib
import subprocess
from datetime import time
from pathlib import Path

from src.ingestion.normalization.names import normalize_name
from src.models import ParsedMatch


def repository_commit_sha(root: str | Path) -> str | None:
    root_path = Path(root)
    try:
        completed = subprocess.run(
            ["git", "-C", str(root_path), "rev-parse", "HEAD"],
            check=True,
            capture_output=True,
            text=True,
        )
    except (OSError, subprocess.CalledProcessError):
        return None
    value = completed.stdout.strip()
    return value or None


def stable_match_fingerprint(match: ParsedMatch) -> str:
    """Hash stable source identity fields; final score is deliberately absent."""

    date_value = match.match_date.isoformat() if match.match_date else ""
    time_value = match.kickoff_local_time.isoformat() if isinstance(match.kickoff_local_time, time) else ""
    home_value = normalize_name(match.home_source_name or "")
    away_value = normalize_name(match.away_source_name or "")
    identity = "\x1f".join(
        (match.season_label, date_value, time_value, home_value, away_value)
    )
    if not match.match_date or not match.home_source_name or not match.away_source_name:
        identity += f"\x1f{match.source_file}\x1f{match.source_match_index}"
    return hashlib.sha256(identity.encode("utf-8")).hexdigest()

