from pathlib import Path

from src.ingestion.football_txt.file_parser import parse_file
from src.ingestion.normalization.names import normalize_name
from src.ingestion.normalization.aliases import load_alias_maps
from src.ingestion.source import stable_match_fingerprint
from src.ingestion.validation.season_validator import validate_season


FIXTURES = Path(__file__).parent / "fixtures"


def test_full_fixture_reconstructs_rich_match():
    parsed = parse_file(FIXTURES / "sample_full.txt", source_file="fixtures/sample_full.txt")
    report = validate_season(parsed)
    match = parsed.matches[0]
    assert report.parsed_matches == 1
    assert match.home_source_name == "Liverpool"
    assert match.away_source_name == "Bournemouth"
    assert match.match_date.isoformat() == "2025-08-15"
    assert match.utc_offset_minutes == 0
    assert match.venue == "Anfield"
    assert match.location == "Liverpool"
    assert match.attendance == 60315
    assert [(goal.player_name, goal.side) for goal in match.goals] == [
        ("Hugo EKITIKE", "home"),
        ("Cody GAKPO", "home"),
        ("Federico Chiesa", "home"),
        ("MOHAMED SALAH", "home"),
        ("Antoine SEMENYO", "away"),
        ("Antoine SEMENYO", "away"),
    ]
    assert len(match.appearances) == 31
    assert len(match.substitutions) == 9
    assert match.cards[0].card_type == "yellow"
    assert match.referees == ["Anthony Taylor", "Gary Beswick", "Craig Taylor"]
    assert match.raw_match_text.startswith("Fri Aug 15 19:00 UTC+0")
    assert report.goal_count_mismatches == 0


def test_compact_fixture_retains_matchday_and_cross_year_dates():
    parsed = parse_file(FIXTURES / "sample_compact.txt", source_file="fixtures/sample_compact.txt")
    validate_season(parsed)
    assert len(parsed.matches) == 2
    assert parsed.matches[0].matchday == 18
    assert parsed.matches[0].match_date.isoformat() == "2020-12-26"
    assert parsed.matches[1].match_date.isoformat() == "2021-01-02"
    assert parsed.matches[1].home_ht_score is None
    assert parsed.matches[0].appearances == []


def test_normalization_preserves_accents_but_ignores_case_and_captain_marker():
    assert normalize_name("Fábio Carvalho") == normalize_name("FÁBIO CARVALHO [c]")
    assert normalize_name("Joël Piroe") == "joël piroe"
    assert normalize_name("Fer López") == "fer lópez"


def test_flat_team_alias_config_is_loaded():
    aliases = load_alias_maps(Path(__file__).parents[1] / "config" / "team_aliases.json")
    assert aliases.target("team", "Liverpool FC") == "Liverpool"


def test_match_fingerprint_does_not_depend_on_final_score():
    parsed = parse_file(FIXTURES / "sample_compact.txt", source_file="fixtures/sample_compact.txt")
    first = parsed.matches[0]
    before = stable_match_fingerprint(first)
    first.score = type(first.score)(0, 9, None, None, "0-9")
    assert stable_match_fingerprint(first) == before
