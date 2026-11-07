from datetime import date

from src.ingestion.football_txt.header_parser import parse_date_parts, parse_header, resolve_date


def test_header_variants_and_expected_values():
    header = parse_header(
        "= England | Premier League 2025/26\n"
        "# Dates    Fri Aug 15 2025 - Sun May 24 2026 (282d)\n"
        "# Teams    20\n# Matches  380\n"
    )
    assert header.season_label == "2025-26"
    assert header.competition_name == "Premier League"
    assert header.country == "England"
    assert header.start_date == date(2025, 8, 15)
    assert header.end_date == date(2026, 5, 24)
    assert header.expected_team_count == 20
    assert header.expected_match_count == 380


def test_yearless_date_resolves_across_december_january_transition():
    header = parse_header(
        "= English Premier League 2020/21\n"
        "# Date Sat Dec 26 2020 - Sat Jan 2 2021 (8d)\n"
    )
    assert resolve_date(parse_date_parts("Sat Dec 26", full_line=True), header) == date(2020, 12, 26)
    assert resolve_date(parse_date_parts("Sat Jan 2", full_line=True), header) == date(2021, 1, 2)

