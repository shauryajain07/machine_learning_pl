from src.ingestion.football_txt.goal_parser import parse_goal_block
from src.ingestion.football_txt.card_parser import parse_card_line
from src.ingestion.football_txt.metadata_parser import parse_full_metadata_line
from src.ingestion.football_txt.minute_parser import parse_minute
from src.ingestion.football_txt.score_parser import parse_match_result_line, parse_score
from src.ingestion.football_txt.header_parser import parse_header


def test_minute_preserves_regulation_and_stoppage_time():
    mark = parse_minute("90+8'")
    assert (mark.minute, mark.stoppage_time, mark.absolute_minute, mark.raw) == (90, 8, 98, "90+8'")


def test_score_variants():
    assert parse_score("4-2 (1-0)").home_ht_score == 1
    assert parse_score("0-0").away_ht_score is None
    parsed = parse_match_result_line("  Liverpool v Bournemouth  4-2 (1-0)")
    assert parsed[0:2] == ("Liverpool", "Bournemouth")


def test_repeated_scorer_minutes_are_two_goals_for_one_player():
    goals, issues = parse_goal_block(
        "(Chris WOOD 5', 45+2'; IGOR THIAGO 78'(p))",
        home_score=2,
        away_score=1,
    )
    assert not issues
    assert [(goal.player_name, goal.minute_mark.absolute_minute) for goal in goals] == [
        ("Chris WOOD", 5),
        ("Chris WOOD", 47),
        ("IGOR THIAGO", 78),
    ]
    assert goals[-1].penalty is True


def test_unseparated_away_only_block_uses_score_for_attribution():
    goals, issues = parse_goal_block("(Riccardo Calafiori 13')", home_score=0, away_score=1)
    assert not issues
    assert goals[0].side == "away"


def test_own_goal_is_preserved_as_annotation():
    goals, _ = parse_goal_block("(RODRIGO MUNIZ 58'(og))", home_score=0, away_score=1)
    assert goals[0].own_goal is True
    assert goals[0].annotation == "og"


def test_source_utc_offsets_are_taken_from_each_metadata_line():
    header = parse_header(
        "= England | Premier League 2025/26\n"
        "# Dates Fri Aug 15 2025 - Sun May 24 2026 (282d)\n"
    )
    utc_zero = parse_full_metadata_line(
        "Fri Aug 15 19:00 UTC+0 @ Anfield, Liverpool, Att: 60315", header
    )
    utc_one = parse_full_metadata_line(
        "Sat Aug 16 12:30 UTC+1 @ Villa Park, Birmingham, Att: 42526", header
    )
    assert utc_zero["utc_offset_minutes"] == 0
    assert utc_one["utc_offset_minutes"] == 60
    assert utc_zero["kickoff_utc"].hour == 19
    assert utc_one["kickoff_utc"].hour == 11


def test_explicit_yellow_red_card_label_is_not_reclassified():
    cards, issues = parse_card_line(
        "Yellow-Red: Ladislav KREJCI 61'", side="away", card_type="yellow_red"
    )
    assert not issues
    assert cards[0].card_type == "yellow_red"
