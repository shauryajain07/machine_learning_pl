from src.ingestion.football_txt.lineup_parser import parse_lineup, split_lineup_entries


def test_captain_and_nested_substitutions_are_parsed_without_positions():
    appearances, substitutions, issues = parse_lineup(
        "Adam SMITH [c] (10' Álex Jiménez (45' Lewis Cook)), Veljko Milosavljevic - Jean-Clair TODIBO",
        side="home",
    )
    assert not issues
    assert [appearance.player_name for appearance in appearances] == [
        "Adam SMITH",
        "Álex Jiménez",
        "Lewis Cook",
        "Veljko Milosavljevic",
        "Jean-Clair TODIBO",
    ]
    assert appearances[0].captain is True
    assert appearances[0].exited.absolute_minute == 10
    assert appearances[1].entered.absolute_minute == 10
    assert appearances[1].exited.absolute_minute == 45
    assert [(item.player_off_name, item.player_on_name) for item in substitutions] == [
        ("Adam SMITH", "Álex Jiménez"),
        ("Álex Jiménez", "Lewis Cook"),
    ]


def test_hyphenated_names_are_not_treated_as_group_separators():
    entries = split_lineup_entries("Jean-Clair TODIBO, Aaron WAN-BISSAKA - João Silva")
    assert entries == [("Jean-Clair TODIBO", 0), ("Aaron WAN-BISSAKA", 0), ("João Silva", 1)]

