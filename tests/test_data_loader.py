import pandas as pd

from src.data.loader import DataLoader


def test_sanitize_matches_removes_empty_and_duplicate_rows():
    frame = pd.DataFrame(
        [
            {
                "MatchID": 1,
                "Year": 2022,
                "Home Team Name": "Brazil",
                "Away Team Name": "Germany",
                "Home Team Goals": 2,
                "Away Team Goals": 0,
            },
            {
                "MatchID": 1,
                "Year": 2022,
                "Home Team Name": "Brazil",
                "Away Team Name": "Germany",
                "Home Team Goals": 2,
                "Away Team Goals": 0,
            },
            {
                "MatchID": None,
                "Year": None,
                "Home Team Name": None,
                "Away Team Name": None,
                "Home Team Goals": None,
                "Away Team Goals": None,
            },
        ]
    )

    clean = DataLoader._sanitize_matches(frame)

    assert len(clean) == 1
    assert clean.iloc[0]["Home Team Name"] == "Brazil"


def test_sanitize_matches_repairs_embedded_html_fragment():
    frame = pd.DataFrame(
        [
            {
                "MatchID": 1,
                "Year": 2002,
                "Home Team Name": 'rn">Republic of Ireland',
                "Away Team Name": "Germany",
                "Home Team Goals": 1,
                "Away Team Goals": 1,
            }
        ]
    )

    clean = DataLoader._sanitize_matches(frame)
    assert clean.iloc[0]["Home Team Name"] == "Republic of Ireland"
