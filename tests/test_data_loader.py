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


def test_completed_2026_results_are_adapted_for_training(tmp_path):
    schedule_path = tmp_path / "schedule.csv"
    pd.DataFrame(
        [
            {
                "match_id": 1,
                "home_team": "Brasil",
                "away_team": "Alemanha",
                "date": "2026-06-20",
                "time": "15:00",
                "home_score": 2,
                "away_score": 1,
                "status": "Completed",
            },
            {
                "match_id": 2,
                "home_team": "França",
                "away_team": "Espanha",
                "date": "2026-06-21",
                "time": "15:00",
                "home_score": None,
                "away_score": None,
                "status": "Scheduled",
            },
        ]
    ).to_csv(schedule_path, index=False)

    current = DataLoader(data_dir=str(tmp_path)).load_current_world_cup_matches(
        schedule_path
    )

    assert len(current) == 1
    assert current.iloc[0]["Year"] == 2026
    assert current.iloc[0]["Home Team Name"] == "Brasil"
    assert current.iloc[0]["Home Team Goals"] == 2
