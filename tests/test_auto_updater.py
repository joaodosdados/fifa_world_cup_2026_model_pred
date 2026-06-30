import pandas as pd

from src.data.auto_updater import AutoDataUpdater


class FakeScraper:
    def fetch_match_results(self):
        return [
            {
                "team_a": "Haiti",
                "team_b": "Escócia",
                "score_a": 0,
                "score_b": 1,
                "status": "completed",
            }
        ]


class FakeKnockoutScraper:
    def fetch_match_results(self):
        return [
            {
                "team_a": "Brasil",
                "team_b": "Japão",
                "date": "2026-06-29",
                "time": "",
                "venue": "Houston Stadium",
                "stage": "Round of 32",
                "group": "",
                "score_a": 2,
                "score_b": 1,
                "status": "completed",
            }
        ]


def test_updater_matches_reverse_home_away_order(tmp_path):
    schedule_path = tmp_path / "schedule.csv"
    pd.DataFrame(
        [
            {
                "match_id": 1,
                "group": "C",
                "home_team": "Escócia",
                "away_team": "Haiti",
                "date": "2026-06-12",
                "time": "15:00",
                "venue": "Test",
                "stage": "Group Stage",
                "home_score": None,
                "away_score": None,
                "status": "Scheduled",
            }
        ]
    ).to_csv(schedule_path, index=False)

    updater = AutoDataUpdater(schedule_path=str(schedule_path))
    updater.scraper = FakeScraper()
    result = updater.update_from_fifa()
    updated = pd.read_csv(schedule_path)

    assert result["success"] is True
    assert result["matched_matches"] == 1
    assert updated.iloc[0]["home_score"] == 1
    assert updated.iloc[0]["away_score"] == 0
    assert updated.iloc[0]["status"] == "Completed"


def test_updater_appends_new_knockout_fixture(tmp_path):
    schedule_path = tmp_path / "schedule.csv"
    pd.DataFrame(
        [
            {
                "match_id": 72,
                "group": "L",
                "home_team": "Panamá",
                "away_team": "Inglaterra",
                "date": "2026-06-27",
                "time": "18:00",
                "venue": "Test",
                "stage": "Group Stage",
                "home_score": 0,
                "away_score": 2,
                "status": "Completed",
            }
        ]
    ).to_csv(schedule_path, index=False)

    updater = AutoDataUpdater(schedule_path=str(schedule_path))
    updater.scraper = FakeKnockoutScraper()
    result = updater.update_from_fifa()
    updated = pd.read_csv(schedule_path)

    knockout = updated[updated["stage"].eq("Round of 32")].iloc[0]
    assert result["success"] is True
    assert result["appended_matches"] == 1
    assert knockout["match_id"] == 73
    assert knockout["home_team"] == "Brasil"
    assert knockout["away_team"] == "Japão"
    assert knockout["home_score"] == 2
    assert knockout["away_score"] == 1
    assert knockout["status"] == "Completed"
