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
