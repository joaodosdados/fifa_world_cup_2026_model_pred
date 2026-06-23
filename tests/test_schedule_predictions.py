import pandas as pd

from src.models.schedule_predictions import (
    prediction_for_match,
    precompute_schedule_predictions,
    schedule_fingerprint,
)


class DummyPredictor:
    def __init__(self):
        self.calls = 0

    def predict_match(self, home_team, away_team, is_home_a=True):
        self.calls += 1
        return {
            "home_win": 0.6,
            "draw": 0.2,
            "away_win": 0.2,
            "expected_goals_home": 1.8,
            "expected_goals_away": 0.9,
            "most_likely_score": (2, 1),
        }


def sample_schedule():
    return pd.DataFrame(
        [
            {
                "match_id": 1,
                "home_team": "Brasil",
                "away_team": "Alemanha",
                "date": "2026-06-20",
                "time": "15:00",
                "home_score": None,
                "away_score": None,
                "status": "Scheduled",
            },
            {
                "match_id": 2,
                "home_team": "França",
                "away_team": "Espanha",
                "date": "2026-06-21",
                "time": "15:00",
                "home_score": 1,
                "away_score": 1,
                "status": "Completed",
            },
        ]
    )


def test_precomputes_one_prediction_per_fixture():
    predictor = DummyPredictor()

    predictions = precompute_schedule_predictions(predictor, sample_schedule())

    assert predictor.calls == 2
    assert predictions.loc[1, "predicted_winner"] == "Brasil"

    cached = prediction_for_match(predictions, sample_schedule().iloc[0])
    assert cached["home_win"] == 0.6
    assert cached["most_likely_score"] == (2, 1)


def test_schedule_fingerprint_changes_when_result_changes():
    original = sample_schedule()
    changed = sample_schedule()
    changed.loc[1, "home_score"] = 2

    assert schedule_fingerprint(original) != schedule_fingerprint(changed)
