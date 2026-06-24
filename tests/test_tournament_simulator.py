import pandas as pd

from src.models.schedule_predictions import precompute_schedule_predictions
from src.simulation.tournament_simulator import SimulatedMatch, TournamentSimulator


class FixedPredictor:
    def predict_match(self, home_team, away_team, is_home_a=True):
        return {
            "home_win": 0.55,
            "draw": 0.2,
            "away_win": 0.25,
            "expected_goals_home": 1.6,
            "expected_goals_away": 1.0,
            "most_likely_score": (2, 1),
        }


def synthetic_schedule():
    rows = []
    match_id = 1
    for group_number in range(12):
        group = chr(ord("A") + group_number)
        teams = [f"{group}{index}" for index in range(1, 5)]
        pairings = [(0, 1), (2, 3), (0, 2), (3, 1), (3, 0), (1, 2)]
        for home_idx, away_idx in pairings:
            rows.append(
                {
                    "match_id": match_id,
                    "group": group,
                    "home_team": teams[home_idx],
                    "away_team": teams[away_idx],
                    "date": "2026-06-11",
                    "time": "15:00",
                    "venue": "Test",
                    "stage": "Group Stage",
                    "home_score": None,
                    "away_score": None,
                    "status": "Scheduled",
                }
            )
            match_id += 1
    return pd.DataFrame(rows)


def test_tournament_simulator_returns_probability_tables():
    schedule = synthetic_schedule()
    predictor = FixedPredictor()
    predictions = precompute_schedule_predictions(predictor, schedule)

    result = TournamentSimulator(
        predictor,
        schedule,
        predictions,
        random_seed=7,
    ).simulate(n_simulations=50)

    probabilities = result["probabilities"]

    assert len(probabilities) == 48
    assert probabilities["pass_group"].between(0, 1).all()
    assert probabilities["title"].between(0, 1).all()
    assert probabilities["title"].sum() == 1
    assert not result["group_difficulty"].empty
    assert not result["matchups"].empty


def test_group_ranking_uses_head_to_head_before_overall_goal_difference():
    simulator = TournamentSimulator(
        FixedPredictor(),
        synthetic_schedule(),
        pd.DataFrame(),
        random_seed=1,
    )
    standings = {
        "A": {
            "team": "A",
            "points": 6,
            "played": 3,
            "wins": 2,
            "draws": 0,
            "losses": 1,
            "goals_for": 4,
            "goals_against": 5,
            "goal_difference": -1,
        },
        "B": {
            "team": "B",
            "points": 6,
            "played": 3,
            "wins": 2,
            "draws": 0,
            "losses": 1,
            "goals_for": 7,
            "goals_against": 2,
            "goal_difference": 5,
        },
        "C": {
            "team": "C",
            "points": 3,
            "played": 3,
            "wins": 1,
            "draws": 0,
            "losses": 2,
            "goals_for": 3,
            "goals_against": 4,
            "goal_difference": -1,
        },
        "D": {
            "team": "D",
            "points": 3,
            "played": 3,
            "wins": 1,
            "draws": 0,
            "losses": 2,
            "goals_for": 2,
            "goals_against": 5,
            "goal_difference": -3,
        },
    }
    results = [
        SimulatedMatch("A", "B", 1, 0),
        SimulatedMatch("A", "C", 0, 4),
        SimulatedMatch("A", "D", 3, 1),
        SimulatedMatch("B", "C", 3, 0),
        SimulatedMatch("B", "D", 4, 1),
        SimulatedMatch("C", "D", 1, 0),
    ]

    ranking = simulator._rank_group(standings, results)

    assert ranking[0]["team"] == "A"
    assert ranking[1]["team"] == "B"
