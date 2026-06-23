import pytest

from src.data.loader import DataLoader
from src.models.model_catalog import (
    build_model_catalog,
    default_model_id,
    rank_model_ids,
)


@pytest.fixture(scope="module")
def catalog():
    matches = DataLoader().load_international_matches(
        min_year=2010,
        max_date="2026-06-10",
        min_team_matches=30,
    )
    return build_model_catalog(matches)


def test_catalog_contains_ml_and_statistical_models(catalog):
    assert "elo_poisson" in catalog
    assert any(item["kind"] == "machine_learning" for item in catalog.values())
    assert default_model_id(catalog) in catalog


def test_every_available_model_returns_normalized_probabilities(catalog):
    for item in catalog.values():
        prediction = item["predictor"].predict_match("Brazil", "Germany")
        total = sum(
            prediction[key] for key in ("home_win", "draw", "away_win")
        )
        assert total == pytest.approx(1.0)
        assert prediction["expected_goals_home"] >= 0
        assert prediction["expected_goals_away"] >= 0
        assert len(prediction["most_likely_score"]) == 2
        assert all(
            isinstance(goals, int)
            for goals in prediction["most_likely_score"]
        )


def test_model_change_produces_model_specific_predictions(catalog):
    signatures = set()
    for item in catalog.values():
        prediction = item["predictor"].predict_match("Brazil", "Germany")
        signatures.add(
            tuple(
                round(prediction[key], 5)
                for key in ("home_win", "draw", "away_win")
            )
        )

    assert len(signatures) > 1


def test_display_and_historical_team_names_produce_same_prediction(catalog):
    for item in catalog.values():
        localized = item["predictor"].predict_match("Brasil", "Alemanha")
        historical = item["predictor"].predict_match("Brazil", "Germany")

        assert localized["home_win"] == pytest.approx(historical["home_win"])
        assert localized["draw"] == pytest.approx(historical["draw"])
        assert localized["away_win"] == pytest.approx(historical["away_win"])


def test_ml_models_use_dedicated_goal_predictions(catalog):
    for item in catalog.values():
        if item["kind"] != "machine_learning":
            continue

        prediction = item["predictor"].predict_match("Brazil", "Germany")
        home_goals, away_goals = prediction["most_likely_score"]

        assert prediction["goal_model_name"] != "historical_fallback"
        assert prediction["expected_goals_home"] >= 0.05
        assert prediction["expected_goals_away"] >= 0.05
        assert 0 <= home_goals <= 8
        assert 0 <= away_goals <= 8


def test_model_rank_orders_highest_accuracy_first(catalog):
    accuracy = {
        model_id: float(index)
        for index, model_id in enumerate(catalog)
    }
    ranked = rank_model_ids(catalog, accuracy)
    assert accuracy[ranked[0]] == max(accuracy.values())
