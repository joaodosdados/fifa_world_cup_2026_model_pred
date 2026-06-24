import pandas as pd
import pytest

from src.features.match_features import (
    FEATURE_NAMES,
    build_temporal_training_data,
    calculate_team_stats,
)


def sample_matches():
    return pd.DataFrame(
        [
            {
                "Year": 2000,
                "MatchID": 1,
                "Home Team Name": "Brazil",
                "Away Team Name": "Germany",
                "Home Team Goals": 2,
                "Away Team Goals": 0,
            },
            {
                "Year": 2004,
                "MatchID": 2,
                "Home Team Name": "Germany",
                "Away Team Name": "Brazil",
                "Home Team Goals": 1,
                "Away Team Goals": 1,
            },
        ]
    )


def test_temporal_features_use_only_previous_matches():
    X, y, context = build_temporal_training_data(sample_matches())

    assert X.shape == (2, len(FEATURE_NAMES))
    assert X[0, 0] == pytest.approx(1.3)
    assert X[0, 7] == pytest.approx(1.3)
    assert X[1, 0] == pytest.approx(0.0)
    assert X[1, 7] == pytest.approx(2.0)
    assert X[1, FEATURE_NAMES.index("home_recent_goal_diff")] == pytest.approx(-2.0)
    assert X[1, FEATURE_NAMES.index("away_recent_goal_diff")] == pytest.approx(2.0)
    assert X[1, FEATURE_NAMES.index("elo_diff")] < 0
    assert y.tolist() == [2, 1]
    assert context["year"].tolist() == [2000, 2004]


def test_future_result_does_not_change_past_feature_row():
    original = sample_matches()
    changed = original.copy()
    changed.loc[1, "Home Team Goals"] = 9

    original_X, _, _ = build_temporal_training_data(original)
    changed_X, _, _ = build_temporal_training_data(changed)

    assert original_X[0].tolist() == changed_X[0].tolist()


def test_runtime_team_stats_use_all_sanitized_history():
    stats = calculate_team_stats(sample_matches())

    assert stats["Brazil"]["goals_scored"] == pytest.approx(1.5)
    assert stats["Germany"]["goals_conceded"] == pytest.approx(1.5)
