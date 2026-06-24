"""Helpers to precompute and reuse fixture predictions within the app."""

from __future__ import annotations

from typing import Any, Dict

import pandas as pd


PREDICTION_COLUMNS = [
    "home_win",
    "draw",
    "away_win",
    "expected_goals_home",
    "expected_goals_away",
    "most_likely_home_score",
    "most_likely_away_score",
    "modal_home_score",
    "modal_away_score",
    "score_adjusted_to_outcome",
    "predicted_winner",
]


def schedule_fingerprint(schedule: pd.DataFrame) -> str:
    """Return a stable fingerprint for prediction/cache invalidation."""
    relevant_columns = [
        column
        for column in [
            "match_id",
            "home_team",
            "away_team",
            "date",
            "time",
            "home_score",
            "away_score",
            "status",
        ]
        if column in schedule.columns
    ]
    relevant = schedule[relevant_columns].copy().fillna("")
    hashed = pd.util.hash_pandas_object(relevant, index=True).astype("uint64")
    return f"{len(schedule)}-{int(hashed.sum())}"


def winner_from_prediction(
    prediction: Dict[str, Any],
    home_team: str,
    away_team: str,
) -> str:
    """Convert probability keys into the display winner label."""
    predicted_key = max(
        ("home_win", "draw", "away_win"),
        key=lambda key: float(prediction.get(key, 0.0)),
    )
    if predicted_key == "home_win":
        return home_team
    if predicted_key == "away_win":
        return away_team
    return "Draw"


def precompute_schedule_predictions(
    predictor: Any,
    schedule: pd.DataFrame,
) -> pd.DataFrame:
    """
    Predict every fixture once for a model and return rows indexed by match_id.

    Streamlit reruns the script whenever a widget changes; keeping this work in
    one table prevents group selector changes from repeatedly invoking model
    inference for the same matches.
    """
    rows = []
    for index, match in schedule.iterrows():
        match_id = match.get("match_id", index)
        home_team = match["home_team"]
        away_team = match["away_team"]
        try:
            prediction = predictor.predict_match(
                home_team,
                away_team,
                is_home_a=True,
            )
            predicted_home, predicted_away = prediction["most_likely_score"]
            modal_home, modal_away = prediction.get(
                "modal_score",
                prediction["most_likely_score"],
            )
            rows.append({
                "match_id": match_id,
                "home_win": float(prediction["home_win"]),
                "draw": float(prediction["draw"]),
                "away_win": float(prediction["away_win"]),
                "expected_goals_home": float(prediction["expected_goals_home"]),
                "expected_goals_away": float(prediction["expected_goals_away"]),
                "most_likely_home_score": int(predicted_home),
                "most_likely_away_score": int(predicted_away),
                "modal_home_score": int(modal_home),
                "modal_away_score": int(modal_away),
                "score_adjusted_to_outcome": bool(
                    prediction.get("score_adjusted_to_outcome", False)
                ),
                "predicted_winner": winner_from_prediction(
                    prediction,
                    home_team,
                    away_team,
                ),
            })
        except Exception as exc:
            rows.append({
                "match_id": match_id,
                "prediction_error": str(exc),
            })

    predictions = pd.DataFrame(rows)
    if predictions.empty:
        return predictions
    return predictions.set_index("match_id", drop=False)


def prediction_for_match(
    predictions: pd.DataFrame | None,
    match: pd.Series,
) -> Dict[str, Any] | None:
    """Fetch a cached prediction row and convert it to the legacy dict shape."""
    if predictions is None or predictions.empty:
        return None

    match_id = match.get("match_id")
    if match_id not in predictions.index:
        return None

    row = predictions.loc[match_id]
    if isinstance(row, pd.DataFrame):
        row = row.iloc[0]
    if pd.notna(row.get("prediction_error", None)):
        raise RuntimeError(str(row["prediction_error"]))

    return {
        "home_win": float(row["home_win"]),
        "draw": float(row["draw"]),
        "away_win": float(row["away_win"]),
        "expected_goals_home": float(row["expected_goals_home"]),
        "expected_goals_away": float(row["expected_goals_away"]),
        "most_likely_score": (
            int(row["most_likely_home_score"]),
            int(row["most_likely_away_score"]),
        ),
        "modal_score": (
            int(row.get("modal_home_score", row["most_likely_home_score"])),
            int(row.get("modal_away_score", row["most_likely_away_score"])),
        ),
        "score_adjusted_to_outcome": bool(
            row.get("score_adjusted_to_outcome", False)
        ),
        "predicted_winner": str(row["predicted_winner"]),
    }
