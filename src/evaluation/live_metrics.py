"""Live evaluation metrics against completed 2026 fixtures."""

from __future__ import annotations

from typing import Any, Dict

import pandas as pd

from src.models.schedule_predictions import precompute_schedule_predictions


def calculate_completed_match_accuracy(
    predictor: Any,
    schedule: pd.DataFrame,
) -> Dict[str, float]:
    """Calculate winner accuracy on completed fixtures."""
    predictions = precompute_schedule_predictions(predictor, schedule)
    completed = schedule[
        schedule["status"].astype(str).str.lower().eq("completed")
    ].copy()
    if completed.empty:
        return {
            "correct": 0,
            "incorrect": 0,
            "total": 0,
            "accuracy": 0.0,
        }

    correct = 0
    evaluated = 0
    for _, match in completed.iterrows():
        match_id = match.get("match_id")
        if match_id not in predictions.index:
            continue

        if match["home_score"] > match["away_score"]:
            actual_winner = match["home_team"]
        elif match["away_score"] > match["home_score"]:
            actual_winner = match["away_team"]
        else:
            actual_winner = "Draw"

        predicted_row = predictions.loc[match_id]
        if isinstance(predicted_row, pd.DataFrame):
            predicted_row = predicted_row.iloc[0]
        correct += predicted_row.get("predicted_winner") == actual_winner
        evaluated += 1

    return {
        "correct": int(correct),
        "incorrect": int(evaluated - correct),
        "total": int(evaluated),
        "accuracy": float(correct / evaluated * 100) if evaluated else 0.0,
    }
