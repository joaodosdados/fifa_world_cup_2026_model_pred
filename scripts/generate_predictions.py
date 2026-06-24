"""Generate schedule prediction CSV for the best available model."""

from __future__ import annotations

import sys
from pathlib import Path

import pandas as pd

PROJECT_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(PROJECT_ROOT))

from src.data.loader import DataLoader
from src.models.model_catalog import build_model_catalog, default_model_id
from src.pipelines.prediction_pipeline import generate_schedule_predictions


def main() -> None:
    """Persist predictions for the default/best ranked training model."""
    schedule = pd.read_csv(PROJECT_ROOT / "data" / "2026_world_cup_schedule.csv")
    matches = DataLoader().load_international_matches(
        min_year=2010,
        max_date="2026-06-10",
        min_team_matches=30,
    )
    catalog = build_model_catalog(matches)
    model_id = default_model_id(catalog)
    output_path = PROJECT_ROOT / "data" / "predictions" / f"{model_id}_schedule_predictions.csv"
    predictions = generate_schedule_predictions(
        catalog[model_id]["predictor"],
        schedule,
        output_path,
    )
    print(f"Generated {len(predictions)} predictions: {output_path}")


if __name__ == "__main__":
    main()
