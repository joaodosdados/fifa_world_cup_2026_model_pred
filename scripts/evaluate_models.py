"""Evaluate available models against completed 2026 fixtures."""

from __future__ import annotations

import json
import sys
from pathlib import Path

import pandas as pd

PROJECT_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(PROJECT_ROOT))

from src.data.loader import DataLoader
from src.evaluation.live_metrics import calculate_completed_match_accuracy
from src.models.model_catalog import build_model_catalog


def main() -> None:
    """Print live winner accuracy for every available model."""
    schedule = pd.read_csv(PROJECT_ROOT / "data" / "2026_world_cup_schedule.csv")
    matches = DataLoader().load_international_matches(
        min_year=2010,
        max_date="2026-06-10",
        min_team_matches=30,
    )
    catalog = build_model_catalog(matches)

    rows = []
    for model_id, item in catalog.items():
        metrics = calculate_completed_match_accuracy(item["predictor"], schedule)
        rows.append({
            "model_id": model_id,
            "label": item["label"],
            **metrics,
        })

    rows = sorted(rows, key=lambda row: row["accuracy"], reverse=True)
    print(json.dumps(rows, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
