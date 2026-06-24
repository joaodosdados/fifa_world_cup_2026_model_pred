"""Generate model predictions for the current World Cup schedule."""

from __future__ import annotations

from pathlib import Path
from typing import Any

import pandas as pd

from src.models.schedule_predictions import precompute_schedule_predictions


def generate_schedule_predictions(
    predictor: Any,
    schedule: pd.DataFrame,
    output_path: str | Path | None = None,
) -> pd.DataFrame:
    """Generate predictions and optionally persist them as CSV."""
    predictions = precompute_schedule_predictions(predictor, schedule)
    if output_path is not None:
        output_path = Path(output_path)
        output_path.parent.mkdir(parents=True, exist_ok=True)
        predictions.to_csv(output_path, index=False)
    return predictions
