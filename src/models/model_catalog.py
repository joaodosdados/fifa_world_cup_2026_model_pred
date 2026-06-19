"""Runtime catalog that keeps model selection, prediction and metadata aligned."""

from __future__ import annotations

import logging
from typing import Any, Dict

import pandas as pd

from src.models.ensemble_predictor import EnsemblePredictor
from src.models.model_manager import ModelManager
from src.models.sklearn_adapter import SklearnMatchPredictor


STATISTICAL_MODEL_ID = "elo_poisson"
logger = logging.getLogger(__name__)


def _display_name(model_id: str, model: Any) -> str:
    names = {
        "ensemble_top3": "ML · Ensemble Top 3",
        "svm": "ML · SVM",
        "naive_bayes": "ML · Naive Bayes",
        "logistic_regression": "ML · Logistic Regression",
        "gradient_boosting": "ML · Gradient Boosting",
        "random_forest": "ML · Random Forest",
        "k-nearest_neighbors": "ML · K-Nearest Neighbors",
    }
    return names.get(model_id, f"ML · {type(model).__name__}")


def build_model_catalog(matches: pd.DataFrame) -> Dict[str, Dict[str, Any]]:
    """Load all persisted ML models plus the statistical ELO/Poisson model."""
    manager = ModelManager()
    catalog: Dict[str, Dict[str, Any]] = {}

    for model_id, estimator in manager.get_available_models().items():
        metadata = manager.metadata.get(model_id, {})
        predictor = SklearnMatchPredictor(estimator, matches)
        try:
            sample = matches.iloc[0]
            predictor.predict_match(
                sample["Home Team Name"],
                sample["Away Team Name"],
            )
        except Exception as exc:
            logger.warning("Skipping unusable model %s: %s", model_id, exc)
            continue

        catalog[model_id] = {
            "id": model_id,
            "label": _display_name(model_id, estimator),
            "kind": "machine_learning",
            "predictor": predictor,
            "estimator": estimator,
            "metrics": metadata.get("metrics", {}),
            "description": (
                metadata.get("description")
                or metadata.get("metrics", {}).get("description")
                or "Modelo scikit-learn treinado com dados históricos."
            ),
        }

    statistical = EnsemblePredictor()
    statistical.train(matches)
    catalog[STATISTICAL_MODEL_ID] = {
        "id": STATISTICAL_MODEL_ID,
        "label": "Estatístico · ELO + Poisson",
        "kind": "statistical",
        "predictor": statistical,
        "estimator": statistical,
        "metrics": {
            "elo_weight": statistical.elo_weight,
            "poisson_weight": statistical.poisson_weight,
            "teams": len(statistical.poisson_model.team_stats),
        },
        "description": (
            "Combina força relativa por ELO com distribuição de gols por Poisson."
        ),
    }

    return catalog


def default_model_id(catalog: Dict[str, Dict[str, Any]]) -> str:
    """Select the persisted model with the best recorded accuracy."""
    candidates = [
        item
        for item in catalog.values()
        if item["kind"] == "machine_learning"
    ]
    if not candidates:
        return STATISTICAL_MODEL_ID
    return max(
        candidates,
        key=lambda item: item["metrics"].get("accuracy", -1),
    )["id"]


def rank_model_ids(
    catalog: Dict[str, Dict[str, Any]],
    live_accuracy: Dict[str, float],
) -> list[str]:
    """Rank every model by comparable 2026 accuracy, then training accuracy."""
    return sorted(
        catalog,
        key=lambda model_id: (
            live_accuracy.get(model_id, -1.0),
            catalog[model_id]["metrics"].get("accuracy", -1.0),
            catalog[model_id]["label"],
        ),
        reverse=True,
    )
