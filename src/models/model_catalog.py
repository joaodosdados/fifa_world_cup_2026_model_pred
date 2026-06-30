"""Runtime catalog that keeps model selection, prediction and metadata aligned."""

from __future__ import annotations

import logging
from typing import Any, Dict

import pandas as pd

from src.models.ensemble_predictor import EnsemblePredictor
from src.models.model_manager import ModelManager
from src.models.sklearn_adapter import SklearnMatchPredictor


STATISTICAL_MODEL_ID = "elo_poisson"
LIVE_UPDATED_STATISTICAL_MODEL_ID = "elo_poisson_2026_groups"
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


def _estimator_display_target(model: Any) -> Any:
    return getattr(model, "outcome_model", model)


def _build_statistical_catalog_item(
    model_id: str,
    matches: pd.DataFrame,
    label: str,
    description: str,
    extra_metrics: Dict[str, Any] | None = None,
    training_scope: str = "historical",
) -> Dict[str, Any]:
    statistical = EnsemblePredictor()
    statistical.train(matches)
    metrics = {
        "elo_weight": statistical.elo_weight,
        "poisson_weight": statistical.poisson_weight,
        "teams": len(statistical.poisson_model.team_stats),
    }
    if extra_metrics:
        metrics.update(extra_metrics)

    return {
        "id": model_id,
        "label": label,
        "kind": "statistical",
        "predictor": statistical,
        "estimator": statistical,
        "metrics": metrics,
        "description": description,
        "training_scope": training_scope,
        "training_matches": matches,
    }


def build_model_catalog(
    matches: pd.DataFrame,
    live_updated_matches: pd.DataFrame | None = None,
) -> Dict[str, Dict[str, Any]]:
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
            "label": _display_name(model_id, _estimator_display_target(estimator)),
            "kind": "machine_learning",
            "predictor": predictor,
            "estimator": estimator,
            "metrics": metadata.get("metrics", {}),
            "description": (
                metadata.get("description")
                or metadata.get("metrics", {}).get("description")
                or "Modelo scikit-learn treinado com dados históricos."
            ),
            "training_scope": "historical",
            "training_matches": matches,
        }

    catalog[STATISTICAL_MODEL_ID] = _build_statistical_catalog_item(
        STATISTICAL_MODEL_ID,
        matches,
        "Estatístico · ELO + Poisson",
        (
            "Combina força relativa por ELO com distribuição de gols por Poisson."
        ),
    )

    if live_updated_matches is not None and len(live_updated_matches) > len(matches):
        added_matches = len(live_updated_matches) - len(matches)
        catalog[LIVE_UPDATED_STATISTICAL_MODEL_ID] = _build_statistical_catalog_item(
            LIVE_UPDATED_STATISTICAL_MODEL_ID,
            live_updated_matches,
            "Live · ELO + Poisson + grupos 2026",
            (
                "Retreinado em runtime com o histórico pré-Copa e os jogos da "
                "fase de grupos 2026 já finalizados. Use para prever o mata-mata; "
                "a acurácia da fase de grupos não é comparável porque esses jogos "
                "entraram no treino."
            ),
            extra_metrics={"completed_2026_matches": added_matches},
            training_scope="live_2026_groups",
        )

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
