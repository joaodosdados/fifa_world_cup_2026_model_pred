#!/usr/bin/env python3
"""Train, evaluate and persist every dashboard ML model without Jupyter."""

from __future__ import annotations

import argparse
import json
import os
import pickle
import sys
from dataclasses import dataclass
from datetime import datetime, timezone
from pathlib import Path
from tempfile import TemporaryDirectory
from typing import Any, Dict

import numpy as np
import pandas as pd
import sklearn
from sklearn.base import clone
from sklearn.ensemble import (
    GradientBoostingClassifier,
    RandomForestClassifier,
    VotingClassifier,
)
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score, classification_report, confusion_matrix
from sklearn.model_selection import TimeSeriesSplit, cross_val_score
from sklearn.naive_bayes import GaussianNB
from sklearn.neighbors import KNeighborsClassifier
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import StandardScaler
from sklearn.svm import SVC

PROJECT_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(PROJECT_ROOT))

from src.data.loader import DataLoader
from src.models.features import FEATURE_NAMES, build_temporal_training_data


MODEL_LABELS = {
    "logistic_regression": "Logistic Regression",
    "random_forest": "Random Forest",
    "gradient_boosting": "Gradient Boosting",
    "svm": "SVM",
    "k-nearest_neighbors": "K-Nearest Neighbors",
    "naive_bayes": "Naive Bayes",
    "ensemble_top3": "Ensemble Top 3",
}


@dataclass
class Evaluation:
    model_id: str
    estimator: Any
    accuracy: float
    cv_mean: float
    cv_std: float
    confusion_matrix: list[list[int]]
    classification_report: Dict[str, Any]


def build_estimators(random_state: int) -> Dict[str, Any]:
    """Return deterministic estimators with scaling where it materially helps."""
    return {
        "logistic_regression": make_pipeline(
            StandardScaler(),
            LogisticRegression(max_iter=2000, random_state=random_state),
        ),
        "random_forest": RandomForestClassifier(
            n_estimators=400,
            min_samples_leaf=2,
            class_weight="balanced_subsample",
            random_state=random_state,
            n_jobs=-1,
        ),
        "gradient_boosting": GradientBoostingClassifier(
            n_estimators=200,
            learning_rate=0.05,
            max_depth=3,
            random_state=random_state,
        ),
        "svm": make_pipeline(
            StandardScaler(),
            SVC(
                kernel="rbf",
                class_weight="balanced",
                random_state=random_state,
            ),
        ),
        "k-nearest_neighbors": make_pipeline(
            StandardScaler(),
            KNeighborsClassifier(n_neighbors=11, weights="distance"),
        ),
        "naive_bayes": GaussianNB(),
    }


def temporal_split(
    X: np.ndarray,
    y: np.ndarray,
    test_size: float,
) -> tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray, int]:
    """Reserve the newest observations as a realistic temporal holdout."""
    split_index = int(len(X) * (1 - test_size))
    if split_index <= 0 or split_index >= len(X):
        raise ValueError("test_size must leave observations in train and test sets")
    return X[:split_index], X[split_index:], y[:split_index], y[split_index:], split_index


def evaluate_estimator(
    model_id: str,
    estimator: Any,
    X_train: np.ndarray,
    y_train: np.ndarray,
    X_test: np.ndarray,
    y_test: np.ndarray,
    cv_folds: int,
) -> Evaluation:
    """Fit one estimator and calculate temporal holdout/CV metrics."""
    fitted = clone(estimator).fit(X_train, y_train)
    predictions = fitted.predict(X_test)
    accuracy = accuracy_score(y_test, predictions)

    cv = TimeSeriesSplit(n_splits=cv_folds)
    scores = cross_val_score(
        clone(estimator),
        X_train,
        y_train,
        cv=cv,
        scoring="accuracy",
        n_jobs=1,
        error_score="raise",
    )

    return Evaluation(
        model_id=model_id,
        estimator=estimator,
        accuracy=float(accuracy),
        cv_mean=float(scores.mean()),
        cv_std=float(scores.std()),
        confusion_matrix=confusion_matrix(
            y_test, predictions, labels=[0, 1, 2]
        ).tolist(),
        classification_report=classification_report(
            y_test,
            predictions,
            labels=[0, 1, 2],
            target_names=["away_win", "draw", "home_win"],
            output_dict=True,
            zero_division=0,
        ),
    )


def create_top3_ensemble(
    evaluations: Dict[str, Evaluation],
) -> tuple[VotingClassifier, list[str]]:
    """Create a hard-voting ensemble from the three strongest holdout models."""
    top_ids = sorted(
        evaluations,
        key=lambda model_id: (
            evaluations[model_id].accuracy,
            evaluations[model_id].cv_mean,
        ),
        reverse=True,
    )[:3]
    ensemble = VotingClassifier(
        estimators=[
            (model_id.replace("-", "_"), clone(evaluations[model_id].estimator))
            for model_id in top_ids
        ],
        voting="hard",
    )
    return ensemble, top_ids


def train_pipeline(args: argparse.Namespace) -> pd.DataFrame:
    """Run the complete training pipeline and atomically publish artifacts."""
    matches = DataLoader(data_dir=str(args.data_dir)).load_matches(processed=False)
    X, y, context = build_temporal_training_data(matches)
    X_train, X_test, y_train, y_test, split_index = temporal_split(
        X, y, args.test_size
    )

    print(
        f"Dados válidos: {len(matches)} | treino: {len(X_train)} | "
        f"teste temporal: {len(X_test)}"
    )
    if "year" in context:
        print(
            f"Período de teste: {context.iloc[split_index]['year']}–"
            f"{context.iloc[-1]['year']}"
        )

    evaluations: Dict[str, Evaluation] = {}
    estimators = build_estimators(args.random_state)

    for model_id, estimator in estimators.items():
        print(f"Treinando {MODEL_LABELS[model_id]}...")
        evaluation = evaluate_estimator(
            model_id,
            estimator,
            X_train,
            y_train,
            X_test,
            y_test,
            args.cv_folds,
        )
        evaluations[model_id] = evaluation
        print(
            f"  holdout={evaluation.accuracy:.1%} | "
            f"cv={evaluation.cv_mean:.1%} ± {evaluation.cv_std:.1%}"
        )

    if not args.no_ensemble:
        ensemble, top_ids = create_top3_ensemble(evaluations)
        print(
            "Treinando Ensemble Top 3: "
            + ", ".join(MODEL_LABELS[model_id] for model_id in top_ids)
        )
        ensemble_evaluation = evaluate_estimator(
            "ensemble_top3",
            ensemble,
            X_train,
            y_train,
            X_test,
            y_test,
            args.cv_folds,
        )
        ensemble_evaluation.classification_report["ensemble_models"] = top_ids
        evaluations["ensemble_top3"] = ensemble_evaluation
        print(
            f"  holdout={ensemble_evaluation.accuracy:.1%} | "
            f"cv={ensemble_evaluation.cv_mean:.1%} ± "
            f"{ensemble_evaluation.cv_std:.1%}"
        )

    publish_models(
        evaluations=evaluations,
        X=X,
        y=y,
        output_dir=args.output_dir,
        training_rows=len(X_train),
        test_rows=len(X_test),
        test_start_year=context.iloc[split_index]["year"],
        random_state=args.random_state,
    )

    comparison = pd.DataFrame(
        [
            {
                "rank": 0,
                "model_id": model_id,
                "model": MODEL_LABELS[model_id],
                "accuracy": evaluation.accuracy,
                "cv_mean": evaluation.cv_mean,
                "cv_std": evaluation.cv_std,
            }
            for model_id, evaluation in evaluations.items()
        ]
    ).sort_values(["accuracy", "cv_mean"], ascending=False)
    comparison["rank"] = range(1, len(comparison) + 1)

    print("\nRanking do treinamento")
    print(
        comparison.to_string(
            index=False,
            formatters={
                "accuracy": "{:.1%}".format,
                "cv_mean": "{:.1%}".format,
                "cv_std": "{:.1%}".format,
            },
        )
    )
    print(f"\nModelos publicados em: {args.output_dir}")
    return comparison


def publish_models(
    evaluations: Dict[str, Evaluation],
    X: np.ndarray,
    y: np.ndarray,
    output_dir: Path,
    training_rows: int,
    test_rows: int,
    test_start_year: int,
    random_state: int,
) -> None:
    """Refit on all history and atomically replace models plus metadata."""
    output_dir.mkdir(parents=True, exist_ok=True)
    best_model_id = max(
        evaluations,
        key=lambda model_id: (
            evaluations[model_id].accuracy,
            evaluations[model_id].cv_mean,
        ),
    )
    timestamp = datetime.now(timezone.utc).isoformat()
    metadata = {}

    with TemporaryDirectory(dir=output_dir) as temporary_directory:
        staging = Path(temporary_directory)

        for model_id, evaluation in evaluations.items():
            production_model = clone(evaluation.estimator).fit(X, y)
            staged_model = staging / f"{model_id}.pkl"
            with staged_model.open("wb") as model_file:
                pickle.dump(production_model, model_file)

            report = dict(evaluation.classification_report)
            ensemble_models = report.pop("ensemble_models", None)
            metrics = {
                "accuracy": evaluation.accuracy,
                "cv_mean": evaluation.cv_mean,
                "cv_std": evaluation.cv_std,
                "confusion_matrix": evaluation.confusion_matrix,
                "classification_report": report,
                "features": FEATURE_NAMES,
                "feature_method": "chronological_pre_match_expanding_stats",
                "training_samples": training_rows,
                "test_samples": test_rows,
                "test_start_year": int(test_start_year),
                "random_state": random_state,
                "sklearn_version": sklearn.__version__,
                "numpy_version": np.__version__,
            }
            if ensemble_models:
                metrics["ensemble_models"] = ensemble_models

            default_output_dir = (PROJECT_ROOT / "models").resolve()
            metadata_path = (
                Path("models") / f"{model_id}.pkl"
                if output_dir.resolve() == default_output_dir
                else output_dir / f"{model_id}.pkl"
            )
            metadata[model_id] = {
                "type": "sklearn",
                "metrics": metrics,
                "description": (
                    f"{MODEL_LABELS[model_id]} treinado com validação temporal "
                    "e features pré-jogo sem vazamento de dados."
                ),
                "path": str(metadata_path),
                "saved_at": timestamp,
                "active": model_id == best_model_id,
            }

        staged_metadata = staging / "models_metadata.json"
        staged_metadata.write_text(
            json.dumps(metadata, ensure_ascii=False, indent=2),
            encoding="utf-8",
        )

        for model_id in evaluations:
            os.replace(
                staging / f"{model_id}.pkl",
                output_dir / f"{model_id}.pkl",
            )
        os.replace(staged_metadata, output_dir / "models_metadata.json")


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Treina e publica os modelos usados pelo dashboard."
    )
    parser.add_argument(
        "--data-dir",
        type=Path,
        default=PROJECT_ROOT / "data",
        help="Diretório que contém data/raw/WorldCupMatches.csv.",
    )
    parser.add_argument(
        "--output-dir",
        type=Path,
        default=PROJECT_ROOT / "models",
        help="Diretório de saída dos arquivos .pkl e metadados.",
    )
    parser.add_argument(
        "--test-size",
        type=float,
        default=0.20,
        help="Fração mais recente reservada para teste temporal.",
    )
    parser.add_argument("--cv-folds", type=int, default=5)
    parser.add_argument("--random-state", type=int, default=42)
    parser.add_argument(
        "--no-ensemble",
        action="store_true",
        help="Não treina o ensemble com os três melhores modelos.",
    )
    return parser.parse_args()


if __name__ == "__main__":
    train_pipeline(parse_args())
