"""Composite sklearn artifacts used by the dashboard.

Each dashboard ML entry predicts two related but distinct things:

1. the match outcome class (home win, draw, away win);
2. the expected goals for each team.

Keeping both estimators in one pickled artifact lets the UI still expose a
single selectable model while avoiding the old heuristic where goals were
derived from the classification result.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any


@dataclass
class MatchModelBundle:
    """Outcome classifier plus goal regressor."""

    outcome_model: Any
    goals_model: Any
    model_id: str

    @property
    def classes_(self):
        return getattr(self.outcome_model, "classes_", None)

    def predict(self, X):
        return self.outcome_model.predict(X)

    def predict_proba(self, X):
        return self.outcome_model.predict_proba(X)

    def decision_function(self, X):
        return self.outcome_model.decision_function(X)

    def predict_goals(self, X):
        return self.goals_model.predict(X)

    def get_params(self, deep: bool = True):
        return {
            "outcome_model": self.outcome_model,
            "goals_model": self.goals_model,
            "model_id": self.model_id,
        }
