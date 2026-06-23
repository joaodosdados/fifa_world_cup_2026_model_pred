"""
Sklearn Model Adapter - Adapta modelos sklearn para interface do dashboard
Permite usar modelos sklearn com a mesma interface do EnsemblePredictor
"""

import numpy as np
import pandas as pd
from scipy.stats import poisson
from typing import Dict, Any
from src.models.features import (
    DEFAULT_TEAM_STATS,
    calculate_team_stats,
    create_match_features,
)


class SklearnMatchPredictor:
    """
    Adaptador que permite modelos sklearn fazerem predições de partidas
    usando a mesma interface do EnsemblePredictor
    """
    
    def __init__(self, sklearn_model: Any, historical_data: pd.DataFrame):
        """
        Inicializa o adaptador
        
        Args:
            sklearn_model: Modelo sklearn treinado
            historical_data: Dados históricos para calcular estatísticas
        """
        self.model = sklearn_model
        self.team_stats = calculate_team_stats(historical_data)
    
    def _create_features(self, team_a: str, team_b: str) -> np.ndarray:
        """
        Cria features para predição
        
        Args:
            team_a: Time da casa
            team_b: Time visitante
            
        Returns:
            Array com features
        """
        return create_match_features(self.team_stats, team_a, team_b)

    @staticmethod
    def _softmax(values: np.ndarray) -> np.ndarray:
        values = np.asarray(values, dtype=float)
        values = values - np.max(values)
        exponentials = np.exp(values)
        return exponentials / exponentials.sum()

    def _predict_probabilities(self, X: np.ndarray) -> Dict[int, float]:
        """Return class probabilities using the best capability of the estimator."""
        classes = list(getattr(self.model, 'classes_', [0, 1, 2]))

        values = None

        if hasattr(self.model, 'predict_proba'):
            try:
                values = np.asarray(self.model.predict_proba(X)[0], dtype=float)
            except (AttributeError, NotImplementedError):
                values = None

        if values is None and hasattr(self.model, 'decision_function'):
            try:
                decision = np.asarray(self.model.decision_function(X), dtype=float)
                values = self._softmax(decision.reshape(-1))
            except (AttributeError, NotImplementedError):
                values = None

        if values is None and hasattr(self.model, 'estimators_'):
            votes = [int(estimator.predict(X)[0]) for estimator in self.model.estimators_]
            values = np.array([votes.count(int(label)) for label in classes], dtype=float)
            values = values / values.sum()

        if values is None:
            predicted = int(self.model.predict(X)[0])
            values = np.array([1.0 if int(label) == predicted else 0.0 for label in classes])

        return {int(label): float(probability) for label, probability in zip(classes, values)}

    def _predict_expected_goals(
        self, team_a: str, team_b: str, is_home_a: bool, X: np.ndarray | None = None
    ) -> tuple[float, float]:
        """Predict goals with the dedicated regressor when available."""
        if X is not None and hasattr(self.model, "predict_goals"):
            predicted = np.asarray(self.model.predict_goals(X)[0], dtype=float)
            if len(predicted) >= 2 and np.all(np.isfinite(predicted[:2])):
                return max(0.05, predicted[0]), max(0.05, predicted[1])

        # Backwards-compatible fallback for old artifacts without a goal model.
        from src.utils.team_names import normalize_team_name

        home_stats = self.team_stats.get(
            normalize_team_name(team_a), DEFAULT_TEAM_STATS
        )
        away_stats = self.team_stats.get(
            normalize_team_name(team_b), DEFAULT_TEAM_STATS
        )

        expected_home = (
            home_stats['goals_scored'] + away_stats['goals_conceded']
        ) / 2
        expected_away = (
            away_stats['goals_scored'] + home_stats['goals_conceded']
        ) / 2

        if is_home_a:
            expected_home *= 1.10
            expected_away *= 0.95
        else:
            expected_home *= 0.95
            expected_away *= 1.10

        return max(0.05, expected_home), max(0.05, expected_away)

    @staticmethod
    def _most_likely_score(
        expected_home: float,
        expected_away: float,
        max_goals: int = 8,
    ) -> tuple[int, int]:
        """Select the likeliest Poisson score from the goal regressor outputs."""
        best_score = (0, 0)
        best_probability = -1.0

        for home_goals in range(max_goals + 1):
            for away_goals in range(max_goals + 1):
                probability = poisson.pmf(
                    home_goals, expected_home
                ) * poisson.pmf(away_goals, expected_away)
                if probability > best_probability:
                    best_probability = probability
                    best_score = (home_goals, away_goals)

        return best_score
    
    def predict_match(self, team_a: str, team_b: str, is_home_a: bool = True) -> Dict[str, Any]:
        """
        Prediz resultado de uma partida
        
        Args:
            team_a: Time da casa
            team_b: Time visitante
            is_home_a: Se team_a joga em casa (compatibilidade com EnsemblePredictor)
            
        Returns:
            Dicionário com probabilidades e predições
        """
        # Criar features
        X = self._create_features(team_a, team_b)
        
        # Fazer predição
        probabilities = self._predict_probabilities(X)
        away_win = probabilities.get(0, 0.0)
        draw = probabilities.get(1, 0.0)
        home_win = probabilities.get(2, 0.0)
        
        expected_goals_home, expected_goals_away = self._predict_expected_goals(
            team_a, team_b, is_home_a, X
        )
        most_likely_score = self._most_likely_score(
            expected_goals_home,
            expected_goals_away,
        )
        
        return {
            'home_win': home_win,
            'draw': draw,
            'away_win': away_win,
            'expected_goals_home': expected_goals_home,
            'expected_goals_away': expected_goals_away,
            'most_likely_score': most_likely_score,
            'model_type': 'sklearn',
            'model_name': type(getattr(self.model, "outcome_model", self.model)).__name__,
            'goal_model_name': type(getattr(self.model, "goals_model", None)).__name__
            if hasattr(self.model, "goals_model")
            else "historical_fallback",
        }
    
    def predict_winner(self, team_a: str, team_b: str, is_home_a: bool = True) -> str:
        """
        Prediz o vencedor mais provável
        
        Args:
            team_a: Time da casa
            team_b: Time visitante
            is_home_a: Se team_a joga em casa
            
        Returns:
            Nome do time vencedor ou 'Draw'
        """
        prediction = self.predict_match(team_a, team_b, is_home_a)
        
        if prediction['home_win'] > prediction['draw'] and prediction['home_win'] > prediction['away_win']:
            return team_a
        elif prediction['away_win'] > prediction['draw'] and prediction['away_win'] > prediction['home_win']:
            return team_b
        else:
            return 'Draw'

# Made with Bob
