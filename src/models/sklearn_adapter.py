"""
Sklearn Model Adapter - Adapta modelos sklearn para interface do dashboard
Permite usar modelos sklearn com a mesma interface do EnsemblePredictor
"""

import numpy as np
import pandas as pd
from typing import Dict, Any


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
        self.team_stats = self._calculate_team_stats(historical_data)
    
    def _calculate_team_stats(self, df: pd.DataFrame) -> Dict[str, Dict[str, float]]:
        """Calcula estatísticas históricas para cada time"""
        team_stats = {}
        
        for team in set(df['Home Team Name'].unique()) | set(df['Away Team Name'].unique()):
            home_matches = df[df['Home Team Name'] == team]
            away_matches = df[df['Away Team Name'] == team]
            
            total_matches = len(home_matches) + len(away_matches)
            if total_matches == 0:
                continue
            
            team_stats[team] = {
                'goals_scored': (home_matches['Home Team Goals'].sum() + 
                               away_matches['Away Team Goals'].sum()) / (total_matches + 1),
                'goals_conceded': (home_matches['Away Team Goals'].sum() + 
                                 away_matches['Home Team Goals'].sum()) / (total_matches + 1),
                'wins': ((home_matches['Home Team Goals'] > home_matches['Away Team Goals']).sum() + 
                        (away_matches['Away Team Goals'] > away_matches['Home Team Goals']).sum()) / (total_matches + 1)
            }
        
        return team_stats
    
    def _create_features(self, team_a: str, team_b: str) -> np.ndarray:
        """
        Cria features para predição
        
        Args:
            team_a: Time da casa
            team_b: Time visitante
            
        Returns:
            Array com features
        """
        # Usar estatísticas médias se time não conhecido
        default_stats = {
            'goals_scored': 1.5,
            'goals_conceded': 1.5,
            'wins': 0.33
        }
        
        home_stats = self.team_stats.get(team_a, default_stats)
        away_stats = self.team_stats.get(team_b, default_stats)
        
        features = [
            home_stats['goals_scored'],
            home_stats['goals_conceded'],
            home_stats['wins'],
            away_stats['goals_scored'],
            away_stats['goals_conceded'],
            away_stats['wins'],
            1  # home advantage
        ]
        
        return np.array(features).reshape(1, -1)
    
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
        try:
            # Tentar usar probabilidades
            proba = self.model.predict_proba(X)[0]
            
            # proba[0] = away win, proba[1] = draw, proba[2] = home win
            away_win = float(proba[0]) if len(proba) > 0 else 0.33
            draw = float(proba[1]) if len(proba) > 1 else 0.33
            home_win = float(proba[2]) if len(proba) > 2 else 0.34
        except Exception:
            # Modelo não suporta probabilidades ou não foi treinado com probability=True
            # Usar predição hard
            prediction = self.model.predict(X)[0]
            
            if prediction == 2:  # home win
                home_win, draw, away_win = 0.7, 0.2, 0.1
            elif prediction == 0:  # away win
                home_win, draw, away_win = 0.1, 0.2, 0.7
            else:  # draw
                home_win, draw, away_win = 0.3, 0.4, 0.3
        
        # Calcular placares esperados baseado nas estatísticas
        home_stats = self.team_stats.get(team_a, {'goals_scored': 1.5})
        away_stats = self.team_stats.get(team_b, {'goals_scored': 1.5})
        
        expected_goals_home = home_stats['goals_scored'] * 1.1  # home advantage
        expected_goals_away = away_stats['goals_scored'] * 0.9
        
        # Placar mais provável (arredondado)
        most_likely_score = (
            round(expected_goals_home),
            round(expected_goals_away)
        )
        
        return {
            'home_win': home_win,
            'draw': draw,
            'away_win': away_win,
            'expected_goals_home': expected_goals_home,
            'expected_goals_away': expected_goals_away,
            'most_likely_score': most_likely_score,
            'model_type': 'sklearn',
            'model_name': type(self.model).__name__
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
