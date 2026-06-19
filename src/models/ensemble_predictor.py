"""
Ensemble Predictor combining ELO and Poisson models
Professional betting house approach
"""

import pandas as pd
import numpy as np
from typing import Dict
import logging

from src.models.elo_predictor import ELOPredictor
from src.models.poisson_predictor import PoissonPredictor
from src.utils.team_names import normalize_team_name

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


class EnsemblePredictor:
    """
    Ensemble model combining ELO and Poisson predictions
    """
    
    def __init__(self, elo_weight: float = 0.4, poisson_weight: float = 0.6):
        """
        Initialize ensemble predictor
        
        Args:
            elo_weight: Weight for ELO predictions (0-1)
            poisson_weight: Weight for Poisson predictions (0-1)
        """
        self.elo_weight = elo_weight
        self.poisson_weight = poisson_weight
        
        # Normalize weights
        total_weight = elo_weight + poisson_weight
        self.elo_weight = elo_weight / total_weight
        self.poisson_weight = poisson_weight / total_weight
        
        self.elo_model = ELOPredictor(k_factor=32, home_advantage=100)
        self.poisson_model = PoissonPredictor(home_advantage=1.3)
        
        self.is_trained = False
    
    def train(self, matches_df: pd.DataFrame) -> None:
        """
        Train both models on historical data
        
        Args:
            matches_df: DataFrame with match results
        """
        logger.info("Training ensemble model...")
        
        # Train ELO model
        logger.info("Training ELO model...")
        self.elo_model.train_on_historical_data(matches_df)
        
        # Train Poisson model
        logger.info("Training Poisson model...")
        self.poisson_model.train(matches_df)
        
        self.is_trained = True
        logger.info("Ensemble training complete")
    
    def predict_match(self, team_a: str, team_b: str, 
                     is_home_a: bool = True) -> Dict[str, float]:
        """
        Predict match outcome using ensemble of models
        
        Args:
            team_a: Name of team A
            team_b: Name of team B
            is_home_a: Whether team A is playing at home
            
        Returns:
            Dictionary with combined predictions
        """
        if not self.is_trained:
            raise ValueError("Model must be trained before making predictions")

        model_team_a = normalize_team_name(team_a)
        model_team_b = normalize_team_name(team_b)
        
        # Get predictions from both models
        elo_pred = self.elo_model.predict_match(
            model_team_a, model_team_b, is_home_a
        )
        poisson_pred = self.poisson_model.predict_match(
            model_team_a, model_team_b, is_home_a
        )
        
        # Combine predictions using weighted average
        combined = {
            'home_win': (self.elo_weight * elo_pred['home_win'] + 
                        self.poisson_weight * poisson_pred['home_win']),
            'draw': (self.elo_weight * elo_pred['draw'] + 
                    self.poisson_weight * poisson_pred['draw']),
            'away_win': (self.elo_weight * elo_pred['away_win'] + 
                        self.poisson_weight * poisson_pred['away_win']),
        }

        probability_total = sum(combined.values())
        if probability_total:
            combined = {
                outcome: probability / probability_total
                for outcome, probability in combined.items()
            }
        
        # Add additional information
        combined.update({
            'expected_goals_home': poisson_pred['expected_goals_home'],
            'expected_goals_away': poisson_pred['expected_goals_away'],
            'most_likely_score': poisson_pred['most_likely_score'],
            'home_elo': elo_pred['home_rating'],
            'away_elo': elo_pred['away_rating'],
            'elo_prediction': elo_pred,
            'poisson_prediction': poisson_pred
        })
        
        return combined
    
    def predict_winner(self, team_a: str, team_b: str, 
                      is_home_a: bool = True) -> str:
        """
        Predict the most likely winner
        
        Args:
            team_a: Name of team A
            team_b: Name of team B
            is_home_a: Whether team A is playing at home
            
        Returns:
            Name of predicted winner or 'Draw'
        """
        prediction = self.predict_match(team_a, team_b, is_home_a)
        
        max_prob = max(prediction['home_win'], prediction['draw'], prediction['away_win'])
        
        if max_prob == prediction['home_win']:
            return team_a
        elif max_prob == prediction['away_win']:
            return team_b
        else:
            return 'Draw'
    
    def get_match_summary(self, team_a: str, team_b: str, 
                         is_home_a: bool = True) -> Dict:
        """
        Get comprehensive match prediction summary
        
        Args:
            team_a: Name of team A
            team_b: Name of team B
            is_home_a: Whether team A is playing at home
            
        Returns:
            Dictionary with detailed prediction information
        """
        prediction = self.predict_match(team_a, team_b, is_home_a)
        winner = self.predict_winner(team_a, team_b, is_home_a)
        
        return {
            'team_a': team_a,
            'team_b': team_b,
            'is_home_a': is_home_a,
            'predicted_winner': winner,
            'probabilities': {
                f'{team_a} win': f"{prediction['home_win']:.1%}",
                'Draw': f"{prediction['draw']:.1%}",
                f'{team_b} win': f"{prediction['away_win']:.1%}"
            },
            'expected_score': f"{prediction['expected_goals_home']:.1f} - {prediction['expected_goals_away']:.1f}",
            'most_likely_score': f"{prediction['most_likely_score'][0]} - {prediction['most_likely_score'][1]}",
            'team_ratings': {
                team_a: f"{prediction['home_elo']:.0f}",
                team_b: f"{prediction['away_elo']:.0f}"
            }
        }
    
    def predict_tournament_matches(self, matches: list) -> pd.DataFrame:
        """
        Predict multiple tournament matches
        
        Args:
            matches: List of tuples (team_a, team_b, is_home_a)
            
        Returns:
            DataFrame with predictions for all matches
        """
        predictions = []
        
        for team_a, team_b, is_home_a in matches:
            pred = self.predict_match(team_a, team_b, is_home_a)
            winner = self.predict_winner(team_a, team_b, is_home_a)
            
            predictions.append({
                'Team A': team_a,
                'Team B': team_b,
                'Predicted Winner': winner,
                f'{team_a} Win %': f"{pred['home_win']:.1%}",
                'Draw %': f"{pred['draw']:.1%}",
                f'{team_b} Win %': f"{pred['away_win']:.1%}",
                'Expected Score': f"{pred['expected_goals_home']:.1f}-{pred['expected_goals_away']:.1f}",
                'Most Likely': f"{pred['most_likely_score'][0]}-{pred['most_likely_score'][1]}"
            })
        
        return pd.DataFrame(predictions)


if __name__ == "__main__":
    # Test the ensemble predictor
    from src.data.loader import DataLoader
    
    loader = DataLoader()
    matches = loader.load_matches(processed=False)
    
    # Initialize and train
    ensemble = EnsemblePredictor(elo_weight=0.4, poisson_weight=0.6)
    ensemble.train(matches)
    
    # Test predictions
    print("\n" + "="*60)
    print("ENSEMBLE PREDICTION TEST")
    print("="*60)
    
    test_matches = [
        ('Brazil', 'Germany', True),
        ('Argentina', 'France', True),
        ('Spain', 'England', False)
    ]
    
    for team_a, team_b, is_home in test_matches:
        summary = ensemble.get_match_summary(team_a, team_b, is_home)
        home_label = "home" if is_home else "away"
        
        print(f"\n{team_a} ({home_label}) vs {team_b}:")
        print(f"  Predicted Winner: {summary['predicted_winner']}")
        print(f"  Probabilities: {summary['probabilities']}")
        print(f"  Expected Score: {summary['expected_score']}")
        print(f"  Most Likely Score: {summary['most_likely_score']}")
        print(f"  ELO Ratings: {summary['team_ratings']}")

# Made with Bob
