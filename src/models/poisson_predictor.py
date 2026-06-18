"""
Poisson Distribution Model for Football Match Prediction
Based on professional betting house methodologies
"""

import pandas as pd
import numpy as np
from scipy.stats import poisson
from typing import Dict, Tuple
import logging

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


class PoissonPredictor:
    """
    Poisson distribution model for predicting football match outcomes
    """
    
    def __init__(self, home_advantage: float = 1.3):
        """
        Initialize Poisson predictor
        
        Args:
            home_advantage: Multiplier for home team attack strength
        """
        self.home_advantage = home_advantage
        self.team_stats = {}
        self.league_avg_goals = 0
        
    def calculate_team_stats(self, matches_df: pd.DataFrame) -> None:
        """
        Calculate attack and defense strength for each team
        
        Args:
            matches_df: DataFrame with match results
        """
        logger.info("Calculating team statistics...")
        
        # Calculate league average goals
        total_goals = matches_df['Home Team Goals'].sum() + matches_df['Away Team Goals'].sum()
        total_matches = len(matches_df) * 2  # Each match has 2 teams
        self.league_avg_goals = total_goals / total_matches
        
        teams = set(matches_df['Home Team Name'].unique()) | set(matches_df['Away Team Name'].unique())
        
        for team in teams:
            # Home matches
            home_matches = matches_df[matches_df['Home Team Name'] == team]
            home_scored = home_matches['Home Team Goals'].sum()
            home_conceded = home_matches['Away Team Goals'].sum()
            home_count = len(home_matches)
            
            # Away matches
            away_matches = matches_df[matches_df['Away Team Name'] == team]
            away_scored = away_matches['Away Team Goals'].sum()
            away_conceded = away_matches['Home Team Goals'].sum()
            away_count = len(away_matches)
            
            # Calculate averages
            total_matches = home_count + away_count
            if total_matches > 0:
                avg_scored = (home_scored + away_scored) / total_matches
                avg_conceded = (home_conceded + away_conceded) / total_matches
                
                # Calculate attack and defense strength relative to league average
                attack_strength = avg_scored / self.league_avg_goals if self.league_avg_goals > 0 else 1.0
                defense_strength = avg_conceded / self.league_avg_goals if self.league_avg_goals > 0 else 1.0
                
                self.team_stats[team] = {
                    'attack_strength': attack_strength,
                    'defense_strength': defense_strength,
                    'avg_goals_scored': avg_scored,
                    'avg_goals_conceded': avg_conceded,
                    'matches_played': total_matches
                }
        
        logger.info(f"Statistics calculated for {len(self.team_stats)} teams")
    
    def get_team_stats(self, team: str) -> Dict[str, float]:
        """Get statistics for a team"""
        if team not in self.team_stats:
            # Return league average for unknown teams
            return {
                'attack_strength': 1.0,
                'defense_strength': 1.0,
                'avg_goals_scored': self.league_avg_goals,
                'avg_goals_conceded': self.league_avg_goals,
                'matches_played': 0
            }
        return self.team_stats[team]
    
    def predict_goals(self, team_a: str, team_b: str, 
                     is_home_a: bool = True) -> Tuple[float, float]:
        """
        Predict expected goals for both teams
        
        Args:
            team_a: Name of team A
            team_b: Name of team B
            is_home_a: Whether team A is playing at home
            
        Returns:
            Tuple of (expected_goals_a, expected_goals_b)
        """
        stats_a = self.get_team_stats(team_a)
        stats_b = self.get_team_stats(team_b)
        
        # Calculate expected goals using attack and defense strengths
        if is_home_a:
            expected_a = (stats_a['attack_strength'] * stats_b['defense_strength'] * 
                         self.league_avg_goals * self.home_advantage)
            expected_b = (stats_b['attack_strength'] * stats_a['defense_strength'] * 
                         self.league_avg_goals)
        else:
            expected_a = (stats_a['attack_strength'] * stats_b['defense_strength'] * 
                         self.league_avg_goals)
            expected_b = (stats_b['attack_strength'] * stats_a['defense_strength'] * 
                         self.league_avg_goals * self.home_advantage)
        
        return expected_a, expected_b
    
    def predict_match(self, team_a: str, team_b: str, 
                     is_home_a: bool = True, max_goals: int = 10) -> Dict[str, float]:
        """
        Predict match outcome probabilities using Poisson distribution
        
        Args:
            team_a: Name of team A
            team_b: Name of team B
            is_home_a: Whether team A is playing at home
            max_goals: Maximum goals to consider in calculations
            
        Returns:
            Dictionary with match outcome probabilities
        """
        expected_a, expected_b = self.predict_goals(team_a, team_b, is_home_a)
        
        # Generate probability distributions
        prob_a = [poisson.pmf(i, expected_a) for i in range(max_goals)]
        prob_b = [poisson.pmf(i, expected_b) for i in range(max_goals)]
        
        # Calculate match outcome probabilities
        home_win = 0
        away_win = 0
        draw = 0
        
        for i in range(max_goals):
            for j in range(max_goals):
                prob = prob_a[i] * prob_b[j]
                if i > j:
                    home_win += prob
                elif i < j:
                    away_win += prob
                else:
                    draw += prob
        
        # Calculate most likely score
        most_likely_score = self._get_most_likely_score(prob_a, prob_b, max_goals)
        
        return {
            'home_win': home_win,
            'draw': draw,
            'away_win': away_win,
            'expected_goals_home': expected_a,
            'expected_goals_away': expected_b,
            'most_likely_score': most_likely_score
        }
    
    def _get_most_likely_score(self, prob_a: list, prob_b: list, 
                               max_goals: int) -> Tuple[int, int]:
        """Find the most likely score"""
        max_prob = 0
        best_score = (0, 0)
        
        for i in range(max_goals):
            for j in range(max_goals):
                prob = prob_a[i] * prob_b[j]
                if prob > max_prob:
                    max_prob = prob
                    best_score = (i, j)
        
        return best_score
    
    def predict_score_probabilities(self, team_a: str, team_b: str,
                                   is_home_a: bool = True, 
                                   max_goals: int = 6) -> pd.DataFrame:
        """
        Generate probability matrix for all possible scores
        
        Args:
            team_a: Name of team A
            team_b: Name of team B
            is_home_a: Whether team A is playing at home
            max_goals: Maximum goals to show
            
        Returns:
            DataFrame with score probabilities
        """
        expected_a, expected_b = self.predict_goals(team_a, team_b, is_home_a)
        
        # Generate probability matrix
        prob_matrix = np.zeros((max_goals, max_goals))
        
        for i in range(max_goals):
            for j in range(max_goals):
                prob_matrix[i, j] = poisson.pmf(i, expected_a) * poisson.pmf(j, expected_b)
        
        # Create DataFrame
        df = pd.DataFrame(
            prob_matrix,
            index=[f"{i}" for i in range(max_goals)],
            columns=[f"{j}" for j in range(max_goals)]
        )
        
        return df
    
    def train(self, matches_df: pd.DataFrame) -> None:
        """
        Train the model on historical data
        
        Args:
            matches_df: DataFrame with match results
        """
        self.calculate_team_stats(matches_df)
        
        # Log top attacking and defensive teams
        attack_sorted = sorted(self.team_stats.items(), 
                             key=lambda x: x[1]['attack_strength'], 
                             reverse=True)[:5]
        defense_sorted = sorted(self.team_stats.items(), 
                              key=lambda x: x[1]['defense_strength'])[:5]
        
        logger.info("Top 5 attacking teams:")
        for team, stats in attack_sorted:
            logger.info(f"  {team}: {stats['attack_strength']:.2f}")
        
        logger.info("Top 5 defensive teams (lowest conceded):")
        for team, stats in defense_sorted:
            logger.info(f"  {team}: {stats['defense_strength']:.2f}")


if __name__ == "__main__":
    # Test the Poisson predictor
    from src.data.loader import DataLoader
    
    loader = DataLoader()
    matches = loader.load_matches(processed=False)
    
    # Initialize and train
    poisson = PoissonPredictor(home_advantage=1.3)
    poisson.train(matches)
    
    # Test prediction
    prediction = poisson.predict_match('Brazil', 'Germany', is_home_a=True)
    print("\nBrazil vs Germany (Brazil home):")
    print(f"  Brazil win: {prediction['home_win']:.1%}")
    print(f"  Draw: {prediction['draw']:.1%}")
    print(f"  Germany win: {prediction['away_win']:.1%}")
    print(f"  Expected goals - Brazil: {prediction['expected_goals_home']:.2f}")
    print(f"  Expected goals - Germany: {prediction['expected_goals_away']:.2f}")
    print(f"  Most likely score: {prediction['most_likely_score']}")
    
    # Show score probability matrix
    print("\nScore Probability Matrix (%):")
    score_probs = poisson.predict_score_probabilities('Brazil', 'Germany', is_home_a=True)
    print((score_probs * 100).round(1))

# Made with Bob
