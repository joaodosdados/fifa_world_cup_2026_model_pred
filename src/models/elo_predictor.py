"""
ELO Rating System for Football Match Prediction
Based on professional betting house methodologies
"""

import pandas as pd
import numpy as np
from typing import Dict, Tuple
import logging

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


class ELOPredictor:
    """
    ELO rating system for predicting football match outcomes
    """
    
    def __init__(self, k_factor: int = 32, home_advantage: float = 100):
        """
        Initialize ELO predictor
        
        Args:
            k_factor: K-factor for rating updates (higher = more volatile)
            home_advantage: ELO points advantage for home team
        """
        self.k_factor = k_factor
        self.home_advantage = home_advantage
        self.ratings = {}
        self.initial_rating = 1500
        
    def get_rating(self, team: str) -> float:
        """Get current ELO rating for a team"""
        if team not in self.ratings:
            self.ratings[team] = self.initial_rating
        return self.ratings[team]
    
    def expected_score(self, rating_a: float, rating_b: float) -> float:
        """
        Calculate expected score for team A
        
        Args:
            rating_a: ELO rating of team A
            rating_b: ELO rating of team B
            
        Returns:
            Expected score (0 to 1)
        """
        return 1 / (1 + 10 ** ((rating_b - rating_a) / 400))
    
    def update_ratings(self, team_a: str, team_b: str, score_a: float, 
                      is_home_a: bool = True) -> Tuple[float, float]:
        """
        Update ELO ratings after a match
        
        Args:
            team_a: Name of team A
            team_b: Name of team B
            score_a: Actual score for team A (1=win, 0.5=draw, 0=loss)
            is_home_a: Whether team A is playing at home
            
        Returns:
            Tuple of (new_rating_a, new_rating_b)
        """
        rating_a = self.get_rating(team_a)
        rating_b = self.get_rating(team_b)
        
        # Apply home advantage
        if is_home_a:
            rating_a_adjusted = rating_a + self.home_advantage
            rating_b_adjusted = rating_b
        else:
            rating_a_adjusted = rating_a
            rating_b_adjusted = rating_b + self.home_advantage
        
        # Calculate expected scores
        expected_a = self.expected_score(rating_a_adjusted, rating_b_adjusted)
        expected_b = 1 - expected_a
        
        # Update ratings
        new_rating_a = rating_a + self.k_factor * (score_a - expected_a)
        new_rating_b = rating_b + self.k_factor * ((1 - score_a) - expected_b)
        
        self.ratings[team_a] = new_rating_a
        self.ratings[team_b] = new_rating_b
        
        return new_rating_a, new_rating_b
    
    def predict_match(self, team_a: str, team_b: str, 
                     is_home_a: bool = True) -> Dict[str, float]:
        """
        Predict match outcome probabilities
        
        Args:
            team_a: Name of team A
            team_b: Name of team B
            is_home_a: Whether team A is playing at home
            
        Returns:
            Dictionary with win/draw/loss probabilities
        """
        rating_a = self.get_rating(team_a)
        rating_b = self.get_rating(team_b)
        
        # Apply home advantage
        if is_home_a:
            rating_a_adjusted = rating_a + self.home_advantage
            rating_b_adjusted = rating_b
        else:
            rating_a_adjusted = rating_a
            rating_b_adjusted = rating_b + self.home_advantage
        
        # Calculate expected score
        expected_a = self.expected_score(rating_a_adjusted, rating_b_adjusted)
        
        # Convert to match outcome probabilities
        # Using empirical conversion from ELO to match outcomes
        draw_prob = 0.25  # Base draw probability
        
        if expected_a > 0.5:
            win_a = expected_a - (draw_prob / 2)
            win_b = (1 - expected_a) - (draw_prob / 2)
        else:
            win_b = (1 - expected_a) - (draw_prob / 2)
            win_a = expected_a - (draw_prob / 2)
        
        # Ensure probabilities sum to 1
        total = win_a + draw_prob + win_b
        
        return {
            'home_win': win_a / total,
            'draw': draw_prob / total,
            'away_win': win_b / total,
            'home_rating': rating_a,
            'away_rating': rating_b
        }
    
    def train_on_historical_data(self, matches_df: pd.DataFrame) -> None:
        """
        Train ELO ratings on historical match data
        
        Args:
            matches_df: DataFrame with columns: 
                       ['Home Team Name', 'Away Team Name', 
                        'Home Team Goals', 'Away Team Goals', 'Year']
        """
        logger.info("Training ELO ratings on historical data...")
        
        # Sort by date/year
        if 'Year' in matches_df.columns:
            matches_df = matches_df.sort_values('Year')
        
        for idx, row in matches_df.iterrows():
            home_team = row['Home Team Name']
            away_team = row['Away Team Name']
            home_goals = row['Home Team Goals']
            away_goals = row['Away Team Goals']
            
            # Determine match result
            if home_goals > away_goals:
                score = 1.0  # Home win
            elif home_goals < away_goals:
                score = 0.0  # Away win
            else:
                score = 0.5  # Draw
            
            # Update ratings
            self.update_ratings(home_team, away_team, score, is_home_a=True)
        
        logger.info(f"Training complete. {len(self.ratings)} teams rated.")
        
        # Log top teams
        top_teams = sorted(self.ratings.items(), key=lambda x: x[1], reverse=True)[:10]
        logger.info("Top 10 teams by ELO rating:")
        for team, rating in top_teams:
            logger.info(f"  {team}: {rating:.0f}")
    
    def get_all_ratings(self) -> pd.DataFrame:
        """
        Get all team ratings as DataFrame
        
        Returns:
            DataFrame with team names and ratings
        """
        ratings_df = pd.DataFrame([
            {'team': team, 'elo_rating': rating}
            for team, rating in self.ratings.items()
        ])
        return ratings_df.sort_values('elo_rating', ascending=False)


if __name__ == "__main__":
    # Test the ELO predictor
    from src.data.loader import DataLoader
    
    loader = DataLoader()
    matches = loader.load_matches(processed=False)
    
    # Initialize and train
    elo = ELOPredictor(k_factor=32, home_advantage=100)
    elo.train_on_historical_data(matches)
    
    # Test prediction
    prediction = elo.predict_match('Brazil', 'Germany', is_home_a=True)
    print("\nBrazil vs Germany (Brazil home):")
    print(f"  Brazil win: {prediction['home_win']:.1%}")
    print(f"  Draw: {prediction['draw']:.1%}")
    print(f"  Germany win: {prediction['away_win']:.1%}")
    
    # Show top ratings
    print("\nTop 10 ELO Ratings:")
    print(elo.get_all_ratings().head(10))

# Made with Bob
