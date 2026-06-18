"""
Data Loader Module
Handles loading CSV files from the data directory
"""

import pandas as pd
from pathlib import Path
from typing import Optional, List
import logging

# Set up logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


class DataLoader:
    """Simple data loader for CSV files"""
    
    def __init__(self, data_dir: str = 'data'):
        """
        Initialize the data loader
        
        Args:
            data_dir: Base directory for data files
        """
        self.data_dir = Path(data_dir)
        self.raw_dir = self.data_dir / 'raw'
        self.processed_dir = self.data_dir / 'processed'
        self.cache_dir = self.data_dir / 'cache'
        self.predictions_dir = self.data_dir / 'predictions'
        
        # Create directories if they don't exist
        for directory in [self.raw_dir, self.processed_dir, self.cache_dir, self.predictions_dir]:
            directory.mkdir(parents=True, exist_ok=True)
    
    def load_matches(self, processed: bool = True) -> pd.DataFrame:
        """
        Load match data
        
        Args:
            processed: If True, load processed data; otherwise load raw data
            
        Returns:
            DataFrame with match data
        """
        try:
            if processed:
                file_path = self.processed_dir / 'matches_clean.csv'
                if not file_path.exists():
                    logger.warning(f"Processed file not found: {file_path}")
                    logger.info("Loading raw data instead...")
                    return self.load_matches(processed=False)
            else:
                # Try different possible filenames from Kaggle datasets
                possible_files = [
                    'WorldCupMatches.csv',
                    'world_cup_matches.csv',
                    'matches.csv'
                ]
                
                for filename in possible_files:
                    file_path = self.raw_dir / filename
                    if file_path.exists():
                        break
                else:
                    raise FileNotFoundError(
                        f"No match data found in {self.raw_dir}. "
                        f"Please download data from Kaggle first."
                    )
            
            logger.info(f"Loading matches from: {file_path}")
            df = pd.read_csv(file_path)
            logger.info(f"Loaded {len(df)} matches")
            return df
            
        except Exception as e:
            logger.error(f"Error loading matches: {e}")
            raise
    
    def load_teams(self) -> pd.DataFrame:
        """Load team statistics"""
        file_path = self.processed_dir / 'team_stats.csv'
        
        if not file_path.exists():
            logger.warning(f"Team stats not found: {file_path}")
            logger.info("You may need to run data processing first")
            return pd.DataFrame()
        
        logger.info(f"Loading team stats from: {file_path}")
        return pd.read_csv(file_path)
    
    def load_players(self) -> pd.DataFrame:
        """Load player data"""
        try:
            # Try different possible filenames
            possible_files = [
                'WorldCupPlayers.csv',
                'world_cup_players.csv',
                'players.csv'
            ]
            
            for filename in possible_files:
                file_path = self.raw_dir / filename
                if file_path.exists():
                    logger.info(f"Loading players from: {file_path}")
                    return pd.read_csv(file_path)
            
            logger.warning("No player data found")
            return pd.DataFrame()
            
        except Exception as e:
            logger.error(f"Error loading players: {e}")
            return pd.DataFrame()
    
    def load_rankings(self, ranking_type: str = 'fifa') -> pd.DataFrame:
        """
        Load FIFA or ELO rankings
        
        Args:
            ranking_type: 'fifa' or 'elo'
            
        Returns:
            DataFrame with rankings
        """
        if ranking_type == 'fifa':
            file_path = self.raw_dir / 'fifa_rankings.csv'
        else:
            file_path = self.raw_dir / 'elo_ratings.csv'
        
        if not file_path.exists():
            logger.warning(f"Rankings file not found: {file_path}")
            return pd.DataFrame()
        
        logger.info(f"Loading {ranking_type} rankings from: {file_path}")
        return pd.read_csv(file_path)
    
    def load_international_results(self) -> pd.DataFrame:
        """Load international football results dataset"""
        try:
            file_path = self.raw_dir / 'results.csv'
            
            if not file_path.exists():
                logger.warning(f"International results not found: {file_path}")
                return pd.DataFrame()
            
            logger.info(f"Loading international results from: {file_path}")
            df = pd.read_csv(file_path)
            logger.info(f"Loaded {len(df)} international matches")
            return df
            
        except Exception as e:
            logger.error(f"Error loading international results: {e}")
            return pd.DataFrame()
    
    def save_processed(self, df: pd.DataFrame, filename: str) -> Path:
        """
        Save processed data
        
        Args:
            df: DataFrame to save
            filename: Name of the file
            
        Returns:
            Path to saved file
        """
        output_path = self.processed_dir / filename
        df.to_csv(output_path, index=False)
        logger.info(f"Saved processed data to: {output_path}")
        return output_path
    
    def save_predictions(self, df: pd.DataFrame, filename: str = 'predictions.csv') -> Path:
        """
        Save prediction results
        
        Args:
            df: DataFrame with predictions
            filename: Name of the file
            
        Returns:
            Path to saved file
        """
        output_path = self.predictions_dir / filename
        df.to_csv(output_path, index=False)
        logger.info(f"Saved predictions to: {output_path}")
        return output_path
    
    def list_available_files(self) -> dict:
        """
        List all available data files
        
        Returns:
            Dictionary with file categories and their files
        """
        files = {
            'raw': list(self.raw_dir.glob('*.csv')),
            'processed': list(self.processed_dir.glob('*.csv')),
            'predictions': list(self.predictions_dir.glob('*.csv'))
        }
        
        logger.info("Available files:")
        for category, file_list in files.items():
            logger.info(f"  {category}: {len(file_list)} files")
            for file in file_list:
                logger.info(f"    - {file.name}")
        
        return files


if __name__ == "__main__":
    # Test the loader
    loader = DataLoader()
    
    print("\n=== Testing Data Loader ===\n")
    
    # List available files
    loader.list_available_files()
    
    # Try loading matches
    try:
        matches = loader.load_matches(processed=False)
        print(f"\nLoaded {len(matches)} matches")
        print("\nFirst few rows:")
        print(matches.head())
    except Exception as e:
        print(f"\nCould not load matches: {e}")
        print("Please download data from Kaggle first")

# Made with Bob
