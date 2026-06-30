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
            df = self._sanitize_matches(df)
            logger.info("Loaded %d valid matches", len(df))
            return df
            
        except Exception as e:
            logger.error(f"Error loading matches: {e}")
            raise

    def load_current_world_cup_matches(
        self,
        schedule_path: str | Path | None = None,
        stages: list[str] | None = None,
    ) -> pd.DataFrame:
        """
        Load completed 2026 World Cup matches from the dashboard schedule and
        adapt them to the historical training schema.

        Scheduled fixtures are deliberately ignored because they do not have a
        target result yet. The returned rows can be appended to either the
        WorldCupMatches dataset or the expanded international dataset.
        """
        file_path = Path(schedule_path) if schedule_path else self.data_dir / "2026_world_cup_schedule.csv"
        if not file_path.exists():
            logger.warning("2026 schedule not found: %s", file_path)
            return pd.DataFrame()

        schedule = pd.read_csv(file_path)
        required = [
            "match_id",
            "home_team",
            "away_team",
            "date",
            "home_score",
            "away_score",
            "status",
        ]
        missing = [column for column in required if column not in schedule.columns]
        if missing:
            raise ValueError(f"2026 schedule is missing columns: {missing}")

        completed = schedule[
            schedule["status"].astype(str).str.lower().eq("completed")
        ].copy()
        if stages and "stage" in completed.columns:
            completed = completed[completed["stage"].isin(stages)].copy()
        if completed.empty:
            return pd.DataFrame()

        completed["home_score"] = pd.to_numeric(
            completed["home_score"], errors="coerce"
        )
        completed["away_score"] = pd.to_numeric(
            completed["away_score"], errors="coerce"
        )
        completed = completed.dropna(
            subset=["home_team", "away_team", "home_score", "away_score"]
        )
        if completed.empty:
            return pd.DataFrame()

        time_values = (
            completed["time"].fillna("00:00")
            if "time" in completed.columns
            else "00:00"
        )
        dates = pd.to_datetime(
            completed["date"].astype(str) + " " + time_values.astype(str),
            errors="coerce",
        )

        adapted = pd.DataFrame({
            "Year": dates.dt.year.fillna(2026).astype(int),
            "Datetime": completed["date"].astype(str),
            "Home Team Name": completed["home_team"].astype(str).str.strip(),
            "Away Team Name": completed["away_team"].astype(str).str.strip(),
            "Home Team Goals": completed["home_score"].astype(float),
            "Away Team Goals": completed["away_score"].astype(float),
            "Tournament": "FIFA World Cup",
            "Neutral": ~completed["home_team"].isin(["México", "Canadá", "EUA"]),
            "MatchID": 2_026_000 + pd.to_numeric(
                completed["match_id"], errors="coerce"
            ).fillna(0).astype(int),
        })
        return self._sanitize_matches(adapted)

    def append_current_world_cup_results(
        self,
        matches: pd.DataFrame,
        schedule_path: str | Path | None = None,
        stages: list[str] | None = None,
    ) -> pd.DataFrame:
        """Append completed 2026 matches to an existing training frame."""
        current_matches = self.load_current_world_cup_matches(schedule_path, stages)
        if current_matches.empty:
            return matches.reset_index(drop=True)

        combined = pd.concat([matches, current_matches], ignore_index=True)
        combined = self._sanitize_matches(combined)
        logger.info(
            "Training data now includes %d completed 2026 matches",
            len(current_matches),
        )
        return combined.reset_index(drop=True)

    @staticmethod
    def _sanitize_matches(df: pd.DataFrame) -> pd.DataFrame:
        """Remove empty/duplicated rows and enforce the prediction schema."""
        required = [
            'Home Team Name',
            'Away Team Name',
            'Home Team Goals',
            'Away Team Goals',
        ]
        missing = [column for column in required if column not in df.columns]
        if missing:
            raise ValueError(f"Match dataset is missing columns: {missing}")

        clean = df.dropna(subset=required).copy()
        clean['Home Team Name'] = clean['Home Team Name'].astype(str).str.strip()
        clean['Away Team Name'] = clean['Away Team Name'].astype(str).str.strip()
        for column in ['Home Team Name', 'Away Team Name']:
            clean[column] = clean[column].str.replace(
                r'^.*">', '', regex=True
            ).str.strip()
        clean = clean[
            clean['Home Team Name'].ne('')
            & clean['Away Team Name'].ne('')
            & clean['Home Team Name'].str.lower().ne('nan')
            & clean['Away Team Name'].str.lower().ne('nan')
        ]
        clean['Home Team Goals'] = pd.to_numeric(
            clean['Home Team Goals'], errors='coerce'
        )
        clean['Away Team Goals'] = pd.to_numeric(
            clean['Away Team Goals'], errors='coerce'
        )
        clean = clean.dropna(subset=['Home Team Goals', 'Away Team Goals'])

        dedupe_columns = [
            column
            for column in ['MatchID', 'Year', 'Home Team Name', 'Away Team Name']
            if column in clean.columns
        ]
        clean = clean.drop_duplicates(subset=dedupe_columns or required)
        return clean.reset_index(drop=True)
    
    def load_international_matches(
        self,
        include_friendlies: bool = True,
        min_year: int | None = None,
        max_date: str | None = None,
        min_team_matches: int | None = 30,
    ) -> pd.DataFrame:
        """
        Load the expanded international results dataset (World Cup,
        qualifiers, continental tournaments, friendlies) and adapt it to
        the same schema produced by load_matches(), so it can be used as a
        drop-in replacement in the training pipeline.

        Args:
            include_friendlies: If False, excludes "Friendly" matches,
                keeping only competitive/official matches.
            min_year: If set, drop matches before this year.
            max_date: If set, drop matches after this YYYY-MM-DD date.
            min_team_matches: If set, keeps only teams with at least this many
                matches inside the selected date window.

        Returns:
            DataFrame with the same columns as load_matches(): 'Year',
            'Datetime', 'Home Team Name', 'Away Team Name',
            'Home Team Goals', 'Away Team Goals', 'MatchID'.
        """
        file_path = self.raw_dir / 'international_results.csv'
        if not file_path.exists():
            raise FileNotFoundError(
                f"{file_path} não encontrado. Rode "
                "scripts/fetch_external_data.py primeiro."
            )

        logger.info(f"Loading international matches from: {file_path}")
        df = pd.read_csv(file_path)

        if not include_friendlies:
            df = df[df['tournament'].ne('Friendly')]

        df = df.dropna(subset=['home_team', 'away_team', 'home_score', 'away_score'])

        adapted = pd.DataFrame({
            'Datetime': df['date'],
            'Year': pd.to_datetime(df['date'], errors='coerce').dt.year,
            'Home Team Name': df['home_team'].astype(str).str.strip(),
            'Away Team Name': df['away_team'].astype(str).str.strip(),
            'Home Team Goals': pd.to_numeric(df['home_score'], errors='coerce'),
            'Away Team Goals': pd.to_numeric(df['away_score'], errors='coerce'),
            'Tournament': df['tournament'].astype(str).str.strip(),
            'Neutral': df['neutral'].astype(bool),
        })
        adapted = adapted.dropna(
            subset=['Year', 'Home Team Goals', 'Away Team Goals']
        )
        if min_year is not None:
            adapted = adapted[adapted['Year'] >= min_year]
        if max_date is not None:
            cutoff = pd.to_datetime(max_date, errors='coerce')
            match_dates = pd.to_datetime(adapted['Datetime'], errors='coerce')
            adapted = adapted[match_dates <= cutoff]
        if min_team_matches:
            team_counts = pd.concat(
                [adapted['Home Team Name'], adapted['Away Team Name']]
            ).value_counts()
            eligible = set(team_counts[team_counts >= min_team_matches].index)
            adapted = adapted[
                adapted['Home Team Name'].isin(eligible)
                & adapted['Away Team Name'].isin(eligible)
            ]

        adapted = adapted.sort_values(
            ['Datetime'], kind='stable'
        ).reset_index(drop=True)
        adapted['MatchID'] = adapted.index
        adapted['Year'] = adapted['Year'].astype(int)

        logger.info("Loaded %d international matches", len(adapted))
        return adapted

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
        output_path.parent.mkdir(parents=True, exist_ok=True)
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
        output_path.parent.mkdir(parents=True, exist_ok=True)
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
