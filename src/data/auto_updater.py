"""
Automatic Data Updater for FIFA World Cup 2026
Fetches live data from FIFA and updates the schedule CSV automatically
"""

import pandas as pd
import logging
from pathlib import Path
from datetime import datetime
from typing import Dict, List, Optional
from src.data.fifa_scraper import FIFADataScraper

# Try to import Selenium scraper, fallback to regular scraper
try:
    from src.data.fifa_scraper_selenium import FIFASeleniumScraper
    SELENIUM_AVAILABLE = True
except ImportError:
    SELENIUM_AVAILABLE = False
    logger.warning("Selenium not available, using regular scraper")

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


class AutoDataUpdater:
    """Automatically updates match data from FIFA website"""
    
    def __init__(self, schedule_path: str = 'data/2026_world_cup_schedule.csv'):
        self.schedule_path = Path(schedule_path)
        
        # Use Selenium scraper if available, otherwise use regular scraper
        if SELENIUM_AVAILABLE:
            logger.info("Using Selenium-based FIFA scraper")
            self.scraper = FIFASeleniumScraper()
        else:
            logger.info("Using requests-based FIFA scraper")
            self.scraper = FIFADataScraper()
        
        self.last_update = None
    
    def load_schedule(self) -> pd.DataFrame:
        """Load current schedule from CSV"""
        try:
            df = pd.read_csv(self.schedule_path)
            logger.info(f"Loaded schedule with {len(df)} matches")
            return df
        except Exception as e:
            logger.error(f"Error loading schedule: {e}")
            raise
    
    def save_schedule(self, df: pd.DataFrame) -> bool:
        """Save updated schedule to CSV"""
        try:
            df.to_csv(self.schedule_path, index=False)
            logger.info(f"Schedule saved successfully to {self.schedule_path}")
            return True
        except Exception as e:
            logger.error(f"Error saving schedule: {e}")
            return False
    
    def normalize_team_name(self, name: str) -> str:
        """Normalize team names for matching"""
        # Mapping from CSV format to scraper format
        name_mapping = {
            'México': 'Mexico',
            'África do Sul': 'South Africa',
            'República da Coreia': 'Korea Republic',
            'Tchéquia': 'Czech Republic',
            'Suíça': 'Switzerland',
            'Bósnia e Herzegovina': 'Bosnia and Herzegovina',
            'Canadá': 'Canada',
            'Catar': 'Qatar',
            'Escócia': 'Scotland',
            'Haiti': 'Haiti',
            'Marrocos': 'Morocco',
            'Brasil': 'Brazil',
            'EUA': 'USA',
            'Paraguai': 'Paraguay',
            'Inglaterra': 'England',
            'França': 'France',
            'Alemanha': 'Germany',
            'Espanha': 'Spain',
            'Portugal': 'Portugal',
            'Holanda': 'Netherlands',
            'Itália': 'Italy',
            'Bélgica': 'Belgium',
            'Croácia': 'Croatia',
            'Uruguai': 'Uruguay',
            'Colômbia': 'Colombia',
            'Chile': 'Chile',
            'Dinamarca': 'Denmark',
            'Suécia': 'Sweden',
            'Polônia': 'Poland',
            'Senegal': 'Senegal',
            'Nigéria': 'Nigeria',
            'Gana': 'Ghana',
            'Camarões': 'Cameroon',
            'Tunísia': 'Tunisia',
            'Egito': 'Egypt',
            'Argélia': 'Algeria',
            'Japão': 'Japan',
            'Irã': 'Iran',
            'Austrália': 'Australia',
            'Arábia Saudita': 'Saudi Arabia',
            'Costa Rica': 'Costa Rica',
            'Panamá': 'Panama',
            'Jamaica': 'Jamaica',
            'Honduras': 'Honduras',
            'Equador': 'Ecuador',
            'Peru': 'Peru',
            'Venezuela': 'Venezuela',
            'Bolívia': 'Bolivia',
            'Argentina': 'Argentina',
        }
        return name_mapping.get(name, name)
    
    def update_from_fifa(self) -> Dict[str, any]:
        """
        Fetch data from FIFA and update schedule
        
        Returns:
            Dictionary with update statistics
        """
        logger.info("=" * 60)
        logger.info("Starting automatic data update from FIFA...")
        logger.info("=" * 60)
        
        try:
            # Load current schedule
            schedule_df = self.load_schedule()
            original_completed = len(schedule_df[schedule_df['status'] == 'Completed'])
            
            # Fetch live data from FIFA
            logger.info("Fetching live data from FIFA website...")
            fifa_matches = self.scraper.fetch_match_results()
            
            if not fifa_matches:
                logger.warning("No matches fetched from FIFA")
                return {
                    'success': False,
                    'message': 'No data fetched from FIFA',
                    'updated_matches': 0
                }
            
            logger.info(f"Fetched {len(fifa_matches)} matches from FIFA")
            
            # Update schedule with FIFA data
            updated_count = 0
            for fifa_match in fifa_matches:
                if fifa_match.get('status') != 'completed':
                    continue
                
                # Normalize team names
                fifa_home = self.normalize_team_name(fifa_match.get('team_a', ''))
                fifa_away = self.normalize_team_name(fifa_match.get('team_b', ''))
                
                # Find matching row in schedule
                for idx, row in schedule_df.iterrows():
                    schedule_home = self.normalize_team_name(row['home_team'])
                    schedule_away = self.normalize_team_name(row['away_team'])
                    
                    # Check if teams match
                    if (schedule_home == fifa_home and schedule_away == fifa_away):
                        # Update only if not already completed or scores changed
                        current_status = row.get('status', 'Scheduled')
                        
                        if current_status != 'Completed' or pd.isna(row.get('home_score')):
                            schedule_df.at[idx, 'home_score'] = float(fifa_match.get('score_a', 0))
                            schedule_df.at[idx, 'away_score'] = float(fifa_match.get('score_b', 0))
                            schedule_df.at[idx, 'status'] = 'Completed'
                            updated_count += 1
                            
                            logger.info(f"✓ Updated: {fifa_home} {fifa_match.get('score_a')} - {fifa_match.get('score_b')} {fifa_away}")
            
            # Save updated schedule
            if updated_count > 0:
                self.save_schedule(schedule_df)
                logger.info(f"✓ Successfully updated {updated_count} matches")
            else:
                logger.info("No new updates found")
            
            new_completed = len(schedule_df[schedule_df['status'] == 'Completed'])
            self.last_update = datetime.now()
            
            logger.info("=" * 60)
            logger.info(f"Update complete! Completed matches: {original_completed} → {new_completed}")
            logger.info("=" * 60)
            
            return {
                'success': True,
                'updated_matches': updated_count,
                'total_completed': new_completed,
                'last_update': self.last_update,
                'message': f'Updated {updated_count} matches from FIFA'
            }
            
        except Exception as e:
            logger.error(f"Error during update: {e}")
            return {
                'success': False,
                'message': f'Error: {str(e)}',
                'updated_matches': 0
            }


def run_auto_update() -> Dict[str, any]:
    """
    Convenience function to run automatic update
    
    Returns:
        Update statistics
    """
    updater = AutoDataUpdater()
    return updater.update_from_fifa()


if __name__ == "__main__":
    # Test the updater
    result = run_auto_update()
    print(f"\nUpdate Result: {result}")

# Made with Bob
