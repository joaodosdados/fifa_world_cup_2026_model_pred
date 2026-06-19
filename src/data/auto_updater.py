"""
Automatic Data Updater for FIFA World Cup 2026
Fetches live data from FIFA and updates the schedule CSV automatically
"""

import pandas as pd
import logging
import os
from pathlib import Path
from datetime import datetime
from typing import Any, Dict

# Selenium is required because FIFA renders the fixtures client-side.
try:
    from src.data.fifa_scraper_selenium import FIFASeleniumScraper
    SELENIUM_AVAILABLE = True
except ImportError:
    SELENIUM_AVAILABLE = False

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


class AutoDataUpdater:
    """Automatically updates match data from FIFA website"""
    
    def __init__(self, schedule_path: str = 'data/2026_world_cup_schedule.csv'):
        self.schedule_path = Path(schedule_path)
        
        # Never fall back to sample results: a failed scrape must not mutate data.
        if SELENIUM_AVAILABLE:
            logger.info("Using Selenium-based FIFA scraper")
            self.scraper = FIFASeleniumScraper()
        else:
            self.scraper = None
        
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
        """Atomically save the updated schedule CSV."""
        temporary_path = self.schedule_path.with_suffix(".tmp")
        try:
            df.to_csv(temporary_path, index=False)
            os.replace(temporary_path, self.schedule_path)
            logger.info(f"Schedule saved successfully to {self.schedule_path}")
            return True
        except Exception as e:
            logger.error(f"Error saving schedule: {e}")
            temporary_path.unlink(missing_ok=True)
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
        clean_name = str(name).strip()
        return name_mapping.get(clean_name, clean_name).casefold()
    
    def update_from_fifa(self) -> Dict[str, Any]:
        """
        Fetch data from FIFA and update schedule
        
        Returns:
            Dictionary with update statistics
        """
        logger.info("=" * 60)
        logger.info("Starting automatic data update from FIFA...")
        logger.info("=" * 60)
        
        try:
            if self.scraper is None:
                return {
                    'success': False,
                    'message': 'Selenium não está disponível neste ambiente.',
                    'updated_matches': 0,
                    'fetched_matches': 0,
                    'matched_matches': 0,
                }

            # Load current schedule
            schedule_df = self.load_schedule()
            original_completed = int(
                schedule_df["status"].astype(str).str.lower().eq("completed").sum()
            )
            
            # Fetch live data from FIFA
            logger.info("Fetching live data from FIFA website...")
            try:
                fifa_matches = self.scraper.fetch_match_results()
            except Exception as scraper_error:
                logger.warning(f"⚠️  FIFA scraper failed: {scraper_error}")
                return {
                    'success': False,
                    'message': f'Falha ao consultar a FIFA: {str(scraper_error)}',
                    'updated_matches': 0,
                    'fetched_matches': 0,
                    'matched_matches': 0,
                }
            
            if not fifa_matches:
                logger.warning("⚠️  No match data fetched from FIFA website")
                return {
                    'success': False,
                    'message': 'A FIFA não retornou partidas finalizadas.',
                    'updated_matches': 0,
                    'fetched_matches': 0,
                    'matched_matches': 0,
                }
            
            logger.info(f"✓ Fetched {len(fifa_matches)} matches from FIFA")
            
            # Update schedule with FIFA data
            updated_count = 0
            matched_count = 0
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
                    
                    same_order = (
                        schedule_home == fifa_home and schedule_away == fifa_away
                    )
                    reverse_order = (
                        schedule_home == fifa_away and schedule_away == fifa_home
                    )

                    # Group-stage CSVs sometimes list the same fixture with the
                    # opposite home/away order. Scores must be swapped in that case.
                    if same_order or reverse_order:
                        matched_count += 1
                        if same_order:
                            new_home_score = float(fifa_match['score_a'])
                            new_away_score = float(fifa_match['score_b'])
                        else:
                            new_home_score = float(fifa_match['score_b'])
                            new_away_score = float(fifa_match['score_a'])
                        current_home_score = row.get('home_score')
                        current_away_score = row.get('away_score')
                        current_status = str(row.get('status', 'Scheduled'))

                        changed = (
                            current_status.lower() != 'completed'
                            or pd.isna(current_home_score)
                            or pd.isna(current_away_score)
                            or float(current_home_score) != new_home_score
                            or float(current_away_score) != new_away_score
                        )

                        if changed:
                            schedule_df.at[idx, 'home_score'] = new_home_score
                            schedule_df.at[idx, 'away_score'] = new_away_score
                            schedule_df.at[idx, 'status'] = 'Completed'
                            updated_count += 1
                            
                            logger.info(
                                "✓ Updated: %s %s - %s %s",
                                fifa_home,
                                fifa_match['score_a'],
                                fifa_match['score_b'],
                                fifa_away,
                            )
                        break
            
            # Save updated schedule
            if updated_count > 0:
                if not self.save_schedule(schedule_df):
                    return {
                        'success': False,
                        'message': 'Resultados encontrados, mas o CSV não pôde ser salvo.',
                        'updated_matches': 0,
                        'fetched_matches': len(fifa_matches),
                        'matched_matches': matched_count,
                    }
                logger.info(f"✓ Successfully updated {updated_count} matches")
            else:
                logger.info("No new updates found")
            
            new_completed = int(
                schedule_df["status"].astype(str).str.lower().eq("completed").sum()
            )
            self.last_update = datetime.now()
            
            logger.info("=" * 60)
            logger.info(f"Update complete! Completed matches: {original_completed} → {new_completed}")
            logger.info("=" * 60)
            
            return {
                'success': True,
                'updated_matches': updated_count,
                'fetched_matches': len(fifa_matches),
                'matched_matches': matched_count,
                'total_completed': new_completed,
                'last_update': self.last_update,
                'message': (
                    f'{updated_count} resultado(s) atualizado(s). '
                    f'{matched_count} partida(s) do calendário conferida(s).'
                ),
            }
            
        except Exception as e:
            logger.error(f"Error during update: {e}")
            return {
                'success': False,
                'message': f'Error: {str(e)}',
                'updated_matches': 0,
                'fetched_matches': 0,
                'matched_matches': 0,
            }


def run_auto_update() -> Dict[str, Any]:
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
