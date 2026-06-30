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
            'Mexico': 'Mexico',
            'África do Sul': 'South Africa',
            'South Africa': 'South Africa',
            'República da Coreia': 'Korea Republic',
            'Korea Republic': 'Korea Republic',
            'Tchéquia': 'Czech Republic',
            'Czech Republic': 'Czech Republic',
            'Suíça': 'Switzerland',
            'Switzerland': 'Switzerland',
            'Bósnia e Herzegovina': 'Bosnia and Herzegovina',
            'Bosnia and Herzegovina': 'Bosnia and Herzegovina',
            'Canadá': 'Canada',
            'Canada': 'Canada',
            'Catar': 'Qatar',
            'Qatar': 'Qatar',
            'Escócia': 'Scotland',
            'Scotland': 'Scotland',
            'Haiti': 'Haiti',
            'Marrocos': 'Morocco',
            'Morocco': 'Morocco',
            'Brasil': 'Brazil',
            'Brazil': 'Brazil',
            'EUA': 'USA',
            'USA': 'USA',
            'Estados Unidos': 'USA',
            'Paraguai': 'Paraguay',
            'Paraguay': 'Paraguay',
            'Inglaterra': 'England',
            'England': 'England',
            'França': 'France',
            'France': 'France',
            'Alemanha': 'Germany',
            'Germany': 'Germany',
            'Espanha': 'Spain',
            'Spain': 'Spain',
            'Portugal': 'Portugal',
            'Holanda': 'Netherlands',
            'Netherlands': 'Netherlands',
            'Itália': 'Italy',
            'Italy': 'Italy',
            'Bélgica': 'Belgium',
            'Belgium': 'Belgium',
            'Croácia': 'Croatia',
            'Croatia': 'Croatia',
            'Uruguai': 'Uruguay',
            'Uruguay': 'Uruguay',
            'Colômbia': 'Colombia',
            'Colombia': 'Colombia',
            'Chile': 'Chile',
            'Dinamarca': 'Denmark',
            'Denmark': 'Denmark',
            'Suécia': 'Sweden',
            'Sweden': 'Sweden',
            'Polônia': 'Poland',
            'Poland': 'Poland',
            'Senegal': 'Senegal',
            'Nigéria': 'Nigeria',
            'Nigeria': 'Nigeria',
            'Gana': 'Ghana',
            'Ghana': 'Ghana',
            'Camarões': 'Cameroon',
            'Cameroon': 'Cameroon',
            'Tunísia': 'Tunisia',
            'Tunisia': 'Tunisia',
            'Egito': 'Egypt',
            'Egypt': 'Egypt',
            'Argélia': 'Algeria',
            'Algeria': 'Algeria',
            'Japão': 'Japan',
            'Japan': 'Japan',
            'Irã': 'Iran',
            'RI do Irã': 'Iran',
            'IR Iran': 'Iran',
            'Iran': 'Iran',
            'Austrália': 'Australia',
            'Australia': 'Australia',
            'Arábia Saudita': 'Saudi Arabia',
            'Saudi Arabia': 'Saudi Arabia',
            'Costa Rica': 'Costa Rica',
            'Panamá': 'Panama',
            'Panama': 'Panama',
            'Jamaica': 'Jamaica',
            'Honduras': 'Honduras',
            'Equador': 'Ecuador',
            'Ecuador': 'Ecuador',
            'Peru': 'Peru',
            'Venezuela': 'Venezuela',
            'Bolívia': 'Bolivia',
            'Bolivia': 'Bolivia',
            'Argentina': 'Argentina',
            'Noruega': 'Norway',
            'Norway': 'Norway',
            'Iraque': 'Iraq',
            'Iraq': 'Iraq',
            'Cabo Verde': 'Cape Verde',
            'Cape Verde': 'Cape Verde',
            'Curaçau': 'Curacao',
            'Curaçao': 'Curacao',
            'Curacao': 'Curacao',
            'Costa do Marfim': 'Ivory Coast',
            "Côte d'Ivoire": 'Ivory Coast',
            'Ivory Coast': 'Ivory Coast',
            'Turquia': 'Turkey',
            'Türkiye': 'Turkey',
            'Turkey': 'Turkey',
            'Nova Zelândia': 'New Zealand',
            'New Zealand': 'New Zealand',
            'Áustria': 'Austria',
            'Austria': 'Austria',
            'Jordânia': 'Jordan',
            'Jordan': 'Jordan',
            'Uzbequistão': 'Uzbekistan',
            'Uzbekistan': 'Uzbekistan',
            'RD do Congo': 'DR Congo',
            'Congo DR': 'DR Congo',
            'DR Congo': 'DR Congo',
        }
        clean_name = str(name).strip()
        return name_mapping.get(clean_name, clean_name).casefold()

    @staticmethod
    def _values_differ(current_value: Any, new_value: Any) -> bool:
        """Return True when a schedule cell needs to be updated."""
        if pd.isna(current_value) and pd.isna(new_value):
            return False
        if pd.isna(current_value) and new_value in (None, ""):
            return False
        return str(current_value) != str(new_value)

    def _find_fixture_index(
        self,
        schedule_df: pd.DataFrame,
        fifa_home: str,
        fifa_away: str,
    ) -> int | None:
        """Find a schedule row matching the FIFA fixture, ignoring team order."""
        for idx, row in schedule_df.iterrows():
            schedule_home = self.normalize_team_name(row['home_team'])
            schedule_away = self.normalize_team_name(row['away_team'])
            if {schedule_home, schedule_away} == {fifa_home, fifa_away}:
                return idx
        return None

    @staticmethod
    def _next_match_id(schedule_df: pd.DataFrame) -> int:
        if "match_id" not in schedule_df.columns or schedule_df.empty:
            return 1
        ids = pd.to_numeric(schedule_df["match_id"], errors="coerce").dropna()
        return int(ids.max()) + 1 if not ids.empty else len(schedule_df) + 1

    def _new_fixture_row(
        self,
        schedule_df: pd.DataFrame,
        fifa_match: Dict[str, Any],
        match_id: int,
    ) -> Dict[str, Any]:
        new_status = (
            "Completed"
            if fifa_match.get("status") == "completed"
            else "Scheduled"
        )
        row = {
            "match_id": match_id,
            "group": fifa_match.get("group") or "",
            "home_team": fifa_match.get("team_a"),
            "away_team": fifa_match.get("team_b"),
            "date": fifa_match.get("date"),
            "time": fifa_match.get("time") or "",
            "venue": fifa_match.get("venue") or "",
            "stage": fifa_match.get("stage") or "",
            "home_score": pd.NA,
            "away_score": pd.NA,
            "status": new_status,
        }

        if new_status == "Completed":
            row["home_score"] = float(fifa_match["score_a"])
            row["away_score"] = float(fifa_match["score_b"])

        return {
            column: row.get(column, pd.NA)
            for column in schedule_df.columns
        }
    
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
                if hasattr(self.scraper, "fetch_fixtures"):
                    fifa_matches = self.scraper.fetch_fixtures()
                else:
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
                    'message': 'A FIFA não retornou partidas.',
                    'updated_matches': 0,
                    'fetched_matches': 0,
                    'matched_matches': 0,
                }
            
            logger.info(f"✓ Fetched {len(fifa_matches)} matches from FIFA")
            
            # Update schedule with FIFA data
            updated_count = 0
            matched_count = 0
            appended_count = 0
            next_match_id = self._next_match_id(schedule_df)
            for fifa_match in fifa_matches:
                fifa_home = self.normalize_team_name(fifa_match.get('team_a', ''))
                fifa_away = self.normalize_team_name(fifa_match.get('team_b', ''))
                idx = self._find_fixture_index(schedule_df, fifa_home, fifa_away)
                if idx is None:
                    if fifa_match.get("stage") == "Group Stage":
                        continue
                    if not fifa_match.get("team_a") or not fifa_match.get("team_b"):
                        continue

                    schedule_df.loc[len(schedule_df)] = self._new_fixture_row(
                        schedule_df,
                        fifa_match,
                        next_match_id,
                    )
                    next_match_id += 1
                    appended_count += 1
                    matched_count += 1
                    logger.info(
                        "✓ Added fixture: %s vs %s (%s %s)",
                        fifa_match.get("team_a"),
                        fifa_match.get("team_b"),
                        fifa_match.get("date"),
                        fifa_match.get("time") or fifa_match.get("status"),
                    )
                    continue

                matched_count += 1
                current_row = schedule_df.loc[idx]
                has_fixture_metadata = bool(
                    fifa_match.get("date")
                    or fifa_match.get("time")
                    or fifa_match.get("venue")
                )
                new_status = (
                    "Completed"
                    if fifa_match.get('status') == 'completed'
                    else "Scheduled"
                )
                if has_fixture_metadata:
                    new_values = {
                        "group": fifa_match.get("group") or current_row.get("group"),
                        "home_team": fifa_match.get("team_a"),
                        "away_team": fifa_match.get("team_b"),
                        "date": fifa_match.get("date"),
                        "time": fifa_match.get("time") or current_row.get("time"),
                        "venue": fifa_match.get("venue") or current_row.get("venue"),
                        "stage": fifa_match.get("stage") or current_row.get("stage"),
                        "status": new_status,
                    }
                else:
                    new_values = {"status": new_status}

                if new_status == "Completed":
                    if has_fixture_metadata:
                        new_values["home_score"] = float(fifa_match["score_a"])
                        new_values["away_score"] = float(fifa_match["score_b"])
                    else:
                        schedule_home = self.normalize_team_name(
                            current_row["home_team"]
                        )
                        same_order = schedule_home == fifa_home
                        new_values["home_score"] = float(
                            fifa_match["score_a" if same_order else "score_b"]
                        )
                        new_values["away_score"] = float(
                            fifa_match["score_b" if same_order else "score_a"]
                        )
                else:
                    new_values["home_score"] = pd.NA
                    new_values["away_score"] = pd.NA

                changed = False
                for column, new_value in new_values.items():
                    if column not in schedule_df.columns:
                        continue
                    if self._values_differ(current_row.get(column), new_value):
                        schedule_df.at[idx, column] = new_value
                        changed = True

                if changed:
                    updated_count += 1
                    logger.info(
                        "✓ Updated fixture: %s vs %s (%s %s)",
                        fifa_match.get("team_a"),
                        fifa_match.get("team_b"),
                        fifa_match.get("date"),
                        fifa_match.get("time") or new_status,
                    )
            
            # Save updated schedule
            changed_count = updated_count + appended_count
            if changed_count > 0:
                if not self.save_schedule(schedule_df):
                    return {
                        'success': False,
                        'message': 'Resultados encontrados, mas o CSV não pôde ser salvo.',
                        'updated_matches': 0,
                        'fetched_matches': len(fifa_matches),
                        'matched_matches': matched_count,
                    }
                logger.info(
                    "✓ Successfully changed %d matches (%d updated, %d added)",
                    changed_count,
                    updated_count,
                    appended_count,
                )
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
                'updated_matches': changed_count,
                'changed_matches': changed_count,
                'appended_matches': appended_count,
                'fetched_matches': len(fifa_matches),
                'matched_matches': matched_count,
                'total_completed': new_completed,
                'last_update': self.last_update,
                'message': (
                    f'{changed_count} partida(s) alterada(s). '
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
