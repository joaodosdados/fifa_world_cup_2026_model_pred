"""
FIFA Official API Client
Integration with FIFA's public API for enhanced data
Documentation: https://givevoicetofootball.github.io/api/
"""

import requests
import logging
from typing import Dict, List, Optional
import pandas as pd

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


class FIFAAPIClient:
    """
    Client for FIFA's official public API
    Provides access to:
    - Seasons data
    - Match details
    - Team statistics
    - Player information
    - Competition stages
    """
    
    BASE_URL = "https://givevoicetofootball.fifa.com/api/v1"
    
    def __init__(self):
        self.session = requests.Session()
        self.session.headers.update({
            'User-Agent': 'Mozilla/5.0 (Macintosh; Intel Mac OS X 10_15_7) AppleWebKit/537.36',
            'Accept': 'application/json'
        })
    
    def search_seasons(self, competition_name: str = "FIFA World Cup 2026") -> List[Dict]:
        """
        Search for seasons by competition name
        
        Args:
            competition_name: Name of the competition
            
        Returns:
            List of season dictionaries
        """
        try:
            url = f"{self.BASE_URL}/seasons/search"
            params = {'name': competition_name}
            
            response = self.session.get(url, params=params, timeout=10)
            response.raise_for_status()
            
            data = response.json()
            logger.info(f"Found {len(data.get('results', []))} seasons")
            return data.get('results', [])
            
        except Exception as e:
            logger.error(f"Error searching seasons: {e}")
            return []
    
    def get_season_details(self, season_id: str) -> Optional[Dict]:
        """
        Get detailed information about a specific season
        
        Args:
            season_id: Season identifier
            
        Returns:
            Season details dictionary
        """
        try:
            url = f"{self.BASE_URL}/seasons/{season_id}"
            
            response = self.session.get(url, timeout=10)
            response.raise_for_status()
            
            return response.json()
            
        except Exception as e:
            logger.error(f"Error getting season details: {e}")
            return None
    
    def get_season_stages(self, season_id: str, competition_id: str) -> List[Dict]:
        """
        Get all stages of a season
        
        Args:
            season_id: Season identifier
            competition_id: Competition identifier
            
        Returns:
            List of stage dictionaries
        """
        try:
            url = f"{self.BASE_URL}/stages"
            params = {
                'idSeason': season_id,
                'idCompetition': competition_id
            }
            
            response = self.session.get(url, params=params, timeout=10)
            response.raise_for_status()
            
            data = response.json()
            logger.info(f"Found {len(data.get('results', []))} stages")
            return data.get('results', [])
            
        except Exception as e:
            logger.error(f"Error getting stages: {e}")
            return []
    
    def get_stage_squads(self, stage_id: str) -> List[Dict]:
        """
        Get all squads (teams) in a stage
        
        Args:
            stage_id: Stage identifier
            
        Returns:
            List of squad dictionaries with team information
        """
        try:
            url = f"{self.BASE_URL}/squads"
            params = {'idStage': stage_id}
            
            response = self.session.get(url, params=params, timeout=10)
            response.raise_for_status()
            
            data = response.json()
            logger.info(f"Found {len(data.get('results', []))} squads")
            return data.get('results', [])
            
        except Exception as e:
            logger.error(f"Error getting squads: {e}")
            return []
    
    def get_stage_matches(self, stage_id: str) -> List[Dict]:
        """
        Get all matches in a stage
        
        Args:
            stage_id: Stage identifier
            
        Returns:
            List of match dictionaries
        """
        try:
            url = f"{self.BASE_URL}/matches"
            params = {'idStage': stage_id}
            
            response = self.session.get(url, params=params, timeout=10)
            response.raise_for_status()
            
            data = response.json()
            logger.info(f"Found {len(data.get('results', []))} matches")
            return data.get('results', [])
            
        except Exception as e:
            logger.error(f"Error getting matches: {e}")
            return []
    
    def get_match_details(self, match_id: str) -> Optional[Dict]:
        """
        Get detailed information about a specific match
        
        Args:
            match_id: Match identifier
            
        Returns:
            Match details dictionary with:
            - Teams
            - Score
            - Statistics
            - Events (goals, cards, substitutions)
        """
        try:
            url = f"{self.BASE_URL}/matches/{match_id}"
            
            response = self.session.get(url, timeout=10)
            response.raise_for_status()
            
            return response.json()
            
        except Exception as e:
            logger.error(f"Error getting match details: {e}")
            return None
    
    def get_team_statistics(self, team_id: str, season_id: str) -> Optional[Dict]:
        """
        Get team statistics for a season
        
        Args:
            team_id: Team identifier
            season_id: Season identifier
            
        Returns:
            Team statistics dictionary
        """
        try:
            url = f"{self.BASE_URL}/teams/{team_id}/statistics"
            params = {'idSeason': season_id}
            
            response = self.session.get(url, params=params, timeout=10)
            response.raise_for_status()
            
            return response.json()
            
        except Exception as e:
            logger.error(f"Error getting team statistics: {e}")
            return None
    
    def get_world_cup_2026_data(self) -> Dict:
        """
        Get comprehensive data for World Cup 2026
        
        Returns:
            Dictionary with all available data:
            - seasons
            - stages
            - teams
            - matches
        """
        logger.info("Fetching World Cup 2026 data from FIFA API...")
        
        # Search for World Cup 2026 season
        seasons = self.search_seasons("FIFA World Cup 2026")
        
        if not seasons:
            logger.warning("World Cup 2026 season not found in API")
            return {}
        
        season = seasons[0]
        season_id = season.get('IdSeason')
        competition_id = season.get('IdCompetition')
        
        # Get stages
        stages = self.get_season_stages(season_id, competition_id)
        
        # Get matches from all stages
        all_matches = []
        all_teams = []
        
        for stage in stages:
            stage_id = stage.get('IdStage')
            
            # Get matches
            matches = self.get_stage_matches(stage_id)
            all_matches.extend(matches)
            
            # Get teams
            squads = self.get_stage_squads(stage_id)
            all_teams.extend(squads)
        
        return {
            'season': season,
            'stages': stages,
            'matches': all_matches,
            'teams': all_teams
        }


def convert_fifa_matches_to_dataframe(matches: List[Dict]) -> pd.DataFrame:
    """
    Convert FIFA API matches to pandas DataFrame
    
    Args:
        matches: List of match dictionaries from FIFA API
        
    Returns:
        DataFrame with match information
    """
    rows = []
    
    for match in matches:
        row = {
            'match_id': match.get('IdMatch'),
            'home_team': match.get('HomeTeam', {}).get('TeamName', [{}])[0].get('Description'),
            'away_team': match.get('AwayTeam', {}).get('TeamName', [{}])[0].get('Description'),
            'home_score': match.get('HomeTeamScore'),
            'away_score': match.get('AwayTeamScore'),
            'match_date': match.get('Date'),
            'stadium': match.get('Stadium', {}).get('Name', [{}])[0].get('Description'),
            'stage': match.get('StageName', [{}])[0].get('Description'),
            'status': match.get('MatchStatus')
        }
        rows.append(row)
    
    return pd.DataFrame(rows)


if __name__ == "__main__":
    # Test the API client
    client = FIFAAPIClient()
    
    # Try to get World Cup 2026 data
    data = client.get_world_cup_2026_data()
    
    if data:
        print(f"\nSeason: {data.get('season', {}).get('Name')}")
        print(f"Stages: {len(data.get('stages', []))}")
        print(f"Matches: {len(data.get('matches', []))}")
        print(f"Teams: {len(data.get('teams', []))}")
        
        # Convert matches to DataFrame
        if data.get('matches'):
            df = convert_fifa_matches_to_dataframe(data['matches'])
            print(f"\nMatches DataFrame shape: {df.shape}")
            print(df.head())
    else:
        print("No data available from FIFA API yet")

# Made with Bob
