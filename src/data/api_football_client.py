"""
API-Football (API-Sports) Client
Commercial API with comprehensive football data
Documentation: https://www.api-football.com/documentation-v3
"""

import requests
import logging
from typing import Dict, List, Optional
import pandas as pd
from datetime import datetime
import os

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


class APIFootballClient:
    """
    Client for API-Football (API-Sports)
    
    Features:
    - Live scores and fixtures
    - Team statistics
    - Player statistics
    - Head-to-head records
    - Predictions
    - Standings
    - Odds
    
    Pricing:
    - Free tier: 100 requests/day
    - Paid tiers: More requests and features
    
    Get API key: https://www.api-football.com/
    """
    
    BASE_URL = "https://v3.football.api-sports.io"
    
    def __init__(self, api_key: Optional[str] = None):
        """
        Initialize API-Football client
        
        Args:
            api_key: API key from api-football.com
                    If not provided, will try to get from environment variable API_FOOTBALL_KEY
        """
        self.api_key = api_key or os.getenv('API_FOOTBALL_KEY')
        
        if not self.api_key:
            logger.warning("No API key provided. Set API_FOOTBALL_KEY environment variable or pass api_key parameter")
        
        self.session = requests.Session()
        self.session.headers.update({
            'x-rapidapi-key': self.api_key,
            'x-rapidapi-host': 'v3.football.api-sports.io'
        })
    
    def _make_request(self, endpoint: str, params: Dict = None) -> Optional[Dict]:
        """Make API request with error handling"""
        try:
            url = f"{self.BASE_URL}/{endpoint}"
            response = self.session.get(url, params=params, timeout=10)
            response.raise_for_status()
            
            data = response.json()
            
            # Check API response
            if data.get('errors'):
                logger.error(f"API errors: {data['errors']}")
                return None
            
            return data
            
        except Exception as e:
            logger.error(f"Error making request to {endpoint}: {e}")
            return None
    
    def get_world_cup_2026_league_id(self) -> Optional[int]:
        """
        Get the league ID for World Cup 2026
        
        Returns:
            League ID or None
        """
        # World Cup league ID is typically 1 (FIFA World Cup)
        # Season 2026
        return 1
    
    def get_fixtures(self, league_id: int = 1, season: int = 2026) -> List[Dict]:
        """
        Get all fixtures for World Cup 2026
        
        Args:
            league_id: League ID (1 for World Cup)
            season: Season year (2026)
            
        Returns:
            List of fixture dictionaries
        """
        data = self._make_request('fixtures', {
            'league': league_id,
            'season': season
        })
        
        if data and data.get('response'):
            logger.info(f"Found {len(data['response'])} fixtures")
            return data['response']
        
        return []
    
    def get_live_fixtures(self) -> List[Dict]:
        """
        Get all live fixtures
        
        Returns:
            List of live fixture dictionaries
        """
        data = self._make_request('fixtures', {'live': 'all'})
        
        if data and data.get('response'):
            # Filter for World Cup matches
            wc_matches = [f for f in data['response'] 
                         if f.get('league', {}).get('id') == 1]
            logger.info(f"Found {len(wc_matches)} live World Cup matches")
            return wc_matches
        
        return []
    
    def get_fixture_details(self, fixture_id: int) -> Optional[Dict]:
        """
        Get detailed information about a specific fixture
        
        Args:
            fixture_id: Fixture ID
            
        Returns:
            Fixture details dictionary
        """
        data = self._make_request('fixtures', {'id': fixture_id})
        
        if data and data.get('response'):
            return data['response'][0]
        
        return None
    
    def get_fixture_statistics(self, fixture_id: int) -> List[Dict]:
        """
        Get match statistics
        
        Args:
            fixture_id: Fixture ID
            
        Returns:
            List of team statistics
        """
        data = self._make_request('fixtures/statistics', {'fixture': fixture_id})
        
        if data and data.get('response'):
            return data['response']
        
        return []
    
    def get_team_statistics(self, team_id: int, league_id: int = 1, season: int = 2026) -> Optional[Dict]:
        """
        Get team statistics for a season
        
        Args:
            team_id: Team ID
            league_id: League ID
            season: Season year
            
        Returns:
            Team statistics dictionary
        """
        data = self._make_request('teams/statistics', {
            'team': team_id,
            'league': league_id,
            'season': season
        })
        
        if data and data.get('response'):
            return data['response']
        
        return None
    
    def get_head_to_head(self, team1_id: int, team2_id: int) -> List[Dict]:
        """
        Get head-to-head matches between two teams
        
        Args:
            team1_id: First team ID
            team2_id: Second team ID
            
        Returns:
            List of H2H matches
        """
        data = self._make_request('fixtures/headtohead', {
            'h2h': f"{team1_id}-{team2_id}"
        })
        
        if data and data.get('response'):
            logger.info(f"Found {len(data['response'])} H2H matches")
            return data['response']
        
        return []
    
    def get_predictions(self, fixture_id: int) -> Optional[Dict]:
        """
        Get AI predictions for a fixture
        
        Args:
            fixture_id: Fixture ID
            
        Returns:
            Predictions dictionary with:
            - Winner prediction
            - Win probabilities
            - Goals predictions
            - Advice
        """
        data = self._make_request('predictions', {'fixture': fixture_id})
        
        if data and data.get('response'):
            return data['response'][0]
        
        return None
    
    def get_standings(self, league_id: int = 1, season: int = 2026) -> List[Dict]:
        """
        Get league standings
        
        Args:
            league_id: League ID
            season: Season year
            
        Returns:
            List of standings by group
        """
        data = self._make_request('standings', {
            'league': league_id,
            'season': season
        })
        
        if data and data.get('response'):
            return data['response']
        
        return []
    
    def search_team(self, team_name: str) -> List[Dict]:
        """
        Search for a team by name
        
        Args:
            team_name: Team name to search
            
        Returns:
            List of matching teams
        """
        data = self._make_request('teams', {'search': team_name})
        
        if data and data.get('response'):
            return data['response']
        
        return []


def convert_api_football_fixtures_to_dataframe(fixtures: List[Dict]) -> pd.DataFrame:
    """
    Convert API-Football fixtures to pandas DataFrame
    
    Args:
        fixtures: List of fixture dictionaries
        
    Returns:
        DataFrame with match information
    """
    rows = []
    
    for fixture in fixtures:
        row = {
            'fixture_id': fixture['fixture']['id'],
            'date': fixture['fixture']['date'],
            'status': fixture['fixture']['status']['long'],
            'home_team': fixture['teams']['home']['name'],
            'away_team': fixture['teams']['away']['name'],
            'home_score': fixture['goals']['home'],
            'away_score': fixture['goals']['away'],
            'venue': fixture['fixture']['venue']['name'],
            'city': fixture['fixture']['venue']['city'],
            'round': fixture['league']['round']
        }
        rows.append(row)
    
    return pd.DataFrame(rows)


if __name__ == "__main__":
    # Test the API client
    print("API-Football Client Test")
    print("=" * 50)
    
    # Check for API key
    api_key = os.getenv('API_FOOTBALL_KEY')
    if not api_key:
        print("\n⚠️  No API key found!")
        print("To use API-Football:")
        print("1. Sign up at https://www.api-football.com/")
        print("2. Get your API key")
        print("3. Set environment variable: export API_FOOTBALL_KEY='your-key'")
        print("\nFree tier: 100 requests/day")
    else:
        client = APIFootballClient(api_key)
        
        print(f"\n✓ API key configured")
        print("\nTesting API...")
        
        # Try to get World Cup fixtures
        fixtures = client.get_fixtures(league_id=1, season=2026)
        
        if fixtures:
            print(f"\n✓ Found {len(fixtures)} World Cup 2026 fixtures")
            
            # Convert to DataFrame
            df = convert_api_football_fixtures_to_dataframe(fixtures)
            print(f"\nFixtures DataFrame shape: {df.shape}")
            print(df.head())
        else:
            print("\n⚠️  No fixtures found (World Cup 2026 data may not be available yet)")

# Made with Bob
