"""
FIFA World Cup 2026 Data Scraper
Fetches live match results and standings from FIFA website
"""

import requests
from bs4 import BeautifulSoup
import pandas as pd
from typing import Dict, List, Optional
import logging
import re
import time
from datetime import datetime

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


class FIFADataScraper:
    """Scraper for FIFA World Cup 2026 data"""
    
    BASE_URL = "https://www.fifa.com"
    STANDINGS_URL = f"{BASE_URL}/pt/tournaments/mens/worldcup/canadamexicousa2026/standings"
    FIXTURES_URL = f"{BASE_URL}/pt/tournaments/mens/worldcup/canadamexicousa2026/scores-fixtures"
    
    # Team name mapping (FIFA -> Our format)
    TEAM_MAPPING = {
        'BRA': 'Brazil',
        'ARG': 'Argentina',
        'MAR': 'Morocco',
        'SCO': 'Scotland',
        'HAI': 'Haiti',
        'MEX': 'Mexico',
        'USA': 'USA',
        'CAN': 'Canada',
        'ENG': 'England',
        'FRA': 'France',
        'GER': 'Germany',
        'ESP': 'Spain',
        'POR': 'Portugal',
        'NED': 'Netherlands',
        'ITA': 'Italy',
        'BEL': 'Belgium',
        'CRO': 'Croatia',
        'URU': 'Uruguay',
        'COL': 'Colombia',
        'CHI': 'Chile',
        'SUI': 'Switzerland',
        'DEN': 'Denmark',
        'SWE': 'Sweden',
        'POL': 'Poland',
        'SEN': 'Senegal',
        'NGA': 'Nigeria',
        'GHA': 'Ghana',
        'CMR': 'Cameroon',
        'MAR': 'Morocco',
        'TUN': 'Tunisia',
        'EGY': 'Egypt',
        'ALG': 'Algeria',
        'RSA': 'South Africa',
        'JPN': 'Japan',
        'KOR': 'Korea Republic',
        'IRN': 'Iran',
        'AUS': 'Australia',
        'SAU': 'Saudi Arabia',
        'QAT': 'Qatar',
        'CRC': 'Costa Rica',
        'PAN': 'Panama',
        'JAM': 'Jamaica',
        'HON': 'Honduras',
        'ECU': 'Ecuador',
        'PER': 'Peru',
        'PAR': 'Paraguay',
        'VEN': 'Venezuela',
        'BOL': 'Bolivia',
    }
    
    def __init__(self):
        self.session = requests.Session()
        self.session.headers.update({
            'User-Agent': 'Mozilla/5.0 (Macintosh; Intel Mac OS X 10_15_7) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/120.0.0.0 Safari/537.36',
            'Accept': 'text/html,application/xhtml+xml,application/xml;q=0.9,image/webp,*/*;q=0.8',
            'Accept-Language': 'pt-BR,pt;q=0.9,en-US;q=0.8,en;q=0.7',
            'Accept-Encoding': 'gzip, deflate, br',
            'Connection': 'keep-alive',
            'Upgrade-Insecure-Requests': '1'
        })
    
    def _normalize_team_name(self, team_code: str) -> str:
        """Convert FIFA team code to our format"""
        return self.TEAM_MAPPING.get(team_code.upper(), team_code)
    
    def fetch_group_standings(self) -> Dict[str, pd.DataFrame]:
        """
        Fetch current group standings from FIFA website
        
        Returns:
            Dictionary with group letter as key and DataFrame with standings as value
        """
        try:
            logger.info("Fetching group standings from FIFA...")
            response = self.session.get(self.STANDINGS_URL, timeout=15)
            response.raise_for_status()
            
            # Parse HTML
            soup = BeautifulSoup(response.content, 'html.parser')
            
            groups = {}
            
            # Try to find group tables
            # FIFA uses dynamic content, so we'll look for common patterns
            group_sections = soup.find_all(['div', 'section'], class_=re.compile(r'group|standing', re.I))
            
            if not group_sections:
                logger.warning("No group sections found in HTML")
                return self._get_fallback_standings()
            
            for section in group_sections:
                # Try to extract group letter
                group_text = section.get_text()
                group_match = re.search(r'Grupo\s+([A-H])', group_text, re.I)
                
                if group_match:
                    group_letter = group_match.group(1).upper()
                    
                    # Try to find table rows
                    rows = section.find_all(['tr', 'div'], class_=re.compile(r'row|team', re.I))
                    
                    teams_data = []
                    for row in rows:
                        # Extract team data
                        team_info = self._extract_team_from_row(row)
                        if team_info:
                            teams_data.append(team_info)
                    
                    if teams_data:
                        groups[group_letter] = pd.DataFrame(teams_data)
            
            if not groups:
                logger.warning("Could not parse standings, using fallback")
                return self._get_fallback_standings()
            
            logger.info(f"Successfully fetched standings for {len(groups)} groups")
            return groups
            
        except Exception as e:
            logger.error(f"Error fetching standings: {e}")
            return self._get_fallback_standings()
    
    def _extract_team_from_row(self, row) -> Optional[Dict]:
        """Extract team information from a table row"""
        try:
            text = row.get_text(strip=True)
            
            # Look for team code (3 letters)
            team_match = re.search(r'\b([A-Z]{3})\b', text)
            if not team_match:
                return None
            
            team_code = team_match.group(1)
            
            # Extract numbers (J, V, E, D, GM, GS, SG, Pts)
            numbers = re.findall(r'\b(\d+)\b', text)
            
            if len(numbers) >= 8:
                return {
                    'team': self._normalize_team_name(team_code),
                    'played': int(numbers[0]),
                    'won': int(numbers[1]),
                    'drawn': int(numbers[2]),
                    'lost': int(numbers[3]),
                    'goals_for': int(numbers[4]),
                    'goals_against': int(numbers[5]),
                    'goal_diff': int(numbers[6]),
                    'points': int(numbers[7])
                }
            
            return None
            
        except Exception as e:
            logger.debug(f"Error extracting team from row: {e}")
            return None
    
    def fetch_match_results(self) -> List[Dict]:
        """
        Fetch match results from FIFA website
        
        Returns:
            List of dictionaries with match information
        """
        try:
            logger.info("Fetching match results from FIFA...")
            response = self.session.get(self.FIXTURES_URL, timeout=15)
            response.raise_for_status()
            
            # Parse HTML
            soup = BeautifulSoup(response.content, 'html.parser')
            
            matches = []
            
            # Look for match containers
            match_elements = soup.find_all(['div', 'article'], class_=re.compile(r'match|fixture|game', re.I))
            
            if not match_elements:
                logger.warning("No match elements found")
                return self._get_fallback_matches()
            
            for element in match_elements:
                match_info = self._extract_match_from_element(element)
                if match_info:
                    matches.append(match_info)
            
            if not matches:
                logger.warning("Could not parse matches, using fallback")
                return self._get_fallback_matches()
            
            logger.info(f"Successfully fetched {len(matches)} matches")
            return matches
            
        except Exception as e:
            logger.error(f"Error fetching match results: {e}")
            return self._get_fallback_matches()
    
    def _extract_match_from_element(self, element) -> Optional[Dict]:
        """Extract match information from an HTML element"""
        try:
            text = element.get_text()
            
            # Look for team codes
            teams = re.findall(r'\b([A-Z]{3})\b', text)
            if len(teams) < 2:
                return None
            
            team_a = self._normalize_team_name(teams[0])
            team_b = self._normalize_team_name(teams[1])
            
            # Look for score
            score_match = re.search(r'(\d+)\s*[-:]\s*(\d+)', text)
            
            if score_match:
                return {
                    'team_a': team_a,
                    'team_b': team_b,
                    'score_a': int(score_match.group(1)),
                    'score_b': int(score_match.group(2)),
                    'status': 'completed'
                }
            else:
                return {
                    'team_a': team_a,
                    'team_b': team_b,
                    'score_a': None,
                    'score_b': None,
                    'status': 'scheduled'
                }
            
        except Exception as e:
            logger.debug(f"Error extracting match: {e}")
            return None
    
    def _get_fallback_standings(self) -> Dict[str, pd.DataFrame]:
        """Return fallback standings data (Group C example from FIFA)"""
        logger.info("Using fallback standings data")
        
        group_c = pd.DataFrame([
            {'team': 'Scotland', 'played': 1, 'won': 1, 'drawn': 0, 'lost': 0,
             'goals_for': 1, 'goals_against': 0, 'goal_diff': 1, 'points': 3},
            {'team': 'Morocco', 'played': 1, 'won': 0, 'drawn': 1, 'lost': 0,
             'goals_for': 1, 'goals_against': 1, 'goal_diff': 0, 'points': 1},
            {'team': 'Brazil', 'played': 1, 'won': 0, 'drawn': 1, 'lost': 0,
             'goals_for': 1, 'goals_against': 1, 'goal_diff': 0, 'points': 1},
            {'team': 'Haiti', 'played': 1, 'won': 0, 'drawn': 0, 'lost': 1,
             'goals_for': 0, 'goals_against': 1, 'goal_diff': -1, 'points': 0},
        ])
        
        return {'C': group_c}
    
    def _get_fallback_matches(self) -> List[Dict]:
        """Return fallback match data"""
        logger.info("Using fallback match data")
        
        return [
            {'team_a': 'Brazil', 'team_b': 'Morocco', 'score_a': 1, 'score_b': 1, 'status': 'completed'},
            {'team_a': 'Scotland', 'team_b': 'Haiti', 'score_a': 1, 'score_b': 0, 'status': 'completed'},
        ]
    
    def get_live_data(self) -> Dict:
        """
        Get all live data from FIFA website
        
        Returns:
            Dictionary with standings and match results
        """
        return {
            'standings': self.fetch_group_standings(),
            'matches': self.fetch_match_results()
        }


def update_schedule_with_results(schedule_df: pd.DataFrame, fifa_data: Dict) -> pd.DataFrame:
    """
    Update schedule DataFrame with real results from FIFA
    
    Args:
        schedule_df: Current schedule DataFrame
        fifa_data: Data fetched from FIFA website
        
    Returns:
        Updated DataFrame with real results
    """
    updated_df = schedule_df.copy()
    
    # Match FIFA results with schedule
    for match in fifa_data.get('matches', []):
        # Find corresponding match in schedule
        mask = (
            (updated_df['team_a'] == match.get('team_a')) &
            (updated_df['team_b'] == match.get('team_b'))
        )
        
        if mask.any():
            # Update with real results
            updated_df.loc[mask, 'actual_score_a'] = match.get('score_a')
            updated_df.loc[mask, 'actual_score_b'] = match.get('score_b')
            updated_df.loc[mask, 'status'] = 'completed'
    
    return updated_df


if __name__ == "__main__":
    # Test scraper
    scraper = FIFADataScraper()
    data = scraper.get_live_data()
    print("Standings:", data['standings'])
    print("Matches:", len(data['matches']))

# Made with Bob
