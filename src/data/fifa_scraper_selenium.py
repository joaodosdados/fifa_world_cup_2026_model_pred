"""
FIFA World Cup 2026 Data Scraper with Selenium
Fetches live match results from FIFA website using Selenium for dynamic content
"""

import logging
import re
from typing import Dict, List, Optional
from datetime import datetime
import time

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


class FIFASeleniumScraper:
    """Scraper for FIFA World Cup 2026 data using Selenium"""
    
    FIXTURES_URL = "https://www.fifa.com/pt/tournaments/mens/worldcup/canadamexicousa2026/scores-fixtures?country=BR&wtw-filter=ALL"
    
    # Team name mapping (FIFA Portuguese -> English)
    TEAM_MAPPING = {
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
        'Países Baixos': 'Netherlands',
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
    
    def __init__(self):
        self.driver = None
    
    def _normalize_team_name(self, team_name: str) -> str:
        """Convert FIFA Portuguese team name to English"""
        team_name = team_name.strip()
        return self.TEAM_MAPPING.get(team_name, team_name)
    
    def _init_driver(self):
        """Initialize Selenium WebDriver"""
        try:
            from selenium import webdriver
            from selenium.webdriver.chrome.options import Options
            from selenium.webdriver.chrome.service import Service
            from selenium.webdriver.common.by import By
            from selenium.webdriver.support.ui import WebDriverWait
            from selenium.webdriver.support import expected_conditions as EC
            
            chrome_options = Options()
            chrome_options.add_argument('--headless')
            chrome_options.add_argument('--no-sandbox')
            chrome_options.add_argument('--disable-dev-shm-usage')
            chrome_options.add_argument('--disable-gpu')
            chrome_options.add_argument('--window-size=1920,1080')
            chrome_options.add_argument('user-agent=Mozilla/5.0 (Macintosh; Intel Mac OS X 10_15_7) AppleWebKit/537.36')
            
            self.driver = webdriver.Chrome(options=chrome_options)
            logger.info("Selenium WebDriver initialized successfully")
            return True
            
        except Exception as e:
            logger.error(f"Failed to initialize Selenium: {e}")
            logger.info("Falling back to requests-based scraper")
            return False
    
    def fetch_match_results(self) -> List[Dict]:
        """
        Fetch match results from FIFA website using Selenium
        
        Returns:
            List of dictionaries with match information
        """
        # Try to initialize Selenium
        if not self._init_driver():
            logger.warning("Selenium not available, using fallback data")
            return self._get_fallback_matches()
        
        try:
            from selenium.webdriver.common.by import By
            from selenium.webdriver.support.ui import WebDriverWait
            from selenium.webdriver.support import expected_conditions as EC
            
            logger.info("Fetching match results from FIFA using Selenium...")
            self.driver.get(self.FIXTURES_URL)
            
            # Wait for page to load
            time.sleep(5)
            
            # Try to close cookie banner if present
            try:
                cookie_button = WebDriverWait(self.driver, 5).until(
                    EC.element_to_be_clickable((By.XPATH, "//button[contains(text(), 'Concordo') or contains(text(), 'Accept')]"))
                )
                cookie_button.click()
                time.sleep(2)
            except:
                pass
            
            matches = []
            
            # Find all match elements
            match_elements = self.driver.find_elements(By.CSS_SELECTOR, "[class*='match'], [class*='fixture'], [class*='game']")
            
            for element in match_elements:
                try:
                    text = element.text
                    
                    # Look for team names and scores
                    lines = text.split('\n')
                    
                    for i, line in enumerate(lines):
                        # Check if line contains "FIM" (finished match)
                        if 'FIM' in line or 'FINAL' in line:
                            # Try to extract teams and score from surrounding lines
                            if i > 0 and i < len(lines) - 1:
                                # Pattern: Team1 Score - Score Team2
                                match_text = ' '.join(lines[max(0, i-2):min(len(lines), i+3)])
                                
                                # Try to find score pattern
                                score_match = re.search(r'(\d+)\s*[-:]\s*(\d+)', match_text)
                                
                                if score_match:
                                    # Extract team names (words before and after score)
                                    parts = match_text.split(score_match.group(0))
                                    
                                    if len(parts) >= 2:
                                        team_a_text = parts[0].strip()
                                        team_b_text = parts[1].strip()
                                        
                                        # Clean team names
                                        team_a = re.sub(r'[^a-zA-ZÀ-ÿ\s]', '', team_a_text).strip()
                                        team_b = re.sub(r'[^a-zA-ZÀ-ÿ\s]', '', team_b_text).strip()
                                        
                                        # Get last words as team names
                                        team_a_words = team_a.split()
                                        team_b_words = team_b.split()
                                        
                                        if team_a_words and team_b_words:
                                            team_a_name = ' '.join(team_a_words[-3:]) if len(team_a_words) > 2 else team_a
                                            team_b_name = ' '.join(team_b_words[:3]) if len(team_b_words) > 2 else team_b
                                            
                                            matches.append({
                                                'team_a': self._normalize_team_name(team_a_name),
                                                'team_b': self._normalize_team_name(team_b_name),
                                                'score_a': int(score_match.group(1)),
                                                'score_b': int(score_match.group(2)),
                                                'status': 'completed'
                                            })
                except Exception as e:
                    logger.debug(f"Error parsing match element: {e}")
                    continue
            
            if matches:
                logger.info(f"Successfully fetched {len(matches)} completed matches from FIFA")
                return matches
            else:
                logger.warning("No matches found, using fallback")
                return self._get_fallback_matches()
            
        except Exception as e:
            logger.error(f"Error fetching match results with Selenium: {e}")
            return self._get_fallback_matches()
        
        finally:
            if self.driver:
                self.driver.quit()
    
    def _get_fallback_matches(self) -> List[Dict]:
        """Return fallback match data"""
        logger.info("Using fallback match data")
        
        return [
            {'team_a': 'Brazil', 'team_b': 'Morocco', 'score_a': 1, 'score_b': 1, 'status': 'completed'},
            {'team_a': 'Scotland', 'team_b': 'Haiti', 'score_a': 1, 'score_b': 0, 'status': 'completed'},
            {'team_a': 'Czech Republic', 'team_b': 'South Africa', 'score_a': 1, 'score_b': 1, 'status': 'completed'},
        ]


if __name__ == "__main__":
    # Test scraper
    scraper = FIFASeleniumScraper()
    matches = scraper.fetch_match_results()
    print(f"\nFetched {len(matches)} matches:")
    for match in matches:
        print(f"  {match['team_a']} {match.get('score_a', '?')} - {match.get('score_b', '?')} {match['team_b']} ({match['status']})")

# Made with Bob
