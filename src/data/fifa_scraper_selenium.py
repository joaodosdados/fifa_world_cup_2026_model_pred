"""
Selenium RPA for the FIFA World Cup 2026 fixtures page.

The FIFA page is rendered client-side. Selenium is used only to render and
scroll the page; parsing is performed from the final HTML with BeautifulSoup.
"""

from __future__ import annotations

import logging
import time
from typing import Dict, List

from bs4 import BeautifulSoup

logger = logging.getLogger(__name__)


class FIFAScraperError(RuntimeError):
    """Raised when the FIFA page cannot be rendered or parsed safely."""


class FIFASeleniumScraper:
    """Fetch completed World Cup matches from FIFA's official fixtures page."""

    FIXTURES_URL = (
        "https://www.fifa.com/pt/tournaments/mens/worldcup/"
        "canadamexicousa2026/scores-fixtures?country=BR&wtw-filter=ALL"
    )

    MATCH_ROW_SELECTOR = "[class*='match-row_matchRowContainer']"
    TEAM_SELECTOR = "[class*='match-row_team']"
    SCORE_SELECTOR = "[class*='match-row_score']"
    STATUS_SELECTOR = "[class*='match-row_statusLabel']"

    COMPLETED_LABELS = {"FIM", "FINAL", "FULL TIME", "FT"}

    def __init__(self, timeout: int = 30, headless: bool = True):
        self.timeout = timeout
        self.headless = headless
        self.driver = None

    def _init_driver(self) -> None:
        """Initialize Chrome using Selenium Manager."""
        try:
            from selenium import webdriver
            from selenium.webdriver.chrome.options import Options

            options = Options()
            if self.headless:
                options.add_argument("--headless=new")
            options.add_argument("--no-sandbox")
            options.add_argument("--disable-dev-shm-usage")
            options.add_argument("--disable-gpu")
            options.add_argument("--disable-extensions")
            options.add_argument("--window-size=1920,1080")
            options.add_argument("--lang=pt-BR")
            options.add_argument(
                "--user-agent=Mozilla/5.0 (Macintosh; Intel Mac OS X 10_15_7) "
                "AppleWebKit/537.36 (KHTML, like Gecko) "
                "Chrome/124.0.0.0 Safari/537.36"
            )

            self.driver = webdriver.Chrome(options=options)
            self.driver.set_page_load_timeout(self.timeout)
            logger.info("Selenium WebDriver initialized")
        except Exception as exc:
            raise FIFAScraperError(
                "Chrome/Chromedriver could not be initialized"
            ) from exc

    def _accept_cookies(self) -> None:
        """Close FIFA's cookie banner when it is displayed."""
        from selenium.webdriver.common.by import By

        for label in ("Concordo", "Aceitar", "Accept"):
            buttons = self.driver.find_elements(
                By.XPATH, f"//button[normalize-space()='{label}']"
            )
            if buttons:
                try:
                    buttons[0].click()
                    return
                except Exception:
                    logger.debug("Cookie button found but could not be clicked")

    def _load_all_matches(self) -> None:
        """Scroll until FIFA stops adding match cards to the page."""
        from selenium.webdriver.common.by import By
        from selenium.webdriver.support.ui import WebDriverWait
        from selenium.webdriver.support import expected_conditions as EC

        WebDriverWait(self.driver, self.timeout).until(
            EC.presence_of_element_located((By.CSS_SELECTOR, self.MATCH_ROW_SELECTOR))
        )

        previous_count = -1
        stable_rounds = 0

        for _ in range(20):
            rows = self.driver.find_elements(By.CSS_SELECTOR, self.MATCH_ROW_SELECTOR)
            current_count = len(rows)

            if current_count == previous_count:
                stable_rounds += 1
            else:
                stable_rounds = 0

            if stable_rounds >= 2:
                break

            previous_count = current_count
            self.driver.execute_script(
                "window.scrollTo(0, document.body.scrollHeight);"
            )
            time.sleep(0.5)

        self.driver.execute_script("window.scrollTo(0, 0);")
        logger.info("FIFA page rendered with %d match cards", previous_count)

    @staticmethod
    def _team_name(team_element) -> str:
        """Extract the desktop full team name, falling back to visible text."""
        full_name = team_element.select_one("span.d-none.d-md-block")
        if full_name:
            return full_name.get_text(" ", strip=True)

        spans = team_element.find_all("span")
        return spans[-1].get_text(" ", strip=True) if spans else ""

    @classmethod
    def parse_match_results(cls, html: str) -> List[Dict]:
        """Parse completed matches from an already-rendered FIFA HTML page."""
        soup = BeautifulSoup(html, "html.parser")
        parsed: List[Dict] = []
        seen = set()

        for row in soup.select(cls.MATCH_ROW_SELECTOR):
            status_element = row.select_one(cls.STATUS_SELECTOR)
            status = (
                status_element.get_text(" ", strip=True).upper()
                if status_element
                else ""
            )
            if status not in cls.COMPLETED_LABELS:
                continue

            teams = row.select(cls.TEAM_SELECTOR)
            scores = row.select(cls.SCORE_SELECTOR)
            if len(teams) != 2 or len(scores) != 2:
                logger.debug("Skipped malformed completed FIFA match card")
                continue

            team_a = cls._team_name(teams[0])
            team_b = cls._team_name(teams[1])

            try:
                score_a = int(scores[0].get_text(strip=True))
                score_b = int(scores[1].get_text(strip=True))
            except (TypeError, ValueError):
                logger.debug("Skipped match with invalid score: %s vs %s", team_a, team_b)
                continue

            if not team_a or not team_b:
                continue

            key = (team_a, team_b, score_a, score_b)
            if key in seen:
                continue
            seen.add(key)

            parsed.append(
                {
                    "team_a": team_a,
                    "team_b": team_b,
                    "score_a": score_a,
                    "score_b": score_b,
                    "status": "completed",
                }
            )

        return parsed

    def fetch_match_results(self) -> List[Dict]:
        """Render FIFA's page and return all completed match results."""
        self._init_driver()

        try:
            logger.info("Opening FIFA fixtures page")
            self.driver.get(self.FIXTURES_URL)
            self._accept_cookies()
            self._load_all_matches()

            matches = self.parse_match_results(self.driver.page_source)
            if not matches:
                raise FIFAScraperError(
                    "FIFA page loaded, but no completed matches were parsed"
                )

            logger.info("Fetched %d completed FIFA matches", len(matches))
            return matches
        except FIFAScraperError:
            raise
        except Exception as exc:
            raise FIFAScraperError(
                f"Could not fetch FIFA results: {exc}"
            ) from exc
        finally:
            if self.driver:
                self.driver.quit()
                self.driver = None


if __name__ == "__main__":
    results = FIFASeleniumScraper().fetch_match_results()
    print(f"Fetched {len(results)} completed matches")
    for match in results:
        print(
            f"{match['team_a']} {match['score_a']} - "
            f"{match['score_b']} {match['team_b']}"
        )
