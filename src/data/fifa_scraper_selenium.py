"""
Selenium RPA for the FIFA World Cup 2026 fixtures page.

The FIFA page is rendered client-side. Selenium is used only to render and
scroll the page; parsing is performed from the final HTML with BeautifulSoup.
"""

from __future__ import annotations

import logging
import re
import time
from datetime import date
from typing import Dict, List

from bs4 import BeautifulSoup

logger = logging.getLogger(__name__)


class FIFAScraperError(RuntimeError):
    """Raised when the FIFA page cannot be rendered or parsed safely."""


class FIFASeleniumScraper:
    """Fetch World Cup fixtures from FIFA's official fixtures page."""

    FIXTURES_URL = (
        "https://www.fifa.com/pt/tournaments/mens/worldcup/"
        "canadamexicousa2026/scores-fixtures?country=BR&wtw-filter=ALL"
    )

    MATCH_ROW_SELECTOR = "[class*='match-row_matchRowContainer']"
    TEAM_SELECTOR = "[class*='match-row_team']"
    SCORE_SELECTOR = "[class*='match-row_score']"
    STATUS_SELECTOR = "[class*='match-row_statusLabel']"
    TIME_SELECTOR = "[class*='match-row_matchTime']"
    DATE_TITLE_SELECTOR = "[class*='matches-container_title']"
    BOTTOM_LABEL_SELECTOR = "[class*='match-row_bottomLabel']"
    STADIUM_CITY_SELECTOR = "[class*='match-row_stadiumCityLabels']"

    COMPLETED_LABELS = {"FIM", "FINAL", "FULL TIME", "FT"}
    MONTHS_PT = {
        "janeiro": 1,
        "fevereiro": 2,
        "março": 3,
        "abril": 4,
        "maio": 5,
        "junho": 6,
        "julho": 7,
        "agosto": 8,
        "setembro": 9,
        "outubro": 10,
        "novembro": 11,
        "dezembro": 12,
    }

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
    def _parse_date(cls, label: str) -> str:
        """Parse FIFA's Portuguese date label into YYYY-MM-DD."""
        normalized = " ".join(label.casefold().split())
        match = re.search(
            r"(\d{1,2})\s+([a-zç]+)\s+(\d{4})",
            normalized,
        )
        if not match:
            raise FIFAScraperError(f"Could not parse FIFA date label: {label}")

        day = int(match.group(1))
        month_name = match.group(2)
        year = int(match.group(3))
        month = cls.MONTHS_PT.get(month_name)
        if not month:
            raise FIFAScraperError(f"Unknown FIFA month name: {month_name}")

        return date(year, month, day).isoformat()

    @classmethod
    def _match_status(cls, row) -> tuple[str, str]:
        """Return normalized status and visible time/status label."""
        status_element = row.select_one(cls.STATUS_SELECTOR)
        if status_element:
            raw_status = status_element.get_text(" ", strip=True).upper()
            if raw_status in cls.COMPLETED_LABELS:
                return "completed", raw_status
            return "scheduled", ""

        time_element = row.select_one(cls.TIME_SELECTOR)
        kickoff_time = time_element.get_text(" ", strip=True) if time_element else ""
        if not re.fullmatch(r"\d{1,2}:\d{2}", kickoff_time):
            kickoff_time = ""
        return "scheduled", kickoff_time

    @classmethod
    def _match_metadata(cls, row) -> tuple[str, str, str]:
        """Extract stage, group and venue labels from a FIFA match row."""
        labels = [
            label.get_text(" ", strip=True)
            for label in row.select(cls.BOTTOM_LABEL_SELECTOR)
            if label.get_text(" ", strip=True)
        ]

        stage = labels[-2] if len(labels) >= 2 else ""
        group_label = labels[-1] if labels else ""
        group_match = re.search(r"Grupo\s+([A-Z])", group_label, re.IGNORECASE)
        group = group_match.group(1).upper() if group_match else ""

        venue_element = row.select_one(cls.STADIUM_CITY_SELECTOR)
        venue = (
            " ".join(venue_element.get_text(" ", strip=True).split())
            if venue_element
            else ""
        )
        return stage, group, venue

    @classmethod
    def parse_fixtures(cls, html: str) -> List[Dict]:
        """Parse all fixtures from an already-rendered FIFA HTML page."""
        soup = BeautifulSoup(html, "html.parser")
        parsed: List[Dict] = []
        seen = set()

        for title in soup.select(cls.DATE_TITLE_SELECTOR):
            match_date = cls._parse_date(title.get_text(" ", strip=True))
            date_block = title.find_parent("div")
            for _ in range(2):
                if date_block is not None:
                    date_block = date_block.find_parent("div")
            if date_block is None:
                continue

            for row in date_block.select(cls.MATCH_ROW_SELECTOR):
                teams = row.select(cls.TEAM_SELECTOR)
                if len(teams) != 2:
                    logger.debug("Skipped malformed FIFA match card")
                    continue

                team_a = cls._team_name(teams[0])
                team_b = cls._team_name(teams[1])
                if not team_a or not team_b:
                    continue

                status, time_or_status = cls._match_status(row)
                stage, group, venue = cls._match_metadata(row)

                scores = row.select(cls.SCORE_SELECTOR)
                score_a = None
                score_b = None
                if status == "completed" and len(scores) == 2:
                    try:
                        score_a = int(scores[0].get_text(strip=True))
                        score_b = int(scores[1].get_text(strip=True))
                    except (TypeError, ValueError):
                        logger.debug(
                            "Skipped invalid score for completed FIFA match: %s vs %s",
                            team_a,
                            team_b,
                        )
                        continue

                key = (match_date, team_a, team_b)
                if key in seen:
                    continue
                seen.add(key)

                parsed.append(
                    {
                        "team_a": team_a,
                        "team_b": team_b,
                        "date": match_date,
                        "time": "" if status == "completed" else time_or_status,
                        "venue": venue,
                        "stage": "Group Stage"
                        if stage.casefold() == "primeira fase"
                        else stage,
                        "group": group,
                        "score_a": score_a,
                        "score_b": score_b,
                        "status": status,
                    }
                )

        return parsed

    @classmethod
    def parse_match_results(cls, html: str) -> List[Dict]:
        """Parse completed matches from an already-rendered FIFA HTML page."""
        fixtures = [
            match
            for match in cls.parse_fixtures(html)
            if match.get("status") == "completed"
        ]
        if fixtures:
            return fixtures

        soup = BeautifulSoup(html, "html.parser")
        parsed: List[Dict] = []
        seen = set()

        for row in soup.select(cls.MATCH_ROW_SELECTOR):
            status, _ = cls._match_status(row)
            if status != "completed":
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

    def fetch_fixtures(self) -> List[Dict]:
        """Render FIFA's page and return all parsed fixtures."""
        self._init_driver()

        try:
            logger.info("Opening FIFA fixtures page")
            self.driver.get(self.FIXTURES_URL)
            self._accept_cookies()
            self._load_all_matches()

            matches = self.parse_fixtures(self.driver.page_source)
            if not matches:
                raise FIFAScraperError(
                    "FIFA page loaded, but no fixtures were parsed"
                )

            logger.info("Fetched %d FIFA fixtures", len(matches))
            return matches
        except FIFAScraperError:
            raise
        except Exception as exc:
            raise FIFAScraperError(
                f"Could not fetch FIFA fixtures: {exc}"
            ) from exc
        finally:
            if self.driver:
                self.driver.quit()
                self.driver = None

    def fetch_match_results(self) -> List[Dict]:
        """Render FIFA's page and return all completed match results."""
        matches = [
            match
            for match in self.fetch_fixtures()
            if match.get("status") == "completed"
        ]
        if not matches:
            raise FIFAScraperError(
                "FIFA page loaded, but no completed matches were parsed"
            )
        logger.info("Fetched %d completed FIFA matches", len(matches))
        return matches


if __name__ == "__main__":
    results = FIFASeleniumScraper().fetch_match_results()
    print(f"Fetched {len(results)} completed matches")
    for match in results:
        print(
            f"{match['team_a']} {match['score_a']} - "
            f"{match['score_b']} {match['team_b']}"
        )
