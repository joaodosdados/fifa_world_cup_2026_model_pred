from src.data.fifa_scraper_selenium import FIFASeleniumScraper


def test_parse_completed_match_and_ignore_scheduled_match():
    html = """
    <div class="match-row_matchRowContainer__abc">
      <div class="match-row_team__abc">
        <span class="d-none d-md-block">México</span>
      </div>
      <div>
        <span class="match-row_score__abc">2</span>
        <span class="match-row_statusLabel__abc">FIM</span>
        <span class="match-row_score__abc">0</span>
      </div>
      <div class="match-row_team__abc">
        <span class="d-none d-md-block">África do Sul</span>
      </div>
    </div>
    <div class="match-row_matchRowContainer__def">
      <div class="match-row_team__def">
        <span class="d-none d-md-block">EUA</span>
      </div>
      <div><span class="match-row_statusLabel__def">16:00</span></div>
      <div class="match-row_team__def">
        <span class="d-none d-md-block">Austrália</span>
      </div>
    </div>
    """

    assert FIFASeleniumScraper.parse_match_results(html) == [
        {
            "team_a": "México",
            "team_b": "África do Sul",
            "score_a": 2,
            "score_b": 0,
            "status": "completed",
        }
    ]


def test_parse_deduplicates_matches():
    card = """
    <div class="match-row_matchRowContainer__abc">
      <div class="match-row_team__abc"><span class="d-none d-md-block">Brasil</span></div>
      <span class="match-row_score__abc">3</span>
      <span class="match-row_statusLabel__abc">FIM</span>
      <span class="match-row_score__abc">1</span>
      <div class="match-row_team__abc"><span class="d-none d-md-block">Marrocos</span></div>
    </div>
    """

    results = FIFASeleniumScraper.parse_match_results(card + card)
    assert len(results) == 1
