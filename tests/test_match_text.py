from src.components.match_text import compact_venue_name, format_fixture_caption


def test_compact_venue_shortens_long_san_francisco_label():
    venue = "Estádio da Baía de São Francisco (Área da baía de São Francisco)"

    assert compact_venue_name(venue) == "SF Bay Area"


def test_fixture_caption_keeps_card_text_short():
    match = {
        "date": "2026-06-23",
        "time": "15:00",
        "venue": "Estádio da Baía de São Francisco (Área da baía de São Francisco)",
    }

    assert format_fixture_caption(match) == "2026-06-23 15:00 • SF Bay Area"


def test_compact_venue_prefers_city_inside_parentheses():
    assert compact_venue_name("Estádio de Boston (Boston)") == "Boston"
