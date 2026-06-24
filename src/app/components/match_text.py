"""Display text helpers for fixtures."""

from __future__ import annotations

from typing import Any

import pandas as pd


SHORT_VENUE_NAMES = {
    "Área da baía de São Francisco": "SF Bay Area",
    "Baía de São Francisco": "SF Bay Area",
    "Estádio da Baía de São Francisco": "SF Bay Area",
    "Nova York/Nova Jersey": "NY/NJ",
}


def format_fixture_datetime(match: Any) -> str:
    """Format fixture date/time from a Series-like object."""
    match_date = str(match.get("date", "")).strip()
    match_time = match.get("time", "")
    if pd.isna(match_time) or str(match_time).strip() == "":
        return match_date
    return f"{match_date} {str(match_time).strip()}"


def compact_venue_name(venue: Any, max_length: int = 26) -> str:
    """Return a compact venue label for dense match cards."""
    if pd.isna(venue):
        return ""

    venue_text = " ".join(str(venue).strip().split())
    if not venue_text:
        return ""

    for source, replacement in SHORT_VENUE_NAMES.items():
        if source.casefold() in venue_text.casefold():
            return replacement

    if "(" in venue_text and ")" in venue_text:
        city = venue_text.rsplit("(", 1)[-1].rstrip(")").strip()
        if city:
            return SHORT_VENUE_NAMES.get(city, city)

    for prefix in ("Estádio de ", "Estádio da ", "Estádio do "):
        if venue_text.startswith(prefix):
            venue_text = venue_text[len(prefix):]
            break

    if len(venue_text) <= max_length:
        return venue_text
    return venue_text[: max_length - 1].rstrip() + "…"


def format_fixture_caption(match: Any) -> str:
    """Format compact date/time/venue caption for match cards."""
    fixture_datetime = format_fixture_datetime(match)
    venue = compact_venue_name(match.get("venue", ""))
    if venue:
        return f"{fixture_datetime} • {venue}"
    return fixture_datetime
