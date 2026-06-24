"""Utilities for selecting football scorelines from expected goals."""

from __future__ import annotations

from scipy.stats import poisson


def most_likely_poisson_score(
    expected_home: float,
    expected_away: float,
    predicted_outcome: str | None = None,
    max_goals: int = 10,
) -> tuple[int, int]:
    """
    Select the likeliest score under independent Poisson goal assumptions.

    When ``predicted_outcome`` is provided, the search is constrained to a
    scoreline that matches that outcome: ``home``, ``away`` or ``draw``.
    """
    best_score = (0, 0)
    best_probability = -1.0

    for home_goals in range(max_goals + 1):
        for away_goals in range(max_goals + 1):
            if predicted_outcome == "home" and home_goals <= away_goals:
                continue
            if predicted_outcome == "away" and away_goals <= home_goals:
                continue
            if predicted_outcome == "draw" and home_goals != away_goals:
                continue

            probability = poisson.pmf(home_goals, expected_home) * poisson.pmf(
                away_goals,
                expected_away,
            )
            if probability > best_probability:
                best_probability = probability
                best_score = (home_goals, away_goals)

    return best_score


def outcome_key_from_probabilities(
    home_win: float,
    draw: float,
    away_win: float,
) -> str:
    """Return the most likely outcome key for three-way probabilities."""
    return max(
        {
            "home": home_win,
            "draw": draw,
            "away": away_win,
        },
        key={
            "home": home_win,
            "draw": draw,
            "away": away_win,
        }.get,
    )
