"""Text helpers for making match predictions easier to interpret."""

from __future__ import annotations

from typing import Any, Dict


def score_label(prediction: Dict[str, Any]) -> str:
    """Return the UI label for the displayed score."""
    if prediction.get("score_adjusted_to_outcome"):
        return "Placar de referência"
    return "Placar provável"


def format_expected_goals(prediction: Dict[str, Any]) -> str:
    """Format expected goals with one decimal place."""
    return (
        f"{float(prediction['expected_goals_home']):.1f}–"
        f"{float(prediction['expected_goals_away']):.1f}"
    )


def format_score_summary(prediction: Dict[str, Any]) -> str:
    """
    Build a compact score/xG summary.

    ``most_likely_score`` is the score shown to the user. For ML models it may
    be constrained to match the most probable outcome class. ``modal_score`` is
    the unconstrained Poisson modal score derived only from expected goals.
    """
    displayed_home, displayed_away = prediction["most_likely_score"]
    modal_home, modal_away = prediction.get(
        "modal_score",
        (displayed_home, displayed_away),
    )
    summary = (
        f"{score_label(prediction)}: {displayed_home}–{displayed_away} · "
        f"xG: {format_expected_goals(prediction)}"
    )
    if prediction.get("score_adjusted_to_outcome"):
        summary += f" · modal por xG: {modal_home}–{modal_away}"
    return summary


def score_explanation(prediction: Dict[str, Any]) -> str | None:
    """Return an explanatory note when the displayed score was adjusted."""
    if not prediction.get("score_adjusted_to_outcome"):
        return None

    displayed_home, displayed_away = prediction["most_likely_score"]
    modal_home, modal_away = prediction.get(
        "modal_score",
        (displayed_home, displayed_away),
    )
    return (
        "O placar de referência foi alinhado ao resultado mais provável do "
        f"classificador ({displayed_home}–{displayed_away}). Pelos gols "
        f"esperados isoladamente, o placar modal seria {modal_home}–{modal_away}."
    )
