from src.app.components.prediction_text import (
    format_score_summary,
    score_explanation,
    score_label,
)


def test_score_summary_explains_adjusted_score():
    prediction = {
        "most_likely_score": (1, 2),
        "modal_score": (1, 1),
        "score_adjusted_to_outcome": True,
        "expected_goals_home": 1.5,
        "expected_goals_away": 1.8,
    }

    assert score_label(prediction) == "Placar de referência"
    assert (
        format_score_summary(prediction)
        == "Placar de referência: 1–2 · xG: 1.5–1.8 · modal por xG: 1–1"
    )
    assert score_explanation(prediction) is not None


def test_score_summary_keeps_unadjusted_score_simple():
    prediction = {
        "most_likely_score": (1, 1),
        "score_adjusted_to_outcome": False,
        "expected_goals_home": 1.5,
        "expected_goals_away": 1.8,
    }

    assert score_label(prediction) == "Placar provável"
    assert format_score_summary(prediction) == "Placar provável: 1–1 · xG: 1.5–1.8"
    assert score_explanation(prediction) is None
