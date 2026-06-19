"""Shared feature engineering for training and runtime prediction."""

from __future__ import annotations

from collections import defaultdict
from typing import Dict, Tuple

import numpy as np
import pandas as pd

from src.utils.team_names import normalize_team_name


FEATURE_NAMES = [
    "home_goals_scored",
    "home_goals_conceded",
    "home_win_rate",
    "away_goals_scored",
    "away_goals_conceded",
    "away_win_rate",
    "home_advantage",
]

DEFAULT_TEAM_STATS = {
    "goals_scored": 1.3,
    "goals_conceded": 1.3,
    "wins": 0.33,
}


def result_label(home_goals: float, away_goals: float) -> int:
    """Encode away win, draw and home win as 0, 1 and 2."""
    if home_goals > away_goals:
        return 2
    if home_goals < away_goals:
        return 0
    return 1


def calculate_team_stats(df: pd.DataFrame) -> Dict[str, Dict[str, float]]:
    """Calculate aggregate team statistics for runtime inference."""
    totals = defaultdict(
        lambda: {"matches": 0, "goals_scored": 0.0, "goals_conceded": 0.0, "wins": 0}
    )

    home_index = df.columns.get_loc("Home Team Name")
    away_index = df.columns.get_loc("Away Team Name")
    home_goals_index = df.columns.get_loc("Home Team Goals")
    away_goals_index = df.columns.get_loc("Away Team Goals")

    for row in df.itertuples(index=False, name=None):
        # itertuples field names are unstable with spaces; use positional values.
        home = normalize_team_name(row[home_index])
        away = normalize_team_name(row[away_index])
        home_goals = float(row[home_goals_index])
        away_goals = float(row[away_goals_index])
        _update_totals(totals, home, away, home_goals, away_goals)

    return {
        team: _stats_from_totals(values)
        for team, values in totals.items()
    }


def _stats_from_totals(values: Dict[str, float]) -> Dict[str, float]:
    matches = values["matches"]
    if not matches:
        return DEFAULT_TEAM_STATS.copy()
    return {
        "goals_scored": values["goals_scored"] / matches,
        "goals_conceded": values["goals_conceded"] / matches,
        "wins": values["wins"] / matches,
    }


def _feature_row(
    home_stats: Dict[str, float],
    away_stats: Dict[str, float],
) -> list[float]:
    return [
        home_stats["goals_scored"],
        home_stats["goals_conceded"],
        home_stats["wins"],
        away_stats["goals_scored"],
        away_stats["goals_conceded"],
        away_stats["wins"],
        1.0,
    ]


def create_match_features(
    team_stats: Dict[str, Dict[str, float]],
    home_team: str,
    away_team: str,
) -> np.ndarray:
    """Create the seven model features for an upcoming match."""
    home_stats = team_stats.get(
        normalize_team_name(home_team), DEFAULT_TEAM_STATS
    )
    away_stats = team_stats.get(
        normalize_team_name(away_team), DEFAULT_TEAM_STATS
    )
    return np.asarray(_feature_row(home_stats, away_stats)).reshape(1, -1)


def build_temporal_training_data(
    matches: pd.DataFrame,
) -> Tuple[np.ndarray, np.ndarray, pd.DataFrame]:
    """
    Build leakage-free pre-match features in chronological order.

    Each row uses only matches observed before it. The returned context frame
    preserves the year and teams for temporal splitting and diagnostics.
    """
    ordered = matches.copy()
    sort_columns = [column for column in ("Year",) if column in ordered]
    if "Datetime" in ordered:
        ordered["_parsed_datetime"] = pd.to_datetime(
            ordered["Datetime"].astype(str).str.replace(" - ", " ", regex=False),
            dayfirst=True,
            errors="coerce",
        )
        sort_columns.append("_parsed_datetime")
    if "MatchID" in ordered:
        sort_columns.append("MatchID")
    ordered = ordered.sort_values(
        sort_columns, kind="stable", na_position="last"
    ).reset_index(drop=True)
    totals = defaultdict(
        lambda: {"matches": 0, "goals_scored": 0.0, "goals_conceded": 0.0, "wins": 0}
    )
    features = []
    labels = []
    context = []

    for _, match in ordered.iterrows():
        home = normalize_team_name(match["Home Team Name"])
        away = normalize_team_name(match["Away Team Name"])
        home_goals = float(match["Home Team Goals"])
        away_goals = float(match["Away Team Goals"])

        home_stats = _stats_from_totals(totals[home])
        away_stats = _stats_from_totals(totals[away])
        features.append(_feature_row(home_stats, away_stats))
        labels.append(result_label(home_goals, away_goals))
        context.append(
            {
                "year": int(match["Year"]) if pd.notna(match.get("Year")) else None,
                "home_team": home,
                "away_team": away,
            }
        )

        _update_totals(totals, home, away, home_goals, away_goals)

    return np.asarray(features), np.asarray(labels), pd.DataFrame(context)


def _update_totals(
    totals: Dict[str, Dict[str, float]],
    home: str,
    away: str,
    home_goals: float,
    away_goals: float,
) -> None:
    totals[home]["matches"] += 1
    totals[home]["goals_scored"] += home_goals
    totals[home]["goals_conceded"] += away_goals
    totals[home]["wins"] += int(home_goals > away_goals)

    totals[away]["matches"] += 1
    totals[away]["goals_scored"] += away_goals
    totals[away]["goals_conceded"] += home_goals
    totals[away]["wins"] += int(away_goals > home_goals)
