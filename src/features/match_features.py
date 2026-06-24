"""Shared feature engineering for training and runtime prediction."""

from __future__ import annotations

from collections import defaultdict, deque
from typing import Dict, Tuple

import numpy as np
import pandas as pd

from src.utils.team_names import normalize_team_name


FEATURE_NAMES = [
    "home_goals_scored",
    "home_goals_conceded",
    "home_win_rate",
    "home_matches_played",
    "home_recent_points",
    "home_recent_goal_diff",
    "home_elo",
    "away_goals_scored",
    "away_goals_conceded",
    "away_win_rate",
    "away_matches_played",
    "away_recent_points",
    "away_recent_goal_diff",
    "away_elo",
    "elo_diff",
    "home_advantage",
    "neutral_site",
    "tournament_importance",
]

DEFAULT_TEAM_STATS = {
    "goals_scored": 1.3,
    "goals_conceded": 1.3,
    "wins": 0.33,
    "matches_played": 0.0,
    "recent_points": 1.0,
    "recent_goal_diff": 0.0,
    "elo": 1500.0,
}

BASE_ELO = 1500.0
RECENT_WINDOW = 5
HOME_ELO_ADVANTAGE = 65.0


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
    recent = defaultdict(lambda: deque(maxlen=RECENT_WINDOW))
    elos = defaultdict(lambda: BASE_ELO)

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
        _update_after_match(totals, recent, elos, home, away, home_goals, away_goals)

    return {
        team: _stats_from_totals(values, recent[team], elos[team])
        for team, values in totals.items()
    }


def _stats_from_totals(
    values: Dict[str, float],
    recent_results=None,
    elo: float = BASE_ELO,
) -> Dict[str, float]:
    matches = values["matches"]
    if not matches:
        return DEFAULT_TEAM_STATS.copy()
    recent_results = list(recent_results or [])
    return {
        "goals_scored": values["goals_scored"] / matches,
        "goals_conceded": values["goals_conceded"] / matches,
        "wins": values["wins"] / matches,
        "matches_played": float(matches),
        "recent_points": (
            float(np.mean([result["points"] for result in recent_results]))
            if recent_results
            else DEFAULT_TEAM_STATS["recent_points"]
        ),
        "recent_goal_diff": (
            float(np.mean([result["goal_diff"] for result in recent_results]))
            if recent_results
            else DEFAULT_TEAM_STATS["recent_goal_diff"]
        ),
        "elo": float(elo),
    }


def _feature_row(
    home_stats: Dict[str, float],
    away_stats: Dict[str, float],
    *,
    neutral_site: float = 1.0,
    tournament_importance: float = 1.0,
) -> list[float]:
    home_advantage = 0.0 if neutral_site else 1.0
    return [
        home_stats["goals_scored"],
        home_stats["goals_conceded"],
        home_stats["wins"],
        np.log1p(home_stats["matches_played"]),
        home_stats["recent_points"],
        home_stats["recent_goal_diff"],
        home_stats["elo"] / 1000.0,
        away_stats["goals_scored"],
        away_stats["goals_conceded"],
        away_stats["wins"],
        np.log1p(away_stats["matches_played"]),
        away_stats["recent_points"],
        away_stats["recent_goal_diff"],
        away_stats["elo"] / 1000.0,
        (home_stats["elo"] - away_stats["elo"]) / 400.0,
        home_advantage,
        float(neutral_site),
        tournament_importance,
    ]


def create_match_features(
    team_stats: Dict[str, Dict[str, float]],
    home_team: str,
    away_team: str,
    *,
    neutral_site: bool = True,
    tournament: str = "FIFA World Cup",
) -> np.ndarray:
    """Create model features for an upcoming match."""
    home_stats = team_stats.get(
        normalize_team_name(home_team), DEFAULT_TEAM_STATS
    )
    away_stats = team_stats.get(
        normalize_team_name(away_team), DEFAULT_TEAM_STATS
    )
    return np.asarray(
        _feature_row(
            home_stats,
            away_stats,
            neutral_site=float(neutral_site),
            tournament_importance=_tournament_importance(tournament),
        )
    ).reshape(1, -1)


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
    recent = defaultdict(lambda: deque(maxlen=RECENT_WINDOW))
    elos = defaultdict(lambda: BASE_ELO)
    features = []
    labels = []
    context = []

    for _, match in ordered.iterrows():
        home = normalize_team_name(match["Home Team Name"])
        away = normalize_team_name(match["Away Team Name"])
        home_goals = float(match["Home Team Goals"])
        away_goals = float(match["Away Team Goals"])
        neutral_site = _neutral_site(match)
        tournament_importance = _tournament_importance(match.get("Tournament"))

        home_stats = _stats_from_totals(totals[home], recent[home], elos[home])
        away_stats = _stats_from_totals(totals[away], recent[away], elos[away])
        features.append(
            _feature_row(
                home_stats,
                away_stats,
                neutral_site=neutral_site,
                tournament_importance=tournament_importance,
            )
        )
        labels.append(result_label(home_goals, away_goals))
        context.append(
            {
                "year": int(match["Year"]) if pd.notna(match.get("Year")) else None,
                "home_team": home,
                "away_team": away,
                "home_goals": home_goals,
                "away_goals": away_goals,
                "tournament": match.get("Tournament", "Unknown"),
                "neutral": bool(neutral_site),
            }
        )

        _update_after_match(
            totals,
            recent,
            elos,
            home,
            away,
            home_goals,
            away_goals,
            neutral_site=bool(neutral_site),
            tournament_importance=tournament_importance,
        )

    return np.asarray(features), np.asarray(labels), pd.DataFrame(context)


def _update_after_match(
    totals: Dict[str, Dict[str, float]],
    recent: Dict[str, deque],
    elos: Dict[str, float],
    home: str,
    away: str,
    home_goals: float,
    away_goals: float,
    *,
    neutral_site: bool = True,
    tournament_importance: float = 1.0,
) -> None:
    totals[home]["matches"] += 1
    totals[home]["goals_scored"] += home_goals
    totals[home]["goals_conceded"] += away_goals
    totals[home]["wins"] += int(home_goals > away_goals)

    totals[away]["matches"] += 1
    totals[away]["goals_scored"] += away_goals
    totals[away]["goals_conceded"] += home_goals
    totals[away]["wins"] += int(away_goals > home_goals)

    if home_goals > away_goals:
        home_points, away_points = 3.0, 0.0
    elif away_goals > home_goals:
        home_points, away_points = 0.0, 3.0
    else:
        home_points, away_points = 1.0, 1.0

    recent[home].append({"points": home_points, "goal_diff": home_goals - away_goals})
    recent[away].append({"points": away_points, "goal_diff": away_goals - home_goals})

    _update_elo(
        elos,
        home,
        away,
        home_goals,
        away_goals,
        neutral_site=neutral_site,
        tournament_importance=tournament_importance,
    )


def _update_elo(
    elos: Dict[str, float],
    home: str,
    away: str,
    home_goals: float,
    away_goals: float,
    *,
    neutral_site: bool,
    tournament_importance: float,
) -> None:
    home_rating = elos[home] + (0 if neutral_site else HOME_ELO_ADVANTAGE)
    away_rating = elos[away]
    expected_home = 1 / (1 + 10 ** ((away_rating - home_rating) / 400))

    if home_goals > away_goals:
        actual_home = 1.0
    elif home_goals < away_goals:
        actual_home = 0.0
    else:
        actual_home = 0.5

    goal_diff = abs(home_goals - away_goals)
    goal_multiplier = 1.0 if goal_diff <= 1 else np.log1p(goal_diff)
    k_factor = 18.0 * float(tournament_importance) * goal_multiplier
    delta = k_factor * (actual_home - expected_home)
    elos[home] += delta
    elos[away] -= delta


def _neutral_site(match: pd.Series) -> float:
    if "Neutral" in match and pd.notna(match.get("Neutral")):
        return float(bool(match.get("Neutral")))
    if "neutral" in match and pd.notna(match.get("neutral")):
        return float(bool(match.get("neutral")))
    return 1.0


def _tournament_importance(tournament: object) -> float:
    value = str(tournament or "").lower()
    if "world cup" in value and "qual" not in value:
        return 2.2
    if "qual" in value:
        return 1.7
    if any(term in value for term in ["uefa euro", "copa américa", "copa america", "african cup", "asian cup", "gold cup", "nations league"]):
        return 1.5
    if "friendly" in value:
        return 0.7
    return 1.0
