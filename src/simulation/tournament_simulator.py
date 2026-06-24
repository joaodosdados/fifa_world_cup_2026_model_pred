"""Monte Carlo simulator for the FIFA World Cup tournament."""

from __future__ import annotations

from collections import Counter, defaultdict
from dataclasses import dataclass
from typing import Any, Dict, Iterable

import numpy as np
import pandas as pd


KNOCKOUT_STAGES = [
    ("round_of_32", "Round of 32"),
    ("round_of_16", "Oitavas"),
    ("quarter_finals", "Quartas"),
    ("semi_finals", "Semifinal"),
    ("final", "Final"),
]


@dataclass(frozen=True)
class SimulatedMatch:
    """Compact match result used by the simulator."""

    home_team: str
    away_team: str
    home_goals: int
    away_goals: int

    @property
    def winner(self) -> str:
        if self.home_goals > self.away_goals:
            return self.home_team
        if self.away_goals > self.home_goals:
            return self.away_team
        return "Draw"


def _safe_float(value: Any, default: float = 0.0) -> float:
    try:
        if pd.isna(value):
            return default
        return float(value)
    except (TypeError, ValueError):
        return default


def _normalize_probabilities(values: Iterable[float]) -> np.ndarray:
    probabilities = np.asarray(list(values), dtype=float)
    probabilities = np.clip(probabilities, 0.0, None)
    total = probabilities.sum()
    if total <= 0:
        return np.ones(len(probabilities)) / len(probabilities)
    return probabilities / total


def _empty_standing(team: str) -> Dict[str, Any]:
    return {
        "team": team,
        "points": 0,
        "played": 0,
        "wins": 0,
        "draws": 0,
        "losses": 0,
        "goals_for": 0,
        "goals_against": 0,
        "goal_difference": 0,
    }


class TournamentSimulator:
    """Run Monte Carlo simulations using the active dashboard model."""

    def __init__(
        self,
        predictor: Any,
        schedule: pd.DataFrame,
        predictions: pd.DataFrame,
        random_seed: int = 42,
    ):
        self.predictor = predictor
        self.schedule = schedule.copy()
        self.predictions = predictions
        self.rng = np.random.default_rng(random_seed)
        self._knockout_probability_cache: Dict[tuple[str, str], Dict[str, float]] = {}
        self.group_matches = self._prepare_group_matches()

    def simulate(self, n_simulations: int = 10_000) -> Dict[str, Any]:
        """Run the tournament simulation and return summary artifacts."""
        n_simulations = int(n_simulations)
        if n_simulations <= 0:
            raise ValueError("n_simulations must be positive")

        teams = sorted(
            set(self.schedule["home_team"].dropna())
            | set(self.schedule["away_team"].dropna())
        )
        stage_counts = {
            team: Counter()
            for team in teams
        }
        points_sum = Counter()
        path_counts = {
            team: {stage_key: Counter() for stage_key, _ in KNOCKOUT_STAGES}
            for team in teams
        }
        matchup_counts = Counter()
        third_place_points = Counter()

        for _ in range(n_simulations):
            group_rankings = self._simulate_group_stage()
            qualifiers = self._select_group_qualifiers(group_rankings)

            for group, ranking in group_rankings.items():
                if len(ranking) >= 3:
                    third_place_points[group] += ranking[2]["points"]
                for row in ranking:
                    points_sum[row["team"]] += row["points"]

            for team in (row["team"] for row in qualifiers):
                stage_counts[team]["round_of_32"] += 1

            current_round = self._seed_round_of_32(qualifiers)
            for stage_key, _stage_label in KNOCKOUT_STAGES:
                winners = []
                for team_a, team_b in current_round:
                    path_counts[team_a][stage_key][team_b] += 1
                    path_counts[team_b][stage_key][team_a] += 1
                    matchup_counts[(stage_key, *sorted((team_a, team_b)))] += 1

                    winner = self._simulate_knockout_winner(team_a, team_b)
                    winners.append(winner)

                if stage_key == "final":
                    stage_counts[winners[0]]["title"] += 1
                    break

                next_stage_key = self._next_stage_key(stage_key)
                for winner in winners:
                    stage_counts[winner][next_stage_key] += 1
                current_round = self._pair_adjacent(winners)

        probabilities = self._build_probability_table(
            teams,
            stage_counts,
            points_sum,
            n_simulations,
        )
        matchups = self._build_matchup_table(matchup_counts, n_simulations)
        group_difficulty = self._build_group_difficulty_table(
            third_place_points,
            n_simulations,
        )

        return {
            "probabilities": probabilities,
            "matchups": matchups,
            "group_difficulty": group_difficulty,
            "paths": path_counts,
            "n_simulations": n_simulations,
            "bracket_method": (
                "Seed 1–32 por campanha simulada na fase de grupos; "
                "empates de ranking usam sorteio."
            ),
        }

    def _simulate_group_stage(self) -> Dict[str, list[Dict[str, Any]]]:
        rankings = {}

        for group, matches in self.group_matches.items():
            teams = sorted(
                {match["home_team"] for match in matches}
                | {match["away_team"] for match in matches}
            )
            standings = {team: _empty_standing(team) for team in teams}
            results = []

            for match in matches:
                result = self._simulate_group_match(match)
                self._apply_match_to_standings(standings, result)
                results.append(result)

            table = self._rank_group(standings, results)
            rankings[group] = table

        return rankings

    def _prepare_group_matches(self) -> Dict[str, list[Dict[str, Any]]]:
        """Precompute group-stage match inputs for fast simulation loops."""
        prepared: Dict[str, list[Dict[str, Any]]] = defaultdict(list)
        group_matches = self.schedule[
            self.schedule["stage"].astype(str).eq("Group Stage")
        ].copy()

        for _, match in group_matches.iterrows():
            home_team = str(match["home_team"])
            away_team = str(match["away_team"])
            status = str(match.get("status", "")).casefold()
            home_score = match.get("home_score")
            away_score = match.get("away_score")

            record = {
                "group": str(match["group"]),
                "home_team": home_team,
                "away_team": away_team,
                "completed": (
                    status == "completed"
                    and pd.notna(home_score)
                    and pd.notna(away_score)
                ),
                "home_score": None,
                "away_score": None,
            }

            if record["completed"]:
                record["home_score"] = int(float(home_score))
                record["away_score"] = int(float(away_score))
            else:
                prediction = self._prediction_for_match(match)
                record.update(prediction)

            prepared[record["group"]].append(record)

        return dict(prepared)

    def _simulate_group_match(self, match: Dict[str, Any]) -> SimulatedMatch:
        home_team = str(match["home_team"])
        away_team = str(match["away_team"])

        if match["completed"]:
            return SimulatedMatch(
                home_team,
                away_team,
                int(match["home_score"]),
                int(match["away_score"]),
            )

        outcome = self.rng.choice(
            ["home", "draw", "away"],
            p=_normalize_probabilities(
                [
                    match["home_win"],
                    match["draw"],
                    match["away_win"],
                ]
            ),
        )
        home_goals, away_goals = self._sample_scoreline(
            match["expected_goals_home"],
            match["expected_goals_away"],
            outcome,
        )
        return SimulatedMatch(home_team, away_team, home_goals, away_goals)

    def _prediction_for_match(self, match: pd.Series) -> Dict[str, float]:
        match_id = match.get("match_id")
        if (
            self.predictions is not None
            and not self.predictions.empty
            and match_id in self.predictions.index
        ):
            row = self.predictions.loc[match_id]
            if isinstance(row, pd.DataFrame):
                row = row.iloc[0]
            return {
                "home_win": _safe_float(row.get("home_win"), 1 / 3),
                "draw": _safe_float(row.get("draw"), 1 / 3),
                "away_win": _safe_float(row.get("away_win"), 1 / 3),
                "expected_goals_home": max(
                    0.05,
                    _safe_float(row.get("expected_goals_home"), 1.2),
                ),
                "expected_goals_away": max(
                    0.05,
                    _safe_float(row.get("expected_goals_away"), 1.2),
                ),
            }

        prediction = self.predictor.predict_match(
            match["home_team"],
            match["away_team"],
            is_home_a=True,
        )
        return {
            "home_win": _safe_float(prediction.get("home_win"), 1 / 3),
            "draw": _safe_float(prediction.get("draw"), 1 / 3),
            "away_win": _safe_float(prediction.get("away_win"), 1 / 3),
            "expected_goals_home": max(
                0.05,
                _safe_float(prediction.get("expected_goals_home"), 1.2),
            ),
            "expected_goals_away": max(
                0.05,
                _safe_float(prediction.get("expected_goals_away"), 1.2),
            ),
        }

    def _sample_scoreline(
        self,
        expected_home: float,
        expected_away: float,
        outcome: str,
    ) -> tuple[int, int]:
        if outcome == "home":
            away_goals = int(self.rng.poisson(expected_away))
            margin = max(1, int(self.rng.poisson(max(0.35, expected_home - expected_away + 0.5))))
            return away_goals + margin, away_goals
        if outcome == "away":
            home_goals = int(self.rng.poisson(expected_home))
            margin = max(1, int(self.rng.poisson(max(0.35, expected_away - expected_home + 0.5))))
            return home_goals, home_goals + margin
        goals = int(self.rng.poisson(max(0.05, (expected_home + expected_away) / 2)))
        return goals, goals

    @staticmethod
    def _apply_match_to_standings(
        standings: Dict[str, Dict[str, Any]],
        result: SimulatedMatch,
    ) -> None:
        home = standings[result.home_team]
        away = standings[result.away_team]

        home["played"] += 1
        away["played"] += 1
        home["goals_for"] += result.home_goals
        home["goals_against"] += result.away_goals
        away["goals_for"] += result.away_goals
        away["goals_against"] += result.home_goals
        home["goal_difference"] = home["goals_for"] - home["goals_against"]
        away["goal_difference"] = away["goals_for"] - away["goals_against"]

        if result.home_goals > result.away_goals:
            home["points"] += 3
            home["wins"] += 1
            away["losses"] += 1
        elif result.away_goals > result.home_goals:
            away["points"] += 3
            away["wins"] += 1
            home["losses"] += 1
        else:
            home["points"] += 1
            away["points"] += 1
            home["draws"] += 1
            away["draws"] += 1

    def _rank_group(
        self,
        standings: Dict[str, Dict[str, Any]],
        results: list[SimulatedMatch],
    ) -> list[Dict[str, Any]]:
        """
        Rank a group using the available 2026 rules.

        Implemented criteria:
        1. points;
        2. head-to-head points among tied teams;
        3. head-to-head goal difference among tied teams;
        4. head-to-head goals scored among tied teams;
        5. overall goal difference;
        6. overall goals scored.

        Fair-play points and FIFA ranking are official later criteria, but this
        dataset does not yet contain cards or current FIFA ranking snapshots, so
        remaining ties are resolved by simulation randomness.
        """
        teams_by_points: Dict[int, list[Dict[str, Any]]] = defaultdict(list)
        for row in standings.values():
            teams_by_points[int(row["points"])].append(row.copy())

        ranked_rows = []
        for points in sorted(teams_by_points, reverse=True):
            tied_rows = teams_by_points[points]
            if len(tied_rows) == 1:
                ranked_rows.append(tied_rows[0])
                continue

            h2h = self._head_to_head_stats(
                [row["team"] for row in tied_rows],
                results,
            )
            ranked_rows.extend(
                sorted(
                    tied_rows,
                    key=lambda row: (
                        h2h[row["team"]]["points"],
                        h2h[row["team"]]["goal_difference"],
                        h2h[row["team"]]["goals_for"],
                        row["goal_difference"],
                        row["goals_for"],
                        self.rng.random(),
                    ),
                    reverse=True,
                )
            )

        return ranked_rows

    @staticmethod
    def _head_to_head_stats(
        tied_teams: list[str],
        results: list[SimulatedMatch],
    ) -> Dict[str, Dict[str, int]]:
        """Build head-to-head mini-table stats for tied teams."""
        tied_set = set(tied_teams)
        table = {
            team: {
                "points": 0,
                "goals_for": 0,
                "goals_against": 0,
                "goal_difference": 0,
            }
            for team in tied_teams
        }

        for result in results:
            if result.home_team not in tied_set or result.away_team not in tied_set:
                continue

            home = table[result.home_team]
            away = table[result.away_team]
            home["goals_for"] += result.home_goals
            home["goals_against"] += result.away_goals
            away["goals_for"] += result.away_goals
            away["goals_against"] += result.home_goals

            if result.home_goals > result.away_goals:
                home["points"] += 3
            elif result.away_goals > result.home_goals:
                away["points"] += 3
            else:
                home["points"] += 1
                away["points"] += 1

        for row in table.values():
            row["goal_difference"] = row["goals_for"] - row["goals_against"]

        return table

    def _select_group_qualifiers(
        self,
        group_rankings: Dict[str, list[Dict[str, Any]]],
    ) -> list[Dict[str, Any]]:
        qualifiers = []
        third_places = []

        for group, ranking in sorted(group_rankings.items()):
            for position, row in enumerate(ranking):
                record = row.copy()
                record["group"] = group
                record["group_position"] = int(position) + 1
                record["_tie_breaker"] = self.rng.random()
                if position < 2:
                    qualifiers.append(record)
                elif position == 2:
                    third_places.append(record)

        best_thirds = sorted(
            third_places,
            key=lambda row: (
                row["points"],
                row["goal_difference"],
                row["goals_for"],
                row["_tie_breaker"],
            ),
            reverse=True,
        )[:8]
        qualifiers.extend(best_thirds)

        return qualifiers

    def _seed_round_of_32(
        self,
        qualifiers: list[Dict[str, Any]],
    ) -> list[tuple[str, str]]:
        seeded = [
            row["team"]
            for row in sorted(
                qualifiers,
                key=lambda row: (
                    row["points"],
                    row["goal_difference"],
                    row["goals_for"],
                    -row["group_position"],
                    row["_tie_breaker"],
                ),
                reverse=True,
            )
        ]

        if len(seeded) % 2 != 0:
            raise ValueError("The knockout field must contain an even number of teams")

        return [
            (seeded[index], seeded[-index - 1])
            for index in range(len(seeded) // 2)
        ]

    def _simulate_knockout_winner(self, team_a: str, team_b: str) -> str:
        probabilities = self._knockout_probabilities(team_a, team_b)
        p_team_a = _normalize_probabilities(
            [probabilities["home_win"], probabilities["away_win"]]
        )[0]
        return team_a if self.rng.random() < p_team_a else team_b

    def _knockout_probabilities(self, team_a: str, team_b: str) -> Dict[str, float]:
        key = (team_a, team_b)
        if key not in self._knockout_probability_cache:
            prediction = self.predictor.predict_match(team_a, team_b, is_home_a=True)
            self._knockout_probability_cache[key] = {
                "home_win": _safe_float(prediction.get("home_win"), 0.375)
                + _safe_float(prediction.get("draw"), 0.25) / 2,
                "away_win": _safe_float(prediction.get("away_win"), 0.375)
                + _safe_float(prediction.get("draw"), 0.25) / 2,
            }
        return self._knockout_probability_cache[key]

    @staticmethod
    def _next_stage_key(stage_key: str) -> str:
        order = ["round_of_32", "round_of_16", "quarter_finals", "semi_finals", "final"]
        return order[order.index(stage_key) + 1]

    @staticmethod
    def _pair_adjacent(teams: list[str]) -> list[tuple[str, str]]:
        return [
            (teams[index], teams[index + 1])
            for index in range(0, len(teams), 2)
        ]

    @staticmethod
    def _build_probability_table(
        teams: list[str],
        stage_counts: Dict[str, Counter],
        points_sum: Counter,
        n_simulations: int,
    ) -> pd.DataFrame:
        rows = []
        for team in teams:
            counts = stage_counts[team]
            rows.append(
                {
                    "team": team,
                    "avg_group_points": points_sum[team] / n_simulations,
                    "pass_group": counts["round_of_32"] / n_simulations,
                    "round_of_16": counts["round_of_16"] / n_simulations,
                    "quarter_finals": counts["quarter_finals"] / n_simulations,
                    "semi_finals": counts["semi_finals"] / n_simulations,
                    "final": counts["final"] / n_simulations,
                    "title": counts["title"] / n_simulations,
                }
            )
        return pd.DataFrame(rows).sort_values("title", ascending=False)

    @staticmethod
    def _build_matchup_table(
        matchup_counts: Counter,
        n_simulations: int,
        limit: int = 40,
    ) -> pd.DataFrame:
        rows = [
            {
                "stage": stage,
                "team_a": team_a,
                "team_b": team_b,
                "probability": count / n_simulations,
                "simulations": count,
            }
            for (stage, team_a, team_b), count in matchup_counts.items()
        ]
        if not rows:
            return pd.DataFrame(columns=["stage", "team_a", "team_b", "probability"])
        return (
            pd.DataFrame(rows)
            .sort_values("probability", ascending=False)
            .head(limit)
            .reset_index(drop=True)
        )

    @staticmethod
    def _build_group_difficulty_table(
        third_place_points: Counter,
        n_simulations: int,
    ) -> pd.DataFrame:
        rows = [
            {
                "group": group,
                "avg_third_place_points": points / n_simulations,
            }
            for group, points in third_place_points.items()
        ]
        return pd.DataFrame(rows).sort_values(
            "avg_third_place_points",
            ascending=False,
        )
