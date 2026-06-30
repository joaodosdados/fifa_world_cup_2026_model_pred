"""
Tournament Bracket Page
Complete tournament visualization with all stages
"""
import streamlit as st
import pandas as pd
from src.utils.flag_images import get_flag_html
from src.utils.team_names import normalize_team_name
from src.app.components.match_display import display_match_card_compact

ROUND_OF_32_FIXTURES = [
    {
        "match_number": 73,
        "date": "2026-06-28",
        "time": "1:00 p.m. ET",
        "venue": "Toronto Stadium",
        "home": ("position", "B", 2),
        "away": ("position", "A", 2),
    },
    {
        "match_number": 76,
        "date": "2026-06-29",
        "time": "1:00 p.m. ET",
        "venue": "Houston Stadium",
        "home": ("position", "C", 1),
        "away": ("position", "F", 2),
    },
    {
        "match_number": 74,
        "date": "2026-06-29",
        "time": "4:30 p.m. ET",
        "venue": "Boston Stadium",
        "home": ("position", "E", 1),
        "away": ("third", "D"),
    },
    {
        "match_number": 75,
        "date": "2026-06-29",
        "time": "9:00 p.m. ET",
        "venue": "Monterrey Stadium",
        "home": ("position", "F", 1),
        "away": ("position", "C", 2),
    },
    {
        "match_number": 78,
        "date": "2026-06-30",
        "time": "1:00 p.m. ET",
        "venue": "Dallas Stadium",
        "home": ("position", "E", 2),
        "away": ("position", "I", 2),
    },
    {
        "match_number": 77,
        "date": "2026-06-30",
        "time": "5:00 p.m. ET",
        "venue": "New York/New Jersey Stadium",
        "home": ("position", "I", 1),
        "away": ("third", "F"),
    },
    {
        "match_number": 79,
        "date": "2026-06-30",
        "time": "9:00 p.m. ET",
        "venue": "Mexico City Stadium",
        "home": ("position", "A", 1),
        "away": ("third", "E"),
    },
    {
        "match_number": 80,
        "date": "2026-07-01",
        "time": "12:00 p.m. ET",
        "venue": "Atlanta Stadium",
        "home": ("position", "L", 1),
        "away": ("third", "K"),
    },
    {
        "match_number": 82,
        "date": "2026-07-01",
        "time": "4:00 p.m. ET",
        "venue": "Seattle Stadium",
        "home": ("position", "G", 1),
        "away": ("third", "I"),
    },
    {
        "match_number": 81,
        "date": "2026-07-01",
        "time": "8:00 p.m. ET",
        "venue": "San Francisco Bay Stadium",
        "home": ("position", "D", 1),
        "away": ("third", "B"),
    },
    {
        "match_number": 84,
        "date": "2026-07-02",
        "time": "3:00 p.m. ET",
        "venue": "Los Angeles Stadium",
        "home": ("position", "H", 1),
        "away": ("position", "J", 2),
    },
    {
        "match_number": 83,
        "date": "2026-07-02",
        "time": "7:00 p.m. ET",
        "venue": "Toronto Stadium",
        "home": ("position", "K", 2),
        "away": ("position", "L", 2),
    },
    {
        "match_number": 85,
        "date": "2026-07-02",
        "time": "11:00 p.m. ET",
        "venue": "BC Place Vancouver",
        "home": ("position", "B", 1),
        "away": ("third", "J"),
    },
    {
        "match_number": 88,
        "date": "2026-07-03",
        "time": "2:00 p.m. ET",
        "venue": "Dallas Stadium",
        "home": ("position", "D", 2),
        "away": ("position", "G", 2),
    },
    {
        "match_number": 86,
        "date": "2026-07-03",
        "time": "6:00 p.m. ET",
        "venue": "Miami Stadium",
        "home": ("position", "J", 1),
        "away": ("position", "H", 2),
    },
    {
        "match_number": 87,
        "date": "2026-07-03",
        "time": "9:30 p.m. ET",
        "venue": "Kansas City Stadium",
        "home": ("position", "K", 1),
        "away": ("third", "L"),
    },
]

# Access global state
ensemble = st.session_state.get("predictor")
schedule = st.session_state.get("schedule_2026")
predictions = st.session_state.get("schedule_predictions")
show_prob = st.session_state.get("show_probabilities", True)
show_scores = st.session_state.get("show_scores", True)

def show_tournament_bracket(ensemble, schedule, predictions, show_prob, show_scores):
    """Display tournament in bracket format"""
    
    # Tournament stages tabs
    tab1, tab2, tab3, tab4, tab5 = st.tabs([
        "Group Stage", "Mata-mata", "Quarter-Finals", "Semi-Finals", "Final"
    ])
    
    with tab1:
        show_group_stage_expanded(ensemble, schedule, predictions, show_prob, show_scores)
    
    with tab2:
        show_knockout_stage(ensemble, schedule, predictions, "Round of 32", show_prob, show_scores)
    
    with tab3:
        st.info("Quarter-finals will be determined after Octave-finals")
        show_knockout_stage(ensemble, schedule, predictions, "Quarter-finals", show_prob, show_scores)
    
    with tab4:
        st.info("Semi-finals will be determined after Quarter-finals")
        show_knockout_stage(ensemble, schedule, predictions, "Semi-finals", show_prob, show_scores)
    
    with tab5:
        st.info("Final will be determined after Semi-finals")
        show_knockout_stage(ensemble, schedule, predictions, "Final", show_prob, show_scores)


def show_group_stage_expanded(ensemble, schedule, predictions, show_prob, show_scores):
    """Display group stage matches with selector"""
    
    # Get group stage matches
    group_matches = schedule[schedule['stage'] == 'Group Stage'].copy()
    
    # Get unique groups
    groups = sorted([g for g in group_matches['group'].unique() if pd.notna(g)])
    
    if len(groups) == 0:
        st.warning("No group data available")
        return
    
    # Group selector
    selected_group = st.selectbox(
        "Selecione o Grupo",
        options=groups,
        format_func=lambda x: f"Grupo {x}",
        key="group_selector"
    )
    
    st.markdown("---")
    
    # Display selected group
    display_group_row(ensemble, group_matches, predictions, selected_group, show_prob, show_scores)


def display_group_row(ensemble, matches, predictions, group, show_prob, show_scores):
    """Display ONE GROUP as a complete ROW with all teams and matches"""
    
    group_data = matches[matches['group'] == group].sort_values(['date', 'time'])
    
    # Group header only
    st.markdown(f"### Grupo {group}")
    
    # Show all matches for this group in columns
    if len(group_data) > 0:
        # Display matches in 2 columns for better readability
        match_cols = st.columns(2)
        for idx, (_, match) in enumerate(group_data.iterrows()):
            with match_cols[idx % 2]:
                display_match_card_compact(
                    ensemble,
                    match,
                    show_prob,
                    show_scores,
                    predictions=predictions,
                )


def show_knockout_stage(ensemble, schedule, predictions, stage_name, show_prob, show_scores):
    """Display knockout stage matches with visual bracket layout"""
    
    stage_matches = schedule[schedule['stage'] == stage_name]

    if stage_name == "Round of 32":
        show_round_of_32_from_group_results(ensemble, schedule, show_prob, show_scores)
        return
    
    # Round of 16 - 8 matches
    if stage_name == "Round of 16":
        st.markdown("### Octave-finals - Knockout Stage")
        st.markdown("*Matches will be determined after group stage completion*")
        st.markdown("---")
        
        matches_data = [
            {"label": "Match 1", "team1": "Winner Group A", "team2": "Runner-up Group B"},
            {"label": "Match 2", "team1": "Winner Group C", "team2": "Runner-up Group D"},
            {"label": "Match 3", "team1": "Winner Group E", "team2": "Runner-up Group F"},
            {"label": "Match 4", "team1": "Winner Group G", "team2": "Runner-up Group H"},
            {"label": "Match 5", "team1": "Winner Group B", "team2": "Runner-up Group A"},
            {"label": "Match 6", "team1": "Winner Group D", "team2": "Runner-up Group C"},
            {"label": "Match 7", "team1": "Winner Group F", "team2": "Runner-up Group E"},
            {"label": "Match 8", "team1": "Winner Group H", "team2": "Runner-up Group G"},
        ]
        
        col1, col2 = st.columns(2)
        
        for i, match_info in enumerate(matches_data):
            col = col1 if i < 4 else col2
            
            with col:
                with st.container():
                    st.markdown(f"#### {match_info['label']}")
                    team_col1, vs_col, team_col2 = st.columns([2, 1, 2])
                    
                    with team_col1:
                        st.markdown(f"**{match_info['team1']}**")
                    with vs_col:
                        st.markdown("<p style='text-align: center;'>vs</p>", unsafe_allow_html=True)
                    with team_col2:
                        st.markdown(f"**{match_info['team2']}**")
                    
                    st.caption("Date: TBD | Venue: TBD")
                    st.markdown("---")
        
        return
    
    # Quarter-Finals - 4 matches
    elif stage_name == "Quarter-finals":
        st.markdown("### Quarter-Finals")
        st.markdown("*Matches will be determined after Octave-finals*")
        st.markdown("---")
        
        matches_data = [
            {"label": "Quarter-Final 1", "team1": "Winner Match 1", "team2": "Winner Match 2"},
            {"label": "Quarter-Final 2", "team1": "Winner Match 3", "team2": "Winner Match 4"},
            {"label": "Quarter-Final 3", "team1": "Winner Match 5", "team2": "Winner Match 6"},
            {"label": "Quarter-Final 4", "team1": "Winner Match 7", "team2": "Winner Match 8"},
        ]
        
        col1, col2 = st.columns(2)
        
        for i, match_info in enumerate(matches_data):
            col = col1 if i < 2 else col2
            
            with col:
                with st.container():
                    st.markdown(f"#### {match_info['label']}")
                    team_col1, vs_col, team_col2 = st.columns([2, 1, 2])
                    
                    with team_col1:
                        st.markdown(f"**{match_info['team1']}**")
                    with vs_col:
                        st.markdown("<p style='text-align: center;'>vs</p>", unsafe_allow_html=True)
                    with team_col2:
                        st.markdown(f"**{match_info['team2']}**")
                    
                    st.caption("Date: TBD | Venue: TBD")
                    st.markdown("---")
        
        return
    
    # Semi-Finals - 2 matches
    elif stage_name == "Semi-finals":
        st.markdown("### Semi-Finals")
        st.markdown("*Matches will be determined after Quarter-Finals*")
        st.markdown("---")
        
        matches_data = [
            {"label": "Semi-Final 1", "team1": "Winner QF1", "team2": "Winner QF2"},
            {"label": "Semi-Final 2", "team1": "Winner QF3", "team2": "Winner QF4"},
        ]
        
        col1, col2 = st.columns(2)
        
        for i, match_info in enumerate(matches_data):
            col = col1 if i == 0 else col2
            
            with col:
                with st.container():
                    st.markdown(f"#### {match_info['label']}")
                    team_col1, vs_col, team_col2 = st.columns([2, 1, 2])
                    
                    with team_col1:
                        st.markdown(f"**{match_info['team1']}**")
                    with vs_col:
                        st.markdown("<p style='text-align: center;'>vs</p>", unsafe_allow_html=True)
                    with team_col2:
                        st.markdown(f"**{match_info['team2']}**")
                    
                    st.caption("Date: TBD | Venue: TBD")
                    st.markdown("---")
        
        return
    
    # Final - 1 match
    elif stage_name == "Final":
        st.markdown("### Final - World Cup 2026")
        st.markdown("*The ultimate match will be determined after Semi-Finals*")
        st.markdown("---")
        
        # Center the final match
        col1, col2, col3 = st.columns([1, 2, 1])
        
        with col2:
            with st.container():
                st.markdown("#### World Cup Final")
                team_col1, vs_col, team_col2 = st.columns([2, 1, 2])
                
                with team_col1:
                    st.markdown("**Winner SF1**")
                with vs_col:
                    st.markdown("<p style='text-align: center;'>vs</p>", unsafe_allow_html=True)
                with team_col2:
                    st.markdown("**Winner SF2**")
                
                st.caption("Date: TBD | Venue: TBD")
                st.markdown("---")
                
                # Add third place match
                st.markdown("#### Third Place Match")
                team_col1, vs_col, team_col2 = st.columns([2, 1, 2])
                
                with team_col1:
                    st.markdown("**Loser SF1**")
                with vs_col:
                    st.markdown("<p style='text-align: center;'>vs</p>", unsafe_allow_html=True)
                with team_col2:
                    st.markdown("**Loser SF2**")
                
                st.caption("Date: TBD | Venue: TBD")
        
        return
    
    # For other stages or if data exists, use existing logic
    if len(stage_matches) == 0:
        st.warning(f"No {stage_name} matches scheduled yet")
        return
    
    cols = st.columns(2)
    
    for idx, match in stage_matches.iterrows():
        col = cols[idx % 2]
        with col:
            display_match_card_compact(
                ensemble,
                match,
                show_prob,
                show_scores,
                predictions=predictions,
            )


def show_round_of_32_from_group_results(ensemble, schedule, show_prob, show_scores):
    """Build the first knockout round from completed group-stage standings."""
    group_matches = schedule[schedule["stage"].astype(str).eq("Group Stage")].copy()
    if group_matches.empty:
        st.warning("No group-stage data available.")
        return

    rankings = build_group_rankings(group_matches)
    incomplete_groups = [
        group
        for group, ranking in rankings.items()
        if any(row["played"] < 3 for row in ranking)
    ]
    if incomplete_groups:
        st.info(
            "A fase de grupos ainda não está completa para: "
            + ", ".join(f"Grupo {group}" for group in incomplete_groups)
            + "."
        )
        return

    qualifiers = select_knockout_qualifiers(rankings)
    if len(qualifiers) < 32:
        st.warning("Ainda não há 32 classificados para montar o mata-mata.")
        return

    round_matches = build_round_of_32_matches(rankings, schedule)
    top_two = [row for row in qualifiers if row["group_position"] <= 2]
    best_thirds = [row for row in qualifiers if row["group_position"] == 3]

    st.markdown("### Mata-mata definido pela fase de grupos")
    st.caption(
        "Formato 2026: os dois primeiros de cada grupo e os 8 melhores terceiros "
        "avançam. Os confrontos abaixo seguem o chaveamento da Rodada de 32."
    )

    metric_cols = st.columns(4)
    metric_cols[0].metric("Classificados", len(qualifiers))
    metric_cols[1].metric("Top-2", len(top_two))
    metric_cols[2].metric("Melhores 3º", len(best_thirds))
    metric_cols[3].metric("Jogos", len(round_matches))

    with st.expander("Ver classificados e melhores terceiros", expanded=False):
        display_qualifiers_table(qualifiers)

    st.markdown("---")
    st.markdown("#### Rodada de 32")

    cols = st.columns(2)
    for index, match in enumerate(round_matches, start=1):
        with cols[(index - 1) % 2]:
            display_knockout_match_card(
                ensemble,
                match,
                show_prob,
                show_scores,
            )


def build_group_rankings(group_matches):
    """Return deterministic standings for every completed group."""
    rankings = {}
    for group, matches in group_matches.groupby("group", sort=True):
        teams = sorted(
            set(matches["home_team"].dropna().astype(str))
            | set(matches["away_team"].dropna().astype(str))
        )
        standings = {team: empty_standing(team) for team in teams}
        results = []

        for _, match in matches.iterrows():
            if not is_completed_match(match):
                continue

            home_team = str(match["home_team"])
            away_team = str(match["away_team"])
            home_goals = int(float(match["home_score"]))
            away_goals = int(float(match["away_score"]))
            results.append(
                {
                    "home_team": home_team,
                    "away_team": away_team,
                    "home_goals": home_goals,
                    "away_goals": away_goals,
                }
            )
            apply_result_to_standings(
                standings,
                home_team,
                away_team,
                home_goals,
                away_goals,
            )

        rankings[str(group)] = rank_group(standings, results)
    return rankings


def empty_standing(team):
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


def is_completed_match(match):
    return (
        str(match.get("status", "")).casefold() == "completed"
        and pd.notna(match.get("home_score"))
        and pd.notna(match.get("away_score"))
    )


def apply_result_to_standings(standings, home_team, away_team, home_goals, away_goals):
    home = standings[home_team]
    away = standings[away_team]
    home["played"] += 1
    away["played"] += 1
    home["goals_for"] += home_goals
    home["goals_against"] += away_goals
    away["goals_for"] += away_goals
    away["goals_against"] += home_goals
    home["goal_difference"] = home["goals_for"] - home["goals_against"]
    away["goal_difference"] = away["goals_for"] - away["goals_against"]

    if home_goals > away_goals:
        home["points"] += 3
        home["wins"] += 1
        away["losses"] += 1
    elif away_goals > home_goals:
        away["points"] += 3
        away["wins"] += 1
        home["losses"] += 1
    else:
        home["points"] += 1
        away["points"] += 1
        home["draws"] += 1
        away["draws"] += 1


def rank_group(standings, results):
    """Rank a group using available FIFA criteria plus stable team-name fallback."""
    rows = [row.copy() for row in standings.values()]
    ranked = []
    for points in sorted({row["points"] for row in rows}, reverse=True):
        tied_rows = [row for row in rows if row["points"] == points]
        if len(tied_rows) == 1:
            ranked.extend(tied_rows)
            continue

        h2h = head_to_head_stats([row["team"] for row in tied_rows], results)
        ranked.extend(
            sorted(
                tied_rows,
                key=lambda row: (
                    h2h[row["team"]]["points"],
                    h2h[row["team"]]["goal_difference"],
                    h2h[row["team"]]["goals_for"],
                    row["goal_difference"],
                    row["goals_for"],
                    row["team"],
                ),
                reverse=True,
            )
        )
    return ranked


def head_to_head_stats(tied_teams, results):
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
        if result["home_team"] not in tied_set or result["away_team"] not in tied_set:
            continue

        home = table[result["home_team"]]
        away = table[result["away_team"]]
        home["goals_for"] += result["home_goals"]
        home["goals_against"] += result["away_goals"]
        away["goals_for"] += result["away_goals"]
        away["goals_against"] += result["home_goals"]

        if result["home_goals"] > result["away_goals"]:
            home["points"] += 3
        elif result["away_goals"] > result["home_goals"]:
            away["points"] += 3
        else:
            home["points"] += 1
            away["points"] += 1

    for row in table.values():
        row["goal_difference"] = row["goals_for"] - row["goals_against"]
    return table


def select_knockout_qualifiers(rankings):
    qualifiers = []
    third_places = []

    for group, ranking in sorted(rankings.items()):
        for position, row in enumerate(ranking, start=1):
            record = row.copy()
            record["group"] = group
            record["group_position"] = position
            if position <= 2:
                qualifiers.append(record)
            elif position == 3:
                third_places.append(record)

    best_thirds = sorted(
        third_places,
        key=lambda row: (
            row["points"],
            row["goal_difference"],
            row["goals_for"],
            row["team"],
        ),
        reverse=True,
    )[:8]
    qualifiers.extend(best_thirds)
    return sorted(
        qualifiers,
        key=lambda row: (
            row["points"],
            row["goal_difference"],
            row["goals_for"],
            -row["group_position"],
            row["team"],
        ),
        reverse=True,
    )


def seed_round_of_32(qualifiers):
    seeded = [row["team"] for row in qualifiers]
    return [
        (seeded[index], seeded[-index - 1])
        for index in range(len(seeded) // 2)
    ]


def build_round_of_32_matches(rankings, schedule=None):
    """Resolve official Round of 32 fixtures from completed standings."""
    matches = []
    for fixture in ROUND_OF_32_FIXTURES:
        home_team = resolve_fixture_team(rankings, fixture["home"])
        away_team = resolve_fixture_team(rankings, fixture["away"])
        if not home_team or not away_team:
            continue
        match = fixture.copy()
        match["home_team"] = home_team
        match["away_team"] = away_team
        result = completed_knockout_result(schedule, match) if schedule is not None else None
        if result:
            match.update(result)
        matches.append(match)
    return matches


def completed_knockout_result(schedule, generated_match):
    """Find a completed schedule row for a generated knockout fixture."""
    if schedule is None or schedule.empty:
        return None

    knockout_rows = schedule[
        ~schedule["stage"].astype(str).eq("Group Stage")
        & schedule["status"].astype(str).str.casefold().eq("completed")
    ].copy()
    if knockout_rows.empty:
        return None

    match_number = generated_match.get("match_number")
    if "match_id" in knockout_rows.columns and match_number is not None:
        match_ids = pd.to_numeric(knockout_rows["match_id"], errors="coerce")
        same_id = knockout_rows[match_ids.eq(int(match_number))]
        for _, row in same_id.iterrows():
            if row_matches_generated_fixture(row, generated_match):
                return result_from_schedule_row(row, generated_match, schedule)

    generated_teams = {
        normalized_team_key(generated_match["home_team"]),
        normalized_team_key(generated_match["away_team"]),
    }
    for _, row in knockout_rows.iterrows():
        if row_matches_generated_fixture(row, generated_match):
            return result_from_schedule_row(row, generated_match, schedule)

    return None


def normalized_team_key(team):
    return normalize_team_name(str(team)).strip().casefold()


def row_matches_generated_fixture(row, generated_match):
    generated_teams = {
        normalized_team_key(generated_match["home_team"]),
        normalized_team_key(generated_match["away_team"]),
    }
    row_teams = {
        normalized_team_key(row.get("home_team")),
        normalized_team_key(row.get("away_team")),
    }
    return row_teams == generated_teams


def result_from_schedule_row(row, generated_match, schedule=None):
    if pd.isna(row.get("home_score")) or pd.isna(row.get("away_score")):
        return None

    generated_home = normalized_team_key(generated_match["home_team"])
    row_home = normalized_team_key(row.get("home_team"))
    same_order = row_home == generated_home

    home_score = row.get("home_score" if same_order else "away_score")
    away_score = row.get("away_score" if same_order else "home_score")

    result = {
        "home_score": float(home_score),
        "away_score": float(away_score),
        "status": "Completed",
        "result_source_match_id": row.get("match_id"),
    }
    if float(home_score) == float(away_score):
        actual_winner = infer_knockout_winner_from_future_fixtures(
            schedule,
            generated_match,
            row,
        )
        if actual_winner:
            result["actual_winner"] = actual_winner
    return result


def infer_knockout_winner_from_future_fixtures(schedule, generated_match, result_row):
    if schedule is None or schedule.empty:
        return None

    teams = [generated_match["home_team"], generated_match["away_team"]]
    team_keys = {normalized_team_key(team): team for team in teams}
    result_date = pd.to_datetime(result_row.get("date"), errors="coerce")

    candidates = set()
    future_rows = schedule[~schedule["stage"].astype(str).eq("Group Stage")].copy()
    for _, row in future_rows.iterrows():
        if row_matches_generated_fixture(row, generated_match):
            continue

        row_date = pd.to_datetime(row.get("date"), errors="coerce")
        if pd.notna(result_date) and pd.notna(row_date) and row_date <= result_date:
            continue

        row_teams = [
            normalized_team_key(row.get("home_team")),
            normalized_team_key(row.get("away_team")),
        ]
        for team_key, display_name in team_keys.items():
            if team_key in row_teams:
                candidates.add(display_name)

    return next(iter(candidates)) if len(candidates) == 1 else None


def resolve_fixture_team(rankings, selector):
    selector_type = selector[0]
    group = selector[1]

    if selector_type == "position":
        position = selector[2]
        ranking = rankings.get(group, [])
        if len(ranking) < position:
            return None
        return ranking[position - 1]["team"]

    if selector_type == "third":
        ranking = rankings.get(group, [])
        if len(ranking) < 3:
            return None
        return ranking[2]["team"]

    return None


def display_qualifiers_table(qualifiers):
    table = pd.DataFrame(qualifiers)
    table.insert(0, "seed", range(1, len(table) + 1))
    table["Grupo"] = "Grupo " + table["group"].astype(str)
    table["Pos"] = table["group_position"].astype(str) + "º"
    table = table.rename(
        columns={
            "seed": "Seed",
            "team": "Seleção",
            "points": "Pts",
            "played": "J",
            "wins": "V",
            "draws": "E",
            "losses": "D",
            "goals_for": "GP",
            "goals_against": "GC",
            "goal_difference": "SG",
        }
    )
    st.dataframe(
        table[["Seed", "Seleção", "Grupo", "Pos", "Pts", "J", "V", "E", "D", "GP", "GC", "SG"]],
        hide_index=True,
        width="stretch",
    )


def display_knockout_match_card(
    ensemble,
    match,
    show_prob,
    show_scores,
):
    team_a = match["home_team"]
    team_b = match["away_team"]
    flag_a = get_flag_html(team_a, size="w40", height=20)
    flag_b = get_flag_html(team_b, size="w40", height=20)

    try:
        prediction = ensemble.predict_match(team_a, team_b, is_home_a=True)
        home_advances = float(prediction.get("home_win", 0)) + float(prediction.get("draw", 0)) / 2
        away_advances = float(prediction.get("away_win", 0)) + float(prediction.get("draw", 0)) / 2
        total = home_advances + away_advances
        if total > 0:
            home_advances /= total
            away_advances /= total
        predicted_home, predicted_away = prediction.get("most_likely_score", ("-", "-"))
        predicted_winner = team_a if home_advances >= away_advances else team_b
    except Exception as exc:
        prediction = None
        home_advances = away_advances = 0
        predicted_home = predicted_away = "-"
        predicted_winner = None
        st.error(f"Error: {str(exc)}")

    is_completed = (
        str(match.get("status", "")).casefold() == "completed"
        and pd.notna(match.get("home_score"))
        and pd.notna(match.get("away_score"))
    )

    with st.container():
        st.caption(
            f"Jogo {match['match_number']} | {match['date']} {match['time']} | "
            f"{match['venue']}"
        )
        team_col1, vs_col, team_col2 = st.columns([5, 1, 5])

        with team_col1:
            st.markdown(
                f"<h4 style='margin-bottom: 0.25rem;'>{flag_a} {team_a}</h4>",
                unsafe_allow_html=True,
            )
            if is_completed:
                st.markdown(
                    f"Score: **{int(float(match['home_score']))}**",
                    unsafe_allow_html=True,
                )
            if show_prob and prediction is not None:
                st.progress(home_advances)
                st.caption(f"Avança: {home_advances:.1%}")

        with vs_col:
            st.markdown("<h4 style='text-align: center;'>vs</h4>", unsafe_allow_html=True)

        with team_col2:
            st.markdown(
                f"<h4 style='margin-bottom: 0.25rem;'>{flag_b} {team_b}</h4>",
                unsafe_allow_html=True,
            )
            if is_completed:
                st.markdown(
                    f"Score: **{int(float(match['away_score']))}**",
                    unsafe_allow_html=True,
                )
            if show_prob and prediction is not None:
                st.progress(away_advances)
                st.caption(f"Avança: {away_advances:.1%}")

        if show_scores and prediction is not None:
            st.caption(f"Placar mais provável no tempo normal: {predicted_home} - {predicted_away}")

        if is_completed and prediction is not None:
            display_knockout_prediction_feedback(
                team_a,
                team_b,
                match,
                predicted_winner,
                home_advances,
                away_advances,
            )
        st.divider()


def display_knockout_prediction_feedback(
    team_a,
    team_b,
    match,
    predicted_winner,
    home_advances,
    away_advances,
):
    home_score = int(float(match["home_score"]))
    away_score = int(float(match["away_score"]))
    actual_winner = match.get("actual_winner")
    if home_score > away_score:
        actual_winner = team_a
    elif away_score > home_score:
        actual_winner = team_b

    feedback_cols = st.columns(2)
    with feedback_cols[0]:
        st.markdown("**Prediction**")
        predicted_prob = home_advances if predicted_winner == team_a else away_advances
        predicted_flag = get_flag_html(predicted_winner, size="w40", height=20)
        st.markdown(
            (
                '<div style="padding: 10px; background-color: #d1ecf1; '
                'border-radius: 5px; color: #0c5460;">'
                f"{predicted_flag}<b>{predicted_winner}</b> avança "
                f"({predicted_prob:.0%})</div>"
            ),
            unsafe_allow_html=True,
        )

    with feedback_cols[1]:
        st.markdown("**Actual Result**")
        if actual_winner is None:
            st.markdown(
                (
                    '<div style="padding: 10px; background-color: #fff3cd; '
                    'border-radius: 5px; color: #856404;">'
                    "Empate no CSV; vencedor nos pênaltis não disponível</div>"
                ),
                unsafe_allow_html=True,
            )
            return

        actual_flag = get_flag_html(actual_winner, size="w40", height=20)
        correct = predicted_winner == actual_winner
        background = "#d4edda" if correct else "#f8d7da"
        color = "#155724" if correct else "#721c24"
        label = "CORRECT" if correct else "INCORRECT"
        st.markdown(
            (
                f'<div style="padding: 10px; background-color: {background}; '
                f'border-radius: 5px; color: {color};">'
                f"{label}: {actual_flag}<b>{actual_winner}</b> avançou</div>"
            ),
            unsafe_allow_html=True,
        )


# Execute page
if ensemble and schedule is not None:
    show_tournament_bracket(ensemble, schedule, predictions, show_prob, show_scores)
else:
    st.error("Failed to load predictor or schedule data")
