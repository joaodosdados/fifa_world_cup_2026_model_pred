"""
Tournament Bracket Page
Complete tournament visualization with all stages
"""
import streamlit as st
import pandas as pd
from src.utils.flag_images import get_flag_html
from src.components.match_display import display_match_card_compact

# Access global state
ensemble = st.session_state.get("predictor")
schedule = st.session_state.get("schedule_2026")
show_prob = st.session_state.get("show_probabilities", True)
show_scores = st.session_state.get("show_scores", True)

def show_tournament_bracket(ensemble, schedule, show_prob, show_scores):
    """Display tournament in bracket format"""
    
    # Tournament stages tabs
    tab1, tab2, tab3, tab4, tab5 = st.tabs([
        "Group Stage", "Octave-finals", "Quarter-Finals", "Semi-Finals", "Final"
    ])
    
    with tab1:
        show_group_stage_expanded(ensemble, schedule, show_prob, show_scores)
    
    with tab2:
        st.info("Octave-finals matches will be determined after group stage completion")
        show_knockout_stage(ensemble, schedule, "Round of 16", show_prob, show_scores)
    
    with tab3:
        st.info("Quarter-finals will be determined after Octave-finals")
        show_knockout_stage(ensemble, schedule, "Quarter-finals", show_prob, show_scores)
    
    with tab4:
        st.info("Semi-finals will be determined after Quarter-finals")
        show_knockout_stage(ensemble, schedule, "Semi-finals", show_prob, show_scores)
    
    with tab5:
        st.info("Final will be determined after Semi-finals")
        show_knockout_stage(ensemble, schedule, "Final", show_prob, show_scores)


def show_group_stage_expanded(ensemble, schedule, show_prob, show_scores):
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
    display_group_row(ensemble, group_matches, selected_group, show_prob, show_scores)


def display_group_row(ensemble, matches, group, show_prob, show_scores):
    """Display ONE GROUP as a complete ROW with all teams and matches"""
    
    group_data = matches[matches['group'] == group].sort_values('date')
    
    # Group header only
    st.markdown(f"### Grupo {group}")
    
    # Show all matches for this group in columns
    if len(group_data) > 0:
        # Display matches in 2 columns for better readability
        match_cols = st.columns(2)
        for idx, (_, match) in enumerate(group_data.iterrows()):
            with match_cols[idx % 2]:
                display_match_card_compact(ensemble, match, show_prob, show_scores)


def show_knockout_stage(ensemble, schedule, stage_name, show_prob, show_scores):
    """Display knockout stage matches with visual bracket layout"""
    
    stage_matches = schedule[schedule['stage'] == stage_name]
    
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
            display_match_card_compact(ensemble, match, show_prob, show_scores)


# Execute page
if ensemble and schedule is not None:
    show_tournament_bracket(ensemble, schedule, show_prob, show_scores)
else:
    st.error("Failed to load predictor or schedule data")
