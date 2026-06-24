"""
Group Stage Details Page
Detailed group stage analysis
"""
import streamlit as st
import pandas as pd
from src.utils.flag_images import get_flag_html
from src.app.components.match_text import format_fixture_datetime
from src.models.schedule_predictions import prediction_for_match
from src.app.components.prediction_text import (
    format_expected_goals,
    score_explanation,
    score_label,
)

# Access global state
ensemble = st.session_state.get("predictor")
schedule = st.session_state.get("schedule_2026")
predictions = st.session_state.get("schedule_predictions")
show_prob = st.session_state.get("show_probabilities", True)
show_scores = st.session_state.get("show_scores", True)

def show_group_stage_details(ensemble, schedule, predictions, show_prob, show_scores):
    """Show detailed group stage view"""
    
    st.markdown("## Group Stage - Detailed View")
    
    group_matches = schedule[schedule['stage'] == 'Group Stage'].copy()
    groups = sorted([g for g in group_matches['group'].unique() if pd.notna(g)])
    
    selected_group = st.selectbox("Select Group", groups)
    
    group_data = group_matches[group_matches['group'] == selected_group].sort_values(
        ['date', 'time']
    )
    
    st.markdown(f"### Group {selected_group}")
    
    for idx, match in group_data.iterrows():
        with st.expander(
            f"{match['home_team']} vs {match['away_team']} - "
            f"{format_fixture_datetime(match)}"
        ):
            display_match_detailed(ensemble, match, predictions)

def display_match_detailed(ensemble, match, predictions=None):
    """Display detailed match information"""
    
    team_a = match['home_team']
    team_b = match['away_team']
    flag_a = get_flag_html(team_a, size='w40', height=20)
    flag_b = get_flag_html(team_b, size='w40', height=20)
    
    # Check if match is completed
    status = match.get('status', 'Scheduled')
    is_completed = status == 'Completed' and pd.notna(match.get('home_score')) and pd.notna(match.get('away_score'))
    
    try:
        prediction = prediction_for_match(predictions, match)
        if prediction is None:
            prediction = ensemble.predict_match(team_a, team_b, is_home_a=True)
        
        predicted_home, predicted_away = prediction['most_likely_score']

        score_col, goals_col = st.columns(2)
        score_col.metric(
            score_label(prediction),
            f"{predicted_home} - {predicted_away}",
        )
        goals_col.metric(
            "Gols esperados (xG)",
            format_expected_goals(prediction).replace("–", " - "),
        )
        explanation = score_explanation(prediction)
        if explanation:
            st.caption(explanation)

        # Win probabilities
        col1, col2, col3 = st.columns(3)
        
        with col1:
            st.markdown(f"**{flag_a} {team_a}**", unsafe_allow_html=True)
            st.markdown('<div style="display: inline-block; padding: 2px 8px; background-color: #e3f2fd; border-radius: 4px; margin-top: 4px;"><small>Win</small></div>', unsafe_allow_html=True)
            st.progress(prediction['home_win'])
            st.markdown(f"{prediction['home_win']:.1%}")
        
        with col2:
            st.markdown("**&nbsp;**", unsafe_allow_html=True)
            st.markdown('<div style="display: inline-block; padding: 2px 8px; background-color: #fff3cd; border-radius: 4px; margin-top: 4px;"><small><b>Draw</b></small></div>', unsafe_allow_html=True)
            st.progress(prediction['draw'])
            st.markdown(f"{prediction['draw']:.1%}")
        
        with col3:
            st.markdown(f"**{flag_b} {team_b}**", unsafe_allow_html=True)
            st.markdown('<div style="display: inline-block; padding: 2px 8px; background-color: #e3f2fd; border-radius: 4px; margin-top: 4px;"><small>Win</small></div>', unsafe_allow_html=True)
            st.progress(prediction['away_win'])
            st.markdown(f"{prediction['away_win']:.1%}")
        
        st.markdown("---")
        
        # Prediction vs Reality comparison
        if is_completed:
            st.markdown("### Prediction vs Reality")
            
            actual_home = int(match['home_score'])
            actual_away = int(match['away_score'])
            col1, col2 = st.columns(2)
            
            with col1:
                st.markdown("**Model Prediction**")
                st.markdown(f"### {predicted_home} - {predicted_away}")
                
                # Determine predicted winner
                if predicted_home > predicted_away:
                    st.markdown(f"Predicted Winner: {flag_a} **{team_a}**", unsafe_allow_html=True)
                elif predicted_away > predicted_home:
                    st.markdown(f"Predicted Winner: {flag_b} **{team_b}**", unsafe_allow_html=True)
                else:
                    st.markdown("Predicted: **Draw**")
            
            with col2:
                st.markdown("**Actual Result**")
                st.markdown(f"### {actual_home} - {actual_away}")
                
                # Determine actual winner
                if actual_home > actual_away:
                    st.markdown(f"Actual Winner: {flag_a} **{team_a}**", unsafe_allow_html=True)
                elif actual_away > actual_home:
                    st.markdown(f"Actual Winner: {flag_b} **{team_b}**", unsafe_allow_html=True)
                else:
                    st.markdown("Result: **Draw**")
            
            # Accuracy check
            st.markdown("---")
            predicted_winner = "home" if predicted_home > predicted_away else ("away" if predicted_away > predicted_home else "draw")
            actual_winner = "home" if actual_home > actual_away else ("away" if actual_away > actual_home else "draw")
            
            if predicted_winner == actual_winner:
                st.success("✓ Model correctly predicted the winner!")
            else:
                st.error("✗ Model prediction was incorrect")
            
            st.markdown("---")
        
    except Exception as e:
        st.error(f"Error: {str(e)}")


# Execute page
if ensemble and schedule is not None:
    show_group_stage_details(ensemble, schedule, predictions, show_prob, show_scores)
else:
    st.error("Failed to load predictor or schedule data")
