"""
Shared match display components
"""
import streamlit as st
import pandas as pd
from src.utils.flag_images import get_flag_html
from src.utils.team_names import get_team_abbreviation, get_team_abbreviation_with_tooltip
from src.models.schedule_predictions import prediction_for_match
from src.components.match_text import format_fixture_caption
from src.components.prediction_text import format_score_summary

def display_match_card_compact(
    ensemble,
    match,
    show_prob,
    show_scores,
    predictions=None,
):
    """Display compact match card using Streamlit components"""
    
    team_a = match['home_team']
    team_b = match['away_team']
    status = match['status']
    
    try:
        # Get precomputed prediction when available; fall back to direct
        # inference for callers/tests that do not pass a prediction table.
        prediction = prediction_for_match(predictions, match)
        if prediction is None:
            prediction = ensemble.predict_match(team_a, team_b, is_home_a=True)
        predicted_winner = prediction.get("predicted_winner")
        if not predicted_winner:
            predicted_winner = ensemble.predict_winner(
                team_a,
                team_b,
                is_home_a=True,
            )
        
        # Determine winner probability
        if predicted_winner == team_a:
            winner_prob = prediction['home_win']
        elif predicted_winner == team_b:
            winner_prob = prediction['away_win']
        else:
            winner_prob = prediction['draw']

        predicted_home, predicted_away = prediction['most_likely_score']
        
        # Get flags
        flag_a = get_flag_html(team_a, size='w40', height=20)
        flag_b = get_flag_html(team_b, size='w40', height=20)
        
        # Create container for match
        with st.container():
            # Match info
            st.caption(format_fixture_caption(match))
            
            # Teams and scores - balanced layout
            col1, col2, col3 = st.columns([5, 1, 5])
            
            with col1:
                team_a_abbr = get_team_abbreviation_with_tooltip(team_a)
                st.markdown(f"<div style='text-align: left;'><h4 style='margin-bottom: 0.5rem;'>{flag_a} {team_a_abbr}</h4></div>", unsafe_allow_html=True)
                if status == 'Completed' and pd.notna(match['home_score']):
                    st.markdown(f"<div style='text-align: left;'>Score: <strong>{int(match['home_score'])}</strong></div>", unsafe_allow_html=True)
            
            with col2:
                st.markdown("<div style='text-align: center;'><h4>vs</h4></div>", unsafe_allow_html=True)
            
            with col3:
                team_b_abbr = get_team_abbreviation_with_tooltip(team_b)
                st.markdown(f"<div style='text-align: left;'><h4 style='margin-bottom: 0.5rem;'>{flag_b} {team_b_abbr}</h4></div>", unsafe_allow_html=True)
                if status == 'Completed' and pd.notna(match['away_score']):
                    st.markdown(f"<div style='text-align: left;'>Score: <strong>{int(match['away_score'])}</strong></div>", unsafe_allow_html=True)
            
            # Show prediction and result side by side for completed matches
            if status == 'Completed' and pd.notna(match['home_score']):
                actual_a = int(match['home_score'])
                actual_b = int(match['away_score'])
                
                if actual_a > actual_b:
                    actual_winner = team_a
                elif actual_b > actual_a:
                    actual_winner = team_b
                else:
                    actual_winner = "Draw"
                
                correct = predicted_winner == actual_winner
                
                # Two columns: Prediction vs Actual Result
                result_col1, result_col2 = st.columns(2)
                
                with result_col1:
                    st.markdown("**Prediction**")
                    if predicted_winner != "Draw":
                        pred_flag = get_flag_html(predicted_winner, size='w40', height=20)
                        pred_abbr = get_team_abbreviation_with_tooltip(predicted_winner)
                        st.markdown(f'<div style="padding: 10px; background-color: #d1ecf1; border-radius: 5px; color: #0c5460;">{pred_flag} <b>{pred_abbr}</b> ({winner_prob:.0%})</div>', unsafe_allow_html=True)
                    else:
                        st.markdown(f'<div style="padding: 10px; background-color: #d1ecf1; border-radius: 5px; color: #0c5460;"><b>Draw</b> ({winner_prob:.0%})</div>', unsafe_allow_html=True)
                
                with result_col2:
                    st.markdown("**Actual Result**")
                    if actual_winner != "Draw":
                        win_flag = get_flag_html(actual_winner, size='w40', height=20)
                        win_abbr = get_team_abbreviation(actual_winner)
                        if correct:
                            st.markdown(f'<div style="padding: 10px; background-color: #d4edda; border-radius: 5px; color: #155724;">CORRECT: {win_flag}<b>{win_abbr}</b> Won</div>', unsafe_allow_html=True)
                        else:
                            st.markdown(f'<div style="padding: 10px; background-color: #f8d7da; border-radius: 5px; color: #721c24;">INCORRECT: {win_flag}<b>{win_abbr}</b> Won</div>', unsafe_allow_html=True)
                    else:
                        if correct:
                            st.markdown('<div style="padding: 10px; background-color: #d4edda; border-radius: 5px; color: #155724;">CORRECT: <b>Draw</b></div>', unsafe_allow_html=True)
                        else:
                            st.markdown('<div style="padding: 10px; background-color: #f8d7da; border-radius: 5px; color: #721c24;">INCORRECT: <b>Draw</b></div>', unsafe_allow_html=True)
            
            # Show only prediction for scheduled matches
            elif show_prob and status != 'Completed':
                if predicted_winner != "Draw":
                    pred_flag = get_flag_html(predicted_winner, size='w40', height=20)
                    pred_abbr = get_team_abbreviation(predicted_winner)
                    st.markdown(f'<div style="padding: 10px; background-color: #d1ecf1; border-radius: 5px; color: #0c5460;"><b>Prediction:</b> {pred_flag}<b>{pred_abbr}</b> ({winner_prob:.0%})</div>', unsafe_allow_html=True)
                else:
                    st.markdown(f'<div style="padding: 10px; background-color: #d1ecf1; border-radius: 5px; color: #0c5460;"><b>Prediction:</b> Draw ({winner_prob:.0%})</div>', unsafe_allow_html=True)

            if show_scores:
                st.caption(format_score_summary(prediction))
            
            st.divider()
        
    except Exception as e:
        st.error(f"Error: {str(e)}")
