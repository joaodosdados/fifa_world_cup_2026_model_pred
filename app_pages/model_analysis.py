"""
Model Analysis Page
ELO ratings, statistics, and model performance
"""
import streamlit as st
import plotly.graph_objects as go
import pandas as pd
from src.data.loader import DataLoader

def show_ml_analysis(models):
    """Display ML model analysis"""
    st.markdown("## ML Model Analysis (SVM)")
    
    ml_model = models.get('ml_model')
    ml_metrics = models.get('ml_metrics', {})
    
    # Model Performance
    st.markdown("### 📊 Model Performance")
    col1, col2, col3 = st.columns(3)
    
    with col1:
        st.metric("Training Accuracy", f"{ml_metrics.get('accuracy', 0):.1%}")
    with col2:
        st.metric("Cross-Validation", f"{ml_metrics.get('cv_mean', 0):.1%}")
    with col3:
        st.metric("CV Std Dev", f"{ml_metrics.get('cv_std', 0):.3f}")
    
    st.markdown("---")
    
    # Feature Importance (if available)
    st.markdown("### 🎯 Model Features")
    st.info("""
    **Features utilizadas pelo modelo SVM:**
    - Goals Scored (Home/Away)
    - Goals Conceded (Home/Away)
    - Win Rate (Home/Away)
    - Home Advantage Factor
    
    O modelo foi treinado em dados históricos de Copas do Mundo.
    """)
    
    # Training Data
    st.markdown("### 📚 Training Data")
    try:
        loader = DataLoader()
        matches_df = loader.load_matches(processed=False)
        
        col1, col2, col3 = st.columns(3)
        with col1:
            st.metric("Total Matches", f"{len(matches_df):,}")
        with col2:
            if 'Year' in matches_df.columns:
                st.metric("Period", f"{int(matches_df['Year'].min())}-{int(matches_df['Year'].max())}")
        with col3:
            if 'Home Team Name' in matches_df.columns:
                unique_teams = pd.concat([matches_df['Home Team Name'], matches_df['Away Team Name']]).nunique()
                st.metric("Teams", unique_teams)
        
        st.markdown("#### Sample Data")
        display_cols = ['Year', 'Home Team Name', 'Away Team Name', 'Home Team Goals', 'Away Team Goals']
        if all(col in matches_df.columns for col in display_cols):
            st.dataframe(matches_df[display_cols].head(10), use_container_width=True)
        else:
            st.dataframe(matches_df.head(10), use_container_width=True)
        
    except Exception as e:
        st.error(f"Error loading training data: {e}")

def show_elo_analysis(models):
    """Display ELO model analysis"""
    st.markdown("## ELO Model Analysis")
    
    analysis_model = models.get("elo_model")
    
    if analysis_model is None:
        st.error("ELO model not available")
        return
    
    tab1, tab2, tab3, tab4 = st.tabs(["ELO Ratings", "Team Statistics", "Complete Rankings", "Training Data"])
    
    with tab1:
        ratings_df = analysis_model.elo_model.get_all_ratings()
        # Filter out None, nan, and invalid teams
        ratings_df = ratings_df[
            (ratings_df['team'] != 'nan') &
            (ratings_df['team'].notna()) &
            (ratings_df['team'] != 'None') &
            (ratings_df['team'] != None)
        ].head(20)
        
        st.markdown("### Top 20 Teams by ELO Rating")
        st.dataframe(ratings_df, use_container_width=True, height=600)
    
    with tab2:
        team_stats = []
        for team, stats in analysis_model.poisson_model.team_stats.items():
            # Filter out None, nan, and invalid teams
            if team and team != 'nan' and team != 'None' and str(team).lower() != 'none':
                team_stats.append({
                    'Team': team,
                    'Attack': stats['attack_strength'],
                    'Defense': stats['defense_strength'],
                    'Matches': stats['matches_played']
                })
        
        stats_df = pd.DataFrame(team_stats).sort_values('Attack', ascending=False)
        st.markdown("### Team Attack & Defense")
        st.dataframe(stats_df, use_container_width=True, height=600)
    
    with tab3:
        ratings_df = analysis_model.elo_model.get_all_ratings()
        # Filter out None, nan, and invalid teams
        ratings_df = ratings_df[
            (ratings_df['team'] != 'nan') &
            (ratings_df['team'].notna()) &
            (ratings_df['team'] != 'None') &
            (ratings_df['team'] != None)
        ].reset_index(drop=True)
        ratings_df.index = ratings_df.index + 1
        
        st.markdown("### Complete Team Rankings")
        st.caption("All teams ranked by ELO rating")
        st.dataframe(
            ratings_df.rename(columns={'team': 'Team', 'elo_rating': 'ELO Rating'}),
            use_container_width=True,
            height=600
        )
    
    with tab4:
        st.markdown("### Historical Training Data")
        st.caption("Sample of World Cup matches used to train the ELO and Poisson models")
        
        try:
            loader = DataLoader()
            matches_df = loader.load_matches(processed=False)
            
            col1, col2, col3 = st.columns(3)
            with col1:
                st.metric("Total Matches", f"{len(matches_df):,}")
            with col2:
                if 'Year' in matches_df.columns:
                    st.metric("Period", f"{int(matches_df['Year'].min())}-{int(matches_df['Year'].max())}")
            with col3:
                if 'Home Team Name' in matches_df.columns:
                    unique_teams = pd.concat([matches_df['Home Team Name'], matches_df['Away Team Name']]).nunique()
                    st.metric("Teams", unique_teams)
            
            st.markdown("#### Sample Data")
            display_cols = ['Year', 'Home Team Name', 'Away Team Name', 'Home Team Goals', 'Away Team Goals']
            if all(col in matches_df.columns for col in display_cols):
                st.dataframe(matches_df[display_cols].head(10), use_container_width=True)
            else:
                st.dataframe(matches_df.head(10), use_container_width=True)
            
        except Exception as e:
            st.error(f"Error loading training data: {e}")

def show_model_analysis(ensemble):
    """Display model analysis based on active model"""
    
    # Get active model info
    active_model = st.session_state.get('active_model', 'ELO')
    models = st.session_state.get("models", {})
    
    # Show appropriate analysis
    if active_model == 'ML (SVM)':
        show_ml_analysis(models)
    else:
        show_elo_analysis(models)

# Execute page
predictor = st.session_state.get("predictor")

if predictor:
    show_model_analysis(predictor)
else:
    st.error("Failed to load predictor")
