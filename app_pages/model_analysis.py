"""
Model Analysis Page
ELO ratings, statistics, and model performance
"""
import streamlit as st
import plotly.graph_objects as go
import pandas as pd
from src.data.loader import DataLoader

# Access global state
ensemble = st.session_state.get("predictor")

def show_model_analysis(ensemble):
    """Display model analysis"""
    
    st.markdown("## Model Analysis")
    
    tab1, tab2, tab3, tab4 = st.tabs(["ELO Ratings", "Team Statistics", "Complete Rankings", "Training Data"])
    
    with tab1:
        ratings_df = ensemble.elo_model.get_all_ratings()
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
        for team, stats in ensemble.poisson_model.team_stats.items():
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
        ratings_df = ensemble.elo_model.get_all_ratings()
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
        st.caption("Sample of World Cup matches used to train the prediction models")
        
        try:
            loader = DataLoader()
            matches_df = loader.load_matches(processed=False)
            
            # Show statistics
            col1, col2, col3, col4 = st.columns(4)
            
            with col1:
                st.metric("Total Matches", f"{len(matches_df):,}")
            
            with col2:
                if 'Year' in matches_df.columns:
                    st.metric("Period", f"{int(matches_df['Year'].min())}-{int(matches_df['Year'].max())}")
                else:
                    st.metric("Period", "N/A")
            
            with col3:
                unique_teams = set()
                if 'Home Team Name' in matches_df.columns:
                    unique_teams.update(matches_df['Home Team Name'].unique())
                if 'Away Team Name' in matches_df.columns:
                    unique_teams.update(matches_df['Away Team Name'].unique())
                st.metric("Teams", len(unique_teams))
            
            with col4:
                if 'Home Team Goals' in matches_df.columns and 'Away Team Goals' in matches_df.columns:
                    total_goals = matches_df['Home Team Goals'].sum() + matches_df['Away Team Goals'].sum()
                    st.metric("Total Goals", f"{int(total_goals):,}")
                else:
                    st.metric("Total Goals", "N/A")
            
            st.markdown("---")
            
            # Show sample data
            st.markdown("#### Sample Matches")
            
            # Select relevant columns
            display_cols = []
            if 'Year' in matches_df.columns:
                display_cols.append('Year')
            if 'Stage' in matches_df.columns:
                display_cols.append('Stage')
            if 'Home Team Name' in matches_df.columns:
                display_cols.append('Home Team Name')
            if 'Home Team Goals' in matches_df.columns:
                display_cols.append('Home Team Goals')
            if 'Away Team Goals' in matches_df.columns:
                display_cols.append('Away Team Goals')
            if 'Away Team Name' in matches_df.columns:
                display_cols.append('Away Team Name')
            if 'Attendance' in matches_df.columns:
                display_cols.append('Attendance')
            
            if display_cols:
                sample_df = matches_df[display_cols].head(100)
                st.dataframe(sample_df, use_container_width=True, height=400)
                
                # Download button
                csv = matches_df.to_csv(index=False)
                st.download_button(
                    label="📥 Download Complete Training Data (CSV)",
                    data=csv,
                    file_name="world_cup_training_data.csv",
                    mime="text/csv"
                )
            else:
                st.warning("No displayable columns found in training data")
                
        except Exception as e:
            st.error(f"Could not load training data: {e}")
            st.info("Training data should be located in data/raw/WorldCupMatches.csv")

def show_fifa_standings(fifa_standings, ensemble):
    """Display FIFA official standings"""
    
    st.markdown("## FIFA Official Standings")
    
    if not fifa_standings:
        st.warning("No FIFA standings data available. Using fallback data.")
        st.info("The FIFA website uses dynamic JavaScript content. Real-time data fetching may be limited.")
        
        # Show fallback data
        scraper = FIFADataScraper()
        fifa_standings = scraper._get_fallback_standings()
    
    st.markdown("### Group Stage Standings")
    st.caption("Live data from FIFA.com (updated every 5 minutes)")
    
    # Display standings in columns
    groups = sorted(fifa_standings.keys())
    cols_per_row = 3
    
    for i in range(0, len(groups), cols_per_row):
        cols = st.columns(cols_per_row)
        
        for j, col in enumerate(cols):
            if i + j < len(groups):
                group = groups[i + j]
                with col:
                    display_group_standings(fifa_standings[group], group, ensemble)
    
    # Knockout bracket visualization
    st.markdown("---")
    st.markdown("### Knockout Stage Bracket")
    st.info("Knockout bracket will be generated after group stage completion")
    
    # Show qualified teams if available
    qualified_teams = []
    for group, standings in fifa_standings.items():
        if len(standings) >= 2:
            # Top 2 teams qualify
            top_teams = standings.head(2)
            for _, team in top_teams.iterrows():
                qualified_teams.append({
                    'Group': group,
                    'Team': team['team'],
                    'Points': team['points'],
                    'Position': 1 if _ == 0 else 2
                })
    
    if qualified_teams:
        st.markdown("#### Qualified Teams")
        qualified_df = pd.DataFrame(qualified_teams)
        
        col1, col2 = st.columns(2)
        
        with col1:
            st.markdown("**Group Winners**")
            winners = qualified_df[qualified_df['Position'] == 1].sort_values('Group')
            for _, team in winners.iterrows():
                flag = get_flag_html(team['Team'], size='w40', height=20)
                st.markdown(f"**Group {team['Group']}:** {flag}{team['Team']} ({team['Points']} pts)", unsafe_allow_html=True)
        
        with col2:
            st.markdown("**Runners-up**")
            runners = qualified_df[qualified_df['Position'] == 2].sort_values('Group')
            for _, team in runners.iterrows():
                flag = get_flag_html(team['Team'], size='w40', height=20)
                st.markdown(f"**Group {team['Group']}:** {flag}{team['Team']} ({team['Points']} pts)", unsafe_allow_html=True)

def display_group_standings(standings_df, group, ensemble):
    """Display standings for a specific group"""
    
    st.markdown(f"#### Group {group}")
    
    # Legend for status indicators
    st.caption("Q = Qualified | P = Possible 3rd place | E = Eliminated")
    
    # Create standings table
    for idx, row in standings_df.iterrows():
        team = row['team']
        flag = get_flag_html(team, size='w40', height=20)
        points = row['points']
        played = row['played']
        goal_diff = row['goal_diff']
        
        # Position indicator
        position = idx + 1
        if position <= 2:
            pos_status = "Q"  # Qualified
            status_color = "🟢"
        elif position == 3:
            pos_status = "P"  # Possible 3rd place
            status_color = "🟡"
        else:
            pos_status = "E"  # Eliminated
            status_color = "🔴"
        
        # Display team row
        col1, col2, col3 = st.columns([1, 4, 2])
        
        with col1:
            st.markdown(f"**{position}. {pos_status}**")
        
        with col2:
            st.markdown(f"{flag}**{team}**", unsafe_allow_html=True)
        
        with col3:
            st.markdown(f"**{points}** pts ({played}J, {goal_diff:+d}GD)")
    
    st.divider()


# Execute page
if ensemble:
    show_model_analysis(ensemble)
else:
    st.error("Failed to load predictor")
