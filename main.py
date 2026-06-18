"""
World Cup 2026 Predictor - Main Application
Central router using native Streamlit multipage architecture
"""

import streamlit as st
import sys
from pathlib import Path

# Add project root to path
sys.path.insert(0, str(Path(__file__).parent))

from src.models.ensemble_predictor import EnsemblePredictor
from src.data.loader import DataLoader
from src.data.fifa_scraper import FIFADataScraper
from src.components.styles import apply_custom_css
import pandas as pd

# 1. MUST be first Streamlit command
st.set_page_config(
    page_title="World Cup 2026 Predictor",
    page_icon="⚽",
    layout="wide",
    initial_sidebar_state="expanded"
)

# 2. Apply custom CSS
apply_custom_css()

# 3. Initialize models and data (cached)
@st.cache_resource
def load_models():
    """Load ML models once and persist across reruns"""
    from src.data.loader import DataLoader
    
    ensemble = EnsemblePredictor()
    
    # Train models on historical data
    try:
        loader = DataLoader()
        matches_df = loader.load_matches(processed=False)
        ensemble.train(matches_df)
    except Exception as e:
        st.warning(f"Could not train models: {e}")
    
    return ensemble

@st.cache_data(ttl=300)
def load_2026_schedule():
    """Load 2026 World Cup schedule"""
    return pd.read_csv('data/2026_world_cup_schedule.csv')

@st.cache_data(ttl=300)
def fetch_fifa_data():
    """Fetch FIFA data with fallback"""
    scraper = FIFADataScraper()
    return scraper.fetch_group_standings()

# 4. Initialize global session state
if 'predictor' not in st.session_state:
    st.session_state.predictor = load_models()

if 'schedule_2026' not in st.session_state:
    st.session_state.schedule_2026 = load_2026_schedule()

if 'fifa_data' not in st.session_state:
    st.session_state.fifa_data = fetch_fifa_data()

if 'show_probabilities' not in st.session_state:
    st.session_state.show_probabilities = True

if 'show_scores' not in st.session_state:
    st.session_state.show_scores = True

# 5. Calculate accuracy on completed 2026 matches
def calculate_accuracy():
    """Calculate model accuracy on completed 2026 World Cup matches"""
    schedule = st.session_state.schedule_2026
    predictor = st.session_state.predictor
    
    # Check for both 'Completed' and 'completed' status
    completed = schedule[schedule['status'].str.lower() == 'completed'].copy()
    if len(completed) == 0:
        return None
    
    correct = 0
    for _, match in completed.iterrows():
        prediction = predictor.predict_match(match['home_team'], match['away_team'])
        
        # Determine actual winner
        actual_winner = None
        if match['home_score'] > match['away_score']:
            actual_winner = 'home'
        elif match['away_score'] > match['home_score']:
            actual_winner = 'away'
        else:
            actual_winner = 'draw'
        
        # Determine predicted winner based on highest probability
        predicted_winner = None
        if prediction['home_win'] > prediction['draw'] and prediction['home_win'] > prediction['away_win']:
            predicted_winner = 'home'
        elif prediction['away_win'] > prediction['draw'] and prediction['away_win'] > prediction['home_win']:
            predicted_winner = 'away'
        else:
            predicted_winner = 'draw'
        
        if predicted_winner == actual_winner:
            correct += 1
    
    return {
        'correct': correct,
        'incorrect': len(completed) - correct,
        'total': len(completed),
        'accuracy': (correct / len(completed)) * 100
    }

if 'accuracy_stats' not in st.session_state:
    st.session_state.accuracy_stats = calculate_accuracy()

# 6. Define pages programmatically
page_bracket = st.Page(
    "app_pages/tournament_bracket.py",
    title="Tournament Bracket",
    icon="🏆",
    default=True
)

page_groups = st.Page(
    "app_pages/group_stage.py",
    title="Group Stage Details",
    icon="📊"
)

page_analysis = st.Page(
    "app_pages/model_analysis.py",
    title="Model Analysis",
    icon="📈"
)

page_about = st.Page(
    "app_pages/about.py",
    title="About",
    icon="ℹ️"
)

# 7. Organize pages into sections
pages_dict = {
    "Tournament": [page_bracket, page_groups],
    "Analysis": [page_analysis],
    "Information": [page_about]
}

# 8. Create navigation
current_page = st.navigation(pages_dict)

# 8.5. Add centered title
st.markdown("""
<h1 style='text-align: center; color: #1f77b4; margin-bottom: 2rem;'>
    ⚽ FIFA World Cup 2026 Prediction Dashboard
</h1>
""", unsafe_allow_html=True)

# 9. Global sidebar elements (persist across all pages)
with st.sidebar:
    # Model accuracy on 2026 predictions (only if matches completed)
    if st.session_state.accuracy_stats:
        st.markdown("### 🎯 Model Accuracy")
        
        accuracy_stats = st.session_state.accuracy_stats
        st.metric(
            "Prediction Accuracy",
            f"{accuracy_stats['accuracy']:.1f}%",
            delta=f"{accuracy_stats['correct']}/{accuracy_stats['total']} correct"
        )
        
        col1, col2 = st.columns(2)
        col1.metric("✓ Correct", accuracy_stats['correct'])
        col2.metric("✗ Incorrect", accuracy_stats['incorrect'])
        
        st.markdown("---")
    
    st.markdown("### ℹ️ About")
    st.markdown("""
    This dashboard uses machine learning to predict World Cup 2026 match outcomes.
    
    **Models:**
    - ELO Rating System
    - Poisson Distribution
    - Ensemble Predictor
    """)

# 10. Execute selected page
current_page.run()

# Made with Bob
