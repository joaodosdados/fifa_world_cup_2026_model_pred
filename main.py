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
from src.models.model_manager import ModelManager, ModelSelector
from src.models.sklearn_adapter import SklearnMatchPredictor
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
    """Load dual models: ML (primary) + ELO (statistical)"""
    from src.data.loader import DataLoader
    
    # Carregar dados históricos
    loader = DataLoader()
    matches_df = loader.load_matches(processed=False)
    
    # Sempre carregar ELO para análises estatísticas
    elo_ensemble = EnsemblePredictor()
    try:
        elo_ensemble.train(matches_df)
    except Exception as e:
        pass  # Silenciar erros
    
    # Inicializar ModelManager
    manager = ModelManager()
    
    # Tentar carregar o melhor modelo ML treinado
    best_model, best_name = manager.get_best_model()
    
    if best_model is not None:
        # Envolver modelo sklearn no adaptador
        ml_model = SklearnMatchPredictor(best_model, matches_df)
        metadata = manager.metadata.get(best_name, {})
        
        return {
            'ml_model': ml_model,
            'ml_name': best_name,
            'ml_metrics': metadata.get('metrics', {}),
            'elo_model': elo_ensemble,
            'primary': 'ml'
        }
    
    # Fallback: usar apenas ELO se não houver modelos ML
    return {
        'ml_model': None,
        'ml_name': None,
        'ml_metrics': {},
        'elo_model': elo_ensemble,
        'primary': 'elo'
    }

@st.cache_data(ttl=300)
def auto_update_schedule():
    """Automatically fetch and update schedule from FIFA"""
    from src.data.auto_updater import run_auto_update
    
    # Run automatic update
    update_result = run_auto_update()
    
    # Show update status in sidebar (if successful)
    if update_result.get('success') and update_result.get('updated_matches', 0) > 0:
        st.toast(f"✓ Updated {update_result['updated_matches']} matches from FIFA!", icon="⚽")
    
    return update_result

@st.cache_data(ttl=300)
def load_2026_schedule():
    """Load 2026 World Cup schedule (after auto-update)"""
    return pd.read_csv('data/2026_world_cup_schedule.csv')

@st.cache_data(ttl=300)
def fetch_fifa_data():
    """Fetch FIFA data with fallback"""
    scraper = FIFADataScraper()
    return scraper.fetch_group_standings()

# 4. Initialize global session state
if 'models' not in st.session_state:
    st.session_state.models = load_models()

# Manter compatibilidade: predictor aponta para o modelo primário
if 'predictor' not in st.session_state:
    models = st.session_state.models
    if models['primary'] == 'ml':
        st.session_state.predictor = models['ml_model']
    else:
        st.session_state.predictor = models['elo_model']

# Run auto-update BEFORE loading schedule
if 'update_result' not in st.session_state:
    st.session_state.update_result = auto_update_schedule()

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
    predictor_info = st.session_state.get('predictor_info', {})
    
    # Check for both 'Completed' and 'completed' status
    completed = schedule[schedule['status'].str.lower() == 'completed'].copy()
    if len(completed) == 0:
        return None
    
    correct = 0
    for _, match in completed.iterrows():
        try:
            # Todos os modelos agora suportam predict_match()
            prediction = predictor.predict_match(match['home_team'], match['away_team'])
        except Exception as e:
            # Se houver erro, pular este jogo
            continue
        
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

# Inicializar seleção de modelo ativo
if 'active_model' not in st.session_state:
    models = st.session_state.models
    st.session_state.active_model = 'ML (SVM)' if models.get('ml_model') else 'ELO'

# Recalcular acurácia quando modelo mudar
if 'last_active_model' not in st.session_state or st.session_state.last_active_model != st.session_state.active_model:
    st.session_state.last_active_model = st.session_state.active_model
    st.session_state.accuracy_stats = calculate_accuracy()
elif 'accuracy_stats' not in st.session_state:
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
    st.markdown("### 🤖 Active Model")
    
    # Model selector
    models = st.session_state.models
    available_models = []
    
    if models.get('ml_model'):
        available_models.append('ML (SVM)')
    if models.get('elo_model'):
        available_models.append('ELO')
    
    selected_model = st.selectbox(
        "Select prediction model:",
        available_models,
        index=available_models.index(st.session_state.active_model) if st.session_state.active_model in available_models else 0,
        key='model_selector'
    )
    
    # Update active model and predictor when selection changes
    if selected_model != st.session_state.active_model:
        st.session_state.active_model = selected_model
        
        # Update predictor
        if selected_model == 'ML (SVM)':
            st.session_state.predictor = models['ml_model']
        else:
            st.session_state.predictor = models['elo_model']
        
        # Force recalculation of accuracy
        st.session_state.accuracy_stats = calculate_accuracy()
        st.rerun()
    
    st.markdown("---")
    
    # Model accuracy on 2026 predictions (only if matches completed)
    if st.session_state.accuracy_stats:
        st.markdown("### 🎯 2026 Predictions")
        
        accuracy_stats = st.session_state.accuracy_stats
        st.metric(
            "Accuracy",
            f"{accuracy_stats['accuracy']:.1f}%",
            delta=f"{accuracy_stats['correct']}/{accuracy_stats['total']} correct"
        )
        
        col1, col2 = st.columns(2)
        col1.metric("✓ Correct", accuracy_stats['correct'])
        col2.metric("✗ Incorrect", accuracy_stats['incorrect'])
        
        st.markdown("---")
    
    st.markdown("### ℹ️ About")
    st.markdown("""
    Dashboard de predição da Copa do Mundo 2026 usando Machine Learning e modelos estatísticos.
    
    **Modelos disponíveis:**
    - **ML (SVM)**: Support Vector Machine treinado em dados históricos
    - **ELO**: Sistema de rating ELO + Distribuição de Poisson
    
    Alterne entre os modelos para comparar predições e análises.
    """)

# 10. Execute selected page
current_page.run()

# Made with Bob
