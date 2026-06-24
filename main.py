"""
World Cup 2026 Predictor - Main Application
Central router using native Streamlit multipage architecture
"""

import streamlit as st
import sys
from pathlib import Path
from datetime import datetime

# Add project root to path
sys.path.insert(0, str(Path(__file__).parent))

from src.data.loader import DataLoader
from src.models.model_catalog import (
    build_model_catalog,
    default_model_id,
    rank_model_ids,
)
from src.models.schedule_predictions import (
    precompute_schedule_predictions,
    schedule_fingerprint,
)
from src.app.components.styles import apply_custom_css
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

def update_schedule():
    """Fetch FIFA results without hiding failures behind cached data."""
    from src.data.auto_updater import run_auto_update

    return run_auto_update()

def load_2026_schedule():
    """Load 2026 World Cup schedule (after auto-update)"""
    return pd.read_csv('data/2026_world_cup_schedule.csv')

# 3. Initialize models and data (cached)
@st.cache_resource
def load_runtime(schedule_version: str):
    """
    Load sanitized pre-2026 training data and build all predictors.

    The schedule_version argument invalidates the cache when FIFA updates
    change the local schedule CSV. Completed 2026 matches are intentionally not
    appended here, otherwise live accuracy on those same matches would leak the
    answer into the model.
    """
    loader = DataLoader()
    try:
        training_df = loader.load_international_matches(
            min_year=2010,
            max_date="2026-06-10",
        )
    except FileNotFoundError:
        training_df = loader.load_matches(processed=False)
    return {
        "historical_matches": training_df,
        "training_matches": training_df,
        "catalog": build_model_catalog(training_df),
    }

# Every new Streamlit session performs a fresh FIFA update. Widget reruns do not
# launch another browser; users can explicitly refresh with the sidebar button.
if "initial_fifa_update_done" not in st.session_state:
    st.session_state.update_result = update_schedule()
    st.session_state.initial_fifa_update_done = True

st.session_state.schedule_2026 = load_2026_schedule()
st.session_state.schedule_version = schedule_fingerprint(
    st.session_state.schedule_2026
)

# 4. Initialize global session state
runtime = load_runtime(st.session_state.schedule_version)
catalog = runtime["catalog"]
st.session_state.training_matches = runtime["training_matches"]

if "active_model_id" not in st.session_state:
    st.session_state.active_model_id = default_model_id(catalog)

if st.session_state.active_model_id not in catalog:
    st.session_state.active_model_id = default_model_id(catalog)

active_model = catalog[st.session_state.active_model_id]
st.session_state.active_model = active_model
st.session_state.predictor = active_model["predictor"]


def get_schedule_predictions(model_id: str) -> pd.DataFrame:
    """Return all fixture predictions for a model, cached for fast page reruns."""
    cache = st.session_state.setdefault("schedule_prediction_cache", {})
    cache_key = (model_id, st.session_state.schedule_version)
    if cache_key not in cache:
        cache[cache_key] = precompute_schedule_predictions(
            catalog[model_id]["predictor"],
            st.session_state.schedule_2026,
        )
    return cache[cache_key]

# 5. Calculate accuracy on completed 2026 matches
def calculate_accuracy(model_id=None):
    """Calculate model accuracy on completed 2026 World Cup matches"""
    schedule = st.session_state.schedule_2026
    model_id = model_id or st.session_state.active_model_id
    predictions = get_schedule_predictions(model_id)
    
    # Check for both 'Completed' and 'completed' status
    completed = schedule[schedule['status'].str.lower() == 'completed'].copy()
    if len(completed) == 0:
        return None
    
    correct = 0
    for _, match in completed.iterrows():
        match_id = match.get("match_id")
        if match_id not in predictions.index:
            continue
        
        # Determine actual winner
        actual_winner = None
        if match['home_score'] > match['away_score']:
            actual_winner = match["home_team"]
        elif match['away_score'] > match['home_score']:
            actual_winner = match["away_team"]
        else:
            actual_winner = 'Draw'

        predicted_row = predictions.loc[match_id]
        if isinstance(predicted_row, pd.DataFrame):
            predicted_row = predicted_row.iloc[0]
        predicted_winner = predicted_row.get("predicted_winner")
        
        if predicted_winner == actual_winner:
            correct += 1
    
    return {
        'correct': correct,
        'incorrect': len(completed) - correct,
        'total': len(completed),
        'accuracy': (correct / len(completed)) * 100
    }

# 6. Define pages programmatically
page_bracket = st.Page(
    "src/app/pages/tournament_bracket.py",
    title="Tournament Bracket",
    icon="🏆",
    default=True
)

page_groups = st.Page(
    "src/app/pages/group_stage.py",
    title="Group Stage Details",
    icon="📊"
)

page_simulation = st.Page(
    "src/app/pages/tournament_simulation.py",
    title="Tournament Simulation",
    icon="🎲"
)

page_analysis = st.Page(
    "src/app/pages/model_analysis.py",
    title="Model Analysis",
    icon="📈"
)

page_about = st.Page(
    "src/app/pages/about.py",
    title="About",
    icon="ℹ️"
)

# 7. Organize pages into sections
pages_dict = {
    "Tournament": [page_bracket, page_groups, page_simulation],
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
    st.markdown("### 🤖 Modelo")

    model_accuracy = {}
    model_accuracy_stats = {}
    for model_id, model in catalog.items():
        stats = calculate_accuracy(model_id)
        model_accuracy_stats[model_id] = stats
        model_accuracy[model_id] = stats["accuracy"] if stats else -1.0

    model_ids = rank_model_ids(catalog, model_accuracy)
    model_rank = {
        model_id: position
        for position, model_id in enumerate(model_ids, start=1)
    }

    # On a new session, start with the best model under the same live ranking
    # displayed to the user. Subsequent selections remain untouched.
    if "model_rank_initialized" not in st.session_state:
        st.session_state.active_model_id = model_ids[0]
        st.session_state.model_rank_initialized = True
    if st.session_state.active_model_id not in model_ids:
        st.session_state.active_model_id = model_ids[0]

    selected_model_id = st.selectbox(
        "Modelo de previsão",
        options=model_ids,
        index=model_ids.index(st.session_state.active_model_id),
        format_func=lambda model_id: (
            f"#{model_rank[model_id]} · {catalog[model_id]['label']} · "
            f"{model_accuracy[model_id]:.1f}%"
        ),
        label_visibility="collapsed",
        key="active_model_id",
    )

    active_model = catalog[selected_model_id]
    st.session_state.active_model = active_model
    st.session_state.predictor = active_model["predictor"]
    st.session_state.schedule_predictions = get_schedule_predictions(
        selected_model_id
    )
    accuracy_stats = model_accuracy_stats[selected_model_id]
    st.session_state.accuracy_stats = accuracy_stats

    if accuracy_stats:
        metric_col, result_col = st.columns([1.15, 1])
        metric_col.metric("Acurácia 2026", f"{accuracy_stats['accuracy']:.1f}%")
        result_col.metric(
            "Acertos",
            f"{accuracy_stats['correct']}/{accuracy_stats['total']}",
        )

    st.caption(active_model["description"])
    st.markdown("---")

    update_col, status_col = st.columns([1.15, 1])
    refresh_clicked = update_col.button(
        "↻ Atualizar FIFA",
        width="stretch",
        help="Busca novamente os resultados na página oficial da FIFA.",
    )

    update_result = st.session_state.get("update_result", {})
    status_col.caption(
        f"{update_result.get('matched_matches', 0)} jogos conferidos"
        if update_result.get("success")
        else "FIFA indisponível"
    )

    if refresh_clicked:
        with st.spinner("Atualizando..."):
            st.session_state.update_result = update_schedule()
        st.session_state.schedule_2026 = load_2026_schedule()
        st.session_state.schedule_version = schedule_fingerprint(
            st.session_state.schedule_2026
        )
        st.session_state.schedule_prediction_cache = {}
        st.session_state.accuracy_stats = calculate_accuracy(
            st.session_state.active_model_id
        )
        st.rerun()

    update_result = st.session_state.get("update_result", {})
    last_update = update_result.get("last_update")
    if isinstance(last_update, datetime):
        st.caption(
            f"Última consulta {last_update:%d/%m %H:%M} · "
            f"{update_result.get('fetched_matches', 0)} finalizados"
        )
    elif update_result.get("message"):
        st.caption(update_result["message"])

# 10. Execute selected page
current_page.run()

# Made with Bob
