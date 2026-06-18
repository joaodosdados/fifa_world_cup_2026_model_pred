"""
Model Comparison Page - Compare and switch between trained models
"""

import streamlit as st
import pandas as pd
import plotly.graph_objects as go
import plotly.express as px
from pathlib import Path
from src.models.sklearn_adapter import SklearnMatchPredictor
from src.data.loader import DataLoader

st.title("🔬 Model Comparison & Selection")

# Add refresh button at the top
col1, col2 = st.columns([4, 1])
with col2:
    if st.button("🔄 Refresh Models", help="Reload models from disk"):
        # Clear cache to force reload
        st.cache_resource.clear()
        st.rerun()

# Get model manager from session state
if 'predictor' not in st.session_state:
    st.error("⚠️ Predictor not initialized. Please restart the application.")
    st.stop()

predictor_info = st.session_state.get('predictor_info', {})
manager = predictor_info.get('manager')

if manager is None:
    st.error("⚠️ Model Manager not available.")
    st.stop()

# Reload metadata to get latest models
manager.metadata = manager._load_metadata()

# Get list of available models
models_list = manager.list_models()

if not models_list:
    st.warning("📭 No trained models found. Please run the training notebook first.")
    st.info("💡 Run `notebooks/model_training_analysis.ipynb` to train and save models.")
    st.stop()

# === SECTION 1: Current Active Model ===
st.markdown("## 🎯 Currently Active Model")

current_model_name = predictor_info.get('name', 'Unknown')
current_model_type = predictor_info.get('type', 'Unknown')

col1, col2, col3 = st.columns([2, 1, 1])

with col1:
    st.metric("Active Model", current_model_name)

with col2:
    st.metric("Type", current_model_type.upper())

with col3:
    if current_model_type == 'sklearn':
        current_metrics = manager.metadata.get(current_model_name, {}).get('metrics', {})
        st.metric("Accuracy", f"{current_metrics.get('accuracy', 0):.1%}")

st.markdown("---")

# === SECTION 2: Models Comparison Table ===
st.markdown("## 📊 All Available Models")

# Create comparison DataFrame
comparison_df = manager.compare_models()

if not comparison_df.empty:
    # Format metrics as percentages
    for col in ['accuracy', 'cv_mean', 'cv_std']:
        if col in comparison_df.columns:
            comparison_df[col] = comparison_df[col].apply(lambda x: f"{x:.1%}")
    
    # Highlight active model
    def highlight_active(row):
        if row['Active'] == '✓':
            return ['background-color: #90EE90'] * len(row)
        return [''] * len(row)
    
    styled_df = comparison_df.style.apply(highlight_active, axis=1)
    st.dataframe(styled_df, use_container_width=True, hide_index=True)
else:
    st.info("No models to compare.")

st.markdown("---")

# === SECTION 3: Visual Comparison ===
st.markdown("## 📈 Performance Visualization")

if models_list:
    # Prepare data for plotting
    model_names = [m['name'] for m in models_list]
    accuracies = [m['metrics'].get('accuracy', 0) for m in models_list]
    cv_scores = [m['metrics'].get('cv_mean', 0) for m in models_list]
    
    # Create bar chart
    fig = go.Figure()
    
    fig.add_trace(go.Bar(
        name='Test Accuracy',
        x=model_names,
        y=accuracies,
        marker_color='steelblue',
        text=[f"{acc:.1%}" for acc in accuracies],
        textposition='outside'
    ))
    
    fig.add_trace(go.Bar(
        name='CV Score',
        x=model_names,
        y=cv_scores,
        marker_color='coral',
        text=[f"{cv:.1%}" for cv in cv_scores],
        textposition='outside'
    ))
    
    fig.update_layout(
        title="Model Performance Comparison",
        xaxis_title="Model",
        yaxis_title="Score",
        yaxis_tickformat='.0%',
        barmode='group',
        height=500,
        hovermode='x unified'
    )
    
    st.plotly_chart(fig, use_container_width=True)

st.markdown("---")

# === SECTION 4: Model Selection ===
st.markdown("## 🔄 Switch Active Model")

col1, col2 = st.columns([3, 1])

with col1:
    # Create selectbox with model names
    model_options = [m['name'] for m in models_list]
    current_index = model_options.index(current_model_name) if current_model_name in model_options else 0
    
    selected_model = st.selectbox(
        "Select a model to activate:",
        options=model_options,
        index=current_index,
        help="Choose which model to use for predictions"
    )
    
    # Show selected model details
    selected_info = next((m for m in models_list if m['name'] == selected_model), None)
    if selected_info:
        st.info(f"""
        **Description:** {selected_info['description']}
        
        **Metrics:**
        - Accuracy: {selected_info['metrics'].get('accuracy', 0):.1%}
        - CV Score: {selected_info['metrics'].get('cv_mean', 0):.1%} (±{selected_info['metrics'].get('cv_std', 0):.1%})
        
        **Saved:** {selected_info['saved_at'][:10]}
        """)

with col2:
    st.markdown("<br>", unsafe_allow_html=True)
    if st.button("🔄 Activate Model", type="primary", use_container_width=True):
        if selected_model != current_model_name:
            # Load and activate the selected model
            sklearn_model = manager.load_model(selected_model)
            if sklearn_model is not None:
                manager.set_active_model(selected_model)
                
                # Carregar dados históricos para o adaptador
                loader = DataLoader()
                matches_df = loader.load_matches(processed=False)
                
                # Envolver modelo sklearn no adaptador
                adapted_model = SklearnMatchPredictor(sklearn_model, matches_df)
                
                # Update session state
                new_predictor_info = {
                    'type': 'sklearn',
                    'model': adapted_model,
                    'name': selected_model,
                    'manager': manager
                }
                st.session_state.predictor = adapted_model  # Manter compatibilidade
                st.session_state.predictor_info = new_predictor_info
                
                # Recalculate accuracy with new model
                st.session_state.accuracy_stats = None  # Will be recalculated
                
                st.success(f"✅ Model '{selected_model}' is now active!")
                st.balloons()
                st.rerun()
            else:
                st.error(f"❌ Failed to load model '{selected_model}'")
        else:
            st.info("ℹ️ This model is already active.")

st.markdown("---")

# === SECTION 5: Model Details ===
with st.expander("📋 Detailed Model Information"):
    for model_info in models_list:
        st.markdown(f"### {model_info['name']}")
        
        col1, col2, col3 = st.columns(3)
        
        with col1:
            st.metric("Type", model_info['type'].upper())
        
        with col2:
            st.metric("Accuracy", f"{model_info['metrics'].get('accuracy', 0):.1%}")
        
        with col3:
            st.metric("CV Score", f"{model_info['metrics'].get('cv_mean', 0):.1%}")
        
        st.caption(f"**Description:** {model_info['description']}")
        st.caption(f"**Saved:** {model_info['saved_at']}")
        
        if model_info['active']:
            st.success("✓ Currently Active")
        
        st.markdown("---")

# === SECTION 6: Tips ===
st.markdown("## 💡 Tips")

st.info("""
**How to train new models:**
1. Open `notebooks/model_training_analysis.ipynb`
2. Run all cells to train multiple ML models
3. Models will be automatically saved and available here
4. The best model will be set as active by default

**Model Selection:**
- Higher accuracy = better predictions on test data
- CV Score shows model consistency across different data splits
- Consider both metrics when choosing a model
""")

# Made with Bob
