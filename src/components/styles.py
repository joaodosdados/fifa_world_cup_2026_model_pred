"""
Shared CSS styles for the application
"""

import streamlit as st

def apply_custom_css():
    """Apply custom CSS styling to the application"""
    st.markdown("""
    <style>
        .main-header {
            font-size: 3rem;
            font-weight: bold;
            text-align: center;
            color: #1f77b4;
            margin-bottom: 2rem;
        }
        .group-header {
            background-color: #1f77b4;
            color: white;
            padding: 1rem;
            border-radius: 10px;
            font-size: 1.5rem;
            font-weight: bold;
            text-align: center;
            margin-bottom: 1rem;
        }
        .match-box {
            background-color: #f8f9fa;
            border: 2px solid #dee2e6;
            border-radius: 8px;
            padding: 1rem;
            margin-bottom: 0.5rem;
            transition: all 0.3s;
        }
        .match-box:hover {
            border-color: #1f77b4;
            box-shadow: 0 4px 8px rgba(0,0,0,0.1);
        }
        .team-row {
            display: flex;
            justify-content: space-between;
            align-items: center;
            padding: 0.5rem;
            border-bottom: 1px solid #dee2e6;
        }
        .team-row:last-child {
            border-bottom: none;
        }
        .team-name {
            font-weight: bold;
            font-size: 1.1rem;
        }
        .prediction-badge {
            background-color: #28a745;
            color: white;
            padding: 0.25rem 0.5rem;
            border-radius: 5px;
            font-size: 0.9rem;
            font-weight: bold;
        }
        .score-display {
            font-size: 1.2rem;
            font-weight: bold;
            color: #1f77b4;
        }
        .probability-bar {
            height: 8px;
            background-color: #e9ecef;
            border-radius: 4px;
            overflow: hidden;
            margin-top: 0.5rem;
        }
        .probability-fill {
            height: 100%;
            background: linear-gradient(90deg, #28a745, #ffc107);
            transition: width 0.3s;
        }
        .stage-badge {
            display: inline-block;
            background-color: #ff7f0e;
            color: white;
            padding: 0.25rem 0.75rem;
            border-radius: 15px;
            font-size: 0.85rem;
            font-weight: bold;
            margin-bottom: 0.5rem;
        }
    </style>
    """, unsafe_allow_html=True)

# Made with Bob
