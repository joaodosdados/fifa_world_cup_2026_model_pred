"""
About Page
Information about the World Cup 2026 Prediction System
"""

import streamlit as st

# Access global state
schedule = st.session_state.get("schedule_2026")

# Page content
st.markdown("## About World Cup 2026 Prediction System")

# About the Developer - First and always visible
st.markdown("### About the Developer")
col1, col2 = st.columns([1, 2])

with col1:
    try:
        st.image("assets/profile.png", width=200)
    except:
        st.info("Profile image not found")

with col2:
    st.markdown("""
    **João dos Dados**
    
    Data-driven professional with over 6 years of experience building scalable machine learning and AI solutions across media, mining, retail, telecommunications, and enterprise environments.
    
    **Contact:**
    - Email: lucas.xms@gmail.com
    - GitHub: [github.com/joaodosdados](https://github.com/joaodosdados)
    - LinkedIn: [linkedin.com/in/joaodosdados](https://www.linkedin.com/in/joaodosdados/)
    """)

st.markdown("---")

# Tournament Format
with st.expander("Tournament Format", expanded=True):
    col1, col2 = st.columns(2)
    
    with col1:
        st.markdown("""
        **Tournament Structure:**
        - **Group Stage**: 12 groups of 4 teams
        - **Octave-finals**: Top 2 from each group + best 3rd place teams
        - **Quarter-Finals**: 8 teams
        - **Semi-Finals**: 4 teams
        - **Final**: Championship match
        
        **Dates:**
        - Opening Match: June 11, 2026
        - Group Stage: June 11-27, 2026
        - Knockout Stage: June 30 - July 19, 2026
        """)
    
    with col2:
        st.markdown("""
        **Venues:**
        - United States: 11 cities
        - Canada: 2 cities
        - Mexico: 3 cities
        
        **Key Stadiums:**
        - MetLife Stadium (Final)
        - AT&T Stadium
        - SoFi Stadium
        - Estadio Azteca
        - And more...
        """)

# Prediction Model
with st.expander("Prediction Model", expanded=False):
    st.markdown("""
    **Ensemble Model**
    
    Our prediction system combines two powerful models:
    
    1. **ELO Rating System (40%)**
       - Dynamic team strength calculation
       - K-factor: 32
       - Home advantage: 100 points
       - Updated after each match
    
    2. **Poisson Distribution Model (60%)**
       - Statistical goal prediction
       - Attack strength analysis
       - Defense strength analysis
       - Historical performance
    
    **Training Data:**
    - 4,572 historical World Cup matches (1930-2014)
    - 84 teams rated
    - Continuous learning from new results
    
    **Accuracy:**
    - Real-time calculation based on completed matches
    - Comparison with actual FIFA results
    - Visual indicators (CORRECT/INCORRECT) for each prediction
    """)

# Data Sources & Credits
with st.expander("Data Sources & Credits", expanded=False):
    st.markdown("""
    **Historical Data:**
    - [Kaggle: FIFA World Cup Dataset](https://www.kaggle.com/datasets/abecklas/fifa-world-cup) (1930-2014)
    - 4,572 matches with detailed statistics
    
    **Live Data:**
    - [FIFA.com Official Website](https://www.fifa.com/fifaplus/en/tournaments/mens/worldcup/canadamexicousa2026)
    - Real-time standings and results
    - Group stage matches and scores
    
    **Technologies & Libraries:**
    - [Streamlit](https://streamlit.io/) - Dashboard framework
    - [Pandas](https://pandas.pydata.org/) - Data manipulation
    - [NumPy](https://numpy.org/) - Numerical computing
    - [Plotly](https://plotly.com/) - Interactive visualizations
    - [Requests](https://requests.readthedocs.io/) - HTTP library
    - [BeautifulSoup4](https://www.crummy.com/software/BeautifulSoup/) - Web scraping
    
    **Flag Images:**
    - [FlagCDN](https://flagcdn.com/) - Country flag API
    """)

# How to Use
with st.expander("How to Use", expanded=False):
    st.markdown("""
    **Navigation:**
    1. **Tournament Bracket**: Overview of all stages
    2. **Group Stage Details**: Detailed analysis by group
    3. **Model Analysis**: ELO ratings and statistics
    
    **Features:**
    - Toggle win probabilities on/off
    - Toggle predicted scores on/off
    - View real-time model accuracy
    - Compare predictions vs actual results
    
    **Tips:**
    - Use the group selector to focus on specific groups
    - Check FIFA Standings for official data
    - Predictions update automatically as matches complete
    """)

# Made with Bob
