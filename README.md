# ⚽ FIFA World Cup 2026 Prediction Dashboard

A machine learning-powered prediction system for the FIFA World Cup 2026, featuring an interactive dashboard built with Streamlit's native multipage architecture.

![Python](https://img.shields.io/badge/python-3.8+-blue.svg)
![Streamlit](https://img.shields.io/badge/streamlit-1.58+-red.svg)
![License](https://img.shields.io/badge/license-MIT-green.svg)

## 📋 Overview

This project uses ensemble machine learning models to predict match outcomes for the FIFA World Cup 2026. It combines historical data analysis with real-time FIFA data to provide accurate predictions and comprehensive analytics.

### ✨ Key Features

- **🤖 Ensemble Prediction Model**: Combines ELO Rating (40%) and Poisson Distribution (60%)
- **📊 Interactive Dashboard**: Real-time visualization with native Streamlit multipage architecture
- **📈 Historical Analysis**: Trained on 4,572 World Cup matches (1930-2014)
- **🌐 Live Data Integration**: Fetches current standings and results from FIFA.com
- **🎯 Accuracy Tracking**: Real-time model performance monitoring
- **🏆 Tournament Visualization**: Complete bracket with group stage and knockout rounds

## 🚀 Quick Start

### Prerequisites

- Python 3.8 or higher
- pip package manager

### Installation

1. **Clone the repository**
```bash
git clone https://github.com/yourusername/bolao_previsao.git
cd bolao_previsao
```

2. **Create virtual environment** (recommended)
```bash
python -m venv venv
source venv/bin/activate  # On Windows: venv\Scripts\activate
```

3. **Install dependencies**
```bash
pip install -r requirements.txt
```

4. **Download historical data**
   - Download the [FIFA World Cup Dataset](https://www.kaggle.com/datasets/abecklas/fifa-world-cup) from Kaggle
   - Place `WorldCupMatches.csv` in the `data/raw/` directory

### Running the Application

**Launch the dashboard**
```bash
streamlit run main.py
```

The models are automatically trained on startup using historical data. Access the dashboard at `http://localhost:8501`

## 📁 Project Structure

```
bolao_previsao/
├── main.py                         # Main application entry point
├── requirements.txt                # Python dependencies
├── .gitignore                      # Git ignore rules
│
├── app_pages/                      # Streamlit pages
│   ├── tournament_bracket.py      # Tournament visualization
│   ├── group_stage.py             # Group stage details
│   ├── model_analysis.py          # Model analytics
│   └── about.py                   # Project information
│
├── src/                            # Source code
│   ├── models/                     # Prediction models
│   │   ├── elo_predictor.py       # ELO rating system
│   │   ├── poisson_predictor.py   # Poisson distribution model
│   │   └── ensemble_predictor.py  # Ensemble model
│   │
│   ├── data/                       # Data processing
│   │   ├── loader.py              # Data loading utilities
│   │   └── fifa_scraper.py        # FIFA.com data scraper
│   │
│   ├── components/                 # UI components
│   │   ├── match_display.py       # Match visualization
│   │   └── styles.py              # Custom CSS styles
│   │
│   └── utils/                      # Utility functions
│       ├── team_names.py          # Team name abbreviations
│       ├── country_flags.py       # Flag utilities
│       └── flag_images.py         # Flag image handling
│
├── data/                           # Data directory
│   ├── 2026_world_cup_schedule.csv # 2026 tournament schedule
│   ├── raw/                        # Raw data files
│   │   └── WorldCupMatches.csv    # Historical match data
│   ├── processed/                  # Processed data
│   ├── predictions/                # Model predictions
│   └── cache/                      # Cache directory
│
├── scripts/                        # Utility scripts
│   └── train_models.py            # Standalone model training
│
├── notebooks/                      # Jupyter notebooks
│   └── model_explanation.ipynb    # Model training explanation
│
├── docs/                           # Documentation
│   └── CONTRIBUTING.md            # Contribution guidelines
│
└── assets/                         # Static assets
    └── profile.png                # Developer profile image
```

## 🤖 Model Architecture

### Ensemble Predictor

The system uses an ensemble approach combining two complementary models:

#### 1. ELO Rating System (40% weight)
- **Purpose**: Measures relative team strength
- **K-factor**: 32 (determines rating volatility)
- **Home Advantage**: 100 points
- **Initial Rating**: 1500 for all teams
- **Updates**: Dynamic after each match

#### 2. Poisson Distribution Model (60% weight)
- **Purpose**: Predicts goal probabilities
- **Features**:
  - Attack strength analysis
  - Defense strength analysis
  - Historical performance patterns
- **Output**: Expected goals and most likely scores

### Training Data

- **Source**: [Kaggle FIFA World Cup Dataset](https://www.kaggle.com/datasets/abecklas/fifa-world-cup)
- **Period**: 1930-2014
- **Matches**: 4,572 historical games
- **Teams**: 84 national teams

## 📊 Dashboard Features

### Pages

1. **🏆 Tournament Bracket**
   - Complete tournament visualization
   - Group stage and knockout rounds
   - Match predictions with probabilities
   - Team abbreviations for compact display

2. **📊 Group Stage Details**
   - Detailed match analysis by group
   - Win/Draw/Loss probabilities
   - Expected vs actual scores
   - Prediction accuracy indicators
   - Color-coded result boxes

3. **📈 Model Analysis**
   - ELO ratings (top 20 teams)
   - Team statistics (attack/defense)
   - Complete rankings
   - Training data insights with CSV download

4. **ℹ️ About**
   - Project information
   - Model explanation
   - Data sources and credits
   - Developer information

### Sidebar Features

- **🎯 Model Accuracy**: Real-time prediction accuracy tracking (appears when matches are completed)
- **ℹ️ About**: Quick model information

### Real-time Features

- Live FIFA data integration (5-minute cache)
- Model accuracy tracking on completed matches
- Prediction vs reality comparison
- Interactive visualizations with Plotly

## 🔧 Configuration

### Data Update Frequency

The dashboard automatically refreshes data every 5 minutes. To force an immediate update:
1. Click the menu (☰) in the top-right corner
2. Select "Clear cache"
3. The page will reload with fresh data

### Manual Data Updates

To update match results manually:
1. Edit `data/2026_world_cup_schedule.csv`
2. Update scores and status for completed matches
3. Wait 5 minutes or clear cache

## 📚 Technologies Used

- **[Streamlit](https://streamlit.io/)** - Dashboard framework with native multipage support
- **[Pandas](https://pandas.pydata.org/)** - Data manipulation and analysis
- **[NumPy](https://numpy.org/)** - Numerical computing
- **[Plotly](https://plotly.com/)** - Interactive visualizations
- **[Requests](https://requests.readthedocs.io/)** - HTTP library for data fetching
- **[BeautifulSoup4](https://www.crummy.com/software/BeautifulSoup/)** - Web scraping

## 📖 Documentation

For detailed documentation, see:
- [Model Explanation Notebook](notebooks/model_explanation.ipynb) - Detailed model training process
- [Training Script](scripts/train_models.py) - Standalone model training
- [Contributing Guidelines](docs/CONTRIBUTING.md) - How to contribute

## 🤝 Contributing

Contributions are welcome! Please read our [Contributing Guidelines](docs/CONTRIBUTING.md) before submitting a Pull Request.

## 👨‍💻 Developer

**João dos Dados**

Profissional orientado a dados, com mais de 6 anos de experiência no desenvolvimento de soluções inovadoras e escaláveis. Possui mais de 1 ano de experiência na liderança e gestão de times de Ciência de Dados.

- **Email**: lucas.xms@gmail.com
- **GitHub**: [@joaodosdados](https://github.com/joaodosdados)
- **LinkedIn**: [linkedin.com/in/joaodosdados](https://www.linkedin.com/in/joaodosdados/)

## 📄 License

This project is open source and available under the MIT License.

## 🙏 Acknowledgments

- **FIFA** for official tournament data
- **Kaggle** for historical World Cup dataset ([abecklas/fifa-world-cup](https://www.kaggle.com/datasets/abecklas/fifa-world-cup))
- **FlagCDN** for country flag images
- **Streamlit** community for the amazing framework

## 🐛 Known Issues

- FIFA web scraper currently uses fallback data (FIFA.com structure may have changed)
- Some team names may need manual mapping for historical data compatibility

## 🔮 Future Enhancements

- [ ] Real-time FIFA API integration
- [ ] Monte Carlo simulation for tournament outcomes
- [ ] Player-level statistics integration
- [ ] Mobile-responsive design improvements
- [ ] Multi-language support

---

**Made with ❤️ for World Cup 2026**