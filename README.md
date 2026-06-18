# ⚽ FIFA World Cup 2026 Prediction Dashboard

A machine learning-powered prediction system for the FIFA World Cup 2026, featuring an interactive dashboard built with Streamlit's native multipage architecture.

![Dashboard Screenshot](assets/home.png)

![Python](https://img.shields.io/badge/python-3.8+-blue.svg)
![Streamlit](https://img.shields.io/badge/streamlit-1.58+-red.svg)
![License](https://img.shields.io/badge/license-MIT-green.svg)

## 📋 Overview

This project uses ensemble machine learning models to predict match outcomes for the FIFA World Cup 2026. It combines historical data analysis with real-time FIFA data to provide accurate predictions and comprehensive analytics.

### ✨ Key Features

- **🤖 Dual Model System**: Switch between ML (SVM 93.3% accuracy) and Statistical (ELO + Poisson) models
- **📊 Interactive Dashboard**: Real-time visualization with native Streamlit multipage architecture
- **🧠 Machine Learning**: Support Vector Machine trained on historical World Cup data
- **📈 Historical Analysis**: Trained on 4,572 World Cup matches (1930-2014)
- **🌐 Live Data Integration**: Fetches current standings and results from FIFA.com
- **🎯 Real-time Accuracy**: Track model performance on 2026 World Cup matches
- **🏆 Tournament Visualization**: Complete bracket with group stage and knockout rounds
- **🔄 Model Comparison**: Compare predictions and analysis between different models

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
│   ├── model_analysis.py          # Model analytics (dynamic per model)
│   ├── model_comparison.py        # Model comparison (legacy)
│   └── about.py                   # Project information
│
├── src/                            # Source code
│   ├── models/                     # Prediction models
│   │   ├── elo_predictor.py       # ELO rating system
│   │   ├── poisson_predictor.py   # Poisson distribution model
│   │   ├── ensemble_predictor.py  # Ensemble model (ELO + Poisson)
│   │   ├── sklearn_adapter.py     # ML model adapter
│   │   └── model_manager.py       # Model management system
│   │
│   ├── data/                       # Data processing
│   │   ├── loader.py              # Data loading utilities
│   │   ├── fifa_scraper.py        # FIFA.com data scraper (legacy)
│   │   ├── fifa_scraper_selenium.py # Selenium-based scraper
│   │   ├── auto_updater.py        # Automatic data updater
│   │   ├── fifa_api_client.py     # FIFA Official API client (reference)
│   │   └── api_football_client.py # API-Football client (reference)
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
├── models/                         # Trained models directory
│   └── models_metadata.json       # Model metadata and metrics
│
├── scripts/                        # Utility scripts
│   ├── train_models.py            # Standalone model training
│   └── optimize_ensemble.py       # Ensemble weight optimization
│
├── notebooks/                      # Jupyter notebooks
│   ├── model_explanation.ipynb    # Model training explanation
│   └── model_training_analysis.ipynb # ML model training & comparison
│
├── docs/                           # Documentation
│   ├── CONTRIBUTING.md            # Contribution guidelines
│   └── FIFA_API_INTEGRATION.md    # API integration guide
│
└── assets/                         # Static assets
    └── profile.png                # Developer profile image
```

## 🤖 Model Architecture

### Dual Model System

The system offers two prediction approaches that you can switch between:

#### 1. Machine Learning Model (Primary) - SVM
- **Type**: Support Vector Machine (SVM)
- **Accuracy**: 93.3% on test data
- **Cross-Validation**: 92.9% (5-fold CV)
- **Features** (7 total):
  - Goals Scored (Home/Away)
  - Goals Conceded (Home/Away)
  - Win Rate (Home/Away)
  - Home Advantage Factor
- **Training**: Scikit-learn with RBF kernel
- **Use Case**: Best for match outcome predictions

#### 2. Statistical Model (Alternative) - ELO + Poisson
- **Type**: Ensemble of ELO Rating (40%) and Poisson Distribution (60%)
- **ELO Component**:
  - K-factor: 32 (rating volatility)
  - Home Advantage: 100 points
  - Initial Rating: 1500 for all teams
  - Dynamic updates after each match
- **Poisson Component**:
  - Attack/Defense strength analysis
  - Expected goals calculation
  - Most likely score prediction
- **Use Case**: Best for statistical analysis and team rankings

### Model Management

- **Automatic Selection**: System loads the best trained model on startup
- **Manual Switching**: Toggle between ML and Statistical models in the sidebar
- **Real-time Accuracy**: Track each model's performance on 2026 matches
- **Model Persistence**: Trained models saved with metadata for reuse

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

3. **📈 Model Analysis** (Dynamic per active model)
   - **ML Model (SVM)**:
     - Model performance metrics
     - Feature importance
     - Training data statistics
   - **Statistical Model (ELO)**:
     - ELO ratings (top 20 teams)
     - Team statistics (attack/defense)
     - Complete rankings
     - Training data insights

4. **ℹ️ About**
   - Project information
   - Model explanation
   - Data sources and credits
   - Developer information

### Sidebar Features

- **🤖 Active Model**: Dropdown to switch between ML (SVM) and ELO models
- **🎯 2026 Predictions**: Real-time accuracy tracking on completed matches
- **ℹ️ About**: Model information and usage guide (in Portuguese)

### Real-time Features

- **Automatic Data Updates**: Fetches latest results from FIFA.com every 5 minutes
- **Selenium Web Scraping**: Handles dynamic JavaScript-rendered content
- **Model Accuracy Tracking**: Real-time performance monitoring on completed matches
- **Prediction vs Reality**: Side-by-side comparison of predictions and actual results
- **Interactive Visualizations**: Dynamic charts with Plotly

## 🔧 Configuration

### Automatic Data Updates

The system automatically fetches the latest match results from FIFA.com:

- **Update Frequency**: Every 5 minutes (TTL cache)
- **Auto-run on Startup**: Updates run automatically when launching the app
- **Notification**: Toast message shows when new data is fetched
- **Fallback**: Uses sample data if FIFA.com is unavailable

To force an immediate update:
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
- **[Selenium](https://www.selenium.dev/)** - Browser automation for dynamic content scraping
- **[BeautifulSoup4](https://www.crummy.com/software/BeautifulSoup/)** - HTML parsing
- **[Requests](https://requests.readthedocs.io/)** - HTTP library for data fetching

## 📖 Documentation

For detailed documentation, see:
- [Model Training & Analysis](notebooks/model_training_analysis.ipynb) - ML model training and comparison
- [Model Explanation Notebook](notebooks/model_explanation.ipynb) - Statistical model details
- [Model Selection Guide](docs/MODEL_SELECTION_GUIDE.md) - How to choose and train models
- [Training Script](scripts/train_models.py) - Standalone model training
- [Ensemble Optimization](scripts/optimize_ensemble.py) - Optimize ensemble weights
- [FIFA API Integration Guide](docs/FIFA_API_INTEGRATION.md) - External API options
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

- Selenium scraper uses fallback data when FIFA.com structure changes
- Some team names may need manual mapping for historical data compatibility
- Chrome/Chromium required for Selenium scraper (auto-installed via webdriver-manager)

## 🔮 Future Enhancements

- [ ] Deep Learning models (Neural Networks)
- [ ] Integrate commercial APIs for enhanced predictions (API-Football)
- [ ] Monte Carlo simulation for tournament outcomes
- [ ] Player-level statistics integration
- [ ] Mobile-responsive design improvements
- [ ] Multi-language support (currently Portuguese sidebar)
- [ ] Advanced team form analysis
- [ ] Automated model retraining pipeline

---

**Made with ❤️ for World Cup 2026**