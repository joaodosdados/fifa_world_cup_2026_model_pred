# Contributing to World Cup 2026 Predictor

Thank you for your interest in contributing to the World Cup 2026 Prediction Dashboard! This document provides guidelines and instructions for contributing to the project.

## Table of Contents

1. [Code of Conduct](#code-of-conduct)
2. [Getting Started](#getting-started)
3. [Development Setup](#development-setup)
4. [Project Structure](#project-structure)
5. [Coding Standards](#coding-standards)
6. [Making Changes](#making-changes)
7. [Testing](#testing)
8. [Submitting Changes](#submitting-changes)
9. [Areas for Contribution](#areas-for-contribution)

## Code of Conduct

This project follows a simple code of conduct:
- Be respectful and inclusive
- Provide constructive feedback
- Focus on what is best for the community
- Show empathy towards other contributors

## Getting Started

### Prerequisites

- Python 3.8 or higher
- Git
- Basic understanding of machine learning concepts
- Familiarity with Streamlit (helpful but not required)

### Fork and Clone

1. Fork the repository on GitHub
2. Clone your fork locally:
```bash
git clone https://github.com/YOUR_USERNAME/bolao_previsao.git
cd bolao_previsao
```

3. Add the upstream repository:
```bash
git remote add upstream https://github.com/ORIGINAL_OWNER/bolao_previsao.git
```

## Development Setup

### 1. Create Virtual Environment

```bash
python -m venv venv
source venv/bin/activate  # On Windows: venv\Scripts\activate
```

### 2. Install Dependencies

```bash
pip install -r requirements.txt
```

### 3. Download Training Data

Download the historical World Cup data from Kaggle:
- Dataset: [World Cup Dataset](https://www.kaggle.com/datasets/abecklas/fifa-world-cup)
- Place `WorldCupMatches.csv` in `data/raw/`

### 4. Train Models

```bash
python scripts/train_models.py
```

### 5. Run the Application

```bash
streamlit run app.py
```

## Project Structure

```
bolao_previsao/
├── app.py                          # Main Streamlit application
├── requirements.txt                # Python dependencies
├── README.md                       # Project documentation
├── CONTRIBUTING.md                 # This file
├── OPTIMIZATIONS.md                # Performance optimizations
├── .gitignore                      # Git ignore rules
├── assets/                         # Static assets
│   └── profile.png                 # Developer profile image
├── data/                           # Data directory
│   ├── 2026_world_cup_schedule.csv # Tournament schedule
│   ├── times.txt                   # FIFA standings source
│   ├── cache/                      # Cache files
│   ├── predictions/                # Prediction outputs
│   ├── processed/                  # Processed data
│   └── raw/                        # Raw historical data
├── models/                         # Trained model files
├── notebooks/                      # Jupyter notebooks
│   └── model_explanation.ipynb     # Model documentation
├── scripts/                        # Utility scripts
│   ├── parse_fifa_data.py          # FIFA data parser
│   └── train_models.py             # Model training pipeline
└── src/                            # Source code
    ├── data/                       # Data modules
    │   ├── loader.py               # Data loading utilities
    │   └── fifa_scraper.py         # FIFA web scraper
    ├── models/                     # ML models
    │   ├── elo_predictor.py        # ELO rating system
    │   ├── poisson_predictor.py    # Poisson distribution model
    │   └── ensemble_predictor.py   # Ensemble predictor
    └── utils/                      # Utility modules
        ├── __init__.py             # Package initialization
        └── flag_images.py          # Flag image utilities
```

## Coding Standards

### Python Style Guide

Follow [PEP 8](https://pep8.org/) style guide:
- Use 4 spaces for indentation
- Maximum line length: 100 characters
- Use descriptive variable names
- Add docstrings to all functions and classes

### Example Function Documentation

```python
def predict_match(home_team: str, away_team: str) -> dict:
    """
    Predict the outcome of a match between two teams.
    
    Parameters:
    -----------
    home_team : str
        Name of the home team
    away_team : str
        Name of the away team
    
    Returns:
    --------
    dict
        Dictionary containing prediction results with keys:
        - 'home_win_prob': float
        - 'draw_prob': float
        - 'away_win_prob': float
        - 'predicted_score': tuple
    """
    # Implementation here
    pass
```

### Streamlit Best Practices

1. **Use Caching Appropriately**
   - `@st.cache_resource` for ML models and connections
   - `@st.cache_data` for data loading and transformations

2. **Optimize Performance**
   - Minimize reruns with proper state management
   - Use `st.session_state` for persistent data
   - Implement lazy loading for heavy computations

3. **Clean UI/UX**
   - Use containers and columns for layout
   - Provide clear error messages
   - Add loading indicators for long operations

## Making Changes

### Branch Naming Convention

- `feature/description` - New features
- `fix/description` - Bug fixes
- `docs/description` - Documentation updates
- `refactor/description` - Code refactoring
- `test/description` - Test additions/updates

Example:
```bash
git checkout -b feature/add-knockout-predictions
```

### Commit Message Format

Use clear, descriptive commit messages:

```
<type>: <subject>

<body>

<footer>
```

Types:
- `feat`: New feature
- `fix`: Bug fix
- `docs`: Documentation changes
- `style`: Code style changes (formatting, etc.)
- `refactor`: Code refactoring
- `test`: Test additions/updates
- `chore`: Maintenance tasks

Example:
```
feat: Add knockout stage prediction visualization

- Implement Round of 16 bracket display
- Add quarter-finals prediction logic
- Update UI with match progression indicators

Closes #123
```

## Testing

### Manual Testing Checklist

Before submitting changes, verify:

- [ ] Application starts without errors
- [ ] All pages load correctly
- [ ] Predictions display properly
- [ ] No console errors in browser
- [ ] Data loads from cache correctly
- [ ] UI is responsive on different screen sizes

### Running Tests

```bash
# Run unit tests (when available)
pytest tests/

# Run linting
flake8 src/ app.py

# Check code formatting
black --check src/ app.py
```

## Submitting Changes

### Pull Request Process

1. **Update Your Fork**
```bash
git fetch upstream
git rebase upstream/main
```

2. **Push Your Changes**
```bash
git push origin feature/your-feature-name
```

3. **Create Pull Request**
   - Go to GitHub and create a PR from your fork
   - Fill out the PR template completely
   - Link any related issues
   - Add screenshots for UI changes

### PR Checklist

- [ ] Code follows project style guidelines
- [ ] Documentation updated (if needed)
- [ ] Tests added/updated (if applicable)
- [ ] All tests pass
- [ ] No merge conflicts
- [ ] PR description is clear and complete

### PR Template

```markdown
## Description
Brief description of changes

## Type of Change
- [ ] Bug fix
- [ ] New feature
- [ ] Documentation update
- [ ] Performance improvement
- [ ] Code refactoring

## Testing
Describe testing performed

## Screenshots (if applicable)
Add screenshots for UI changes

## Checklist
- [ ] Code follows style guidelines
- [ ] Self-review completed
- [ ] Documentation updated
- [ ] No new warnings generated
```

## Areas for Contribution

### High Priority

1. **Model Improvements**
   - Implement additional prediction models
   - Improve ensemble weighting strategy
   - Add player-level statistics integration

2. **Data Enhancement**
   - Automated FIFA data scraping
   - Historical match data updates
   - Player performance tracking

3. **UI/UX Enhancements**
   - Mobile responsiveness improvements
   - Interactive tournament bracket
   - Real-time score updates

### Medium Priority

4. **Testing**
   - Unit tests for prediction models
   - Integration tests for data pipeline
   - UI testing with Selenium

5. **Documentation**
   - Video tutorials
   - API documentation
   - Deployment guides

6. **Features**
   - Export predictions to CSV/PDF
   - Comparison with betting odds
   - Historical accuracy tracking

### Low Priority

7. **Optimization**
   - Database integration for faster queries
   - Parallel processing for predictions
   - CDN for static assets

8. **Internationalization**
   - Multi-language support
   - Localized team names
   - Regional date/time formats

## Questions or Issues?

- **Bug Reports**: Open an issue with detailed description
- **Feature Requests**: Open an issue with use case explanation
- **Questions**: Start a discussion in GitHub Discussions
- **Security Issues**: Email directly (do not open public issue)

## Recognition

Contributors will be recognized in:
- README.md Contributors section
- Release notes
- Project documentation

Thank you for contributing to the World Cup 2026 Predictor! 🎉⚽