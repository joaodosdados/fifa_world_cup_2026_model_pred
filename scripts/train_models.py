"""
Model Training Script
Trains the ELO and Poisson models on historical World Cup data
"""

import sys
from pathlib import Path

# Add project root to path
project_root = Path(__file__).parent.parent
sys.path.insert(0, str(project_root))

from src.models.ensemble_predictor import EnsemblePredictor
from src.data.loader import DataLoader

def main():
    """Train and save prediction models"""
    
    print("=" * 60)
    print("FIFA World Cup 2026 - Model Training")
    print("=" * 60)
    print()
    
    # Initialize data loader
    print("📊 Loading historical data...")
    loader = DataLoader()
    
    try:
        matches_df = loader.load_matches(processed=False)
        print(f"✓ Loaded {len(matches_df)} historical matches")
        if 'Year' in matches_df.columns:
            print(f"  Period: {matches_df['Year'].min()}-{matches_df['Year'].max()}")
        print()
    except FileNotFoundError:
        print("❌ Error: WorldCupMatches.csv not found!")
        print()
        print("Please download the dataset from:")
        print("https://www.kaggle.com/datasets/abecklas/fifa-world-cup")
        print()
        print("And place it in the data/raw/ directory")
        return 1
    
    # Initialize ensemble predictor
    print("🤖 Initializing prediction models...")
    ensemble = EnsemblePredictor()
    print("✓ Models initialized")
    print()
    
    # Train models
    print("🎯 Training models on historical data...")
    print("  - ELO Rating System (K=32, Home Advantage=100)")
    print("  - Poisson Distribution Model")
    print()
    
    ensemble.train(matches_df)
    
    print("✓ Training complete!")
    print()
    
    # Display statistics
    print("📈 Model Statistics:")
    print("-" * 60)
    
    # ELO statistics
    ratings_df = ensemble.elo_model.get_all_ratings()
    valid_ratings = ratings_df[ratings_df['team'] != 'nan']
    print(f"  Teams rated: {len(valid_ratings)}")
    print(f"  Top team: {valid_ratings.iloc[0]['team']} (ELO: {valid_ratings.iloc[0]['elo_rating']:.0f})")
    print()
    
    # Poisson statistics
    team_count = len([t for t in ensemble.poisson_model.team_stats.keys() if t != 'nan'])
    print(f"  Teams analyzed: {team_count}")
    print(f"  League average goals: {ensemble.poisson_model.league_avg_goals:.2f}")
    print()
    
    print("=" * 60)
    print("✅ Training completed successfully!")
    print("=" * 60)
    print()
    print("Next steps:")
    print("  1. Run the dashboard: streamlit run app.py")
    print("  2. View predictions at: http://localhost:8501")
    print()
    
    return 0

if __name__ == "__main__":
    exit(main())

# Made with Bob
