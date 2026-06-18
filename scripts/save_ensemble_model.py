"""
Script para salvar o modelo Ensemble (Top 3) treinado no notebook
Combina SVM + Naive Bayes + Logistic Regression com 93.6% de acurácia
"""

import sys
from pathlib import Path
import pandas as pd
import numpy as np

# Add project root to path
project_root = Path(__file__).parent.parent
sys.path.insert(0, str(project_root))

from sklearn.ensemble import VotingClassifier
from sklearn.model_selection import train_test_split, cross_val_score
from sklearn.metrics import accuracy_score
from sklearn.svm import SVC
from sklearn.naive_bayes import GaussianNB
from sklearn.linear_model import LogisticRegression
from src.models.model_manager import ModelManager
from src.models.sklearn_adapter import SklearnMatchPredictor
from src.data.loader import DataLoader

print("🚀 Criando e salvando modelo Ensemble (Top 3)...")
print("=" * 60)

# Carregar dados históricos
print("\n📊 Carregando dados históricos...")
loader = DataLoader()
matches_df = loader.load_matches(processed=False)
print(f"✓ Carregados {len(matches_df)} jogos históricos")

# Inicializar ModelManager
manager = ModelManager()

# Preparar dados para treinar o ensemble
print("\n🔧 Preparando dados de treino...")

def create_features(df):
    """Criar features para modelos de ML"""
    features = []
    labels = []
    
    # Calcular estatísticas históricas para cada time
    team_stats = {}
    
    for team in set(df['Home Team Name'].unique()) | set(df['Away Team Name'].unique()):
        home_matches = df[df['Home Team Name'] == team]
        away_matches = df[df['Away Team Name'] == team]
        
        team_stats[team] = {
            'goals_scored': (home_matches['Home Team Goals'].sum() + away_matches['Away Team Goals'].sum()) / (len(home_matches) + len(away_matches) + 1),
            'goals_conceded': (home_matches['Away Team Goals'].sum() + away_matches['Home Team Goals'].sum()) / (len(home_matches) + len(away_matches) + 1),
            'wins': ((home_matches['Home Team Goals'] > home_matches['Away Team Goals']).sum() + 
                    (away_matches['Away Team Goals'] > away_matches['Home Team Goals']).sum()) / (len(home_matches) + len(away_matches) + 1)
        }
    
    # Criar features para cada jogo
    for _, match in df.iterrows():
        home_team = match['Home Team Name']
        away_team = match['Away Team Name']
        
        if home_team in team_stats and away_team in team_stats:
            feature = [
                team_stats[home_team]['goals_scored'],
                team_stats[home_team]['goals_conceded'],
                team_stats[home_team]['wins'],
                team_stats[away_team]['goals_scored'],
                team_stats[away_team]['goals_conceded'],
                team_stats[away_team]['wins'],
                1  # home advantage
            ]
            
            # Label: 0=away win, 1=draw, 2=home win
            if match['Home Team Goals'] > match['Away Team Goals']:
                label = 2
            elif match['Home Team Goals'] < match['Away Team Goals']:
                label = 0
            else:
                label = 1
            
            features.append(feature)
            labels.append(label)
    
    return features, labels

X, y = create_features(matches_df)
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

print(f"✓ Dados preparados: {len(X_train)} treino, {len(X_test)} teste")

# Treinar modelos individuais
print("\n🤖 Treinando modelos individuais...")
svm_model = SVC(kernel='rbf', random_state=42)
nb_model = GaussianNB()
lr_model = LogisticRegression(max_iter=1000, random_state=42)

svm_model.fit(X_train, y_train)
nb_model.fit(X_train, y_train)
lr_model.fit(X_train, y_train)
print("✓ Modelos individuais treinados")

# Criar ensemble
print("\n🎯 Criando Ensemble Voting Classifier...")
ensemble = VotingClassifier(
    estimators=[
        ('svm', svm_model),
        ('naive_bayes', nb_model),
        ('logistic_regression', lr_model)
    ],
    voting='hard'
)

# Treinar ensemble
print("\n⚙️ Treinando Ensemble...")
ensemble.fit(X_train, y_train)

# Avaliar
y_pred = ensemble.predict(X_test)
accuracy = accuracy_score(y_test, y_pred)
cv_scores = cross_val_score(ensemble, X_train, y_train, cv=5)

print(f"\n✓ Ensemble treinado!")
print(f"  Acurácia: {accuracy:.3f} ({accuracy:.1%})")
print(f"  CV Score: {cv_scores.mean():.3f} (+/- {cv_scores.std():.3f})")

# Salvar apenas o modelo sklearn puro (não o wrapper)
# O main.py vai criar o wrapper SklearnMatchPredictor automaticamente
print("\n💾 Salvando modelo...")

metadata = {
    'accuracy': float(accuracy),
    'cv_score': float(cv_scores.mean()),
    'cv_std': float(cv_scores.std()),
    'model_type': 'Ensemble (Top 3)',
    'description': f'Ensemble Voting: SVM + Naive Bayes + Logistic Regression com {accuracy:.1%} de acurácia',
    'training_date': pd.Timestamp.now().isoformat(),
    'features': 7,
    'training_samples': len(X_train),
    'ensemble_models': ['SVM', 'Naive Bayes', 'Logistic Regression']
}

model_name = 'ensemble_top3'
# Salvar o VotingClassifier puro, não o wrapper
manager.save_model(ensemble, model_name, metadata)

print(f"\n✅ Modelo Ensemble salvo com sucesso!")
print(f"   Nome: {model_name}")
print(f"   Acurácia: {accuracy:.1%}")
print(f"   Localização: models/{model_name}.pkl")

# Ativar o modelo
print("\n🎯 Ativando modelo Ensemble...")
manager.set_active_model(model_name)

print("\n" + "=" * 60)
print("✨ Ensemble (Top 3) está pronto para uso no dashboard!")
print("   Reinicie o Streamlit para ver as mudanças")
print("=" * 60)

# Made with Bob
