"""
Ensemble Weight Optimization Script
Finds optimal weights for ELO + Poisson ensemble using grid search
"""

import sys
from pathlib import Path
import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split
from itertools import product

# Add project root to path
project_root = Path(__file__).parent.parent
sys.path.insert(0, str(project_root))

from src.data.loader import DataLoader
from src.models.ensemble_predictor import EnsemblePredictor
from src.models.model_manager import ModelManager


def evaluate_weights(elo_weight: float, poisson_weight: float, 
                     train_df: pd.DataFrame, test_df: pd.DataFrame) -> dict:
    """
    Avalia um conjunto de pesos para o ensemble
    
    Args:
        elo_weight: Peso do modelo ELO (0-1)
        poisson_weight: Peso do modelo Poisson (0-1)
        train_df: Dados de treino
        test_df: Dados de teste
        
    Returns:
        Dicionário com métricas de performance
    """
    # Criar e treinar ensemble com pesos específicos
    ensemble = EnsemblePredictor(elo_weight=elo_weight, poisson_weight=poisson_weight)
    ensemble.train(train_df)
    
    # Avaliar no conjunto de teste
    correct = 0
    total = 0
    
    for _, match in test_df.iterrows():
        try:
            prediction = ensemble.predict_match(
                match['Home Team Name'], 
                match['Away Team Name']
            )
            
            # Determinar vencedor real
            if match['Home Team Goals'] > match['Away Team Goals']:
                actual = 'home'
            elif match['Home Team Goals'] < match['Away Team Goals']:
                actual = 'away'
            else:
                actual = 'draw'
            
            # Determinar vencedor previsto
            if prediction['home_win'] > prediction['draw'] and prediction['home_win'] > prediction['away_win']:
                predicted = 'home'
            elif prediction['away_win'] > prediction['draw'] and prediction['away_win'] > prediction['home_win']:
                predicted = 'away'
            else:
                predicted = 'draw'
            
            if predicted == actual:
                correct += 1
            total += 1
            
        except Exception as e:
            continue
    
    accuracy = correct / total if total > 0 else 0
    
    return {
        'elo_weight': elo_weight,
        'poisson_weight': poisson_weight,
        'accuracy': accuracy,
        'correct': correct,
        'total': total
    }


def grid_search_weights(train_df: pd.DataFrame, test_df: pd.DataFrame, 
                       step: float = 0.1) -> pd.DataFrame:
    """
    Busca em grade pelos melhores pesos
    
    Args:
        train_df: Dados de treino
        test_df: Dados de teste
        step: Tamanho do passo na busca (default: 0.1)
        
    Returns:
        DataFrame com resultados ordenados por acurácia
    """
    print(f"🔍 Iniciando busca em grade com passo de {step}...")
    print(f"📊 Treino: {len(train_df)} jogos | Teste: {len(test_df)} jogos\n")
    
    # Gerar combinações de pesos
    weights = np.arange(0, 1 + step, step)
    combinations = [(w1, 1-w1) for w1 in weights]
    
    results = []
    total_combinations = len(combinations)
    
    for i, (elo_w, poisson_w) in enumerate(combinations, 1):
        print(f"[{i}/{total_combinations}] Testando ELO={elo_w:.1f}, Poisson={poisson_w:.1f}...", end=' ')
        
        result = evaluate_weights(elo_w, poisson_w, train_df, test_df)
        results.append(result)
        
        print(f"Acurácia: {result['accuracy']:.1%}")
    
    # Criar DataFrame e ordenar por acurácia
    results_df = pd.DataFrame(results)
    results_df = results_df.sort_values('accuracy', ascending=False)
    
    return results_df


def main():
    """Função principal"""
    print("=" * 80)
    print("🎯 OTIMIZAÇÃO DE PESOS DO ENSEMBLE")
    print("=" * 80)
    print()
    
    # Carregar dados
    print("📥 Carregando dados históricos...")
    loader = DataLoader()
    matches_df = loader.load_matches(processed=False)
    print(f"✓ Carregados {len(matches_df)} jogos\n")
    
    # Dividir em treino e teste
    train_df, test_df = train_test_split(matches_df, test_size=0.2, random_state=42)
    
    # Executar busca em grade
    results_df = grid_search_weights(train_df, test_df, step=0.05)
    
    # Mostrar resultados
    print("\n" + "=" * 80)
    print("📊 TOP 10 MELHORES COMBINAÇÕES")
    print("=" * 80)
    print()
    
    top_10 = results_df.head(10)
    for i, row in top_10.iterrows():
        print(f"{i+1}. ELO: {row['elo_weight']:.2f} | Poisson: {row['poisson_weight']:.2f} | "
              f"Acurácia: {row['accuracy']:.1%} ({row['correct']}/{row['total']})")
    
    # Melhor combinação
    best = results_df.iloc[0]
    print("\n" + "=" * 80)
    print("🏆 MELHOR COMBINAÇÃO")
    print("=" * 80)
    print(f"ELO Weight: {best['elo_weight']:.2f}")
    print(f"Poisson Weight: {best['poisson_weight']:.2f}")
    print(f"Acurácia: {best['accuracy']:.1%}")
    print(f"Corretos: {best['correct']}/{best['total']}")
    print()
    
    # Salvar ensemble otimizado
    print("💾 Salvando ensemble otimizado...")
    
    # Treinar ensemble com melhores pesos
    optimized_ensemble = EnsemblePredictor(
        elo_weight=best['elo_weight'],
        poisson_weight=best['poisson_weight']
    )
    optimized_ensemble.train(matches_df)
    
    # Salvar usando ModelManager
    manager = ModelManager()
    manager.save_model(
        model=optimized_ensemble,
        model_name='optimized_ensemble',
        metrics={
            'accuracy': best['accuracy'],
            'elo_weight': best['elo_weight'],
            'poisson_weight': best['poisson_weight']
        },
        model_type='ensemble',
        description=f'Optimized Ensemble (ELO: {best["elo_weight"]:.2f}, Poisson: {best["poisson_weight"]:.2f})'
    )
    
    print(f"✓ Ensemble otimizado salvo!")
    print()
    
    # Salvar resultados completos
    results_path = project_root / 'models' / 'ensemble_optimization_results.csv'
    results_df.to_csv(results_path, index=False)
    print(f"✓ Resultados completos salvos em: {results_path}")
    print()
    
    print("=" * 80)
    print("✅ OTIMIZAÇÃO CONCLUÍDA!")
    print("=" * 80)


if __name__ == "__main__":
    main()

# Made with Bob
