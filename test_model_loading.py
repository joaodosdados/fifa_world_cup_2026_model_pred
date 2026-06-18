#!/usr/bin/env python3
"""
Script para testar o carregamento do modelo ativo
"""

from src.models.model_manager import ModelManager
from src.data.loader import DataLoader

print("\n" + "="*60)
print("TESTE DE CARREGAMENTO DE MODELO")
print("="*60)

# Inicializar ModelManager
manager = ModelManager()

# Listar todos os modelos
print("\n📋 Modelos disponíveis:")
for name, info in manager.metadata.items():
    active = "✓ ATIVO" if info.get('active', False) else ""
    acc = info['metrics'].get('accuracy', 0)
    print(f"  {name}: {acc:.3f} ({acc:.1%}) {active}")

# Carregar modelo ativo
print("\n🔄 Carregando modelo ativo...")
model, model_name = manager.get_active_model()

if model is not None:
    print(f"✅ Modelo carregado: {model_name}")
    print(f"   Tipo: {type(model).__name__}")
    
    # Verificar se é um VotingClassifier
    if hasattr(model, 'estimators_'):
        print(f"   Estimadores: {[name for name, _ in model.estimators]}")
else:
    print("❌ Nenhum modelo foi carregado!")

print("="*60 + "\n")

# Made with Bob
