# 🤖 Model Selection Guide

Guia completo sobre o sistema de modelos do FIFA World Cup 2026 Prediction Dashboard.

## 📋 Visão Geral

O sistema oferece dois tipos de modelos de predição que podem ser alternados dinamicamente:

1. **ML Model (SVM)** - Machine Learning com Support Vector Machine
2. **Statistical Model (ELO)** - Modelo estatístico com ELO Rating + Poisson

## 🎯 Quando Usar Cada Modelo

### ML Model (SVM) - Recomendado para Predições

**Use quando:**
- Você quer a maior acurácia possível (93.3%)
- Precisa de predições de resultado (vitória/empate/derrota)
- Quer aproveitar padrões complexos dos dados históricos
- Está fazendo apostas ou análises críticas

**Características:**
- ✅ Alta acurácia (93.3% no teste, 92.9% cross-validation)
- ✅ Aprende padrões não-lineares
- ✅ Robusto a outliers
- ❌ Não fornece rankings de times
- ❌ Menos interpretável

### Statistical Model (ELO) - Recomendado para Análises

**Use quando:**
- Você quer entender rankings de times
- Precisa de análise estatística detalhada
- Quer ver força de ataque/defesa dos times
- Está fazendo análise exploratória

**Características:**
- ✅ Rankings interpretáveis (ELO ratings)
- ✅ Estatísticas detalhadas por time
- ✅ Previsão de placares (Poisson)
- ✅ Fácil de entender e explicar
- ❌ Acurácia menor que ML

## 🔄 Como Alternar Entre Modelos

### No Dashboard

1. Abra o dashboard (`streamlit run main.py`)
2. No sidebar, localize **🤖 Active Model**
3. Selecione o modelo desejado no dropdown:
   - `ML (SVM)` - Machine Learning
   - `ELO` - Statistical Model
4. O sistema automaticamente:
   - Atualiza todas as predições
   - Recalcula a acurácia
   - Ajusta a página Model Analysis

### Programaticamente

```python
from src.models.model_manager import ModelManager

# Inicializar gerenciador
manager = ModelManager()

# Listar modelos disponíveis
models = manager.list_models()
print(models)

# Carregar melhor modelo
best_model, best_name = manager.get_best_model()
print(f"Best model: {best_name}")

# Carregar modelo específico
model, name = manager.load_model("svm")
```

## 🏋️ Treinando Novos Modelos

### Usando o Notebook

O método recomendado é usar o notebook interativo:

```bash
jupyter notebook notebooks/model_training_analysis.ipynb
```

O notebook treina automaticamente 6 modelos:
1. Logistic Regression
2. Random Forest
3. Gradient Boosting
4. **SVM (melhor)** ⭐
5. K-Nearest Neighbors
6. Naive Bayes

### Usando Script

Para treinar modelos via linha de comando:

```bash
python scripts/train_models.py
```

### Otimizando Ensemble

Para encontrar os melhores pesos do ensemble ELO + Poisson:

```bash
python scripts/optimize_ensemble.py
```

## 📊 Estrutura dos Modelos Salvos

Os modelos são salvos em `models/` com a seguinte estrutura:

```
models/
├── svm.pkl                    # Modelo treinado
├── random_forest.pkl          # Outro modelo
├── logistic_regression.pkl    # Outro modelo
└── models_metadata.json       # Metadados de todos os modelos
```

### Formato do Metadata

```json
{
  "svm": {
    "type": "sklearn",
    "description": "SVM trained on World Cup historical data",
    "metrics": {
      "accuracy": 0.933,
      "cv_mean": 0.929,
      "cv_std": 0.015
    },
    "active": true,
    "created_at": "2026-06-18T20:00:00"
  }
}
```

## 🎓 Features do ML Model

O modelo SVM usa 7 features calculadas automaticamente:

1. **goals_scored_home** - Média de gols marcados em casa
2. **goals_scored_away** - Média de gols marcados fora
3. **goals_conceded_home** - Média de gols sofridos em casa
4. **goals_conceded_away** - Média de gols sofridos fora
5. **win_rate_home** - Taxa de vitórias em casa
6. **win_rate_away** - Taxa de vitórias fora
7. **home_advantage** - Fator de vantagem de jogar em casa (1.0)

### Cálculo Automático

O `SklearnMatchPredictor` calcula essas features automaticamente:

```python
from src.models.sklearn_adapter import SklearnMatchPredictor
from src.data.loader import DataLoader

# Carregar dados históricos
loader = DataLoader()
matches_df = loader.load_matches(processed=False)

# Criar adaptador (calcula features automaticamente)
predictor = SklearnMatchPredictor(model, matches_df)

# Fazer predição
result = predictor.predict_match("Brazil", "Argentina")
print(result)
# {
#   'home_win': 0.65,
#   'draw': 0.20,
#   'away_win': 0.15,
#   'expected_goals_home': 2.1,
#   'expected_goals_away': 1.3,
#   'most_likely_score': (2, 1)
# }
```

## 📈 Métricas de Avaliação

### Acurácia de Treinamento

Medida durante o treinamento no conjunto de teste:

- **SVM**: 93.3%
- **Random Forest**: 91.2%
- **Gradient Boosting**: 90.8%

### Cross-Validation

Validação cruzada 5-fold para evitar overfitting:

- **SVM**: 92.9% ± 1.5%
- **Random Forest**: 90.5% ± 2.1%
- **Gradient Boosting**: 89.8% ± 2.3%

### Acurácia em Tempo Real

Medida nos jogos completados da Copa 2026:

- Atualizada automaticamente no sidebar
- Mostra X/Y correct
- Permite comparar modelos

## 🔧 Configuração Avançada

### Ajustar Pesos do Ensemble

Edite `main.py`:

```python
ensemble = EnsemblePredictor(
    elo_weight=0.4,      # 40% ELO
    poisson_weight=0.6   # 60% Poisson
)
```

### Treinar Modelo Customizado

```python
from sklearn.ensemble import RandomForestClassifier
from src.models.model_manager import ModelManager
from src.data.loader import DataLoader

# Carregar dados
loader = DataLoader()
X_train, X_test, y_train, y_test = loader.prepare_ml_data()

# Treinar modelo
model = RandomForestClassifier(n_estimators=200)
model.fit(X_train, y_train)

# Salvar modelo
manager = ModelManager()
manager.save_model(
    model=model,
    name="random_forest_custom",
    model_type="sklearn",
    description="Custom Random Forest with 200 trees",
    metrics={
        'accuracy': model.score(X_test, y_test),
        'n_estimators': 200
    }
)
```

## 🐛 Troubleshooting

### Modelo não carrega

**Problema**: `FileNotFoundError: No models found`

**Solução**: Execute o notebook de treinamento:
```bash
jupyter notebook notebooks/model_training_analysis.ipynb
```

### Acurácia muito baixa

**Problema**: Modelo com acurácia < 50%

**Possíveis causas**:
1. Dados de treinamento insuficientes
2. Features mal calculadas
3. Modelo não adequado para o problema

**Solução**: Retreine o modelo com mais dados ou tente outro algoritmo

### Predições inconsistentes

**Problema**: Resultados muito diferentes entre modelos

**Explicação**: Normal! Cada modelo usa abordagem diferente:
- ML aprende padrões complexos
- ELO usa força relativa dos times
- Poisson usa estatísticas de gols

## 📚 Referências

- [Scikit-learn Documentation](https://scikit-learn.org/)
- [ELO Rating System](https://en.wikipedia.org/wiki/Elo_rating_system)
- [Poisson Distribution](https://en.wikipedia.org/wiki/Poisson_distribution)
- [Support Vector Machines](https://scikit-learn.org/stable/modules/svm.html)

## 🤝 Contribuindo

Para adicionar novos modelos:

1. Treine o modelo no notebook
2. Salve usando `ModelManager`
3. Teste no dashboard
4. Documente as métricas
5. Abra um Pull Request

---

**Última atualização**: 2026-06-18