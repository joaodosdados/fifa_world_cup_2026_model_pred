# Notebooks de Análise - FIFA World Cup 2026

Este diretório contém notebooks Jupyter para análise e treinamento de modelos de predição.

## 📓 Notebooks Disponíveis

### 1. `model_training_analysis.ipynb` (RECOMENDADO)

Notebook completo e atualizado que inclui:

- ✅ Carregamento correto dos dados usando `DataLoader`
- ✅ Exploração e visualização dos dados históricos
- ✅ Treinamento do modelo customizado (ELO + Poisson Ensemble)
- ✅ **Teste automático de múltiplos modelos de ML** usando scikit-learn:
  - Logistic Regression
  - Random Forest
  - Gradient Boosting
  - SVM
  - K-Nearest Neighbors
  - Naive Bayes
- ✅ Comparação de performance entre todos os modelos
- ✅ Seleção automática do melhor modelo
- ✅ Visualizações e métricas detalhadas

**Este notebook responde à sua pergunta sobre testar múltiplos modelos automaticamente!**

### 2. `model_explanation.ipynb` (LEGADO)

Notebook original com explicações teóricas dos modelos ELO e Poisson.

## 🚀 Como Usar

### Pré-requisitos

1. Instalar dependências:
```bash
pip install -r ../requirements.txt
```

2. Certifique-se de que os dados estão em `data/raw/WorldCupMatches.csv`

### Executar o Notebook

1. Abrir Jupyter Lab ou VS Code:
```bash
# Opção 1: Jupyter Lab
jupyter lab

# Opção 2: VS Code
code .
```

2. Navegar até `notebooks/model_training_analysis.ipynb`

3. Executar todas as células em ordem (Cell > Run All)

## 📊 O que o Notebook Faz

### Parte 1: Modelo Customizado
- Treina os modelos ELO e Poisson desenvolvidos especificamente para futebol
- Mostra rankings de times e estatísticas
- Faz predições de exemplo

### Parte 2: Teste Automático de Modelos ML
- Prepara features automaticamente dos dados históricos
- Testa 6 algoritmos diferentes de Machine Learning
- Compara performance usando:
  - Acurácia no conjunto de teste
  - Cross-validation scores
  - Matriz de confusão
  - Relatório de classificação
- **Seleciona automaticamente o melhor modelo**
- Salva o melhor modelo para uso futuro

## 🎯 Vantagens da Abordagem com Scikit-Learn

### Por que testar múltiplos modelos?

1. **Descoberta Automática**: Não precisa adivinhar qual algoritmo funciona melhor
2. **Comparação Objetiva**: Métricas claras para cada modelo
3. **Facilidade**: Scikit-learn facilita testar vários algoritmos rapidamente
4. **Flexibilidade**: Fácil adicionar novos modelos ou features

### Modelos Testados

| Modelo | Características | Quando Usar |
|--------|----------------|-------------|
| **Logistic Regression** | Simples, rápido, interpretável | Baseline, relações lineares |
| **Random Forest** | Robusto, lida bem com não-linearidades | Boa escolha geral |
| **Gradient Boosting** | Alta performance, captura padrões complexos | Quando precisar de máxima acurácia |
| **SVM** | Bom para dados de alta dimensão | Problemas de classificação complexos |
| **K-Nearest Neighbors** | Simples, não-paramétrico | Dados com padrões locais |
| **Naive Bayes** | Rápido, funciona bem com poucos dados | Baseline probabilístico |

## 🔧 Personalizações

### Adicionar Mais Features

Edite a função `create_features()` no notebook para incluir:
- Rankings FIFA
- Forma recente (últimos 5 jogos)
- Estatísticas de jogadores
- Condições do jogo (clima, estádio, etc.)

### Adicionar Mais Modelos

```python
from sklearn.neural_network import MLPClassifier
from xgboost import XGBClassifier

models['Neural Network'] = MLPClassifier(hidden_layer_sizes=(100, 50))
models['XGBoost'] = XGBClassifier(n_estimators=100)
```

### Otimizar Hiperparâmetros

```python
from sklearn.model_selection import GridSearchCV

param_grid = {
    'n_estimators': [50, 100, 200],
    'max_depth': [5, 10, 15]
}

grid_search = GridSearchCV(RandomForestClassifier(), param_grid, cv=5)
grid_search.fit(X_train, y_train)
best_model = grid_search.best_estimator_
```

## 📈 Próximos Passos

1. **Feature Engineering**: Criar features mais sofisticadas
2. **Ensemble de Ensembles**: Combinar modelo customizado com ML
3. **Validação Temporal**: Testar em dados cronológicos
4. **Deep Learning**: Experimentar redes neurais para padrões mais complexos
5. **AutoML**: Usar bibliotecas como Auto-sklearn ou TPOT para otimização automática

## 🤝 Contribuindo

Para adicionar novos modelos ou melhorias:
1. Faça suas modificações no notebook
2. Documente os resultados
3. Atualize este README

## 📚 Recursos Adicionais

- [Scikit-learn Documentation](https://scikit-learn.org/)
- [Kaggle: FIFA World Cup Dataset](https://www.kaggle.com/datasets/abecklas/fifa-world-cup)
- [Football Prediction Models](https://www.pinnacle.com/en/betting-articles/Soccer/how-to-calculate-poisson-distribution/)

## ❓ FAQ

**P: Qual modelo devo usar em produção?**
R: O notebook seleciona automaticamente o melhor baseado em acurácia. Considere também interpretabilidade e velocidade.

**P: Por que não usar apenas o modelo customizado?**
R: O modelo customizado (ELO + Poisson) é excelente e interpretável, mas testar ML pode revelar padrões que não capturamos manualmente.

**P: Posso combinar ambas abordagens?**
R: Sim! Você pode usar as predições do modelo customizado como features para os modelos de ML (stacking).

**P: Como adicionar dados mais recentes?**
R: Atualize o arquivo `data/raw/WorldCupMatches.csv` e re-execute o notebook.