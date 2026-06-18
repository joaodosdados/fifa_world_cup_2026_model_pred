# Instruções para Treinar Modelos Avançados

## 📋 Pré-requisitos

Antes de executar o notebook `model_training_analysis.ipynb`, você precisa instalar as bibliotecas de modelos avançados:

```bash
# Instalar XGBoost
pip install xgboost

# Instalar LightGBM
pip install lightgbm

# Instalar CatBoost
pip install catboost
```

## 🚀 Como Executar

1. **Abra o Jupyter Notebook:**
   ```bash
   cd notebooks
   jupyter notebook model_training_analysis.ipynb
   ```

2. **Execute todas as células em ordem:**
   - Seção 1-4: Carregamento e análise dos dados
   - Seção 5: Treinamento dos 6 modelos sklearn básicos
   - **Seção 6: Treinamento dos modelos avançados (XGBoost, LightGBM, CatBoost)**
   - **Seção 7: Criação do Ensemble Voting com os 3 melhores modelos**
   - **Seção 8: Salvamento do melhor modelo para uso no dashboard**

## 📊 O que o Notebook Faz

### Modelos Testados:
1. **Modelos Básicos (sklearn):**
   - Logistic Regression
   - Random Forest
   - Gradient Boosting
   - SVM (atual melhor: 93.3%)
   - K-Nearest Neighbors
   - Naive Bayes

2. **Modelos Avançados (novos):**
   - **XGBoost**: Gradient boosting otimizado
   - **LightGBM**: Gradient boosting rápido e eficiente
   - **CatBoost**: Gradient boosting com tratamento automático de categorias

3. **Ensemble Voting:**
   - Combina os 3 melhores modelos
   - Usa votação por maioria
   - Pode superar modelos individuais

### Features Utilizadas (7):
1. Goals scored home (média histórica)
2. Goals conceded home (média histórica)
3. Win rate home (taxa de vitórias)
4. Goals scored away (média histórica)
5. Goals conceded away (média histórica)
6. Win rate away (taxa de vitórias)
7. Home advantage (vantagem de jogar em casa)

## 🎯 Objetivo

Encontrar o modelo com **maior acurácia** que supere o SVM atual (93.3%) e salvá-lo automaticamente para uso no dashboard.

## 📈 Resultados Esperados

Após executar o notebook, você verá:

1. **Comparação de todos os modelos** (básicos + avançados + ensemble)
2. **Melhor modelo identificado** automaticamente
3. **Modelo salvo** em `models/` com metadados completos
4. **Visualizações** de performance e matriz de confusão

## 🔄 Integração com Dashboard

O modelo salvo será automaticamente:
- Detectado pelo `ModelManager`
- Carregado no dashboard se for melhor que o atual
- Disponível para seleção no sidebar
- Usado para predições da Copa 2026

## ⚠️ Notas Importantes

- O treinamento pode levar alguns minutos
- XGBoost, LightGBM e CatBoost são opcionais (o notebook funciona sem eles)
- Se alguma biblioteca não estiver instalada, o notebook pula esse modelo
- O ensemble só é criado se houver pelo menos 3 modelos treinados
- Todos os modelos são avaliados com cross-validation (5-fold)

## 🐛 Troubleshooting

### Erro: "No match data found"
```bash
# Certifique-se de que os dados estão em data/raw/
ls data/raw/WorldCupMatches.csv
```

### Erro: "Module not found: xgboost/lightgbm/catboost"
```bash
# Instale a biblioteca faltante
pip install xgboost lightgbm catboost
```

### Erro: "Cannot import ModelManager"
```bash
# Execute o notebook a partir da pasta notebooks/
cd notebooks
jupyter notebook
```

## 📝 Próximos Passos

Após treinar o modelo:

1. Verifique o arquivo salvo em `models/`
2. Execute o dashboard: `streamlit run main.py`
3. Selecione o novo modelo no sidebar
4. Compare a acurácia com o modelo anterior
5. Use o novo modelo para predições da Copa 2026!

---

**Boa sorte com o treinamento! 🚀⚽**