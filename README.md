# FIFA World Cup 2026 Prediction Dashboard

Dashboard Streamlit para acompanhar resultados e comparar previsões de modelos
de machine learning e do ensemble estatístico ELO + Poisson.

## Execução

Requisitos:

- Python 3.10+
- Google Chrome ou Chromium, usado pelo RPA da FIFA

```bash
python -m venv .venv
source .venv/bin/activate
pip install -r requirements.txt
streamlit run main.py
```

Ao abrir uma nova sessão, o aplicativo consulta a página oficial da FIFA e
atualiza `data/2026_world_cup_schedule.csv`. O botão **↻ Atualizar FIFA** permite
repetir a consulta manualmente sem reiniciar o dashboard.

## Arquitetura

- `main.py`: composição do aplicativo, estado da sessão e sidebar.
- `app_pages/`: chaveamento, grupos, análise dinâmica e informações.
- `src/data/`: carga sanitizada dos dados, RPA Selenium e reconciliação do CSV.
- `src/models/`: catálogo, adaptadores e modelos estatísticos.
- `src/components/`: componentes visuais reutilizáveis.
- `tests/`: testes automatizados do parser, dados e modelos.

## Modelos

O seletor mostra todos os artefatos disponíveis em `models/`, além de
**ELO + Poisson**. Ao trocar de modelo:

1. o preditor ativo muda imediatamente;
2. todas as partidas são recalculadas;
3. a acurácia de 2026 é recalculada;
4. a página **Model Analysis** passa a apresentar a heurística e os diagnósticos
   específicos daquele estimador.

O dataset histórico é sanitizado antes do uso: linhas vazias, resultados
inválidos e duplicatas são removidos.

## Treinamento dos modelos

O treinamento não depende de Jupyter, JupyterLab ou configuração de kernel.
Com o ambiente virtual ativo, execute:

```bash
python scripts/train_models.py
```

O pipeline:

1. carrega e sanitiza o histórico;
2. cria features pré-jogo usando apenas partidas anteriores;
3. reserva os 20% mais recentes para teste temporal;
4. treina Logistic Regression, Random Forest, Gradient Boosting, SVM, KNN e
   Naive Bayes;
5. cria um ensemble de votação com os três melhores;
6. calcula holdout, validação temporal, matriz de confusão e relatório por
   classe;
7. retreina os modelos com todo o histórico e publica os arquivos em `models/`
   de forma atômica.

Opções disponíveis:

```bash
python scripts/train_models.py --help
python scripts/train_models.py --test-size 0.25 --cv-folds 4
python scripts/train_models.py --no-ensemble
```

Depois do treinamento, reinicie o Streamlit para limpar o cache de recursos e
carregar os novos artefatos.

As versões de NumPy e scikit-learn usadas para persistência são registradas em
`models/models_metadata.json`. O `requirements.txt` fixa a versão do
scikit-learn para evitar incompatibilidade ao carregar arquivos pickle.

Para desenvolvimento e testes:

```bash
pip install -r requirements-dev.txt
pytest -q
```

## Atualização FIFA

O único scraper mantido é `src/data/fifa_scraper_selenium.py`. Ele:

- renderiza a página dinâmica da FIFA;
- extrai somente cartões com status finalizado;
- não utiliza resultados fictícios em caso de falha;
- concilia nomes e mando invertido;
- salva o CSV de forma atômica.

Consulte `docs/DATA_UPDATE_GUIDE.md` para detalhes operacionais.
