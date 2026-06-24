# Contribuição

Antes de enviar mudanças:

```bash
pip install -r requirements-dev.txt
python scripts/train_models.py --output-dir /tmp/bolao-models-smoke --cv-folds 3
python -m compileall -q main.py src scripts tests
python -m pytest -q
```

Regras do projeto:

- mantenha apenas um caminho de scraping;
- nunca grave dados de fallback como resultados oficiais;
- adicione novos modelos ao `ModelManager`; o catálogo os descobrirá;
- análises específicas devem usar capacidades reais do estimador;
- páginas Streamlit vivem em `src/app/pages/`;
- componentes visuais vivem em `src/app/components/`;
- features pré-jogo vivem em `src/features/`;
- não versione caches, saídas de treinamento ou modelos duplicados de notebooks.
