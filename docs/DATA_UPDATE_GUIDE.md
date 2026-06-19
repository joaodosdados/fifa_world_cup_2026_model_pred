# Atualização dos resultados

O dashboard consulta:

`https://www.fifa.com/pt/tournaments/mens/worldcup/canadamexicousa2026/scores-fixtures?country=BR&wtw-filter=ALL`

## Comportamento

- Uma atualização é executada ao abrir cada nova sessão do Streamlit.
- Interações comuns e trocas de modelo não abrem novamente o navegador.
- O botão **↻ Atualizar FIFA** força uma nova consulta.
- Falhas de rede ou mudanças no HTML não alteram o CSV.
- Apenas partidas com status finalizado e placares válidos são aceitas.

O sidebar informa a hora da última consulta, partidas finalizadas encontradas e
quantas foram conciliadas com o calendário local.

## Requisitos

Selenium precisa encontrar Google Chrome ou Chromium no ambiente. Em
deployments conteinerizados, instale o navegador no sistema operacional além das
dependências de `requirements.txt`.

## Diagnóstico

Execute o teste do parser:

```bash
python -m pytest tests/test_fifa_scraper.py
```

Para testar a consulta real:

```bash
python -c "from src.data.auto_updater import run_auto_update; print(run_auto_update())"
```
