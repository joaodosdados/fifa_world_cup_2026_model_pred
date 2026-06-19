"""Project information."""

import streamlit as st


st.markdown("## Sobre o projeto")

developer_col, description_col = st.columns([1, 3])

with developer_col:
    st.image("assets/profile.png", width=150)

with description_col:
    st.markdown(
        """
        **João "dos Dados" Oliveira**

        Data-driven professional with over 6 years of experience building scalable machine learning and AI solutions across media, mining, retail, telecommunications, and enterprise environments..

        [GitHub](https://github.com/joaodosdados) ·
        [LinkedIn](https://www.linkedin.com/in/joaodosdados/)
        """
    )

st.markdown("---")

with st.expander("Como o dashboard funciona", expanded=True):
    st.markdown(
        """
        - O calendário é atualizado pela página oficial da FIFA ao abrir a sessão.
        - O seletor da barra lateral troca imediatamente o preditor ativo.
        - As previsões, a acurácia de 2026 e a página **Model Analysis** usam
          sempre o mesmo modelo selecionado.
        - Os dados históricos são sanitizados antes de qualquer treinamento ou
          análise.
        """
    )

with st.expander("Modelos"):
    st.markdown(
        """
        O catálogo inclui os artefatos scikit-learn compatíveis encontrados em
        `models/` e o modelo estatístico **ELO + Poisson**.

        Cada estimador possui uma análise própria baseada nas capacidades que
        realmente expõe: coeficientes, importância de features, vetores de
        suporte, vizinhos, distribuições por classe ou componentes de votação.
        """
    )

with st.expander("Dados e atualização"):
    st.markdown(
        """
        - Histórico: dataset público de Copas do Mundo, 1930–2014.
        - Resultados de 2026: página oficial da FIFA.
        - RPA: Selenium + BeautifulSoup.
        - Falhas no scraping nunca geram resultados de fallback no CSV.
        """
    )
