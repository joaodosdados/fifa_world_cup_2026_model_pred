"""Analysis page driven by the model currently selected in the sidebar."""

from __future__ import annotations

import numpy as np
import pandas as pd
import streamlit as st

from src.data.loader import DataLoader


FEATURE_NAMES = [
    "Gols marcados · mandante",
    "Gols sofridos · mandante",
    "Taxa de vitórias · mandante",
    "Gols marcados · visitante",
    "Gols sofridos · visitante",
    "Taxa de vitórias · visitante",
    "Vantagem de mando",
]


def metric_value(metrics, *names, default=None):
    for name in names:
        if name in metrics:
            return metrics[name]
    return default


def show_metadata(model):
    metrics = model["metrics"]
    accuracy = metric_value(metrics, "accuracy")
    validation = metric_value(metrics, "cv_mean", "cv_score")
    validation_std = metric_value(metrics, "cv_std")

    columns = st.columns(3)
    columns[0].metric(
        "Acurácia registrada",
        f"{accuracy:.1%}" if accuracy is not None else "Não informada",
    )
    columns[1].metric(
        "Validação cruzada",
        f"{validation:.1%}" if validation is not None else "Não informada",
    )
    columns[2].metric(
        "Desvio da validação",
        f"{validation_std:.3f}" if validation_std is not None else "Não informado",
    )

    st.info(model["description"])


def show_feature_values(title, values):
    values = np.asarray(values, dtype=float).reshape(-1)
    if len(values) != len(FEATURE_NAMES):
        st.warning("O estimador não expõe valores compatíveis com as sete features.")
        return

    frame = pd.DataFrame({"Feature": FEATURE_NAMES, "Valor": values})
    frame["Magnitude"] = frame["Valor"].abs()
    frame = frame.sort_values("Magnitude", ascending=False).drop(columns="Magnitude")
    st.markdown(f"#### {title}")
    st.dataframe(frame, width="stretch", hide_index=True)


def show_estimator_analysis(estimator):
    if type(estimator).__name__ == "Pipeline":
        steps = list(estimator.named_steps.values())
        preprocessing = [type(step).__name__ for step in steps[:-1]]
        estimator = steps[-1]
        if preprocessing:
            st.caption(
                "Pré-processamento: " + " → ".join(preprocessing)
            )

    estimator_name = type(estimator).__name__
    st.markdown(f"### Heurística: {estimator_name}")

    if estimator_name == "VotingClassifier":
        voting = getattr(estimator, "voting", "hard")
        components = [
            {
                "Identificador": name,
                "Estimador": type(component).__name__,
            }
            for name, component in getattr(estimator, "estimators", [])
        ]
        st.write(
            "Combina as decisões dos estimadores abaixo por "
            f"**votação {voting}**. Cada componente enxerga as mesmas sete features."
        )
        st.dataframe(pd.DataFrame(components), width="stretch", hide_index=True)
        return

    if hasattr(estimator, "feature_importances_"):
        show_feature_values("Importância das features", estimator.feature_importances_)
        st.caption(
            "Valores maiores indicam features mais usadas nas divisões e decisões "
            "do modelo."
        )
        return

    if hasattr(estimator, "coef_"):
        coefficients = np.asarray(estimator.coef_)
        show_feature_values(
            "Magnitude média dos coeficientes",
            np.mean(np.abs(coefficients), axis=0),
        )
        st.caption(
            "A tabela resume a influência linear média de cada feature entre as classes."
        )
        return

    if estimator_name == "SVC":
        columns = st.columns(3)
        columns[0].metric("Kernel", getattr(estimator, "kernel", "—"))
        columns[1].metric(
            "Vetores de suporte",
            int(np.sum(getattr(estimator, "n_support_", [0]))),
        )
        columns[2].metric("C", getattr(estimator, "C", "—"))
        st.write(
            "O SVM separa os resultados no espaço de features usando os vetores "
            "de suporte mais próximos das fronteiras entre vitória, empate e derrota."
        )
        return

    if estimator_name == "CalibratedClassifierCV":
        base_estimator = getattr(estimator, "estimator", None)
        st.write(
            "O classificador base produz margens de decisão e uma calibração "
            "sigmoide as converte em probabilidades comparáveis."
        )
        if base_estimator is not None:
            columns = st.columns(3)
            columns[0].metric("Base", type(base_estimator).__name__)
            columns[1].metric("Calibração", getattr(estimator, "method", "—"))
            columns[2].metric("Folds internos", getattr(estimator, "cv", "—"))
        return

    if estimator_name == "GaussianNB":
        class_counts = getattr(estimator, "class_count_", [])
        classes = getattr(estimator, "classes_", [])
        frame = pd.DataFrame(
            {
                "Classe": classes,
                "Amostras": class_counts,
                "Prior": getattr(estimator, "class_prior_", [None] * len(classes)),
            }
        )
        st.write(
            "O Naive Bayes estima distribuições gaussianas por classe e combina "
            "as probabilidades assumindo independência condicional das features."
        )
        st.dataframe(frame, width="stretch", hide_index=True)
        return

    if estimator_name == "KNeighborsClassifier":
        columns = st.columns(3)
        columns[0].metric("Vizinhos", getattr(estimator, "n_neighbors", "—"))
        columns[1].metric("Pesos", getattr(estimator, "weights", "—"))
        columns[2].metric("Métrica", getattr(estimator, "metric", "—"))
        st.write(
            "A previsão é determinada pelos jogos historicamente mais próximos "
            "no espaço das sete features."
        )
        return

    st.warning(
        "Este estimador não publica uma heurística visual específica. "
        "As informações técnicas disponíveis são exibidas abaixo."
    )
    st.json(
        {
            "tipo": estimator_name,
            "possui_predict_proba": hasattr(estimator, "predict_proba"),
            "possui_decision_function": hasattr(estimator, "decision_function"),
            "parâmetros": (
                estimator.get_params(deep=False)
                if hasattr(estimator, "get_params")
                else {}
            ),
        }
    )


def show_statistical_analysis(model):
    predictor = model["predictor"]
    st.markdown("### Heurística: ELO + Poisson")

    columns = st.columns(3)
    columns[0].metric("Peso ELO", f"{predictor.elo_weight:.0%}")
    columns[1].metric("Peso Poisson", f"{predictor.poisson_weight:.0%}")
    columns[2].metric("Times avaliados", len(predictor.poisson_model.team_stats))

    st.write(
        "**ELO** atualiza a força relativa após cada partida. **Poisson** estima "
        "gols esperados a partir das forças de ataque e defesa. As probabilidades "
        "finais são a média ponderada das duas abordagens."
    )

    ratings = predictor.elo_model.get_all_ratings().head(20).rename(
        columns={"team": "Seleção", "elo_rating": "Rating ELO"}
    )
    st.markdown("#### Top 20 por ELO")
    st.dataframe(ratings, width="stretch", hide_index=True)

    team_stats = pd.DataFrame(
        [
            {
                "Seleção": team,
                "Ataque": stats["attack_strength"],
                "Defesa": stats["defense_strength"],
                "Jogos": stats["matches_played"],
            }
            for team, stats in predictor.poisson_model.team_stats.items()
        ]
    ).sort_values("Ataque", ascending=False)
    st.markdown("#### Forças estimadas por Poisson")
    st.dataframe(team_stats, width="stretch", hide_index=True, height=420)


def show_training_data():
    matches = DataLoader().load_matches(processed=False)
    columns = st.columns(3)
    columns[0].metric("Partidas válidas", f"{len(matches):,}")
    columns[1].metric(
        "Período",
        f"{int(matches['Year'].min())}–{int(matches['Year'].max())}",
    )
    teams = pd.concat(
        [matches["Home Team Name"], matches["Away Team Name"]]
    ).nunique()
    columns[2].metric("Seleções", teams)

    st.caption(
        "Linhas vazias, times ausentes, gols inválidos e duplicatas são removidos "
        "antes do treinamento e das análises."
    )
    st.dataframe(
        matches[
            [
                "Year",
                "Home Team Name",
                "Away Team Name",
                "Home Team Goals",
                "Away Team Goals",
            ]
        ].head(20),
        width="stretch",
        hide_index=True,
    )


active_model = st.session_state.get("active_model")

if not active_model:
    st.error("Nenhum modelo ativo.")
else:
    st.markdown(f"## {active_model['label']}")
    overview_tab, heuristic_tab, data_tab = st.tabs(
        ["Visão geral", "Heurística e diagnóstico", "Dados"]
    )

    with overview_tab:
        show_metadata(active_model)
        live_accuracy = st.session_state.get("accuracy_stats")
        if live_accuracy:
            st.markdown("### Desempenho nos resultados de 2026")
            columns = st.columns(3)
            columns[0].metric("Acurácia", f"{live_accuracy['accuracy']:.1f}%")
            columns[1].metric("Acertos", live_accuracy["correct"])
            columns[2].metric("Partidas", live_accuracy["total"])

    with heuristic_tab:
        if active_model["kind"] == "statistical":
            show_statistical_analysis(active_model)
        else:
            show_estimator_analysis(active_model["estimator"])

    with data_tab:
        show_training_data()
