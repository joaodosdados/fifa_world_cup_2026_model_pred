"""Analysis page driven by the model currently selected in the sidebar."""

from __future__ import annotations

import numpy as np
import pandas as pd
import streamlit as st
from sklearn.inspection import permutation_importance

from src.data.loader import DataLoader
from src.features.match_features import FEATURE_NAMES as MODEL_FEATURE_NAMES
from src.features.match_features import build_temporal_training_data


FEATURE_LABELS = {
    "home_goals_scored": "Gols marcados · mandante",
    "home_goals_conceded": "Gols sofridos · mandante",
    "home_win_rate": "Taxa de vitórias · mandante",
    "home_matches_played": "Experiência histórica · mandante",
    "home_recent_points": "Forma recente · mandante",
    "home_recent_goal_diff": "Saldo recente · mandante",
    "home_elo": "ELO pré-jogo · mandante",
    "away_goals_scored": "Gols marcados · visitante",
    "away_goals_conceded": "Gols sofridos · visitante",
    "away_win_rate": "Taxa de vitórias · visitante",
    "away_matches_played": "Experiência histórica · visitante",
    "away_recent_points": "Forma recente · visitante",
    "away_recent_goal_diff": "Saldo recente · visitante",
    "away_elo": "ELO pré-jogo · visitante",
    "elo_diff": "Diferença de ELO",
    "home_advantage": "Vantagem de mando",
    "neutral_site": "Campo neutro",
    "tournament_importance": "Importância do torneio",
}
FEATURE_NAMES = [FEATURE_LABELS.get(name, name) for name in MODEL_FEATURE_NAMES]

FEATURE_GROUPS = {
    "Médias históricas": [
        "home_goals_scored",
        "home_goals_conceded",
        "home_win_rate",
        "away_goals_scored",
        "away_goals_conceded",
        "away_win_rate",
    ],
    "Forma recente": [
        "home_recent_points",
        "home_recent_goal_diff",
        "away_recent_points",
        "away_recent_goal_diff",
    ],
    "Força relativa": [
        "home_elo",
        "away_elo",
        "elo_diff",
    ],
    "Contexto da partida": [
        "home_matches_played",
        "away_matches_played",
        "home_advantage",
        "neutral_site",
        "tournament_importance",
    ],
}


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
    goals_mae = metric_value(metrics, "goals_mae")

    columns = st.columns(4)
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
    columns[3].metric(
        "MAE gols",
        f"{goals_mae:.2f}" if goals_mae is not None else "Não informado",
    )

    st.info(model["description"])


def show_probability_metrics(model):
    metrics = model["metrics"]
    log_loss_value = metric_value(metrics, "log_loss")
    brier = metric_value(metrics, "brier_score")
    ece = metric_value(metrics, "expected_calibration_error")

    st.markdown("### Qualidade das probabilidades")
    columns = st.columns(3)
    columns[0].metric(
        "Log Loss",
        f"{log_loss_value:.3f}" if log_loss_value is not None else "Não informado",
        help="Penaliza previsões confiantes erradas. Quanto menor, melhor.",
    )
    columns[1].metric(
        "Brier Score",
        f"{brier:.3f}" if brier is not None else "Não informado",
        help="Erro quadrático das probabilidades multi-classe. Quanto menor, melhor.",
    )
    columns[2].metric(
        "ECE",
        f"{ece:.3f}" if ece is not None else "Não informado",
        help="Expected Calibration Error: distância entre confiança e acurácia observada.",
    )

    if log_loss_value is None and brier is None and ece is None:
        st.caption(
            "Este modelo não possui métricas probabilísticas registradas. "
            "Reexecute `python scripts/train_models.py` para gerá-las."
        )


def training_matches() -> pd.DataFrame:
    """Return the same training frame used by the runtime models."""
    matches = st.session_state.get("training_matches")
    if matches is not None:
        return matches

    loader = DataLoader()
    try:
        return loader.load_international_matches(
            min_year=2010,
            max_date="2026-06-10",
            min_team_matches=30,
        )
    except FileNotFoundError:
        return loader.load_matches(processed=False)


def parsed_match_dates(matches: pd.DataFrame) -> pd.Series:
    if "Datetime" not in matches:
        return pd.Series(dtype="datetime64[ns]")
    return pd.to_datetime(matches["Datetime"], errors="coerce")


def show_training_setup(model):
    """Explain the data contract behind the selected model."""
    metrics = model.get("metrics", {})
    matches = training_matches()
    dates = parsed_match_dates(matches)

    data_source = metrics.get("data_source", "runtime")
    cutoff = metrics.get("cutoff_date", "2026-06-10")
    min_year = metrics.get("min_year", int(matches["Year"].min()))
    min_team_matches = metrics.get("min_team_matches", 30)
    leakage_flag = metrics.get("includes_current_2026_results", False)

    st.markdown("### Dados e recorte do treino")
    columns = st.columns(5)
    columns[0].metric("Fonte", str(data_source))
    columns[1].metric("Partidas", f"{len(matches):,}")
    columns[2].metric("Features", len(MODEL_FEATURE_NAMES))
    columns[3].metric("Leakage 2026", "Sim" if leakage_flag else "Não")
    columns[4].metric("Regressor gols", str(metrics.get("goal_model", "—")))

    st.caption(
        "O modelo padrão é treinado apenas com dados disponíveis antes da Copa "
        f"2026: ano mínimo `{min_year}`, cutoff `{cutoff}`, e pelo menos "
        f"`{min_team_matches}` partidas por seleção no recorte internacional."
    )

    if not dates.dropna().empty:
        st.caption(
            "Janela efetiva carregada no app: "
            f"{dates.min():%d/%m/%Y} até {dates.max():%d/%m/%Y}."
        )

    if leakage_flag:
        st.warning(
            "Este artefato foi treinado incluindo resultados já concluídos de "
            "2026. Não use a acurácia nesses mesmos jogos como avaliação honesta."
        )
    else:
        st.success(
            "Os resultados já concluídos da Copa 2026 ficam fora do treino e "
            "são usados apenas para avaliação live/exibição."
        )


def show_feature_catalog():
    st.markdown("### Catálogo de features")
    rows = []
    for group, features in FEATURE_GROUPS.items():
        for feature in features:
            rows.append(
                {
                    "Grupo": group,
                    "Feature técnica": feature,
                    "Descrição": FEATURE_LABELS.get(feature, feature),
                }
            )
    st.dataframe(pd.DataFrame(rows), width="stretch", hide_index=True)


def show_calibration_bins(model):
    bins = model.get("metrics", {}).get("calibration_bins", [])
    st.markdown("### Calibração por faixa de confiança")
    if not bins:
        st.info(
            "Este modelo não possui bins de calibração registrados. "
            "Modelos estatísticos como ELO + Poisson ainda não publicam essa análise."
        )
        return

    frame = pd.DataFrame(bins)
    display = frame.rename(
        columns={
            "bin": "Faixa",
            "samples": "Amostras",
            "confidence": "Confiança média",
            "accuracy": "Acurácia observada",
            "gap": "Gap",
        }
    )[["Faixa", "Amostras", "Confiança média", "Acurácia observada", "Gap"]]

    st.caption(
        "Um modelo bem calibrado fica perto da diagonal: quando prevê 70% de "
        "confiança, deveria acertar aproximadamente 70% dos jogos naquela faixa."
    )
    chart_frame = display.set_index("Faixa")[
        ["Confiança média", "Acurácia observada"]
    ]
    st.line_chart(chart_frame, height=300)
    st.dataframe(display, width="stretch", hide_index=True)


def show_feature_values(title, values):
    values = np.asarray(values, dtype=float).reshape(-1)
    if len(values) != len(FEATURE_NAMES):
        st.warning(
            "O estimador não expõe valores compatíveis com o conjunto atual "
            "de features."
        )
        return

    frame = pd.DataFrame({"Feature": FEATURE_NAMES, "Valor": values})
    frame["Magnitude"] = frame["Valor"].abs()
    frame = frame.sort_values("Magnitude", ascending=False).drop(columns="Magnitude")
    st.markdown(f"#### {title}")
    st.dataframe(frame, width="stretch", hide_index=True)


def unwrap_pipeline(estimator):
    """Return the final estimator and a human-readable preprocessing chain."""
    if type(estimator).__name__ != "Pipeline":
        return estimator, []
    steps = list(estimator.named_steps.values())
    return steps[-1], [type(step).__name__ for step in steps[:-1]]


def outcome_estimator(estimator):
    """Return the classifier inside a MatchModelBundle, if present."""
    return getattr(estimator, "outcome_model", estimator)


def goal_estimator(estimator):
    """Return the goal regressor inside a MatchModelBundle, if present."""
    return getattr(estimator, "goals_model", None)


def make_importance_frame(
    values,
    *,
    method: str,
    signed_values=None,
    std_values=None,
) -> pd.DataFrame | None:
    values = np.asarray(values, dtype=float).reshape(-1)
    if len(values) != len(FEATURE_NAMES):
        return None

    frame = pd.DataFrame({
        "Feature": FEATURE_NAMES,
        "Importância": values,
        "Método": method,
    })
    if signed_values is not None:
        signed_values = np.asarray(signed_values, dtype=float).reshape(-1)
        if len(signed_values) == len(FEATURE_NAMES):
            frame["Sinal médio"] = signed_values
    if std_values is not None:
        std_values = np.asarray(std_values, dtype=float).reshape(-1)
        if len(std_values) == len(FEATURE_NAMES):
            frame["Desvio"] = std_values

    frame["Importância"] = frame["Importância"].astype(float)
    return frame.sort_values("Importância", ascending=False).reset_index(drop=True)


def native_feature_importance(estimator) -> tuple[pd.DataFrame | None, str]:
    """
    Try estimator-native importance first.

    Tree models expose feature_importances_; linear models expose coefficients.
    Pipeline estimators are unwrapped only for native inspection, while
    permutation importance uses the full pipeline.
    """
    final_estimator, preprocessing = unwrap_pipeline(estimator)
    preprocessing_note = (
        "Pré-processamento: " + " → ".join(preprocessing)
        if preprocessing
        else ""
    )

    if hasattr(final_estimator, "feature_importances_"):
        frame = make_importance_frame(
            final_estimator.feature_importances_,
            method="Importância nativa do estimador",
        )
        return frame, preprocessing_note

    if hasattr(final_estimator, "coef_"):
        coefficients = np.asarray(final_estimator.coef_, dtype=float)
        frame = make_importance_frame(
            np.mean(np.abs(coefficients), axis=0),
            method="Magnitude média dos coeficientes",
            signed_values=np.mean(coefficients, axis=0),
        )
        return frame, preprocessing_note

    return None, preprocessing_note


def voting_feature_importance(estimator) -> pd.DataFrame | None:
    """Aggregate importances from fitted components of a VotingClassifier."""
    fitted_estimators = getattr(estimator, "estimators_", None)
    configured_estimators = getattr(estimator, "estimators", [])
    if not fitted_estimators or not configured_estimators:
        return None

    frames = []
    names = [name for name, _ in configured_estimators]
    for name, component in zip(names, fitted_estimators):
        frame, _ = native_feature_importance(component)
        if frame is None:
            continue
        component_frame = frame[["Feature", "Importância"]].copy()
        total = component_frame["Importância"].abs().sum()
        if total > 0:
            component_frame["Importância"] = component_frame["Importância"] / total
        component_frame["Componente"] = name
        frames.append(component_frame)

    if not frames:
        return None

    combined = pd.concat(frames, ignore_index=True)
    aggregated = (
        combined.groupby("Feature", as_index=False)["Importância"]
        .mean()
        .sort_values("Importância", ascending=False)
        .reset_index(drop=True)
    )
    aggregated["Método"] = "Média normalizada dos componentes do ensemble"
    return aggregated


def permutation_feature_importance(estimator, matches: pd.DataFrame) -> pd.DataFrame | None:
    """Model-agnostic fallback for estimators without native importance."""
    if matches is None or matches.empty:
        return None

    X, y, _ = build_temporal_training_data(matches)
    if len(X) < 30:
        return None

    sample_size = min(350, len(X))
    X_sample = X[-sample_size:]
    y_sample = y[-sample_size:]
    result = permutation_importance(
        estimator,
        X_sample,
        y_sample,
        n_repeats=8,
        random_state=42,
        scoring="accuracy",
        n_jobs=1,
    )
    values = np.maximum(result.importances_mean, 0)
    return make_importance_frame(
        values,
        method=f"Permutation importance nos {sample_size} jogos mais recentes",
        std_values=result.importances_std,
    )


def render_importance_frame(frame: pd.DataFrame, explanation: str) -> None:
    if frame is None or frame.empty:
        st.warning(
            "Não foi possível calcular uma importância de features confiável "
            "para este modelo."
        )
        return

    st.caption(explanation)
    chart_data = frame[["Feature", "Importância"]].set_index("Feature")
    st.bar_chart(chart_data, height=320)
    st.dataframe(frame, width="stretch", hide_index=True)


def show_model_feature_importance(model):
    """Display model-specific feature importance in the selected model context."""
    st.markdown("### Feature importance do modelo selecionado")

    if model["kind"] == "statistical":
        show_statistical_feature_importance(model)
        return

    estimator = outcome_estimator(model["estimator"])
    estimator_name = type(estimator).__name__
    matches = st.session_state.get("training_matches")

    if estimator_name == "VotingClassifier":
        frame = voting_feature_importance(estimator)
        if frame is not None:
            render_importance_frame(
                frame,
                "O ensemble combina modelos base. A importância abaixo é a média "
                "normalizada das importâncias nativas disponíveis nos componentes.",
            )
            return

    frame, preprocessing_note = native_feature_importance(estimator)
    if frame is not None:
        if preprocessing_note:
            st.caption(preprocessing_note)
        render_importance_frame(
            frame,
            "Esta importância vem diretamente da estrutura do estimador treinado.",
        )
        return

    frame = permutation_feature_importance(estimator, matches)
    render_importance_frame(
        frame,
        "Este modelo não publica importância nativa. Por isso, a tabela usa "
        "permutation importance: cada feature é embaralhada e medimos quanto a "
        "acurácia cai. Quedas maiores indicam maior dependência do modelo.",
    )


def show_statistical_feature_importance(model):
    predictor = model["predictor"]
    st.info(
        "ELO + Poisson não usa as features tabulares dos modelos ML. Ele é um modelo "
        "estatístico: ELO mede força relativa; Poisson mede gols esperados a "
        "partir de ataque e defesa."
    )

    frame = pd.DataFrame(
        [
            {
                "Fator": "Rating ELO relativo",
                "Peso/uso": f"{predictor.elo_weight:.0%}",
                "Interpretação": "Times com ELO maior tendem a receber maior probabilidade de vitória.",
            },
            {
                "Fator": "Força de ataque Poisson",
                "Peso/uso": f"{predictor.poisson_weight:.0%}",
                "Interpretação": "Aumenta os gols esperados do time.",
            },
            {
                "Fator": "Força de defesa Poisson",
                "Peso/uso": f"{predictor.poisson_weight:.0%}",
                "Interpretação": "Reduz os gols esperados do adversário quando a defesa é forte.",
            },
            {
                "Fator": "Vantagem de mando",
                "Peso/uso": "Aplicada internamente",
                "Interpretação": "Ajusta expectativas/probabilidades quando há mandante definido.",
            },
        ]
    )
    st.dataframe(frame, width="stretch", hide_index=True)


def show_estimator_analysis(estimator):
    goals_model = goal_estimator(estimator)
    if goals_model is not None:
        st.markdown("### Arquitetura: resultado + gols")
        cols = st.columns(2)
        cols[0].metric("Modelo de resultado", type(outcome_estimator(estimator)).__name__)
        cols[1].metric("Modelo de gols", type(goals_model).__name__)
        st.write(
            "Este artefato separa duas tarefas: um classificador prevê "
            "**vitória/empate/derrota** e um regressor multi-output prevê "
            "**gols esperados do mandante e visitante**."
        )
        estimator = outcome_estimator(estimator)

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
            f"**votação {voting}**. Cada componente enxerga o mesmo conjunto "
            "de features tabulares."
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
            "no espaço das features tabulares."
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
    matches = training_matches()
    dates = parsed_match_dates(matches)
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
        "antes do treinamento e das análises. O recorte padrão termina em "
        "`2026-06-10` para não vazar jogos da Copa 2026 na avaliação live."
    )

    if not dates.dropna().empty:
        st.caption(
            f"Data mínima/máxima no treino: {dates.min():%d/%m/%Y} → "
            f"{dates.max():%d/%m/%Y}."
        )

    if "Tournament" in matches:
        st.markdown("#### Principais competições no treino")
        tournament_counts = (
            matches["Tournament"]
            .fillna("Unknown")
            .value_counts()
            .head(12)
            .rename_axis("Competição")
            .reset_index(name="Partidas")
        )
        st.dataframe(tournament_counts, width="stretch", hide_index=True)

    show_feature_catalog()

    st.markdown("#### Amostra dos dados usados")
    preview_columns = [
        column
        for column in [
            "Datetime",
            "Year",
            "Tournament",
            "Home Team Name",
            "Away Team Name",
            "Home Team Goals",
            "Away Team Goals",
            "Neutral",
        ]
        if column in matches.columns
    ]
    st.dataframe(
        matches[preview_columns].tail(20),
        width="stretch",
        hide_index=True,
    )


active_model = st.session_state.get("active_model")

if not active_model:
    st.error("Nenhum modelo ativo.")
else:
    st.markdown(f"## {active_model['label']}")
    overview_tab, heuristic_tab, importance_tab, calibration_tab, data_tab = st.tabs(
        [
            "Visão geral",
            "Heurística e diagnóstico",
            "Feature importance",
            "Calibração",
            "Dados",
        ]
    )

    with overview_tab:
        show_metadata(active_model)
        show_training_setup(active_model)
        show_probability_metrics(active_model)
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

    with importance_tab:
        show_model_feature_importance(active_model)

    with calibration_tab:
        show_probability_metrics(active_model)
        show_calibration_bins(active_model)

    with data_tab:
        show_training_data()
