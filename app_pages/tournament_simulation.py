"""Monte Carlo tournament simulation page."""

from __future__ import annotations

import pandas as pd
import streamlit as st

from src.simulation.tournament_simulator import KNOCKOUT_STAGES, TournamentSimulator


predictor = st.session_state.get("predictor")
schedule = st.session_state.get("schedule_2026")
predictions = st.session_state.get("schedule_predictions")
active_model = st.session_state.get("active_model", {})
active_model_id = st.session_state.get("active_model_id", "unknown")
schedule_version = st.session_state.get("schedule_version", "unknown")


STAGE_LABELS = {
    "round_of_32": "Rodada de 32",
    "round_of_16": "Oitavas",
    "quarter_finals": "Quartas",
    "semi_finals": "Semifinal",
    "final": "Final",
}


def simulation_cache_key(n_simulations: int, seed: int) -> tuple:
    """Stable cache key for Streamlit session cache."""
    return (
        active_model_id,
        schedule_version,
        int(n_simulations),
        int(seed),
    )


def run_or_get_simulation(n_simulations: int, seed: int) -> dict:
    """Run simulation once per model/schedule/settings combination."""
    cache = st.session_state.setdefault("tournament_simulation_cache", {})
    key = simulation_cache_key(n_simulations, seed)
    if key not in cache:
        cache[key] = TournamentSimulator(
            predictor=predictor,
            schedule=schedule,
            predictions=predictions,
            random_seed=seed,
        ).simulate(n_simulations=n_simulations)
    return cache[key]


def percentage_table(probabilities: pd.DataFrame) -> pd.DataFrame:
    """Return display-friendly probability table."""
    table = probabilities.copy()
    table = table.rename(
        columns={
            "team": "Seleção",
            "avg_group_points": "Pts médios",
            "pass_group": "Passa do grupo",
            "round_of_16": "Oitavas",
            "quarter_finals": "Quartas",
            "semi_finals": "Semifinal",
            "final": "Final",
            "title": "Título",
        }
    )
    for column in [
        "Passa do grupo",
        "Oitavas",
        "Quartas",
        "Semifinal",
        "Final",
        "Título",
    ]:
        table[column] = table[column].map(lambda value: f"{value:.1%}")
    table["Pts médios"] = table["Pts médios"].map(lambda value: f"{value:.2f}")
    return table


def matchup_table(matchups: pd.DataFrame) -> pd.DataFrame:
    """Return display-friendly knockout matchup table."""
    if matchups.empty:
        return matchups

    table = matchups.copy()
    table["Confronto"] = table["team_a"] + " x " + table["team_b"]
    table["Fase"] = table["stage"].map(STAGE_LABELS).fillna(table["stage"])
    table["Chance"] = table["probability"].map(lambda value: f"{value:.1%}")
    return table[["Fase", "Confronto", "Chance", "simulations"]].rename(
        columns={"simulations": "Simulações"}
    )


def path_table(paths: dict, selected_team: str, n_simulations: int) -> pd.DataFrame:
    """Build likely path rows for the selected team."""
    rows = []
    team_paths = paths.get(selected_team, {})
    for stage_key, stage_label in KNOCKOUT_STAGES:
        counter = team_paths.get(stage_key)
        if not counter:
            continue
        for opponent, count in counter.most_common(3):
            rows.append(
                {
                    "Fase": STAGE_LABELS.get(stage_key, stage_label),
                    "Possível adversário": opponent,
                    "Chance": f"{count / n_simulations:.1%}",
                }
            )
    return pd.DataFrame(rows)


def group_difficulty_table(group_difficulty: pd.DataFrame) -> pd.DataFrame:
    """Return display-friendly group difficulty table."""
    table = group_difficulty.copy()
    table["Grupo"] = "Grupo " + table["group"].astype(str)
    table["Pts médios do 3º"] = table["avg_third_place_points"].map(
        lambda value: f"{value:.2f}"
    )
    return table[["Grupo", "Pts médios do 3º"]]


def show_simulation_page():
    """Render tournament simulation page."""
    st.markdown("## 🎲 Simulação Monte Carlo da Copa")

    if predictor is None or schedule is None or predictions is None:
        st.error("Preditor, calendário ou previsões não estão carregados.")
        return

    st.caption(
        "A simulação usa resultados reais já finalizados e sorteia os jogos "
        "restantes com as probabilidades do modelo selecionado."
    )

    col_settings, col_note = st.columns([1, 2])
    with col_settings:
        n_simulations = st.select_slider(
            "Número de simulações",
            options=[1_000, 5_000, 10_000],
            value=10_000,
        )
        seed = st.number_input("Seed", min_value=1, value=42, step=1)
        force_refresh = st.button("Rodar novamente", width="stretch")

    with col_note:
        st.info(
            "Regras implementadas: 12 grupos de 4, top-2 + 8 melhores terceiros, "
            "Rodada de 32, Oitavas, Quartas, Semis e Final. Desempate de grupos "
            "usa pontos, confronto direto, saldo, gols marcados e sorteio quando "
            "faltam dados de fair play/ranking FIFA."
        )
        st.caption(f"Modelo ativo: {active_model.get('label', active_model_id)}")

    cache = st.session_state.setdefault("tournament_simulation_cache", {})
    key = simulation_cache_key(n_simulations, seed)
    if force_refresh and key in cache:
        del cache[key]

    with st.spinner(f"Simulando {n_simulations:,} Copas...".replace(",", ".")):
        result = run_or_get_simulation(n_simulations, seed)

    probabilities = result["probabilities"]
    champion = probabilities.iloc[0]
    hardest_group = result["group_difficulty"].iloc[0]

    metric_cols = st.columns(4)
    metric_cols[0].metric("Favorito ao título", champion["team"])
    metric_cols[1].metric("Chance de título", f"{champion['title']:.1%}")
    metric_cols[2].metric("Grupo mais difícil", f"Grupo {hardest_group['group']}")
    metric_cols[3].metric(
        "Pts médios do 3º",
        f"{hardest_group['avg_third_place_points']:.2f}",
    )

    tab_prob, tab_path, tab_groups, tab_matchups = st.tabs(
        [
            "Probabilidades",
            "Caminho provável",
            "Grupos difíceis",
            "Confrontos possíveis",
        ]
    )

    with tab_prob:
        st.dataframe(
            percentage_table(probabilities),
            width="stretch",
            hide_index=True,
        )

    with tab_path:
        teams = probabilities["team"].tolist()
        selected_team = st.selectbox(
            "Seleção",
            options=teams,
            index=0,
        )
        path = path_table(result["paths"], selected_team, result["n_simulations"])
        if path.empty:
            st.warning("A seleção quase nunca chega ao mata-mata nesta simulação.")
        else:
            st.dataframe(path, width="stretch", hide_index=True)

    with tab_groups:
        st.caption(
            "Aqui o grupo mais difícil é aproximado pelo número médio de pontos "
            "do terceiro colocado: quanto maior, mais caro é sobreviver no grupo."
        )
        st.dataframe(
            group_difficulty_table(result["group_difficulty"]),
            width="stretch",
            hide_index=True,
        )

    with tab_matchups:
        st.caption(
            "Confrontos mais recorrentes no mata-mata dentro das simulações. "
            f"Chaveamento: {result['bracket_method']}"
        )
        st.dataframe(
            matchup_table(result["matchups"]),
            width="stretch",
            hide_index=True,
        )


show_simulation_page()
