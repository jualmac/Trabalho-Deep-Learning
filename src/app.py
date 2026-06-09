"""Streamlit interface for the neuroevolution MVP."""

from __future__ import annotations

import os

os.environ.setdefault("MPLCONFIGDIR", "/tmp/matplotlib")

import matplotlib.pyplot as plt
import pandas as pd
import streamlit as st
import torch

from neuroevolution_mvp.artifacts import load_genome, load_json
from neuroevolution_mvp.config import DEFAULT_ARTIFACT_DIR, ExperimentConfig
from neuroevolution_mvp.data import load_mnist, set_seed
from neuroevolution_mvp.experiment import artifact_exists, run_experiment
from neuroevolution_mvp.interpretability import (
    compare_maps,
    occlusion_map,
    prediction_summary,
)
from neuroevolution_mvp.models import BaselineMLP, EvolvedTopologyMLP, FixedNeatTopologyMLP
from neuroevolution_mvp.neat_runner import load_neat_config


st.set_page_config(page_title="NEAT + Interpretabilidade", layout="wide")


def main() -> None:
    _apply_theme()
    config = _sidebar_config()
    results = _load_or_run_results(config)

    st.markdown(
        """
        <section class="hero">
            <div>
                <p class="eyebrow">Neuroevolucao + interpretabilidade</p>
                <h1>Onde cada rede esta olhando?</h1>
                <p>
                    O MVP separa a arquitetura encontrada pelo NEAT dos pesos:
                    NEAT puro evolui arquitetura e pesos; NEAT + SGD usa a
                    mesma arquitetura evoluida, mas treina os pesos com SGD.
                </p>
            </div>
        </section>
        """,
        unsafe_allow_html=True,
    )

    if not results:
        st.info("Rode um experimento na barra lateral para gerar os artefatos.")
        return

    overview, examples, topology, tracking = st.tabs(
        ["Visao geral", "Exemplos de olhar", "Topologia NEAT", "Rastreamento"]
    )

    with overview:
        _show_overview(results)
    with examples:
        _show_examples(config)
    with topology:
        _show_topology(config, results)
    with tracking:
        _show_tracking(results)


def _sidebar_config() -> ExperimentConfig:
    defaults = ExperimentConfig()
    st.sidebar.header("Experimento")
    st.sidebar.caption("Valores pequenos deixam a demo rapida para sala.")
    generations = st.sidebar.slider("Geracoes NEAT", 1, 12, defaults.neat_generations)
    epochs = st.sidebar.slider("Epocas SGD", 1, 12, defaults.baseline_epochs)
    train_limit = st.sidebar.slider("Amostras de treino", 200, 2_500, defaults.train_limit, step=100)
    test_limit = st.sidebar.slider("Amostras de teste", 100, 1_000, defaults.test_limit, step=100)
    neat_eval_limit = st.sidebar.slider(
        "Amostras por avaliacao NEAT",
        100,
        1_000,
        defaults.neat_eval_limit,
        step=100,
    )

    return ExperimentConfig(
        neat_generations=generations,
        baseline_epochs=epochs,
        evolved_sgd_epochs=epochs,
        train_limit=train_limit,
        test_limit=test_limit,
        neat_eval_limit=neat_eval_limit,
        artifact_dir=DEFAULT_ARTIFACT_DIR,
    )


def _load_or_run_results(config: ExperimentConfig) -> dict | None:
    run_clicked = st.sidebar.button("Rodar novo experimento", type="primary", use_container_width=True)
    if run_clicked:
        with st.spinner("Treinando modelos, evoluindo topologia e calculando mapas..."):
            return run_experiment(config)

    if artifact_exists(config.artifact_dir):
        results = load_json(config.artifact_dir / "results.json")
        if int(results["config"].get("image_size", 0)) != config.image_size:
            st.sidebar.warning("Artefatos antigos detectados. Rode um novo experimento.")
            return None
        st.sidebar.success("Artefatos carregados.")
        return results

    return None


def _show_overview(results: dict) -> None:
    interpretation = results.get("interpretation_summary")
    summary = results["summary"]

    st.subheader("Resumo interpretavel")
    if not interpretation:
        st.info("Rode o experimento novamente para gerar as metricas de interpretacao.")
        return

    cols = st.columns(4)
    cols[0].metric("NEAT puro vs NEAT+SGD", f"{interpretation['neat_pure_vs_neat_sgd_cosine']:.2f}")
    cols[1].metric("Pixels-chave comuns", f"{interpretation['neat_pure_vs_neat_sgd_top_overlap']:.1%}")
    cols[2].metric("Diferenca media", f"{interpretation['neat_pure_vs_neat_sgd_difference']:.2f}")
    cols[3].metric("Predicoes iguais", f"{interpretation['neat_pure_vs_neat_sgd_prediction_agreement']:.1%}")

    st.markdown(
        f"""
        <div class="insight">
            <strong>Leitura da demo:</strong>
            estes numeros foram calculados em {int(interpretation['probe_samples'])} imagens.
            A comparacao principal usa a mesma arquitetura evoluida. O que muda
            e o otimizador dos pesos: NEAT no modelo puro, SGD no modelo
            retreinado. Similaridade baixa sugere que trocar o otimizador de
            pesos muda as regioes usadas na decisao.
        </div>
        """,
        unsafe_allow_html=True,
    )

    st.subheader("Controles de sanidade")
    control_cols = st.columns(4)
    control_cols[0].metric("Baseline SGD", f"{summary['baseline_sgd_accuracy']:.1%}")
    control_cols[1].metric("NEAT puro", f"{summary['neat_accuracy']:.1%}")
    control_cols[2].metric("NEAT + SGD", f"{summary['evolved_topology_sgd_accuracy']:.1%}")
    control_cols[3].metric("Conexoes NEAT", int(summary["evolved_enabled_connections"]))
    st.caption("Acuracia ajuda a checar se ha sinal de aprendizagem, mas nao decide a pergunta do trabalho.")


def _show_examples(config: ExperimentConfig) -> None:
    data, baseline, neat_pure, evolved, _ = _load_demo_objects(config)

    st.subheader("Exemplos: regioes usadas pela decisao")
    st.caption(
        "Use os botoes de digito para abrir um exemplo daquele rotulo, ou navegue "
        "amostra por amostra. O ultimo mapa mostra onde NEAT puro e NEAT + SGD divergem."
    )

    if "sample_index" not in st.session_state:
        st.session_state.sample_index = 0

    controls = st.columns([1, 1, 4])
    if controls[0].button("Anterior", use_container_width=True):
        st.session_state.sample_index = max(0, st.session_state.sample_index - 1)
    if controls[1].button("Proxima", use_container_width=True):
        st.session_state.sample_index = min(len(data.x_test) - 1, st.session_state.sample_index + 1)

    selected = controls[2].slider(
        "Amostra MNIST",
        0,
        len(data.x_test) - 1,
        st.session_state.sample_index,
    )
    st.session_state.sample_index = selected

    digit_examples = _first_index_by_digit(data.y_test)
    st.markdown('<div class="digit-picker">Exemplo por digito</div>', unsafe_allow_html=True)
    quick_cols = st.columns(10)
    for digit, col in enumerate(quick_cols):
        disabled = digit not in digit_examples
        if col.button(str(digit), use_container_width=True, disabled=disabled):
            st.session_state.sample_index = digit_examples[digit]

    image = data.x_test[st.session_state.sample_index].reshape(config.image_size, config.image_size)
    label = int(data.y_test[st.session_state.sample_index].item())
    st.caption(f"Amostra selecionada: indice {st.session_state.sample_index}, rotulo {label}.")

    baseline_map = occlusion_map(baseline, image, config.image_size)
    neat_map = occlusion_map(neat_pure, image, config.image_size)
    evolved_map = occlusion_map(evolved, image, config.image_size)
    difference_map = torch.abs(neat_map - evolved_map)
    map_comparison = compare_maps(neat_map, evolved_map)
    baseline_prediction = prediction_summary(baseline, image)
    neat_prediction = prediction_summary(neat_pure, image)
    evolved_prediction = prediction_summary(evolved, image)

    metric_cols = st.columns(6)
    metric_cols[0].metric("Rotulo real", label)
    metric_cols[1].metric("Baseline SGD", int(baseline_prediction["class"]))
    metric_cols[2].metric("NEAT puro", int(neat_prediction["class"]))
    metric_cols[3].metric("NEAT + SGD", int(evolved_prediction["class"]))
    metric_cols[4].metric("Similaridade NEAT/SGD", f"{map_comparison['cosine_similarity']:.2f}")
    metric_cols[5].metric("Pixels comuns", f"{map_comparison['top_pixel_overlap']:.1%}")

    fig_cols = st.columns(5)
    fig_cols[0].pyplot(_heatmap_figure(image, "Digito original", "Greys"), use_container_width=True)
    fig_cols[0].markdown(_prediction_badge("Rotulo", label, None), unsafe_allow_html=True)
    fig_cols[1].pyplot(_heatmap_figure(baseline_map, "Olhar: baseline SGD", "Greens"), use_container_width=True)
    fig_cols[1].markdown(
        _prediction_badge(
            "Predicao",
            int(baseline_prediction["class"]),
            float(baseline_prediction["confidence"]),
        ),
        unsafe_allow_html=True,
    )
    fig_cols[2].pyplot(_heatmap_figure(neat_map, "Olhar: NEAT puro", "YlGn"), use_container_width=True)
    fig_cols[2].markdown(
        _prediction_badge(
            "Predicao",
            int(neat_prediction["class"]),
            float(neat_prediction["confidence"]),
        ),
        unsafe_allow_html=True,
    )
    fig_cols[3].pyplot(_heatmap_figure(evolved_map, "Olhar: NEAT + SGD", "Greens"), use_container_width=True)
    fig_cols[3].markdown(
        _prediction_badge(
            "Predicao",
            int(evolved_prediction["class"]),
            float(evolved_prediction["confidence"]),
        ),
        unsafe_allow_html=True,
    )
    fig_cols[4].pyplot(_heatmap_figure(difference_map, "Diferenca NEAT vs SGD", "summer"), use_container_width=True)
    fig_cols[4].markdown(
        _prediction_badge("Comparacao", "NEAT vs SGD", None),
        unsafe_allow_html=True,
    )


def _show_topology(config: ExperimentConfig, results: dict) -> None:
    _, _, _, _, genome = _load_demo_objects(config)
    neat_config = load_neat_config(config.neat_config_path)
    evolved = EvolvedTopologyMLP(genome, neat_config, input_size=config.input_size)

    summary = results["summary"]
    hidden_nodes = sorted(
        node
        for layer in evolved.layers
        for node in layer
        if node not in evolved.output_keys
    )

    st.subheader("Visao da topologia evoluida")
    cols = st.columns(4)
    cols[0].metric("Camadas computaveis", len(evolved.layers))
    cols[1].metric("Neuronios ocultos", len(hidden_nodes))
    cols[2].metric("Saidas", len(evolved.output_keys))
    cols[3].metric("Conexoes ativas", int(summary["evolved_enabled_connections"]))

    st.pyplot(_network_figure(genome, evolved, max_inputs=24), use_container_width=True)
    st.caption(
        "A figura agrupa parte dos pixels de entrada para manter a leitura limpa. "
        "Linhas mais fortes representam conexoes com maior peso absoluto no genoma vencedor."
    )

    details = []
    for index, layer in enumerate(evolved.layers, start=1):
        hidden_count = sum(1 for node in layer if node not in evolved.output_keys)
        output_count = sum(1 for node in layer if node in evolved.output_keys)
        details.append(
            {
                "camada": index,
                "neuronios_ocultos": hidden_count,
                "saidas": output_count,
                "nos": ", ".join(str(node) for node in layer[:12]),
            }
        )
    st.dataframe(pd.DataFrame(details), hide_index=True, use_container_width=True)


def _show_tracking(results: dict) -> None:
    st.subheader("Rastreamento do experimento")

    train_rows = []
    for row in results["baseline_history"]:
        train_rows.append({"modelo": "Baseline SGD", **row})
    for row in results["evolved_sgd_history"]:
        train_rows.append({"modelo": "Topologia NEAT + SGD", **row})

    train_frame = pd.DataFrame(train_rows)
    neat_frame = pd.DataFrame(results["neat_history"])

    left, right = st.columns(2)
    with left:
        st.caption("Controle SGD")
        st.line_chart(train_frame, x="epoch", y="test_accuracy", color="modelo")
        st.dataframe(train_frame, hide_index=True, use_container_width=True)
    with right:
        st.caption("Evolucao NEAT")
        st.line_chart(neat_frame, x="generation", y=["best_fitness", "mean_fitness"])
        st.dataframe(neat_frame, hide_index=True, use_container_width=True)


def _load_demo_objects(
    config: ExperimentConfig,
) -> tuple[object, BaselineMLP, FixedNeatTopologyMLP, EvolvedTopologyMLP, object]:
    data = load_mnist(
        image_size=config.image_size,
        train_limit=100,
        test_limit=config.test_limit,
        batch_size=config.batch_size,
        seed=config.seed,
    )
    neat_config = load_neat_config(config.neat_config_path)
    genome = load_genome(config.artifact_dir / "winner_genome.pkl")
    baseline = BaselineMLP(config.input_size, config.hidden_units)
    neat_pure = FixedNeatTopologyMLP(genome, neat_config, input_size=config.input_size)
    evolved = EvolvedTopologyMLP(genome, neat_config, input_size=config.input_size)

    baseline.load_state_dict(torch.load(config.artifact_dir / "baseline_mlp.pt"))
    evolved.load_state_dict(torch.load(config.artifact_dir / "evolved_topology_sgd.pt"))
    return data, baseline, neat_pure, evolved, genome


def _heatmap_figure(matrix: torch.Tensor, title: str, cmap: str):
    fig, ax = plt.subplots(figsize=(3, 3), facecolor="#f7fbf2")
    ax.imshow(matrix.detach().numpy(), cmap=cmap, interpolation="nearest")
    ax.set_title(title, fontsize=11, color="#17351f")
    ax.set_xticks([])
    ax.set_yticks([])
    for spine in ax.spines.values():
        spine.set_color("#b8d8ad")
    fig.tight_layout(pad=0.3)
    return fig


def _first_index_by_digit(labels: torch.Tensor) -> dict[int, int]:
    examples: dict[int, int] = {}
    for index, label in enumerate(labels.tolist()):
        digit = int(label)
        if digit not in examples:
            examples[digit] = index
        if len(examples) == 10:
            break
    return examples


def _prediction_badge(label: str, value: int | str, confidence: float | None) -> str:
    confidence_text = "" if confidence is None else f"<span>{confidence:.1%} confianca</span>"
    return f"""
    <div class="prediction-badge">
        <strong>{label}: {value}</strong>
        {confidence_text}
    </div>
    """


def _network_figure(genome, evolved: EvolvedTopologyMLP, max_inputs: int = 24):
    fig, ax = plt.subplots(figsize=(10, 5.6), facecolor="#f7fbf2")
    ax.set_facecolor("#f7fbf2")
    ax.axis("off")

    shown_inputs = evolved.input_keys[:max_inputs]
    output_keys = evolved.output_keys
    hidden_layers = [
        [node for node in layer if node not in output_keys]
        for layer in evolved.layers
    ]

    positions: dict[int, tuple[float, float]] = {}
    _place_nodes(positions, shown_inputs, x=0.0)
    for idx, layer in enumerate(hidden_layers, start=1):
        if layer:
            _place_nodes(positions, layer, x=float(idx))
    output_x = float(max(2, len(hidden_layers) + 1))
    _place_nodes(positions, output_keys, x=output_x)

    visible_nodes = set(positions)
    weights = [
        abs(connection.weight)
        for connection in genome.connections.values()
        if connection.enabled
    ]
    max_weight = max(weights) if weights else 1.0

    for (source, target), connection in genome.connections.items():
        if not connection.enabled or source not in visible_nodes or target not in visible_nodes:
            continue
        x1, y1 = positions[source]
        x2, y2 = positions[target]
        width = 0.4 + 2.4 * abs(connection.weight) / max_weight
        color = "#1b7f45" if connection.weight >= 0 else "#7d6b00"
        ax.plot([x1, x2], [y1, y2], color=color, linewidth=width, alpha=0.28)

    _draw_nodes(ax, positions, shown_inputs, "#dff4d7", "#3c7f44", size=80)
    for layer in hidden_layers:
        _draw_nodes(ax, positions, layer, "#8fcb7a", "#1d5b34", size=170)
    _draw_nodes(ax, positions, output_keys, "#1f8f55", "#0d3924", size=230)

    ax.text(0.0, 1.08, f"Entradas visiveis: {len(shown_inputs)} de {len(evolved.input_keys)} pixels", color="#17351f")
    ax.text(output_x, 1.08, "10 classes", color="#17351f", ha="right")
    fig.tight_layout()
    return fig


def _place_nodes(positions: dict[int, tuple[float, float]], nodes: list[int], x: float) -> None:
    if not nodes:
        return
    if len(nodes) == 1:
        positions[nodes[0]] = (x, 0.5)
        return
    for index, node in enumerate(nodes):
        positions[node] = (x, 1.0 - index / (len(nodes) - 1))


def _draw_nodes(ax, positions: dict[int, tuple[float, float]], nodes: list[int], color: str, edge: str, size: int) -> None:
    for node in nodes:
        if node not in positions:
            continue
        x, y = positions[node]
        ax.scatter([x], [y], s=size, color=color, edgecolor=edge, linewidth=1.4, zorder=3)


def _apply_theme() -> None:
    st.markdown(
        """
        <style>
        :root {
            --deep-green: #12351f;
            --leaf: #1f8f55;
            --moss: #7bb661;
            --soft: #f3faef;
            --line: #c9dfbd;
        }
        .stApp {
            background:
                radial-gradient(circle at 18% 8%, rgba(123, 182, 97, 0.20), transparent 28%),
                linear-gradient(180deg, #f7fbf2 0%, #eef7eb 100%);
            color: var(--deep-green);
        }
        .hero {
            border: 1px solid var(--line);
            background: linear-gradient(135deg, rgba(18, 53, 31, 0.96), rgba(31, 143, 85, 0.86));
            color: white;
            padding: 26px 30px;
            border-radius: 8px;
            margin-bottom: 18px;
        }
        .hero h1 {
            font-size: 40px;
            line-height: 1.08;
            margin: 4px 0 10px;
            letter-spacing: 0;
        }
        .hero p {
            max-width: 900px;
            font-size: 16px;
            margin: 0;
        }
        .eyebrow {
            text-transform: uppercase;
            font-size: 12px !important;
            letter-spacing: 0.08em;
            color: #ccefc0;
            font-weight: 700;
        }
        .insight {
            border-left: 5px solid var(--leaf);
            background: rgba(223, 244, 215, 0.78);
            padding: 14px 16px;
            border-radius: 6px;
            margin: 12px 0 24px;
        }
        .digit-picker {
            color: var(--deep-green);
            font-size: 13px;
            font-weight: 800;
            margin: 14px 0 8px;
        }
        .prediction-badge {
            background: rgba(255, 255, 255, 0.84);
            border: 1px solid var(--line);
            border-radius: 8px;
            color: var(--deep-green);
            padding: 9px 10px;
            text-align: center;
            margin-top: -4px;
            min-height: 48px;
        }
        .prediction-badge strong {
            display: block;
            font-size: 14px;
            line-height: 1.2;
        }
        .prediction-badge span {
            display: block;
            color: #4c7354;
            font-size: 12px;
            margin-top: 3px;
        }
        div[data-testid="stMetric"] {
            background: rgba(255, 255, 255, 0.72);
            border: 1px solid var(--line);
            border-radius: 8px;
            padding: 12px 14px;
        }
        .stButton > button {
            border-radius: 8px;
            border: 1px solid #93c684;
            background: #ffffff;
            color: var(--deep-green);
            font-weight: 700;
            min-height: 40px;
        }
        .stButton > button:hover {
            border-color: var(--leaf);
            color: var(--leaf);
            background: #edf8e9;
        }
        .stButton > button:focus,
        .stButton > button:active {
            border-color: #1f8f55;
            color: #0f3822;
            background: #dff4d7;
            box-shadow: 0 0 0 2px rgba(31, 143, 85, 0.20);
        }
        .stButton > button:disabled {
            color: #7f9378;
            background: #eef4ea;
            border-color: #d2dfcb;
        }
        .stTabs [data-baseweb="tab-list"] {
            gap: 8px;
            border-bottom: 1px solid var(--line);
            padding-bottom: 8px;
        }
        .stTabs [data-baseweb="tab"] {
            border-radius: 8px;
            border: 1px solid var(--line);
            background: rgba(255, 255, 255, 0.75);
            padding: 8px 14px;
            color: var(--deep-green) !important;
        }
        .stTabs [data-baseweb="tab"] p {
            color: var(--deep-green) !important;
            font-weight: 800;
            opacity: 1 !important;
        }
        .stTabs [data-baseweb="tab"]:hover {
            background: #edf8e9;
            border-color: #93c684;
        }
        .stTabs [data-baseweb="tab"][aria-selected="true"] {
            background: #1f8f55;
            border-color: #176b40;
        }
        .stTabs [data-baseweb="tab"][aria-selected="true"] p {
            color: #ffffff !important;
        }
        </style>
        """,
        unsafe_allow_html=True,
    )


if __name__ == "__main__":
    set_seed(ExperimentConfig().seed)
    main()
