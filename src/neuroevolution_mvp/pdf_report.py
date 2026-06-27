"""PDF export for visual interpretability validation."""

from __future__ import annotations

import csv
import os
import textwrap
from pathlib import Path
from typing import Any

os.environ.setdefault("MPLCONFIGDIR", "/tmp/matplotlib")

import matplotlib.pyplot as plt
import torch
from matplotlib.backends.backend_pdf import PdfPages

from .artifacts import load_genome, load_json
from .config import ExperimentConfig
from .data import load_dataset
from .interpretability import compare_maps, occlusion_map, prediction_summary
from .features import FEATURE_SIZE, PROTOTYPE_FEATURE_SIZE
from .models import (
    BaselineMLP,
    EvolvedTopologyMLP,
    FeatureWrappedModel,
    FixedNeatTopologyMLP,
    GeneratedSubstrateMLP,
    PrototypeWrappedModel,
)
from .neat_runner import (
    build_hyperneat_feature_substrate,
    build_hyperneat_prototype_substrate,
    build_hyperneat_substrate,
    load_neat_config,
)


def export_interpretability_pdf(
    config: ExperimentConfig,
    output_path: Path,
    sample_count: int = 12,
) -> Path:
    """Export the image and NEAT/SGD attention maps to a standalone PDF."""

    output_path.parent.mkdir(parents=True, exist_ok=True)
    results = load_json(config.artifact_dir / "results.json")
    data = load_dataset(
        config.dataset,
        image_size=config.image_size,
        train_limit=100,
        test_limit=min(10_000, max(1_000, sample_count * 25)),
        batch_size=config.batch_size,
        seed=config.seed,
    )
    neat_config = load_neat_config(_selected_config_path(config, results))
    genome = load_genome(config.artifact_dir / "winner_genome.pkl")
    family = results.get("summary", {}).get("selected_neat_family", "direct")

    baseline = BaselineMLP(
        config.input_size,
        config.hidden_units,
        image_channels=config.image_channels,
    )
    neat_pure, evolved = _load_neat_pair(
        family=family,
        genome=genome,
        neat_config=neat_config,
        input_size=config.input_size,
        image_size=config.image_size,
        artifact_dir=config.artifact_dir,
    )
    baseline.load_state_dict(torch.load(config.artifact_dir / "baseline_mlp.pt", map_location="cpu"))
    evolved.load_state_dict(torch.load(config.artifact_dir / "evolved_topology_sgd.pt", map_location="cpu"))

    selected_examples = _diagnostic_indices(
        data.x_test,
        data.y_test,
        neat_pure,
        evolved,
        sample_count,
    )
    summary_figure = _summary_dashboard_figure(results)
    summary_png = output_path.with_name(
        f"{config.dataset}_interpretability_summary.png"
    )
    summary_figure.savefig(summary_png, dpi=180, bbox_inches="tight")
    _write_summary_csv(
        results,
        output_path.with_name(f"{config.dataset}_interpretability_summary.csv"),
    )

    with PdfPages(output_path) as pdf:
        _write_cover(pdf, results, len(selected_examples), config.dataset)
        pdf.savefig(summary_figure)
        plt.close(summary_figure)
        definitions_figure = _metric_definitions_figure(results)
        pdf.savefig(definitions_figure)
        plt.close(definitions_figure)
        for page_number, (index, outcome) in enumerate(selected_examples, start=1):
            image = _reshape_image(data.x_test[index], config.image_size, config.image_channels)
            label = int(data.y_test[index].item())
            pdf.savefig(
                _sample_figure(
                    page_number=page_number,
                    sample_index=index,
                    label=label,
                    image=image,
                    baseline=baseline,
                    neat_pure=neat_pure,
                    evolved=evolved,
                    image_size=config.image_size,
                    outcome=outcome,
                )
            )
            plt.close()

    return output_path


def _load_neat_pair(
    family: str,
    genome,
    neat_config,
    input_size: int,
    image_size: int,
    artifact_dir: Path,
):
    if family == "hyperneat":
        weight_matrix, bias = build_hyperneat_substrate(genome, neat_config, image_size=image_size)
        weight_tensor = torch.from_numpy(weight_matrix).float()
        bias_tensor = torch.from_numpy(bias).float()
        return (
            GeneratedSubstrateMLP(weight_tensor, bias_tensor, trainable=False),
            GeneratedSubstrateMLP(weight_tensor, bias_tensor, trainable=True),
        )

    if family == "hyperneat_features":
        weight_matrix, bias = build_hyperneat_feature_substrate(genome, neat_config)
        weight_tensor = torch.from_numpy(weight_matrix).float()
        bias_tensor = torch.from_numpy(bias).float()
        return (
            FeatureWrappedModel(
                GeneratedSubstrateMLP(weight_tensor, bias_tensor, trainable=False),
                image_size,
            ),
            FeatureWrappedModel(
                GeneratedSubstrateMLP(weight_tensor, bias_tensor, trainable=True),
                image_size,
            ),
        )

    if family == "hyperneat_prototypes":
        centroids = torch.load(
            artifact_dir / "prototype_centroids.pt",
            map_location="cpu",
        )
        weight_matrix, bias = build_hyperneat_prototype_substrate(genome, neat_config)
        weight_tensor = torch.from_numpy(weight_matrix).float()
        bias_tensor = torch.from_numpy(bias).float()
        return (
            PrototypeWrappedModel(
                GeneratedSubstrateMLP(weight_tensor, bias_tensor, trainable=False),
                centroids,
            ),
            PrototypeWrappedModel(
                GeneratedSubstrateMLP(weight_tensor, bias_tensor, trainable=True),
                centroids,
            ),
        )

    if family in {"prototypes", "seeded_prototypes"}:
        centroids = torch.load(
            artifact_dir / "prototype_centroids.pt",
            map_location="cpu",
        )
        return (
            PrototypeWrappedModel(
                FixedNeatTopologyMLP(genome, neat_config, input_size=PROTOTYPE_FEATURE_SIZE),
                centroids,
            ),
            PrototypeWrappedModel(
                EvolvedTopologyMLP(genome, neat_config, input_size=PROTOTYPE_FEATURE_SIZE),
                centroids,
            ),
        )

    if family == "features":
        return (
            FeatureWrappedModel(
                FixedNeatTopologyMLP(genome, neat_config, input_size=FEATURE_SIZE),
                image_size,
            ),
            FeatureWrappedModel(
                EvolvedTopologyMLP(genome, neat_config, input_size=FEATURE_SIZE),
                image_size,
            ),
        )

    return (
        FixedNeatTopologyMLP(genome, neat_config, input_size=input_size),
        EvolvedTopologyMLP(genome, neat_config, input_size=input_size),
    )


def _write_cover(
    pdf: PdfPages,
    results: dict[str, Any],
    sample_count: int,
    dataset_name: str,
) -> None:
    summary = results.get("summary", {})
    interpretation = results.get("interpretation_summary", {})
    conditioned = interpretation.get("pairwise_conditioned", {}).get(
        "neat_pure_vs_neat_sgd", {}
    )
    variants = results.get("neat_variant_results", [])
    fig = plt.figure(figsize=(11.69, 8.27), facecolor="white")
    fig.text(
        0.07,
        0.93,
        f"Validacao visual das predicoes - {dataset_name.upper()}",
        fontsize=21,
        weight="bold",
    )
    fig.text(
        0.07,
        0.875,
        "Imagem original, mapas de oclusao e diferenca entre NEAT puro e NEAT + SGD.",
        fontsize=12,
    )
    lines = [
        f"Amostras no PDF: {sample_count}",
        f"Imagens na metrica global: {int(interpretation.get('probe_samples', 0))} "
        f"({interpretation.get('evaluation_scope', 'n/a')})",
        f"Metodo de atribuicao: {interpretation.get('attribution_method', 'n/a')}",
        f"Dataset: {dataset_name.upper()}",
        f"Resolucao: {summary.get('image_size', 'n/a')}x{summary.get('image_size', 'n/a')}",
        f"Retreino dos pesos: {str(summary.get('optimizer', 'sgd')).upper()}",
        f"Inicializacao pos-NEAT: {summary.get('sgd_weight_initialization', 'n/a')}",
        f"Variante NEAT escolhida: {summary.get('selected_neat_variant', 'desconhecida')}",
        f"Baseline: {_fmt_accuracy(summary.get('baseline_sgd_accuracy'))}",
        f"NEAT puro: {_fmt_accuracy(summary.get('neat_accuracy'))}",
        f"Topologia NEAT + SGD: {_fmt_accuracy(summary.get('evolved_topology_sgd_accuracy'))}",
        f"Conexoes ativas: {summary.get('evolved_enabled_connections', 'n/a')}",
    ]
    if conditioned:
        correct = conditioned["both_correct"]
        wrong = conditioned["both_wrong"]
        controlled_wrong = conditioned["both_wrong_same_prediction"]
        controlled_gap = conditioned["controlled_attention_gap_weighted_jaccard"]
        primary = interpretation.get("primary_metric", {})
        lines.extend(
            [
                "",
                "NEAT puro vs NEAT + SGD (mesma arquitetura):",
                f"Divergencia de atencao no dataset: "
                f"{_fmt_metric(primary.get('value'))}",
                f"Divergencia quando preveem a mesma classe: "
                f"{_fmt_metric(conditioned.get('dataset_attention_divergence_same_prediction'))} "
                f"(n={conditioned.get('same_prediction_sample_count', 0)})",
                f"Cobertura de mapas validos: "
                f"{_fmt_accuracy(conditioned.get('valid_map_coverage'))}",
                f"Jaccard ponderado quando ambos acertam: "
                f"{_fmt_metric(correct['weighted_jaccard']['mean'])} "
                f"(n={correct['sample_count']})",
                f"Jaccard ponderado quando ambos erram: "
                f"{_fmt_metric(wrong['weighted_jaccard']['mean'])} "
                f"(n={wrong['sample_count']})",
                f"Gap de concordancia acerto - erro: "
                f"{_fmt_metric(conditioned['attention_agreement_gap_weighted_jaccard'], signed=True)}",
                f"Erros com mesma classe prevista: "
                f"{_fmt_metric(controlled_wrong['weighted_jaccard']['mean'])} "
                f"(n={controlled_wrong['sample_count']})",
                f"Gap controlado acerto - erro: "
                f"{_fmt_metric(controlled_gap, signed=True)}",
                f"Erros conjuntos com classes diferentes: "
                f"{_fmt_accuracy(conditioned['wrong_prediction_disagreement'])}",
            ]
        )
    fig.text(
        0.07,
        0.82,
        "\n".join(lines),
        fontsize=9.5,
        linespacing=1.3,
        verticalalignment="top",
    )

    if variants:
        table_rows = [
            [
                row["variant"],
                f"{row['accuracy']:.1%}",
                f"{row.get('evolved_topology_accuracy', 0.0):.1%}",
                f"{row['best_fitness']:.3f}",
                str(row["enabled_connections"]),
            ]
            for row in variants
        ]
        ax = fig.add_axes((0.07, 0.04, 0.86, 0.25))
        ax.axis("off")
        table = ax.table(
            cellText=table_rows,
            colLabels=["Variante", "NEAT puro", "NEAT + SGD", "Fitness", "Conexoes"],
            loc="center",
            cellLoc="center",
        )
        table.auto_set_font_size(False)
        table.set_fontsize(8)
        table.scale(1, 1.4)

    pdf.savefig(fig)
    plt.close(fig)


def _summary_dashboard_figure(results: dict[str, Any]):
    article = _article_summary(results)
    fig, axes = plt.subplots(1, 2, figsize=(11.69, 6.2), facecolor="white")
    fig.suptitle("Sumarizacao quantitativa da interpretabilidade", fontsize=20, weight="bold")

    dad_labels = ["DAD global", "DAD macro", "DAD mesma\nprevisao"]
    dad_values = [
        article.get("dataset_attention_divergence"),
        article.get("dad_macro"),
        article.get("dad_same_prediction"),
    ]
    _bar_chart(
        axes[0],
        dad_labels,
        dad_values,
        "Divergencia espacial (maior = mais diferente)",
        "#16866f",
    )

    agreement_labels = ["Ambos\nacertam", "Ambos\nerram", "Erram a\nmesma classe"]
    agreement_values = [
        article.get("both_correct_weighted_jaccard"),
        article.get("both_wrong_weighted_jaccard"),
        article.get("both_wrong_same_prediction_weighted_jaccard"),
    ]
    _bar_chart(
        axes[1],
        agreement_labels,
        agreement_values,
        "Jaccard ponderado (maior = mais semelhante)",
        "#4f9d57",
    )

    lines = [
        f"Dataset: {article.get('dataset', 'n/a')} | seed: {article.get('seed', 'n/a')} | "
        f"imagens: {article.get('evaluated_images', 'n/a')}",
        f"Acuracias - baseline: {_fmt_accuracy(article.get('baseline_accuracy'))} | "
        f"NEAT puro: {_fmt_accuracy(article.get('neat_pure_accuracy'))} | "
        f"NEAT + SGD: {_fmt_accuracy(article.get('neat_sgd_accuracy'))}",
        f"DAD: {_fmt_metric(article.get('dataset_attention_divergence'))} "
        f"[IC95% {_fmt_metric(article.get('dad_ci95_low'))}, "
        f"{_fmt_metric(article.get('dad_ci95_high'))}] | "
        f"cobertura: {_fmt_accuracy(article.get('valid_map_coverage'))}",
        f"Gap acerto-erro: {_fmt_metric(article.get('attention_agreement_gap'), signed=True)} | "
        f"gap controlado: {_fmt_metric(article.get('controlled_attention_gap'), signed=True)}",
        f"Grupos - ambos acertam: {article.get('both_correct_n', 'n/a')} | "
        f"ambos erram: {article.get('both_wrong_n', 'n/a')} | "
        f"apenas um acerta: {article.get('one_correct_n', 'n/a')}",
    ]
    fig.text(0.07, 0.06, "\n".join(lines), fontsize=10.5, linespacing=1.5)
    fig.tight_layout(rect=(0.04, 0.25, 0.98, 0.91))
    return fig


def _bar_chart(ax, labels: list[str], values: list[object], title: str, color: str) -> None:
    numeric = [float(value) if isinstance(value, int | float) else 0.0 for value in values]
    bars = ax.bar(labels, numeric, color=color, alpha=0.88)
    ax.set_ylim(0.0, 1.0)
    ax.set_title(title, fontsize=12)
    ax.set_ylabel("valor")
    ax.grid(axis="y", alpha=0.2)
    for bar, original in zip(bars, values, strict=True):
        label = "n/a" if not isinstance(original, int | float) else f"{float(original):.3f}"
        ax.text(
            bar.get_x() + bar.get_width() / 2,
            min(bar.get_height() + 0.025, 0.96),
            label,
            ha="center",
            fontsize=10,
        )


def _metric_definitions_figure(results: dict[str, Any]):
    definitions = results.get("metric_definitions", {})
    fig = plt.figure(figsize=(11.69, 8.27), facecolor="white")
    fig.text(0.06, 0.93, "Definicoes das metricas", fontsize=20, weight="bold")
    fig.text(
        0.06,
        0.88,
        "A DAD e uma metrica operacional deste estudo, baseada na similaridade min-max.",
        fontsize=11,
    )
    rows = []
    for name, definition in definitions.items():
        rows.append(
            [
                textwrap.fill(name, 25),
                textwrap.fill(definition.get("definition", ""), 48),
                definition.get("range", ""),
                textwrap.fill(definition.get("interpretation", ""), 48),
            ]
        )
    ax = fig.add_axes((0.05, 0.08, 0.90, 0.74))
    ax.axis("off")
    if rows:
        table = ax.table(
            cellText=rows,
            colLabels=["Metrica", "Definicao", "Faixa", "Interpretacao"],
            cellLoc="left",
            loc="upper center",
            colWidths=[0.20, 0.31, 0.09, 0.40],
        )
        table.auto_set_font_size(False)
        table.set_fontsize(8)
        table.scale(1.0, 2.7)
    return fig


def _write_summary_csv(results: dict[str, Any], path: Path) -> None:
    article = _article_summary(results)
    if not article:
        return
    with path.open("w", newline="", encoding="utf-8") as file:
        writer = csv.DictWriter(file, fieldnames=list(article))
        writer.writeheader()
        writer.writerow(article)


def _article_summary(results: dict[str, Any]) -> dict[str, Any]:
    article = results.get("article_summary")
    if article:
        return article
    summary = results.get("summary", {})
    interpretation = results.get("interpretation_summary", {})
    pair = interpretation.get("pairwise_conditioned", {}).get(
        "neat_pure_vs_neat_sgd", {}
    )
    primary = interpretation.get("primary_metric", {})
    return {
        "dataset": summary.get("dataset"),
        "evaluated_images": interpretation.get("probe_samples"),
        "baseline_accuracy": summary.get("baseline_sgd_accuracy"),
        "neat_pure_accuracy": summary.get("neat_accuracy"),
        "neat_sgd_accuracy": summary.get("evolved_topology_sgd_accuracy"),
        "dataset_attention_divergence": primary.get("value"),
        "dad_macro": pair.get("dataset_attention_divergence_macro"),
        "dad_same_prediction": pair.get("dataset_attention_divergence_same_prediction"),
        "valid_map_coverage": pair.get("valid_map_coverage"),
    }


def _sample_figure(
    page_number: int,
    sample_index: int,
    label: int,
    image: torch.Tensor,
    baseline: BaselineMLP,
    neat_pure: FixedNeatTopologyMLP,
    evolved: EvolvedTopologyMLP,
    image_size: int,
    outcome: str,
):
    baseline_map = occlusion_map(baseline, image, image_size)
    neat_map = occlusion_map(neat_pure, image, image_size)
    evolved_map = occlusion_map(evolved, image, image_size)
    difference_map = torch.abs(neat_map - evolved_map)
    comparison = compare_maps(neat_map, evolved_map)
    predictions = [
        ("Baseline", prediction_summary(baseline, image)),
        ("NEAT puro", prediction_summary(neat_pure, image)),
        ("NEAT + SGD", prediction_summary(evolved, image)),
    ]

    fig, axes = plt.subplots(1, 5, figsize=(13.5, 3.6), facecolor="white")
    panels = [
        (image, "Imagem original", "Greys"),
        (baseline_map, "Sensibilidade: baseline", "Greens"),
        (neat_map, "Sensibilidade: NEAT puro", "YlGn"),
        (evolved_map, "Sensibilidade: NEAT + SGD", "Greens"),
        (difference_map, "Diferenca NEAT vs SGD", "summer"),
    ]
    for ax, (matrix, title, cmap) in zip(axes, panels, strict=True):
        if matrix.ndim == 3:
            ax.imshow(matrix.detach().permute(1, 2, 0).numpy(), interpolation="nearest")
        else:
            ax.imshow(matrix.detach().numpy(), cmap=cmap, interpolation="nearest")
        ax.set_title(title, fontsize=10)
        ax.set_xticks([])
        ax.set_yticks([])

    prediction_text = " | ".join(
        f"{name}: {int(row['class'])} ({float(row['confidence']):.1%})"
        for name, row in predictions
    )
    fig.suptitle(
        (
            f"Pagina {page_number} - {outcome} - amostra {sample_index} - rotulo real {label}\n"
            f"{prediction_text} | similaridade NEAT/SGD: {comparison['cosine_similarity']:.2f} | "
            f"pixels comuns: {comparison['top_pixel_overlap']:.1%}"
        ),
        fontsize=12,
    )
    fig.tight_layout(rect=(0, 0, 1, 0.82))
    return fig


def _reshape_image(image: torch.Tensor, image_size: int, image_channels: int) -> torch.Tensor:
    if image_channels == 1:
        return image.reshape(image_size, image_size)
    return image.reshape(image_channels, image_size, image_size)


@torch.no_grad()
def _diagnostic_indices(
    images: torch.Tensor,
    labels: torch.Tensor,
    neat_pure,
    evolved,
    sample_count: int,
) -> list[tuple[int, str]]:
    neat_pure.eval()
    evolved.eval()
    neat_predictions = neat_pure(images).argmax(dim=1)
    sgd_predictions = evolved(images).argmax(dim=1)
    neat_correct = neat_predictions.eq(labels)
    sgd_correct = sgd_predictions.eq(labels)

    categories = [
        (
            "discordam - apenas NEAT puro acerta",
            torch.where(neat_correct & ~sgd_correct)[0].tolist(),
        ),
        (
            "discordam - apenas NEAT + SGD acerta",
            torch.where(~neat_correct & sgd_correct)[0].tolist(),
        ),
        (
            "discordam - ambos erram classes diferentes",
            torch.where(
                ~neat_correct
                & ~sgd_correct
                & neat_predictions.ne(sgd_predictions)
            )[0].tolist(),
        ),
        (
            "concordam - ambos acertam",
            torch.where(neat_correct & sgd_correct)[0].tolist(),
        ),
        (
            "concordam - ambos erram a mesma classe",
            torch.where(
                ~neat_correct
                & ~sgd_correct
                & neat_predictions.eq(sgd_predictions)
            )[0].tolist(),
        ),
    ]
    selected: list[tuple[int, str]] = []
    used: set[int] = set()
    quota = max(1, sample_count // len(categories))
    for outcome, candidates in categories:
        for index in _class_diverse_indices(candidates, labels, quota):
            if index not in used and len(selected) < sample_count:
                selected.append((index, outcome))
                used.add(index)

    disagreement_candidates = torch.where(neat_predictions.ne(sgd_predictions))[0].tolist()
    fill_order = [
        ("discordam - previsoes diferentes", disagreement_candidates),
        ("cobertura adicional", list(range(len(labels)))),
    ]
    for outcome, candidates in fill_order:
        for index in _class_diverse_indices(candidates, labels, sample_count):
            if index not in used and len(selected) < sample_count:
                selected.append((index, outcome))
                used.add(index)
        if len(selected) >= sample_count:
            break
    return selected


def _class_diverse_indices(
    candidates: list[int],
    labels: torch.Tensor,
    limit: int,
) -> list[int]:
    by_class: dict[int, list[int]] = {}
    for index in candidates:
        by_class.setdefault(int(labels[index].item()), []).append(index)
    selected: list[int] = []
    offset = 0
    classes = sorted(by_class)
    while len(selected) < min(limit, len(candidates)):
        added = False
        for label in classes:
            rows = by_class[label]
            if offset < len(rows):
                selected.append(rows[offset])
                added = True
                if len(selected) >= limit:
                    break
        if not added:
            break
        offset += 1
    return selected


def _selected_config_path(config: ExperimentConfig, results: dict[str, Any]) -> Path:
    selected = results.get("summary", {}).get("selected_neat_variant")
    for path in config.neat_variant_paths or (config.neat_config_path,):
        if path.stem == selected:
            return path
    return config.neat_config_path


def _fmt_accuracy(value: object) -> str:
    if isinstance(value, int | float):
        return f"{value:.1%}"
    return "n/a"


def _fmt_metric(value: object, signed: bool = False) -> str:
    if isinstance(value, int | float):
        return f"{value:+.3f}" if signed else f"{value:.3f}"
    return "n/a"
