"""PDF export for visual interpretability validation."""

from __future__ import annotations

import os
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
        test_limit=max(config.test_limit, sample_count),
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

    indices = _representative_indices(data.y_test, sample_count)

    with PdfPages(output_path) as pdf:
        _write_cover(pdf, results, len(indices), config.dataset)
        for page_number, index in enumerate(indices, start=1):
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
    variants = results.get("neat_variant_results", [])
    fig = plt.figure(figsize=(11.69, 8.27), facecolor="white")
    fig.text(
        0.07,
        0.88,
        f"Validacao visual das predicoes - {dataset_name.upper()}",
        fontsize=24,
        weight="bold",
    )
    fig.text(
        0.07,
        0.82,
        "Imagem original, mapas de oclusao e diferenca entre NEAT puro e NEAT + SGD.",
        fontsize=12,
    )
    lines = [
        f"Amostras no PDF: {sample_count}",
        f"Dataset: {dataset_name.upper()}",
        f"Resolucao: {summary.get('image_size', 'n/a')}x{summary.get('image_size', 'n/a')}",
        f"Retreino dos pesos: {str(summary.get('optimizer', 'sgd')).upper()}",
        f"Variante NEAT escolhida: {summary.get('selected_neat_variant', 'desconhecida')}",
        f"Baseline: {_fmt_accuracy(summary.get('baseline_sgd_accuracy'))}",
        f"NEAT puro: {_fmt_accuracy(summary.get('neat_accuracy'))}",
        f"Topologia NEAT + SGD: {_fmt_accuracy(summary.get('evolved_topology_sgd_accuracy'))}",
        f"Conexoes ativas: {summary.get('evolved_enabled_connections', 'n/a')}",
    ]
    fig.text(0.07, 0.70, "\n".join(lines), fontsize=13, linespacing=1.7)

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
        ax = fig.add_axes((0.07, 0.16, 0.86, 0.34))
        ax.axis("off")
        table = ax.table(
            cellText=table_rows,
            colLabels=["Variante", "NEAT puro", "NEAT + SGD", "Fitness", "Conexoes"],
            loc="center",
            cellLoc="center",
        )
        table.auto_set_font_size(False)
        table.set_fontsize(10)
        table.scale(1, 1.4)

    pdf.savefig(fig)
    plt.close(fig)


def _sample_figure(
    page_number: int,
    sample_index: int,
    label: int,
    image: torch.Tensor,
    baseline: BaselineMLP,
    neat_pure: FixedNeatTopologyMLP,
    evolved: EvolvedTopologyMLP,
    image_size: int,
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
        (baseline_map, "Olhar: baseline", "Greens"),
        (neat_map, "Olhar: NEAT puro", "YlGn"),
        (evolved_map, "Olhar: NEAT + SGD", "Greens"),
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
            f"Pagina {page_number} - amostra {sample_index} - rotulo real {label}\n"
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


def _representative_indices(labels: torch.Tensor, sample_count: int) -> list[int]:
    indices: list[int] = []
    seen_digits: set[int] = set()
    for index, label in enumerate(labels.tolist()):
        digit = int(label)
        if digit not in seen_digits:
            indices.append(index)
            seen_digits.add(digit)
        if len(indices) >= min(10, sample_count):
            break

    for index in range(len(labels)):
        if len(indices) >= sample_count:
            break
        if index not in indices:
            indices.append(index)
    return indices


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
