"""PDF export for visual interpretability validation."""

from __future__ import annotations

import argparse
import os
from dataclasses import fields
from pathlib import Path
from typing import Any
os.environ.setdefault("MPLCONFIGDIR", "/tmp/matplotlib")

import matplotlib.pyplot as plt
import torch
import torch.nn as nn
from matplotlib.backends.backend_pdf import PdfPages

import pandas as pd

from src.neuroevolution_mvp.export_interpretability_metrics import dad
from src.neuroevolution_mvp.export_interpretability_metrics import weighted_jaccard

from .data import load_dataset
from .interpretability import compare_maps, occlusion_map, prediction_summary, block_occlusion_map
from .artifacts import load_genome, load_json
from .config import ALL_NEAT_VARIANT_CONFIGS, ExperimentConfig
from .features import FEATURE_SIZE, PROTOTYPE_FEATURE_SIZE
from .models import (
    BaselineMLP,
    CnnEmbeddingWrappedModel,
    EvolvedTopologyMLP,
    FeatureWrappedModel,
    FixedNeatTopologyMLP,
    GeneratedSubstrateMLP,
    PrototypeWrappedModel,
    SmallCifarCNN,
)
from .neat_runner import (
    build_hyperneat_feature_substrate,
    build_hyperneat_prototype_substrate,
    build_hyperneat_substrate,
    load_neat_config,
)

class FlattenInputModel(nn.Module):
    def __init__(self, model: nn.Module) -> None:
        super().__init__()
        self.model = model

    def forward(self, features: torch.Tensor) -> torch.Tensor:
        if features.ndim > 2:
            features = features.reshape(features.size(0), -1)

        return self.model(features)

def _load_results_from_artifact(artifact_dir: Path) -> dict[str, Any]:
    default_path = artifact_dir / "results.json"

    if default_path.exists():
        return load_json(default_path)

    result_files = sorted(artifact_dir.glob("results*.json"))

    if not result_files:
        raise FileNotFoundError(
            f"No results JSON found in artifact dir: {artifact_dir}"
        )

    return load_json(result_files[0])


def export_interpretability_pdf(
    config: ExperimentConfig,
    output_path: Path,
    sample_count: int = 12,
    pdf_group: str = "both_correct",
    occlusion_window: int = 3,
) -> Path:
    """Export the image and NEAT/SGD attention maps to a standalone PDF."""

    output_path.parent.mkdir(parents=True, exist_ok=True)
    results = _load_results_from_artifact(config.artifact_dir)
    summary = results.get("summary", {})
    family = str(summary.get("selected_neat_family", "direct")).strip().lower()

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

    use_cnn_embeddings = config.dataset in {"cifar10", "cifar100_10"}

    image_channels = data.image_channels
    raw_input_size = config.image_size * config.image_size * image_channels

    if use_cnn_embeddings:
        baseline = SmallCifarCNN(
            image_size=config.image_size,
            num_classes=10,
            feature_size=128,
        )
        neat_input_size = neat_config.genome_config.num_inputs
    else:
        baseline = BaselineMLP(
            raw_input_size,
            config.hidden_units,
            image_channels=image_channels,
        )
        neat_input_size = raw_input_size

    neat_pure, evolved = _load_neat_pair(
        family=family,
        genome=genome,
        neat_config=neat_config,
        input_size=neat_input_size,
        image_size=config.image_size,
        artifact_dir=config.artifact_dir,
    )

    baseline.load_state_dict(
        torch.load(
            config.artifact_dir / "baseline_mlp.pt",
            map_location="cpu",
        )
    )

    evolved.load_state_dict(
        torch.load(
            config.artifact_dir / "evolved_topology_sgd.pt",
            map_location="cpu",
        )
    )

    if use_cnn_embeddings:
        neat_pure = CnnEmbeddingWrappedModel(
            encoder=baseline,
            classifier=neat_pure,
        )
        evolved = CnnEmbeddingWrappedModel(
            encoder=baseline,
            classifier=evolved,
        )
    else:
        neat_pure = FlattenInputModel(neat_pure)
        evolved = FlattenInputModel(evolved)

    groups = _correctness_groups(
        labels=data.y_test,
        x_test=data.x_test,
        baseline=baseline,
        evolved=evolved,
        image_size=config.image_size,
        image_channels=image_channels
    )

    indices = groups[pdf_group][:sample_count]

    if len(indices) < sample_count:
        remaining = sample_count - len(indices)
        indices.extend(groups["evolved_correct_baseline_wrong"][:remaining])

    records: list[dict[str, int | float]] = []

    with PdfPages(output_path) as pdf:
        _write_cover(pdf, results, len(indices), config.dataset)
        for page_number, index in enumerate(indices, start=1):
            image = _reshape_image(
                data.x_test[index],
                config.image_size,
                image_channels,
            )
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
                        records=records,
                        occlusion_window=occlusion_window,
                    )
            )
            plt.close()
    
    records_data = pd.DataFrame(records)
    records_data.to_csv(
        config.artifact_dir / "interpretability_records.csv",
        index=False,
    )
    return output_path


def _load_neat_pair(
    family: str,
    genome,
    neat_config,
    input_size: int,
    image_size: int,
    artifact_dir: Path,
):
    
    family_name = family.strip().lower()

    is_cppn_config = (
    neat_config.genome_config.num_inputs == 5
    and neat_config.genome_config.num_outputs == 1
    )

    if family_name == "direct" and is_cppn_config:
        family_name = "hyperneat"

    if family_name == "hyperneat":
        weight_matrix, bias = build_hyperneat_substrate(
            genome=genome,
            config=neat_config,
            image_size=image_size,
        )       

        weight_tensor = torch.from_numpy(weight_matrix).float()
        bias_tensor = torch.from_numpy(bias).float()

        neat_pure = GeneratedSubstrateMLP(
            weight_matrix=weight_tensor,
            bias=bias_tensor,
            trainable=False,
        )
        evolved = GeneratedSubstrateMLP(
            weight_matrix=weight_tensor,
            bias=bias_tensor,
            trainable=True,
        )

    elif family_name == "hyperneat_features":
        weight_matrix, bias = build_hyperneat_feature_substrate(
            genome=genome,
            config=neat_config,
        )

        weight_tensor = torch.from_numpy(weight_matrix).float()
        bias_tensor = torch.from_numpy(bias).float()

        neat_pure = FeatureWrappedModel(
            GeneratedSubstrateMLP(
                weight_matrix=weight_tensor,
                bias=bias_tensor,
                trainable=False,
            ),
            image_size,
        )
        evolved = FeatureWrappedModel(
            GeneratedSubstrateMLP(
                weight_matrix=weight_tensor,
                bias=bias_tensor,
                trainable=True,
            ),
            image_size,
        )

    elif family_name == "hyperneat_prototypes":
        centroids = torch.load(
            artifact_dir / "prototype_centroids.pt",
            map_location="cpu",
        )

        weight_matrix, bias = build_hyperneat_prototype_substrate(
            genome=genome,
            config=neat_config,
        )

        weight_tensor = torch.from_numpy(weight_matrix).float()
        bias_tensor = torch.from_numpy(bias).float()

        neat_pure = PrototypeWrappedModel(
            GeneratedSubstrateMLP(
                weight_matrix=weight_tensor,
                bias=bias_tensor,
                trainable=False,
            ),
            centroids,
        )
        evolved = PrototypeWrappedModel(
            GeneratedSubstrateMLP(
                weight_matrix=weight_tensor,
                bias=bias_tensor,
                trainable=True,
            ),
            centroids,
        )

    elif family_name in {"prototypes", "seeded_prototypes"}:
        centroids = torch.load(
            artifact_dir / "prototype_centroids.pt",
            map_location="cpu",
        )

        neat_pure = PrototypeWrappedModel(
            FixedNeatTopologyMLP(
                genome,
                neat_config,
                input_size=PROTOTYPE_FEATURE_SIZE,
            ),
            centroids,
        )
        evolved = PrototypeWrappedModel(
            EvolvedTopologyMLP(
                genome,
                neat_config,
                input_size=PROTOTYPE_FEATURE_SIZE,
            ),
            centroids,
        )

    elif family_name == "features":
        neat_pure = FeatureWrappedModel(
            FixedNeatTopologyMLP(
                genome,
                neat_config,
                input_size=FEATURE_SIZE,
            ),
            image_size,
        )
        evolved = FeatureWrappedModel(
            EvolvedTopologyMLP(
                genome,
                neat_config,
                input_size=FEATURE_SIZE,
            ),
            image_size,
        )

    else:
        neat_pure = FixedNeatTopologyMLP(
            genome,
            neat_config,
            input_size=input_size,
        )
        evolved = EvolvedTopologyMLP(
            genome,
            neat_config,
            input_size=input_size,
        )

    return neat_pure, evolved

def _write_cover(
    pdf: PdfPages,
    results: dict[str, Any],
    sample_count: int,
    dataset_name: str,
) -> None:
    summary = results.get("summary", {})
    variants = results.get("neat_variant_results", [])
    optimizer_name = str(summary.get("optimizer", "sgd")).upper()
    learning_rate = summary.get("learning_rate", None)
    selected_family = str(summary.get("selected_neat_family", "direct"))
    selected_variant = str(summary.get("selected_neat_variant", "desconhecida"))
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
        f"Retreino dos pesos: {optimizer_name}",
        f"Learning rate: {learning_rate if learning_rate is not None else 'n/a'}",
        f"Familia selecionada: {selected_family}",
        f"Variante escolhida: {selected_variant}",
        f"Baseline + {optimizer_name}: {_fmt_accuracy(summary.get('baseline_sgd_accuracy'))}",
        f"NEAT puro: {_fmt_accuracy(summary.get('neat_accuracy'))}",
        f"Topologia NEAT + {optimizer_name}: {_fmt_accuracy(summary.get('evolved_topology_sgd_accuracy'))}",
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
    baseline,
    neat_pure,
    evolved,
    image_size: int,
    records: list[dict[str, int | float]],
    occlusion_window: int,
):
    baseline_map = block_occlusion_map(
        baseline,
        image,
        image_size,
        patch_size=occlusion_window,
        stride=1,
    )
    neat_map = block_occlusion_map(
        neat_pure,
        image,
        image_size,
        patch_size=occlusion_window,
        stride=1,
    )
    evolved_map = block_occlusion_map(
        evolved,
        image,
        image_size,
        patch_size=occlusion_window,
        stride=1,
    )

    difference_map = torch.abs(neat_map - evolved_map)
    comparison = compare_maps(neat_map, evolved_map)

    predictions = [
        ("Baseline", prediction_summary(baseline, image)),
        ("NEAT puro", prediction_summary(neat_pure, image)),
        ("NEAT + SGD", prediction_summary(evolved, image)),
    ]

    baseline_prediction = int(predictions[0][1]["class"])
    neat_pure_prediction = int(predictions[1][1]["class"])
    evolved_prediction = int(predictions[2][1]["class"])

    neat_map_numpy = neat_map.detach().cpu().numpy()
    evolved_map_numpy = evolved_map.detach().cpu().numpy()

    record = {
        "sample_index": int(sample_index),
        "true_label": int(label),
        "baseline_prediction": baseline_prediction,
        "neat_pure_prediction": neat_pure_prediction,
        "neat_gradient_prediction": evolved_prediction,
        "baseline_correct": int(baseline_prediction == label),
        "neat_pure_correct": int(neat_pure_prediction == label),
        "neat_gradient_correct": int(evolved_prediction == label),
        "same_prediction": int(neat_pure_prediction == evolved_prediction),
        "weighted_jaccard": weighted_jaccard(neat_map_numpy, evolved_map_numpy),
        "dad": dad(neat_map_numpy, evolved_map_numpy),
    }

    records.append(record)

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
            ax.imshow(
                matrix.detach().permute(1, 2, 0).numpy(),
                interpolation="nearest",
            )
        else:
            ax.imshow(
                matrix.detach().numpy(),
                cmap=cmap,
                interpolation="nearest",
            )
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
            f"{prediction_text} | similaridade NEAT/SGD: "
            f"{comparison['cosine_similarity']:.2f} | "
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


def _selected_config_path(
    config: ExperimentConfig,
    results: dict,
) -> Path:
    selected_variant = str(
        results.get("summary", {}).get("selected_neat_variant", "")
    )

    for row in results.get("neat_variant_results", []):
        if str(row.get("variant", "")) == selected_variant:
            return Path(str(row["config_path"]))

    candidate_paths = (
        *config.neat_variant_paths,
        *ALL_NEAT_VARIANT_CONFIGS,
        config.neat_config_path,
    )

    for config_path in candidate_paths:
        if config_path.stem == selected_variant:
            return config_path

    raise ValueError(
        f"Could not find config path for selected variant: {selected_variant}"
    )

def _predicted_class(model, image: torch.Tensor) -> int:
    prediction = prediction_summary(model, image)
    return int(prediction["class"])


def _correctness_groups(
    labels: torch.Tensor,
    x_test: torch.Tensor,
    baseline,
    evolved,
    image_size: int,
    image_channels: int,
) -> dict[str, list[int]]:
    groups = {
        "both_correct": [],
        "baseline_correct_evolved_wrong": [],
        "evolved_correct_baseline_wrong": [],
        "both_wrong": [],
    }

    for index, label_tensor in enumerate(labels):
        label = int(label_tensor.item())
        image = _reshape_image(x_test[index], image_size, image_channels)

        baseline_correct = _predicted_class(baseline, image) == label
        evolved_correct = _predicted_class(evolved, image) == label

        if baseline_correct and evolved_correct:
            groups["both_correct"].append(index)
        elif baseline_correct and not evolved_correct:
            groups["baseline_correct_evolved_wrong"].append(index)
        elif evolved_correct and not baseline_correct:
            groups["evolved_correct_baseline_wrong"].append(index)
        else:
            groups["both_wrong"].append(index)

    return groups

def _fmt_accuracy(value: object) -> str:
    if isinstance(value, int | float):
        return f"{value:.1%}"
    return "n/a"
def _config_from_artifact(artifact_dir: Path) -> ExperimentConfig:
    results = _load_results_from_artifact(artifact_dir)
    config_values = dict(results.get("config", {}))

    valid_fields = {field.name for field in fields(ExperimentConfig)}
    filtered_values = {
        key: value
        for key, value in config_values.items()
        if key in valid_fields
    }

    filtered_values["artifact_dir"] = artifact_dir

    if "neat_config_path" in filtered_values:
        filtered_values["neat_config_path"] = Path(
            str(filtered_values["neat_config_path"])
        )

    if "neat_variant_paths" in filtered_values:
        filtered_values["neat_variant_paths"] = tuple(
            Path(str(path))
            for path in filtered_values["neat_variant_paths"]
        )

    return ExperimentConfig(**filtered_values)


def _parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Export interpretability PDF and per-sample metrics."
    )
    parser.add_argument(
        "--artifact-dir",
        type=Path,
        required=True,
        help="Directory containing trained artifacts.",
    )
    parser.add_argument(
        "--num-samples",
        type=int,
        default=12,
        help="Number of samples to include in the PDF and CSV.",
    )
    parser.add_argument(
        "--occlusion-window",
        type=int,
        default=3,
        help="Occlusion patch size.",
    )
    parser.add_argument(
        "--pdf-group",
        type=str,
        default="both_correct",
        choices=[
            "both_correct",
            "baseline_correct_evolved_wrong",
            "evolved_correct_baseline_wrong",
            "both_wrong",
        ],
        help="Group of samples used in the PDF.",
    )
    parser.add_argument("--output-name", type=str, default=None, help="Nome do PDF de saída.")
    return parser.parse_args()


def main() -> None:
    args = _parse_args()
    config = _config_from_artifact(args.artifact_dir)

    output_name = args.output_name or f"{config.dataset}_interpretability_validation.pdf"
    output_path = args.artifact_dir / output_name
    
    pdf_path = export_interpretability_pdf(
        config=config,
        output_path=output_path,
        sample_count=args.num_samples,
        pdf_group=args.pdf_group,
        occlusion_window=args.occlusion_window,
    )

    records_path = args.artifact_dir / "interpretability_records.csv"

    print(f"PDF salvo em: {pdf_path}")
    print(f"CSV salvo em: {records_path}")


if __name__ == "__main__":
    main()