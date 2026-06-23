"""Command line entry point for running the MVP experiment."""

from __future__ import annotations

import argparse
from pathlib import Path

from .artifacts import load_json, save_json
from .config import (
    ALL_NEAT_VARIANT_CONFIGS,
    DEFAULT_ARTIFACT_DIR,
    DEFAULT_NEAT_VARIANT_CONFIGS,
    ExperimentConfig,
)
from .data import SUPPORTED_DATASETS
from .experiment import enrich_results_for_reporting, run_experiment
from .pdf_report import export_interpretability_pdf


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="Run the NEAT + SGD image-classification MVP.")
    variant_names = [path.stem for path in ALL_NEAT_VARIANT_CONFIGS]
    parser.add_argument("--seed", type=int, default=ExperimentConfig.seed)
    parser.add_argument(
        "--dataset",
        choices=SUPPORTED_DATASETS,
        default=ExperimentConfig.dataset,
        help="Dataset used by the experiment. Default keeps the original MNIST setup.",
    )
    parser.add_argument("--generations", type=int, default=ExperimentConfig.neat_generations)
    parser.add_argument(
        "--image-size",
        type=int,
        default=None,
        help="Spatial resolution. Defaults to 26 for MNIST and 32 for CIFAR-10.",
    )
    parser.add_argument("--train-limit", type=int, default=ExperimentConfig.train_limit)
    parser.add_argument("--test-limit", type=int, default=ExperimentConfig.test_limit)
    parser.add_argument(
        "--interpretability-samples",
        type=int,
        default=ExperimentConfig.interpretability_samples,
        help="Images used for attention metrics; 0 evaluates the full loaded test set.",
    )
    parser.add_argument("--epochs", type=int, default=ExperimentConfig.baseline_epochs)
    parser.add_argument("--evolved-epochs", type=int, default=ExperimentConfig.evolved_sgd_epochs)
    parser.add_argument(
        "--neat-variant",
        choices=["all", *variant_names],
        default="all",
        help="Use 'all' to benchmark every configured NEAT variant, or choose one variant.",
    )
    parser.add_argument("--artifact-dir", type=Path, default=None)
    parser.add_argument("--pdf-samples", type=int, default=12)
    parser.add_argument(
        "--report-only",
        action="store_true",
        help="Regenerate JSON summary, CSV, PNG and PDF from current artifacts without training.",
    )
    parser.add_argument(
        "--pdf-path",
        type=Path,
        default=None,
    )
    return parser


def main() -> None:
    args = build_parser().parse_args()
    image_size = args.image_size or _default_image_size(args.dataset)
    artifact_dir = args.artifact_dir or DEFAULT_ARTIFACT_DIR / args.dataset
    variant_paths = _selected_variant_paths(args.neat_variant, image_size, args.dataset)
    config = ExperimentConfig(
        seed=args.seed,
        dataset=args.dataset,
        image_size=image_size,
        neat_generations=args.generations,
        train_limit=args.train_limit,
        test_limit=args.test_limit,
        interpretability_samples=args.interpretability_samples,
        baseline_epochs=args.epochs,
        evolved_sgd_epochs=args.evolved_epochs,
        artifact_dir=artifact_dir,
        neat_variant_paths=variant_paths,
    )
    pdf_path = args.pdf_path or artifact_dir / f"{args.dataset}_interpretability_validation.pdf"
    if args.report_only:
        results_path = artifact_dir / "results.json"
        results = enrich_results_for_reporting(load_json(results_path))
        save_json(results_path, results)
        stored_config = results.get("config", {})
        config = ExperimentConfig(
            seed=int(stored_config.get("seed", args.seed)),
            dataset=str(stored_config.get("dataset", args.dataset)),
            image_size=int(stored_config.get("image_size", image_size)),
            test_limit=int(stored_config.get("test_limit", args.test_limit)),
            batch_size=int(stored_config.get("batch_size", ExperimentConfig.batch_size)),
            hidden_units=int(stored_config.get("hidden_units", ExperimentConfig.hidden_units)),
            artifact_dir=artifact_dir,
            neat_variant_paths=variant_paths,
        )
    else:
        results = run_experiment(config)
    pdf_path = export_interpretability_pdf(config, pdf_path, sample_count=args.pdf_samples)

    print("\nSonda de interpretacao")
    _print_interpretation_summary(results["interpretation_summary"])

    print("\nControles do experimento")
    for name, value in results["summary"].items():
        print(f"- {name}: {value}")

    print(f"\nPDF de validacao visual: {pdf_path}")
    print(
        "Tabela CSV: "
        f"{pdf_path.with_name(f'{args.dataset}_interpretability_summary.csv')}"
    )
    print(
        "Grafico PNG: "
        f"{pdf_path.with_name(f'{args.dataset}_interpretability_summary.png')}"
    )


def _selected_variant_paths(name: str, image_size: int, dataset: str) -> tuple[Path, ...]:
    if name == "all":
        return DEFAULT_NEAT_VARIANT_CONFIGS
    for path in ALL_NEAT_VARIANT_CONFIGS:
        if path.stem == name:
            if dataset == "cifar10" and _raw_pixel_variant(path):
                raise ValueError(
                    "Raw-pixel variants in this repo are grayscale MNIST configs. "
                    "For CIFAR-10, choose a features/prototypes/HyperNEAT-features variant."
                )
            if image_size != 14 and _requires_14x14_pixels(path):
                raise ValueError(
                    "This raw-pixel NEAT config expects 14x14 images. "
                    "Use --image-size 14 or choose a features/prototypes/HyperNEAT-features variant."
                )
            return (path,)
    raise ValueError(f"Unknown NEAT variant: {name}")


def _requires_14x14_pixels(path: Path) -> bool:
    stem = path.stem
    return "14x14" in stem


def _raw_pixel_variant(path: Path) -> bool:
    stem = path.stem
    return "14x14" in stem or stem == "hyperneat_mnist_cppn"


def _default_image_size(dataset: str) -> int:
    return 32 if dataset == "cifar10" else ExperimentConfig.image_size


def _print_interpretation_summary(summary: dict) -> None:
    primary = summary.get("primary_metric", {})
    print(f"- imagens avaliadas: {int(summary['probe_samples'])}")
    print(f"- escopo: {summary.get('evaluation_scope', 'desconhecido')}")
    print(
        "- divergencia de atencao do dataset (1 - Jaccard): "
        f"{_format_interval(primary.get('value'), primary.get('ci95', {}))}"
    )
    pair = summary.get("pairwise_conditioned", {}).get("neat_pure_vs_neat_sgd")
    if not pair:
        print(f"- cosseno NEAT/SGD: {summary['neat_pure_vs_neat_sgd_cosine']:.3f}")
        print(f"- pixels-chave comuns: {summary['neat_pure_vs_neat_sgd_top_overlap']:.1%}")
        return

    correct = pair["both_correct"]
    wrong = pair["both_wrong"]
    controlled_wrong = pair["both_wrong_same_prediction"]
    print(
        "- divergencia macro por classe: "
        f"{_format_optional(pair['dataset_attention_divergence_macro'])}"
    )
    print(
        "- divergencia quando preveem a mesma classe: "
        f"{_format_interval(pair['dataset_attention_divergence_same_prediction'], pair['dataset_attention_divergence_same_prediction_ci95'])} "
        f"(grupo={pair['same_prediction_sample_count']})"
    )
    print(f"- cobertura de mapas validos: {pair['valid_map_coverage']:.1%}")
    print(
        "- Jaccard ponderado, ambos acertam: "
        f"{_format_mean(correct['weighted_jaccard'])} (grupo={correct['sample_count']})"
    )
    print(
        "- Jaccard ponderado, ambos erram: "
        f"{_format_mean(wrong['weighted_jaccard'])} (grupo={wrong['sample_count']})"
    )
    print(
        "- Jaccard ponderado, ambos erram a mesma classe: "
        f"{_format_mean(controlled_wrong['weighted_jaccard'])} "
        f"(grupo={controlled_wrong['sample_count']})"
    )
    print(
        "- gap acerto - erro: "
        f"{_format_gap(pair['attention_agreement_gap_weighted_jaccard'], pair['attention_agreement_gap_weighted_jaccard_ci95'])}"
    )
    print(
        "- gap controlado acerto - erro: "
        f"{_format_gap(pair['controlled_attention_gap_weighted_jaccard'], pair['controlled_attention_gap_weighted_jaccard_ci95'])}"
    )
    print(
        "- erros conjuntos com classes diferentes: "
        f"{_format_percent(pair['wrong_prediction_disagreement'])}"
    )
    print("- detalhes e IC95%: results.json")


def _format_mean(metric: dict) -> str:
    value = metric.get("mean")
    if value is None:
        return "n/a"
    low = metric.get("ci95_low")
    high = metric.get("ci95_high")
    if low is None or high is None:
        return f"{value:.3f} (IC95% n/a)"
    return f"{value:.3f} (IC95% {low:.3f}-{high:.3f})"


def _format_optional(value: float | None, signed: bool = False) -> str:
    if value is None:
        return "n/a"
    return f"{value:+.3f}" if signed else f"{value:.3f}"


def _format_percent(value: float | None) -> str:
    return "n/a" if value is None else f"{value:.1%}"


def _format_gap(value: float | None, interval: dict) -> str:
    formatted = _format_optional(value, signed=True)
    low = interval.get("low")
    high = interval.get("high")
    if low is None or high is None:
        return f"{formatted} (IC95% bootstrap n/a)"
    return f"{formatted} (IC95% bootstrap {low:+.3f} a {high:+.3f})"


def _format_interval(value: float | None, interval: dict) -> str:
    formatted = _format_optional(value)
    low = interval.get("low")
    high = interval.get("high")
    if low is None or high is None:
        return f"{formatted} (IC95% bootstrap n/a)"
    return f"{formatted} (IC95% bootstrap {low:.3f} a {high:.3f})"


if __name__ == "__main__":
    main()
