"""Command line entry point for running the MVP experiment."""

from __future__ import annotations

import argparse
from pathlib import Path

from .config import (
    ALL_NEAT_VARIANT_CONFIGS,
    DEFAULT_ARTIFACT_DIR,
    DEFAULT_NEAT_VARIANT_CONFIGS,
    ExperimentConfig,
)
from .data import SUPPORTED_DATASETS
from .experiment import run_experiment
from .pdf_report import export_interpretability_pdf


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="Run the NEAT + SGD image-classification MVP.")
    variant_names = [path.stem for path in ALL_NEAT_VARIANT_CONFIGS]
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
        dataset=args.dataset,
        image_size=image_size,
        neat_generations=args.generations,
        train_limit=args.train_limit,
        test_limit=args.test_limit,
        baseline_epochs=args.epochs,
        evolved_sgd_epochs=args.evolved_epochs,
        artifact_dir=artifact_dir,
        neat_variant_paths=variant_paths,
    )
    results = run_experiment(config)
    pdf_path = args.pdf_path or artifact_dir / f"{args.dataset}_interpretability_validation.pdf"
    pdf_path = export_interpretability_pdf(config, pdf_path, sample_count=args.pdf_samples)

    print("\nSonda de interpretacao")
    for name, value in results["interpretation_summary"].items():
        print(f"- {name}: {value}")

    print("\nControles do experimento")
    for name, value in results["summary"].items():
        print(f"- {name}: {value}")

    print(f"\nPDF de validacao visual: {pdf_path}")


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


if __name__ == "__main__":
    main()
