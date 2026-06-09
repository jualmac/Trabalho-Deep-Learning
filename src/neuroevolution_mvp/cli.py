"""Command line entry point for running the MVP experiment."""

from __future__ import annotations

import argparse
from pathlib import Path

from .config import ExperimentConfig
from .experiment import run_experiment


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="Run the NEAT + SGD MNIST MVP.")
    parser.add_argument("--generations", type=int, default=ExperimentConfig.neat_generations)
    parser.add_argument("--train-limit", type=int, default=ExperimentConfig.train_limit)
    parser.add_argument("--test-limit", type=int, default=ExperimentConfig.test_limit)
    parser.add_argument("--epochs", type=int, default=ExperimentConfig.baseline_epochs)
    parser.add_argument("--artifact-dir", type=Path, default=ExperimentConfig.artifact_dir)
    return parser


def main() -> None:
    args = build_parser().parse_args()
    config = ExperimentConfig(
        neat_generations=args.generations,
        train_limit=args.train_limit,
        test_limit=args.test_limit,
        baseline_epochs=args.epochs,
        evolved_sgd_epochs=args.epochs,
        artifact_dir=args.artifact_dir,
    )
    results = run_experiment(config)

    print("\nSonda de interpretacao")
    for name, value in results["interpretation_summary"].items():
        print(f"- {name}: {value}")

    print("\nControles do experimento")
    for name, value in results["summary"].items():
        print(f"- {name}: {value}")


if __name__ == "__main__":
    main()
