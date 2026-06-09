"""End-to-end orchestration for the neuroevolution MVP."""

from __future__ import annotations

from pathlib import Path
from typing import Any

from .artifacts import save_genome, save_json, save_model
from .config import ExperimentConfig
from .data import load_mnist, set_seed
from .interpretability import compare_maps, occlusion_map, prediction_summary
from .models import BaselineMLP, EvolvedTopologyMLP, FixedNeatTopologyMLP
from .neat_runner import evaluate_winner, evolve_genome, load_neat_config
from .training import train_classifier


def run_experiment(config: ExperimentConfig) -> dict[str, Any]:
    """Run baseline SGD, NEAT, and evolved-topology SGD comparisons."""

    set_seed(config.seed)
    config.artifact_dir.mkdir(parents=True, exist_ok=True)

    data = load_mnist(
        image_size=config.image_size,
        train_limit=config.train_limit,
        test_limit=config.test_limit,
        batch_size=config.batch_size,
        seed=config.seed,
    )

    baseline = BaselineMLP(config.input_size, config.hidden_units)
    baseline_history = train_classifier(
        baseline,
        data.train_loader,
        data.test_loader,
        epochs=config.baseline_epochs,
        learning_rate=config.learning_rate,
    )

    neat_config = load_neat_config(config.neat_config_path)
    winner, neat_history = evolve_genome(
        neat_config,
        data.x_train,
        data.y_train,
        generations=config.neat_generations,
        eval_limit=config.neat_eval_limit,
    )
    neat_accuracy = evaluate_winner(
        winner,
        neat_config,
        data.x_test,
        data.y_test,
        limit=config.neat_winner_eval_limit,
    )

    evolved_sgd = EvolvedTopologyMLP(winner, neat_config, input_size=config.input_size)
    evolved_sgd_history = train_classifier(
        evolved_sgd,
        data.train_loader,
        data.test_loader,
        epochs=config.evolved_sgd_epochs,
        learning_rate=config.learning_rate,
    )
    neat_pure = FixedNeatTopologyMLP(winner, neat_config, input_size=config.input_size)
    interpretation_summary = _compare_interpretations(
        baseline,
        neat_pure,
        evolved_sgd,
        data.x_test,
        image_size=config.image_size,
        sample_count=min(24, len(data.x_test)),
    )

    results = {
        "config": _serializable_config(config),
        "summary": {
            "baseline_sgd_accuracy": baseline_history[-1]["test_accuracy"],
            "neat_accuracy": neat_accuracy,
            "evolved_topology_sgd_accuracy": evolved_sgd_history[-1]["test_accuracy"],
            "evolved_nodes": len(winner.nodes),
            "evolved_enabled_connections": sum(
                1 for connection in winner.connections.values() if connection.enabled
            ),
        },
        "interpretation_summary": interpretation_summary,
        "baseline_history": baseline_history,
        "neat_history": neat_history,
        "evolved_sgd_history": evolved_sgd_history,
    }

    save_json(config.artifact_dir / "results.json", results)
    save_genome(config.artifact_dir / "winner_genome.pkl", winner)
    save_model(config.artifact_dir / "baseline_mlp.pt", baseline)
    save_model(config.artifact_dir / "evolved_topology_sgd.pt", evolved_sgd)
    return results


def _compare_interpretations(
    baseline: BaselineMLP,
    neat_pure: FixedNeatTopologyMLP,
    evolved_sgd: EvolvedTopologyMLP,
    x_test,
    image_size: int,
    sample_count: int,
) -> dict[str, float]:
    """Compare attribution maps over a small fixed probe set."""

    baseline_vs_sgd_rows = []
    neat_vs_sgd_rows = []
    baseline_vs_neat_rows = []
    neat_sgd_prediction_agreements = []

    for image in x_test[:sample_count]:
        reshaped = image.reshape(image_size, image_size)
        baseline_map = occlusion_map(baseline, reshaped, image_size)
        neat_map = occlusion_map(neat_pure, reshaped, image_size)
        evolved_map = occlusion_map(evolved_sgd, reshaped, image_size)
        baseline_vs_sgd_rows.append(compare_maps(baseline_map, evolved_map))
        neat_vs_sgd_rows.append(compare_maps(neat_map, evolved_map))
        baseline_vs_neat_rows.append(compare_maps(baseline_map, neat_map))

        neat_prediction = prediction_summary(neat_pure, reshaped)
        evolved_prediction = prediction_summary(evolved_sgd, reshaped)
        neat_sgd_prediction_agreements.append(
            float(neat_prediction["class"] == evolved_prediction["class"])
        )

    return {
        "probe_samples": float(sample_count),
        "baseline_vs_neat_sgd_cosine": _mean(baseline_vs_sgd_rows, "cosine_similarity"),
        "baseline_vs_neat_sgd_top_overlap": _mean(baseline_vs_sgd_rows, "top_pixel_overlap"),
        "baseline_vs_neat_pure_cosine": _mean(baseline_vs_neat_rows, "cosine_similarity"),
        "neat_pure_vs_neat_sgd_cosine": _mean(neat_vs_sgd_rows, "cosine_similarity"),
        "neat_pure_vs_neat_sgd_top_overlap": _mean(neat_vs_sgd_rows, "top_pixel_overlap"),
        "neat_pure_vs_neat_sgd_difference": _mean(neat_vs_sgd_rows, "mean_absolute_difference"),
        "neat_pure_vs_neat_sgd_prediction_agreement": (
            sum(neat_sgd_prediction_agreements)
            / max(len(neat_sgd_prediction_agreements), 1)
        ),
    }


def artifact_exists(artifact_dir: Path) -> bool:
    """Return whether a previous run can be loaded by the UI."""

    return (artifact_dir / "results.json").exists() and (
        artifact_dir / "winner_genome.pkl"
    ).exists()


def _mean(rows: list[dict[str, float]], key: str) -> float:
    return sum(row[key] for row in rows) / max(len(rows), 1)


def _serializable_config(config: ExperimentConfig) -> dict[str, Any]:
    payload = dict(config.__dict__)
    payload["artifact_dir"] = str(config.artifact_dir)
    payload["neat_config_path"] = str(config.neat_config_path)
    return payload
