"""End-to-end orchestration for the neuroevolution MVP."""

from __future__ import annotations

from logging import config
from pathlib import Path
from typing import Any

import torch
from torch.utils.data import DataLoader, TensorDataset


from .artifacts import save_genome, save_json, save_model
from .config import ExperimentConfig
from .data import load_dataset, set_seed
from .interpretability import compare_maps, occlusion_map, prediction_summary
from .features import FEATURE_SIZE, PROTOTYPE_FEATURE_SIZE
from .models import (
    BaselineMLP,
    EvolvedTopologyMLP,
    FeatureWrappedModel,
    FixedNeatTopologyMLP,
    GeneratedSubstrateMLP,
    PrototypeWrappedModel,
    SmallCifarCNN,
    CnnEmbeddingWrappedModel
)
from .neat_runner import (
    build_hyperneat_feature_substrate,
    build_hyperneat_prototype_substrate,
    build_hyperneat_substrate,
    evolve_variants,
)
from .training import train_classifier, extract_cnn_embeddings


def run_experiment(config: ExperimentConfig) -> dict[str, Any]:
    """Run baseline SGD, NEAT, and evolved-topology SGD comparisons."""

    set_seed(config.seed)
    config.artifact_dir.mkdir(parents=True, exist_ok=True)

    data = load_dataset(
        config.dataset,
        image_size=config.image_size,
        train_limit=config.train_limit,
        test_limit=config.test_limit,
        batch_size=config.batch_size,
        seed=config.seed,
    )

    image_channels = data.image_channels
    raw_input_size = config.image_size * config.image_size * image_channels
    use_cnn_embeddings = config.dataset in {"cifar10", "cifar100_10"}

    if use_cnn_embeddings:
        baseline = SmallCifarCNN(
            image_size=config.image_size,
            num_classes=10,
            feature_size=128,
        )
    else:
        baseline = BaselineMLP(
            raw_input_size,
            config.hidden_units,
            image_channels=image_channels,
        )

    baseline_history = train_classifier(
        baseline,
        data.train_loader,
        data.test_loader,
        epochs=config.baseline_epochs,
        learning_rate=config.learning_rate,
        optimizer_name=config.optimizer_name,
    )

    if use_cnn_embeddings:
        x_train_for_neat = extract_cnn_embeddings(
            model=baseline,
            inputs=data.x_train,
            image_size=config.image_size,
        )
        x_test_for_neat = extract_cnn_embeddings(
            model=baseline,
            inputs=data.x_test,
            image_size=config.image_size,
        )
        input_size_for_neat = 128

        train_loader_for_candidate = DataLoader(
            TensorDataset(x_train_for_neat, data.y_train),
            batch_size=config.batch_size,
            shuffle=True,
        )
        test_loader_for_candidate = DataLoader(
            TensorDataset(x_test_for_neat, data.y_test),
            batch_size=config.batch_size,
            shuffle=False,
        )
    else:
        x_train_for_neat = data.x_train
        x_test_for_neat = data.x_test
        input_size_for_neat = raw_input_size
        train_loader_for_candidate = data.train_loader
        test_loader_for_candidate = data.test_loader

    
    evolved_candidates = []
    neat_variants = evolve_variants(
        config.neat_variant_paths or (config.neat_config_path,),
        x_train_for_neat,
        data.y_train,
        x_test_for_neat,
        data.y_test,
        image_size=config.image_size,
        generations=config.neat_generations,
        eval_limit=config.neat_eval_limit,
        winner_eval_limit=config.neat_winner_eval_limit,
    )

    for index, candidate in enumerate(neat_variants):
        set_seed(config.seed + 101 + index)

        candidate_model = _candidate_sgd_model(
            candidate,
            input_size_for_neat,
            config.image_size,
        )

        candidate_history = train_classifier(
            candidate_model,
            train_loader_for_candidate,
            test_loader_for_candidate,
            epochs=config.evolved_sgd_epochs,
            learning_rate=config.learning_rate,
            optimizer_name=config.optimizer_name,
        )

        if use_cnn_embeddings:
            train_loader_for_candidate = DataLoader(
                TensorDataset(x_train_for_neat, data.y_train),
                batch_size=config.batch_size,
                shuffle=True,
            )
            test_loader_for_candidate = DataLoader(
                TensorDataset(x_test_for_neat, data.y_test),
                batch_size=config.batch_size,
                shuffle=False,
            )
        else:
            train_loader_for_candidate = data.train_loader
            test_loader_for_candidate = data.test_loader
        
        candidate["result"]["evolved_topology_accuracy"] = _best_accuracy(candidate_history)
        candidate_payload = {
            "winner": candidate["winner"],
            "config": candidate["config"],
            "history": candidate["history"],
            "result": candidate["result"],
            "model": candidate_model,
            "model_history": candidate_history,
        }
        if "centroids" in candidate:
            candidate_payload["centroids"] = candidate["centroids"]
        evolved_candidates.append(candidate_payload)

    best_candidate = max(
    evolved_candidates,
    key=lambda row: (
        row["result"].get("evolved_topology_accuracy", 0.0),
        row["result"].get("accuracy", 0.0),
        row["result"].get("best_fitness", 0.0),
    ),
    )

    winner = best_candidate["winner"]
    neat_config = best_candidate["config"]
    neat_history = best_candidate["history"]
    neat_accuracy = best_candidate["result"]["accuracy"]
    evolved_sgd = best_candidate["model"]
    evolved_sgd_history = best_candidate["model_history"]
    neat_variant_results = [candidate["result"] for candidate in evolved_candidates]

    neat_pure = _candidate_pure_model(
            best_candidate,
            input_size_for_neat,
            config.image_size,
        )

    if use_cnn_embeddings:
        neat_pure_for_interpretation = CnnEmbeddingWrappedModel(
            encoder=baseline,
            classifier=neat_pure,
        )
        evolved_for_interpretation = CnnEmbeddingWrappedModel(
            encoder=baseline,
            classifier=evolved_sgd,
        )
    else:
        neat_pure_for_interpretation = neat_pure
        evolved_for_interpretation = evolved_sgd

    interpretation_summary = _compare_interpretations(
        baseline,
        neat_pure,
        evolved_sgd,
        data.x_test,
        image_size=config.image_size,
        image_channels=config.image_channels,
        sample_count=min(config.probe_samples, len(data.x_test)),
    )

    results = {
        "config": _serializable_config(config),
        "summary": {
            "baseline_sgd_accuracy": _best_accuracy(baseline_history),
            "neat_accuracy": neat_accuracy,
            "evolved_topology_sgd_accuracy": _best_accuracy(evolved_sgd_history),
            "optimizer": config.optimizer_name,
            "dataset": config.dataset,
            "image_size": config.image_size,
            "image_channels": image_channels,
            "selected_neat_variant": str(best_candidate["result"]["variant"]),
            "selected_neat_family": str(best_candidate["result"].get("family", "direct")),
            "evolved_nodes": len(winner.nodes),
            "evolved_enabled_connections": sum(
                1 for connection in winner.connections.values() if connection.enabled
            ),
        },
        "interpretation_summary": interpretation_summary,
        "baseline_history": baseline_history,
        "neat_history": neat_history,
        "neat_variant_results": neat_variant_results,
        "evolved_sgd_history": evolved_sgd_history,
    }

    save_json(config.artifact_dir / "results.json", results)
    save_genome(config.artifact_dir / "winner_genome.pkl", winner)
    save_model(config.artifact_dir / "evolved_topology_sgd.pt", evolved_sgd)
    save_model(config.artifact_dir / "baseline_mlp.pt", baseline)
    if best_candidate["result"].get("family") in {"prototypes", "seeded_prototypes", "hyperneat_prototypes"}:
        torch.save(best_candidate["centroids"], config.artifact_dir / "prototype_centroids.pt")
    return results


def _compare_interpretations(
    baseline: BaselineMLP,
    neat_pure: FixedNeatTopologyMLP,
    evolved_sgd: EvolvedTopologyMLP,
    x_test,
    image_size: int,
    image_channels: int,
    sample_count: int,
) -> dict[str, float]:
    """Compare attribution maps over a small fixed probe set."""

    baseline_vs_sgd_rows = []
    neat_vs_sgd_rows = []
    baseline_vs_neat_rows = []
    neat_sgd_prediction_agreements = []

    for image in x_test[:sample_count]:
        reshaped = _reshape_image(image, image_size, image_channels)
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


def _reshape_image(image: torch.Tensor, image_size: int, image_channels: int) -> torch.Tensor:
    if image_channels == 1:
        return image.reshape(image_size, image_size)
    return image.reshape(image_channels, image_size, image_size)

def _best_accuracy(history: list[dict[str, float]]) -> float:
    return max((row["test_accuracy"] for row in history), default=0.0)

def _candidate_sgd_model(candidate: dict[str, Any], input_size: int, image_size: int):
    if candidate["result"].get("family") in {"prototypes", "seeded_prototypes"}:
        classifier = EvolvedTopologyMLP(
            candidate["winner"],
            candidate["config"],
            input_size=PROTOTYPE_FEATURE_SIZE,
            initialize_from_genome=True,
        )
        return PrototypeWrappedModel(classifier, candidate["centroids"])

    if candidate["result"].get("family") == "hyperneat_prototypes":
        weight_matrix, bias = build_hyperneat_prototype_substrate(
            candidate["winner"],
            candidate["config"],
        )
        classifier = GeneratedSubstrateMLP(
            weight_matrix=torch.from_numpy(weight_matrix).float(),
            bias=torch.from_numpy(bias).float(),
            trainable=True,
        )
        return PrototypeWrappedModel(classifier, candidate["centroids"])

    if candidate["result"].get("family") == "features":
        classifier = EvolvedTopologyMLP(
            candidate["winner"],
            candidate["config"],
            input_size=FEATURE_SIZE,
            initialize_from_genome=True,
        )
        return FeatureWrappedModel(classifier, image_size)

    if candidate["result"].get("family") == "hyperneat_features":
        weight_matrix, bias = build_hyperneat_feature_substrate(
            candidate["winner"],
            candidate["config"],
        )
        classifier = GeneratedSubstrateMLP(
            weight_matrix=torch.from_numpy(weight_matrix).float(),
            bias=torch.from_numpy(bias).float(),
            trainable=True,
        )
        return FeatureWrappedModel(classifier, image_size)

    if candidate["result"].get("family") == "hyperneat":
        weight_matrix, bias = build_hyperneat_substrate(
            candidate["winner"],
            candidate["config"],
            image_size=image_size,
        )
        return GeneratedSubstrateMLP(
            weight_matrix=torch.from_numpy(weight_matrix).float(),
            bias=torch.from_numpy(bias).float(),
            trainable=True,
        )

    return EvolvedTopologyMLP(
        candidate["winner"],
        candidate["config"],
        input_size=input_size,
        initialize_from_genome=True,
    )


def _candidate_pure_model(candidate: dict[str, Any], input_size: int, image_size: int):
    if candidate["result"].get("family") in {"prototypes", "seeded_prototypes"}:
        classifier = FixedNeatTopologyMLP(
            candidate["winner"],
            candidate["config"],
            input_size=PROTOTYPE_FEATURE_SIZE,
        )
        return PrototypeWrappedModel(classifier, candidate["centroids"])

    if candidate["result"].get("family") == "hyperneat_prototypes":
        weight_matrix, bias = build_hyperneat_prototype_substrate(
            candidate["winner"],
            candidate["config"],
        )
        classifier = GeneratedSubstrateMLP(
            weight_matrix=torch.from_numpy(weight_matrix).float(),
            bias=torch.from_numpy(bias).float(),
            trainable=False,
        )
        return PrototypeWrappedModel(classifier, candidate["centroids"])

    if candidate["result"].get("family") == "features":
        classifier = FixedNeatTopologyMLP(
            candidate["winner"],
            candidate["config"],
            input_size=FEATURE_SIZE,
        )
        return FeatureWrappedModel(classifier, image_size)

    if candidate["result"].get("family") == "hyperneat_features":
        weight_matrix, bias = build_hyperneat_feature_substrate(
            candidate["winner"],
            candidate["config"],
        )
        classifier = GeneratedSubstrateMLP(
            weight_matrix=torch.from_numpy(weight_matrix).float(),
            bias=torch.from_numpy(bias).float(),
            trainable=False,
        )
        return FeatureWrappedModel(classifier, image_size)

    if candidate["result"].get("family") == "hyperneat":
        weight_matrix, bias = build_hyperneat_substrate(
            candidate["winner"],
            candidate["config"],
            image_size=image_size,
        )
        return GeneratedSubstrateMLP(
            weight_matrix=torch.from_numpy(weight_matrix).float(),
            bias=torch.from_numpy(bias).float(),
            trainable=False,
        )

    return FixedNeatTopologyMLP(
        candidate["winner"],
        candidate["config"],
        input_size=input_size,
    )


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
    payload["neat_variant_paths"] = [str(path) for path in config.neat_variant_paths]
    return payload


def _selected_variant(rows: list[dict[str, Any]]) -> str:
    if not rows:
        return ""

    best = max(
        rows,
        key=lambda row: (
            row.get("evolved_topology_accuracy", 0.0),
            row.get("accuracy", 0.0),
            row.get("best_fitness", 0.0),
        ),
    )

    return str(best["variant"])