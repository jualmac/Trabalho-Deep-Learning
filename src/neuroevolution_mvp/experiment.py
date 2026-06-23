"""End-to-end orchestration for the neuroevolution MVP."""

from __future__ import annotations

import math
from pathlib import Path
from typing import Any

import torch

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
)
from .neat_runner import (
    build_hyperneat_feature_substrate,
    build_hyperneat_prototype_substrate,
    build_hyperneat_substrate,
    evolve_variants,
)
from .training import train_classifier


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

    baseline = BaselineMLP(
        config.input_size,
        config.hidden_units,
        image_channels=config.image_channels,
    )
    baseline_history = train_classifier(
        baseline,
        data.train_loader,
        data.test_loader,
        epochs=config.baseline_epochs,
        learning_rate=config.learning_rate,
        optimizer_name=config.optimizer_name,
    )

    evolved_candidates = []
    neat_variants = evolve_variants(
        config.neat_variant_paths or (config.neat_config_path,),
        data.x_train,
        data.y_train,
        data.x_test,
        data.y_test,
        image_size=config.image_size,
        generations=config.neat_generations,
        eval_limit=config.neat_eval_limit,
        winner_eval_limit=config.neat_winner_eval_limit,
    )

    for index, candidate in enumerate(neat_variants):
        set_seed(config.seed + 101 + index)
        candidate_model = _candidate_sgd_model(candidate, config.input_size, config.image_size)
        candidate_history = train_classifier(
            candidate_model,
            data.train_loader,
            data.test_loader,
            epochs=config.evolved_sgd_epochs,
            learning_rate=config.learning_rate,
            optimizer_name=config.optimizer_name,
        )
        candidate["result"]["evolved_topology_accuracy"] = candidate_history[-1]["test_accuracy"]
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
            row["result"]["accuracy"],
            row["result"]["best_fitness"],
            row["result"]["evolved_topology_accuracy"],
        ),
    )
    winner = best_candidate["winner"]
    neat_config = best_candidate["config"]
    neat_history = best_candidate["history"]
    neat_accuracy = best_candidate["result"]["accuracy"]
    evolved_sgd = best_candidate["model"]
    evolved_sgd_history = best_candidate["model_history"]
    neat_variant_results = [candidate["result"] for candidate in evolved_candidates]

    neat_pure = _candidate_pure_model(best_candidate, config.input_size, config.image_size)
    interpretation_sample_count = (
        len(data.x_test)
        if config.interpretability_samples <= 0
        else min(config.interpretability_samples, len(data.x_test))
    )
    interpretation_summary = _compare_interpretations(
        baseline,
        neat_pure,
        evolved_sgd,
        data.x_test,
        data.y_test,
        image_size=config.image_size,
        image_channels=config.image_channels,
        sample_count=interpretation_sample_count,
        grid_size=config.interpretability_grid_size,
    )
    article_summary = _build_article_summary(
        config=config,
        interpretation=interpretation_summary,
        baseline_accuracy=baseline_history[-1]["test_accuracy"],
        neat_accuracy=neat_accuracy,
        neat_sgd_accuracy=evolved_sgd_history[-1]["test_accuracy"],
        selected_variant=_selected_variant(neat_variant_results),
    )

    results = {
        "config": _serializable_config(config),
        "summary": {
            "baseline_sgd_accuracy": baseline_history[-1]["test_accuracy"],
            "neat_accuracy": neat_accuracy,
            "evolved_topology_sgd_accuracy": evolved_sgd_history[-1]["test_accuracy"],
            "optimizer": config.optimizer_name,
            "dataset": config.dataset,
            "image_size": config.image_size,
            "image_channels": config.image_channels,
            "sgd_weight_initialization": "random_reset_preserving_neat_topology",
            "selected_neat_variant": _selected_variant(neat_variant_results),
            "selected_neat_family": str(best_candidate["result"].get("family", "direct")),
            "evolved_nodes": len(winner.nodes),
            "evolved_enabled_connections": sum(
                1 for connection in winner.connections.values() if connection.enabled
            ),
        },
        "interpretation_summary": interpretation_summary,
        "article_summary": article_summary,
        "metric_definitions": _metric_definitions(),
        "baseline_history": baseline_history,
        "neat_history": neat_history,
        "neat_variant_results": neat_variant_results,
        "evolved_sgd_history": evolved_sgd_history,
    }

    save_json(config.artifact_dir / "results.json", results)
    save_genome(config.artifact_dir / "winner_genome.pkl", winner)
    save_model(config.artifact_dir / "baseline_mlp.pt", baseline)
    save_model(config.artifact_dir / "evolved_topology_sgd.pt", evolved_sgd)
    if best_candidate["result"].get("family") in {"prototypes", "seeded_prototypes", "hyperneat_prototypes"}:
        torch.save(best_candidate["centroids"], config.artifact_dir / "prototype_centroids.pt")
    return results


def _build_article_summary(
    config: ExperimentConfig,
    interpretation: dict[str, Any],
    baseline_accuracy: float,
    neat_accuracy: float,
    neat_sgd_accuracy: float,
    selected_variant: str,
) -> dict[str, Any]:
    pair = interpretation["pairwise_conditioned"]["neat_pure_vs_neat_sgd"]
    primary = interpretation["primary_metric"]
    correct = pair["both_correct"]
    wrong = pair["both_wrong"]
    controlled_wrong = pair["both_wrong_same_prediction"]
    one_correct = pair["one_correct"]
    gap_ci = pair["attention_agreement_gap_weighted_jaccard_ci95"]
    controlled_ci = pair["controlled_attention_gap_weighted_jaccard_ci95"]
    same_prediction_ci = pair["dataset_attention_divergence_same_prediction_ci95"]
    return {
        "dataset": config.dataset,
        "seed": config.seed,
        "selected_neat_variant": selected_variant,
        "evaluated_images": int(interpretation["probe_samples"]),
        "evaluation_scope": interpretation["evaluation_scope"],
        "baseline_accuracy": baseline_accuracy,
        "neat_pure_accuracy": neat_accuracy,
        "neat_sgd_accuracy": neat_sgd_accuracy,
        "dataset_attention_divergence": primary["value"],
        "dad_ci95_low": primary["ci95"]["low"],
        "dad_ci95_high": primary["ci95"]["high"],
        "dad_macro": pair["dataset_attention_divergence_macro"],
        "dad_same_prediction": pair["dataset_attention_divergence_same_prediction"],
        "dad_same_prediction_ci95_low": same_prediction_ci["low"],
        "dad_same_prediction_ci95_high": same_prediction_ci["high"],
        "same_prediction_n": pair["same_prediction_sample_count"],
        "valid_map_coverage": pair["valid_map_coverage"],
        "both_correct_n": correct["sample_count"],
        "both_correct_weighted_jaccard": correct["weighted_jaccard"]["mean"],
        "both_wrong_n": wrong["sample_count"],
        "both_wrong_weighted_jaccard": wrong["weighted_jaccard"]["mean"],
        "both_wrong_same_prediction_n": controlled_wrong["sample_count"],
        "both_wrong_same_prediction_weighted_jaccard": controlled_wrong[
            "weighted_jaccard"
        ]["mean"],
        "one_correct_n": one_correct["sample_count"],
        "attention_agreement_gap": pair[
            "attention_agreement_gap_weighted_jaccard"
        ],
        "attention_gap_ci95_low": gap_ci["low"],
        "attention_gap_ci95_high": gap_ci["high"],
        "controlled_attention_gap": pair[
            "controlled_attention_gap_weighted_jaccard"
        ],
        "controlled_gap_ci95_low": controlled_ci["low"],
        "controlled_gap_ci95_high": controlled_ci["high"],
        "wrong_prediction_disagreement": pair["wrong_prediction_disagreement"],
        "attribution_method": interpretation["attribution_method"],
        "dataset_metric": interpretation["dataset_metric"],
    }


def _metric_definitions() -> dict[str, dict[str, str]]:
    return {
        "dataset_attention_divergence": {
            "definition": "1 menos a media do Jaccard ponderado entre mapas pareados.",
            "range": "0 a 1",
            "interpretation": "Maior valor indica maior divergencia espacial media.",
        },
        "dad_macro": {
            "definition": "DAD calculada com o mesmo peso para cada classe.",
            "range": "0 a 1",
            "interpretation": "Controla desequilibrio na quantidade de imagens por classe.",
        },
        "dad_same_prediction": {
            "definition": "DAD apenas quando os modelos preveem a mesma classe.",
            "range": "0 a 1",
            "interpretation": "Compara mapas que explicam a mesma classe-alvo.",
        },
        "valid_map_coverage": {
            "definition": "Fracao de imagens com sinal de oclusao valido nos dois mapas.",
            "range": "0 a 1",
            "interpretation": "Valores proximos de 1 indicam boa cobertura da sumarizacao.",
        },
        "both_correct_weighted_jaccard": {
            "definition": "Concordancia min-max media quando ambos acertam.",
            "range": "0 a 1",
            "interpretation": "Maior valor indica regioes mais semelhantes nos acertos.",
        },
        "both_wrong_weighted_jaccard": {
            "definition": "Concordancia min-max media quando ambos erram.",
            "range": "0 a 1",
            "interpretation": "Maior valor indica regioes mais semelhantes nos erros.",
        },
        "both_wrong_same_prediction_weighted_jaccard": {
            "definition": "Concordancia nos erros em que ambos preveem a mesma classe.",
            "range": "0 a 1",
            "interpretation": "Controla a classe-alvo ao comparar os erros.",
        },
        "attention_agreement_gap": {
            "definition": "Jaccard medio nos acertos menos Jaccard medio nos erros.",
            "range": "-1 a 1",
            "interpretation": "Positivo indica maior concordancia espacial nos acertos.",
        },
        "controlled_attention_gap": {
            "definition": "Gap usando somente erros com a mesma classe incorreta.",
            "range": "-1 a 1",
            "interpretation": "Controla a classe-alvo explicada pelos heatmaps.",
        },
        "wrong_prediction_disagreement": {
            "definition": "Fracao dos erros conjuntos com classes previstas diferentes.",
            "range": "0 a 1",
            "interpretation": "Maior valor indica erros de decisao mais distintos.",
        },
    }


def enrich_results_for_reporting(results: dict[str, Any]) -> dict[str, Any]:
    """Add flat article metrics and definitions to a current-protocol result."""

    summary = results["summary"]
    interpretation = results["interpretation_summary"]
    if summary.get("sgd_weight_initialization") != "random_reset_preserving_neat_topology":
        raise ValueError("Artifacts predate the independent post-NEAT weight reset.")
    if interpretation.get("attribution_method") != "occlusion_absolute_predicted_logit_change_v2":
        raise ValueError("Artifacts predate the logit-based occlusion maps.")
    stored_config = results.get("config", {})
    config = ExperimentConfig(
        seed=int(stored_config.get("seed", 42)),
        dataset=str(stored_config.get("dataset", summary.get("dataset", "mnist"))),
        image_size=int(stored_config.get("image_size", summary.get("image_size", 26))),
    )
    results["article_summary"] = _build_article_summary(
        config=config,
        interpretation=interpretation,
        baseline_accuracy=float(summary["baseline_sgd_accuracy"]),
        neat_accuracy=float(summary["neat_accuracy"]),
        neat_sgd_accuracy=float(summary["evolved_topology_sgd_accuracy"]),
        selected_variant=str(summary["selected_neat_variant"]),
    )
    results["metric_definitions"] = _metric_definitions()
    return results


def _compare_interpretations(
    baseline: BaselineMLP,
    neat_pure: FixedNeatTopologyMLP,
    evolved_sgd: EvolvedTopologyMLP,
    x_test,
    y_test,
    image_size: int,
    image_channels: int,
    sample_count: int,
    grid_size: int,
) -> dict[str, Any]:
    """Compare attention maps, stratified by joint prediction outcome."""

    pair_rows: dict[str, list[dict[str, Any]]] = {
        "baseline_vs_neat_pure": [],
        "baseline_vs_neat_sgd": [],
        "neat_pure_vs_neat_sgd": [],
    }
    probe_indices = _balanced_probe_indices(y_test, sample_count)
    patch_size = max(1, math.ceil(image_size / grid_size))

    progress_step = max(10, len(probe_indices) // 20)
    for position, index in enumerate(probe_indices, start=1):
        if position == 1 or position % progress_step == 0 or position == len(probe_indices):
            print(
                f"Interpretabilidade: {position}/{len(probe_indices)} imagens",
                flush=True,
            )
        image = x_test[index]
        label = int(y_test[index].item())
        reshaped = _reshape_image(image, image_size, image_channels)
        maps = {
            "baseline": occlusion_map(
                baseline, reshaped, image_size, patch_size=patch_size
            ),
            "neat_pure": occlusion_map(
                neat_pure, reshaped, image_size, patch_size=patch_size
            ),
            "neat_sgd": occlusion_map(
                evolved_sgd, reshaped, image_size, patch_size=patch_size
            ),
        }
        predictions = {
            "baseline": int(prediction_summary(baseline, reshaped)["class"]),
            "neat_pure": int(prediction_summary(neat_pure, reshaped)["class"]),
            "neat_sgd": int(prediction_summary(evolved_sgd, reshaped)["class"]),
        }
        _append_pair_row(
            pair_rows["baseline_vs_neat_pure"],
            maps["baseline"], maps["neat_pure"],
            predictions["baseline"], predictions["neat_pure"], label,
        )
        _append_pair_row(
            pair_rows["baseline_vs_neat_sgd"],
            maps["baseline"], maps["neat_sgd"],
            predictions["baseline"], predictions["neat_sgd"], label,
        )
        _append_pair_row(
            pair_rows["neat_pure_vs_neat_sgd"],
            maps["neat_pure"], maps["neat_sgd"],
            predictions["neat_pure"], predictions["neat_sgd"], label,
        )

    conditioned = {name: _summarize_pair(rows) for name, rows in pair_rows.items()}
    neat_sgd_rows = pair_rows["neat_pure_vs_neat_sgd"]
    baseline_sgd_rows = pair_rows["baseline_vs_neat_sgd"]
    baseline_neat_rows = pair_rows["baseline_vs_neat_pure"]
    return {
        "probe_samples": float(len(probe_indices)),
        "evaluation_scope": (
            "full_loaded_test_set"
            if len(probe_indices) == len(y_test)
            else "class_balanced_subset"
        ),
        "attribution_method": "occlusion_absolute_predicted_logit_change_v2",
        "dataset_metric": "dad_weighted_jaccard_v1",
        "occlusion_grid_size": grid_size,
        "occlusion_patch_size": patch_size,
        "top_fraction": 0.15,
        "primary_metric": {
            "name": "dataset_attention_divergence",
            "pair": "neat_pure_vs_neat_sgd",
            "value": conditioned["neat_pure_vs_neat_sgd"][
                "dataset_attention_divergence"
            ],
            "ci95": conditioned["neat_pure_vs_neat_sgd"][
                "dataset_attention_divergence_ci95"
            ],
        },
        "baseline_vs_neat_sgd_cosine": _valid_mean(baseline_sgd_rows, "cosine_similarity"),
        "baseline_vs_neat_sgd_top_overlap": _valid_mean(baseline_sgd_rows, "top_pixel_overlap"),
        "baseline_vs_neat_pure_cosine": _valid_mean(baseline_neat_rows, "cosine_similarity"),
        "neat_pure_vs_neat_sgd_cosine": _valid_mean(neat_sgd_rows, "cosine_similarity"),
        "neat_pure_vs_neat_sgd_top_overlap": _valid_mean(neat_sgd_rows, "top_pixel_overlap"),
        "neat_pure_vs_neat_sgd_difference": _valid_mean(neat_sgd_rows, "mean_absolute_difference"),
        "neat_pure_vs_neat_sgd_prediction_agreement": _mean_value(
            neat_sgd_rows, "same_prediction"
        ),
        "pairwise_conditioned": conditioned,
    }


def _append_pair_row(
    rows: list[dict[str, Any]],
    first_map: torch.Tensor,
    second_map: torch.Tensor,
    first_prediction: int,
    second_prediction: int,
    label: int,
) -> None:
    first_correct = first_prediction == label
    second_correct = second_prediction == label
    rows.append(
        {
            **compare_maps(first_map, second_map),
            "first_correct": float(first_correct),
            "second_correct": float(second_correct),
            "both_correct": float(first_correct and second_correct),
            "both_wrong": float(not first_correct and not second_correct),
            "both_wrong_same_prediction": float(
                not first_correct
                and not second_correct
                and first_prediction == second_prediction
            ),
            "both_wrong_different_prediction": float(
                not first_correct
                and not second_correct
                and first_prediction != second_prediction
            ),
            "one_correct": float(first_correct != second_correct),
            "same_prediction": float(first_prediction == second_prediction),
            "label": label,
        }
    )


def _summarize_pair(rows: list[dict[str, Any]]) -> dict[str, Any]:
    groups = {
        "all": rows,
        "both_correct": [row for row in rows if row["both_correct"]],
        "both_wrong": [row for row in rows if row["both_wrong"]],
        "both_wrong_same_prediction": [
            row for row in rows if row["both_wrong_same_prediction"]
        ],
        "both_wrong_different_prediction": [
            row for row in rows if row["both_wrong_different_prediction"]
        ],
        "one_correct": [row for row in rows if row["one_correct"]],
        "same_prediction": [row for row in rows if row["same_prediction"]],
        "different_prediction": [row for row in rows if not row["same_prediction"]],
    }
    summaries = {name: _summarize_group(group) for name, group in groups.items()}
    correct_jaccard = summaries["both_correct"]["weighted_jaccard"]["mean"]
    wrong_jaccard = summaries["both_wrong"]["weighted_jaccard"]["mean"]
    correct_iou = summaries["both_correct"]["top_pixel_iou"]["mean"]
    wrong_iou = summaries["both_wrong"]["top_pixel_iou"]["mean"]
    controlled_wrong_jaccard = summaries["both_wrong_same_prediction"][
        "weighted_jaccard"
    ]["mean"]
    controlled_wrong_iou = summaries["both_wrong_same_prediction"][
        "top_pixel_iou"
    ]["mean"]
    gap_ci = _bootstrap_difference_ci(
        groups["both_correct"], groups["both_wrong"], "weighted_jaccard"
    )
    controlled_gap_ci = _bootstrap_difference_ci(
        groups["both_correct"],
        groups["both_wrong_same_prediction"],
        "weighted_jaccard",
    )
    jointly_wrong = groups["both_wrong"]
    all_jaccard = summaries["all"]["weighted_jaccard"]["mean"]
    all_jaccard_ci = _bootstrap_mean_ci(groups["all"], "weighted_jaccard")
    macro_jaccard = _macro_class_mean(groups["all"], "weighted_jaccard")
    same_prediction_jaccard = summaries["same_prediction"]["weighted_jaccard"]["mean"]
    same_prediction_ci = _bootstrap_mean_ci(
        groups["same_prediction"], "weighted_jaccard"
    )
    return {
        **summaries,
        "dataset_attention_agreement": all_jaccard,
        "dataset_attention_agreement_ci95": all_jaccard_ci,
        "dataset_attention_divergence": (
            1.0 - all_jaccard if all_jaccard is not None else None
        ),
        "dataset_attention_divergence_ci95": _invert_interval(all_jaccard_ci),
        "dataset_attention_agreement_macro": macro_jaccard,
        "dataset_attention_divergence_macro": (
            1.0 - macro_jaccard if macro_jaccard is not None else None
        ),
        "dataset_attention_divergence_same_prediction": (
            1.0 - same_prediction_jaccard
            if same_prediction_jaccard is not None
            else None
        ),
        "dataset_attention_divergence_same_prediction_ci95": _invert_interval(
            same_prediction_ci
        ),
        "same_prediction_sample_count": len(groups["same_prediction"]),
        "valid_map_coverage": (
            summaries["all"]["valid_map_pairs"] / max(len(rows), 1)
        ),
        "prediction_agreement": _mean_value(rows, "same_prediction"),
        "wrong_prediction_disagreement": (
            1.0 - _mean_value(jointly_wrong, "same_prediction")
            if jointly_wrong else None
        ),
        "attention_agreement_gap_weighted_jaccard": _difference(
            correct_jaccard, wrong_jaccard
        ),
        "attention_agreement_gap_top_iou": _difference(correct_iou, wrong_iou),
        "attention_agreement_gap_weighted_jaccard_ci95": gap_ci,
        "controlled_attention_gap_weighted_jaccard": _difference(
            correct_jaccard, controlled_wrong_jaccard
        ),
        "controlled_attention_gap_top_iou": _difference(
            correct_iou, controlled_wrong_iou
        ),
        "controlled_attention_gap_weighted_jaccard_ci95": controlled_gap_ci,
    }


def _summarize_group(rows: list[dict[str, Any]]) -> dict[str, Any]:
    valid_rows = [row for row in rows if row["valid_attribution"]]
    return {
        "sample_count": len(rows),
        "valid_map_pairs": len(valid_rows),
        "cosine_similarity": _metric_with_ci(valid_rows, "cosine_similarity"),
        "top_pixel_overlap": _metric_with_ci(valid_rows, "top_pixel_overlap"),
        "top_pixel_iou": _metric_with_ci(valid_rows, "top_pixel_iou"),
        "weighted_jaccard": _metric_with_ci(valid_rows, "weighted_jaccard"),
        "mean_absolute_difference": _metric_with_ci(
            valid_rows, "mean_absolute_difference"
        ),
    }


def _metric_with_ci(rows: list[dict[str, Any]], key: str) -> dict[str, Any]:
    values = [float(row[key]) for row in rows]
    if not values:
        return {"mean": None, "ci95_low": None, "ci95_high": None, "n": 0}
    mean = sum(values) / len(values)
    if len(values) == 1:
        return {"mean": mean, "ci95_low": None, "ci95_high": None, "n": 1}
    variance = sum((value - mean) ** 2 for value in values) / (len(values) - 1)
    margin = 1.96 * math.sqrt(variance / len(values))
    return {
        "mean": mean,
        "ci95_low": max(0.0, mean - margin),
        "ci95_high": min(1.0, mean + margin),
        "n": len(values),
    }


def _balanced_probe_indices(labels: torch.Tensor, limit: int) -> list[int]:
    by_class: dict[int, list[int]] = {}
    for index, label in enumerate(labels.tolist()):
        by_class.setdefault(int(label), []).append(index)
    selected: list[int] = []
    offset = 0
    classes = sorted(by_class)
    while len(selected) < min(limit, len(labels)):
        added = False
        for label in classes:
            class_indices = by_class[label]
            if offset < len(class_indices):
                selected.append(class_indices[offset])
                added = True
                if len(selected) >= min(limit, len(labels)):
                    break
        if not added:
            break
        offset += 1
    return selected


def _valid_mean(rows: list[dict[str, Any]], key: str) -> float:
    valid_rows = [row for row in rows if row["valid_attribution"]]
    return _mean_value(valid_rows, key)


def _mean_value(rows: list[dict[str, Any]], key: str) -> float:
    return sum(float(row[key]) for row in rows) / max(len(rows), 1)


def _difference(first: float | None, second: float | None) -> float | None:
    if first is None or second is None:
        return None
    return first - second


def _bootstrap_difference_ci(
    first_rows: list[dict[str, Any]],
    second_rows: list[dict[str, Any]],
    key: str,
    iterations: int = 2_000,
) -> dict[str, Any]:
    first = torch.tensor(
        [float(row[key]) for row in first_rows if row["valid_attribution"]],
        dtype=torch.float32,
    )
    second = torch.tensor(
        [float(row[key]) for row in second_rows if row["valid_attribution"]],
        dtype=torch.float32,
    )
    if len(first) < 2 or len(second) < 2:
        return {"low": None, "high": None, "iterations": 0}
    generator = torch.Generator().manual_seed(42)
    first_indices = torch.randint(
        len(first), (iterations, len(first)), generator=generator
    )
    second_indices = torch.randint(
        len(second), (iterations, len(second)), generator=generator
    )
    differences = first[first_indices].mean(dim=1) - second[second_indices].mean(dim=1)
    return {
        "low": float(torch.quantile(differences, 0.025).item()),
        "high": float(torch.quantile(differences, 0.975).item()),
        "iterations": iterations,
    }


def _bootstrap_mean_ci(
    rows: list[dict[str, Any]],
    key: str,
    iterations: int = 2_000,
) -> dict[str, Any]:
    values = torch.tensor(
        [float(row[key]) for row in rows if row["valid_attribution"]],
        dtype=torch.float32,
    )
    if len(values) < 2:
        return {"low": None, "high": None, "iterations": 0}
    generator = torch.Generator().manual_seed(42)
    indices = torch.randint(
        len(values), (iterations, len(values)), generator=generator
    )
    means = values[indices].mean(dim=1)
    return {
        "low": float(torch.quantile(means, 0.025).item()),
        "high": float(torch.quantile(means, 0.975).item()),
        "iterations": iterations,
    }


def _macro_class_mean(rows: list[dict[str, Any]], key: str) -> float | None:
    by_class: dict[int, list[float]] = {}
    for row in rows:
        if row["valid_attribution"]:
            by_class.setdefault(int(row["label"]), []).append(float(row[key]))
    class_means = [sum(values) / len(values) for values in by_class.values() if values]
    if not class_means:
        return None
    return sum(class_means) / len(class_means)


def _invert_interval(interval: dict[str, Any]) -> dict[str, Any]:
    low = interval.get("low")
    high = interval.get("high")
    return {
        "low": 1.0 - high if high is not None else None,
        "high": 1.0 - low if low is not None else None,
        "iterations": interval.get("iterations", 0),
    }


def _reshape_image(image: torch.Tensor, image_size: int, image_channels: int) -> torch.Tensor:
    if image_channels == 1:
        return image.reshape(image_size, image_size)
    return image.reshape(image_channels, image_size, image_size)


def _candidate_sgd_model(candidate: dict[str, Any], input_size: int, image_size: int):
    if candidate["result"].get("family") in {"prototypes", "seeded_prototypes"}:
        classifier = EvolvedTopologyMLP(
            candidate["winner"],
            candidate["config"],
            input_size=PROTOTYPE_FEATURE_SIZE,
            initialize_from_genome=False,
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
            reset_weights=True,
            preserve_connection_mask=True,
        )
        return PrototypeWrappedModel(classifier, candidate["centroids"])

    if candidate["result"].get("family") == "features":
        classifier = EvolvedTopologyMLP(
            candidate["winner"],
            candidate["config"],
            input_size=FEATURE_SIZE,
            initialize_from_genome=False,
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
            reset_weights=True,
            preserve_connection_mask=True,
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
            reset_weights=True,
            preserve_connection_mask=True,
        )

    return EvolvedTopologyMLP(
        candidate["winner"],
        candidate["config"],
        input_size=input_size,
        initialize_from_genome=False,
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
            row.get("accuracy", 0.0),
            row.get("best_fitness", 0.0),
            row.get("evolved_topology_accuracy", 0.0),
        ),
    )
    return str(best["variant"])
