"""NEAT evolution utilities for discovering compact image-classifier topologies."""

from __future__ import annotations

from pathlib import Path
from typing import Any

import neat
import numpy as np
import torch

from .features import (
    FEATURE_SIZE,
    PROTOTYPE_FEATURE_SIZE,
    class_centroids,
    extract_neat_features_numpy,
    extract_prototype_features_numpy,
)


def load_neat_config(path: Path | str) -> neat.Config:
    """Load the NEAT config used by the MVP."""

    return neat.Config(
        neat.DefaultGenome,
        neat.DefaultReproduction,
        neat.DefaultSpeciesSet,
        neat.DefaultStagnation,
        str(path),
    )


def evolve_genome(
    config: neat.Config,
    x_train: torch.Tensor,
    y_train: torch.Tensor,
    generations: int,
    eval_limit: int,
    seed_initializer=None,
) -> tuple[neat.DefaultGenome, list[dict[str, float]]]:
    """Evolve a topology and return the winning genome plus generation stats."""

    population = neat.Population(config)
    if seed_initializer is not None:
        seed_initializer(population.population, config)

    reporter = _HistoryReporter()
    population.add_reporter(reporter)
    population.add_reporter(neat.StdOutReporter(show_species_detail=False))

    eval_indices = _balanced_indices(y_train, eval_limit)
    x_eval = x_train[eval_indices].numpy()
    y_eval = y_train[eval_indices].numpy()

    def evaluate_population(genomes, neat_config):
        for _, genome in genomes:
            genome.fitness = _fitness_genome(
                genome,
                neat_config,
                x_eval,
                y_eval,
            )

    winner = population.run(evaluate_population, generations)
    return winner, reporter.history

def evolve_hyperneat_genome(
    config: neat.Config,
    x_train: torch.Tensor,
    y_train: torch.Tensor,
    generations: int,
    eval_limit: int,
    image_size: int,
) -> tuple[neat.DefaultGenome, list[dict[str, float]]]:
    """Evolve a CPPN genome used to generate HyperNEAT substrate weights."""

    population = neat.Population(config)

    reporter = _HistoryReporter()
    population.add_reporter(reporter)
    population.add_reporter(neat.StdOutReporter(show_species_detail=False))

    eval_indices = _balanced_indices(y_train, eval_limit)
    x_eval = x_train[eval_indices].numpy()
    y_eval = y_train[eval_indices].numpy()

    def evaluate_population(genomes, neat_config):
        for _, genome in genomes:
            genome.fitness = _fitness_hyperneat_genome(
                genome=genome,
                neat_config=neat_config,
                x_eval=x_eval,
                y_eval=y_eval,
                image_size=image_size,
            )

    winner = population.run(evaluate_population, generations)
    return winner, reporter.history

def _fitness_hyperneat_genome(
    genome: neat.DefaultGenome,
    neat_config: neat.Config,
    x_eval: np.ndarray,
    y_eval: np.ndarray,
    image_size: int,
) -> float:
    weight_matrix, bias = build_hyperneat_substrate(
        genome=genome,
        config=neat_config,
        image_size=image_size,
    )

    logits = x_eval @ weight_matrix.T + bias
    predictions = np.argmax(logits, axis=1)

    return float(np.mean(predictions == y_eval))

def evolve_best_variant(
    config_paths: tuple[Path, ...],
    x_train: torch.Tensor,
    y_train: torch.Tensor,
    x_test: torch.Tensor,
    y_test: torch.Tensor,
    image_size: int,
    generations: int,
    eval_limit: int,
    winner_eval_limit: int,
) -> tuple[neat.DefaultGenome, neat.Config, list[dict[str, float]], list[dict[str, Any]]]:
    """Run several NEAT connectivity variants and keep the strongest winner."""

    variants = evolve_variants(
        config_paths,
        x_train,
        y_train,
        x_test,
        y_test,
        image_size=image_size,
        generations=generations,
        eval_limit=eval_limit,
        winner_eval_limit=winner_eval_limit,
    )
    if not variants:
        raise ValueError("No NEAT variant configs were provided.")

    best = max(
        variants,
        key=lambda row: row["result"]["accuracy"] + 0.05 * row["result"]["best_fitness"],
    )
    return (
        best["winner"],
        best["config"],
        best["history"],
        [row["result"] for row in variants],
    )

def _variant_family(config_path: Path) -> str:
    variant_name = config_path.stem.lower()

    if "hyperneat" in variant_name:
        return "hyperneat"

    return "direct"


def evolve_variants(
    config_paths: tuple[Path, ...],
    x_train: torch.Tensor,
    y_train: torch.Tensor,
    x_test: torch.Tensor,
    y_test: torch.Tensor,
    image_size: int,
    generations: int,
    eval_limit: int,
    winner_eval_limit: int,
) -> list[dict[str, Any]]:
    """Run all configured NEAT variants and return their winners and metrics."""

    variant_results: list[dict[str, Any]] = []
    total_variants = len(config_paths)

    for index, config_path in enumerate(config_paths, start=1):
        family = _variant_family(config_path)
        print(
            f"\n=== Variante NEAT {index}/{total_variants}: {config_path.stem} ({family}) ===",
            flush=True,
        )
        neat_config = load_neat_config(config_path)
        if family in {"prototypes", "seeded_prototypes"}:
            centroids = class_centroids(x_train, y_train)
            x_train_variant = torch.from_numpy(extract_prototype_features_numpy(x_train.numpy(), centroids))
            x_test_variant = torch.from_numpy(extract_prototype_features_numpy(x_test.numpy(), centroids))
            seed_initializer = None
            if family == "seeded_prototypes":
                seed_initializer = _seed_prototype_population
            winner, history = evolve_genome(
                neat_config,
                x_train_variant,
                y_train,
                generations=generations,
                eval_limit=eval_limit,
                seed_initializer=seed_initializer,
            )
            accuracy = evaluate_winner(
                winner,
                neat_config,
                x_test_variant,
                y_test,
                limit=winner_eval_limit,
            )
            enabled_connections = sum(
                1 for connection in winner.connections.values() if connection.enabled
            )
            extra = {"centroids": centroids}
        elif family == "features":
            x_train_variant = torch.from_numpy(extract_neat_features_numpy(x_train.numpy(), image_size))
            x_test_variant = torch.from_numpy(extract_neat_features_numpy(x_test.numpy(), image_size))
            winner, history = evolve_genome(
                neat_config,
                x_train_variant,
                y_train,
                generations=generations,
                eval_limit=eval_limit,
            )
            accuracy = evaluate_winner(
                winner,
                neat_config,
                x_test_variant,
                y_test,
                limit=winner_eval_limit,
            )
            enabled_connections = sum(
                1 for connection in winner.connections.values() if connection.enabled
            )
            extra = {}
        elif family == "hyperneat_prototypes":
            centroids = class_centroids(x_train, y_train)
            x_train_variant = torch.from_numpy(extract_prototype_features_numpy(x_train.numpy(), centroids))
            x_test_variant = torch.from_numpy(extract_prototype_features_numpy(x_test.numpy(), centroids))
            winner, history = evolve_hyperneat_prototype_genome(
                neat_config,
                x_train_variant,
                y_train,
                generations=generations,
                eval_limit=eval_limit,
            )
            accuracy = evaluate_hyperneat_prototype_winner(
                winner,
                neat_config,
                x_test_variant,
                y_test,
                limit=winner_eval_limit,
            )
            enabled_connections = sum(
                1 for connection in winner.connections.values() if connection.enabled
            )
            extra = {"centroids": centroids}
        elif family == "hyperneat_features":
            x_train_variant = torch.from_numpy(extract_neat_features_numpy(x_train.numpy(), image_size))
            x_test_variant = torch.from_numpy(extract_neat_features_numpy(x_test.numpy(), image_size))
            winner, history = evolve_hyperneat_feature_genome(
                neat_config,
                x_train_variant,
                y_train,
                generations=generations,
                eval_limit=eval_limit,
            )
            accuracy = evaluate_hyperneat_feature_winner(
                winner,
                neat_config,
                x_test_variant,
                y_test,
                limit=winner_eval_limit,
            )
            enabled_connections = sum(
                1 for connection in winner.connections.values() if connection.enabled
            )
            extra = {}
        elif family == "hyperneat":
            winner, history = evolve_hyperneat_genome(
                neat_config,
                x_train,
                y_train,
                generations=generations,
                eval_limit=eval_limit,
                image_size=image_size,
            )
            accuracy = evaluate_hyperneat_winner(
                winner,
                neat_config,
                x_test,
                y_test,
                limit=winner_eval_limit,
                image_size=image_size,
            )
            enabled_connections = sum(
                1 for connection in winner.connections.values() if connection.enabled
            )
            extra = {}
        else:
            winner, history = evolve_genome(
                neat_config,
                x_train,
                y_train,
                generations=generations,
                eval_limit=eval_limit,
            )
            accuracy = evaluate_winner(
                winner,
                neat_config,
                x_test,
                y_test,
                limit=winner_eval_limit,
            )
            enabled_connections = sum(
                1 for connection in winner.connections.values() if connection.enabled
            )
            extra = {}
        best_fitness = max((row["best_fitness"] for row in history), default=0.0)
        result = {
            "variant": config_path.stem,
            "family": family,
            "config_path": str(config_path),
            "accuracy": accuracy,
            "best_fitness": best_fitness,
            "nodes": len(winner.nodes),
            "enabled_connections": enabled_connections,
        }
        variant_results.append(
            {
                "winner": winner,
                "config": neat_config,
                "history": history,
                "result": result,
                **extra,
            }
        )

    return variant_results


def evolve_hyperneat_genome(
    config: neat.Config,
    x_train: torch.Tensor,
    y_train: torch.Tensor,
    generations: int,
    eval_limit: int,
    image_size: int,
) -> tuple[neat.DefaultGenome, list[dict[str, float]]]:
    """Evolve a CPPN that generates a pixel-to-class substrate."""

    population = neat.Population(config)
    reporter = _HistoryReporter()
    population.add_reporter(reporter)
    population.add_reporter(neat.StdOutReporter(show_species_detail=False))

    eval_indices = _balanced_indices(y_train, eval_limit)
    x_eval = x_train[eval_indices].numpy()
    y_eval = y_train[eval_indices].numpy()

    def evaluate_population(genomes: list[tuple[int, neat.DefaultGenome]], neat_config: neat.Config) -> None:
        for _, genome in genomes:
            weight_matrix, bias = build_hyperneat_substrate(genome, neat_config, image_size)
            logits = x_eval @ weight_matrix.T + bias
            genome.fitness = _fitness_logits(logits, y_eval)

    winner = population.run(evaluate_population, generations)
    return winner, reporter.history


def evolve_hyperneat_feature_genome(
    config: neat.Config,
    x_train: torch.Tensor,
    y_train: torch.Tensor,
    generations: int,
    eval_limit: int,
) -> tuple[neat.DefaultGenome, list[dict[str, float]]]:
    """Evolve a CPPN that generates weights over compact image features."""

    population = neat.Population(config)
    reporter = _HistoryReporter()
    population.add_reporter(reporter)
    population.add_reporter(neat.StdOutReporter(show_species_detail=False))

    eval_indices = _balanced_indices(y_train, eval_limit)
    x_eval = x_train[eval_indices].numpy()
    y_eval = y_train[eval_indices].numpy()

    def evaluate_population(genomes: list[tuple[int, neat.DefaultGenome]], neat_config: neat.Config) -> None:
        for _, genome in genomes:
            weight_matrix, bias = build_hyperneat_feature_substrate(genome, neat_config)
            logits = x_eval @ weight_matrix.T + bias
            genome.fitness = _fitness_logits(logits, y_eval)

    winner = population.run(evaluate_population, generations)
    return winner, reporter.history


def evaluate_hyperneat_feature_winner(
    genome: neat.DefaultGenome,
    config: neat.Config,
    x_test: torch.Tensor,
    y_test: torch.Tensor,
    limit: int,
) -> float:
    """Evaluate a feature-substrate HyperNEAT winner without SGD retraining."""

    eval_indices = _balanced_indices(y_test, limit)
    weight_matrix, bias = build_hyperneat_feature_substrate(genome, config)
    logits = x_test[eval_indices].numpy() @ weight_matrix.T + bias
    predictions = np.argmax(logits, axis=1)
    labels = y_test[eval_indices].numpy()
    return float(np.mean(predictions == labels)) if len(labels) else 0.0


def evolve_hyperneat_prototype_genome(
    config: neat.Config,
    x_train: torch.Tensor,
    y_train: torch.Tensor,
    generations: int,
    eval_limit: int,
) -> tuple[neat.DefaultGenome, list[dict[str, float]]]:
    """Evolve a CPPN that generates weights over prototype similarity features."""

    population = neat.Population(config)
    reporter = _HistoryReporter()
    population.add_reporter(reporter)
    population.add_reporter(neat.StdOutReporter(show_species_detail=False))

    eval_indices = _balanced_indices(y_train, eval_limit)
    x_eval = x_train[eval_indices].numpy()
    y_eval = y_train[eval_indices].numpy()

    def evaluate_population(genomes: list[tuple[int, neat.DefaultGenome]], neat_config: neat.Config) -> None:
        for _, genome in genomes:
            weight_matrix, bias = build_hyperneat_prototype_substrate(genome, neat_config)
            logits = x_eval @ weight_matrix.T + bias
            genome.fitness = _fitness_logits(logits, y_eval)

    winner = population.run(evaluate_population, generations)
    return winner, reporter.history


def evaluate_hyperneat_prototype_winner(
    genome: neat.DefaultGenome,
    config: neat.Config,
    x_test: torch.Tensor,
    y_test: torch.Tensor,
    limit: int,
) -> float:
    """Evaluate a prototype-substrate HyperNEAT winner without SGD retraining."""

    eval_indices = _balanced_indices(y_test, limit)
    weight_matrix, bias = build_hyperneat_prototype_substrate(genome, config)
    logits = x_test[eval_indices].numpy() @ weight_matrix.T + bias
    predictions = np.argmax(logits, axis=1)
    labels = y_test[eval_indices].numpy()
    return float(np.mean(predictions == labels)) if len(labels) else 0.0


def evaluate_hyperneat_winner(
    winner: neat.DefaultGenome,
    config: neat.Config,
    x_test: torch.Tensor,
    y_test: torch.Tensor,
    limit: int,
    image_size: int,
) -> float:
    if limit > 0:
        x_eval = x_test[:limit].numpy()
        y_eval = y_test[:limit].numpy()
    else:
        x_eval = x_test.numpy()
        y_eval = y_test.numpy()

    return _fitness_hyperneat_genome(
        genome=winner,
        neat_config=config,
        x_eval=x_eval,
        y_eval=y_eval,
        image_size=image_size,
    )


def build_hyperneat_substrate(
    genome: neat.DefaultGenome,
    config: neat.Config,
    image_size: int,
    output_size: int = 10,
    weight_scale: float = 5.0,
    bias_scale: float = 2.0,
    threshold: float = 0.03,
) -> tuple[np.ndarray, np.ndarray]:
    """Generate dense classifier weights from a CPPN genome."""

    cppn = neat.nn.FeedForwardNetwork.create(genome, config)
    pixel_coords = _pixel_coordinates(image_size)
    class_coords = np.linspace(-1.0, 1.0, output_size)
    weight_matrix = np.zeros((output_size, image_size * image_size), dtype=np.float64)
    bias = np.zeros(output_size, dtype=np.float64)

    for class_index, class_x in enumerate(class_coords):
        for pixel_index, (x_coord, y_coord) in enumerate(pixel_coords):
            distance = float(np.sqrt((x_coord - class_x) ** 2 + y_coord**2))
            raw = cppn.activate([x_coord, y_coord, float(class_x), distance, 1.0])[0]
            weight = (2.0 * float(raw) - 1.0) * weight_scale
            if abs(weight) >= threshold:
                weight_matrix[class_index, pixel_index] = weight

        raw_bias = cppn.activate([0.0, 0.0, float(class_x), 0.0, -1.0])[0]
        bias[class_index] = (2.0 * float(raw_bias) - 1.0) * bias_scale

    return weight_matrix, bias


def build_hyperneat_feature_substrate(
    genome: neat.DefaultGenome,
    config: neat.Config,
    output_size: int = 10,
    weight_scale: float = 4.0,
    bias_scale: float = 2.0,
    threshold: float = 0.02,
) -> tuple[np.ndarray, np.ndarray]:
    """Generate classifier weights for compact NEAT features from a CPPN."""

    cppn = neat.nn.FeedForwardNetwork.create(genome, config)
    feature_coords = np.linspace(-1.0, 1.0, FEATURE_SIZE)
    class_coords = np.linspace(-1.0, 1.0, output_size)
    weight_matrix = np.zeros((output_size, FEATURE_SIZE), dtype=np.float64)
    bias = np.zeros(output_size, dtype=np.float64)

    for class_index, class_x in enumerate(class_coords):
        for feature_index, feature_x in enumerate(feature_coords):
            distance = abs(float(feature_x - class_x))
            raw = cppn.activate([float(feature_x), 0.0, float(class_x), distance, 1.0])[0]
            weight = (2.0 * float(raw) - 1.0) * weight_scale
            if abs(weight) >= threshold:
                weight_matrix[class_index, feature_index] = weight

        raw_bias = cppn.activate([0.0, 0.0, float(class_x), 0.0, -1.0])[0]
        bias[class_index] = (2.0 * float(raw_bias) - 1.0) * bias_scale

    return weight_matrix, bias


def build_hyperneat_prototype_substrate(
    genome: neat.DefaultGenome,
    config: neat.Config,
    output_size: int = 10,
    weight_scale: float = 5.0,
    bias_scale: float = 2.0,
    threshold: float = 0.02,
) -> tuple[np.ndarray, np.ndarray]:
    """Generate classifier weights for prototype features from a CPPN."""

    cppn = neat.nn.FeedForwardNetwork.create(genome, config)
    class_coords = np.linspace(-1.0, 1.0, output_size)
    feature_coords = np.tile(class_coords, 2)
    channel_coords = np.repeat([-0.5, 0.5], output_size)
    weight_matrix = np.zeros((output_size, PROTOTYPE_FEATURE_SIZE), dtype=np.float64)
    bias = np.zeros(output_size, dtype=np.float64)

    for class_index, class_x in enumerate(class_coords):
        for feature_index, (feature_x, channel_y) in enumerate(zip(feature_coords, channel_coords, strict=True)):
            distance = float(np.sqrt((feature_x - class_x) ** 2 + channel_y**2))
            raw = cppn.activate([float(feature_x), float(channel_y), float(class_x), distance, 1.0])[0]
            weight = (2.0 * float(raw) - 1.0) * weight_scale
            if abs(weight) >= threshold:
                weight_matrix[class_index, feature_index] = weight

        raw_bias = cppn.activate([0.0, 0.0, float(class_x), 0.0, -1.0])[0]
        bias[class_index] = (2.0 * float(raw_bias) - 1.0) * bias_scale

    return weight_matrix, bias


def evaluate_winner(
    genome: neat.DefaultGenome,
    config: neat.Config,
    x_test: torch.Tensor,
    y_test: torch.Tensor,
    limit: int,
) -> float:
    """Evaluate the evolved NEAT network without SGD retraining."""

    eval_indices = _balanced_indices(y_test, limit)
    return _accuracy_genome(genome, config, x_test[eval_indices].numpy(), y_test[eval_indices].numpy())


def _accuracy_genome(
    genome: neat.DefaultGenome,
    config: neat.Config,
    x_data: np.ndarray,
    y_data: np.ndarray,
) -> float:
    network = neat.nn.FeedForwardNetwork.create(genome, config)
    correct = 0

    for image, label in zip(x_data, y_data, strict=True):
        output = network.activate(image.tolist())
        correct += int(int(np.argmax(output)) == int(label))

    return correct / max(len(y_data), 1)


def _fitness_genome(
    genome: neat.DefaultGenome,
    config: neat.Config,
    x_data: np.ndarray,
    y_data: np.ndarray,
) -> float:
    """Continuous NEAT fitness with accuracy, probability, margin, and log loss."""

    network = neat.nn.FeedForwardNetwork.create(genome, config)
    correct = 0
    true_class_scores = []
    one_vs_all_scores = []
    margins = []
    losses = []

    for image, label in zip(x_data, y_data, strict=True):
        output = np.asarray(network.activate(image.tolist()), dtype=np.float64)
        probabilities = _softmax(output)
        true_label = int(label)
        predicted = int(np.argmax(probabilities))
        correct += int(predicted == true_label)
        true_probability = float(np.clip(probabilities[true_label], 1e-9, 1.0))
        true_class_scores.append(true_probability)
        losses.append(-np.log(true_probability))
        competitors = np.delete(probabilities, true_label)
        margins.append(true_probability - float(np.max(competitors)))
        target = np.zeros_like(output)
        target[true_label] = 1.0
        clipped_output = np.clip(output, 1e-6, 1.0 - 1e-6)
        binary_loss = -np.mean(
            target * np.log(clipped_output) + (1.0 - target) * np.log(1.0 - clipped_output)
        )
        one_vs_all_scores.append(1.0 / (1.0 + float(binary_loss)))

    accuracy = correct / max(len(y_data), 1)
    confidence = float(np.mean(true_class_scores)) if true_class_scores else 0.0
    margin = float(np.mean(margins)) if margins else -1.0
    loss_score = 1.0 / (1.0 + float(np.mean(losses))) if losses else 0.0
    one_vs_all = float(np.mean(one_vs_all_scores)) if one_vs_all_scores else 0.0
    return 1.55 * accuracy + 0.35 * confidence + 0.25 * loss_score + 0.25 * one_vs_all + 0.20 * margin


def _fitness_logits(logits: np.ndarray, labels: np.ndarray) -> float:
    probabilities = _softmax_rows(logits)
    predictions = np.argmax(probabilities, axis=1)
    accuracy = float(np.mean(predictions == labels)) if len(labels) else 0.0
    true_probabilities = np.clip(probabilities[np.arange(len(labels)), labels], 1e-9, 1.0)
    competitors = probabilities.copy()
    competitors[np.arange(len(labels)), labels] = -np.inf
    margins = true_probabilities - np.max(competitors, axis=1)
    loss_score = 1.0 / (1.0 + float(np.mean(-np.log(true_probabilities))))
    return 1.70 * accuracy + 0.45 * loss_score + 0.25 * float(np.mean(margins))


def _softmax(values: np.ndarray) -> np.ndarray:
    shifted = values - np.max(values)
    exp_values = np.exp(shifted)
    return exp_values / np.sum(exp_values)


def _softmax_rows(values: np.ndarray) -> np.ndarray:
    shifted = values - np.max(values, axis=1, keepdims=True)
    exp_values = np.exp(shifted)
    return exp_values / np.sum(exp_values, axis=1, keepdims=True)


def _pixel_coordinates(image_size: int) -> list[tuple[float, float]]:
    if image_size == 1:
        return [(0.0, 0.0)]
    coords = np.linspace(-1.0, 1.0, image_size)
    return [(float(x_coord), float(y_coord)) for y_coord in coords for x_coord in coords]


def _variant_family(config_path: Path) -> str:
    variant_name = config_path.stem.lower()

    if "hyperneat_prototypes" in variant_name:
        return "hyperneat_prototypes"

    if "hyperneat_features" in variant_name:
        return "hyperneat_features"

    if "hyperneat" in variant_name:
        return "hyperneat"

    if "seeded_prototypes" in variant_name:
        return "seeded_prototypes"

    if "prototypes" in variant_name:
        return "prototypes"

    if "features" in variant_name:
        return "features"

    return "direct"


def _seed_prototype_population(
    population: dict[int, neat.DefaultGenome],
    config: neat.Config,
) -> None:
    """Seed a few genomes with the obvious prototype-to-class mapping."""

    input_keys = list(config.genome_config.input_keys)
    output_keys = list(config.genome_config.output_keys)
    if len(input_keys) < 20 or len(output_keys) != 10:
        return

    seeded_count = max(1, min(len(population), len(population) // 5))
    for genome_index, genome in enumerate(list(population.values())[:seeded_count]):
        strength = 4.0 + 0.25 * genome_index
        off_strength = -0.35
        for output_position, output_key in enumerate(output_keys):
            if output_key in genome.nodes:
                genome.nodes[output_key].bias = -0.6

            for feature_position, input_key in enumerate(input_keys):
                connection = genome.connections.get((input_key, output_key))
                if connection is None:
                    continue
                digit_position = feature_position % 10
                if digit_position == output_position:
                    connection.weight = strength if feature_position >= 10 else strength * 0.65
                    connection.enabled = True
                else:
                    connection.weight = off_strength
                    connection.enabled = True


def _balanced_indices(labels: torch.Tensor, limit: int) -> list[int]:
    """Pick a deterministic class-balanced subset for NEAT fitness/evaluation."""

    by_digit: dict[int, list[int]] = {digit: [] for digit in range(10)}
    for index, label in enumerate(labels.tolist()):
        by_digit[int(label)].append(index)

    selected: list[int] = []
    cursor = 0
    target = min(limit, len(labels))
    while len(selected) < target:
        added = False
        for digit in range(10):
            rows = by_digit[digit]
            if cursor < len(rows):
                selected.append(rows[cursor])
                added = True
                if len(selected) == target:
                    break
        if not added:
            break
        cursor += 1

    return selected


class _HistoryReporter(neat.reporting.BaseReporter):
    """Collect best and mean fitness for the UI."""

    def __init__(self) -> None:
        self.history: list[dict[str, float]] = []
        self._generation = 0

    def start_generation(self, generation: int) -> None:
        self._generation = generation

    def post_evaluate(self, config, population, species, best_genome) -> None:  # noqa: ANN001
        fitness_values = [genome.fitness or 0.0 for genome in population.values()]
        self.history.append(
            {
                "generation": float(self._generation),
                "best_fitness": float(best_genome.fitness or 0.0),
                "mean_fitness": float(np.mean(fitness_values)),
                "nodes": float(len(best_genome.nodes)),
                "enabled_connections": float(
                    sum(1 for connection in best_genome.connections.values() if connection.enabled)
                ),
            }
        )
