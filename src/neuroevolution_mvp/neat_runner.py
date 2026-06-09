"""NEAT evolution utilities for discovering a small MNIST topology."""

from __future__ import annotations

from pathlib import Path

import neat
import numpy as np
import torch


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
) -> tuple[neat.DefaultGenome, list[dict[str, float]]]:
    """Evolve a topology and return the winning genome plus generation stats."""

    population = neat.Population(config)
    reporter = _HistoryReporter()
    population.add_reporter(reporter)
    population.add_reporter(neat.StdOutReporter(show_species_detail=False))

    x_eval = x_train[:eval_limit].numpy()
    y_eval = y_train[:eval_limit].numpy()

    def evaluate_population(genomes: list[tuple[int, neat.DefaultGenome]], neat_config: neat.Config) -> None:
        for _, genome in genomes:
            genome.fitness = _fitness_genome(genome, neat_config, x_eval, y_eval)

    winner = population.run(evaluate_population, generations)
    return winner, reporter.history


def evaluate_winner(
    genome: neat.DefaultGenome,
    config: neat.Config,
    x_test: torch.Tensor,
    y_test: torch.Tensor,
    limit: int,
) -> float:
    """Evaluate the evolved NEAT network without SGD retraining."""

    return _accuracy_genome(genome, config, x_test[:limit].numpy(), y_test[:limit].numpy())


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
    """Continuous NEAT fitness: accuracy plus confidence on the true class."""

    network = neat.nn.FeedForwardNetwork.create(genome, config)
    correct = 0
    true_class_scores = []

    for image, label in zip(x_data, y_data, strict=True):
        output = np.asarray(network.activate(image.tolist()), dtype=np.float64)
        probabilities = _softmax(output)
        correct += int(int(np.argmax(probabilities)) == int(label))
        true_class_scores.append(float(probabilities[int(label)]))

    accuracy = correct / max(len(y_data), 1)
    confidence = float(np.mean(true_class_scores)) if true_class_scores else 0.0
    return 0.8 * accuracy + 0.2 * confidence


def _softmax(values: np.ndarray) -> np.ndarray:
    shifted = values - np.max(values)
    exp_values = np.exp(shifted)
    return exp_values / np.sum(exp_values)


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
