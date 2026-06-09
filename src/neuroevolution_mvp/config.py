"""Configuration objects used by the neuroevolution MVP."""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path


PROJECT_ROOT = Path(__file__).resolve().parents[2]
DEFAULT_ARTIFACT_DIR = PROJECT_ROOT / "src" / "artifacts"
DEFAULT_NEAT_CONFIG = PROJECT_ROOT / "src" / "configs" / "neat_mnist_14x14.ini"


@dataclass(frozen=True)
class ExperimentConfig:
    """Small defaults intended to finish on a normal laptop during a demo."""

    seed: int = 42
    image_size: int = 14
    train_limit: int = 1_000
    test_limit: int = 300
    neat_eval_limit: int = 500
    neat_generations: int = 5
    neat_winner_eval_limit: int = 300
    baseline_epochs: int = 5
    evolved_sgd_epochs: int = 5
    batch_size: int = 64
    learning_rate: float = 0.01
    hidden_units: int = 64
    artifact_dir: Path = DEFAULT_ARTIFACT_DIR
    neat_config_path: Path = DEFAULT_NEAT_CONFIG

    @property
    def input_size(self) -> int:
        """Flattened image size expected by the networks."""

        return self.image_size * self.image_size
