"""Configuration objects used by the neuroevolution MVP."""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path


PROJECT_ROOT = Path(__file__).resolve().parents[2]
DEFAULT_ARTIFACT_DIR = PROJECT_ROOT / "src" / "artifacts"
DEFAULT_NEAT_CONFIG = PROJECT_ROOT / "src" / "configs" / "neat_mnist_features_full.ini"
ALL_NEAT_VARIANT_CONFIGS = (
    PROJECT_ROOT / "src" / "configs" / "neat_mnist_features_full.ini",
    PROJECT_ROOT / "src" / "configs" / "neat_mnist_seeded_prototypes_full.ini",
    PROJECT_ROOT / "src" / "configs" / "hyperneat_mnist_prototypes_cppn.ini",
    PROJECT_ROOT / "src" / "configs" / "hyperneat_mnist_features_cppn.ini",
    PROJECT_ROOT / "src" / "configs" / "neat_mnist_prototypes_full.ini",
    PROJECT_ROOT / "src" / "configs" / "hyperneat_mnist_cppn.ini",
    PROJECT_ROOT / "src" / "configs" / "neat_mnist_14x14.ini",
    PROJECT_ROOT / "src" / "configs" / "neat_mnist_14x14_fs.ini",
    PROJECT_ROOT / "src" / "configs" / "neat_mnist_14x14_dense.ini",
    PROJECT_ROOT / "src" / "configs" / "neat_mnist_14x14_full.ini",
    PROJECT_ROOT / "src" / "configs" / "neat_mnist_14x14_hidden.ini",
)
DEFAULT_NEAT_VARIANT_CONFIGS = (
    DEFAULT_NEAT_CONFIG,
    PROJECT_ROOT / "src" / "configs" / "neat_mnist_seeded_prototypes_full.ini",
    PROJECT_ROOT / "src" / "configs" / "hyperneat_mnist_prototypes_cppn.ini",
)


@dataclass(frozen=True)
class ExperimentConfig:
    """Defaults tuned for quality over quick classroom demos."""

    seed: int = 42
    dataset: str = "mnist"
    image_size: int = 26
    train_limit: int = 5_000
    test_limit: int = 10_000
    neat_eval_limit: int = 2_000
    neat_generations: int = 35
    neat_winner_eval_limit: int = 10_000
    interpretability_samples: int = 0
    interpretability_grid_size: int = 8
    baseline_epochs: int = 15
    evolved_sgd_epochs: int = 25
    batch_size: int = 64
    learning_rate: float = 0.03
    optimizer_name: str = "sgd"
    hidden_units: int = 64
    artifact_dir: Path = DEFAULT_ARTIFACT_DIR
    neat_config_path: Path = DEFAULT_NEAT_CONFIG
    neat_variant_paths: tuple[Path, ...] = DEFAULT_NEAT_VARIANT_CONFIGS

    @property
    def image_channels(self) -> int:
        """Number of channels used by the selected dataset."""

        return 3 if self.dataset.lower() == "cifar10" else 1

    @property
    def input_size(self) -> int:
        """Flattened image size expected by the networks."""

        return self.image_channels * self.image_size * self.image_size
