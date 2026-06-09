"""MNIST loading helpers for small, repeatable experiments."""

from __future__ import annotations

import random
from dataclasses import dataclass
from pathlib import Path

import numpy as np
import torch
from torch.utils.data import DataLoader, Subset, TensorDataset
from torchvision import datasets, transforms


@dataclass(frozen=True)
class MnistData:
    """Tensors and loaders used by training, NEAT evaluation, and the UI."""

    x_train: torch.Tensor
    y_train: torch.Tensor
    x_test: torch.Tensor
    y_test: torch.Tensor
    train_loader: DataLoader
    test_loader: DataLoader


def set_seed(seed: int) -> None:
    """Make the demo deterministic enough for comparison slides."""

    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)


def _subset(dataset: datasets.MNIST, limit: int, seed: int) -> Subset:
    generator = torch.Generator().manual_seed(seed)
    size = min(limit, len(dataset))
    indices = torch.randperm(len(dataset), generator=generator)[:size].tolist()
    return Subset(dataset, indices)


def _stack_subset(subset: Subset) -> tuple[torch.Tensor, torch.Tensor]:
    images: list[torch.Tensor] = []
    labels: list[int] = []

    for image, label in subset:
        images.append(image.flatten())
        labels.append(int(label))

    return torch.stack(images), torch.tensor(labels, dtype=torch.long)


def load_mnist(
    image_size: int,
    train_limit: int,
    test_limit: int,
    batch_size: int,
    seed: int,
    data_dir: Path | str = "data",
) -> MnistData:
    """Load a resized MNIST subset.

    The images are intentionally downsampled. NEAT evaluates many candidate
    networks, so using 8x8 images keeps the input graph small enough for class
    demos while preserving the recognizable digit structure.
    """

    transform = transforms.Compose(
        [
            transforms.Resize((image_size, image_size)),
            transforms.ToTensor(),
        ]
    )

    train = datasets.MNIST(data_dir, train=True, download=True, transform=transform)
    test = datasets.MNIST(data_dir, train=False, download=True, transform=transform)

    train_subset = _subset(train, train_limit, seed)
    test_subset = _subset(test, test_limit, seed + 1)
    x_train, y_train = _stack_subset(train_subset)
    x_test, y_test = _stack_subset(test_subset)

    train_loader = DataLoader(
        TensorDataset(x_train, y_train),
        batch_size=batch_size,
        shuffle=True,
        generator=torch.Generator().manual_seed(seed),
    )
    test_loader = DataLoader(
        TensorDataset(x_test, y_test),
        batch_size=batch_size,
        shuffle=False,
    )

    return MnistData(
        x_train=x_train,
        y_train=y_train,
        x_test=x_test,
        y_test=y_test,
        train_loader=train_loader,
        test_loader=test_loader,
    )
