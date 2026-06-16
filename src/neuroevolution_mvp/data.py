"""Dataset loading helpers for small, repeatable experiments."""

from __future__ import annotations

import random
from dataclasses import dataclass
from pathlib import Path

import numpy as np
import torch
from torch.utils.data import DataLoader, Subset, TensorDataset
from torchvision import datasets, transforms


SUPPORTED_DATASETS = ("mnist", "cifar10")


@dataclass(frozen=True)
class ImageDatasetData:
    """Tensors and loaders used by training, NEAT evaluation, and the UI."""

    name: str
    image_channels: int
    class_names: tuple[str, ...]
    x_train: torch.Tensor
    y_train: torch.Tensor
    x_test: torch.Tensor
    y_test: torch.Tensor
    train_loader: DataLoader
    test_loader: DataLoader


MnistData = ImageDatasetData


def set_seed(seed: int) -> None:
    """Make the demo deterministic enough for comparison slides."""

    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)


def _subset(dataset, limit: int, seed: int) -> Subset:
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


def load_dataset(
    dataset_name: str,
    image_size: int,
    train_limit: int,
    test_limit: int,
    batch_size: int,
    seed: int,
    data_dir: Path | str = "data",
) -> ImageDatasetData:
    """Load a resized image-classification subset.

    The images are intentionally resized. NEAT evaluates many candidate
    networks, so keeping the spatial resolution controlled is important for
    repeatable experiments.
    """

    transform = transforms.Compose(
        [
            transforms.Resize((image_size, image_size)),
            transforms.ToTensor(),
        ]
    )
    normalized_name = dataset_name.strip().lower()

    if normalized_name == "mnist":
        train = datasets.MNIST(data_dir, train=True, download=True, transform=transform)
        test = datasets.MNIST(data_dir, train=False, download=True, transform=transform)
        image_channels = 1
        class_names = tuple(str(label) for label in range(10))
    elif normalized_name == "cifar10":
        train = datasets.CIFAR10(data_dir, train=True, download=True, transform=transform)
        test = datasets.CIFAR10(data_dir, train=False, download=True, transform=transform)
        image_channels = 3
        class_names = tuple(train.classes)
    else:
        raise ValueError(
            f"Unsupported dataset '{dataset_name}'. Use one of: {', '.join(SUPPORTED_DATASETS)}."
        )

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

    return ImageDatasetData(
        name=normalized_name,
        image_channels=image_channels,
        class_names=class_names,
        x_train=x_train,
        y_train=y_train,
        x_test=x_test,
        y_test=y_test,
        train_loader=train_loader,
        test_loader=test_loader,
    )


def load_mnist(
    image_size: int,
    train_limit: int,
    test_limit: int,
    batch_size: int,
    seed: int,
    data_dir: Path | str = "data",
) -> ImageDatasetData:
    """Backward-compatible MNIST loader used by older scripts."""

    return load_dataset(
        "mnist",
        image_size=image_size,
        train_limit=train_limit,
        test_limit=test_limit,
        batch_size=batch_size,
        seed=seed,
        data_dir=data_dir,
    )
