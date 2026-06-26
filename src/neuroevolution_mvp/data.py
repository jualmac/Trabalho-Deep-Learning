"""Dataset loading helpers for small, repeatable experiments."""

from __future__ import annotations

import random
from dataclasses import dataclass
from pathlib import Path

import numpy as np
import torch
from torch.utils.data import DataLoader, Subset, TensorDataset
from torchvision import datasets, transforms


SUPPORTED_DATASETS = ("mnist", "mnist100", "fashion_mnist", "cifar10", 'kmnist')


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

def _dataset_targets(dataset) -> list[int]:
    targets = getattr(dataset, "targets", None)

    if targets is None:
        return [int(dataset[index][1]) for index in range(len(dataset))]

    if isinstance(targets, torch.Tensor):
        return [int(target) for target in targets.tolist()]

    return [int(target) for target in targets]


def _balanced_subset(
    dataset,
    samples_per_class: int,
    seed: int,
    class_count: int = 10,
) -> Subset:
    targets = _dataset_targets(dataset)
    shuffled_indices = list(range(len(targets)))
    random.Random(seed).shuffle(shuffled_indices)

    indices_by_class: dict[int, list[int]] = {
        class_id: [] for class_id in range(class_count)
    }

    for index in shuffled_indices:
        class_id = targets[index]

        if class_id not in indices_by_class:
            continue

        if len(indices_by_class[class_id]) >= samples_per_class:
            continue

        indices_by_class[class_id].append(index)

        if all(
            len(class_indices) == samples_per_class
            for class_indices in indices_by_class.values()
        ):
            break

    missing_classes = [
        class_id
        for class_id, class_indices in indices_by_class.items()
        if len(class_indices) < samples_per_class
    ]

    if missing_classes:
        raise ValueError(
            "Could not build a balanced subset for classes: "
            f"{', '.join(str(class_id) for class_id in missing_classes)}."
        )

    selected_indices = [
        index
        for class_id in range(class_count)
        for index in indices_by_class[class_id]
    ]

    return Subset(dataset, selected_indices)

def _stack_subset(subset: Subset) -> tuple[torch.Tensor, torch.Tensor]:
    images: list[torch.Tensor] = []
    labels: list[int] = []

    for image, label in subset:
        images.append(image.flatten())
        labels.append(int(label))

    return torch.stack(images), torch.tensor(labels, dtype=torch.long)

def _build_transform(dataset_name: str, image_size: int, train: bool):
    normalized_name = dataset_name.strip().lower()

    if normalized_name in {"cifar10", "cifar100_10"}:
        normalize = transforms.Normalize(
            mean=(0.4914, 0.4822, 0.4465),
            std=(0.2470, 0.2435, 0.2616),
        )

        if train:
            return transforms.Compose(
                [
                    transforms.Resize((image_size, image_size)),
                    transforms.RandomCrop(image_size, padding=4),
                    transforms.RandomHorizontalFlip(),
                    transforms.ToTensor(),
                    normalize,
                ]
            )

        return transforms.Compose(
            [
                transforms.Resize((image_size, image_size)),
                transforms.ToTensor(),
                normalize,
            ]
        )

    return transforms.Compose(
        [
            transforms.Resize((image_size, image_size)),
            transforms.ToTensor(),
        ]
    )


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
    elif normalized_name == "mnist100":
        train = datasets.MNIST(data_dir, train=True, download=True, transform=transform)
        test = datasets.MNIST(data_dir, train=False, download=True, transform=transform)
        image_channels = 1
        class_names = tuple(str(label) for label in range(10))
    elif normalized_name == "kmnist":
        train = datasets.KMNIST(
            root=data_dir,
            train=True,
            download=True,
            transform=transform,
        )
        test = datasets.KMNIST(
            root=data_dir,
            train=False,
            download=True,
            transform=transform,
        )
        image_channels = 1
        class_names = tuple(str(label) for label in range(10))
    elif normalized_name == "fashion_mnist":
        train = datasets.FashionMNIST(
            data_dir,
            train=True,
            download=True,
            transform=transform,
        )
        test = datasets.FashionMNIST(
            data_dir,
            train=False,
            download=True,
            transform=transform,
        )
        image_channels = 1
        class_names = tuple(train.classes)
    elif normalized_name == "cifar10":
        train_transform = _build_transform(normalized_name, image_size, train=True)
        test_transform = _build_transform(normalized_name, image_size, train=False)
        train = datasets.CIFAR10(data_dir, train=True, download=True, transform=train_transform)
        test = datasets.CIFAR10(data_dir, train=False, download=True, transform=test_transform)
        image_channels = 3
        class_names = tuple(train.classes)
    else:
        raise ValueError(
            f"Unsupported dataset '{dataset_name}'. Use one of: {', '.join(SUPPORTED_DATASETS)}."
        )

    if normalized_name == "mnist100":
        train_subset = _balanced_subset(
            train,
            samples_per_class=100,
            seed=seed,
        )
    else:
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
