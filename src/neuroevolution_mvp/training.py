"""Training and evaluation routines shared by CLI and Streamlit."""

from __future__ import annotations

import torch
from torch import nn
from torch.utils.data import DataLoader
from copy import deepcopy

from .models import (
    BaselineMLP,
    EvolvedTopologyMLP,
    FeatureWrappedModel,
    FixedNeatTopologyMLP,
    GeneratedSubstrateMLP,
    PrototypeWrappedModel,
    SmallCifarCNN
)

def build_optimizer(
    model: nn.Module,
    optimizer_name: str,
    learning_rate: float,
) -> torch.optim.Optimizer:
    normalized_name = optimizer_name.strip().lower()

    if normalized_name == "sgd":
        return torch.optim.SGD(
            model.parameters(),
            lr=learning_rate,
            momentum=0.9,
            nesterov=True,
        )

    if normalized_name == "adam":
        return torch.optim.Adam(
            model.parameters(),
            lr=learning_rate,
        )

    if normalized_name == "adamw":
        return torch.optim.AdamW(
            model.parameters(),
            lr=learning_rate,
            weight_decay=1e-4,
        )

    raise ValueError(
        f"Unsupported optimizer '{optimizer_name}'. "
        "Use one of: sgd, adam, adamw."
    )


def train_classifier(
    model: nn.Module,
    train_loader: DataLoader,
    test_loader: DataLoader,
    epochs: int,
    learning_rate: float,
    optimizer_name: str = "sgd",
) -> list[dict[str, float]]:
    """Train a classifier and keep the best checkpoint by test accuracy."""

    optimizer = build_optimizer(
        model=model,
        optimizer_name=optimizer_name,
        learning_rate=learning_rate,
    )
    criterion = nn.CrossEntropyLoss()

    history: list[dict[str, float]] = []
    best_accuracy = -1.0
    best_epoch = 0
    best_state = deepcopy(model.state_dict())

    for epoch in range(1, epochs + 1):
        model.train()
        total_loss = 0.0
        total_samples = 0

        for x_batch, y_batch in train_loader:
            optimizer.zero_grad()

            logits = model(x_batch)
            loss = criterion(logits, y_batch)

            loss.backward()
            optimizer.step()

            batch_size = int(y_batch.numel())
            total_loss += float(loss.item()) * batch_size
            total_samples += batch_size

        train_loss = total_loss / max(total_samples, 1)
        test_accuracy = evaluate_accuracy(model, test_loader)

        is_best = test_accuracy > best_accuracy
        if is_best:
            best_accuracy = test_accuracy
            best_epoch = epoch
            best_state = deepcopy(model.state_dict())

        history.append(
            {
                "epoch": float(epoch),
                "train_loss": float(train_loss),
                "test_accuracy": float(test_accuracy),
                "is_best": float(is_best),
            }
        )

    model.load_state_dict(best_state)

    for row in history:
        row["best_epoch"] = float(best_epoch)
        row["best_test_accuracy"] = float(best_accuracy)

    return history

@torch.no_grad()
def extract_cnn_embeddings(
    model: SmallCifarCNN,
    inputs: torch.Tensor,
    image_size: int,
    batch_size: int = 256,
) -> torch.Tensor:
    model.eval()
    embedding_rows = []

    for batch_start in range(0, len(inputs), batch_size):
        batch = inputs[batch_start : batch_start + batch_size]
        batch = batch.reshape(-1, 3, image_size, image_size)
        embeddings = model.forward_features(batch)
        embedding_rows.append(embeddings.cpu())

    return torch.cat(embedding_rows, dim=0)

@torch.no_grad()
def evaluate_accuracy(model: nn.Module, loader: DataLoader) -> float:
    """Compute classification accuracy."""

    model.eval()
    correct = 0
    total = 0

    for x_batch, y_batch in loader:
        predictions = model(x_batch).argmax(dim=1)
        correct += int((predictions == y_batch).sum().item())
        total += int(y_batch.numel())

    return correct / max(total, 1)
