"""Training and evaluation routines shared by CLI and Streamlit."""

from __future__ import annotations

import torch
from torch import nn
from torch.utils.data import DataLoader

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
    """Train a classifier and return per-epoch metrics."""

    optimizer = build_optimizer(
        model=model,
        optimizer_name=optimizer_name,
        learning_rate=learning_rate,
    )
    criterion = nn.CrossEntropyLoss()
    history: list[dict[str, float]] = []

    for epoch in range(1, epochs + 1):
        model.train()
        total_loss = 0.0
        total_examples = 0

        for x_batch, y_batch in train_loader:
            optimizer.zero_grad()
            logits = model(x_batch)
            loss = criterion(logits, y_batch)
            loss.backward()
            optimizer.step()

            total_loss += float(loss.item()) * x_batch.shape[0]
            total_examples += x_batch.shape[0]

        test_accuracy = evaluate_accuracy(model, test_loader)
        history.append(
            {
                "epoch": float(epoch),
                "train_loss": total_loss / max(total_examples, 1),
                "test_accuracy": test_accuracy,
            }
        )

    return history


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
