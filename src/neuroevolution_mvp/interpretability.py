"""Simple visual interpretability helpers."""

from __future__ import annotations

import torch
from torch import nn


@torch.no_grad()
def prediction_summary(model: nn.Module, image: torch.Tensor) -> dict[str, float | int]:
    """Return the predicted class and confidence for a single flattened image."""

    model.eval()
    probabilities = torch.softmax(model(image.flatten().unsqueeze(0)), dim=1)
    predicted_class = int(probabilities.argmax(dim=1).item())
    return {
        "class": predicted_class,
        "confidence": float(probabilities[0, predicted_class].item()),
    }


@torch.no_grad()
def occlusion_map(model: nn.Module, image: torch.Tensor, image_size: int) -> torch.Tensor:
    """Estimate pixel importance by zeroing one pixel at a time.

    The map measures how much the model's confidence in its original prediction
    drops when each pixel is hidden. This is slower than gradients, but it works
    for both ordinary PyTorch models and sparse evolved topologies.
    """

    model.eval()
    flat = image.flatten().unsqueeze(0)
    channels = _infer_channels(flat.shape[1], image_size)
    logits = model(flat)
    probabilities = torch.softmax(logits, dim=1)
    predicted_class = int(probabilities.argmax(dim=1).item())
    baseline_score = probabilities[0, predicted_class]

    importance = torch.zeros(image_size * image_size)
    for pixel in range(image_size * image_size):
        occluded = flat.clone()
        for channel in range(channels):
            occluded[0, channel * image_size * image_size + pixel] = 0.0
        occluded_probability = torch.softmax(model(occluded), dim=1)[0, predicted_class]
        importance[pixel] = torch.clamp(baseline_score - occluded_probability, min=0.0)

    max_value = importance.max()
    if max_value > 0:
        importance = importance / max_value
    return importance.reshape(image_size, image_size)


def _infer_channels(flat_size: int, image_size: int) -> int:
    pixels = image_size * image_size
    if flat_size % pixels != 0:
        raise ValueError(
            f"Flattened input size {flat_size} is incompatible with image size {image_size}."
        )
    return flat_size // pixels

def block_occlusion_map(
    model: torch.nn.Module,
    image: torch.Tensor,
    image_size: int,
    patch_size: int = 3,
    stride: int = 1,
    occlusion_value: float = 0.0,
) -> torch.Tensor:
    model.eval()

    with torch.no_grad():
        original_logits = model(image.unsqueeze(0))
        probabilities = torch.softmax(original_logits, dim=1)
        target_class = int(probabilities.argmax(dim=1).item())
        original_confidence = float(probabilities[0, target_class].item())

    if image.ndim == 2:
        heatmap = torch.zeros_like(image)
        height, width = image.shape
    else:
        heatmap = torch.zeros(image.shape[-2:], dtype=image.dtype)
        height, width = image.shape[-2:]

    for row_start in range(0, height, stride):
        for col_start in range(0, width, stride):
            row_end = min(row_start + patch_size, height)
            col_end = min(col_start + patch_size, width)

            occluded = image.clone()
            if occluded.ndim == 2:
                occluded[row_start:row_end, col_start:col_end] = occlusion_value
            else:
                occluded[:, row_start:row_end, col_start:col_end] = occlusion_value

            with torch.no_grad():
                logits = model(occluded.unsqueeze(0))
                probabilities = torch.softmax(logits, dim=1)
                occluded_confidence = float(probabilities[0, target_class].item())

            importance = max(original_confidence - occluded_confidence, 0.0)
            heatmap[row_start:row_end, col_start:col_end] = torch.maximum(
                heatmap[row_start:row_end, col_start:col_end],
                torch.tensor(importance, dtype=heatmap.dtype),
            )

    max_value = float(heatmap.max().item())
    if max_value > 0:
        heatmap = heatmap / max_value

    return heatmap

def compare_maps(
    first: torch.Tensor,
    second: torch.Tensor,
    top_fraction: float = 0.15,
) -> dict[str, float]:
    """Compare two normalized attribution maps.

    Cosine similarity captures whether both maps point in a similar direction.
    Top-k overlap asks whether the most relevant pixels are roughly the same.
    Mean absolute difference is easier to read visually: lower means closer.
    """

    first_flat = first.flatten().float()
    second_flat = second.flatten().float()
    denominator = torch.linalg.norm(first_flat) * torch.linalg.norm(second_flat)
    cosine = 0.0
    if denominator > 0:
        cosine = float(torch.dot(first_flat, second_flat) / denominator)

    k = max(1, int(first_flat.numel() * top_fraction))
    first_top = set(torch.topk(first_flat, k).indices.tolist())
    second_top = set(torch.topk(second_flat, k).indices.tolist())
    top_overlap = len(first_top.intersection(second_top)) / k

    return {
        "cosine_similarity": cosine,
        "top_pixel_overlap": float(top_overlap),
        "mean_absolute_difference": float(torch.mean(torch.abs(first_flat - second_flat)).item()),
    }
