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
def occlusion_map(
    model: nn.Module,
    image: torch.Tensor,
    image_size: int,
    occlusion_batch_size: int = 256,
    patch_size: int = 1,
) -> torch.Tensor:
    """Estimate spatial importance by zeroing one pixel or patch at a time.

    The map measures the absolute change in the predicted class logit when each
    pixel or patch is hidden. Logits avoid softmax saturation, and the absolute
    change captures both supporting and inhibiting local influence. This is
    slower than gradients, but works for ordinary and evolved topologies.
    """

    model.eval()
    flat = image.flatten().unsqueeze(0)
    channels = _infer_channels(flat.shape[1], image_size)
    logits = model(flat)
    predicted_class = int(logits.argmax(dim=1).item())
    baseline_score = logits[0, predicted_class]

    patch_size = max(1, patch_size)
    pixel_count = image_size * image_size
    regions = []
    for row_start in range(0, image_size, patch_size):
        for col_start in range(0, image_size, patch_size):
            spatial_indices = []
            for row in range(row_start, min(row_start + patch_size, image_size)):
                for col in range(col_start, min(col_start + patch_size, image_size)):
                    spatial_indices.append(row * image_size + col)
            regions.append(torch.tensor(spatial_indices, device=flat.device))
    importance = torch.zeros(pixel_count, device=flat.device)
    for start in range(0, len(regions), occlusion_batch_size):
        stop = min(start + occlusion_batch_size, len(regions))
        occluded = flat.repeat(stop - start, 1)
        for local_index, spatial_indices in enumerate(regions[start:stop]):
            for channel in range(channels):
                channel_indices = channel * pixel_count + spatial_indices
                occluded[local_index, channel_indices] = 0.0
        occluded_scores = model(occluded)[:, predicted_class]
        region_scores = torch.abs(baseline_score - occluded_scores)
        for local_index, spatial_indices in enumerate(regions[start:stop]):
            importance[spatial_indices] = region_scores[local_index]

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


def compare_maps(
    first: torch.Tensor,
    second: torch.Tensor,
    top_fraction: float = 0.15,
) -> dict[str, float]:
    """Compare two normalized attribution maps.

    Cosine similarity captures whether both maps point in a similar direction.
    Top-k overlap asks whether the most relevant pixels are roughly the same.
    Top-k IoU divides their intersection by their union. Weighted Jaccard
    compares the full attribution mass instead of applying a hard threshold.
    """

    first_flat = first.flatten().float()
    second_flat = second.flatten().float()
    first_norm = torch.linalg.norm(first_flat)
    second_norm = torch.linalg.norm(second_flat)
    denominator = first_norm * second_norm
    valid_attribution = bool(first_norm > 1e-12 and second_norm > 1e-12)
    cosine = 0.0
    if valid_attribution:
        cosine = float(torch.dot(first_flat, second_flat) / denominator)

    requested_k = max(1, int(first_flat.numel() * top_fraction))
    first_positive = int((first_flat > 0).sum().item())
    second_positive = int((second_flat > 0).sum().item())
    k = min(requested_k, first_positive, second_positive)
    if k == 0:
        return {
            "cosine_similarity": cosine,
            "top_pixel_overlap": 0.0,
            "top_pixel_iou": 0.0,
            "weighted_jaccard": 0.0,
            "mean_absolute_difference": float(
                torch.mean(torch.abs(first_flat - second_flat)).item()
            ),
            "valid_attribution": 0.0,
        }
    first_top = set(torch.topk(first_flat, k).indices.tolist())
    second_top = set(torch.topk(second_flat, k).indices.tolist())
    intersection_size = len(first_top.intersection(second_top))
    union_size = len(first_top.union(second_top))
    top_overlap = intersection_size / k
    top_iou = intersection_size / max(union_size, 1)

    weighted_union = torch.maximum(first_flat, second_flat).sum()
    weighted_jaccard = 0.0
    if weighted_union > 1e-12:
        weighted_jaccard = float(
            torch.minimum(first_flat, second_flat).sum() / weighted_union
        )

    return {
        "cosine_similarity": cosine,
        "top_pixel_overlap": float(top_overlap),
        "top_pixel_iou": float(top_iou),
        "weighted_jaccard": weighted_jaccard,
        "mean_absolute_difference": float(torch.mean(torch.abs(first_flat - second_flat)).item()),
        "valid_attribution": float(valid_attribution),
    }
