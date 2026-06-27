"""Feature extraction helpers for NEAT-friendly image inputs."""

from __future__ import annotations

import numpy as np
import torch
import torch.nn.functional as F


FEATURE_GRID_SIZE = 7
FEATURE_SIZE = FEATURE_GRID_SIZE * FEATURE_GRID_SIZE + 2 * FEATURE_GRID_SIZE + 6
PROTOTYPE_FEATURE_SIZE = 20


def extract_neat_features_torch(x: torch.Tensor, image_size: int) -> torch.Tensor:
    """Convert flattened images into compact shape descriptors for NEAT."""

    single = x.ndim == 1
    batch = x.reshape(1, -1) if single else x
    channels = _infer_channels(batch.shape[1], image_size)
    images = batch.reshape(batch.shape[0], channels, image_size, image_size)
    images = images.mean(dim=1, keepdim=True)

    pooled = F.adaptive_avg_pool2d(images, (FEATURE_GRID_SIZE, FEATURE_GRID_SIZE)).flatten(1)
    row_projection = F.adaptive_avg_pool1d(
        images.squeeze(1).mean(dim=2).unsqueeze(1),
        FEATURE_GRID_SIZE,
    ).squeeze(1)
    col_projection = F.adaptive_avg_pool1d(
        images.squeeze(1).mean(dim=1).unsqueeze(1),
        FEATURE_GRID_SIZE,
    ).squeeze(1)

    coords = torch.linspace(-1.0, 1.0, image_size, device=batch.device, dtype=batch.dtype)
    mass = images.sum(dim=(1, 2, 3)).clamp_min(1e-6)
    x_mass = images.squeeze(1).sum(dim=1)
    y_mass = images.squeeze(1).sum(dim=2)
    center_x = (x_mass * coords).sum(dim=1) / mass
    center_y = (y_mass * coords).sum(dim=1) / mass
    var_x = (x_mass * (coords.unsqueeze(0) - center_x.unsqueeze(1)).pow(2)).sum(dim=1) / mass
    var_y = (y_mass * (coords.unsqueeze(0) - center_y.unsqueeze(1)).pow(2)).sum(dim=1) / mass
    density = images.mean(dim=(1, 2, 3))
    ink_strength = images.amax(dim=(1, 2, 3))
    moments = torch.stack([density, ink_strength, center_x, center_y, var_x, var_y], dim=1)

    features = torch.cat([pooled, row_projection, col_projection, moments], dim=1)
    return features.squeeze(0) if single else features


def extract_neat_features_numpy(x_data: np.ndarray, image_size: int) -> np.ndarray:
    tensor = torch.from_numpy(x_data.astype(np.float32, copy=False))
    return extract_neat_features_torch(tensor, image_size).numpy()


def class_centroids(x_data: torch.Tensor, y_data: torch.Tensor, output_size: int = 10) -> torch.Tensor:
    centroids = []
    for label in range(output_size):
        rows = x_data[y_data == label]
        if len(rows) == 0:
            centroids.append(torch.zeros(x_data.shape[1], dtype=x_data.dtype))
        else:
            centroids.append(rows.mean(dim=0))
    return torch.stack(centroids)


def extract_prototype_features_torch(x: torch.Tensor, centroids: torch.Tensor) -> torch.Tensor:
    single = x.ndim == 1
    batch = x.reshape(1, -1) if single else x
    batch_norm = F.normalize(batch, dim=1)
    centroid_norm = F.normalize(centroids.to(batch.device, batch.dtype), dim=1)
    cosine = batch_norm @ centroid_norm.t()
    distances = torch.cdist(batch, centroids.to(batch.device, batch.dtype), p=2)
    similarity = torch.exp(-distances / distances.shape[1] ** 0.5)
    features = torch.cat([cosine, similarity], dim=1)
    return features.squeeze(0) if single else features


def extract_prototype_features_numpy(x_data: np.ndarray, centroids: torch.Tensor) -> np.ndarray:
    tensor = torch.from_numpy(x_data.astype(np.float32, copy=False))
    return extract_prototype_features_torch(tensor, centroids).numpy()


def _infer_channels(flat_size: int, image_size: int) -> int:
    pixels = image_size * image_size
    if flat_size % pixels != 0:
        raise ValueError(
            f"Flattened input size {flat_size} is incompatible with image size {image_size}."
        )
    return flat_size // pixels
