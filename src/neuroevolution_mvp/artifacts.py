"""Persistence helpers for trained demo artifacts."""

from __future__ import annotations

import json
import pickle
from pathlib import Path
from typing import Any

import neat
import torch
from torch import nn


def save_json(path: Path, payload: dict[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(payload, indent=2), encoding="utf-8")


def load_json(path: Path) -> dict[str, Any]:
    return json.loads(path.read_text(encoding="utf-8"))


def save_genome(path: Path, genome: neat.DefaultGenome) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("wb") as file:
        pickle.dump(genome, file)


def load_genome(path: Path) -> neat.DefaultGenome:
    with path.open("rb") as file:
        return pickle.load(file)


def save_model(path: Path, model: nn.Module) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    torch.save(model.state_dict(), path)
