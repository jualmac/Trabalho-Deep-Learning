"""PyTorch models for the SGD baseline and evolved topology retraining."""

from __future__ import annotations

import neat
import torch
from neat.graphs import feed_forward_layers
from torch import nn


class BaselineMLP(nn.Module):
    """A compact MLP used as the conventional SGD baseline."""

    def __init__(self, input_size: int, hidden_units: int = 64, output_size: int = 10):
        super().__init__()
        self.network = nn.Sequential(
            nn.Linear(input_size, hidden_units),
            nn.ReLU(),
            nn.Linear(hidden_units, output_size),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.network(x)


class EvolvedTopologyMLP(nn.Module):
    """Trainable PyTorch network that reuses a NEAT genome's enabled topology.

    NEAT node identifiers are kept as string keys inside ModuleDict and
    ParameterDict because PyTorch parameter names cannot contain arbitrary
    integer keys directly.
    """

    def __init__(
        self,
        genome: neat.DefaultGenome,
        config: neat.Config,
        input_size: int,
        output_size: int = 10,
        initialize_from_genome: bool = False,
        neat_activation: bool = False,
    ):
        super().__init__()
        self.input_keys = list(config.genome_config.input_keys)
        self.output_keys = list(config.genome_config.output_keys)
        self.output_size = output_size
        self.neat_activation = neat_activation
        enabled_connections = _enabled_connections(genome)
        neat_layers = feed_forward_layers(
            self.input_keys,
            self.output_keys,
            enabled_connections,
        )
        layers, required = neat_layers if isinstance(neat_layers, tuple) else (neat_layers, set())
        required_with_inputs = set(required).union(self.input_keys)
        self.layers = [sorted(layer) for layer in layers]

        node_keys = sorted({node for layer in self.layers for node in layer})
        self.biases = nn.ParameterDict(
            {str(node): nn.Parameter(torch.zeros(())) for node in node_keys}
        )
        self.weights = nn.ParameterDict()
        self.incoming_by_node: dict[int, list[tuple[int, int]]] = {
            node: [] for node in node_keys
        }

        for source, target in enabled_connections:
            if target not in node_keys or source not in required_with_inputs:
                continue
            if target in self.incoming_by_node:
                self.incoming_by_node[target].append((source, target))
            self.weights[_edge_key(source, target)] = nn.Parameter(
                torch.empty(())
            )
            if initialize_from_genome:
                weight = genome.connections[(source, target)].weight
                self.weights[_edge_key(source, target)].data.fill_(weight)
            else:
                nn.init.normal_(self.weights[_edge_key(source, target)], mean=0.0, std=0.2)

        if initialize_from_genome:
            for node in node_keys:
                self.biases[str(node)].data.fill_(genome.nodes[node].bias)

        self.fallback = None
        if not self.layers:
            self.fallback = nn.Linear(input_size, output_size)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        if self.fallback is not None:
            return self.fallback(x)

        activations: dict[int, torch.Tensor] = {
            key: x[:, position] for position, key in enumerate(self.input_keys)
        }

        for layer in self.layers:
            for node in layer:
                incoming = [
                    activations[source] * self.weights[_edge_key(source, target)]
                    for source, target in self.incoming_by_node[node]
                    if source in activations
                ]
                if incoming:
                    value = torch.stack(incoming, dim=0).sum(dim=0)
                else:
                    value = torch.zeros(x.shape[0], device=x.device, dtype=x.dtype)

                value = value + self.biases[str(node)]
                if self.neat_activation:
                    activations[node] = torch.sigmoid(value)
                else:
                    activations[node] = value if node in self.output_keys else torch.relu(value)

        outputs = []
        for key in self.output_keys:
            outputs.append(
                activations.get(
                    key,
                    torch.zeros(x.shape[0], device=x.device, dtype=x.dtype),
                )
            )
        return torch.stack(outputs, dim=1)


class FixedNeatTopologyMLP(EvolvedTopologyMLP):
    """PyTorch view of the NEAT winner with topology and weights from evolution."""

    def __init__(
        self,
        genome: neat.DefaultGenome,
        config: neat.Config,
        input_size: int,
        output_size: int = 10,
    ):
        super().__init__(
            genome=genome,
            config=config,
            input_size=input_size,
            output_size=output_size,
            initialize_from_genome=True,
            neat_activation=True,
        )
        for parameter in self.parameters():
            parameter.requires_grad = False


def _enabled_connections(genome: neat.DefaultGenome) -> list[tuple[int, int]]:
    return [
        (source, target)
        for (source, target), connection in genome.connections.items()
        if connection.enabled
    ]


def _edge_key(source: int, target: int) -> str:
    return f"{source}_to_{target}"
