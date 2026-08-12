"""Small, auditable neural baseline for existing evidence features."""

from __future__ import annotations

import torch
from torch import nn


class TabularMLP(nn.Module):
    """GELU MLP producing a single correct-identification logit."""

    def __init__(self, input_dim: int, hidden_dims=(128, 64), dropout=0.15):
        super().__init__()
        if input_dim < 1:
            raise ValueError("input_dim must be positive")
        dims = [int(value) for value in hidden_dims]
        if not dims or any(value < 1 for value in dims):
            raise ValueError("hidden_dims must contain positive integers")
        if not 0.0 <= float(dropout) < 1.0:
            raise ValueError("dropout must be in [0, 1)")

        layers = []
        previous = int(input_dim)
        for width in dims:
            layers.extend([
                nn.Linear(previous, width),
                nn.LayerNorm(width),
                nn.GELU(),
                nn.Dropout(float(dropout)),
            ])
            previous = width
        layers.append(nn.Linear(previous, 1))
        self.network = nn.Sequential(*layers)
        self.input_dim = int(input_dim)
        self.hidden_dims = tuple(dims)
        self.dropout = float(dropout)

    def forward(self, features):
        return self.network(features).squeeze(-1)

    def architecture(self) -> dict:
        return {
            "type": "tabular_mlp_v1",
            "input_dim": self.input_dim,
            "hidden_dims": list(self.hidden_dims),
            "dropout": self.dropout,
            "activation": "gelu",
            "normalization": "layer_norm",
        }


def n_trainable_parameters(model: nn.Module) -> int:
    return int(sum(
        parameter.numel() for parameter in model.parameters()
        if parameter.requires_grad))
