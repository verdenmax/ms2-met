"""Signal-native neural model for paired precursor and fragment XICs."""

from __future__ import annotations

import torch
from torch import nn


class TraceEncoder(nn.Module):
    """Small shared temporal encoder for fixed-length XIC traces."""

    def __init__(self, input_channels: int, hidden_dim: int, dropout: float):
        super().__init__()
        self.network = nn.Sequential(
            nn.Conv1d(input_channels, hidden_dim, kernel_size=3, padding=1),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Conv1d(hidden_dim, hidden_dim, kernel_size=3, padding=1),
            nn.GELU(),
        )
        self.output_dim = 2 * hidden_dim

    def forward(self, values: torch.Tensor) -> torch.Tensor:
        encoded = self.network(values)
        return torch.cat([
            encoded.mean(dim=-1), encoded.amax(dim=-1),
        ], dim=-1)


class MaskedAttentionPool(nn.Module):
    """Learned set pooling with a finite all-masked result."""

    def __init__(self, input_dim: int, attention_dim: int):
        super().__init__()
        self.score = nn.Sequential(
            nn.Linear(input_dim, attention_dim),
            nn.Tanh(),
            nn.Linear(attention_dim, 1),
        )

    def forward(self, values: torch.Tensor,
                mask: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
        logits = self.score(values).squeeze(-1)
        valid = mask.to(dtype=torch.bool)
        logits = logits.masked_fill(~valid, -1e4)
        weights = torch.softmax(logits, dim=-1) * valid.to(values.dtype)
        weights = weights / weights.sum(dim=-1, keepdim=True).clamp_min(1e-12)
        pooled = torch.sum(values * weights.unsqueeze(-1), dim=1)
        return pooled, weights


class XICFusionNetwork(nn.Module):
    """Encode precursor XICs and an unordered set of paired fragment XICs.

    The output is one correct-identification logit. Applying sigmoid therefore
    yields the repository's native ``trust_score=P(correct identification)``.
    """

    def __init__(
        self,
        *,
        precursor_channels: int = 20,
        fragment_channels: int = 10,
        trace_hidden_dim: int = 32,
        embedding_dim: int = 8,
        fragment_hidden_dim: int = 64,
        attention_dim: int = 32,
        fusion_hidden_dims=(128, 64),
        max_fragment_charge: int = 8,
        dropout: float = 0.15,
        include_predicted_intensity: bool = False,
    ):
        super().__init__()
        if min(
                precursor_channels, fragment_channels, trace_hidden_dim,
                embedding_dim, fragment_hidden_dim, attention_dim,
                max_fragment_charge) < 1:
            raise ValueError("XIC architecture dimensions must be positive")
        if not 0 <= float(dropout) < 1:
            raise ValueError("dropout must be in [0, 1)")
        fusion_dims = [int(value) for value in fusion_hidden_dims]
        if not fusion_dims or min(fusion_dims) < 1:
            raise ValueError("fusion_hidden_dims must be nonempty and positive")

        self.precursor_encoder = TraceEncoder(
            precursor_channels, trace_hidden_dim, float(dropout))
        self.fragment_encoder = TraceEncoder(
            fragment_channels, trace_hidden_dim, float(dropout))
        self.ion_embedding = nn.Embedding(3, embedding_dim, padding_idx=0)
        self.charge_embedding = nn.Embedding(
            max_fragment_charge + 1, embedding_dim, padding_idx=0)
        prediction_dim = 2 if include_predicted_intensity else 0
        fragment_input = (
            self.fragment_encoder.output_dim + 2 * embedding_dim
            + prediction_dim)
        self.fragment_projection = nn.Sequential(
            nn.Linear(fragment_input, fragment_hidden_dim),
            nn.LayerNorm(fragment_hidden_dim),
            nn.GELU(),
            nn.Dropout(float(dropout)),
        )
        self.fragment_pool = MaskedAttentionPool(
            fragment_hidden_dim, attention_dim)

        fusion_input = (
            self.precursor_encoder.output_dim + fragment_hidden_dim + 2)
        layers = []
        previous = fusion_input
        for width in fusion_dims:
            layers.extend([
                nn.Linear(previous, width),
                nn.LayerNorm(width),
                nn.GELU(),
                nn.Dropout(float(dropout)),
            ])
            previous = width
        layers.append(nn.Linear(previous, 1))
        self.fusion = nn.Sequential(*layers)
        self.precursor_channels = int(precursor_channels)
        self.fragment_channels = int(fragment_channels)
        self.trace_hidden_dim = int(trace_hidden_dim)
        self.embedding_dim = int(embedding_dim)
        self.fragment_hidden_dim = int(fragment_hidden_dim)
        self.attention_dim = int(attention_dim)
        self.fusion_hidden_dims = tuple(fusion_dims)
        self.max_fragment_charge = int(max_fragment_charge)
        self.dropout = float(dropout)
        self.include_predicted_intensity = bool(include_predicted_intensity)

    def _validate_indices(self, batch: dict) -> None:
        if torch.any(batch["fragment_ion_type"] > 2):
            raise ValueError("fragment ion type is outside the embedding schema")
        if torch.any(batch["fragment_charge"] > self.max_fragment_charge):
            raise ValueError("fragment charge exceeds configured maximum")

    def forward(self, batch: dict, *, return_attention: bool = False):
        self._validate_indices(batch)
        precursor = self.precursor_encoder(batch["precursor"])
        fragments = batch["fragment"]
        batch_size, n_fragments, channels, trace = fragments.shape
        encoded = self.fragment_encoder(
            fragments.reshape(batch_size * n_fragments, channels, trace))
        encoded = encoded.reshape(batch_size, n_fragments, -1)
        parts = [
            encoded,
            self.ion_embedding(batch["fragment_ion_type"]),
            self.charge_embedding(batch["fragment_charge"]),
        ]
        if self.include_predicted_intensity:
            if "fragment_prediction" not in batch:
                raise ValueError(
                    "prediction-enabled XIC model requires prediction inputs")
            parts.append(batch["fragment_prediction"])
        fragment_values = self.fragment_projection(torch.cat(parts, dim=-1))
        pooled, attention = self.fragment_pool(
            fragment_values, batch["fragment_mask"])
        logits = self.fusion(torch.cat([
            precursor, pooled, batch["signal_scale"],
        ], dim=-1)).squeeze(-1)
        if return_attention:
            return logits, attention
        return logits

    def architecture(self) -> dict:
        return {
            "type": "xic_fusion_attention_v2",
            "precursor_channels": self.precursor_channels,
            "fragment_channels": self.fragment_channels,
            "trace_hidden_dim": self.trace_hidden_dim,
            "embedding_dim": self.embedding_dim,
            "fragment_hidden_dim": self.fragment_hidden_dim,
            "attention_dim": self.attention_dim,
            "fusion_hidden_dims": list(self.fusion_hidden_dims),
            "max_fragment_charge": self.max_fragment_charge,
            "dropout": self.dropout,
            "include_predicted_intensity": self.include_predicted_intensity,
        }


def model_from_architecture(architecture: dict) -> XICFusionNetwork:
    if architecture.get("type") != "xic_fusion_attention_v2":
        raise ValueError(
            f"unsupported XIC architecture: {architecture.get('type')!r}")
    kwargs = dict(architecture)
    kwargs.pop("type")
    return XICFusionNetwork(**kwargs)


def n_trainable_parameters(model: nn.Module) -> int:
    return int(sum(
        parameter.numel() for parameter in model.parameters()
        if parameter.requires_grad))
