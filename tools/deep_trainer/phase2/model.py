"""Signal-native neural model for paired precursor and fragment XICs."""

from __future__ import annotations

import hashlib
import inspect

import torch
from torch import nn


_TRACE_FEATURES_PER_CHANNEL = 5
_SCAN_MASK_FEATURE_INDEX = 3


class XICModel(nn.Module):
    """Common training/checkpoint interface for all Phase 2 XIC models."""

    include_predicted_intensity: bool

    def architecture(self) -> dict:
        raise NotImplementedError


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


class XICFusionNetwork(XICModel):
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
        if batch.get("_embedding_indices_validated", False):
            return
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


def _masked_mean_max(
    values: torch.Tensor,
    mask: torch.Tensor,
    *,
    dim: int,
) -> tuple[torch.Tensor, torch.Tensor]:
    """Reduce one dimension with a broadcastable mask and finite empty set."""
    valid = torch.broadcast_to(mask.to(dtype=torch.bool), values.shape)
    weight = valid.to(values.dtype)
    count = weight.sum(dim=dim).clamp_min(1.0)
    mean = (values * weight).sum(dim=dim) / count
    maximum = values.masked_fill(
        ~valid, torch.finfo(values.dtype).min).amax(dim=dim)
    maximum = torch.where(
        valid.any(dim=dim), maximum, torch.zeros_like(maximum))
    return mean, maximum


class ResidualTraceBlock(nn.Module):
    """Length-preserving residual temporal block for one XIC channel."""

    def __init__(self, hidden_dim: int, *, dilation: int, dropout: float):
        super().__init__()
        self.network = nn.Sequential(
            nn.Conv1d(
                hidden_dim, hidden_dim, kernel_size=3,
                padding=int(dilation), dilation=int(dilation)),
            nn.GELU(),
            nn.Dropout(float(dropout)),
            nn.Conv1d(hidden_dim, hidden_dim, kernel_size=1),
            nn.Dropout(float(dropout)),
        )
        self.normalization = nn.GroupNorm(1, hidden_dim)

    def forward(self, values: torch.Tensor) -> torch.Tensor:
        return self.normalization(values + self.network(values))


class ResidualTraceEncoder(nn.Module):
    """Encode one five-feature XIC while respecting its real-scan mask."""

    def __init__(self, hidden_dim: int, *, n_blocks: int, dropout: float):
        super().__init__()
        if hidden_dim < 1 or n_blocks < 1:
            raise ValueError("trace hidden_dim and n_blocks must be positive")
        self.input_projection = nn.Conv1d(
            _TRACE_FEATURES_PER_CHANNEL, hidden_dim, kernel_size=1)
        self.blocks = nn.ModuleList([
            ResidualTraceBlock(
                hidden_dim, dilation=2 ** (index % 3), dropout=dropout)
            for index in range(n_blocks)
        ])
        # Keep raw normalized-intensity mean/max alongside the learned trace
        # representation.  This prevents hidden-state normalization from
        # becoming the only route for channel-level abundance evidence.
        self.output_dim = 2 * int(hidden_dim) + 2

    def forward(self, values: torch.Tensor,
                scan_mask: torch.Tensor) -> torch.Tensor:
        raw_mean, raw_maximum = _masked_mean_max(
            values[:, 0, :], scan_mask, dim=-1)
        encoded = self.input_projection(values)
        for block in self.blocks:
            encoded = block(encoded)
        mean, maximum = _masked_mean_max(
            encoded, scan_mask.unsqueeze(1), dim=-1)
        return torch.cat([
            mean, maximum,
            raw_mean.unsqueeze(-1), raw_maximum.unsqueeze(-1),
        ], dim=-1)


class MaskedSetContextBlock(nn.Module):
    """Exchange global context between fragments in linear set size."""

    def __init__(self, hidden_dim: int, dropout: float):
        super().__init__()
        self.update = nn.Sequential(
            nn.Linear(3 * hidden_dim, 2 * hidden_dim),
            nn.GELU(),
            nn.Dropout(float(dropout)),
            nn.Linear(2 * hidden_dim, hidden_dim),
        )
        self.normalization = nn.LayerNorm(hidden_dim)
        self.dropout = nn.Dropout(float(dropout))

    def forward(self, values: torch.Tensor,
                mask: torch.Tensor) -> torch.Tensor:
        mean, maximum = _masked_mean_max(
            values, mask.unsqueeze(-1), dim=1)
        context = torch.cat([mean, maximum], dim=-1).unsqueeze(1)
        context = context.expand(-1, values.shape[1], -1)
        updated = self.normalization(
            values + self.dropout(self.update(torch.cat([
                values, context,
            ], dim=-1))))
        return updated * mask.unsqueeze(-1).to(updated.dtype)


class MultiHeadMaskedAttentionPool(nn.Module):
    """Pool several complementary fragment evidence patterns."""

    def __init__(self, input_dim: int, attention_dim: int, n_heads: int):
        super().__init__()
        if min(input_dim, attention_dim, n_heads) < 1:
            raise ValueError("attention dimensions and heads must be positive")
        self.score = nn.Sequential(
            nn.Linear(input_dim, attention_dim),
            nn.Tanh(),
            nn.Linear(attention_dim, n_heads),
        )
        self.n_heads = int(n_heads)
        self.output_dim = int(input_dim) * self.n_heads

    def forward(self, values: torch.Tensor,
                mask: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
        valid = mask.to(dtype=torch.bool)
        logits = self.score(values).transpose(1, 2)
        logits = logits.masked_fill(~valid.unsqueeze(1), -1e4)
        weights = torch.softmax(logits, dim=-1) \
            * valid.unsqueeze(1).to(values.dtype)
        weights = weights / weights.sum(
            dim=-1, keepdim=True).clamp_min(1e-12)
        pooled = torch.einsum("bhf,bfd->bhd", weights, values)
        return pooled.flatten(start_dim=1), weights


class XICPairInteractionNetwork(XICModel):
    """Signal-only XIC model with explicit light/heavy and set interactions.

    The model consumes the same adapter contract as v2.  It does not add
    fragment ordinal, fragment count, peptide sequence, dataset, or any other
    protocol metadata.  Strength comes from grouping physical XIC channels,
    residual temporal encoding, explicit pair products/differences, and
    linear-cost context exchange across the eligible fragment set.
    """

    def __init__(
        self,
        *,
        precursor_channels: int = 20,
        fragment_channels: int = 10,
        trace_hidden_dim: int = 32,
        trace_blocks: int = 3,
        precursor_hidden_dim: int = 64,
        embedding_dim: int = 8,
        fragment_hidden_dim: int = 64,
        set_blocks: int = 1,
        attention_dim: int = 32,
        attention_heads: int = 2,
        fusion_hidden_dims=(128, 64),
        max_fragment_charge: int = 8,
        dropout: float = 0.15,
        include_predicted_intensity: bool = False,
    ):
        super().__init__()
        if precursor_channels != 4 * _TRACE_FEATURES_PER_CHANNEL:
            raise ValueError(
                "v3 requires four five-feature precursor XIC channels")
        if fragment_channels != 2 * _TRACE_FEATURES_PER_CHANNEL:
            raise ValueError(
                "v3 requires paired five-feature fragment XIC channels")
        if min(
                trace_hidden_dim, trace_blocks, precursor_hidden_dim,
                embedding_dim, fragment_hidden_dim, attention_dim,
                attention_heads, max_fragment_charge) < 1 or set_blocks < 0:
            raise ValueError(
                "v3 dimensions must be positive and set_blocks non-negative")
        if not 0 <= float(dropout) < 1:
            raise ValueError("dropout must be in [0, 1)")
        fusion_dims = [int(value) for value in fusion_hidden_dims]
        if not fusion_dims or min(fusion_dims) < 1:
            raise ValueError("fusion_hidden_dims must be nonempty and positive")

        self.precursor_trace_encoder = ResidualTraceEncoder(
            trace_hidden_dim, n_blocks=trace_blocks, dropout=dropout)
        self.fragment_trace_encoder = ResidualTraceEncoder(
            trace_hidden_dim, n_blocks=trace_blocks, dropout=dropout)
        trace_dim = self.precursor_trace_encoder.output_dim
        # Four raw channel embeddings plus all eight anchored/adjacent isotope
        # pair interactions: light-heavy M0, heavy M0-M1/M2, and M1-M2.
        self.precursor_projection = nn.Sequential(
            nn.Linear(12 * trace_dim, precursor_hidden_dim),
            nn.LayerNorm(precursor_hidden_dim),
            nn.GELU(),
            nn.Dropout(float(dropout)),
        )
        self.ion_embedding = nn.Embedding(3, embedding_dim, padding_idx=0)
        self.charge_embedding = nn.Embedding(
            max_fragment_charge + 1, embedding_dim, padding_idx=0)
        prediction_dim = 2 if include_predicted_intensity else 0
        fragment_input = 4 * trace_dim + 2 * embedding_dim + prediction_dim
        self.fragment_projection = nn.Sequential(
            nn.Linear(fragment_input, fragment_hidden_dim),
            nn.LayerNorm(fragment_hidden_dim),
            nn.GELU(),
            nn.Dropout(float(dropout)),
        )
        self.set_context = nn.ModuleList([
            MaskedSetContextBlock(fragment_hidden_dim, dropout)
            for _ in range(set_blocks)
        ])
        self.fragment_pool = MultiHeadMaskedAttentionPool(
            fragment_hidden_dim, attention_dim, attention_heads)

        fusion_input = (
            precursor_hidden_dim + self.fragment_pool.output_dim
            + 2 * fragment_hidden_dim + 2)
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
        self.trace_blocks = int(trace_blocks)
        self.precursor_hidden_dim = int(precursor_hidden_dim)
        self.embedding_dim = int(embedding_dim)
        self.fragment_hidden_dim = int(fragment_hidden_dim)
        self.set_blocks = int(set_blocks)
        self.attention_dim = int(attention_dim)
        self.attention_heads = int(attention_heads)
        self.fusion_hidden_dims = tuple(fusion_dims)
        self.max_fragment_charge = int(max_fragment_charge)
        self.dropout = float(dropout)
        self.include_predicted_intensity = bool(
            include_predicted_intensity)

    @staticmethod
    def _pair_features(left: torch.Tensor,
                       right: torch.Tensor) -> torch.Tensor:
        return torch.cat([
            left, right, torch.abs(left - right), left * right,
        ], dim=-1)

    @staticmethod
    def _encode_groups(values: torch.Tensor, *, n_groups: int,
                       encoder: ResidualTraceEncoder) -> torch.Tensor:
        batch_size, channels, trace_length = values.shape
        expected = n_groups * _TRACE_FEATURES_PER_CHANNEL
        if channels != expected:
            raise ValueError(
                f"XIC tensor has {channels} channels; expected {expected}")
        grouped = values.reshape(
            batch_size, n_groups, _TRACE_FEATURES_PER_CHANNEL, trace_length)
        scan_mask = grouped[:, :, _SCAN_MASK_FEATURE_INDEX, :] > 0.5
        encoded = encoder(
            grouped.reshape(
                batch_size * n_groups, _TRACE_FEATURES_PER_CHANNEL,
                trace_length),
            scan_mask.reshape(batch_size * n_groups, trace_length),
        )
        return encoded.reshape(batch_size, n_groups, -1)

    def _validate_indices(self, batch: dict) -> None:
        if batch.get("_embedding_indices_validated", False):
            return
        if torch.any(batch["fragment_ion_type"] > 2):
            raise ValueError("fragment ion type is outside the embedding schema")
        if torch.any(batch["fragment_charge"] > self.max_fragment_charge):
            raise ValueError("fragment charge exceeds configured maximum")

    def _encode_eligible_fragments(
        self,
        fragments: torch.Tensor,
        packed: torch.Tensor,
        packed_index: torch.Tensor,
    ) -> torch.Tensor:
        """Encode real fragment pairs without computing padded/masked rows."""
        batch_size, n_fragments, _channels, _trace_length = fragments.shape
        output_shape = (
            batch_size, n_fragments, 2,
            self.fragment_trace_encoder.output_dim,
        )
        if packed.shape[0] == 0:
            return fragments.new_zeros(output_shape)
        eligible = self._encode_groups(
            packed,
            n_groups=2,
            encoder=self.fragment_trace_encoder,
        )
        # ``index_put`` retains the gradient route from every eligible trace
        # while leaving padding and audit-only fragments as exact zeros.  All
        # downstream fragment modules already apply the same mask, so this is
        # numerically equivalent to encoding every padded row and discarding
        # the result later.
        encoded = eligible.new_zeros(output_shape)
        return encoded.index_put(
            (packed_index[:, 0], packed_index[:, 1]), eligible)

    def forward(self, batch: dict, *, return_attention: bool = False,
                return_attention_heads: bool = False):
        self._validate_indices(batch)
        precursor_groups = self._encode_groups(
            batch["precursor"], n_groups=4,
            encoder=self.precursor_trace_encoder)
        light, heavy_m0, heavy_m1, heavy_m2 = precursor_groups.unbind(dim=1)
        precursor = self.precursor_projection(torch.cat([
            precursor_groups.flatten(start_dim=1),
            torch.abs(light - heavy_m0), light * heavy_m0,
            torch.abs(heavy_m0 - heavy_m1), heavy_m0 * heavy_m1,
            torch.abs(heavy_m0 - heavy_m2), heavy_m0 * heavy_m2,
            torch.abs(heavy_m1 - heavy_m2), heavy_m1 * heavy_m2,
        ], dim=-1))

        fragments = batch["fragment"]
        mask = batch["fragment_mask"].to(dtype=torch.bool)
        fragment_groups = self._encode_eligible_fragments(
            fragments,
            batch["fragment_packed"],
            batch["fragment_packed_index"],
        )
        fragment_parts = [
            self._pair_features(
                fragment_groups[:, :, 0], fragment_groups[:, :, 1]),
            self.ion_embedding(batch["fragment_ion_type"]),
            self.charge_embedding(batch["fragment_charge"]),
        ]
        if self.include_predicted_intensity:
            if "fragment_prediction" not in batch:
                raise ValueError(
                    "prediction-enabled XIC model requires prediction inputs")
            fragment_parts.append(batch["fragment_prediction"])
        fragment_values = self.fragment_projection(
            torch.cat(fragment_parts, dim=-1))
        fragment_values = fragment_values * mask.unsqueeze(-1).to(
            fragment_values.dtype)
        for block in self.set_context:
            fragment_values = block(fragment_values, mask)
        attention_pool, attention_heads = self.fragment_pool(
            fragment_values, mask)
        fragment_mean, fragment_max = _masked_mean_max(
            fragment_values, mask.unsqueeze(-1), dim=1)
        logits = self.fusion(torch.cat([
            precursor, attention_pool, fragment_mean, fragment_max,
            batch["signal_scale"],
        ], dim=-1)).squeeze(-1)
        if return_attention_heads:
            return logits, attention_heads
        if return_attention:
            # Preserve the v2 explainability interface by default: one
            # normalized importance per fragment.  Full heads are available
            # explicitly for collapse diagnostics.
            return logits, attention_heads.mean(dim=1)
        return logits

    def architecture(self) -> dict:
        return {
            "type": "xic_pair_interaction_v3",
            "precursor_channels": self.precursor_channels,
            "fragment_channels": self.fragment_channels,
            "trace_hidden_dim": self.trace_hidden_dim,
            "trace_blocks": self.trace_blocks,
            "precursor_hidden_dim": self.precursor_hidden_dim,
            "embedding_dim": self.embedding_dim,
            "fragment_hidden_dim": self.fragment_hidden_dim,
            "set_blocks": self.set_blocks,
            "attention_dim": self.attention_dim,
            "attention_heads": self.attention_heads,
            "fusion_hidden_dims": list(self.fusion_hidden_dims),
            "max_fragment_charge": self.max_fragment_charge,
            "dropout": self.dropout,
            "include_predicted_intensity": self.include_predicted_intensity,
        }


_MODEL_REGISTRY = {
    "xic_fusion_attention_v2": (
        XICFusionNetwork,
        (XICModel, TraceEncoder, MaskedAttentionPool, XICFusionNetwork),
    ),
    "xic_pair_interaction_v3": (
        XICPairInteractionNetwork,
        (
            XICModel, _masked_mean_max, ResidualTraceBlock,
            ResidualTraceEncoder, MaskedSetContextBlock,
            MultiHeadMaskedAttentionPool, XICPairInteractionNetwork,
        ),
    ),
}
SUPPORTED_XIC_MODEL_TYPES = frozenset(_MODEL_REGISTRY)


def model_from_architecture(architecture: dict) -> XICModel:
    model_type = architecture.get("type")
    if model_type not in _MODEL_REGISTRY:
        raise ValueError(
            f"unsupported XIC architecture: {model_type!r}")
    kwargs = dict(architecture)
    kwargs.pop("type")
    factory, _implementation_parts = _MODEL_REGISTRY[model_type]
    return factory(**kwargs)


def model_implementation_sha256(model_type: str) -> str:
    """Hash only the code that interprets one architecture's weights."""
    if model_type not in _MODEL_REGISTRY:
        raise ValueError(f"unsupported XIC architecture: {model_type!r}")
    _factory, implementation_parts = _MODEL_REGISTRY[model_type]
    digest = hashlib.sha256()
    digest.update(
        f"trace_features={_TRACE_FEATURES_PER_CHANNEL};"
        f"scan_mask_index={_SCAN_MASK_FEATURE_INDEX}\n".encode())
    for part in implementation_parts:
        digest.update(inspect.getsource(part).encode("utf-8"))
        digest.update(b"\n")
    return digest.hexdigest()


def n_trainable_parameters(model: nn.Module) -> int:
    return int(sum(
        parameter.numel() for parameter in model.parameters()
        if parameter.requires_grad))
