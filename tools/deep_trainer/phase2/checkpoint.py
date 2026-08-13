"""Portable, dataset-bound checkpoints for Phase 2 XIC models."""

from __future__ import annotations

import os
from pathlib import Path

import torch

from .data import XICDataset
from .model import model_from_architecture
from .training import predict_trust


CHECKPOINT_SCHEMA = "metabolic_label_xic_fusion_checkpoint_v1"


def save_checkpoint(path, fitted, *, dataset_identity: dict, metadata: dict):
    payload = {
        "schema": CHECKPOINT_SCHEMA,
        "architecture": fitted.model.architecture(),
        "model_state_dict": fitted.model.state_dict(),
        "dataset_identity": dict(dataset_identity),
        "metadata": dict(metadata),
    }
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_name(f"{path.name}.tmp.{os.getpid()}")
    torch.save(payload, temporary)
    os.replace(temporary, path)


def load_checkpoint(path, *, source=None, device="cpu"):
    payload = torch.load(path, map_location=device, weights_only=True)
    if payload.get("schema") != CHECKPOINT_SCHEMA:
        raise ValueError(
            f"unsupported Phase 2 checkpoint: {payload.get('schema')!r}")
    model = model_from_architecture(payload["architecture"])
    model.load_state_dict(payload["model_state_dict"])
    model.to(device)
    model.eval()
    if source is not None:
        expected = payload.get("dataset_identity", {}).get(
            "signal_checksums_sha256")
        observed = source.complete.get("checksums_sha256")
        if not expected or expected != observed:
            raise ValueError(
                "checkpoint is bound to a different Phase 2 XIC dataset")
    return model, payload


def predict_indices(path, source, indices, *, device="cpu", batch_size=256,
                    num_workers=0):
    """Schema-safe inference with a checkpoint's declared prediction arm."""
    model, payload = load_checkpoint(path, source=source, device=device)
    include_prediction = bool(
        payload["architecture"].get("include_predicted_intensity", False))
    dataset = XICDataset(
        source, indices,
        include_predicted_intensity=include_prediction)
    return predict_trust(
        model, dataset, batch_size=int(batch_size), device=device,
        num_workers=int(num_workers))
