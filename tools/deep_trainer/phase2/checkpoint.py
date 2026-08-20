"""Portable, dataset-bound checkpoints for Phase 2 XIC models."""

from __future__ import annotations

import os
from pathlib import Path

import torch

from .data import XICDataset, input_adapter_contract
from .model import model_from_architecture, model_implementation_sha256
from .training import predict_trust


CHECKPOINT_SCHEMA = "metabolic_label_xic_fusion_checkpoint_v3"
_TRUSTED_LEGACY_IMPLEMENTATIONS = {
    "xic_fusion_attention_v2": frozenset({
        # Published CPU baseline at git 2b6fd63.  The v2 implementation is
        # unchanged; model.py later gained the independent v3 model arm.
        "ce4814ddead969e87da4b3996a19474bb5b85d1a405575d441f649b7bb197aca",
    }),
}


def model_implementation_contract(model_type: str) -> dict:
    """Bind a checkpoint only to code that interprets its architecture."""
    return {
        "schema": "phase2_xic_model_implementation_v2",
        "model_type": str(model_type),
        "sha256": model_implementation_sha256(str(model_type)),
    }


def save_checkpoint(path, fitted, *, dataset_identity: dict, metadata: dict):
    architecture = fitted.model.architecture()
    payload = {
        "schema": CHECKPOINT_SCHEMA,
        "architecture": architecture,
        "model_state_dict": fitted.model.state_dict(),
        "dataset_identity": dict(dataset_identity),
        "input_adapter": input_adapter_contract(
            include_predicted_intensity=(
                fitted.model.include_predicted_intensity)),
        "model_implementation": model_implementation_contract(
            architecture["type"]),
        "training_history": list(fitted.history),
        "best_epoch": int(fitted.best_epoch),
        "best_validation_score": float(fitted.best_validation_score),
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
    expected_adapter = input_adapter_contract(
        include_predicted_intensity=bool(
            payload["architecture"].get(
                "include_predicted_intensity", False)))
    if payload.get("input_adapter") != expected_adapter:
        raise ValueError(
            "checkpoint uses a different Phase 2 input adapter contract")
    observed_implementation = payload.get("model_implementation")
    model_type = payload.get("architecture", {}).get("type")
    current_implementation = model_implementation_contract(model_type)
    trusted_legacy = (
        isinstance(observed_implementation, dict)
        and observed_implementation.get("schema")
        == "phase2_xic_model_implementation_v1"
        and observed_implementation.get("sha256")
        in _TRUSTED_LEGACY_IMPLEMENTATIONS.get(model_type, frozenset())
    )
    if observed_implementation != current_implementation \
            and not trusted_legacy:
        raise ValueError(
            "checkpoint uses a different Phase 2 model implementation")
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
