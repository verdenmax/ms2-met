"""Portable, fail-closed tabular MLP checkpoints."""

from __future__ import annotations

import os

import torch

from .model import TabularMLP
from .preprocessing import FoldPreprocessor


CHECKPOINT_SCHEMA = "metabolic_label_tabular_mlp_checkpoint_v1"


def save_checkpoint(path, fitted, preprocessor, feature_names, metadata):
    state = preprocessor.to_state()
    for key in ("medians", "means", "scales"):
        state[key] = state[key].tolist()
    payload = {
        "schema": CHECKPOINT_SCHEMA,
        "architecture": fitted.model.architecture(),
        "model_state_dict": fitted.model.state_dict(),
        "preprocessor": state,
        "feature_names": list(feature_names),
        "metadata": dict(metadata),
    }
    path = os.fspath(path)
    os.makedirs(os.path.dirname(path) or ".", exist_ok=True)
    temporary = f"{path}.tmp.{os.getpid()}"
    torch.save(payload, temporary)
    os.replace(temporary, path)


def load_checkpoint(path, *, device="cpu"):
    payload = torch.load(path, map_location=device, weights_only=True)
    if payload.get("schema") != CHECKPOINT_SCHEMA:
        raise ValueError(
            f"unsupported checkpoint schema: {payload.get('schema')!r}")
    architecture = payload["architecture"]
    model = TabularMLP(
        architecture["input_dim"],
        hidden_dims=architecture["hidden_dims"],
        dropout=architecture["dropout"],
    )
    model.load_state_dict(payload["model_state_dict"])
    model.to(device)
    model.eval()
    preprocessor = FoldPreprocessor.from_state(payload["preprocessor"])
    return model, preprocessor, payload
