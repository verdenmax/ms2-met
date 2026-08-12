"""Portable, fail-closed tabular MLP checkpoints."""

from __future__ import annotations

from dataclasses import dataclass
import os

import pandas as pd
import torch

from .model import TabularMLP
from .preprocessing import FoldPreprocessor
from .training import predict_trust


CHECKPOINT_SCHEMA = "metabolic_label_tabular_mlp_checkpoint_v1"


@dataclass
class TabularMLPPredictor:
    """Schema-safe inference interface for one fold checkpoint."""

    model: TabularMLP
    preprocessor: FoldPreprocessor
    feature_names: tuple[str, ...]
    metadata: dict
    device: str = "cpu"

    def predict_frame(self, frame: pd.DataFrame, *, batch_size=4096):
        if not isinstance(frame, pd.DataFrame):
            raise TypeError("predict_frame requires a pandas DataFrame")
        duplicates = frame.columns[frame.columns.duplicated()].tolist()
        if duplicates:
            raise ValueError(f"inference frame has duplicate columns: {duplicates}")
        missing = [name for name in self.feature_names if name not in frame]
        if missing:
            raise ValueError(
                f"inference frame is missing checkpoint features: {missing}")
        raw = frame.loc[:, list(self.feature_names)].to_numpy(dtype="f8")
        values = self.preprocessor.transform(raw)
        return predict_trust(
            self.model, values, batch_size=batch_size, device=self.device)


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


def load_predictor(path, *, device="cpu") -> TabularMLPPredictor:
    model, preprocessor, payload = load_checkpoint(path, device=device)
    feature_names = tuple(payload.get("feature_names", ()))
    if not feature_names or len(set(feature_names)) != len(feature_names):
        raise ValueError("checkpoint feature_names are empty or duplicated")
    if preprocessor.n_source_features != len(feature_names):
        raise ValueError(
            "checkpoint preprocessor/feature schema length mismatch")
    if preprocessor.n_output_features != model.input_dim:
        raise ValueError(
            "checkpoint preprocessor/network input dimension mismatch")
    return TabularMLPPredictor(
        model=model,
        preprocessor=preprocessor,
        feature_names=feature_names,
        metadata=dict(payload.get("metadata", {})),
        device=str(device),
    )
