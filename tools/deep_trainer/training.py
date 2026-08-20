"""PyTorch fitting and inference hidden behind a small training interface."""

from __future__ import annotations

from copy import deepcopy
from dataclasses import dataclass
import logging
import os
import random

import numpy as np
import torch
from torch import nn

from .model import TabularMLP


@dataclass
class FittedMLP:
    model: TabularMLP
    best_epoch: int
    best_validation_score: float
    history: list[dict]
    device: str


def configure_torch(seed: int, *, num_threads: int = 0,
                    deterministic: bool = True) -> None:
    """Configure reproducibility once per fold."""
    # PyTorch raises on CUDA mm/mv/bmm in strict deterministic mode unless
    # cuBLAS receives one of its reproducible workspace configurations.  Set
    # this before any CUDA availability/seed call can initialize cuBLAS.
    if deterministic:
        os.environ.setdefault("CUBLAS_WORKSPACE_CONFIG", ":4096:8")
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)
    if num_threads > 0:
        torch.set_num_threads(num_threads)
    torch.use_deterministic_algorithms(bool(deterministic))


def resolve_device(requested: str = "auto") -> torch.device:
    value = str(requested).strip().lower()
    if value == "auto":
        value = "cuda" if torch.cuda.is_available() else "cpu"
    if value == "cuda" and not torch.cuda.is_available():
        raise ValueError("CUDA was requested but is not available")
    if value not in {"cpu", "cuda"}:
        raise ValueError("device must be one of auto/cpu/cuda")
    return torch.device(value)


def fit_mlp(
    train_x: np.ndarray,
    train_y: np.ndarray,
    valid_x: np.ndarray,
    valid_y: np.ndarray,
    config: dict,
    *,
    validation_score,
    seed: int,
) -> FittedMLP:
    """Fit one fold and early-stop on a caller-supplied validation score.

    ``validation_score(labels, trust_scores)`` is injected by the experiment
    adapter so canonical error-positive evaluation remains centralized in
    ``spec_trainer/cv_core.py``.
    """
    training = config["training"]
    model_cfg = config["model"]
    configure_torch(
        seed,
        num_threads=int(training.get("torch_num_threads", 0)),
        deterministic=bool(training.get("deterministic", True)),
    )
    device = resolve_device(training.get("device", "auto"))
    model = TabularMLP(
        train_x.shape[1],
        hidden_dims=model_cfg.get("hidden_dims", [128, 64]),
        dropout=float(model_cfg.get("dropout", 0.15)),
    ).to(device)

    learning_rate = float(training.get("learning_rate", 1e-3))
    weight_decay = float(training.get("weight_decay", 1e-4))
    optimizer = torch.optim.AdamW(
        model.parameters(), lr=learning_rate, weight_decay=weight_decay)
    loss_function = nn.BCEWithLogitsLoss(reduction="none")
    class_weights = _class_weights(
        train_y, training.get("class_weighting", "none"))
    epochs = int(training.get("epochs", 60))
    batch_size = int(training.get("batch_size", 1024))
    patience = int(training.get("patience", 8))
    min_delta = float(training.get("min_delta", 1e-4))
    gradient_clip = float(training.get("gradient_clip_norm", 5.0))
    if epochs < 1 or batch_size < 1 or patience < 1:
        raise ValueError("epochs, batch_size, and patience must be positive")

    x_train = torch.from_numpy(np.asarray(train_x, dtype="f4"))
    y_train = torch.from_numpy(np.asarray(train_y, dtype="f4"))
    w_train = torch.from_numpy(class_weights.astype("f4"))
    rng = torch.Generator(device="cpu")
    rng.manual_seed(seed)

    best_score = -float("inf")
    best_epoch = 0
    best_state = None
    no_improvement = 0
    history = []
    for epoch in range(1, epochs + 1):
        model.train()
        order = torch.randperm(len(x_train), generator=rng)
        total_loss = 0.0
        for start in range(0, len(order), batch_size):
            indices = order[start:start + batch_size]
            xb = x_train[indices].to(device)
            yb = y_train[indices].to(device)
            wb = w_train[indices].to(device)
            optimizer.zero_grad(set_to_none=True)
            loss = (loss_function(model(xb), yb) * wb).mean()
            loss.backward()
            if gradient_clip > 0:
                nn.utils.clip_grad_norm_(model.parameters(), gradient_clip)
            optimizer.step()
            total_loss += float(loss.detach().cpu()) * len(indices)

        trust = predict_trust(model, valid_x, batch_size=batch_size,
                              device=device)
        score = float(validation_score(valid_y, trust))
        row = {
            "epoch": epoch,
            "train_loss": total_loss / len(train_x),
            "validation_score": score,
        }
        history.append(row)
        logging.info(
            "epoch=%d train_loss=%.6f validation_roc_auc=%.6f",
            epoch, row["train_loss"], score)
        if score > best_score + min_delta:
            best_score = score
            best_epoch = epoch
            best_state = deepcopy(model.state_dict())
            no_improvement = 0
        else:
            no_improvement += 1
            if no_improvement >= patience:
                break

    if best_state is None:
        raise RuntimeError("training did not produce a finite best checkpoint")
    model.load_state_dict(best_state)
    return FittedMLP(
        model=model,
        best_epoch=best_epoch,
        best_validation_score=best_score,
        history=history,
        device=str(device),
    )


def predict_trust(model, values, *, batch_size=4096, device=None):
    """Return ``P(correct identification)`` in input-row order."""
    resolved = torch.device(device or next(model.parameters()).device)
    array = torch.from_numpy(np.asarray(values, dtype="f4"))
    outputs = []
    model.eval()
    with torch.no_grad():
        for start in range(0, len(array), int(batch_size)):
            logits = model(array[start:start + int(batch_size)].to(resolved))
            outputs.append(torch.sigmoid(logits).cpu().numpy())
    result = np.concatenate(outputs).astype("f8") if outputs else np.array([])
    if not np.isfinite(result).all():
        raise ValueError("model produced non-finite trust scores")
    return result


def _class_weights(labels, mode) -> np.ndarray:
    labels = np.asarray(labels, dtype=int)
    if not set(np.unique(labels).tolist()).issubset({0, 1}):
        raise ValueError("training labels must contain only 0/1")
    normalized = str(mode or "none").strip().lower()
    if normalized == "none":
        return np.ones(len(labels), dtype="f8")
    if normalized != "balanced":
        raise ValueError("class_weighting must be none or balanced")
    counts = np.bincount(labels, minlength=2)
    if (counts == 0).any():
        raise ValueError("balanced loss requires both classes")
    per_class = len(labels) / (2.0 * counts)
    return per_class[labels]
