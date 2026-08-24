"""Fitting and order-preserving inference for the Phase 2 XIC model."""

from __future__ import annotations

from copy import deepcopy
from dataclasses import dataclass
import logging
import random
import time

import numpy as np
import torch
from torch import nn
from torch.utils.data import DataLoader

from ..training import _class_weights, configure_torch, resolve_device
from .data import ShardBatchSampler, XICDataset, collate_xic
from .model import XICModel, model_from_architecture


@dataclass
class FittedXICModel:
    model: XICModel
    best_epoch: int
    best_validation_score: float
    history: list[dict]
    device: str
    device_trace: dict


def _model_from_config(config: dict) -> XICModel:
    """Build a declared XIC model through the checkpoint model seam."""
    return model_from_architecture(dict(config["model"]))


def _move_batch(batch: dict, device: torch.device, *,
                non_blocking: bool = False,
                model: XICModel | None = None) -> dict:
    # Validate embedding indices while tensors are still on the CPU.  Doing
    # the same Python ``if torch.any(cuda_tensor)`` checks inside every model
    # forward forces two device synchronizations per batch.
    if model is not None:
        model._validate_indices(batch)
    moved = {
        key: value.to(device, non_blocking=non_blocking)
        if torch.is_tensor(value) else value
        for key, value in batch.items()
    }
    if model is not None:
        moved["_embedding_indices_validated"] = True
    return moved


def _seed_worker(_worker_id: int) -> None:
    """Seed libraries if a future dataset transform adds randomness."""
    worker_seed = torch.initial_seed() % (2 ** 32)
    random.seed(worker_seed)
    np.random.seed(worker_seed)


def runtime_device_trace(device: torch.device) -> dict:
    """Return and log the accelerator identity used by one fitted member."""
    trace = {
        "device": str(device),
        "device_type": device.type,
        "cuda_available": bool(torch.cuda.is_available()),
        "torch_cuda_runtime": torch.version.cuda,
    }
    if device.type != "cuda":
        logging.warning(
            "Phase2 GPU trace device=%s cuda_available=%s torch_cuda=%s; "
            "training will run without GPU acceleration",
            device, trace["cuda_available"], trace["torch_cuda_runtime"])
        return trace
    device_index = (
        int(device.index) if device.index is not None
        else int(torch.cuda.current_device()))
    properties = torch.cuda.get_device_properties(device_index)
    trace.update({
        "device_index": device_index,
        "device_name": properties.name,
        "compute_capability": [
            int(properties.major), int(properties.minor)],
        "total_memory_bytes": int(properties.total_memory),
    })
    logging.info(
        "Phase2 GPU trace device=cuda:%d name=%s capability=%d.%d "
        "memory_gib=%.2f torch_cuda=%s",
        device_index, properties.name, properties.major, properties.minor,
        properties.total_memory / (1024 ** 3), trace["torch_cuda_runtime"])
    return trace


def _loader(dataset: XICDataset, *, batch_size: int, seed: int,
            shuffle: bool, num_workers: int,
            pin_memory: bool = False) -> DataLoader:
    common = {
        "collate_fn": collate_xic,
        "num_workers": int(num_workers),
        "pin_memory": bool(pin_memory),
        "persistent_workers": int(num_workers) > 0,
        "worker_init_fn": _seed_worker if int(num_workers) > 0 else None,
        "generator": torch.Generator().manual_seed(int(seed)),
    }
    if shuffle:
        sampler = ShardBatchSampler(
            dataset, batch_size, seed=seed, shuffle=True)
        return DataLoader(
            dataset, batch_sampler=sampler, **common)
    return DataLoader(
        dataset, batch_size=batch_size, shuffle=False,
        **common)


def predict_trust(
    model: XICModel,
    dataset: XICDataset,
    *,
    batch_size: int,
    device: str | torch.device,
    num_workers: int = 0,
    mixed_precision: bool = False,
) -> np.ndarray:
    """Return ``P(correct identification)`` in dataset index order."""
    resolved = torch.device(device)
    use_pinned_memory = resolved.type == "cuda"
    loader = _loader(
        dataset, batch_size=int(batch_size), seed=0, shuffle=False,
        num_workers=int(num_workers), pin_memory=use_pinned_memory)
    result = _predict_batches(
        model, loader, device=resolved,
        mixed_precision=bool(mixed_precision))
    if len(result) != len(dataset) or not np.isfinite(result).all():
        raise ValueError("XIC model produced incomplete/non-finite trust scores")
    return result


def _predict_batches(
    model: XICModel,
    loader: DataLoader,
    *,
    device: torch.device,
    mixed_precision: bool,
) -> np.ndarray:
    """Predict one reusable loader with a single final device sync."""
    outputs = []
    use_pinned_memory = device.type == "cuda"
    use_mixed_precision = bool(mixed_precision) and device.type == "cuda"
    model.eval()
    with torch.no_grad():
        for batch in loader:
            moved = _move_batch(
                batch, device, non_blocking=use_pinned_memory, model=model)
            with torch.autocast(
                    device_type=device.type, dtype=torch.float16,
                    enabled=use_mixed_precision):
                logits = model(moved)
            outputs.append(torch.sigmoid(logits.float()))
    return (
        torch.cat(outputs).cpu().numpy().astype("f8")
        if outputs else np.array([])
    )


def attention_head_diagnostics(
    model: XICModel,
    dataset: XICDataset,
    *,
    batch_size: int,
    device: str | torch.device,
    num_workers: int = 0,
    mixed_precision: bool = False,
) -> dict:
    """Measure multi-head collapse without assigning semantics to a head."""
    n_heads = int(getattr(model, "attention_heads", 1))
    if n_heads <= 1:
        return {
            "n_heads": n_heads,
            "n_samples_with_multiple_fragments": 0,
            "mean_pairwise_cosine_similarity": None,
            "fraction_pairwise_cosine_ge_0_99": None,
            "mean_effective_fragments_per_head": None,
        }
    resolved = torch.device(device)
    use_pinned_memory = resolved.type == "cuda"
    loader = _loader(
        dataset, batch_size=int(batch_size), seed=0, shuffle=False,
        num_workers=int(num_workers), pin_memory=use_pinned_memory)
    similarities: list[float] = []
    effective_fragments: list[float] = []
    n_samples = 0
    use_mixed_precision = bool(mixed_precision) and resolved.type == "cuda"
    model.eval()
    with torch.no_grad():
        for batch in loader:
            moved = _move_batch(
                batch, resolved, non_blocking=use_pinned_memory, model=model)
            with torch.autocast(
                    device_type=resolved.type, dtype=torch.float16,
                    enabled=use_mixed_precision):
                _logits, heads = model(
                    moved, return_attention_heads=True)
            heads = heads.float()
            mask = moved["fragment_mask"].to(dtype=torch.bool)
            eligible_rows = mask.sum(dim=1) >= 2
            n_eligible_rows = int(eligible_rows.sum().item())
            n_samples += n_eligible_rows
            if n_eligible_rows == 0:
                continue
            # Attention already stores exact zeros outside ``mask``.  Keeping
            # those zeros lets us evaluate the whole batch at once instead of
            # synchronizing CUDA separately for every OOF sample.
            weights = heads[eligible_rows]
            normalized = weights / torch.linalg.vector_norm(
                weights, dim=-1, keepdim=True).clamp_min(1e-12)
            cosine = torch.einsum(
                "bhf,bkf->bhk", normalized, normalized)
            pair_mask = torch.triu(torch.ones(
                n_heads, n_heads, dtype=torch.bool,
                device=cosine.device), diagonal=1)
            similarities.extend(
                cosine[:, pair_mask].reshape(-1).detach().cpu().tolist())
            entropy = -torch.sum(
                weights * torch.log(weights.clamp_min(1e-12)), dim=-1)
            effective_fragments.extend(
                torch.exp(entropy).reshape(-1).detach().cpu().tolist())
    mean_similarity = (
        float(np.mean(similarities)) if similarities else None)
    return {
        "n_heads": n_heads,
        "n_samples_with_multiple_fragments": n_samples,
        "mean_pairwise_cosine_similarity": mean_similarity,
        "fraction_pairwise_cosine_ge_0_99": (
            float(np.mean(np.asarray(similarities) >= 0.99))
            if similarities else None),
        "mean_effective_fragments_per_head": (
            float(np.mean(effective_fragments))
            if effective_fragments else None),
    }


def fit_xic_model(
    source,
    train_indices,
    valid_indices,
    config: dict,
    *,
    validation_score,
    seed: int,
) -> FittedXICModel:
    """Fit one member, early-stopping via the canonical ranking adapter."""
    training = config["training"]
    include_prediction = bool(
        config["model"].get("include_predicted_intensity", False))
    train = XICDataset(
        source, train_indices,
        include_predicted_intensity=include_prediction)
    valid = XICDataset(
        source, valid_indices,
        include_predicted_intensity=include_prediction)
    if not len(train) or not len(valid):
        raise ValueError("Phase 2 train and validation subsets must be nonempty")
    train_labels = source.manifest.iloc[
        train.indices]["label"].to_numpy(dtype=int)
    if set(np.unique(train_labels)) != {0, 1}:
        raise ValueError("Phase 2 training subset requires both stored classes")

    configure_torch(
        seed,
        num_threads=int(training.get("torch_num_threads", 0)),
        deterministic=bool(training.get("deterministic", True)),
    )
    device = resolve_device(training.get("device", "auto"))
    device_trace = runtime_device_trace(device)
    model = _model_from_config(config).to(device)
    optimizer = torch.optim.AdamW(
        model.parameters(),
        lr=float(training.get("learning_rate", 1e-3)),
        weight_decay=float(training.get("weight_decay", 1e-4)),
    )
    loss_function = nn.BCEWithLogitsLoss(reduction="none")
    class_weights = _class_weights(
        train_labels, training.get("class_weighting", "none"))
    uniform_class_weights = bool(np.all(class_weights == 1.0))
    weights_by_source = {} if uniform_class_weights else {
        int(source_index): float(weight)
        for source_index, weight in zip(train.indices, class_weights)
    }
    epochs = int(training.get("epochs", 60))
    batch_size = int(training.get("batch_size", 64))
    inference_batch_size = int(
        training.get("inference_batch_size", batch_size))
    patience = int(training.get("patience", 8))
    min_delta = float(training.get("min_delta", 1e-4))
    gradient_clip = float(training.get("gradient_clip_norm", 5.0))
    num_workers = int(training.get("num_workers", 0))
    mixed_precision_value = training.get("mixed_precision", False)
    if not isinstance(mixed_precision_value, bool):
        raise ValueError("mixed_precision must be a boolean")
    mixed_precision = mixed_precision_value
    if min(epochs, batch_size, inference_batch_size, patience) < 1:
        raise ValueError(
            "epochs, batch sizes, and patience must be positive")
    if num_workers < 0:
        raise ValueError("num_workers must be non-negative")

    logging.info(
        "Phase2 fit model=%s device=%s train=%d valid=%d batch_size=%d "
        "num_workers=%d mixed_precision=%s",
        model.architecture()["type"], device, len(train), len(valid),
        batch_size, num_workers, mixed_precision and device.type == "cuda")
    loader = _loader(
        train, batch_size=batch_size, seed=seed, shuffle=True,
        num_workers=num_workers, pin_memory=device.type == "cuda")
    valid_loader = _loader(
        valid, batch_size=inference_batch_size, seed=0, shuffle=False,
        num_workers=num_workers, pin_memory=device.type == "cuda")
    best_score = -float("inf")
    best_epoch = 0
    best_state = None
    no_improvement = 0
    history = []
    use_mixed_precision = mixed_precision and device.type == "cuda"
    device_trace["mixed_precision"] = use_mixed_precision
    scaler = torch.amp.GradScaler("cuda", enabled=use_mixed_precision)
    valid_labels = source.manifest.iloc[
        valid.indices]["label"].to_numpy(dtype=int)
    for epoch in range(1, epochs + 1):
        model.train()
        if hasattr(loader.batch_sampler, "set_epoch"):
            loader.batch_sampler.set_epoch(epoch)
        total_loss = torch.zeros((), dtype=torch.float32, device=device)
        total_rows = 0
        data_wait_seconds = 0.0
        train_started = time.perf_counter()
        iterator = iter(loader)
        while True:
            wait_started = time.perf_counter()
            try:
                batch = next(iterator)
            except StopIteration:
                break
            data_wait_seconds += time.perf_counter() - wait_started
            batch_rows = len(batch["label"])
            source_indices = (
                None if uniform_class_weights
                else batch["source_index"].numpy()
            )
            batch = _move_batch(
                batch, device, non_blocking=device.type == "cuda", model=model)
            optimizer.zero_grad(set_to_none=True)
            with torch.autocast(
                    device_type=device.type, dtype=torch.float16,
                    enabled=use_mixed_precision):
                losses = loss_function(model(batch), batch["label"])
                if uniform_class_weights:
                    loss = losses.mean()
                else:
                    weights = torch.tensor([
                        weights_by_source[int(index)]
                        for index in source_indices  # type: ignore[union-attr]
                    ], dtype=torch.float32, device=device)
                    loss = (losses * weights).mean()
            scaler.scale(loss).backward()
            if gradient_clip > 0:
                scaler.unscale_(optimizer)
                nn.utils.clip_grad_norm_(model.parameters(), gradient_clip)
            scaler.step(optimizer)
            scaler.update()
            # Keep the epoch reduction on-device.  Calling ``.cpu()`` here
            # synchronizes CUDA once per batch and leaves the GPU idle between
            # thousands of small temporal-convolution steps.
            total_loss.add_(loss.detach().float() * batch_rows)
            total_rows += batch_rows

        if device.type == "cuda":
            torch.cuda.synchronize(device)
        train_seconds = time.perf_counter() - train_started

        validation_started = time.perf_counter()
        trust = _predict_batches(
            model, valid_loader, device=device,
            mixed_precision=use_mixed_precision)
        if len(trust) != len(valid) or not np.isfinite(trust).all():
            raise ValueError(
                "XIC model produced incomplete/non-finite validation scores")
        validation_seconds = time.perf_counter() - validation_started
        score = float(validation_score(valid_labels, trust))
        row = {
            "epoch": epoch,
            "train_loss": float(total_loss.item()) / total_rows,
            "validation_score": score,
            "train_seconds": train_seconds,
            "data_wait_seconds": data_wait_seconds,
            "validation_seconds": validation_seconds,
            "train_rows_per_second": total_rows / train_seconds,
            "data_wait_fraction": data_wait_seconds / train_seconds,
        }
        history.append(row)
        logging.info(
            "Phase2 epoch=%d train_loss=%.6f validation_roc_auc=%.6f "
            "train_seconds=%.1f data_wait_seconds=%.1f "
            "data_wait_fraction=%.3f validation_seconds=%.1f "
            "train_rows_per_second=%.1f",
            epoch, row["train_loss"], score, train_seconds,
            data_wait_seconds, row["data_wait_fraction"],
            validation_seconds, row["train_rows_per_second"])
        if np.isfinite(score) and score > best_score + min_delta:
            best_score = score
            best_epoch = epoch
            best_state = deepcopy(model.state_dict())
            no_improvement = 0
        else:
            no_improvement += 1
            if no_improvement >= patience:
                break

    if best_state is None:
        raise RuntimeError("Phase 2 training produced no finite checkpoint")
    model.load_state_dict(best_state)
    return FittedXICModel(
        model=model, best_epoch=best_epoch,
        best_validation_score=best_score, history=history,
        device=str(device), device_trace=device_trace)
