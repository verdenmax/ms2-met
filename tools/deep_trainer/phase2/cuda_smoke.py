"""Fail-fast CUDA forward/backward smoke test for a declared XIC model."""

from __future__ import annotations

import argparse
from pathlib import Path

import torch
import yaml

from ..training import configure_torch, resolve_device
from .model import model_from_architecture, n_trainable_parameters


def _synthetic_batch() -> dict:
    batch_size, fragments, trace_length = 2, 4, 13
    precursor = torch.zeros(batch_size, 4, 5, trace_length)
    precursor[:, :, 0] = torch.rand(batch_size, 4, trace_length)
    precursor[:, :, 1] = 2.0 * torch.rand(
        batch_size, 4, trace_length) - 1.0
    precursor[:, :, 2] = torch.linspace(-1.0, 1.0, trace_length)
    precursor[:, :, 3] = 1.0
    precursor[:, :, 4] = (
        precursor[:, :, 0] > 0.25).to(precursor.dtype)

    fragment = torch.zeros(batch_size, fragments, 2, 5, trace_length)
    fragment[:, :, :, 0] = torch.rand(
        batch_size, fragments, 2, trace_length)
    fragment[:, :, :, 1] = 2.0 * torch.rand(
        batch_size, fragments, 2, trace_length) - 1.0
    fragment[:, :, :, 2] = torch.linspace(-1.0, 1.0, trace_length)
    fragment[:, :, :, 3] = 1.0
    fragment[:, :, :, 4] = (
        fragment[:, :, :, 0] > 0.25).to(fragment.dtype)
    mask = torch.tensor([
        [True, True, True, False],
        [True, False, False, False],
    ])
    flattened_fragment = fragment.reshape(
        batch_size, fragments, 10, trace_length)
    return {
        "precursor": precursor.reshape(batch_size, 20, trace_length),
        "fragment": flattened_fragment,
        "fragment_packed": flattened_fragment[mask],
        "fragment_packed_index": mask.nonzero(as_tuple=False),
        "fragment_mask": mask,
        "fragment_ion_type": torch.tensor([
            [1, 2, 1, 0], [2, 0, 0, 0],
        ]),
        "fragment_charge": torch.tensor([
            [1, 2, 1, 0], [2, 0, 0, 0],
        ]),
        "signal_scale": torch.tensor([[8.0, 0.8], [9.0, 0.9]]),
        "label": torch.tensor([1.0, 0.0]),
    }


def run_cuda_smoke(config_path: str | Path) -> dict:
    """Run one strict-deterministic optimizer step on the actual CUDA model."""
    configure_torch(20260820, deterministic=True)
    device = resolve_device("cuda")
    config = yaml.safe_load(Path(config_path).read_text(encoding="utf-8"))
    model = model_from_architecture(dict(config["model"])).to(device).train()
    mixed_precision = bool(
        config.get("training", {}).get("mixed_precision", False))
    batch = {
        key: value.to(device) if torch.is_tensor(value) else value
        for key, value in _synthetic_batch().items()
    }
    optimizer = torch.optim.AdamW(model.parameters(), lr=1e-3)
    scaler = torch.amp.GradScaler("cuda", enabled=mixed_precision)
    optimizer.zero_grad(set_to_none=True)
    with torch.autocast(
            device_type="cuda", dtype=torch.float16,
            enabled=mixed_precision):
        logits = model(batch)
        loss = torch.nn.functional.binary_cross_entropy_with_logits(
            logits, batch["label"])
    scaler.scale(loss).backward()
    scaler.step(optimizer)
    scaler.update()
    if not torch.isfinite(loss) or not torch.isfinite(logits).all():
        raise RuntimeError("CUDA XIC smoke test produced non-finite values")
    return {
        "model_type": model.architecture()["type"],
        "device": str(device),
        "parameters": n_trainable_parameters(model),
        "mixed_precision": mixed_precision,
        "loss": float(loss.detach().cpu()),
    }


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", required=True)
    args = parser.parse_args()
    report = run_cuda_smoke(args.config)
    print(
        "CUDA XIC smoke passed: "
        f"model={report['model_type']} device={report['device']} "
        f"parameters={report['parameters']} "
        f"mixed_precision={report['mixed_precision']} "
        f"loss={report['loss']:.6f}")


if __name__ == "__main__":
    main()
