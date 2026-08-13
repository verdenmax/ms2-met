"""Torch input adapter for immutable Phase 2 XIC shards.

The adapter deliberately exposes only signal-derived tensors allowed by the
Phase 2 schema.  Frozen split/fold metadata stays available to the experiment
controller, but never becomes a model input.
"""

from __future__ import annotations

from collections import defaultdict
from collections.abc import Iterable, Iterator, Sequence
import math
from pathlib import Path
import random

import numpy as np
import torch
from torch.utils.data import Dataset, Sampler

from artifact_identity import sha256_file
from .store import SignalDataset


_N_SIGNAL_FEATURES = 5
XIC_INPUT_ADAPTER_SCHEMA = "phase2_xic_input_adapter_v2"


def input_adapter_contract(*, include_predicted_intensity: bool) -> dict:
    """Return the exact model-bound normalization and routing contract."""
    return {
        "schema": XIC_INPUT_ADAPTER_SCHEMA,
        "implementation_sha256": sha256_file(Path(__file__).resolve()),
        "trace_features": [
            "log1p_intensity_div_log1p_sample_max",
            "ppm_error_div_mass_tol_ppm_peak_masked",
            "rt_delta_div_sample_abs_max_scan_masked",
            "scan_mask", "peak_mask",
        ],
        "signal_scale_features": [
            "log1p_sample_max_intensity", "log1p_sample_abs_max_rt_delta",
        ],
        "fragment_context": ["ion_type", "fragment_charge"],
        "fragment_attention_eligibility": (
            "any_real_scan_and_fragment_attempted_and_fragment_separable"),
        "sample_scale_fragment_scope": (
            "fragment_attempted_and_fragment_separable_only"),
        "fragment_ordinal_included": False,
        "fragment_count_included": False,
        "predicted_intensity_included": bool(include_predicted_intensity),
    }


def _masked_max(values: np.ndarray, mask: np.ndarray) -> float:
    selected = np.asarray(values, dtype="f4")[np.asarray(mask, dtype=bool)]
    return float(selected.max()) if selected.size else 0.0


def _normalize_record(record: dict, *, mass_tol_ppm: float,
                      include_predicted_intensity: bool) -> dict:
    """Convert one stored record to bounded signal-derived model inputs."""
    precursor_scan = np.asarray(record["precursor_scan_mask"], dtype=bool)
    precursor_peak = np.asarray(record["precursor_peak_mask"], dtype=bool)
    fragment_scan = np.asarray(record["fragment_scan_mask"], dtype=bool)
    fragment_peak = np.asarray(record["fragment_peak_mask"], dtype=bool)
    fragment_eligible = (
        np.asarray(record["fragment_attempted"], dtype=bool)
        & np.asarray(record["fragment_separable"], dtype=bool)
    )
    eligible_fragment_scan = (
        fragment_scan & fragment_eligible[:, np.newaxis, np.newaxis])

    intensity_scale = max(
        _masked_max(record["precursor_intensity"], precursor_scan),
        _masked_max(record["fragment_intensity"], eligible_fragment_scan),
        1.0,
    )
    precursor_rt = np.asarray(record["precursor_rt_delta"], dtype="f4")
    fragment_rt = np.asarray(record["fragment_rt_delta"], dtype="f4")
    rt_scale = max(
        _masked_max(np.abs(precursor_rt), precursor_scan),
        _masked_max(np.abs(fragment_rt), eligible_fragment_scan),
        1e-3,
    )

    def trace_features(prefix: str, scan: np.ndarray,
                       peak: np.ndarray) -> np.ndarray:
        intensity = np.log1p(np.asarray(
            record[f"{prefix}_intensity"], dtype="f4")) / math.log1p(
                intensity_scale)
        ppm = np.asarray(record[f"{prefix}_ppm_error"], dtype="f4")
        ppm = np.where(peak, ppm / float(mass_tol_ppm), 0.0)
        rt = np.asarray(record[f"{prefix}_rt_delta"], dtype="f4")
        rt = np.where(scan, rt / rt_scale, 0.0)
        intensity = np.where(scan, intensity, 0.0)
        return np.stack([
            intensity, ppm, rt, scan.astype("f4"), peak.astype("f4"),
        ], axis=-2).astype("f4", copy=False)

    precursor = trace_features(
        "precursor", precursor_scan, precursor_peak)
    precursor = precursor.reshape(
        precursor.shape[0] * _N_SIGNAL_FEATURES, precursor.shape[-1])
    fragment = trace_features("fragment", fragment_scan, fragment_peak)
    fragment = fragment.reshape(
        fragment.shape[0],
        fragment.shape[1] * _N_SIGNAL_FEATURES,
        fragment.shape[-1],
    )
    # ``attempted`` and ``separable`` are used only as a hard eligibility
    # gate. They are never exposed as numeric inputs. In particular, a
    # co-isolated same-m/z pair stores a copied heavy trace for audit/parity;
    # allowing it into attention would turn that copy into false evidence.
    fragment_mask = (
        fragment_scan.any(axis=(-1, -2))
        & fragment_eligible
    )

    result = {
        "precursor": precursor,
        "fragment": fragment,
        "fragment_mask": fragment_mask.astype(bool, copy=False),
        # Reserve embedding index 0 for padding.
        "fragment_ion_type": np.asarray(
            record["fragment_ion_type"], dtype="i8") + 1,
        "fragment_charge": np.asarray(
            record["fragment_charge"], dtype="i8"),
        "signal_scale": np.asarray([
            math.log1p(intensity_scale), math.log1p(rt_scale),
        ], dtype="f4"),
        "label": int(record["metadata"]["label"]),
        "sample_id": str(record["metadata"]["sample_id"]),
    }
    if include_predicted_intensity:
        present = np.asarray(
            record["fragment_prediction_present"], dtype=bool)
        prediction = np.asarray(
            record["fragment_predicted_intensity"], dtype="f4")
        prediction = np.where(present, prediction, 0.0)
        result["fragment_prediction"] = np.stack([
            prediction, present.astype("f4"),
        ], axis=-1).astype("f4", copy=False)
    return result


class XICDataset(Dataset):
    """Index view over a :class:`SignalDataset` with no split leakage."""

    def __init__(self, source: SignalDataset, indices: Iterable[int] | None = None,
                 *, include_predicted_intensity: bool = False):
        self.source = source
        self.indices = np.asarray(
            list(range(len(source))) if indices is None else list(indices),
            dtype="i8")
        if self.indices.ndim != 1 or (
                len(self.indices) and (
                    self.indices.min() < 0 or self.indices.max() >= len(source))):
            raise IndexError("XICDataset indices are outside the signal dataset")
        if len(set(self.indices.tolist())) != len(self.indices):
            raise ValueError("XICDataset indices must be unique")
        extraction = source.schema.get("extraction", {})
        self.mass_tol_ppm = float(extraction["mass_tol_ppm"])
        self.include_predicted_intensity = bool(include_predicted_intensity)

    def __len__(self) -> int:
        return len(self.indices)

    def __getitem__(self, index: int) -> dict:
        source_index = int(self.indices[index])
        result = _normalize_record(
            self.source[source_index], mass_tol_ppm=self.mass_tol_ppm,
            include_predicted_intensity=self.include_predicted_intensity)
        result["source_index"] = source_index
        return result


def collate_xic(records: Sequence[dict]) -> dict:
    """Pad a batch's ragged fragment axis and preserve sample order."""
    if not records:
        raise ValueError("cannot collate an empty XIC batch")
    batch_size = len(records)
    max_fragments = max(1, max(
        int(record["fragment"].shape[0]) for record in records))
    fragment_channels = int(records[0]["fragment"].shape[1])
    trace_length = int(records[0]["fragment"].shape[2])
    fragment = np.zeros(
        (batch_size, max_fragments, fragment_channels, trace_length),
        dtype="f4")
    fragment_mask = np.zeros((batch_size, max_fragments), dtype=bool)
    ion_type = np.zeros((batch_size, max_fragments), dtype="i8")
    charge = np.zeros((batch_size, max_fragments), dtype="i8")
    include_prediction = "fragment_prediction" in records[0]
    prediction = np.zeros((batch_size, max_fragments, 2), dtype="f4") \
        if include_prediction else None

    for row, record in enumerate(records):
        if ("fragment_prediction" in record) != include_prediction:
            raise ValueError("mixed prediction-arm records in one XIC batch")
        count = int(record["fragment"].shape[0])
        if count:
            fragment[row, :count] = record["fragment"]
            fragment_mask[row, :count] = record["fragment_mask"]
            ion_type[row, :count] = record["fragment_ion_type"]
            charge[row, :count] = record["fragment_charge"]
            if prediction is not None:
                prediction[row, :count] = record["fragment_prediction"]

    result = {
        "precursor": torch.from_numpy(np.stack([
            record["precursor"] for record in records])),
        "fragment": torch.from_numpy(fragment),
        "fragment_mask": torch.from_numpy(fragment_mask),
        "fragment_ion_type": torch.from_numpy(ion_type),
        "fragment_charge": torch.from_numpy(charge),
        "signal_scale": torch.from_numpy(np.stack([
            record["signal_scale"] for record in records])),
        "label": torch.tensor([
            record["label"] for record in records], dtype=torch.float32),
        "sample_id": [record["sample_id"] for record in records],
        "source_index": torch.tensor([
            record["source_index"] for record in records], dtype=torch.long),
    }
    if prediction is not None:
        result["fragment_prediction"] = torch.from_numpy(prediction)
    return result


class ShardBatchSampler(Sampler[list[int]]):
    """Shuffle batches while keeping every batch inside one mmap shard."""

    def __init__(self, dataset: XICDataset, batch_size: int, *, seed: int,
                 shuffle: bool = True, drop_last: bool = False):
        if batch_size < 1:
            raise ValueError("batch_size must be positive")
        self.dataset = dataset
        self.batch_size = int(batch_size)
        self.seed = int(seed)
        self.shuffle = bool(shuffle)
        self.drop_last = bool(drop_last)
        self.epoch = 0
        by_shard: dict[str, list[int]] = defaultdict(list)
        manifest = dataset.source.manifest
        for local_index, source_index in enumerate(dataset.indices):
            by_shard[str(manifest.iloc[int(source_index)]["shard"])].append(
                local_index)
        self._by_shard = dict(by_shard)

    def set_epoch(self, epoch: int) -> None:
        self.epoch = int(epoch)

    def __iter__(self) -> Iterator[list[int]]:
        rng = random.Random(self.seed + self.epoch)
        shards = list(self._by_shard.items())
        if self.shuffle:
            rng.shuffle(shards)
        for _shard, indices in shards:
            values = list(indices)
            if self.shuffle:
                rng.shuffle(values)
            for start in range(0, len(values), self.batch_size):
                batch = values[start:start + self.batch_size]
                if len(batch) == self.batch_size or not self.drop_last:
                    yield batch

    def __len__(self) -> int:
        if self.drop_last:
            return sum(
                len(values) // self.batch_size
                for values in self._by_shard.values())
        return sum(
            math.ceil(len(values) / self.batch_size)
            for values in self._by_shard.values())
