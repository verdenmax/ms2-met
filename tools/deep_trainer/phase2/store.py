"""Immutable sharded storage for Phase 2 XIC tensors."""

from __future__ import annotations

from collections.abc import Iterable
from dataclasses import dataclass, field
from datetime import datetime, timezone
import json
import logging
import os
from pathlib import Path
import shutil
import tempfile
from typing import Callable
import uuid

import numpy as np
import pandas as pd

from artifact_identity import sha256_file
from .schema import (
    SCHEMA_VERSION, ExtractionSettings, SignalSample, schema_document,
)


_PRECURSOR_ARRAYS = (
    "precursor_intensity", "precursor_ppm_error", "precursor_rt_delta",
    "precursor_scan_mask", "precursor_peak_mask",
)
_FRAGMENT_TRACE_ARRAYS = (
    "fragment_intensity", "fragment_ppm_error", "fragment_rt_delta",
    "fragment_scan_mask", "fragment_peak_mask",
)
_FRAGMENT_VECTOR_ARRAYS = (
    "fragment_ion_type", "fragment_ordinal", "fragment_charge",
    "fragment_light_mz", "fragment_heavy_mz",
    "fragment_predicted_intensity", "fragment_prediction_present",
    "fragment_separable", "fragment_attempted", "fragment_status",
)
_ALL_ARRAYS = _PRECURSOR_ARRAYS + _FRAGMENT_TRACE_ARRAYS + \
    _FRAGMENT_VECTOR_ARRAYS + ("fragment_offsets",)


@dataclass
class StagedValidation:
    """Artifacts produced by validation of the serialized dataset."""

    audit_tables: dict[str, pd.DataFrame] = field(default_factory=dict)
    audit_documents: dict[str, dict] = field(default_factory=dict)
    summary: dict = field(default_factory=dict)


def _json_safe_value(value):
    if isinstance(value, np.generic):
        return value.item()
    if isinstance(value, Path):
        return str(value)
    if isinstance(value, dict):
        return {str(key): _json_safe_value(item)
                for key, item in value.items()}
    if isinstance(value, (list, tuple)):
        return [_json_safe_value(item) for item in value]
    return value


def _json_safe_metadata(metadata: dict) -> dict:
    safe = {}
    for key, value in metadata.items():
        if isinstance(value, (dict, list, tuple)):
            value = json.dumps(
                _json_safe_value(value), ensure_ascii=False, sort_keys=True)
        safe[str(key)] = _json_safe_value(value)
    return safe


def _save_array(path: Path, value: np.ndarray) -> None:
    with path.open("wb") as handle:
        np.save(handle, value, allow_pickle=False)


def _write_shard(root: Path, shard_index: int,
                 samples: list[SignalSample]) -> list[dict]:
    shard_name = f"shard_{shard_index:05d}"
    shard_root = root / "shards" / shard_name
    shard_root.mkdir(parents=True)

    arrays: dict[str, np.ndarray] = {}
    for name in _PRECURSOR_ARRAYS:
        arrays[name] = np.stack([getattr(sample, name) for sample in samples])
    for name in _FRAGMENT_TRACE_ARRAYS + _FRAGMENT_VECTOR_ARRAYS:
        values = [getattr(sample, name) for sample in samples]
        arrays[name] = np.concatenate(values, axis=0)
    lengths = np.asarray([
        len(sample.fragment_ion_type) for sample in samples
    ], dtype="i8")
    arrays["fragment_offsets"] = np.concatenate([
        np.zeros(1, dtype="i8"), np.cumsum(lengths, dtype="i8")])

    for name, value in arrays.items():
        _save_array(shard_root / f"{name}.npy", value)

    rows = []
    for row_index, sample in enumerate(samples):
        row = _json_safe_metadata(sample.metadata)
        row.update({
            "shard": shard_name,
            "shard_row": row_index,
            "fragment_start": int(arrays["fragment_offsets"][row_index]),
            "fragment_end": int(arrays["fragment_offsets"][row_index + 1]),
        })
        rows.append(row)
    return rows


def recover_interrupted_publish(
    output_root: str | Path,
    *,
    cleanup_stale_backups: bool = False,
) -> None:
    """Restore the only backup left by an interrupted atomic overwrite."""
    output_root = Path(output_root).resolve()
    backups = sorted(output_root.parent.glob(
        f".{output_root.name}.backup.*"))
    if output_root.exists():
        if backups and cleanup_stale_backups:
            # Do not destroy the only recoverable copies unless the canonical
            # output is demonstrably complete and immutable.
            SignalDataset(output_root)
            for backup in backups:
                shutil.rmtree(backup)
            logging.warning(
                "removed %d stale Phase 2 backup(s) after verifying current "
                "output: %s", len(backups), output_root)
        elif backups:
            logging.warning(
                "completed Phase 2 output coexists with %d stale backup(s): %s",
                len(backups), ", ".join(str(path) for path in backups))
        return
    if not backups:
        return
    if len(backups) != 1:
        raise RuntimeError(
            "cannot recover interrupted Phase 2 publish unambiguously; "
            f"found backups: {[str(path) for path in backups]}")
    os.replace(backups[0], output_root)
    logging.warning(
        "restored Phase 2 dataset after interrupted overwrite: %s",
        output_root)


def _publish(staging: Path, output_root: Path, overwrite: bool) -> None:
    backup = None
    if output_root.exists():
        if not overwrite:
            raise FileExistsError(f"output path already exists: {output_root}")
        backup = output_root.with_name(
            f".{output_root.name}.backup.{uuid.uuid4().hex}")
        os.replace(output_root, backup)
    try:
        os.replace(staging, output_root)
    except BaseException:
        if backup is not None and backup.exists() and not output_root.exists():
            os.replace(backup, output_root)
        raise
    if backup is not None and backup.exists():
        try:
            shutil.rmtree(backup)
        except OSError:
            logging.warning(
                "published Phase 2 dataset but could not remove backup: %s",
                backup, exc_info=True)


def _write_audit_tables(root: Path,
                        tables: dict[str, pd.DataFrame]) -> None:
    for relative, table in tables.items():
        path = root / "audit" / relative
        path.parent.mkdir(parents=True, exist_ok=True)
        table.to_csv(path, index=False)


def _write_audit_documents(root: Path, documents: dict[str, dict]) -> None:
    for relative, document in documents.items():
        path = root / "audit" / relative
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_text(
            json.dumps(document, indent=2, ensure_ascii=False) + "\n",
            encoding="utf-8")


def write_signal_dataset(
    samples: Iterable[SignalSample],
    output_root: str | Path,
    settings: ExtractionSettings,
    *,
    build_metadata: dict,
    shard_size: int = 1024,
    overwrite: bool = False,
    audit_tables: dict[str, pd.DataFrame] | None = None,
    audit_documents: dict[str, dict] | None = None,
    staged_validator: Callable[["SignalDataset"], StagedValidation] | None = None,
) -> dict:
    """Validate, shard, checksum and atomically publish a signal dataset."""
    if shard_size <= 0:
        raise ValueError("shard_size must be positive")
    output_root = Path(output_root).resolve()
    output_root.parent.mkdir(parents=True, exist_ok=True)
    recover_interrupted_publish(
        output_root, cleanup_stale_backups=overwrite)
    if output_root.exists() and not overwrite:
        raise FileExistsError(f"output path already exists: {output_root}")
    staging = Path(tempfile.mkdtemp(
        prefix=f".{output_root.name}.staging.", dir=output_root.parent))

    rows: list[dict] = []
    pending: list[SignalSample] = []
    seen_ids: set[str] = set()
    class_counts = {"correct_identification": 0,
                    "incorrect_identification": 0}
    n_fragments = 0
    try:
        for sample in samples:
            sample.validate(settings)
            sample_id = str(sample.metadata["sample_id"])
            if sample_id in seen_ids:
                raise ValueError(f"duplicate signal sample_id: {sample_id}")
            seen_ids.add(sample_id)
            label = int(sample.metadata["label"])
            class_counts[
                "correct_identification" if label == 1
                else "incorrect_identification"
            ] += 1
            n_fragments += len(sample.fragment_ion_type)
            pending.append(sample)
            if len(pending) == shard_size:
                rows.extend(_write_shard(
                    staging, len(rows) // shard_size, pending))
                pending = []
        if pending:
            rows.extend(_write_shard(
                staging, len(rows) // shard_size, pending))
        if not rows:
            raise ValueError("refusing to publish an empty signal dataset")

        manifest = pd.DataFrame(rows)
        if manifest["sample_id"].duplicated().any():
            raise ValueError("manifest contains duplicate sample_id values")
        manifest_path = staging / "manifest.parquet"
        manifest.to_parquet(manifest_path, index=False)

        schema = schema_document(settings)
        schema["build"] = _json_safe_value(build_metadata)
        (staging / "schema.json").write_text(
            json.dumps(schema, indent=2, ensure_ascii=False) + "\n",
            encoding="utf-8")

        _write_audit_tables(staging, audit_tables or {})
        _write_audit_documents(staging, audit_documents or {})

        validation = StagedValidation()
        if staged_validator is not None:
            validation = staged_validator(SignalDataset._open_staging(staging))
            if not isinstance(validation, StagedValidation):
                raise TypeError(
                    "staged_validator must return StagedValidation")
            _write_audit_tables(staging, validation.audit_tables)
            _write_audit_documents(staging, validation.audit_documents)

        report = {
            "schema": "phase2_raw_xic_build_report_v1",
            "dataset_schema": SCHEMA_VERSION,
            "created_at_utc": datetime.now(timezone.utc).isoformat(),
            "status": "complete",
            "n_samples": len(rows),
            "n_fragment_charge_records": n_fragments,
            "n_shards": int(manifest["shard"].nunique()),
            "class_counts": class_counts,
            "sample_ids_unique": True,
            "metric_semantics": "error_identification_positive_v1",
            "positive_class": "incorrect_identification",
            "staged_validation": _json_safe_value(validation.summary),
        }
        (staging / "build_report.json").write_text(
            json.dumps(report, indent=2, ensure_ascii=False) + "\n",
            encoding="utf-8")
        checksum_paths = sorted([
            path for path in staging.rglob("*") if path.is_file()
        ])
        checksums = {
            str(path.relative_to(staging)): {
                "sha256": sha256_file(path),
                "size_bytes": path.stat().st_size,
            }
            for path in checksum_paths
        }
        checksums_path = staging / "checksums.json"
        checksums_path.write_text(
            json.dumps(checksums, indent=2, sort_keys=True) + "\n",
            encoding="utf-8")
        (staging / "COMPLETE").write_text(
            json.dumps({
                "schema": SCHEMA_VERSION,
                "status": "complete",
                "checksums_sha256": sha256_file(checksums_path),
                "n_artifacts": len(checksums),
            }, sort_keys=True) + "\n", encoding="utf-8")
        _publish(staging, output_root, overwrite)
        return report
    except BaseException:
        shutil.rmtree(staging, ignore_errors=True)
        raise


class SignalDataset:
    """Read-only mmap view over an immutable Phase 2 signal dataset."""

    def __init__(self, root: str | Path, *, verify_checksums: bool = True):
        self.root = Path(root).resolve()
        required = [
            self.root / "COMPLETE", self.root / "schema.json",
            self.root / "manifest.parquet", self.root / "checksums.json",
            self.root / "build_report.json",
        ]
        missing = [str(path) for path in required if not path.is_file()]
        if missing:
            raise FileNotFoundError(
                "signal dataset is incomplete:\n  " + "\n  ".join(missing))
        complete = json.loads(
            (self.root / "COMPLETE").read_text(encoding="utf-8"))
        if complete.get("schema") != SCHEMA_VERSION or \
                complete.get("status") != "complete":
            raise ValueError("invalid Phase 2 COMPLETE marker")
        if not complete.get("checksums_sha256"):
            raise ValueError(
                "Phase 2 COMPLETE marker does not anchor checksums.json")
        self.complete = complete
        self._load_core()
        if verify_checksums:
            self.verify_checksums()

    @classmethod
    def _open_staging(cls, root: str | Path) -> "SignalDataset":
        """Open unpublished shards for mandatory post-serialization checks."""
        dataset = cls.__new__(cls)
        dataset.root = Path(root).resolve()
        dataset.complete = None
        dataset._load_core()
        return dataset

    def _load_core(self) -> None:
        required = [self.root / "schema.json", self.root / "manifest.parquet"]
        missing = [str(path) for path in required if not path.is_file()]
        if missing:
            raise FileNotFoundError(
                "signal dataset lacks serialized core artifacts:\n  "
                + "\n  ".join(missing))
        self.schema = json.loads(
            (self.root / "schema.json").read_text(encoding="utf-8"))
        if self.schema.get("schema") != SCHEMA_VERSION:
            raise ValueError(
                f"unsupported signal schema: {self.schema.get('schema')!r}")
        self.manifest = pd.read_parquet(self.root / "manifest.parquet")
        if self.manifest["sample_id"].duplicated().any():
            raise ValueError("signal manifest contains duplicate sample IDs")
        self._cache_name: str | None = None
        self._cache: dict[str, np.ndarray] = {}

    def __len__(self) -> int:
        return len(self.manifest)

    def verify_checksums(self) -> None:
        checksums_path = self.root / "checksums.json"
        observed_checksums_hash = sha256_file(checksums_path)
        if observed_checksums_hash != self.complete["checksums_sha256"]:
            raise ValueError("Phase 2 checksums.json differs from COMPLETE")
        checksums = json.loads(checksums_path.read_text(encoding="utf-8"))
        actual = {
            str(path.relative_to(self.root))
            for path in self.root.rglob("*") if path.is_file()
            and path.name not in {"COMPLETE", "checksums.json"}
        }
        expected = set(checksums)
        if actual != expected:
            missing = sorted(expected - actual)
            unexpected = sorted(actual - expected)
            raise ValueError(
                "Phase 2 checksum coverage differs from artifacts: "
                f"missing={missing}, unexpected={unexpected}")
        if int(self.complete.get("n_artifacts", -1)) != len(checksums):
            raise ValueError("Phase 2 COMPLETE artifact count is inconsistent")
        for relative, expected in checksums.items():
            path = self.root / relative
            if not path.is_file() or path.stat().st_size != expected["size_bytes"]:
                raise ValueError(f"signal dataset artifact changed: {relative}")
            if sha256_file(path) != expected["sha256"]:
                raise ValueError(f"signal dataset checksum mismatch: {relative}")

    def _load_shard(self, name: str) -> dict[str, np.ndarray]:
        if self._cache_name != name:
            shard = self.root / "shards" / name
            self._cache = {
                array_name: np.load(
                    shard / f"{array_name}.npy", mmap_mode="r",
                    allow_pickle=False)
                for array_name in _ALL_ARRAYS
            }
            self._cache_name = name
        return self._cache

    def __getitem__(self, index: int) -> dict:
        if index < 0:
            index += len(self)
        if index < 0 or index >= len(self):
            raise IndexError(index)
        row = self.manifest.iloc[index]
        arrays = self._load_shard(str(row["shard"]))
        shard_row = int(row["shard_row"])
        fragment_start = int(row["fragment_start"])
        fragment_end = int(row["fragment_end"])
        result = {
            name: arrays[name][shard_row] for name in _PRECURSOR_ARRAYS
        }
        result.update({
            name: arrays[name][fragment_start:fragment_end]
            for name in _FRAGMENT_TRACE_ARRAYS + _FRAGMENT_VECTOR_ARRAYS
        })
        result["metadata"] = row.to_dict()
        return result

    def sample(self, index: int) -> SignalSample:
        """Return one on-disk record through the canonical tensor contract."""
        record = self[index]
        metadata = record.pop("metadata")
        return SignalSample(metadata=metadata, **record)


def open_signal_dataset(root: str | Path, *,
                        verify_checksums: bool = True) -> SignalDataset:
    """Open the only supported Phase 2 storage adapter."""
    return SignalDataset(root, verify_checksums=verify_checksums)
