"""Fail-closed bridge from frozen fixed-negpool assignments to XIC rows."""

from __future__ import annotations

from dataclasses import dataclass
import hashlib
from pathlib import Path

import numpy as np
import pandas as pd

from ..spec_adapter import PreparedProtocol, prepare_protocol
from .store import SignalDataset, open_signal_dataset


_FLOAT_COLUMNS = frozenset({"rt", "precursor_mz", "q_value"})


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def _assert_equal(left: pd.Series, right: pd.Series, column: str) -> None:
    if column in _FLOAT_COLUMNS:
        a = pd.to_numeric(left, errors="coerce").to_numpy(dtype="f8")
        b = pd.to_numeric(right, errors="coerce").to_numpy(dtype="f8")
        equal = np.isclose(a, b, rtol=1e-12, atol=1e-12, equal_nan=True)
    else:
        a = left.astype("string").fillna("<NA>")
        b = right.astype("string").fillna("<NA>")
        equal = a.eq(b).to_numpy()
    if not bool(np.all(equal)):
        examples = left.index[np.flatnonzero(~equal)[:3]].tolist()
        raise ValueError(
            f"XIC dataset differs from frozen protocol column {column!r}; "
            f"mismatched_rows={int((~equal).sum())}, sample_ids={examples}")


def _validate_build_contract(source: SignalDataset,
                             prepared: PreparedProtocol) -> dict:
    build = source.schema.get("build", {})
    if build.get("mode") != "full_frozen_protocol" or \
            build.get("sample_ids_exactly_equal_frozen_protocol") is not True:
        raise ValueError(
            "Phase 2 training requires the complete full_frozen_protocol XIC "
            "dataset; the balanced pilot is validation-only")
    frozen = build.get("frozen_protocol_contract", {})
    current = prepared.validation.get("frozen_protocol", {})
    if frozen.get("contract") != current.get("contract"):
        raise ValueError("XIC dataset uses a different frozen protocol contract")
    if frozen.get("manifest_sha256") != current.get("manifest_sha256"):
        raise ValueError(
            "XIC dataset protocol hashes differ from the current frozen bundle")
    summary = build.get("frozen_protocol_summary", {})
    summary_path = Path(prepared.protocol_root) / "summary.json"
    if summary.get("sha256") != _sha256(summary_path):
        raise ValueError(
            "XIC dataset was built from a different frozen protocol summary")
    if build.get("frozen_sample_ids_exact_match") is not True:
        raise ValueError("XIC dataset lacks an exact frozen-membership assertion")
    return {
        "dataset_mode": build["mode"],
        "protocol_manifest_sha256": current["manifest_sha256"],
        "protocol_summary_sha256": summary["sha256"],
        "signal_checksums_sha256": source.complete["checksums_sha256"],
        "prediction_included": bool(build.get("prediction_included", False)),
    }


@dataclass
class FrozenXICProtocol:
    """One exact row ordering shared by signals, folds, and baselines."""

    source: SignalDataset
    prepared: PreparedProtocol
    frame: pd.DataFrame
    validation: dict

    def training_frame(self, model_name: str) -> pd.DataFrame:
        if model_name not in self.prepared.model_tiers:
            raise ValueError(f"unknown negative pool model: {model_name}")
        train = self.frame[self.frame[self.prepared.split_col].eq("train")]
        allowed = self.prepared.model_tiers[model_name]
        use = train[self.prepared.target_col].eq(1) | train[
            self.prepared.tier_col].isin(allowed)
        result = train.loc[use].copy().reset_index(drop=True)
        if result.empty or set(result[self.prepared.target_col]) != {0, 1}:
            raise ValueError(f"{model_name} XIC training subset lacks both classes")
        return result

    def test_frame(self) -> pd.DataFrame:
        result = self.frame[
            self.frame[self.prepared.split_col].eq("test")
        ].copy().reset_index(drop=True)
        if result.empty or set(result[self.prepared.target_col]) != {0, 1}:
            raise ValueError("fixed XIC test subset lacks both classes")
        return result


def prepare_xic_protocol(
    signal_root: str,
    split_config_path: str,
    feature_root: str,
    protocol_root: str,
    *,
    verify_checksums: bool = True,
) -> FrozenXICProtocol:
    """Validate and align a full XIC dataset to the current frozen protocol."""
    source = open_signal_dataset(
        signal_root, verify_checksums=verify_checksums)
    prepared = prepare_protocol(
        split_config_path, feature_root, "combined", protocol_root)
    build_validation = _validate_build_contract(source, prepared)

    manifest = source.manifest.copy()
    manifest[prepared.sample_id_col] = manifest[
        prepared.sample_id_col].astype(str)
    if manifest[prepared.sample_id_col].duplicated().any():
        raise ValueError("XIC manifest contains duplicate sample IDs")
    current = prepared.frame.copy()
    current[prepared.sample_id_col] = current[prepared.sample_id_col].astype(str)
    signal_ids = set(manifest[prepared.sample_id_col])
    protocol_ids = set(current[prepared.sample_id_col])
    if signal_ids != protocol_ids or len(manifest) != len(current):
        raise ValueError(
            "full XIC sample membership differs from frozen protocol: "
            f"missing={len(protocol_ids - signal_ids)}, "
            f"unexpected={len(signal_ids - protocol_ids)}")

    signal = manifest.set_index(prepared.sample_id_col, verify_integrity=True)
    frozen = current.set_index(prepared.sample_id_col, verify_integrity=True)
    frozen = frozen.loc[signal.index]
    required = list(dict.fromkeys([
        prepared.target_col, prepared.dataset_col, prepared.tier_col,
        prepared.split_col, prepared.outer_fold_col, prepared.group_col,
        *prepared.inner_valid_cols,
        *prepared.identity_cols,
    ]))
    missing = [column for column in required if column not in signal]
    if missing:
        raise ValueError(
            f"full XIC manifest lacks frozen assignment columns: {missing}")
    for column in required:
        _assert_equal(signal[column], frozen[column], column)

    # Preserve the signal store's order so source_index is a stable, explicit
    # bridge into mmap shards. Protocol-only feature values remain outside the
    # model input adapter.
    frame = frozen.reset_index()
    frame["source_index"] = np.arange(len(frame), dtype="i8")
    validation = {
        "schema": "phase2_xic_protocol_preflight_v1",
        "metric_semantics": "error_identification_positive_v1",
        "positive_class": "incorrect_identification",
        "n_samples": len(frame),
        "sample_ids_exact_match": True,
        "assignments_exact_match": True,
        "model_input_excludes_protocol_metadata": True,
        "build_contract": build_validation,
        "frozen_protocol": prepared.validation.get("frozen_protocol", {}),
    }
    return FrozenXICProtocol(
        source=source, prepared=prepared, frame=frame,
        validation=validation)
