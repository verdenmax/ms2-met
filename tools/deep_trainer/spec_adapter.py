"""Frozen fixed-negpool protocol adapter for neural experiments.

The LightGBM fixed-negpool bundle owns sample membership and every outer/inner
assignment.  Deep trainers may validate the current feature snapshot, but they
must consume those frozen manifests instead of independently choosing a split.
"""

from __future__ import annotations

from dataclasses import dataclass
import hashlib
import json
from pathlib import Path
import sys

import numpy as np
import pandas as pd
import yaml


_PROJECT_ROOT = Path(__file__).resolve().parents[2]
_SPEC_SRC = _PROJECT_ROOT / "tools" / "spec_trainer" / "src"
if str(_SPEC_SRC) not in sys.path:
    sys.path.insert(0, str(_SPEC_SRC))

from fixed_negpool import (  # noqa: E402
    MODEL_TIERS,
    _DATASET,
    _OUTER_FOLD,
    _SAMPLE_ID,
    _SOURCE_ROW,
    _SPLIT,
    _TIER,
    _assert_formal_config,
    feature_paths,
    prepare_combined_fixed_negpool,
    prepare_fixed_negpool,
)
from cv_core import METRIC_SEMANTICS_VERSION  # noqa: E402
from cv_train import _operating_targets, _validate_frame  # noqa: E402


def _sha256(path):
    digest = hashlib.sha256()
    with open(path, "rb") as handle:
        for block in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


@dataclass
class PreparedProtocol:
    frame: pd.DataFrame
    membership: pd.DataFrame
    group_fold_map: pd.DataFrame
    feature_cols: list[str]
    identity_cols: list[str]
    validation: dict
    feature_arm: str
    cohort: str
    target_col: str
    base_group_col: str
    group_col: str
    sample_id_col: str
    source_row_col: str
    dataset_col: str
    tier_col: str
    split_col: str
    outer_fold_col: str
    inner_valid_cols: list[str]
    model_tiers: dict[str, tuple[str, ...]]
    target_fprs: list[float]
    protocol_root: str

    def training_frame(self, model_name: str) -> pd.DataFrame:
        if model_name not in self.model_tiers:
            raise ValueError(
                f"unknown model pool {model_name!r}; expected "
                f"{sorted(self.model_tiers)}")
        train = self.frame[self.frame[self.split_col].eq("train")]
        allowed = self.model_tiers[model_name]
        use = train[self.target_col].eq(1) | train[self.tier_col].isin(allowed)
        result = train.loc[use].copy().reset_index(drop=True)
        _validate_frame(
            result, self.feature_cols, self.target_col, self.group_col)
        return result

    def test_frame(self) -> pd.DataFrame:
        result = self.frame[
            self.frame[self.split_col].eq("test")].copy().reset_index(drop=True)
        _validate_frame(
            result, self.feature_cols, self.target_col, self.group_col)
        return result


def _load_frozen_bundle(protocol_root, dataset, config):
    root = Path(protocol_root)
    required = {
        "summary": root / "summary.json",
        "preflight": root / "preflight.json",
        "membership": root / "manifests" / "membership.csv",
        "fixed_manifest": root / "manifests" / "fixed_test_manifest.csv",
        "fold_map": root / "manifests" / "fold_map.csv",
        "predictions": root / "predictions" / "fixed_test_predictions.csv",
    }
    missing = [str(path) for path in required.values() if not path.is_file()]
    if missing:
        raise FileNotFoundError(
            "frozen fixed-negpool protocol is incomplete:\n  "
            + "\n  ".join(missing))
    summary = json.loads(required["summary"].read_text(encoding="utf-8"))
    frozen_bundle = summary.get("frozen_bundle", {})
    if (frozen_bundle.get("schema") != "fixed_negpool_frozen_bundle_v2"
            or frozen_bundle.get("complete") is not True):
        raise ValueError(
            "frozen protocol lacks a completed artifact-hash contract; "
            "rerun the LightGBM fixed-negpool experiment")
    expected_hashes = frozen_bundle.get("artifact_sha256", {})
    artifact_names = {
        "preflight": "preflight",
        "membership": "membership",
        "fixed_manifest": "fixed_test_manifest",
        "fold_map": "fold_map",
        "predictions": "fixed_test_predictions",
    }
    mismatches = [
        name for key, name in artifact_names.items()
        if not expected_hashes.get(name)
        or _sha256(required[key]) != expected_hashes[name]
    ]
    if mismatches:
        raise ValueError(
            "frozen fixed-negpool artifacts are missing or were modified: "
            f"{mismatches}")
    preflight = json.loads(required["preflight"].read_text(encoding="utf-8"))
    if preflight.get("protocol_schema") != "fixed_negpool_protocol_v2":
        raise ValueError(
            "frozen protocol predates manifest/hash/family validation; "
            "rerun the LightGBM fixed-negpool experiment")
    if summary.get("metric_semantics") != METRIC_SEMANTICS_VERSION:
        raise ValueError("frozen protocol uses legacy metric semantics")
    if summary.get("positive_class") != "incorrect_identification":
        raise ValueError("frozen protocol has the wrong positive class")
    if summary.get("dataset") != dataset:
        raise ValueError(
            f"frozen protocol dataset={summary.get('dataset')!r}, "
            f"requested {dataset!r}")
    design = summary.get("design", {})
    if design.get("cohort") != config["data"]["cohort"]:
        raise ValueError("frozen protocol cohort differs from split config")
    if design.get("feature_arm") != config["data"]["feature_arm"]:
        raise ValueError("frozen protocol feature arm differs from split config")
    return root, required, summary, preflight


def _assert_frozen_inputs(summary, feature_root, dataset):
    frozen = summary.get("provenance", {}).get("inputs", [])
    if not frozen or any(not item.get("sha256") for item in frozen):
        raise ValueError(
            "frozen LightGBM bundle lacks full input SHA256 fingerprints; "
            "rerun train-fixed-test-negpool-combined with current code")
    frozen_by_name = {
        Path(item["path"]).parent.name: item for item in frozen
    }
    sources = ("2da", "5da", "normal") if dataset == "combined" else (dataset,)
    current = [
        Path(feature_root) / f"baseline_{source}_{pool}" / "features.csv"
        for source in sources for pool in ("neg05", "neg10", "neg20")
    ]
    if set(frozen_by_name) != {path.parent.name for path in current}:
        raise ValueError(
            "frozen LightGBM input set differs from the requested feature root")
    mismatches = []
    for path in current:
        if not path.is_file():
            raise FileNotFoundError(f"missing current feature input: {path}")
        if _sha256(path) != frozen_by_name[path.parent.name]["sha256"]:
            mismatches.append(path.parent.name)
    if mismatches:
        raise ValueError(
            "current feature content differs from frozen LightGBM inputs: "
            f"{mismatches}")


def _protocol_parameters(preflight):
    fixed = preflight.get("fixed_split", {})
    try:
        test_fraction = float(fixed["test_fraction_target"])
        split_candidates = int(fixed["n_candidates"])
    except (KeyError, TypeError, ValueError) as exc:
        raise ValueError(
            "frozen preflight lacks split parameters required for validation"
        ) from exc
    minimum = preflight.get(
        "min_test_errors_per_dataset_tier",
        preflight.get("min_test_errors_per_tier"),
    )
    if minimum is None:
        minima = fixed.get("minimum_test_rows_by_stratum", {})
        minimum = min(minima.values()) if minima else 1
    return test_fraction, int(minimum), split_candidates


def _assert_same_ids(current, frozen, label):
    if current[_SAMPLE_ID].duplicated().any():
        raise ValueError(f"current {label} contains duplicate sample_id")
    if frozen[_SAMPLE_ID].duplicated().any():
        raise ValueError(f"frozen {label} contains duplicate sample_id")
    current_ids = set(current[_SAMPLE_ID].astype(str))
    frozen_ids = set(frozen[_SAMPLE_ID].astype(str))
    if current_ids != frozen_ids:
        raise ValueError(
            f"current feature snapshot differs from frozen {label}: "
            f"only_current={len(current_ids - frozen_ids)}, "
            f"only_frozen={len(frozen_ids - current_ids)}")


def _assert_columns_equal(current, frozen, columns, label):
    left = current.set_index(_SAMPLE_ID)
    right = frozen.set_index(_SAMPLE_ID)
    for column in columns:
        if column not in left or column not in right:
            continue
        a = left[column].sort_index().astype("string").fillna("<NA>")
        b = right[column].sort_index().astype("string").fillna("<NA>")
        if not a.equals(b):
            raise ValueError(
                f"current feature snapshot differs from frozen {label} "
                f"column {column!r}")


def _apply_frozen_assignments(prepared, required, config, summary):
    membership = pd.read_csv(required["membership"], low_memory=False)
    manifest = pd.read_csv(required["fixed_manifest"], low_memory=False)
    fold_map = pd.read_csv(required["fold_map"], low_memory=False)
    _assert_same_ids(prepared.membership, membership, "membership")
    _assert_columns_equal(
        prepared.membership, membership,
        [config["data"]["target_col"], _TIER, _DATASET, *prepared.identity_cols],
        "membership",
    )
    _assert_same_ids(prepared.frame, manifest, "cohort manifest")
    _assert_columns_equal(
        prepared.frame, manifest,
        [config["data"]["target_col"], _TIER, _DATASET, *prepared.identity_cols],
        "cohort manifest",
    )

    base_group_col = config["data"]["group_col"]
    split_group_candidates = [
        prepared.split_group_col, summary.get("design", {}).get(
            "split_group_col"), base_group_col,
    ]
    split_group_col = next((
        column for column in split_group_candidates
        if column and column in prepared.frame and column in fold_map
    ), None)
    if split_group_col is None:
        raise ValueError(
            "frozen fold_map cannot be joined to the current leakage group; "
            f"columns={list(fold_map)}")
    if prepared.split_group_col != split_group_col:
        raise ValueError(
            "current candidate-family grouping differs from the frozen "
            "protocol; regenerate the LightGBM fixed-negpool bundle")

    assignment_columns = [_SPLIT, _OUTER_FOLD]
    missing = [column for column in assignment_columns if column not in manifest]
    if missing:
        raise ValueError(f"frozen cohort manifest lacks {missing}")
    frame = prepared.frame.drop(
        columns=[
            column for column in prepared.frame
            if column in assignment_columns
            or column.startswith("inner_valid_for_fold_")
        ],
        errors="ignore",
    )
    assignments = manifest[[_SAMPLE_ID, *assignment_columns]].copy()
    frame = frame.merge(
        assignments, on=_SAMPLE_ID, how="left", validate="one_to_one")
    if frame[assignment_columns].isna().any(axis=None):
        raise ValueError("frozen cohort assignments left rows unmatched")

    n_folds = int(config["training"].get("cv_folds", 5))
    inner_columns = [
        f"inner_valid_for_fold_{fold}" for fold in range(n_folds)]
    missing = [column for column in inner_columns if column not in fold_map]
    if missing:
        raise ValueError(f"frozen fold_map lacks {missing}")
    if fold_map[split_group_col].duplicated().any():
        raise ValueError("frozen fold_map has duplicate leakage groups")
    frame = frame.merge(
        fold_map[[split_group_col, *inner_columns]],
        on=split_group_col, how="left", validate="many_to_one")
    if frame[inner_columns].isna().any(axis=None):
        raise ValueError("frozen fold_map left cohort rows unmatched")
    frame[_OUTER_FOLD] = frame[_OUTER_FOLD].astype(int)
    for column in inner_columns:
        frame[column] = frame[column].astype(bool)

    test = frame[_SPLIT].eq("test")
    if not frame.loc[test, _OUTER_FOLD].eq(-1).all():
        raise ValueError("frozen test rows must have outer_fold=-1")
    train_folds = set(frame.loc[~test, _OUTER_FOLD].tolist())
    if train_folds != set(range(n_folds)):
        raise ValueError(
            f"frozen training folds are incomplete: {sorted(train_folds)}")
    current_groups = set(frame[split_group_col].astype(str))
    frozen_groups = set(fold_map[split_group_col].astype(str))
    if current_groups != frozen_groups:
        raise ValueError("frozen fold_map group membership differs from cohort")
    return frame, membership, fold_map, split_group_col, inner_columns


def prepare_protocol(
    split_config_path: str,
    feature_root: str,
    dataset: str,
    protocol_root: str,
) -> PreparedProtocol:
    """Validate features, then apply one frozen LightGBM protocol bundle."""
    with open(split_config_path, encoding="utf-8") as handle:
        config = yaml.safe_load(handle)
    _assert_formal_config(config)
    root, required, summary, frozen_preflight = _load_frozen_bundle(
        protocol_root, dataset, config)
    _assert_frozen_inputs(summary, feature_root, dataset)
    test_fraction, minimum, split_candidates = _protocol_parameters(
        frozen_preflight)
    if dataset == "combined":
        prepared = prepare_combined_fixed_negpool(
            feature_root,
            config,
            test_fraction=test_fraction,
            min_test_errors_per_tier=minimum,
            split_candidates=split_candidates,
            generate_assignments=False,
        )
    elif dataset in {"2da", "5da", "normal"}:
        prepared = prepare_fixed_negpool(
            feature_paths(feature_root, dataset),
            config,
            test_fraction=test_fraction,
            min_test_errors_per_tier=minimum,
            split_candidates=split_candidates,
            generate_assignments=False,
        )
    else:
        raise ValueError("dataset must be combined, 2da, 5da, or normal")

    frame, membership, fold_map, group_col, inner_columns = (
        _apply_frozen_assignments(prepared, required, config, summary))
    targets, _ = _operating_targets(config)
    validation = dict(prepared.validation)
    validation["frozen_protocol"] = {
        "contract": "lightgbm_fixed_negpool_manifest_v1",
        "root": str(root.resolve()),
        "sample_ids_exact_match": True,
        "cohort_assignments_loaded_from_manifest": True,
        "fold_assignments_loaded_from_manifest": True,
        "split_group_col": group_col,
        "manifest_sha256": {
            key: _sha256(path) for key, path in required.items()
            if key != "summary"
        },
    }
    return PreparedProtocol(
        frame=frame,
        membership=membership,
        group_fold_map=fold_map,
        feature_cols=prepared.feature_cols,
        identity_cols=prepared.identity_cols,
        validation=validation,
        feature_arm=config["data"]["feature_arm"],
        cohort=config["data"]["cohort"],
        target_col=config["data"]["target_col"],
        base_group_col=config["data"]["group_col"],
        group_col=group_col,
        sample_id_col=_SAMPLE_ID,
        source_row_col=_SOURCE_ROW,
        dataset_col=_DATASET,
        tier_col=_TIER,
        split_col=_SPLIT,
        outer_fold_col=_OUTER_FOLD,
        inner_valid_cols=inner_columns,
        model_tiers={key: tuple(value) for key, value in MODEL_TIERS.items()},
        target_fprs=list(targets),
        protocol_root=str(root.resolve()),
    )
