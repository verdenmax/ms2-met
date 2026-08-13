"""Frozen-protocol Phase 2 experiment on signal-native XIC models."""

from __future__ import annotations

import argparse
import datetime as dt
import hashlib
import json
import logging
import os
from pathlib import Path
import platform
import shutil
import subprocess
import sys
import tempfile
import uuid

import numpy as np
import pandas as pd
import torch
import yaml

from ..comparison import (
    load_frozen_lightgbm_predictions, paired_cluster_bootstrap,
)
from .checkpoint import save_checkpoint
from .data import XICDataset, input_adapter_contract
from .model import n_trainable_parameters
from .protocol import prepare_xic_protocol
from .training import fit_xic_model, predict_trust


_PROJECT_ROOT = Path(__file__).resolve().parents[3]
_SPEC_SRC = _PROJECT_ROOT / "tools" / "spec_trainer" / "src"
if str(_SPEC_SRC) not in sys.path:
    sys.path.insert(0, str(_SPEC_SRC))

from cv_core import (  # noqa: E402
    METRIC_SEMANTICS_VERSION, evaluate_at_threshold, evaluate_oof,
    evaluate_ranking, threshold_at_fpr,
)
from cv_train import _split_counts, _validate_split_counts  # noqa: E402


CONFIG_SCHEMA = "phase2_xic_training_config_v1"
_EXPECTED_SEMANTICS = {
    "positive_class": "incorrect_identification",
    "stored_label": "1=correct_identification, 0=incorrect_identification",
    "model_score": "trust_score=P(correct_identification)",
    "metric_score": "error_score=1-trust_score",
}


def _atomic_json(path: Path, value) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_name(f".{path.name}.tmp.{os.getpid()}")
    temporary.write_text(
        json.dumps(value, indent=2, ensure_ascii=False) + "\n",
        encoding="utf-8")
    os.replace(temporary, path)


def _atomic_csv(path: Path, frame: pd.DataFrame) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_name(f".{path.name}.tmp.{os.getpid()}")
    frame.to_csv(temporary, index=False)
    os.replace(temporary, path)


def _atomic_yaml(path: Path, value) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_name(f".{path.name}.tmp.{os.getpid()}")
    temporary.write_text(
        yaml.safe_dump(value, sort_keys=False, allow_unicode=True),
        encoding="utf-8")
    os.replace(temporary, path)


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def _close_staging_log_handlers(staging: Path) -> None:
    """Seal file logs before checksumming or deleting their staging tree."""
    root = logging.getLogger()
    for handler in list(root.handlers):
        if not isinstance(handler, logging.FileHandler):
            continue
        filename = Path(handler.baseFilename).resolve()
        if filename != staging.resolve() and staging.resolve() not in \
                filename.parents:
            continue
        handler.flush()
        root.removeHandler(handler)
        handler.close()


def _finalize_bundle(staging: Path) -> None:
    """Checksum every result artifact and anchor it with COMPLETE."""
    artifacts = sorted(
        path for path in staging.rglob("*")
        if path.is_file() and path.name not in {"checksums.json", "COMPLETE"})
    checksums = {
        str(path.relative_to(staging)): {
            "sha256": _sha256(path), "size_bytes": path.stat().st_size,
        }
        for path in artifacts
    }
    checksums_path = staging / "checksums.json"
    _atomic_json(checksums_path, checksums)
    _atomic_json(staging / "COMPLETE", {
        "schema": "phase2_xic_result_bundle_v1",
        "status": json.loads(
            (staging / "bundle_status.json").read_text(
                encoding="utf-8"))["status"],
        "checksums_sha256": _sha256(checksums_path),
        "n_artifacts": len(checksums),
    })


def _verify_complete_bundle(root: Path) -> None:
    complete_path = root / "COMPLETE"
    checksums_path = root / "checksums.json"
    if not complete_path.is_file() or not checksums_path.is_file():
        raise ValueError(f"incomplete Phase 2 result bundle: {root}")
    complete = json.loads(complete_path.read_text(encoding="utf-8"))
    if complete.get("schema") != "phase2_xic_result_bundle_v1" or \
            complete.get("checksums_sha256") != _sha256(checksums_path):
        raise ValueError(f"invalid Phase 2 result COMPLETE marker: {root}")
    checksums = json.loads(checksums_path.read_text(encoding="utf-8"))
    if complete.get("n_artifacts") != len(checksums):
        raise ValueError("Phase 2 result artifact count is inconsistent")
    actual = {
        str(path.relative_to(root)) for path in root.rglob("*")
        if path.is_file() and path.name not in {"checksums.json", "COMPLETE"}
    }
    if actual != set(checksums):
        raise ValueError(
            "Phase 2 result checksum coverage differs from artifacts: "
            f"missing={sorted(set(checksums) - actual)}, "
            f"unexpected={sorted(actual - set(checksums))}")
    for relative, expected in checksums.items():
        path = root / relative
        if not path.is_file() or path.stat().st_size != expected["size_bytes"]:
            raise ValueError(f"Phase 2 result artifact changed: {relative}")
        if _sha256(path) != expected["sha256"]:
            raise ValueError(f"Phase 2 result checksum mismatch: {relative}")


def _recover_publish(output_root: Path, *, cleanup_stale: bool) -> None:
    backups = sorted(output_root.parent.glob(
        f".{output_root.name}.backup.*"))
    if output_root.exists():
        _verify_complete_bundle(output_root)
        if backups and cleanup_stale:
            for backup in backups:
                shutil.rmtree(backup)
            logging.warning(
                "removed %d stale Phase 2 result backup(s) after verifying "
                "the current bundle", len(backups))
        elif backups:
            logging.warning(
                "complete Phase 2 result coexists with %d stale backup(s): %s",
                len(backups), ", ".join(str(path) for path in backups))
        return
    valid = []
    for backup in backups:
        try:
            _verify_complete_bundle(backup)
            valid.append(backup)
        except (OSError, ValueError, json.JSONDecodeError):
            logging.warning(
                "ignored invalid Phase 2 result backup: %s", backup)
    if len(valid) == 1:
        os.replace(valid[0], output_root)
        logging.warning(
            "restored Phase 2 result after interrupted publish: %s",
            output_root)
    elif len(valid) > 1:
        raise RuntimeError(
            "cannot recover Phase 2 result publish unambiguously: "
            f"{[str(path) for path in valid]}")


def _validate_config(config: dict) -> None:
    if not isinstance(config, dict) or config.get("schema") != CONFIG_SCHEMA:
        raise ValueError(f"Phase 2 config requires schema={CONFIG_SCHEMA}")
    if config.get("evaluation_semantics") != _EXPECTED_SEMANTICS:
        raise ValueError(
            "evaluation_semantics must exactly preserve the canonical "
            "incorrect-identification-positive convention")
    if config.get("model", {}).get("type") != "xic_fusion_attention_v2":
        raise ValueError(
            "Phase 2 config requires model.type=xic_fusion_attention_v2")
    if config.get("training", {}).get("early_stopping_metric") != "roc_auc":
        raise ValueError("Phase 2 early_stopping_metric must be roc_auc")
    if str(config.get("training", {}).get(
            "class_weighting", "none")).lower() != "none":
        raise ValueError(
            "controlled Phase 2 comparison requires class_weighting=none")
    seeds = config.get("training", {}).get("seeds")
    if not isinstance(seeds, list) or not seeds or \
            len(seeds) != len(set(seeds)) or \
            not all(isinstance(seed, int) for seed in seeds):
        raise ValueError("training.seeds must be a nonempty unique integer list")


def _model_names(config: dict, available) -> list[str]:
    values = config.get("experiment", {}).get(
        "negative_pool_models", ["M20"])
    result = [str(value).upper() for value in values]
    if not result or len(result) != len(set(result)) or \
            not set(result).issubset(available):
        raise ValueError(
            f"negative_pool_models must be a unique subset of "
            f"{sorted(available)}")
    return result


def _validation_score(labels, trust_scores) -> float:
    return float(evaluate_ranking(labels, trust_scores)["roc_auc"])


def _member_seed(experiment_seed: int, fold: int) -> int:
    sequence = np.random.SeedSequence([int(experiment_seed), int(fold)])
    return int(sequence.generate_state(1, dtype=np.uint32)[0])


def _assert_fold(protocol, train: pd.DataFrame, fold: int,
                 outer_test: np.ndarray, inner_valid: np.ndarray,
                 inner_train: np.ndarray, config: dict) -> dict:
    if not outer_test.any() or not inner_valid.any() or not inner_train.any():
        raise ValueError(f"Phase 2 fold {fold} contains an empty split")
    group_col = protocol.prepared.group_col
    target_col = protocol.prepared.target_col
    groups = train[group_col]
    group_sets = [
        set(groups[mask]) for mask in (inner_train, inner_valid, outer_test)
    ]
    if any(group_sets[left].intersection(group_sets[right])
           for left in range(3) for right in range(left + 1, 3)):
        raise ValueError(f"Phase 2 fold {fold} has leakage-group overlap")
    counts = {
        "train": _split_counts(
            train[target_col], groups, np.flatnonzero(inner_train)),
        "valid": _split_counts(
            train[target_col], groups, np.flatnonzero(inner_valid)),
        "oof_test": _split_counts(
            train[target_col], groups, np.flatnonzero(outer_test)),
    }
    _validate_split_counts(
        fold, counts,
        int(config["training"].get("min_class_groups_per_split", 1)),
        grouped=True)
    return counts


def _fit_one_pool(protocol, model_name: str, test: pd.DataFrame,
                  config: dict, root: Path, *, seed: int,
                  result_name: str) -> dict:
    train = protocol.training_frame(model_name)
    target_col = protocol.prepared.target_col
    fold_col = protocol.prepared.outer_fold_col
    labels = train[target_col].to_numpy(dtype=int)
    test_labels = test[target_col].to_numpy(dtype=int)
    source_indices = train["source_index"].to_numpy(dtype=int)
    test_indices = test["source_index"].to_numpy(dtype=int)
    folds = train[fold_col].to_numpy(dtype=int)
    n_folds = len(protocol.prepared.inner_valid_cols)
    if set(np.unique(folds)) != set(range(n_folds)):
        raise ValueError(f"{model_name} has incomplete Phase 2 outer folds")

    oof = np.full(len(train), np.nan, dtype="f8")
    member_test_scores = []
    thresholds = {
        target: [] for target in protocol.prepared.target_fprs}
    fold_summaries = []
    include_prediction = bool(
        config["model"].get("include_predicted_intensity", False))
    inference_batch_size = int(
        config["training"].get("inference_batch_size", 256))
    num_workers = int(config["training"].get("num_workers", 0))
    dataset_identity = dict(protocol.validation["build_contract"])

    for fold in range(n_folds):
        outer_test = folds == fold
        inner_valid = train[
            protocol.prepared.inner_valid_cols[fold]].to_numpy(dtype=bool)
        inner_train = (~outer_test) & (~inner_valid)
        counts = _assert_fold(
            protocol, train, fold, outer_test, inner_valid, inner_train,
            config)
        fold_seed = _member_seed(seed, fold)
        fitted = fit_xic_model(
            protocol.source,
            source_indices[inner_train], source_indices[inner_valid], config,
            validation_score=_validation_score, seed=fold_seed)
        oof_dataset = XICDataset(
            protocol.source, source_indices[outer_test],
            include_predicted_intensity=include_prediction)
        test_dataset = XICDataset(
            protocol.source, test_indices,
            include_predicted_intensity=include_prediction)
        oof_scores = predict_trust(
            fitted.model, oof_dataset, batch_size=inference_batch_size,
            device=fitted.device, num_workers=num_workers)
        test_scores = predict_trust(
            fitted.model, test_dataset, batch_size=inference_batch_size,
            device=fitted.device, num_workers=num_workers)
        oof[outer_test] = oof_scores
        member_test_scores.append(test_scores)

        calibration = {}
        for target in protocol.prepared.target_fprs:
            error_threshold = threshold_at_fpr(
                labels[outer_test], oof_scores, target)
            thresholds[target].append(error_threshold)
            point = evaluate_at_threshold(
                labels[outer_test], oof_scores, error_threshold)
            point.update({
                "target_fpr": float(target),
                "threshold_source": "this_model_outer_oof_fold",
            })
            calibration[f"fpr_{int(round(target * 100))}"] = point

        relative_checkpoint = Path("models") / (
            f"{result_name.lower()}.fold{fold}.pt")
        save_checkpoint(
            root / relative_checkpoint, fitted,
            dataset_identity=dataset_identity,
            metadata={
                "fold": fold,
                "negative_pool_model": model_name,
                "training_seed": seed,
                "fold_seed": fold_seed,
                "metric_semantics": METRIC_SEMANTICS_VERSION,
                "positive_class": "incorrect_identification",
                "model_score": "trust_score=P(correct_identification)",
                "best_epoch": fitted.best_epoch,
                "best_validation_roc_auc": fitted.best_validation_score,
                "training_sample_ids_sha256": hashlib.sha256("\n".join(
                    sorted(train.loc[inner_train,
                                     protocol.prepared.sample_id_col].astype(str))
                ).encode("utf-8")).hexdigest(),
                "validation_sample_ids_sha256": hashlib.sha256("\n".join(
                    sorted(train.loc[inner_valid,
                                     protocol.prepared.sample_id_col].astype(str))
                ).encode("utf-8")).hexdigest(),
                "training_config_sha256": hashlib.sha256(json.dumps(
                    config, sort_keys=True, separators=(",", ":")
                ).encode("utf-8")).hexdigest(),
            })
        fold_metrics = evaluate_oof(labels[outer_test], oof_scores)
        fold_summaries.append({
            "fold": fold,
            "fold_seed": fold_seed,
            "n_train": int(inner_train.sum()),
            "n_valid": int(inner_valid.sum()),
            "n_oof": int(outer_test.sum()),
            "split_counts": counts,
            "best_epoch": fitted.best_epoch,
            "best_validation_roc_auc": fitted.best_validation_score,
            "n_trainable_parameters": n_trainable_parameters(fitted.model),
            "checkpoint": str(relative_checkpoint),
            "roc_auc": fold_metrics["roc_auc"],
            "error_pr_auc": fold_metrics["error_pr_auc"],
            "fnr_at_fpr5": fold_metrics["fnr_at_fpr5"],
            "error_recall_at_fpr10": fold_metrics[
                "error_recall_at_fpr10"],
            "calibration_operating_points": calibration,
        })

    if np.isnan(oof).any():
        raise AssertionError("Phase 2 left training rows without OOF scores")
    member_test_scores = np.vstack(member_test_scores)
    trust = member_test_scores.mean(axis=0)
    votes = {}
    operating_points = {}
    for target, member_thresholds in thresholds.items():
        vote_fraction = np.vstack([
            (1.0 - member_score) >= error_threshold
            for member_score, error_threshold in zip(
                member_test_scores, member_thresholds)
        ]).mean(axis=0)
        key = f"fpr_{int(round(target * 100))}"
        votes[key] = vote_fraction
        operating_points[key] = {
            "method": "fold_calibrated_majority_vote",
            "calibration_source": "each_member_outer_oof_fold",
            "member_error_thresholds": [
                float(value) for value in member_thresholds],
            "vote_error_threshold": 0.5,
            "external_ensemble": {
                "method": "fold_calibrated_majority_vote",
                "test_metrics": evaluate_at_threshold(
                    test_labels, 1.0 - vote_fraction, 0.5),
            },
        }

    identity_columns = list(dict.fromkeys(
        column for column in (
            protocol.prepared.sample_id_col, protocol.prepared.dataset_col,
            protocol.prepared.target_col, protocol.prepared.tier_col,
            protocol.prepared.group_col,
        ) if column in train))
    oof_frame = train[identity_columns].copy()
    oof_frame["outer_fold"] = folds
    oof_frame["trust_score"] = oof
    oof_frame["error_score"] = 1.0 - oof
    _atomic_csv(
        root / "predictions" / f"{result_name.lower()}_train_oof.csv",
        oof_frame)
    fixed_metrics = {
        **evaluate_ranking(test_labels, trust),
        "fnr_at_fpr5": operating_points["fpr_5"][
            "external_ensemble"]["test_metrics"]["fnr"],
        "error_recall_at_fpr10": operating_points["fpr_10"][
            "external_ensemble"]["test_metrics"]["error_recall"],
        "operating_points": operating_points,
    }
    return {
        "summary": {
            "n_train": len(train),
            "n_actual_correct_train": int((labels == 1).sum()),
            "n_actual_error_train": int((labels == 0).sum()),
            "oof_metrics": evaluate_oof(labels, oof),
            "fixed_test_metrics": fixed_metrics,
            "architecture": fitted.model.architecture(),
            "training_seed": seed,
            "n_ensemble_members": n_folds,
            "model_input_policy": {
                "raw_xic_signal_native": True,
                "protocol_metadata_excluded": True,
                "predicted_intensity_enabled": include_prediction,
            },
            "folds": fold_summaries,
        },
        "trust_score": trust,
        "vote_fractions": votes,
        "member_thresholds": thresholds,
    }


def _fixed_metrics(labels, values) -> dict:
    operating = {}
    for key in ("fpr_1", "fpr_5", "fpr_10"):
        vote_key = f"{key}_vote_fraction"
        if vote_key not in values:
            continue
        operating[key] = {
            "method": "fold_calibrated_majority_vote",
            "external_ensemble": {
                "method": "fold_calibrated_majority_vote",
                "test_metrics": evaluate_at_threshold(
                    labels, 1.0 - values[vote_key], 0.5),
            },
        }
    return {
        **evaluate_ranking(labels, values["trust_score"]),
        "fnr_at_fpr5": operating["fpr_5"][
            "external_ensemble"]["test_metrics"]["fnr"],
        "error_recall_at_fpr10": operating["fpr_10"][
            "external_ensemble"]["test_metrics"]["error_recall"],
        "operating_points": operating,
    }


def _prediction_values(result: dict) -> dict:
    return {
        "trust_score": result["trust_score"],
        **{
            f"{key}_vote_fraction": value
            for key, value in result["vote_fractions"].items()
        },
    }


def _flat_row(name: str, labels: np.ndarray, metrics: dict) -> dict:
    fpr5 = metrics["operating_points"]["fpr_5"][
        "external_ensemble"]["test_metrics"]
    fpr10 = metrics["operating_points"]["fpr_10"][
        "external_ensemble"]["test_metrics"]
    return {
        "metric_semantics": METRIC_SEMANTICS_VERSION,
        "positive_class": "incorrect_identification",
        "model": name,
        "test_subset": "full_E20",
        "n_rows": len(labels),
        "n_actual_correct": int((labels == 1).sum()),
        "n_actual_error": int((labels == 0).sum()),
        "roc_auc": metrics["roc_auc"],
        "error_pr_auc": metrics["error_pr_auc"],
        "fnr_at_fpr5": fpr5["fnr"],
        "observed_fpr_at_fpr5": fpr5["fpr"],
        "error_recall_at_fpr10": fpr10["error_recall"],
        "observed_fpr_at_fpr10": fpr10["fpr"],
    }


def _seed_ensemble(results: list[dict], labels: np.ndarray) -> dict:
    trust = np.mean([result["trust_score"] for result in results], axis=0)
    keys = list(results[0]["vote_fractions"])
    votes = {
        key: np.mean([
            result["vote_fractions"][key] for result in results
        ], axis=0)
        for key in keys
    }
    values = {
        "trust_score": trust,
        **{f"{key}_vote_fraction": value for key, value in votes.items()},
    }
    return {
        "trust_score": trust,
        "vote_fractions": votes,
        "fixed_test_metrics": _fixed_metrics(labels, values),
        "n_seed_members": len(results),
        "n_fold_members": sum(
            len(result["summary"]["folds"]) for result in results),
        "note": (
            "ranking averages all seed/fold trust scores; formal decisions "
            "majority-vote all fold-local outer-OOF-calibrated members"),
    }


def _generalization_audit(protocol) -> dict:
    prepared = protocol.prepared
    train = protocol.frame[protocol.frame[prepared.split_col].eq("train")]
    test = protocol.frame[protocol.frame[prepared.split_col].eq("test")]
    train_sequences = set(train[prepared.base_group_col].astype(str))
    test_sequences = set(test[prepared.base_group_col].astype(str))
    overlap = train_sequences.intersection(test_sequences)
    return {
        "holdout_type": "internal_fixed_E20_holdout",
        "is_external_entrapment_test": False,
        "train_unique_sequences": len(train_sequences),
        "test_unique_sequences": len(test_sequences),
        "overlap_unique_sequences": len(overlap),
        "allowed_generalization_claim": (
            "internal_unseen_sequence_holdout"
            if not overlap else "internal_domain_holdout_with_sequence_overlap"),
    }


def _provenance(config_path: Path, split_config_path: Path,
                signal_root: Path, protocol_root: Path) -> dict:
    try:
        commit = subprocess.run(
            ["git", "rev-parse", "HEAD"], cwd=_PROJECT_ROOT,
            check=True, capture_output=True, text=True).stdout.strip()
        dirty = bool(subprocess.run(
            ["git", "status", "--porcelain", "--untracked-files=no"],
            cwd=_PROJECT_ROOT, check=True, capture_output=True,
            text=True).stdout.strip())
    except (OSError, subprocess.CalledProcessError):
        commit, dirty = None, None
    return {
        "created_at_utc": dt.datetime.now(dt.timezone.utc).isoformat(),
        "git_commit": commit,
        "git_tracked_files_dirty": dirty,
        "python": platform.python_version(),
        "torch": torch.__version__,
        "numpy": np.__version__,
        "pandas": pd.__version__,
        "cuda_available": torch.cuda.is_available(),
        "config": {"path": str(config_path), "sha256": _sha256(config_path)},
        "split_config": {
            "path": str(split_config_path),
            "sha256": _sha256(split_config_path),
        },
        "signal_root": str(signal_root),
        "signal_complete_sha256": _sha256(signal_root / "COMPLETE"),
        "protocol_root": str(protocol_root),
        "protocol_summary_sha256": _sha256(protocol_root / "summary.json"),
    }


def _publish(staging: Path, output_root: Path, overwrite: bool) -> None:
    _verify_complete_bundle(staging)
    status = json.loads(
        (staging / "bundle_status.json").read_text(encoding="utf-8"))
    if status.get("status") not in {"complete", "prepare_only"}:
        raise ValueError("refusing to publish incomplete Phase 2 results")
    backup = None
    if output_root.exists():
        if not overwrite:
            raise FileExistsError(f"output root already exists: {output_root}")
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
                "published Phase 2 result but could not remove backup: %s",
                backup, exc_info=True)


def _run_staging(config_path: Path, split_config_path: Path,
                 feature_root: str, protocol_root: Path,
                 signal_root: Path, staging: Path, final_root: Path,
                 *, prepare_only: bool) -> dict:
    config = yaml.safe_load(config_path.read_text(encoding="utf-8"))
    _validate_config(config)
    split_config = yaml.safe_load(
        split_config_path.read_text(encoding="utf-8"))
    _atomic_yaml(staging / "config_used.yaml", config)
    _atomic_yaml(staging / "split_config_used.yaml", split_config)
    log_path = staging / "logs" / "train.log"
    log_path.parent.mkdir(parents=True, exist_ok=True)
    logging.basicConfig(
        level=logging.INFO, force=True,
        format="%(asctime)s %(levelname)s %(message)s",
        handlers=[
            logging.FileHandler(log_path, mode="w", encoding="utf-8"),
            logging.StreamHandler(),
        ])

    protocol = prepare_xic_protocol(
        str(signal_root), str(split_config_path), feature_root,
        str(protocol_root))
    include_prediction = bool(
        config["model"].get("include_predicted_intensity", False))
    if include_prediction and not protocol.validation[
            "build_contract"]["prediction_included"]:
        raise ValueError(
            "prediction-enabled model requires an XIC dataset built with "
            "prediction.include=true")
    _atomic_json(staging / "preflight.json", protocol.validation)
    manifest_columns = list(dict.fromkeys(
        column for column in (
            protocol.prepared.sample_id_col,
            protocol.prepared.dataset_col,
            protocol.prepared.target_col,
            protocol.prepared.tier_col,
            protocol.prepared.split_col,
            protocol.prepared.outer_fold_col,
            protocol.prepared.group_col,
            *protocol.prepared.inner_valid_cols,
            "source_index",
        ) if column in protocol.frame))
    _atomic_csv(
        staging / "manifests" / "xic_frozen_membership.csv",
        protocol.frame[manifest_columns])
    if prepare_only:
        result = {
            "mode": "prepare_only", **protocol.validation,
        }
        _atomic_json(staging / "bundle_status.json", {
            "status": "prepare_only",
            "metric_semantics": METRIC_SEMANTICS_VERSION,
            "positive_class": "incorrect_identification",
        })
        _close_staging_log_handlers(staging)
        _finalize_bundle(staging)
        return result

    test = protocol.test_frame()
    labels = test[protocol.prepared.target_col].to_numpy(dtype=int)
    identity = list(dict.fromkeys(
        column for column in (
            protocol.prepared.sample_id_col,
            protocol.prepared.dataset_col,
            protocol.prepared.target_col,
            protocol.prepared.tier_col,
            protocol.prepared.group_col,
        ) if column in test))
    predictions = test[identity].copy()
    model_results = {}
    model_predictions = {}
    flat_rows = []
    model_names = _model_names(config, protocol.prepared.model_tiers)

    baseline = load_frozen_lightgbm_predictions(
        protocol.prepared.protocol_root, test,
        protocol.prepared.sample_id_col, model_names)
    for name, values in baseline.items():
        fixed = _fixed_metrics(labels, values)
        model_results[name] = {
            "source": "frozen_lightgbm_protocol_bundle",
            "fixed_test_metrics": fixed,
        }
        model_predictions[name] = values
        flat_rows.append(_flat_row(name, labels, fixed))
        predictions[f"{name}_trust_score"] = values["trust_score"]
        predictions[f"{name}_error_score"] = 1.0 - values["trust_score"]
        for key in ("fpr_5", "fpr_10"):
            predictions[f"{name}_{key}_error_vote_fraction"] = values[
                f"{key}_vote_fraction"]

    comparison_pairs = []
    for model_name in model_names:
        seed_results = []
        for seed in config["training"]["seeds"]:
            result_name = f"XIC_{model_name}_seed{seed}"
            logging.info(
                "training Phase 2 XIC model pool=%s seed=%d",
                model_name, seed)
            result = _fit_one_pool(
                protocol, model_name, test, config, staging,
                seed=seed, result_name=result_name)
            seed_results.append(result)
            values = _prediction_values(result)
            model_predictions[result_name] = values
            model_results[result_name] = result["summary"]
            fixed = result["summary"]["fixed_test_metrics"]
            flat_rows.append(_flat_row(result_name, labels, fixed))
            predictions[f"{result_name}_trust_score"] = values["trust_score"]
            predictions[f"{result_name}_error_score"] = 1.0 - values[
                "trust_score"]
            for key in result["vote_fractions"]:
                predictions[
                    f"{result_name}_{key}_error_vote_fraction"
                ] = result["vote_fractions"][key]

        ensemble_name = f"XIC_{model_name}_ensemble"
        ensemble = _seed_ensemble(seed_results, labels)
        ensemble_values = {
            "trust_score": ensemble["trust_score"],
            **{
                f"{key}_vote_fraction": value
                for key, value in ensemble["vote_fractions"].items()
            },
        }
        model_predictions[ensemble_name] = ensemble_values
        model_results[ensemble_name] = {
            key: value for key, value in ensemble.items()
            if key not in {"trust_score", "vote_fractions"}
        }
        flat_rows.append(_flat_row(
            ensemble_name, labels, ensemble["fixed_test_metrics"]))
        predictions[f"{ensemble_name}_trust_score"] = ensemble["trust_score"]
        predictions[f"{ensemble_name}_error_score"] = 1.0 - ensemble[
            "trust_score"]
        for key, value in ensemble["vote_fractions"].items():
            predictions[
                f"{ensemble_name}_{key}_error_vote_fraction"] = value
        baseline_name = f"LightGBM_{model_name}"
        comparison_pairs.append((baseline_name, ensemble_name))
        for seed in config["training"]["seeds"]:
            comparison_pairs.append((
                baseline_name, f"XIC_{model_name}_seed{seed}"))

    bootstrap_cfg = config.get("comparison", {})
    bootstrap = paired_cluster_bootstrap(
        test, model_predictions, comparison_pairs,
        reps=int(bootstrap_cfg.get("bootstrap_reps", 1000)),
        seed=int(bootstrap_cfg.get("bootstrap_seed", 20260813)),
        group_col=protocol.prepared.group_col,
        target_col=protocol.prepared.target_col,
        evaluate_ranking=evaluate_ranking,
        evaluate_at_threshold=evaluate_at_threshold)

    domain_rows = []
    for name, values in model_predictions.items():
        for domain in sorted(test[protocol.prepared.dataset_col].unique()):
            mask = test[protocol.prepared.dataset_col].eq(domain).to_numpy()
            subset = {
                key: np.asarray(value)[mask] for key, value in values.items()
            }
            row = _flat_row(name, labels[mask], _fixed_metrics(
                labels[mask], subset))
            row["test_dataset"] = domain
            domain_rows.append(row)

    _atomic_csv(
        staging / "predictions" / "fixed_test_predictions.csv", predictions)
    _atomic_csv(
        staging / "fixed_test_summary.csv", pd.DataFrame(flat_rows))
    _atomic_csv(staging / "paired_model_bootstrap.csv", bootstrap)
    _atomic_csv(staging / "domain_summary.csv", pd.DataFrame(domain_rows))
    summary = {
        "metric_semantics": METRIC_SEMANTICS_VERSION,
        "positive_class": "incorrect_identification",
        "experiment": "phase2_signal_native_xic_frozen_protocol_v1",
        "dataset": "combined",
        "model_score": "trust_score=P(correct_identification)",
        "signal_dataset": {
            "root": str(signal_root),
            "schema": protocol.source.schema["schema"],
            "checksums_sha256": protocol.source.complete[
                "checksums_sha256"],
            "prediction_arm_enabled": include_prediction,
        },
        "split_contract": {
            "source": protocol.prepared.protocol_root,
            "membership": "loaded_and_value-checked_against_frozen_manifest",
            "folds": "loaded_and_value-checked_against_frozen_fold_map",
            "holdout": "internal_fixed_E20_not_external_entrapment",
        },
        "generalization_audit": _generalization_audit(protocol),
        "model_input_contract": {
            "source": "raw_light_heavy_precursor_and_fragment_XIC",
            "metadata_as_model_input": False,
            "theoretical_fragment_context": "ion_type_and_charge_only",
            "fragment_ordinal_as_model_input": False,
            "fragment_count_as_model_input": False,
            "fragment_eligibility_gate": (
                "real_scan_and_separable_and_attempted"),
            "input_adapter": input_adapter_contract(
                include_predicted_intensity=include_prediction),
            "predicted_intensity_enabled": include_prediction,
            "trust_score": "P(correct_identification)",
        },
        "models": model_results,
        "paired_comparison": {
            "resampling_unit": protocol.prepared.group_col,
            "n_requested": int(bootstrap_cfg.get("bootstrap_reps", 1000)),
            "seed": int(bootstrap_cfg.get("bootstrap_seed", 20260813)),
            "note": (
                "paired bootstrap intervals use the identical fixed test; "
                "fold/seed dispersion is not an independent confidence interval"),
        },
        "provenance": _provenance(
            config_path, split_config_path, signal_root, protocol_root),
        "artifacts": {
            "preflight": str(final_root / "preflight.json"),
            "fixed_test_summary": str(final_root / "fixed_test_summary.csv"),
            "paired_model_bootstrap": str(
                final_root / "paired_model_bootstrap.csv"),
            "domain_summary": str(final_root / "domain_summary.csv"),
            "fixed_test_predictions": str(
                final_root / "predictions" / "fixed_test_predictions.csv"),
            "membership": str(
                final_root / "manifests" / "xic_frozen_membership.csv"),
        },
    }
    _atomic_json(staging / "summary.json", summary)
    _atomic_json(staging / "bundle_status.json", {
        "status": "complete",
        "metric_semantics": METRIC_SEMANTICS_VERSION,
        "positive_class": "incorrect_identification",
    })
    _close_staging_log_handlers(staging)
    _finalize_bundle(staging)
    return summary


def run_experiment(config_path: str, split_config_path: str,
                   feature_root: str, protocol_root: str, signal_root: str,
                   output_root: str, *, overwrite: bool = False,
                   prepare_only: bool = False) -> dict:
    config_path_obj = Path(config_path).resolve()
    split_config_obj = Path(split_config_path).resolve()
    protocol_root_obj = Path(protocol_root).resolve()
    signal_root_obj = Path(signal_root).resolve()
    output_root_obj = Path(output_root).resolve()
    for path in (config_path_obj, split_config_obj):
        if not path.is_file():
            raise FileNotFoundError(path)
    output_root_obj.parent.mkdir(parents=True, exist_ok=True)
    _recover_publish(output_root_obj, cleanup_stale=overwrite)
    if output_root_obj.exists() and not overwrite:
        raise FileExistsError(
            f"refusing to overwrite Phase 2 results: {output_root_obj}")
    staging = Path(tempfile.mkdtemp(
        prefix=f".{output_root_obj.name}.staging.",
        dir=output_root_obj.parent))
    try:
        result = _run_staging(
            config_path_obj, split_config_obj, feature_root,
            protocol_root_obj, signal_root_obj, staging, output_root_obj,
            prepare_only=prepare_only)
        _publish(staging, output_root_obj, overwrite)
        return result
    except BaseException:
        _close_staging_log_handlers(staging)
        shutil.rmtree(staging, ignore_errors=True)
        raise


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Train Phase 2 signal-native XIC models")
    parser.add_argument("--config", required=True)
    parser.add_argument("--split-config", required=True)
    parser.add_argument("--feature-root", required=True)
    parser.add_argument("--protocol-root", required=True)
    parser.add_argument("--signal-root", required=True)
    parser.add_argument("--output-root", required=True)
    parser.add_argument("--overwrite", action="store_true")
    parser.add_argument("--prepare-only", action="store_true")
    args = parser.parse_args()
    run_experiment(
        args.config, args.split_config, args.feature_root,
        args.protocol_root, args.signal_root, args.output_root,
        overwrite=args.overwrite, prepare_only=args.prepare_only)


if __name__ == "__main__":
    main()
