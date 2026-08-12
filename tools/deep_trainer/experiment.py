"""Grouped-OOF tabular MLP on the existing immutable E20 fixed test.

The public seam is ``run_experiment``.  Dataset membership, feature selection,
cohort filters, outer folds and inner validation folds come from spec_trainer's
fixed-negpool protocol.  This module changes only the model implementation.
"""

from __future__ import annotations

import argparse
import datetime as dt
import hashlib
import json
import logging
import os
import platform
from pathlib import Path
import shutil
import subprocess
import sys
import tempfile
import uuid

import numpy as np
import pandas as pd
import torch
import yaml

from .checkpoint import save_checkpoint
from .comparison import (
    load_frozen_lightgbm_predictions,
    missingness_audit,
    paired_cluster_bootstrap,
    summarize_missingness,
)
from .model import n_trainable_parameters
from .preprocessing import FoldPreprocessor
from .spec_adapter import prepare_protocol
from .training import fit_mlp, predict_trust


_PROJECT_ROOT = Path(__file__).resolve().parents[2]
_SPEC_SRC = _PROJECT_ROOT / "tools" / "spec_trainer" / "src"
if str(_SPEC_SRC) not in sys.path:
    sys.path.insert(0, str(_SPEC_SRC))

from cv_core import (  # noqa: E402
    METRIC_SEMANTICS_VERSION,
    evaluate_at_threshold,
    evaluate_oof,
    evaluate_ranking,
    threshold_at_fpr,
)
from cv_train import _split_counts, _validate_split_counts  # noqa: E402


_EXPECTED_EVALUATION_SEMANTICS = {
    "positive_class": "incorrect_identification",
    "stored_label": "1=correct_identification, 0=incorrect_identification",
    "model_score": "trust_score=P(correct_identification)",
    "metric_score": "error_score=1-trust_score",
}


def _atomic_json(path, value):
    path = os.fspath(path)
    os.makedirs(os.path.dirname(path) or ".", exist_ok=True)
    temporary = f"{path}.tmp.{os.getpid()}"
    with open(temporary, "w", encoding="utf-8") as handle:
        json.dump(value, handle, indent=2, ensure_ascii=False)
        handle.write("\n")
    os.replace(temporary, path)


def _atomic_csv(path, frame):
    path = os.fspath(path)
    os.makedirs(os.path.dirname(path) or ".", exist_ok=True)
    temporary = f"{path}.tmp.{os.getpid()}"
    frame.to_csv(temporary, index=False)
    os.replace(temporary, path)


def _atomic_yaml(path, value):
    path = os.fspath(path)
    os.makedirs(os.path.dirname(path) or ".", exist_ok=True)
    temporary = f"{path}.tmp.{os.getpid()}"
    with open(temporary, "w", encoding="utf-8") as handle:
        yaml.safe_dump(value, handle, sort_keys=False, allow_unicode=True)
    os.replace(temporary, path)


def _known_outputs(root):
    root = Path(root)
    return [
        root / "summary.json",
        root / "preflight.json",
        root / "config_used.yaml",
        root / "split_config_used.yaml",
        root / "fixed_test_summary.csv",
        root / "domain_summary.csv",
        root / "predictions" / "fixed_test_predictions.csv",
        root / "manifests" / "membership.csv",
        root / "manifests" / "fixed_test_manifest.csv",
        root / "manifests" / "fold_map.csv",
    ]


def _assert_output_available(root, overwrite):
    root = Path(root)
    existing = [path for path in _known_outputs(root) if path.exists()]
    if existing and not overwrite:
        raise FileExistsError(
            "refusing to overwrite an existing deep-trainer result bundle; "
            "choose a new output root or pass --overwrite:\n  "
            + "\n  ".join(map(str, existing)))
    if root.exists() and any(root.iterdir()) and not overwrite:
        raise FileExistsError(
            f"refusing to replace nonempty output directory: {root}")


def _publish_bundle(staging, root, overwrite):
    """Publish a complete staging bundle, restoring the old one on failure."""
    staging, root = Path(staging), Path(root)
    status_path = staging / "bundle_status.json"
    if not status_path.is_file():
        raise ValueError("refusing to publish an incomplete result bundle")
    status = json.loads(status_path.read_text(encoding="utf-8"))
    required = [
        staging / "preflight.json",
        staging / "config_used.yaml",
        staging / "split_config_used.yaml",
        staging / "manifests" / "membership.csv",
        staging / "manifests" / "fixed_test_manifest.csv",
        staging / "manifests" / "fold_map.csv",
    ]
    if status.get("status") == "complete":
        required.extend([
            staging / "summary.json",
            staging / "fixed_test_summary.csv",
            staging / "predictions" / "fixed_test_predictions.csv",
        ])
    elif status.get("status") != "prepare_only":
        raise ValueError(f"unknown bundle status: {status.get('status')!r}")
    missing = [str(path) for path in required if not path.is_file()]
    if missing:
        raise ValueError(
            "refusing to publish a bundle with missing artifacts:\n  "
            + "\n  ".join(missing))
    backup = None
    if root.exists():
        if not overwrite and any(root.iterdir()):
            raise FileExistsError(f"output directory is not empty: {root}")
        backup = root.with_name(f".{root.name}.backup.{uuid.uuid4().hex}")
        os.replace(root, backup)
    try:
        os.replace(staging, root)
    except Exception:
        if backup is not None and backup.exists() and not root.exists():
            os.replace(backup, root)
        raise
    if backup is not None and backup.exists():
        try:
            shutil.rmtree(backup)
        except OSError as exc:
            logging.warning("published bundle but could not remove %s: %s",
                            backup, exc)


def _validate_deep_config(config):
    """Reject silent drift from the controlled baseline contract."""
    if not isinstance(config, dict):
        raise ValueError("deep-trainer config must be a mapping")
    if config.get("evaluation_semantics") != _EXPECTED_EVALUATION_SEMANTICS:
        raise ValueError(
            "evaluation_semantics must exactly match the repository's "
            "error-identification-positive convention")
    if config.get("model", {}).get("type") != "tabular_mlp_v1":
        raise ValueError("phase-1 deep trainer requires model.type=tabular_mlp_v1")
    if config.get("training", {}).get("early_stopping_metric") != "roc_auc":
        raise ValueError(
            "controlled comparison requires "
            "training.early_stopping_metric=roc_auc")
    weighting = str(
        config.get("training", {}).get("class_weighting", "none")
    ).strip().lower()
    if weighting != "none":
        raise ValueError(
            "class weighting is excluded from the controlled LightGBM/MLP "
            "comparison; use class_weighting=none")
    seeds = config.get("training", {}).get("seeds")
    if not isinstance(seeds, list) or not seeds or len(set(seeds)) != len(seeds):
        raise ValueError("training.seeds must be a nonempty unique list")
    if not all(isinstance(seed, int) for seed in seeds):
        raise ValueError("training.seeds must contain integers")
    label_source = config.get("protocol", {}).get("label_source")
    if not label_source:
        raise ValueError("protocol.label_source must describe label provenance")


def _model_names(config, available):
    requested = config.get("experiment", {}).get("negative_pool_models", ["M20"])
    names = [str(value).upper() for value in requested]
    invalid = set(names) - set(available)
    if invalid or not names:
        raise ValueError(
            f"invalid negative_pool_models={names}; expected a nonempty "
            f"subset of {sorted(available)}")
    if len(set(names)) != len(names):
        raise ValueError("negative_pool_models contains duplicates")
    return names


def _validation_score(labels, trust_scores):
    # The existing controlled LightGBM config early-stops on its first metric,
    # AUC.  Complementing labels and scores leaves ROC-AUC unchanged, so this
    # is exactly the same ranking objective under canonical error semantics.
    return evaluate_ranking(labels, trust_scores)["roc_auc"]


def _member_seed(experiment_seed, fold):
    """Derive a stable, collision-free fold seed from one experiment seed."""
    sequence = np.random.SeedSequence([int(experiment_seed), int(fold)])
    return int(sequence.generate_state(1, dtype=np.uint32)[0])


def _fit_one_pool(protocol, model_name, test, config, root, *, seed,
                  result_name):
    train = protocol.training_frame(model_name)
    labels = train[protocol.target_col].to_numpy(dtype=int)
    raw_features = train[protocol.feature_cols].to_numpy(dtype="f8")
    raw_test = test[protocol.feature_cols].to_numpy(dtype="f8")
    test_labels = test[protocol.target_col].to_numpy(dtype=int)
    fold_ids = train[protocol.outer_fold_col].to_numpy(dtype=int)
    n_folds = len(protocol.inner_valid_cols)
    expected_folds = set(range(n_folds))
    if set(np.unique(fold_ids).tolist()) != expected_folds:
        raise ValueError(
            f"outer folds are incomplete for {model_name}: "
            f"{sorted(np.unique(fold_ids))}")

    oof = np.full(len(train), np.nan, dtype="f8")
    member_test_scores = []
    fold_summaries = []
    member_thresholds = {target: [] for target in protocol.target_fprs}
    add_missing = bool(
        config.get("preprocessing", {}).get("add_missing_indicators", True))

    for fold in range(n_folds):
        outer_test = fold_ids == fold
        inner_valid = train[protocol.inner_valid_cols[fold]].to_numpy(dtype=bool)
        inner_train = (~outer_test) & (~inner_valid)
        if outer_test.any() and inner_valid.any() and inner_train.any():
            pass
        else:
            raise ValueError(f"fold {fold} contains an empty split")
        groups = train[protocol.group_col]
        group_sets = [
            set(groups[mask]) for mask in (inner_train, inner_valid, outer_test)]
        if any(group_sets[i].intersection(group_sets[j])
               for i in range(3) for j in range(i + 1, 3)):
            raise ValueError(f"fold {fold} has peptide-group leakage")
        counts = {
            "train": _split_counts(
                train[protocol.target_col], groups,
                np.flatnonzero(inner_train)),
            "valid": _split_counts(
                train[protocol.target_col], groups,
                np.flatnonzero(inner_valid)),
            "oof_test": _split_counts(
                train[protocol.target_col], groups,
                np.flatnonzero(outer_test)),
        }
        _validate_split_counts(
            fold, counts,
            int(config["training"].get("min_class_groups_per_split", 1)),
            grouped=True,
        )

        preprocessor = FoldPreprocessor.fit(
            raw_features[inner_train],
            add_missing_indicators=add_missing,
        )
        x_train = preprocessor.transform(raw_features[inner_train])
        x_valid = preprocessor.transform(raw_features[inner_valid])
        x_oof = preprocessor.transform(raw_features[outer_test])
        x_test = preprocessor.transform(raw_test)
        fold_seed = _member_seed(seed, fold)
        fitted = fit_mlp(
            x_train,
            labels[inner_train],
            x_valid,
            labels[inner_valid],
            config,
            validation_score=_validation_score,
            seed=fold_seed,
        )
        batch_size = int(config["training"].get("inference_batch_size", 4096))
        oof_scores = predict_trust(
            fitted.model, x_oof, batch_size=batch_size, device=fitted.device)
        test_scores = predict_trust(
            fitted.model, x_test, batch_size=batch_size, device=fitted.device)
        oof[outer_test] = oof_scores
        member_test_scores.append(test_scores)

        calibration = {}
        for target in protocol.target_fprs:
            threshold = threshold_at_fpr(
                labels[outer_test], oof_scores, target)
            member_thresholds[target].append(threshold)
            point = evaluate_at_threshold(
                labels[outer_test], oof_scores, threshold)
            point.update({
                "target_fpr": target,
                "threshold_source": "this_model_outer_oof_fold",
            })
            calibration[f"fpr_{int(round(target * 100))}"] = point

        checkpoint = (
            root / "models" /
            f"{result_name.lower()}.fold{fold}.pt")
        save_checkpoint(
            checkpoint,
            fitted,
            preprocessor,
            protocol.feature_cols,
            {
                "fold": fold,
                "negative_pool_model": model_name,
                "training_seed": seed,
                "fold_seed": fold_seed,
                "metric_semantics": METRIC_SEMANTICS_VERSION,
                "positive_class": "incorrect_identification",
                "model_score": "trust_score=P(correct_identification)",
                "best_epoch": fitted.best_epoch,
                "best_validation_roc_auc": (
                    fitted.best_validation_score),
            },
        )
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
            "checkpoint": str(checkpoint),
            "roc_auc": fold_metrics["roc_auc"],
            "error_pr_auc": fold_metrics["error_pr_auc"],
            "fnr_at_fpr5": fold_metrics["fnr_at_fpr5"],
            "error_recall_at_fpr10": fold_metrics[
                "error_recall_at_fpr10"],
            "calibration_operating_points": calibration,
        })

    if np.isnan(oof).any():
        raise AssertionError("some training rows did not receive OOF scores")
    member_test_scores = np.vstack(member_test_scores)
    ensemble_trust = member_test_scores.mean(axis=0)
    vote_fractions = {}
    locked_points = {}
    for target, thresholds in member_thresholds.items():
        error_votes = np.vstack([
            (1.0 - scores) >= threshold
            for scores, threshold in zip(member_test_scores, thresholds)
        ])
        vote_fraction = error_votes.mean(axis=0)
        key = f"fpr_{int(round(target * 100))}"
        vote_fractions[key] = vote_fraction
        locked_points[key] = {
            "method": "fold_calibrated_majority_vote",
            "calibration_source": "each_member_outer_oof_fold",
            "member_error_thresholds": [float(x) for x in thresholds],
            "vote_error_threshold": 0.5,
            "external_ensemble": {
                "method": "fold_calibrated_majority_vote",
                "test_metrics": evaluate_at_threshold(
                    test_labels, 1.0 - vote_fraction, 0.5),
            },
        }

    ranking = evaluate_ranking(test_labels, ensemble_trust)
    fixed_metrics = {
        **ranking,
        "operating_points": locked_points,
    }
    pool_predictions = pd.DataFrame({
        f"{result_name}_trust_score": ensemble_trust,
        f"{result_name}_error_score": 1.0 - ensemble_trust,
        **{
            f"{result_name}_{key}_error_vote_fraction": value
            for key, value in vote_fractions.items()
        },
    })
    oof_identity = list(dict.fromkeys(
        column for column in (
            protocol.sample_id_col, protocol.dataset_col,
            *protocol.identity_cols, protocol.group_col,
            protocol.target_col, protocol.tier_col, protocol.source_row_col,
        ) if column in train
    ))
    oof_frame = train[oof_identity].copy()
    oof_frame["outer_fold"] = fold_ids
    oof_frame["trust_score"] = oof
    oof_frame["error_score"] = 1.0 - oof
    _atomic_csv(
        root / "predictions" /
        f"{result_name.lower()}_train_oof.csv",
        oof_frame,
    )
    return {
        "summary": {
            "n_train": len(train),
            "n_actual_correct_train": int((labels == 1).sum()),
            "n_actual_error_train": int((labels == 0).sum()),
            "n_features": len(protocol.feature_cols),
            "n_network_inputs": int(
                len(protocol.feature_cols) * (2 if add_missing else 1)),
            "oof_metrics": evaluate_oof(labels, oof),
            "fixed_test_metrics": fixed_metrics,
            "architecture": fitted.model.architecture(),
            "training_seed": seed,
            "preprocessing": {
                "fit_scope": "inner_training_rows_only_per_outer_fold",
                "median_imputation": True,
                "standardization": True,
                "add_missing_indicators": add_missing,
            },
            "folds": fold_summaries,
        },
        "predictions": pool_predictions,
        "ensemble_trust": ensemble_trust,
        "vote_fractions": vote_fractions,
    }


def _fit_logistic_pool(protocol, model_name, test, config, root, *, seed):
    """Fit the predeclared linear comparator on the identical fold protocol."""
    import joblib
    from sklearn.linear_model import LogisticRegression

    train = protocol.training_frame(model_name)
    labels = train[protocol.target_col].to_numpy(dtype=int)
    test_labels = test[protocol.target_col].to_numpy(dtype=int)
    raw = train[protocol.feature_cols].to_numpy(dtype="f8")
    raw_test = test[protocol.feature_cols].to_numpy(dtype="f8")
    folds = train[protocol.outer_fold_col].to_numpy(dtype=int)
    n_folds = len(protocol.inner_valid_cols)
    add_missing = bool(
        config.get("preprocessing", {}).get("add_missing_indicators", True))
    oof = np.full(len(train), np.nan, dtype="f8")
    member_test_scores = []
    thresholds = {target: [] for target in protocol.target_fprs}
    fold_summaries = []
    result_name = f"logistic_{model_name.lower()}"
    logistic_cfg = config.get("comparators", {}).get("logistic_regression", {})
    for fold in range(n_folds):
        outer_test = folds == fold
        inner_valid = train[protocol.inner_valid_cols[fold]].to_numpy(dtype=bool)
        inner_train = (~outer_test) & (~inner_valid)
        groups = train[protocol.group_col]
        counts = {
            "train": _split_counts(
                train[protocol.target_col], groups,
                np.flatnonzero(inner_train)),
            "valid": _split_counts(
                train[protocol.target_col], groups,
                np.flatnonzero(inner_valid)),
            "oof_test": _split_counts(
                train[protocol.target_col], groups,
                np.flatnonzero(outer_test)),
        }
        _validate_split_counts(
            fold, counts,
            int(config["training"].get("min_class_groups_per_split", 1)),
            grouped=True,
        )
        preprocessor = FoldPreprocessor.fit(
            raw[inner_train], add_missing_indicators=add_missing)
        x_train = preprocessor.transform(raw[inner_train])
        x_oof = preprocessor.transform(raw[outer_test])
        x_test = preprocessor.transform(raw_test)
        fold_seed = _member_seed(seed, fold)
        model = LogisticRegression(
            C=float(logistic_cfg.get("C", 1.0)),
            max_iter=int(logistic_cfg.get("max_iter", 300)),
            class_weight=None,
            random_state=fold_seed,
            solver=str(logistic_cfg.get("solver", "lbfgs")),
        )
        model.fit(x_train, labels[inner_train])
        oof_scores = model.predict_proba(x_oof)[:, 1]
        test_scores = model.predict_proba(x_test)[:, 1]
        oof[outer_test] = oof_scores
        member_test_scores.append(test_scores)
        calibration = {}
        for target in protocol.target_fprs:
            threshold = threshold_at_fpr(
                labels[outer_test], oof_scores, target)
            thresholds[target].append(threshold)
            point = evaluate_at_threshold(
                labels[outer_test], oof_scores, threshold)
            point.update({
                "target_fpr": target,
                "threshold_source": "this_model_outer_oof_fold",
            })
            calibration[f"fpr_{int(round(target * 100))}"] = point
        state = preprocessor.to_state()
        checkpoint = root / "models" / f"{result_name}.fold{fold}.joblib"
        checkpoint.parent.mkdir(parents=True, exist_ok=True)
        temporary = checkpoint.with_name(
            f"{checkpoint.name}.tmp.{os.getpid()}")
        joblib.dump({
            "schema": "metabolic_label_logistic_checkpoint_v1",
            "model": model,
            "preprocessor": state,
            "feature_names": list(protocol.feature_cols),
            "metric_semantics": METRIC_SEMANTICS_VERSION,
            "positive_class": "incorrect_identification",
        }, temporary)
        os.replace(temporary, checkpoint)
        fold_metrics = evaluate_oof(labels[outer_test], oof_scores)
        fold_summaries.append({
            "fold": fold,
            "fold_seed": fold_seed,
            "n_train": int(inner_train.sum()),
            "n_valid_reserved": int(inner_valid.sum()),
            "n_oof": int(outer_test.sum()),
            "split_counts": counts,
            "checkpoint": str(checkpoint),
            "roc_auc": fold_metrics["roc_auc"],
            "error_pr_auc": fold_metrics["error_pr_auc"],
            "fnr_at_fpr5": fold_metrics["fnr_at_fpr5"],
            "error_recall_at_fpr10": fold_metrics[
                "error_recall_at_fpr10"],
            "calibration_operating_points": calibration,
        })

    if np.isnan(oof).any():
        raise AssertionError("logistic comparator left OOF rows unscored")
    member_test_scores = np.vstack(member_test_scores)
    trust = member_test_scores.mean(axis=0)
    votes = {}
    operating_points = {}
    for target, member_thresholds in thresholds.items():
        vote_fraction = np.vstack([
            (1.0 - score) >= threshold
            for score, threshold in zip(member_test_scores, member_thresholds)
        ]).mean(axis=0)
        key = f"fpr_{int(round(target * 100))}"
        votes[key] = vote_fraction
        operating_points[key] = {
            "method": "fold_calibrated_majority_vote",
            "calibration_source": "each_member_outer_oof_fold",
            "member_error_thresholds": [float(x) for x in member_thresholds],
            "vote_error_threshold": 0.5,
            "external_ensemble": {
                "method": "fold_calibrated_majority_vote",
                "test_metrics": evaluate_at_threshold(
                    test_labels, 1.0 - vote_fraction, 0.5),
            },
        }
    fixed_metrics = {
        **evaluate_ranking(test_labels, trust),
        "operating_points": operating_points,
    }
    predictions = pd.DataFrame({
        f"{result_name}_trust_score": trust,
        f"{result_name}_error_score": 1.0 - trust,
        **{
            f"{result_name}_{key}_error_vote_fraction": value
            for key, value in votes.items()
        },
    })
    return {
        "summary": {
            "n_train": int(len(train)),
            "n_actual_correct_train": int((labels == 1).sum()),
            "n_actual_error_train": int((labels == 0).sum()),
            "n_features": int(len(protocol.feature_cols)),
            "n_network_inputs": None,
            "oof_metrics": evaluate_oof(labels, oof),
            "fixed_test_metrics": fixed_metrics,
            "architecture": {
                "type": "logistic_regression_v1",
                "solver": model.solver,
                "C": float(model.C),
                "class_weight": None,
            },
            "training_seed": seed,
            "preprocessing": {
                "fit_scope": "inner_training_rows_only_per_outer_fold",
                "median_imputation": True,
                "standardization": True,
                "add_missing_indicators": add_missing,
            },
            "folds": fold_summaries,
        },
        "predictions": predictions,
        "ensemble_trust": trust,
        "vote_fractions": votes,
    }


def _flat_row(model_name, labels, metrics):
    fpr5 = metrics["operating_points"]["fpr_5"][
        "external_ensemble"]["test_metrics"]
    fpr10 = metrics["operating_points"]["fpr_10"][
        "external_ensemble"]["test_metrics"]
    return {
        "metric_semantics": METRIC_SEMANTICS_VERSION,
        "positive_class": "incorrect_identification",
        "model": model_name,
        "test_subset": "full_E20",
        "n_rows": len(labels),
        "n_actual_correct": int((labels == 1).sum()),
        "n_actual_error": int((labels == 0).sum()),
        "roc_auc": metrics["roc_auc"],
        "error_pr_auc": metrics["error_pr_auc"],
        "locked_fnr_at_fpr5": fpr5["fnr"],
        "observed_fpr_at_fpr5": fpr5["fpr"],
        "locked_error_recall_at_fpr10": fpr10["error_recall"],
        "observed_fpr_at_fpr10": fpr10["fpr"],
    }


def _file_fingerprint(path):
    digest = hashlib.sha256()
    with open(path, "rb") as handle:
        for block in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(block)
    stat = os.stat(path)
    return {
        "path": str(Path(path).resolve()),
        "size_bytes": int(stat.st_size),
        "sha256": digest.hexdigest(),
    }


def _feature_inputs(feature_root, dataset):
    datasets = ("2da", "5da", "normal") if dataset == "combined" else (dataset,)
    return [
        Path(feature_root) / f"baseline_{source}_{pool}" / "features.csv"
        for source in datasets for pool in ("neg05", "neg10", "neg20")
    ]


def _provenance(config_path, split_config_path, feature_root, dataset,
                protocol_root):
    try:
        commit = subprocess.run(
            ["git", "rev-parse", "HEAD"], check=True, capture_output=True,
            text=True).stdout.strip()
        dirty = bool(subprocess.run(
            ["git", "status", "--porcelain", "--untracked-files=no"],
            check=True, capture_output=True, text=True).stdout.strip())
    except (OSError, subprocess.CalledProcessError):
        commit, dirty = None, None

    def fingerprint(path):
        with open(path, "rb") as handle:
            digest = hashlib.sha256(handle.read()).hexdigest()
        return {"path": str(Path(path).resolve()), "sha256": digest}

    return {
        "created_at_utc": dt.datetime.now(dt.timezone.utc).isoformat(),
        "git_commit": commit,
        "git_tracked_files_dirty": dirty,
        "python": platform.python_version(),
        "torch": torch.__version__,
        "numpy": np.__version__,
        "pandas": pd.__version__,
        "cuda_available": torch.cuda.is_available(),
        "feature_root": str(Path(feature_root).resolve()),
        "deep_config": fingerprint(config_path),
        "split_config": fingerprint(split_config_path),
        "frozen_protocol_root": str(Path(protocol_root).resolve()),
        "inputs": [
            _file_fingerprint(path)
            for path in _feature_inputs(feature_root, dataset)
        ],
    }


def _fixed_metrics(labels, values):
    operating_points = {}
    for key, vote_key in (
        ("fpr_5", "fpr_5_vote_fraction"),
        ("fpr_10", "fpr_10_vote_fraction"),
    ):
        operating_points[key] = {
            "method": "fold_calibrated_majority_vote",
            "external_ensemble": {
                "method": "fold_calibrated_majority_vote",
                "test_metrics": evaluate_at_threshold(
                    labels, 1.0 - values[vote_key], 0.5),
            },
        }
    return {
        **evaluate_ranking(labels, values["trust_score"]),
        "operating_points": operating_points,
    }


def _prediction_values(result):
    return {
        "trust_score": result["ensemble_trust"],
        "fpr_5_vote_fraction": result["vote_fractions"]["fpr_5"],
        "fpr_10_vote_fraction": result["vote_fractions"]["fpr_10"],
    }


def _chemistry_audit(frame):
    result = {}
    for column in ("labeling", "isotope_model"):
        if column in frame:
            values = frame[column].astype("string").fillna("<missing>")
            result[column] = {
                str(key): int(value)
                for key, value in values.value_counts().items()
            }
        else:
            result[column] = {"status": "column_not_available"}
    return result


def _missingness_sensitivity(config):
    enabled = bool(config.get("preprocessing", {}).get(
        "add_missing_indicators", True))
    return {
        "add_missing_indicators": enabled,
        "sensitivity_arm": (
            "tools/deep_trainer/config/tabular_mlp_no_missing_indicators.yaml"
            if enabled else "tools/deep_trainer/config/tabular_mlp.yaml"),
        "comparison_contract": (
            "rerun against the same frozen protocol root and compare paired "
            "fixed-test outputs; do not regenerate membership or folds"),
    }


def _generalization_audit(protocol):
    train = protocol.frame[protocol.frame[protocol.split_col].eq("train")]
    test = protocol.frame[protocol.frame[protocol.split_col].eq("test")]
    train_sequences = set(train[protocol.base_group_col].astype(str))
    test_sequences = set(test[protocol.base_group_col].astype(str))
    overlap = train_sequences.intersection(test_sequences)
    grouping = protocol.validation.get("identity", {}).get(
        "split_grouping", protocol.validation.get("identity", {}).get(
            "grouping", {}))
    family_protected = bool(
        isinstance(grouping, dict)
        and grouping.get("candidate_family_leakage_protected"))
    grouping_limitation = (
        grouping.get("limitation") if isinstance(grouping, dict) else None)
    claim = (
        "internal_unseen_sequence_holdout_with_candidate_family_protection"
        if not overlap and family_protected
        else "internal_unseen_sequence_holdout_sequence_only"
        if not overlap
        else "internal_domain_holdout_with_sequence_overlap"
    )
    return {
        "holdout_type": "internal_fixed_E20_holdout",
        "is_external_entrapment_test": False,
        "train_unique_sequences": int(len(train_sequences)),
        "test_unique_sequences": int(len(test_sequences)),
        "overlap_unique_sequences": int(len(overlap)),
        "candidate_family_leakage_protected": family_protected,
        "allowed_generalization_claim": claim,
        "limitation": (
            None if family_protected else grouping_limitation or
            "pair/candidate-family IDs are absent; only sequence leakage is "
            "audited and prevented"),
    }


def _seed_dispersion(model_results):
    by_pool = {}
    for name, result in model_results.items():
        if not name.startswith("MLP_"):
            continue
        parts = name.split("_seed", 1)[0].split("_", 1)
        pool = parts[1] if len(parts) == 2 else "unknown"
        fixed = result["fixed_test_metrics"]
        fpr5 = fixed["operating_points"]["fpr_5"][
            "external_ensemble"]["test_metrics"]
        fpr10 = fixed["operating_points"]["fpr_10"][
            "external_ensemble"]["test_metrics"]
        by_pool.setdefault(pool, []).append({
            "roc_auc": fixed["roc_auc"],
            "error_pr_auc": fixed["error_pr_auc"],
            "locked_fnr_at_fpr5": fpr5["fnr"],
            "locked_error_recall_at_fpr10": fpr10["error_recall"],
        })
    metrics = (
        "roc_auc", "error_pr_auc", "locked_fnr_at_fpr5",
        "locked_error_recall_at_fpr10",
    )
    result = {}
    for pool, rows in by_pool.items():
        frame = pd.DataFrame(rows)
        result[pool] = {
            metric: {
                "mean": float(frame[metric].mean()),
                "std": float(frame[metric].std(ddof=0)),
            }
            for metric in metrics
        }
    result["note"] = (
            "seed standard deviations reuse the same fixed test; they are "
            "descriptive optimization dispersion, not confidence intervals")
    return result


def _run_into_staging(
    config_path,
    split_config_path,
    feature_root,
    dataset,
    protocol_root,
    staging,
    final_root,
    *,
    prepare_only,
):
    with open(config_path, encoding="utf-8") as handle:
        config = yaml.safe_load(handle)
    _validate_deep_config(config)
    with open(split_config_path, encoding="utf-8") as handle:
        split_config = yaml.safe_load(handle)
    _atomic_yaml(staging / "config_used.yaml", config)
    _atomic_yaml(staging / "split_config_used.yaml", split_config)
    log_path = staging / "logs" / "train.log"
    log_path.parent.mkdir(parents=True, exist_ok=True)
    logging.basicConfig(
        level=logging.INFO,
        force=True,
        format="%(asctime)s %(levelname)s %(message)s",
        handlers=[
            logging.FileHandler(log_path, mode="w", encoding="utf-8"),
            logging.StreamHandler(),
        ],
    )
    protocol = prepare_protocol(
        split_config_path, feature_root, dataset, protocol_root)
    _atomic_json(staging / "preflight.json", protocol.validation)
    _atomic_csv(staging / "manifests" / "membership.csv",
                protocol.membership)
    manifest_columns = list(dict.fromkeys(
        column for column in (
            protocol.sample_id_col, protocol.dataset_col,
            *protocol.identity_cols, protocol.base_group_col,
            protocol.group_col, protocol.target_col, protocol.tier_col,
            protocol.source_row_col, protocol.split_col,
            protocol.outer_fold_col,
        ) if column in protocol.frame
    ))
    _atomic_csv(staging / "manifests" / "fixed_test_manifest.csv",
                protocol.frame[manifest_columns])
    _atomic_csv(staging / "manifests" / "fold_map.csv",
                protocol.group_fold_map)
    if prepare_only:
        _atomic_json(staging / "bundle_status.json", {
            "status": "prepare_only",
            "metric_semantics": METRIC_SEMANTICS_VERSION,
            "positive_class": "incorrect_identification",
        })
        return {"mode": "prepare_only", **protocol.validation}

    test = protocol.test_frame()
    test_labels = test[protocol.target_col].to_numpy(dtype=int)
    identity = list(dict.fromkeys(
        column for column in (
            protocol.sample_id_col, protocol.dataset_col,
            *protocol.identity_cols, protocol.base_group_col,
            protocol.group_col, protocol.target_col, protocol.tier_col,
            protocol.source_row_col,
        ) if column in test
    ))
    predictions = test[identity].copy()
    model_results = {}
    model_predictions = {}
    flat_rows = []
    domain_rows = []
    negative_pool_models = _model_names(config, protocol.model_tiers)

    baseline_predictions = load_frozen_lightgbm_predictions(
        protocol.protocol_root, test, protocol.sample_id_col,
        negative_pool_models)
    for name, values in baseline_predictions.items():
        fixed = _fixed_metrics(test_labels, values)
        model_results[name] = {
            "source": "frozen_lightgbm_protocol_bundle",
            "fixed_test_metrics": fixed,
        }
        model_predictions[name] = values
        flat_rows.append(_flat_row(name, test_labels, fixed))
        predictions[f"{name}_trust_score"] = values["trust_score"]
        predictions[f"{name}_error_score"] = 1.0 - values["trust_score"]
        predictions[f"{name}_fpr_5_error_vote_fraction"] = values[
            "fpr_5_vote_fraction"]
        predictions[f"{name}_fpr_10_error_vote_fraction"] = values[
            "fpr_10_vote_fraction"]

    for model_name in negative_pool_models:
        for seed in config["training"]["seeds"]:
            result_name = f"MLP_{model_name}_seed{seed}"
            logging.info(
                "training tabular MLP pool=%s seed=%d", model_name, seed)
            result = _fit_one_pool(
                protocol, model_name, test, config, staging,
                seed=seed, result_name=result_name)
            predictions = pd.concat(
                [predictions, result["predictions"]], axis=1)
            model_results[result_name] = result["summary"]
            values = _prediction_values(result)
            model_predictions[result_name] = values
            fixed = result["summary"]["fixed_test_metrics"]
            flat_rows.append(_flat_row(result_name, test_labels, fixed))

        if config.get("comparators", {}).get(
                "logistic_regression", {}).get("enabled", True):
            seed = int(config["training"]["seeds"][0])
            result_name = f"Logistic_{model_name}_seed{seed}"
            logging.info("training logistic comparator pool=%s", model_name)
            result = _fit_logistic_pool(
                protocol, model_name, test, config, staging, seed=seed)
            predictions = pd.concat(
                [predictions, result["predictions"]], axis=1)
            model_results[result_name] = result["summary"]
            model_predictions[result_name] = _prediction_values(result)
            fixed = result["summary"]["fixed_test_metrics"]
            flat_rows.append(_flat_row(result_name, test_labels, fixed))

    for model_name, values in model_predictions.items():
        if protocol.dataset_col in test:
            for domain in sorted(test[protocol.dataset_col].unique()):
                mask = test[protocol.dataset_col].eq(domain).to_numpy()
                subset = {
                    key: np.asarray(value)[mask]
                    for key, value in values.items()
                }
                metrics = _fixed_metrics(test_labels[mask], subset)
                row = _flat_row(model_name, test_labels[mask], metrics)
                row["test_dataset"] = domain
                domain_rows.append(row)

    comparison_pairs = []
    for pool in negative_pool_models:
        baseline = f"LightGBM_{pool}"
        for seed in config["training"]["seeds"]:
            comparison_pairs.append((baseline, f"MLP_{pool}_seed{seed}"))
        logistic = f"Logistic_{pool}_seed{config['training']['seeds'][0]}"
        if logistic in model_predictions:
            comparison_pairs.append((baseline, logistic))
            for seed in config["training"]["seeds"]:
                comparison_pairs.append((
                    logistic, f"MLP_{pool}_seed{seed}"))
    bootstrap_cfg = config.get("comparison", {})
    bootstrap = paired_cluster_bootstrap(
        test, model_predictions, comparison_pairs,
        reps=int(bootstrap_cfg.get("bootstrap_reps", 1000)),
        seed=int(bootstrap_cfg.get("bootstrap_seed", 20260812)),
        group_col=protocol.group_col,
        target_col=protocol.target_col,
        evaluate_ranking=evaluate_ranking,
        evaluate_at_threshold=evaluate_at_threshold,
    )

    train_audit = missingness_audit(
        protocol.frame[protocol.frame[protocol.split_col].eq("train")],
        protocol.feature_cols, target_col=protocol.target_col,
        split_name="train", dataset_col=protocol.dataset_col)
    test_audit = missingness_audit(
        test, protocol.feature_cols, target_col=protocol.target_col,
        split_name="fixed_test", dataset_col=protocol.dataset_col)
    missingness = pd.concat([train_audit, test_audit], ignore_index=True)
    _atomic_csv(staging / "predictions" / "fixed_test_predictions.csv",
                predictions)
    _atomic_csv(staging / "fixed_test_summary.csv", pd.DataFrame(flat_rows))
    _atomic_csv(staging / "paired_model_bootstrap.csv", bootstrap)
    _atomic_csv(staging / "missingness_audit.csv", missingness)
    if domain_rows:
        _atomic_csv(staging / "domain_summary.csv", pd.DataFrame(domain_rows))

    chemistry = _chemistry_audit(protocol.frame)
    summary = {
        "metric_semantics": METRIC_SEMANTICS_VERSION,
        "positive_class": "incorrect_identification",
        "experiment": "tabular_phase1_frozen_protocol_v2",
        "dataset": dataset,
        "model_score": "trust_score=P(correct_identification)",
        "feature_arm": protocol.feature_arm,
        "cohort": protocol.cohort,
        "label_source": config["protocol"]["label_source"],
        "chemistry": chemistry,
        "split_contract": {
            "source": protocol.protocol_root,
            "membership": "loaded_from_frozen_LightGBM_manifest",
            "folds": "loaded_from_frozen_LightGBM_fold_map",
            "holdout": "internal_fixed_E20_not_external_entrapment",
        },
        "generalization_audit": _generalization_audit(protocol),
        "feature_names": protocol.feature_cols,
        "models": model_results,
        "seed_dispersion": _seed_dispersion(model_results),
        "missingness_audit": summarize_missingness(
            missingness, target_col=protocol.target_col),
        "missingness_sensitivity": _missingness_sensitivity(config),
        "paired_comparison": {
            "resampling_unit": protocol.group_col,
            "n_requested": int(bootstrap_cfg.get("bootstrap_reps", 1000)),
            "seed": int(bootstrap_cfg.get("bootstrap_seed", 20260812)),
            "note": (
                "paired bootstrap rows are confidence intervals; seed "
                "standard deviations are descriptive only"),
        },
        "artifacts": {
            "preflight": str(final_root / "preflight.json"),
            "config_used": str(final_root / "config_used.yaml"),
            "split_config_used": str(final_root / "split_config_used.yaml"),
            "fixed_test_summary": str(final_root / "fixed_test_summary.csv"),
            "paired_model_bootstrap": str(
                final_root / "paired_model_bootstrap.csv"),
            "missingness_audit": str(final_root / "missingness_audit.csv"),
            "fixed_test_predictions": str(
                final_root / "predictions" / "fixed_test_predictions.csv"),
            "membership": str(
                final_root / "manifests" / "membership.csv"),
            "fixed_test_manifest": str(
                final_root / "manifests" / "fixed_test_manifest.csv"),
            "fold_map": str(final_root / "manifests" / "fold_map.csv"),
        },
        "provenance": _provenance(
            config_path, split_config_path, feature_root, dataset,
            protocol_root),
    }
    staging_text = str(staging)
    final_text = str(final_root)
    summary = json.loads(
        json.dumps(summary).replace(staging_text, final_text))
    _atomic_json(staging / "summary.json", summary)
    _atomic_json(staging / "bundle_status.json", {
        "status": "complete",
        "metric_semantics": METRIC_SEMANTICS_VERSION,
        "positive_class": "incorrect_identification",
    })
    return summary


def run_experiment(
    config_path,
    split_config_path,
    feature_root,
    dataset,
    protocol_root,
    output_root,
    *,
    overwrite=False,
    prepare_only=False,
):
    """Build then atomically publish one Phase-1 result bundle."""
    root = Path(output_root)
    _assert_output_available(root, overwrite)
    root.parent.mkdir(parents=True, exist_ok=True)
    staging = Path(tempfile.mkdtemp(
        prefix=f".{root.name}.staging.", dir=root.parent))
    try:
        result = _run_into_staging(
            config_path, split_config_path, feature_root, dataset,
            protocol_root, staging, root, prepare_only=prepare_only)
        _publish_bundle(staging, root, overwrite)
        return result
    except Exception:
        shutil.rmtree(staging, ignore_errors=True)
        raise


def _parser():
    parser = argparse.ArgumentParser(
        description="Grouped-OOF tabular MLP on the fixed E20 protocol")
    parser.add_argument("--config", required=True)
    parser.add_argument("--split-config", required=True)
    parser.add_argument("--feature-root", required=True)
    parser.add_argument(
        "--protocol-root", required=True,
        help="completed frozen LightGBM fixed-negpool bundle")
    parser.add_argument(
        "--dataset", choices=["combined", "2da", "5da", "normal"],
        default="combined")
    parser.add_argument("--output-root", required=True)
    parser.add_argument("--overwrite", action="store_true")
    parser.add_argument("--prepare-only", action="store_true")
    return parser


def main(argv=None):
    args = _parser().parse_args(argv)
    return run_experiment(
        args.config,
        args.split_config,
        args.feature_root,
        args.dataset,
        args.protocol_root,
        args.output_root,
        overwrite=args.overwrite,
        prepare_only=args.prepare_only,
    )


if __name__ == "__main__":
    main()
