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
import subprocess
import sys

import numpy as np
import pandas as pd
import torch
import yaml

from .checkpoint import save_checkpoint
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
    existing = [path for path in _known_outputs(root) if path.exists()]
    if existing and not overwrite:
        raise FileExistsError(
            "refusing to overwrite an existing deep-trainer result bundle; "
            "choose a new output root or pass --overwrite:\n  "
            + "\n  ".join(map(str, existing)))


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


def _fit_one_pool(protocol, model_name, test, config, root):
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
    seed = int(config["training"].get("seed", 42))
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

        preprocessor = FoldPreprocessor.fit(
            raw_features[inner_train],
            add_missing_indicators=add_missing,
        )
        x_train = preprocessor.transform(raw_features[inner_train])
        x_valid = preprocessor.transform(raw_features[inner_valid])
        x_oof = preprocessor.transform(raw_features[outer_test])
        x_test = preprocessor.transform(raw_test)
        fitted = fit_mlp(
            x_train,
            labels[inner_train],
            x_valid,
            labels[inner_valid],
            config,
            validation_score=_validation_score,
            seed=seed + fold,
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
            f"tabular_mlp_{model_name.lower()}.fold{fold}.pt")
        save_checkpoint(
            checkpoint,
            fitted,
            preprocessor,
            protocol.feature_cols,
            {
                "fold": fold,
                "negative_pool_model": model_name,
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
            "n_train": int(inner_train.sum()),
            "n_valid": int(inner_valid.sum()),
            "n_oof": int(outer_test.sum()),
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
            "test_metrics": evaluate_at_threshold(
                test_labels, 1.0 - vote_fraction, 0.5),
        }

    ranking = evaluate_ranking(test_labels, ensemble_trust)
    fixed_metrics = {
        **ranking,
        "operating_points": locked_points,
    }
    pool_predictions = pd.DataFrame({
        f"mlp_{model_name.lower()}_trust_score": ensemble_trust,
        f"mlp_{model_name.lower()}_error_score": 1.0 - ensemble_trust,
        **{
            f"mlp_{model_name.lower()}_{key}_error_vote_fraction": value
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
        f"tabular_mlp_{model_name.lower()}_train_oof.csv",
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
            "folds": fold_summaries,
        },
        "predictions": pool_predictions,
        "ensemble_trust": ensemble_trust,
        "vote_fractions": vote_fractions,
    }


def _flat_row(model_name, labels, metrics):
    fpr5 = metrics["operating_points"]["fpr_5"]["test_metrics"]
    fpr10 = metrics["operating_points"]["fpr_10"]["test_metrics"]
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


def _provenance(config_path, split_config_path, feature_root):
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
    }


def run_experiment(
    config_path,
    split_config_path,
    feature_root,
    dataset,
    output_root,
    *,
    overwrite=False,
    prepare_only=False,
):
    """Run the first deep-learning phase on the frozen fixed-test protocol."""
    with open(config_path, encoding="utf-8") as handle:
        config = yaml.safe_load(handle)
    _validate_deep_config(config)
    with open(split_config_path, encoding="utf-8") as handle:
        split_config = yaml.safe_load(handle)
    root = Path(output_root)
    _assert_output_available(root, overwrite)
    _atomic_yaml(root / "config_used.yaml", config)
    _atomic_yaml(root / "split_config_used.yaml", split_config)
    log_path = root / "logs" / "train.log"
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
    protocol_cfg = config.get("protocol", {})
    protocol = prepare_protocol(
        split_config_path,
        feature_root,
        dataset,
        test_fraction=float(protocol_cfg.get("test_fraction", 0.20)),
        min_test_errors_per_tier=int(
            protocol_cfg.get("min_test_errors_per_tier", 100)),
        split_candidates=int(protocol_cfg.get("split_candidates", 128)),
    )
    _atomic_json(root / "preflight.json", protocol.validation)
    _atomic_csv(root / "manifests" / "membership.csv",
                protocol.membership)
    fixed_test_manifest = protocol.membership.loc[
        protocol.membership[protocol.split_col].eq("test")].copy()
    _atomic_csv(root / "manifests" / "fixed_test_manifest.csv",
                fixed_test_manifest)
    _atomic_csv(root / "manifests" / "fold_map.csv",
                protocol.group_fold_map)
    if prepare_only:
        return {"mode": "prepare_only", **protocol.validation}

    test = protocol.test_frame()
    test_labels = test[protocol.target_col].to_numpy(dtype=int)
    identity = list(dict.fromkeys(
        column for column in (
            protocol.sample_id_col, protocol.dataset_col,
            *protocol.identity_cols, protocol.group_col, protocol.target_col,
            protocol.tier_col, protocol.source_row_col,
        ) if column in test
    ))
    predictions = test[identity].copy()
    model_results = {}
    flat_rows = []
    domain_rows = []
    for model_name in _model_names(config, protocol.model_tiers):
        logging.info("training tabular MLP negative pool=%s", model_name)
        result = _fit_one_pool(
            protocol, model_name, test, config, root)
        predictions = pd.concat(
            [predictions, result["predictions"]], axis=1)
        model_results[model_name] = result["summary"]
        fixed = result["summary"]["fixed_test_metrics"]
        report_name = f"MLP_{model_name}"
        flat_rows.append(_flat_row(report_name, test_labels, fixed))
        if protocol.dataset_col in test:
            for domain in sorted(test[protocol.dataset_col].unique()):
                mask = test[protocol.dataset_col].eq(domain).to_numpy()
                trust = result["ensemble_trust"][mask]
                domain_metrics = {
                    **evaluate_ranking(test_labels[mask], trust),
                    "operating_points": {},
                }
                for key, values in result["vote_fractions"].items():
                    domain_metrics["operating_points"][key] = {
                        "test_metrics": evaluate_at_threshold(
                            test_labels[mask], 1.0 - values[mask], 0.5)
                    }
                row = _flat_row(
                    report_name, test_labels[mask], domain_metrics)
                row["test_dataset"] = domain
                domain_rows.append(row)

    _atomic_csv(root / "predictions" / "fixed_test_predictions.csv",
                predictions)
    _atomic_csv(root / "fixed_test_summary.csv", pd.DataFrame(flat_rows))
    if domain_rows:
        _atomic_csv(root / "domain_summary.csv", pd.DataFrame(domain_rows))
    summary = {
        "metric_semantics": METRIC_SEMANTICS_VERSION,
        "positive_class": "incorrect_identification",
        "experiment": "tabular_mlp_fixed_negpool_v1",
        "dataset": dataset,
        "model_score": "trust_score=P(correct_identification)",
        "feature_arm": protocol.feature_arm,
        "cohort": protocol.cohort,
        "split_contract": (
            "same fixed E20 test and predefined sequence-grouped outer/inner "
            "folds as spec_trainer fixed-negpool"),
        "feature_names": protocol.feature_cols,
        "models": model_results,
        "artifacts": {
            "preflight": str(root / "preflight.json"),
            "config_used": str(root / "config_used.yaml"),
            "split_config_used": str(root / "split_config_used.yaml"),
            "fixed_test_summary": str(root / "fixed_test_summary.csv"),
            "fixed_test_predictions": str(
                root / "predictions" / "fixed_test_predictions.csv"),
            "membership": str(root / "manifests" / "membership.csv"),
            "fixed_test_manifest": str(
                root / "manifests" / "fixed_test_manifest.csv"),
            "fold_map": str(root / "manifests" / "fold_map.csv"),
        },
        "provenance": _provenance(
            config_path, split_config_path, feature_root),
    }
    _atomic_json(root / "summary.json", summary)
    return summary


def _parser():
    parser = argparse.ArgumentParser(
        description="Grouped-OOF tabular MLP on the fixed E20 protocol")
    parser.add_argument("--config", required=True)
    parser.add_argument("--split-config", required=True)
    parser.add_argument("--feature-root", required=True)
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
        args.output_root,
        overwrite=args.overwrite,
        prepare_only=args.prepare_only,
    )


if __name__ == "__main__":
    main()
