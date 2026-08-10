"""Cross-validated training entry for spec_trainer (production LightGBM).

One CV pass yields OOF predictions that drive: honest evaluation, fold
ensemble (saved per-fold models), and label-noise audit. Does NOT touch
main.py (single-holdout flow unchanged). lightgbm is imported lazily inside
assemble_oof / predict_ensemble / evaluate_cross_test so this module (and path
helpers) import without it.
"""
import argparse
import datetime as dt
import hashlib
import json
import logging
import os
import platform
import re
import subprocess
import sys

import numpy as np
import pandas as pd
import yaml

from cv_core import (METRIC_SEMANTICS_VERSION, average_proba, audit_labels,
                     evaluate_at_threshold, evaluate_oof, make_cv_splits,
                     threshold_at_fpr)
from cohort import apply_training_cohort
from feature_cols import resolve_configured_feature_cols


_SOURCE_FILE = "__source_file"
_SOURCE_ROW = "__source_row"


def read_dataframe(train_files):
    """Concatenate CSVs and retain row/file provenance outside model inputs."""
    frames = []
    for path in train_files:
        frame = pd.read_csv(path)
        frame[_SOURCE_FILE] = os.path.abspath(path)
        frame[_SOURCE_ROW] = np.arange(len(frame), dtype=np.int64)
        frames.append(frame)
    return pd.concat(frames, ignore_index=True)


def derive_paths(cfg):
    """(model_prefix, result_path, suspects_path) from output.{model,result}_path."""
    model_path = cfg["output"]["model_path"]
    result_path = cfg["output"]["result_path"]
    model_prefix = re.sub(r"\.txt$", "", model_path)
    suspects_path = re.sub(r"\.json$", ".suspects.csv", result_path)
    if suspects_path == result_path:
        raise ValueError(
            f"output.result_path must end in '.json' so the suspects CSV path "
            f"doesn't collide with it; got {result_path!r}")
    return model_prefix, result_path, suspects_path


def _build_parser():
    p = argparse.ArgumentParser(description="CV train + ensemble + label audit")
    p.add_argument("--config", required=True)
    p.add_argument("--name", required=True)
    p.add_argument("--logpath", default="./cv_spec.log")
    p.add_argument(
        "--overwrite", action="store_true",
        help="replace an existing result bundle (default: fail closed)")
    return p


def _artifact_paths(result_path):
    """Derive every non-model artifact from the canonical result JSON path."""
    if not result_path.endswith(".json"):
        raise ValueError(f"result path must end in .json: {result_path!r}")
    stem = result_path[:-5]
    return {
        "result": result_path,
        "suspects": f"{stem}.suspects.csv",
        "oof_predictions": f"{stem}.oof.csv",
        "test_predictions": f"{stem}.test_scores.csv",
    }


def _assert_output_bundle_available(paths, overwrite):
    existing = [path for path in paths.values() if os.path.exists(path)]
    if existing and not overwrite:
        raise FileExistsError(
            "refusing to overwrite an existing CV result bundle; choose a new "
            "CV_OUTPUT_ROOT or pass --overwrite:\n  " + "\n  ".join(existing))


def _atomic_json(path, value):
    os.makedirs(os.path.dirname(path) or ".", exist_ok=True)
    tmp = f"{path}.tmp.{os.getpid()}"
    with open(tmp, "w", encoding="utf-8") as handle:
        json.dump(value, handle, indent=2)
    os.replace(tmp, path)


def _atomic_csv(path, frame):
    os.makedirs(os.path.dirname(path) or ".", exist_ok=True)
    tmp = f"{path}.tmp.{os.getpid()}"
    frame.to_csv(tmp, index=False)
    os.replace(tmp, path)


def _operating_targets(cfg):
    op_cfg = cfg.get("operating_point", {})
    raw = op_cfg.get("target_fprs")
    if raw is None:
        raw = [op_cfg.get("target_fpr", 0.10)]
    targets = sorted({float(value) for value in raw})
    if not targets or any(not 0.0 <= value < 1.0 for value in targets):
        raise ValueError(f"invalid operating-point FPR targets: {raw!r}")
    primary = float(op_cfg.get("primary_target_fpr", targets[-1]))
    if primary not in targets:
        raise ValueError(
            f"primary_target_fpr {primary} is not in target_fprs {targets}")
    return targets, primary


def _op_key(target):
    return f"fpr_{int(round(target * 100))}"


def _validate_frame(df, feature_cols, target_col, group_col=None):
    """Fail before fitting on malformed labels, groups, or numeric inputs."""
    if target_col not in df:
        raise ValueError(f"target column {target_col!r} is missing")
    labels = df[target_col]
    if labels.isna().any() or not set(labels.unique().tolist()).issubset({0, 1}):
        raise ValueError(
            f"{target_col} must contain only 0 (error) and 1 (correct)")
    if group_col:
        if group_col not in df:
            raise ValueError(
                f"configured group_col {group_col!r} is missing; refusing "
                "to fall back to ungrouped CV")
        if df[group_col].isna().any():
            raise ValueError(f"group_col {group_col!r} contains missing values")
    missing = [column for column in feature_cols if column not in df]
    if missing:
        raise ValueError(f"feature schema is missing columns: {missing}")
    non_numeric = [
        column for column in feature_cols
        if not pd.api.types.is_numeric_dtype(df[column])]
    if non_numeric:
        raise ValueError(f"model features must be numeric: {non_numeric}")
    infinite = [
        column for column in feature_cols
        if np.isinf(df[column].to_numpy(dtype="f8", copy=False)).any()]
    if infinite:
        raise ValueError(f"model features contain infinite values: {infinite}")


def _missingness_audit(df, feature_cols, target_col):
    def rates(part):
        return {column: float(part[column].isna().mean())
                for column in feature_cols}

    by_class = {
        "correct": rates(df[df[target_col] == 1]),
        "error": rates(df[df[target_col] == 0]),
    }
    by_source = {
        source: rates(part)
        for source, part in df.groupby(_SOURCE_FILE, sort=True)
    }
    class_gap = {
        column: abs(by_class["correct"][column] - by_class["error"][column])
        for column in feature_cols
    }
    source_rates = list(by_source.values())
    source_range = {}
    for column in feature_cols:
        values = [item[column] for item in source_rates]
        source_range[column] = max(values) - min(values) if values else 0.0
    return {
        "overall": rates(df),
        "by_class": by_class,
        "by_source_file": by_source,
        "class_missing_rate_abs_gap": class_gap,
        "source_missing_rate_range": source_range,
        "shortcut_warnings": {
            "class_gap_ge_0_10": sorted(
                column for column, gap in class_gap.items() if gap >= 0.10),
            "source_range_ge_0_10": sorted(
                column for column, gap in source_range.items() if gap >= 0.10),
        },
        "model_missingness_policy": "native_lightgbm_missing_splits_allowed",
    }


def _sequence_overlap(train_df, test_df, group_col):
    if not group_col:
        return None
    train = set(train_df[group_col].astype(str))
    test = set(test_df[group_col].astype(str))
    overlap = train.intersection(test)
    return {
        "group_col": group_col,
        "n_unique_train": len(train),
        "n_unique_test": len(test),
        "n_unique_overlap": len(overlap),
        "test_overlap_fraction": float(len(overlap) / len(test)) if test else None,
        "interpretation": (
            "domain_holdout_not_unseen-group_generalization"
            if overlap else "domain_and_group_holdout"),
    }


def _metrics_by_source(df, labels, trust_scores):
    """Pooled leak-free/ranking metrics split by original feature CSV."""
    labels = np.asarray(labels)
    trust_scores = np.asarray(trust_scores)
    result = {}
    for source, indices in df.groupby(_SOURCE_FILE, sort=True).indices.items():
        idx = np.asarray(indices, dtype=int)
        y_source = labels[idx]
        item = {
            "n_rows": int(len(idx)),
            "n_actual_correct": int((y_source == 1).sum()),
            "n_actual_error": int((y_source == 0).sum()),
        }
        if len(set(y_source.tolist())) == 2:
            item.update(evaluate_oof(y_source, trust_scores[idx]))
        else:
            item["metrics_unavailable_reason"] = "source_contains_one_class"
        result[source] = item
    return result


def _file_fingerprint(path):
    stat = os.stat(path)
    with open(path, "rb") as handle:
        header = handle.readline()
    return {
        "path": os.path.abspath(path),
        "size_bytes": stat.st_size,
        "mtime_ns": stat.st_mtime_ns,
        "header_sha256": hashlib.sha256(header).hexdigest(),
    }


def _provenance(args, cfg, train_files, test_files):
    try:
        commit = subprocess.run(
            ["git", "rev-parse", "HEAD"], check=True, capture_output=True,
            text=True).stdout.strip()
        dirty = bool(subprocess.run(
            ["git", "status", "--porcelain", "--untracked-files=no"],
            check=True, capture_output=True, text=True).stdout.strip())
    except (OSError, subprocess.CalledProcessError):
        commit, dirty = None, None
    try:
        import lightgbm
        import sklearn
        versions = {
            "python": platform.python_version(), "numpy": np.__version__,
            "pandas": pd.__version__, "scikit_learn": sklearn.__version__,
            "lightgbm": lightgbm.__version__,
        }
    except ImportError:
        versions = {"python": platform.python_version(),
                    "numpy": np.__version__, "pandas": pd.__version__}
    with open(args.config, "rb") as handle:
        config_sha256 = hashlib.sha256(handle.read()).hexdigest()
    return {
        "created_at_utc": dt.datetime.now(dt.timezone.utc).isoformat(),
        "run_id": (
            dt.datetime.now(dt.timezone.utc).strftime("%Y%m%dT%H%M%SZ")
            + "-" + config_sha256[:12]),
        "command": [
            sys.executable, "tools/spec_trainer/src/cv_train.py",
            "--config", args.config, "--name", args.name,
            "--logpath", args.logpath,
            *(["--overwrite"] if args.overwrite else []),
        ],
        "config_path": os.path.abspath(args.config),
        "config_sha256": config_sha256,
        "git_commit": commit,
        "git_tracked_files_dirty": dirty,
        "versions": versions,
        "model_params": cfg.get("model", {}).get("params", {}),
        "training_params": cfg.get("training", {}),
        "inputs": [_file_fingerprint(path)
                   for path in list(train_files) + list(test_files or [])],
        "input_fingerprint_note": (
            "size+mtime+header hash; feature snapshots are treated as immutable"),
    }


def _inner_split(X, y, groups, tr_idx, valid_size, seed):
    """Carve an early-stopping validation set from a fold's TRAIN indices.

    Returns (tr2, val) as GLOBAL positional indices — both subsets of tr_idx,
    so the val set never overlaps the OOF test fold. When groups are given,
    split at group level and choose the deterministic candidate closest to
    the global class ratio. Mixed-label groups are supported because a
    synthetic negative shares its parent positive's group. Without groups,
    stratify rows.
    """
    from sklearn.model_selection import GroupShuffleSplit, StratifiedShuffleSplit
    if groups is not None:
        part = pd.DataFrame({
            "group": groups.iloc[tr_idx].to_numpy(),
            "label": y.iloc[tr_idx].to_numpy(),
        })
        if part["group"].isna().any():
            raise ValueError("group_col contains missing values")
        # Synthetic negatives deliberately share a group with their parent
        # positive. A one-label-per-group assumption would either reject this
        # leak-safe representation or force the derived pair across folds.
        # GroupShuffleSplit supports mixed-label groups; try several
        # deterministic candidates and retain the one closest to the global
        # class ratio while keeping both classes on both sides.
        splitter = GroupShuffleSplit(
            n_splits=64, test_size=valid_size, random_state=seed)
        global_rate = float(part["label"].mean())
        best = None
        for loc_tr, loc_val in splitter.split(
                np.zeros(len(part)), part["label"], part["group"]):
            y_tr = part.iloc[loc_tr]["label"]
            y_val = part.iloc[loc_val]["label"]
            if y_tr.nunique() < 2 or y_val.nunique() < 2:
                continue
            size_error = abs(len(loc_val) / len(part) - valid_size)
            ratio_error = abs(float(y_val.mean()) - global_rate)
            score = ratio_error + size_error
            if best is None or score < best[0]:
                best = (score, loc_tr, loc_val)
        if best is None:
            raise ValueError(
                "cannot create a grouped early-stopping split containing "
                "both labels on each side. Reduce valid_size/CV folds or add "
                "more independent positive and negative groups")
        _, loc_tr, loc_val = best
    else:
        inner = StratifiedShuffleSplit(n_splits=1, test_size=valid_size,
                                       random_state=seed)
        loc_tr, loc_val = next(inner.split(X.iloc[tr_idx], y.iloc[tr_idx]))
    return tr_idx[loc_tr], tr_idx[loc_val]


def _split_counts(y, groups, idx):
    """Row/class/group counts for one train/valid/OOF partition."""
    yy = y.iloc[idx]
    out = {
        "n_rows": int(len(idx)),
        "n_correct": int((yy == 1).sum()),
        "n_error": int((yy == 0).sum()),
    }
    if groups is None:
        out.update({"n_groups": None, "n_correct_groups": None,
                    "n_error_groups": None, "n_mixed_groups": None})
        return out

    part = pd.DataFrame({
        "group": groups.iloc[idx].to_numpy(),
        "label": yy.to_numpy(),
    })
    grouped = part.groupby("group", sort=False)["label"]
    nunique = grouped.nunique()
    has_correct = grouped.apply(lambda s: bool((s == 1).any()))
    has_error = grouped.apply(lambda s: bool((s == 0).any()))
    out.update({
        "n_groups": int(len(nunique)),
        "n_correct_groups": int(has_correct.sum()),
        "n_error_groups": int(has_error.sum()),
        "n_mixed_groups": int(((has_correct) & (has_error)).sum()),
    })
    return out


def _validate_split_counts(fold, split_counts, min_class_groups, grouped):
    """Fail fast when a fold cannot support stable binary evaluation."""
    for split_name, counts in split_counts.items():
        if counts["n_correct"] == 0 or counts["n_error"] == 0:
            raise ValueError(
                f"fold {fold} {split_name} has a missing class: {counts}")
        if grouped:
            minority = min(
                counts["n_correct_groups"], counts["n_error_groups"])
            if minority < min_class_groups:
                raise ValueError(
                    f"fold {fold} {split_name} has only {minority} groups in "
                    f"its minority class; require >= {min_class_groups}. "
                    "Reduce training.cv_folds / valid_size or collect more "
                    f"minority-class sequences. Counts: {counts}")


def _predefined_cv_splits(fold_ids, n_rows, n_folds, groups=None):
    """Validate and expand a reusable row-level outer-fold assignment."""
    raw = np.asarray(fold_ids)
    if raw.shape != (n_rows,):
        raise ValueError(
            f"predefined_fold_ids must have shape ({n_rows},), got "
            f"{raw.shape}")
    if not np.issubdtype(raw.dtype, np.integer):
        if not np.isfinite(raw.astype("f8")).all() or not np.equal(
                raw, raw.astype(int)).all():
            raise ValueError("predefined_fold_ids must contain integers")
    fold_ids = raw.astype(int)
    expected = set(range(n_folds))
    actual = set(fold_ids.tolist())
    if actual != expected:
        raise ValueError(
            f"predefined_fold_ids must contain exactly {sorted(expected)}, "
            f"got {sorted(actual)}")
    if groups is not None:
        assigned = pd.DataFrame({
            "group": np.asarray(groups), "fold": fold_ids,
        }).groupby("group", sort=False)["fold"].nunique()
        if (assigned > 1).any():
            raise ValueError(
                "predefined_fold_ids split at least one group across folds")
    return [
        (np.flatnonzero(fold_ids != fold),
         np.flatnonzero(fold_ids == fold))
        for fold in range(n_folds)
    ]


def _predefined_inner_split(tr_idx, te_idx, mask, n_rows, groups=None):
    """Apply one reusable inner-validation assignment for an outer fold."""
    mask = np.asarray(mask)
    if mask.shape != (n_rows,):
        raise ValueError(
            f"predefined inner-valid mask must have shape ({n_rows},), got "
            f"{mask.shape}")
    if mask.dtype != bool:
        if not set(np.unique(mask).tolist()).issubset({0, 1, False, True}):
            raise ValueError("predefined inner-valid mask must be boolean")
        mask = mask.astype(bool)
    if mask[te_idx].any():
        raise ValueError(
            "predefined inner-valid mask marks rows from the outer OOF fold")
    val = tr_idx[mask[tr_idx]]
    tr2 = tr_idx[~mask[tr_idx]]
    if not len(val) or not len(tr2):
        raise ValueError("predefined inner split has an empty train/valid side")
    if groups is not None:
        if set(groups.iloc[tr2]).intersection(set(groups.iloc[val])):
            raise ValueError(
                "predefined inner-valid mask splits a group across train/valid")
    return tr2, val


def assemble_oof(df, X, y, groups, cfg, feature_cols, model_prefix,
                 return_fold_ids=False, predefined_fold_ids=None,
                 predefined_inner_valid=None):
    """Train one model per fold, collect leak-free OOF preds, save fold models.

    Returns (oof_proba, fold_metrics, model_paths), plus fold IDs when
    ``return_fold_ids=True``. lightgbm is imported lazily here.
    """
    # df: unused here; kept for call-site symmetry (caller uses it for label audit)
    from models.model_manager import ModelManager

    n_folds = int(cfg["training"].get("cv_folds", 5))
    seed = int(cfg["training"].get("cv_seed", 42))
    valid_size = float(cfg["training"].get("valid_size", 0.15))
    min_class_groups = int(
        cfg["training"].get("min_class_groups_per_split", 1))
    target_fprs, _ = _operating_targets(cfg)
    if min_class_groups < 1:
        raise ValueError("training.min_class_groups_per_split must be >= 1")
    grp_vals = None if groups is None else groups.values

    if predefined_fold_ids is None:
        splits = make_cv_splits(
            y.values, grp_vals, n_folds=n_folds, seed=seed)
    else:
        splits = _predefined_cv_splits(
            predefined_fold_ids, len(y), n_folds, groups=groups)
    if predefined_inner_valid is not None:
        missing_masks = set(range(n_folds)) - set(predefined_inner_valid)
        if missing_masks:
            raise ValueError(
                "predefined_inner_valid is missing outer folds: "
                f"{sorted(missing_masks)}")
    oof = np.full(len(y), np.nan)
    oof_folds = np.full(len(y), -1, dtype=int)
    fold_metrics, model_paths = [], []

    for k, (tr_idx, te_idx) in enumerate(splits):
        # 折内早停验证集（分组，避免污染 OOF 折）
        if predefined_inner_valid is None:
            tr2, val = _inner_split(
                X, y, groups, tr_idx, valid_size, seed + k)
        else:
            tr2, val = _predefined_inner_split(
                tr_idx, te_idx, predefined_inner_valid[k], len(y),
                groups=groups)
        counts = {
            "train": _split_counts(y, groups, tr2),
            "valid": _split_counts(y, groups, val),
            "oof_test": _split_counts(y, groups, te_idx),
        }
        _validate_split_counts(
            k, counts, min_class_groups, grouped=groups is not None)
        logging.info(
            "fold %d split counts: train=%s valid=%s oof_test=%s",
            k, counts["train"], counts["valid"], counts["oof_test"])

        model = ModelManager.create(cfg, feature_names=feature_cols)
        model.fit(X.iloc[tr2], y.iloc[tr2], X.iloc[val], y.iloc[val])
        oof[te_idx] = model.predict_proba(X.iloc[te_idx])
        oof_folds[te_idx] = k

        mp = f"{model_prefix}.fold{k}.txt"
        os.makedirs(os.path.dirname(mp) or ".", exist_ok=True)
        model_tmp = f"{mp}.tmp.{os.getpid()}"
        model.save(model_tmp)
        os.replace(model_tmp, mp)
        model_paths.append(mp)

        te_y, te_p = y.iloc[te_idx].values, oof[te_idx]
        if len(set(te_y.tolist())) < 2:
            fold_metrics.append({"fold": k, "roc_auc": float("nan"),
                                 "error_pr_auc": float("nan"),
                                 "fnr_at_fpr5": float("nan"),
                                 "error_recall_at_fpr10": float("nan"),
                                 "calibration_operating_points": {},
                                 "split_counts": counts})
        else:
            metrics = evaluate_oof(te_y, te_p)
            calibration = {}
            for target in target_fprs:
                threshold = threshold_at_fpr(te_y, te_p, target)
                point = evaluate_at_threshold(te_y, te_p, threshold)
                point.update({
                    "target_fpr": target,
                    "threshold_source": "this_model_outer_oof_fold",
                })
                calibration[_op_key(target)] = point
            booster = getattr(model, "model", None)
            best_iteration = getattr(booster, "best_iteration", None)
            if not best_iteration and booster is not None:
                best_iteration = booster.current_iteration()
            fold_metrics.append({
                "fold": k,
                "roc_auc": metrics["roc_auc"],
                "error_pr_auc": metrics["error_pr_auc"],
                "fnr_at_fpr5": metrics["fnr_at_fpr5"],
                "error_recall_at_fpr10": metrics[
                    "error_recall_at_fpr10"],
                "best_iteration": (
                    int(best_iteration) if best_iteration is not None else None),
                "calibration_operating_points": calibration,
                "split_counts": counts,
            })

    assert not np.isnan(oof).any(), "OOF has NaN — some sample never predicted"
    assert (oof_folds >= 0).all(), "some sample has no OOF fold assignment"
    result = (oof, fold_metrics, model_paths)
    return (*result, oof_folds) if return_fold_ids else result


def main(argv=None):
    args = _build_parser().parse_args(argv)
    with open(args.config) as f:
        cfg = yaml.safe_load(f)

    model_prefix, result_path, _ = derive_paths(cfg)
    artifact_paths = _artifact_paths(result_path)
    _assert_output_bundle_available(artifact_paths, args.overwrite)
    if os.path.dirname(args.logpath):
        os.makedirs(os.path.dirname(args.logpath), exist_ok=True)
    logging.basicConfig(
        level=logging.INFO, force=True,
        handlers=[logging.FileHandler(
            args.logpath, mode="w", encoding="utf-8")])
    logging.info("starting CV experiment %s from %s", args.name, args.config)

    target_col = cfg["data"]["target_col"]
    train_files = cfg["data"]["train_files"]
    raw_train_df = read_dataframe(train_files)
    df, train_cohort_audit = apply_training_cohort(
        raw_train_df, cfg["data"].get("cohort"), target_col=target_col)
    feature_cols = resolve_configured_feature_cols(
        cfg["data"], train_files, target_col)
    X = df[feature_cols]
    y = df[target_col]
    group_col = cfg["data"].get("group_col")
    _validate_frame(df, feature_cols, target_col, group_col)
    groups = df[group_col] if group_col else None
    logging.info(
        "training cohort %s: %s -> %s; arm=%s, n_features=%d",
        train_cohort_audit["name"], train_cohort_audit["before"],
        train_cohort_audit["after"],
        cfg["data"].get("feature_arm", "legacy_auto"), len(feature_cols))

    oof, fold_metrics, model_paths, oof_folds = assemble_oof(
        df, X, y, groups, cfg, feature_cols, model_prefix,
        return_fold_ids=True)

    target_fprs, primary_target_fpr = _operating_targets(cfg)
    operating_points = {}
    for target_fpr in target_fprs:
        error_threshold = threshold_at_fpr(y, oof, target_fpr)
        train_metrics = evaluate_at_threshold(y, oof, error_threshold)
        operating_points[_op_key(target_fpr)] = {
            "metric_semantics": METRIC_SEMANTICS_VERSION,
            "positive_class": "incorrect_identification",
            "target_fpr": target_fpr,
            "error_threshold": error_threshold,
            "trust_threshold": 1.0 - error_threshold,
            "threshold_source": "pooled_train_oof_single_member_scores",
            "decision_rule": (
                "error_score >= error_threshold => incorrect_identification; "
                "error_score = 1 - trust_score"),
            "train_oof_metrics": train_metrics,
        }

    test_files = cfg["data"].get("test_files")
    test_cohort_audit = None
    test_missingness = None
    sequence_overlap = None
    test_prediction_frame = None
    if test_files and set(test_files) != set(train_files):
        raw_test_df = read_dataframe(test_files)
        test_df, test_cohort_audit = apply_training_cohort(
            raw_test_df, cfg["data"].get("cohort"), target_col=target_col)
        _validate_frame(test_df, feature_cols, target_col, group_col)
        eval_df = test_df
        eval_y = test_df[target_col]
        ens_proba, member_retrospective, test_agg, cross_details = (
            evaluate_cross_test(
                model_paths, test_df[feature_cols], eval_y.values,
                fold_metrics=fold_metrics, return_details=True))
        eval_proba = ens_proba
        retrospective = evaluate_oof(eval_y, eval_proba)
        train_oof = evaluate_oof(y, oof)
        summary = {
            "metric_semantics": METRIC_SEMANTICS_VERSION,
            "positive_class": "incorrect_identification",
            "mode": "cross_test",
            "roc_auc": retrospective["roc_auc"],
            "error_pr_auc": retrospective["error_pr_auc"],
            "retrospective_test_working_points": {
                "threshold_source": "external_test_labels",
                "interpretation": (
                    "oracle discrimination analysis; not deployable locked "
                    "threshold performance"),
                "fnr_at_fpr5": retrospective["fnr_at_fpr5"],
                "error_recall_at_fpr10": retrospective[
                    "error_recall_at_fpr10"],
                "working_points": retrospective["working_points"],
            },
            "member_model_retrospective_metrics": member_retrospective,
            "train_oof_roc_auc": train_oof["roc_auc"],
            "train_oof_error_pr_auc": train_oof["error_pr_auc"],
            "train_oof_fnr_at_fpr5": train_oof["fnr_at_fpr5"],
            "train_fold_metrics": fold_metrics,
            "train_oof_metrics_by_source": _metrics_by_source(df, y, oof),
            "retrospective_test_metrics_by_source": _metrics_by_source(
                test_df, eval_y, eval_proba),
        }
        summary.update(test_agg)
        for key, locked in test_agg["locked_operating_points"].items():
            operating_points[key]["external_ensemble"] = locked
            operating_points[key]["test_metrics"] = locked["test_metrics"]
        test_missingness = _missingness_audit(
            test_df, feature_cols, target_col)
        sequence_overlap = _sequence_overlap(df, test_df, group_col)
        identity = [column for column in (
            _SOURCE_FILE, _SOURCE_ROW, "sequence", "charge", target_col,
            "label_type") if column in test_df]
        test_prediction_frame = test_df[identity].copy()
        test_prediction_frame["ensemble_trust_score"] = ens_proba
        test_prediction_frame["ensemble_error_score"] = 1.0 - ens_proba
        for k, scores in enumerate(cross_details["member_trust_scores"]):
            test_prediction_frame[f"member_{k}_trust_score"] = scores
        for key, fractions in cross_details[
                "fold_calibrated_error_vote_fractions"].items():
            test_prediction_frame[f"{key}_error_vote_fraction"] = fractions
    else:
        eval_df = df
        eval_y = y
        eval_proba = oof
        summary = evaluate_oof(eval_y, eval_proba)
        aucs = [m["roc_auc"] for m in fold_metrics
                if not np.isnan(m["roc_auc"])]
        error_pr_aucs = [m["error_pr_auc"] for m in fold_metrics
                         if not np.isnan(m["error_pr_auc"])]
        fnrs = [m["fnr_at_fpr5"] for m in fold_metrics
                if not np.isnan(m["fnr_at_fpr5"])]
        recalls = [m["error_recall_at_fpr10"] for m in fold_metrics
                   if not np.isnan(m["error_recall_at_fpr10"])]
        summary.update({
            "mode": "in_sample",
            "fold_metrics": fold_metrics,
            "roc_auc_mean": float(np.mean(aucs)) if aucs else float("nan"),
            "roc_auc_std": float(np.std(aucs)) if aucs else float("nan"),
            "error_pr_auc_mean": float(np.mean(error_pr_aucs))
            if error_pr_aucs else float("nan"),
            "error_pr_auc_std": float(np.std(error_pr_aucs))
            if error_pr_aucs else float("nan"),
            "fnr_at_fpr5_mean": float(np.mean(fnrs)) if fnrs else float("nan"),
            "fnr_at_fpr5_std": float(np.std(fnrs)) if fnrs else float("nan"),
            "error_recall_at_fpr10_mean": float(np.mean(recalls))
            if recalls else float("nan"),
            "error_recall_at_fpr10_std": float(np.std(recalls))
            if recalls else float("nan"),
            "fold_dispersion_note": (
                "fold std uses ddof=0 and is descriptive dispersion, not a "
                "confidence interval"),
            "oof_metrics_by_source": _metrics_by_source(df, y, oof),
        })

    primary_key = _op_key(primary_target_fpr)
    summary["operating_points"] = operating_points
    summary["operating_point"] = operating_points[primary_key]
    summary["primary_operating_point"] = primary_key

    summary.update({
        "cv_folds": len(fold_metrics),
        "model_paths": model_paths,
        "n_actual_correct": int((eval_y == 1).sum()),
        "n_actual_error": int((eval_y == 0).sum()),
        "name": args.name,
        "statistical_notes": [
            "fold/member standard deviations are descriptive dispersion, "
            "not confidence intervals",
            "clean/neg05/neg10/neg15/neg20 are nested sensitivity cohorts, "
            "not independent experimental repeats",
        ],
    })
    train_missingness = _missingness_audit(df, feature_cols, target_col)
    summary["experiment"] = {
        "feature_arm": cfg["data"].get("feature_arm"),
        "cohort": train_cohort_audit["name"],
        "drop_features": list(cfg["data"].get("drop_features") or []),
        "n_features": len(feature_cols),
        "feature_cols": feature_cols,
        "train_cohort": train_cohort_audit,
        "test_cohort": test_cohort_audit,
        "train_missing_rate": train_missingness["overall"],
        "train_missingness": train_missingness,
        "test_missingness": test_missingness,
        "train_test_sequence_overlap": sequence_overlap,
    }

    audit_cfg = cfg.get("audit", {})
    susp, suspect_total = audit_labels(
        eval_df, eval_proba, label_col=target_col,
        threshold=float(audit_cfg.get("suspect_threshold", 0.9)),
        top_n=int(audit_cfg.get("suspect_top_n", 200)), return_total=True)
    summary["label_audit"] = {
        "threshold": float(audit_cfg.get("suspect_threshold", 0.9)),
        "total_candidates": suspect_total,
        "rows_saved": len(susp),
        "is_capped": suspect_total > len(susp),
    }

    identity = [column for column in (
        _SOURCE_FILE, _SOURCE_ROW, "sequence", "charge", target_col,
        "label_type") if column in df]
    oof_frame = df[identity].copy()
    oof_frame["oof_fold"] = oof_folds
    oof_frame["trust_score"] = oof
    oof_frame["error_score"] = 1.0 - oof
    summary["artifacts"] = {
        "result_json": artifact_paths["result"],
        "suspects_csv": artifact_paths["suspects"],
        "oof_predictions_csv": artifact_paths["oof_predictions"],
        "test_predictions_csv": (
            artifact_paths["test_predictions"]
            if test_prediction_frame is not None else None),
    }
    summary["provenance"] = _provenance(
        args, cfg, train_files, test_files or [])

    # Write the JSON last: its presence is the completion marker for the bundle.
    _atomic_csv(artifact_paths["suspects"], susp)
    _atomic_csv(artifact_paths["oof_predictions"], oof_frame)
    if test_prediction_frame is not None:
        _atomic_csv(artifact_paths["test_predictions"], test_prediction_frame)
    _atomic_json(artifact_paths["result"], summary)

    primary = operating_points[primary_key]
    applied = primary.get("test_metrics", primary["train_oof_metrics"])
    logging.info(
        "CV(%s) done: error-positive ROC-AUC=%.4f; target FPR<=%.3f, "
        "error threshold=%.6g, observed FPR=%s, error recall=%s; "
        "%d false-negative candidates -> %s",
        summary["mode"], summary["roc_auc"], primary_target_fpr,
        primary["error_threshold"],
        "n/a" if applied["fpr"] is None else f'{applied["fpr"]:.4f}',
        "n/a" if applied["error_recall"] is None
        else f'{applied["error_recall"]:.4f}',
        suspect_total, artifact_paths["suspects"])
    return summary


def predict_ensemble(model_paths, X):
    """Ensemble score for NEW data = mean of the per-fold models' predictions.

    X: numpy array or DataFrame with the same feature columns/order used in
    training (lightgbm matches by position). Used to score external data
    (e.g. cross_test, production) — NOT for in-sample eval (use OOF for that).
    """
    probas = _predict_members(model_paths, X)
    return average_proba(probas)


def _predict_members(model_paths, X):
    """Predict with every member and validate pandas feature names exactly."""
    import lightgbm as lgb
    probas = []
    for path in model_paths:
        booster = lgb.Booster(model_file=path)
        if isinstance(X, pd.DataFrame):
            expected = list(booster.feature_name())
            actual = list(X.columns)
            if actual != expected:
                raise ValueError(
                    f"model/test feature schema mismatch for {path}: "
                    f"expected={expected}, actual={actual}")
            prediction = booster.predict(X, validate_features=True)
        else:
            prediction = booster.predict(X)
        probas.append(prediction)
    return probas


def evaluate_cross_test(model_paths, X, y, fold_metrics=None,
                        return_details=False):
    """Score external test data with each fold model + the ensemble.

    Returns (ens_proba, per_fold, agg), plus member scores/votes when requested:
    - ens_proba: mean of the K fold models' predictions on X (cross_test score).
    - per_fold: error-positive metrics for each fold model on the
      external test (NaN when y is single-class).
    - agg: fold means/std under the same metric convention.
    lightgbm imported lazily.
    """
    y = np.asarray(y)
    probas = _predict_members(model_paths, X)
    ens = average_proba(probas)
    one_class = len(set(y.tolist())) < 2
    per_fold = []
    for k, p in enumerate(probas):
        if one_class:
            per_fold.append({
                "fold": k, "roc_auc": float("nan"),
                "error_pr_auc": float("nan"),
                "oracle_test_fnr_at_fpr5": float("nan"),
                "oracle_test_error_recall_at_fpr10": float("nan")})
        else:
            metrics = evaluate_oof(y, p)
            per_fold.append({
                "fold": k,
                "roc_auc": metrics["roc_auc"],
                "error_pr_auc": metrics["error_pr_auc"],
                "oracle_test_fnr_at_fpr5": metrics["fnr_at_fpr5"],
                "oracle_test_error_recall_at_fpr10": metrics[
                    "error_recall_at_fpr10"],
                "working_point_source": "external_test_labels",
            })
    aucs = [m["roc_auc"] for m in per_fold
            if not np.isnan(m["roc_auc"])]
    error_pr_aucs = [m["error_pr_auc"] for m in per_fold
                     if not np.isnan(m["error_pr_auc"])]
    fnrs = [m["oracle_test_fnr_at_fpr5"] for m in per_fold
            if not np.isnan(m["oracle_test_fnr_at_fpr5"])]
    recalls = [m["oracle_test_error_recall_at_fpr10"] for m in per_fold
               if not np.isnan(m["oracle_test_error_recall_at_fpr10"])]
    agg = {
        "member_model_roc_auc_mean": float(np.mean(aucs)) if aucs else float("nan"),
        "member_model_roc_auc_std": float(np.std(aucs)) if aucs else float("nan"),
        "member_model_error_pr_auc_mean": float(np.mean(error_pr_aucs))
        if error_pr_aucs else float("nan"),
        "member_model_error_pr_auc_std": float(np.std(error_pr_aucs))
        if error_pr_aucs else float("nan"),
        "member_model_oracle_fnr_at_fpr5_mean": (
            float(np.mean(fnrs)) if fnrs else float("nan")),
        "member_model_oracle_fnr_at_fpr5_std": (
            float(np.std(fnrs)) if fnrs else float("nan")),
        "member_model_oracle_error_recall_at_fpr10_mean": float(np.mean(recalls))
        if recalls else float("nan"),
        "member_model_oracle_error_recall_at_fpr10_std": float(np.std(recalls))
        if recalls else float("nan"),
        "member_model_dispersion_note": (
            "same external test set scored by overlapping training models; "
            "std is not an independent-test confidence interval"),
    }
    details = {
        "member_trust_scores": probas,
        "fold_calibrated_error_vote_fractions": {},
    }
    locked = {}
    if fold_metrics is not None:
        if len(fold_metrics) != len(probas):
            raise ValueError("fold_metrics/model_paths length mismatch")
        keys = set(fold_metrics[0].get(
            "calibration_operating_points", {}))
        for metric in fold_metrics[1:]:
            if set(metric.get("calibration_operating_points", {})) != keys:
                raise ValueError("inconsistent fold calibration operating points")
        for key in sorted(keys):
            votes = []
            thresholds = []
            for scores, metric in zip(probas, fold_metrics):
                threshold = metric[
                    "calibration_operating_points"][key]["error_threshold"]
                thresholds.append(float(threshold))
                votes.append((1.0 - np.asarray(scores)) >= threshold)
            vote_fraction = np.mean(np.vstack(votes), axis=0)
            # Canonical helper remains the only confusion-matrix
            # implementation.  The vote fraction is an error score, hence the
            # equivalent trust score is 1-vote_fraction.
            test_metrics = evaluate_at_threshold(
                y, 1.0 - vote_fraction, error_threshold=0.5)
            locked[key] = {
                "method": "fold_calibrated_majority_vote",
                "calibration_source": "each_member_outer_oof_fold",
                "member_target_fpr": metric[
                    "calibration_operating_points"][key]["target_fpr"],
                "member_error_thresholds": thresholds,
                "vote_error_threshold": 0.5,
                "ensemble_fpr_guarantee": (
                    "not guaranteed by member calibration; use observed "
                    "test_metrics.fpr"),
                "test_metrics": test_metrics,
            }
            details["fold_calibrated_error_vote_fractions"][key] = vote_fraction
    agg["locked_operating_points"] = locked
    result = (ens, per_fold, agg)
    return (*result, details) if return_details else result


if __name__ == "__main__":
    main()
