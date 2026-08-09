"""Cross-validated training entry for spec_trainer (production LightGBM).

One CV pass yields OOF predictions that drive: honest evaluation, fold
ensemble (saved per-fold models), and label-noise audit. Does NOT touch
main.py (single-holdout flow unchanged). lightgbm is imported lazily inside
assemble_oof / predict_ensemble / evaluate_cross_test so this module (and path
helpers) import without it.
"""
import argparse
import json
import logging
import os
import re

import numpy as np
import pandas as pd
import yaml

from cv_core import (average_proba, audit_labels, evaluate_at_threshold,
                     evaluate_oof, fnr_at_fpr5, make_cv_splits,
                     threshold_at_fpr)
from cohort import apply_training_cohort
from feature_cols import resolve_configured_feature_cols


def read_dataframe(train_files):
    """Concatenate feature CSVs, keeping all columns (sequence/charge needed)."""
    return pd.concat([pd.read_csv(f) for f in train_files], ignore_index=True)


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
    return p


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
        "n_pos": int((yy == 1).sum()),
        "n_neg": int((yy == 0).sum()),
    }
    if groups is None:
        out.update({"n_groups": None, "n_pos_groups": None,
                    "n_neg_groups": None, "n_mixed_groups": None})
        return out

    part = pd.DataFrame({
        "group": groups.iloc[idx].to_numpy(),
        "label": yy.to_numpy(),
    })
    grouped = part.groupby("group", sort=False)["label"]
    nunique = grouped.nunique()
    has_pos = grouped.apply(lambda s: bool((s == 1).any()))
    has_neg = grouped.apply(lambda s: bool((s == 0).any()))
    out.update({
        "n_groups": int(len(nunique)),
        "n_pos_groups": int(has_pos.sum()),
        "n_neg_groups": int(has_neg.sum()),
        "n_mixed_groups": int(((has_pos) & (has_neg)).sum()),
    })
    return out


def _validate_split_counts(fold, split_counts, min_class_groups, grouped):
    """Fail fast when a fold cannot support stable binary evaluation."""
    for split_name, counts in split_counts.items():
        if counts["n_pos"] == 0 or counts["n_neg"] == 0:
            raise ValueError(
                f"fold {fold} {split_name} has a missing class: {counts}")
        if grouped:
            minority = min(counts["n_pos_groups"], counts["n_neg_groups"])
            if minority < min_class_groups:
                raise ValueError(
                    f"fold {fold} {split_name} has only {minority} groups in "
                    f"its minority class; require >= {min_class_groups}. "
                    "Reduce training.cv_folds / valid_size or collect more "
                    f"minority-class sequences. Counts: {counts}")


def assemble_oof(df, X, y, groups, cfg, feature_cols, model_prefix):
    """Train one model per fold, collect leak-free OOF preds, save fold models.

    Returns (oof_proba, fold_metrics, model_paths). lightgbm imported here.
    """
    # df: unused here; kept for call-site symmetry (caller uses it for label audit)
    from sklearn.metrics import average_precision_score, roc_auc_score
    from models.model_manager import ModelManager

    n_folds = int(cfg["training"].get("cv_folds", 5))
    seed = int(cfg["training"].get("cv_seed", 42))
    valid_size = float(cfg["training"].get("valid_size", 0.15))
    min_class_groups = int(
        cfg["training"].get("min_class_groups_per_split", 1))
    if min_class_groups < 1:
        raise ValueError("training.min_class_groups_per_split must be >= 1")
    grp_vals = None if groups is None else groups.values

    splits = make_cv_splits(y.values, grp_vals, n_folds=n_folds, seed=seed)
    oof = np.full(len(y), np.nan)
    fold_metrics, model_paths = [], []

    for k, (tr_idx, te_idx) in enumerate(splits):
        # 折内早停验证集（分组，避免污染 OOF 折）
        tr2, val = _inner_split(X, y, groups, tr_idx, valid_size, seed)
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

        mp = f"{model_prefix}.fold{k}.txt"
        os.makedirs(os.path.dirname(mp) or ".", exist_ok=True)
        model.save(mp)
        model_paths.append(mp)

        te_y, te_p = y.iloc[te_idx].values, oof[te_idx]
        if len(set(te_y.tolist())) < 2:               # 该折单类 → auc 无定义
            fold_metrics.append({"fold": k, "auc": float("nan"),
                                 "pr_auc_pos": float("nan"),
                                 "pr_auc_neg": float("nan"),
                                 "fnr_at_fpr5": float("nan"),
                                 "split_counts": counts})
        else:
            fold_metrics.append({"fold": k,
                                 "auc": float(roc_auc_score(te_y, te_p)),
                                 "pr_auc_pos": float(
                                     average_precision_score(te_y, te_p)),
                                 "pr_auc_neg": float(
                                     average_precision_score(
                                         1 - te_y, 1.0 - te_p)),
                                 "fnr_at_fpr5": float(fnr_at_fpr5(te_y, te_p)),
                                 "split_counts": counts})

    assert not np.isnan(oof).any(), "OOF has NaN — some sample never predicted"
    return oof, fold_metrics, model_paths


def main(argv=None):
    args = _build_parser().parse_args(argv)
    if os.path.dirname(args.logpath):
        os.makedirs(os.path.dirname(args.logpath), exist_ok=True)
    logging.basicConfig(
        level=logging.INFO,
        handlers=[logging.FileHandler(args.logpath, encoding="utf-8")])

    with open(args.config) as f:
        cfg = yaml.safe_load(f)

    target_col = cfg["data"]["target_col"]
    train_files = cfg["data"]["train_files"]
    raw_train_df = read_dataframe(train_files)
    df, train_cohort_audit = apply_training_cohort(
        raw_train_df, cfg["data"].get("cohort"), target_col=target_col)
    feature_cols = resolve_configured_feature_cols(
        cfg["data"], train_files, target_col)
    X = df[feature_cols]
    y = df[target_col]
    logging.info(
        "training cohort %s: %s -> %s; arm=%s, n_features=%d",
        train_cohort_audit["name"], train_cohort_audit["before"],
        train_cohort_audit["after"],
        cfg["data"].get("feature_arm", "legacy_auto"), len(feature_cols))

    group_col = cfg["data"].get("group_col")
    if group_col and group_col in df.columns:
        groups = df[group_col]
    else:
        groups = None
        if group_col:
            logging.warning("group_col %r not in data — ungrouped CV", group_col)

    model_prefix, result_path, suspects_path = derive_paths(cfg)
    oof, fold_metrics, model_paths = assemble_oof(
        df, X, y, groups, cfg, feature_cols, model_prefix)

    # Deployment operating point: choose once from leak-free TRAIN OOF
    # predictions, then lock it before touching an external test set.
    # The strict selector handles ties conservatively, so empirical train-OOF
    # FPR cannot exceed the requested target.
    op_cfg = cfg.get("operating_point", {})
    target_fpr = float(op_cfg.get("target_fpr", 0.10))
    decision_threshold = threshold_at_fpr(y, oof, target_fpr)
    train_threshold_metrics = evaluate_at_threshold(
        y, oof, decision_threshold)
    operating_point = {
        "target_fpr": target_fpr,
        "threshold": decision_threshold,
        "threshold_source": "train_oof",
        "decision_rule": "score >= threshold => positive",
        "train_oof_metrics": train_threshold_metrics,
    }

    test_files = cfg["data"].get("test_files")
    test_cohort_audit = None
    if test_files and set(test_files) != set(train_files):
        # cross_test 模式：ensemble 给外部测试集打分，在外部测试集评估/审计
        raw_test_df = read_dataframe(test_files)
        test_df, test_cohort_audit = apply_training_cohort(
            raw_test_df, cfg["data"].get("cohort"), target_col=target_col)
        eval_df = test_df
        eval_y = test_df[target_col]
        # position-matched: feature_cols order must equal training's df[feature_cols]
        ens_proba, test_per_fold, test_agg = evaluate_cross_test(
            model_paths, test_df[feature_cols].values, eval_y.values)
        eval_proba = ens_proba
        # generic auc/fnr/working_points on the ensemble scores (NOT OOF in cross_test)
        summary = evaluate_oof(eval_y, eval_proba)
        train_oof = evaluate_oof(y, oof)
        summary.update({
            "mode": "cross_test",
            "test_per_fold": test_per_fold,
            "train_oof_auc": train_oof["auc"],
            "train_oof_fnr_at_fpr5": train_oof["fnr_at_fpr5"],
            "train_fold_metrics": fold_metrics,
        })
        summary.update(test_agg)
        # This is the deployable external-test operating point: the threshold
        # was fixed from train OOF and is NOT re-selected from test labels.
        operating_point["test_metrics"] = evaluate_at_threshold(
            eval_y, eval_proba, decision_threshold)
    else:
        # in_sample 模式（行为同上一里程碑）
        eval_df = df
        eval_y = y
        eval_proba = oof
        summary = evaluate_oof(eval_y, eval_proba)
        aucs = [m["auc"] for m in fold_metrics if not np.isnan(m["auc"])]
        pr_aucs_pos = [m["pr_auc_pos"] for m in fold_metrics
                       if not np.isnan(m["pr_auc_pos"])]
        pr_aucs_neg = [m["pr_auc_neg"] for m in fold_metrics
                       if not np.isnan(m["pr_auc_neg"])]
        fnrs = [m["fnr_at_fpr5"] for m in fold_metrics
                if not np.isnan(m["fnr_at_fpr5"])]
        summary.update({
            "mode": "in_sample",
            "fold_metrics": fold_metrics,
            "auc_mean": float(np.mean(aucs)) if aucs else float("nan"),
            "auc_std": float(np.std(aucs)) if aucs else float("nan"),
            "pr_auc_pos_mean": float(np.mean(pr_aucs_pos))
            if pr_aucs_pos else float("nan"),
            "pr_auc_pos_std": float(np.std(pr_aucs_pos))
            if pr_aucs_pos else float("nan"),
            "pr_auc_neg_mean": float(np.mean(pr_aucs_neg))
            if pr_aucs_neg else float("nan"),
            "pr_auc_neg_std": float(np.std(pr_aucs_neg))
            if pr_aucs_neg else float("nan"),
            "fnr_at_fpr5_mean": float(np.mean(fnrs)) if fnrs else float("nan"),
            "fnr_at_fpr5_std": float(np.std(fnrs)) if fnrs else float("nan"),
        })

    summary["operating_point"] = operating_point

    summary.update({
        "cv_folds": len(fold_metrics),
        "model_paths": model_paths,
        "n_pos": int((eval_y == 1).sum()),
        "n_neg": int((eval_y == 0).sum()),
        "name": args.name,
    })
    summary["experiment"] = {
        "feature_arm": cfg["data"].get("feature_arm"),
        "cohort": train_cohort_audit["name"],
        "drop_features": list(cfg["data"].get("drop_features") or []),
        "n_features": len(feature_cols),
        "feature_cols": feature_cols,
        "train_cohort": train_cohort_audit,
        "test_cohort": test_cohort_audit,
        "train_missing_rate": {
            column: float(df[column].isna().mean())
            for column in feature_cols
        },
    }
    os.makedirs(os.path.dirname(result_path) or ".", exist_ok=True)
    with open(result_path, "w") as f:
        json.dump(summary, f, indent=2)

    audit_cfg = cfg.get("audit", {})
    susp = audit_labels(
        eval_df, eval_proba, label_col=target_col,
        threshold=float(audit_cfg.get("suspect_threshold", 0.9)),
        top_n=int(audit_cfg.get("suspect_top_n", 200)))
    susp.to_csv(suspects_path, index=False)

    applied = operating_point.get(
        "test_metrics", operating_point["train_oof_metrics"])
    logging.info(
        "CV(%s) done: AUC=%.4f; target FPR<=%.3f, threshold=%.6g, "
        "observed FPR=%s, pos recall=%s; %d suspects -> %s",
        summary["mode"], summary["auc"], target_fpr, decision_threshold,
        "n/a" if applied["fpr"] is None else f'{applied["fpr"]:.4f}',
        "n/a" if applied["pos_recall"] is None
        else f'{applied["pos_recall"]:.4f}',
        len(susp), suspects_path)
    return summary


def predict_ensemble(model_paths, X):
    """Ensemble score for NEW data = mean of the per-fold models' predictions.

    X: numpy array or DataFrame with the same feature columns/order used in
    training (lightgbm matches by position). Used to score external data
    (e.g. cross_test, production) — NOT for in-sample eval (use OOF for that).
    """
    import lightgbm as lgb
    probas = [lgb.Booster(model_file=p).predict(X) for p in model_paths]
    return average_proba(probas)


def evaluate_cross_test(model_paths, X, y):
    """Score external test data with each fold model + the ensemble.

    Returns (ens_proba, per_fold, agg):
    - ens_proba: mean of the K fold models' predictions on X (cross_test score).
    - per_fold: [{fold, auc, fnr_at_fpr5}, ...] for each fold model on the
      external test (NaN when y is single-class).
    - agg: {test_auc_mean/std, test_fnr_at_fpr5_mean/std} over non-NaN folds.
    lightgbm imported lazily.
    """
    import lightgbm as lgb
    from sklearn.metrics import average_precision_score, roc_auc_score
    y = np.asarray(y)
    probas = [lgb.Booster(model_file=p).predict(X) for p in model_paths]
    ens = average_proba(probas)
    one_class = len(set(y.tolist())) < 2
    per_fold = []
    for k, p in enumerate(probas):
        if one_class:
            per_fold.append({"fold": k, "auc": float("nan"),
                             "pr_auc_pos": float("nan"),
                             "pr_auc_neg": float("nan"),
                             "fnr_at_fpr5": float("nan")})
        else:
            per_fold.append({"fold": k,
                             "auc": float(roc_auc_score(y, p)),
                             "pr_auc_pos": float(
                                 average_precision_score(y, p)),
                             "pr_auc_neg": float(
                                 average_precision_score(1 - y, 1.0 - p)),
                             "fnr_at_fpr5": float(fnr_at_fpr5(y, p))})
    aucs = [m["auc"] for m in per_fold if not np.isnan(m["auc"])]
    pr_aucs_pos = [m["pr_auc_pos"] for m in per_fold
                   if not np.isnan(m["pr_auc_pos"])]
    pr_aucs_neg = [m["pr_auc_neg"] for m in per_fold
                   if not np.isnan(m["pr_auc_neg"])]
    fnrs = [m["fnr_at_fpr5"] for m in per_fold if not np.isnan(m["fnr_at_fpr5"])]
    agg = {
        "test_auc_mean": float(np.mean(aucs)) if aucs else float("nan"),
        "test_auc_std": float(np.std(aucs)) if aucs else float("nan"),
        "test_pr_auc_pos_mean": float(np.mean(pr_aucs_pos))
        if pr_aucs_pos else float("nan"),
        "test_pr_auc_pos_std": float(np.std(pr_aucs_pos))
        if pr_aucs_pos else float("nan"),
        "test_pr_auc_neg_mean": float(np.mean(pr_aucs_neg))
        if pr_aucs_neg else float("nan"),
        "test_pr_auc_neg_std": float(np.std(pr_aucs_neg))
        if pr_aucs_neg else float("nan"),
        "test_fnr_at_fpr5_mean": float(np.mean(fnrs)) if fnrs else float("nan"),
        "test_fnr_at_fpr5_std": float(np.std(fnrs)) if fnrs else float("nan"),
    }
    return ens, per_fold, agg


if __name__ == "__main__":
    main()
