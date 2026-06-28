"""Cross-validated training entry for spec_trainer (production LightGBM).

One CV pass yields OOF predictions that drive: honest evaluation, fold
ensemble (saved per-fold models), and label-noise audit. Does NOT touch
main.py (single-holdout flow unchanged). lightgbm is imported lazily inside
assemble_oof / predict_ensemble so this module (and path helpers) import without it.
"""
import argparse
import json
import logging
import os
import re

import numpy as np
import pandas as pd
import yaml

from cv_core import (average_proba, audit_labels, evaluate_oof, fnr_at_fpr5,
                     make_cv_splits)
from feature_cols import resolve_feature_cols


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
    so the val set never overlaps the OOF test fold. Grouped split when groups
    is given (no group spans tr2/val), else stratified.
    """
    from sklearn.model_selection import GroupShuffleSplit, StratifiedShuffleSplit
    if groups is not None:
        inner = GroupShuffleSplit(n_splits=1, test_size=valid_size,
                                  random_state=seed)
        loc_tr, loc_val = next(inner.split(
            X.iloc[tr_idx], y.iloc[tr_idx], groups.iloc[tr_idx]))
    else:
        inner = StratifiedShuffleSplit(n_splits=1, test_size=valid_size,
                                       random_state=seed)
        loc_tr, loc_val = next(inner.split(X.iloc[tr_idx], y.iloc[tr_idx]))
    return tr_idx[loc_tr], tr_idx[loc_val]


def assemble_oof(df, X, y, groups, cfg, feature_cols, model_prefix):
    """Train one model per fold, collect leak-free OOF preds, save fold models.

    Returns (oof_proba, fold_metrics, model_paths). lightgbm imported here.
    """
    # df: unused here; kept for call-site symmetry (caller uses it for label audit)
    from sklearn.metrics import roc_auc_score
    from models.model_manager import ModelManager

    n_folds = int(cfg["training"].get("cv_folds", 5))
    seed = int(cfg["training"].get("cv_seed", 42))
    valid_size = float(cfg["training"].get("valid_size", 0.15))
    grp_vals = None if groups is None else groups.values

    splits = make_cv_splits(y.values, grp_vals, n_folds=n_folds, seed=seed)
    oof = np.full(len(y), np.nan)
    fold_metrics, model_paths = [], []

    for k, (tr_idx, te_idx) in enumerate(splits):
        # 折内早停验证集（分组，避免污染 OOF 折）
        tr2, val = _inner_split(X, y, groups, tr_idx, valid_size, seed)

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
                                 "fnr_at_fpr5": float("nan")})
        else:
            fold_metrics.append({"fold": k,
                                 "auc": float(roc_auc_score(te_y, te_p)),
                                 "fnr_at_fpr5": float(fnr_at_fpr5(te_y, te_p))})

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
    df = read_dataframe(train_files)
    feature_cols = resolve_feature_cols(
        cfg["data"].get("feature_cols"), train_files, target_col)
    X = df[feature_cols]
    y = df[target_col]

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

    summary = evaluate_oof(y, oof)
    aucs = [m["auc"] for m in fold_metrics if not np.isnan(m["auc"])]
    summary.update({
        "cv_folds": len(fold_metrics),
        "fold_metrics": fold_metrics,
        "auc_mean": float(np.mean(aucs)) if aucs else float("nan"),
        "auc_std": float(np.std(aucs)) if aucs else float("nan"),
        "model_paths": model_paths,
        "n_pos": int((y == 1).sum()),
        "n_neg": int((y == 0).sum()),
    })
    summary["name"] = args.name
    os.makedirs(os.path.dirname(result_path) or ".", exist_ok=True)
    with open(result_path, "w") as f:
        json.dump(summary, f, indent=2)

    audit_cfg = cfg.get("audit", {})
    susp = audit_labels(
        df, oof, label_col=target_col,
        threshold=float(audit_cfg.get("suspect_threshold", 0.9)),
        top_n=int(audit_cfg.get("suspect_top_n", 200)))
    susp.to_csv(suspects_path, index=False)

    logging.info("CV done: AUC=%.4f FNR@FPR5=%.4f; %d suspects -> %s",
                 summary["auc"], summary["fnr_at_fpr5"], len(susp), suspects_path)
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


if __name__ == "__main__":
    main()
