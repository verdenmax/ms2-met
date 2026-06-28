"""Cross-validated training entry for spec_trainer (production LightGBM).

One CV pass yields OOF predictions that drive: honest evaluation, fold
ensemble (saved per-fold models), and label-noise audit. Does NOT touch
main.py (single-holdout flow unchanged). lightgbm is imported lazily inside
assemble_oof so this module (and path helpers) import without it.
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
