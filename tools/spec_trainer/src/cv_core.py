"""Pure, lightgbm-free helpers for cross-validated training.

Split / metric / audit / ensemble-averaging logic lives here so it can be
unit-tested without importing lightgbm (mirrors the holdout.py / feature_cols.py
extraction pattern in this package).
"""
import numpy as np
from sklearn.model_selection import StratifiedGroupKFold, StratifiedKFold


def make_cv_splits(y, groups, n_folds=5, seed=42):
    """Return a list of (train_idx, test_idx) positional index arrays.

    With groups: StratifiedGroupKFold — no group spans a fold's train+test
    (prevents same-peptide leakage). Without groups (None): StratifiedKFold.
    """
    y = np.asarray(y)
    dummy = np.zeros(len(y))
    if groups is not None:
        groups = np.asarray(groups)
        splitter = StratifiedGroupKFold(
            n_splits=n_folds, shuffle=True, random_state=seed)
        return list(splitter.split(dummy, y, groups))
    splitter = StratifiedKFold(
        n_splits=n_folds, shuffle=True, random_state=seed)
    return list(splitter.split(dummy, y))


def working_points(y_true, y_score, fpr_targets=(0.05, 0.10, 0.20)):
    """Negative-quantile working points — same convention as
    tools/eval_baseline.py:compute_working_points (FPR via neg quantile).
    Returns {"neg_recall_95/90/80": {threshold, pos_recall, neg_recall}}.
    """
    y_true = np.asarray(y_true)
    y_score = np.asarray(y_score)
    pos = y_score[y_true == 1]
    neg = y_score[y_true == 0]
    out = {}
    for fpr in fpr_targets:
        thr = float(np.quantile(neg, 1 - fpr))
        pos_kept = int((pos >= thr).sum())
        neg_kept = int((neg >= thr).sum())
        out[f"neg_recall_{int((1 - fpr) * 100)}"] = {
            "threshold": thr,
            "pos_recall": float(pos_kept / max(len(pos), 1)),
            "neg_recall": float(1 - neg_kept / max(len(neg), 1)),
        }
    return out


def fnr_at_fpr5(y_true, y_score):
    """FNR at FPR<=5% = 1 - pos_recall at the neg-95% working point."""
    return 1.0 - working_points(y_true, y_score)["neg_recall_95"]["pos_recall"]


def evaluate_oof(y_true, oof_proba):
    """Summary metrics on out-of-fold predictions: auc + FNR@FPR5 + working points."""
    from sklearn.metrics import roc_auc_score
    y_true = np.asarray(y_true)
    oof_proba = np.asarray(oof_proba)
    return {
        "auc": float(roc_auc_score(y_true, oof_proba)),
        "fnr_at_fpr5": float(fnr_at_fpr5(y_true, oof_proba)),
        "working_points": working_points(y_true, oof_proba),
    }
