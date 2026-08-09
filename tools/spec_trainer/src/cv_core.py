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


def threshold_at_fpr(y_true, y_score, target_fpr=0.10):
    """Select the most permissive negative-calibrated threshold with
    empirical FPR <= ``target_fpr``.

    Scores greater than or equal to the returned threshold are classified as
    positive.  Unlike a plain quantile, this implementation handles ties at
    the boundary conservatively: all samples tied with the first disallowed
    negative are rejected.  Consequently the calibration-set FPR is
    guaranteed not to exceed the requested target (it can be lower).
    """
    if not 0.0 <= target_fpr < 1.0:
        raise ValueError(
            f"target_fpr must be in [0, 1), got {target_fpr!r}")

    y_true = np.asarray(y_true)
    y_score = np.asarray(y_score, dtype="f8")
    if y_true.shape != y_score.shape:
        raise ValueError(
            f"y_true/y_score shape mismatch: {y_true.shape} vs {y_score.shape}")
    if not np.isfinite(y_score).all():
        raise ValueError("y_score contains NaN or infinite values")

    neg = np.sort(y_score[y_true == 0])[::-1]
    if len(neg) == 0:
        raise ValueError("cannot select an FPR threshold without negatives")

    # At most floor(target_fpr * n_neg) negatives may pass.  The item at
    # ``allowed_fp`` is the first disallowed score; stepping just above it
    # also excludes all ties at that boundary.
    allowed_fp = int(np.floor(target_fpr * len(neg)))
    first_disallowed = neg[allowed_fp]
    threshold = float(np.nextafter(first_disallowed, np.inf))

    observed_fpr = float((neg >= threshold).sum() / len(neg))
    assert observed_fpr <= target_fpr + np.finfo(float).eps
    return threshold


def evaluate_at_threshold(y_true, y_score, threshold):
    """Classification metrics for the inclusive ``score >= threshold`` rule."""
    y_true = np.asarray(y_true)
    y_score = np.asarray(y_score, dtype="f8")
    if y_true.shape != y_score.shape:
        raise ValueError(
            f"y_true/y_score shape mismatch: {y_true.shape} vs {y_score.shape}")

    pred = y_score >= threshold
    pos = y_true == 1
    neg = y_true == 0
    n_pos, n_neg = int(pos.sum()), int(neg.sum())
    tp = int((pred & pos).sum())
    fp = int((pred & neg).sum())
    fn = n_pos - tp
    tn = n_neg - fp
    return {
        "threshold": float(threshold),
        "n_pos": n_pos,
        "n_neg": n_neg,
        "tp": tp,
        "fp": fp,
        "fn": fn,
        "tn": tn,
        "fpr": float(fp / n_neg) if n_neg else None,
        "neg_recall": float(tn / n_neg) if n_neg else None,
        "pos_recall": float(tp / n_pos) if n_pos else None,
        "fnr": float(fn / n_pos) if n_pos else None,
        "pos_precision": float(tp / (tp + fp)) if tp + fp else None,
    }


def fnr_at_fpr5(y_true, y_score):
    """FNR at FPR<=5% = 1 - pos_recall at the neg-95% working point."""
    return 1.0 - working_points(y_true, y_score)["neg_recall_95"]["pos_recall"]


def evaluate_oof(y_true, oof_proba):
    """Summary metrics on out-of-fold predictions.

    The two PR-AUC values use average precision, the standard step-wise area
    summary for a precision-recall curve. ``pr_auc_neg`` treats the minority
    negative identifications as the event of interest and is therefore the
    primary imbalance-sensitive metric; ``pr_auc_pos`` is retained for the
    model's native positive-confidence direction.
    """
    from sklearn.metrics import average_precision_score, roc_auc_score
    y_true = np.asarray(y_true)
    oof_proba = np.asarray(oof_proba)
    return {
        "auc": float(roc_auc_score(y_true, oof_proba)),
        "pr_auc_pos": float(average_precision_score(y_true, oof_proba)),
        "pr_auc_neg": float(average_precision_score(
            1 - y_true, 1.0 - oof_proba)),
        "fnr_at_fpr5": float(fnr_at_fpr5(y_true, oof_proba)),
        "working_points": working_points(y_true, oof_proba),
    }


def audit_labels(df, oof_proba, label_col="label", threshold=0.9, top_n=200,
                 id_cols=("sequence", "charge", "label_type"),
                 diag_cols=("all_p75", "precursor_pearson", "all_cosine_mean",
                            "all_heavy_shape_irregularity_max")):
    """Negatives ranked by how 'positive-looking' their OOF prob is.

    Triage list for manual review (NOT auto-relabel): a negative whose
    out-of-fold prob >= threshold either is a genuine hard negative or a
    mislabel. Returns id+diagnostic cols (only those present), oof desc,
    capped at top_n.
    """
    work = df.copy()
    work["oof_proba"] = np.asarray(oof_proba)
    neg = work[work[label_col] == 0]
    susp = (neg[neg["oof_proba"] >= threshold]
            .sort_values("oof_proba", ascending=False)
            .head(top_n))
    keep = ([c for c in id_cols if c in susp.columns]
            + ["oof_proba"]
            + [c for c in diag_cols if c in susp.columns])
    return susp[keep].reset_index(drop=True)


def average_proba(proba_list):
    """Mean of per-fold predict_proba arrays = ensemble score for new data."""
    arr = np.vstack([np.asarray(p, dtype="f8") for p in proba_list])
    return arr.mean(axis=0)
