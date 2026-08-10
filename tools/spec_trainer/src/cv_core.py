"""Pure helpers for grouped CV and error-identification evaluation.

Storage/training compatibility
------------------------------
Feature CSVs and trained models keep the historical convention
``label=1`` / high model score = a correct, supported identification.

Evaluation convention
---------------------
Every public metric in this module follows the thesis convention:

* actual positive = an incorrect identification;
* actual negative = a correct identification;
* predicted positive = flagged as incorrect/suspicious;
* false positive = a correct identification flagged as incorrect;
* false negative = an incorrect identification accepted as correct.

The conversion is explicit: ``error_truth = 1 - stored_label`` and
``error_score = 1 - trust_score``.
"""
from __future__ import annotations

import numpy as np
from sklearn.model_selection import StratifiedGroupKFold, StratifiedKFold


METRIC_SEMANTICS_VERSION = "error_identification_positive_v1"


def as_error_detection(stored_labels, trust_scores):
    """Return ``(error_truth, error_score)`` under the canonical convention."""
    labels = np.asarray(stored_labels)
    scores = np.asarray(trust_scores, dtype="f8")
    if labels.shape != scores.shape:
        raise ValueError(
            f"stored_labels/trust_scores shape mismatch: "
            f"{labels.shape} vs {scores.shape}")
    if not set(np.unique(labels).tolist()).issubset({0, 1}):
        raise ValueError("stored_labels must contain only 0 (error) and 1 (correct)")
    if not np.isfinite(scores).all():
        raise ValueError("trust_scores contains NaN or infinite values")
    return 1 - labels.astype(int), 1.0 - scores


def make_cv_splits(y, groups, n_folds=5, seed=42):
    """Return positional grouped/stratified CV splits.

    Stratification may use the stored 0/1 labels because complementing a
    binary label does not change the split composition.
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


def threshold_at_fpr(stored_labels, trust_scores, target_fpr=0.10):
    """Select an error-score threshold with empirical FPR <= target.

    FPR is the fraction of actually correct identifications that are flagged
    as errors. The returned threshold is on ``error_score = 1-trust_score``;
    the inclusive decision rule is ``error_score >= threshold => error``.
    Boundary ties are rejected conservatively as a whole.
    """
    if not 0.0 <= target_fpr < 1.0:
        raise ValueError(
            f"target_fpr must be in [0, 1), got {target_fpr!r}")
    error_truth, error_score = as_error_detection(stored_labels, trust_scores)
    correct_scores = np.sort(error_score[error_truth == 0])[::-1]
    if len(correct_scores) == 0:
        raise ValueError(
            "cannot select an FPR threshold without correct identifications")

    allowed_fp = int(np.floor(target_fpr * len(correct_scores)))
    first_disallowed = correct_scores[allowed_fp]
    threshold = float(np.nextafter(first_disallowed, np.inf))
    observed_fpr = float(
        (correct_scores >= threshold).sum() / len(correct_scores))
    assert observed_fpr <= target_fpr + np.finfo(float).eps
    return threshold


def evaluate_at_threshold(stored_labels, trust_scores, error_threshold):
    """Compute a confusion matrix with incorrect identifications positive."""
    error_truth, error_score = as_error_detection(stored_labels, trust_scores)
    predicted_error = error_score >= error_threshold
    actual_error = error_truth == 1
    actual_correct = error_truth == 0

    tp = int((predicted_error & actual_error).sum())
    fp = int((predicted_error & actual_correct).sum())
    fn = int(((~predicted_error) & actual_error).sum())
    tn = int(((~predicted_error) & actual_correct).sum())
    n_error = int(actual_error.sum())
    n_correct = int(actual_correct.sum())
    return {
        "metric_semantics": METRIC_SEMANTICS_VERSION,
        "positive_class": "incorrect_identification",
        "error_threshold": float(error_threshold),
        "trust_threshold": float(1.0 - error_threshold),
        "n_actual_error": n_error,
        "n_actual_correct": n_correct,
        "tp": tp,
        "fp": fp,
        "fn": fn,
        "tn": tn,
        "fpr": float(fp / n_correct) if n_correct else None,
        "fnr": float(fn / n_error) if n_error else None,
        "error_recall": float(tp / n_error) if n_error else None,
        "correct_recall": float(tn / n_correct) if n_correct else None,
        "error_precision": float(tp / (tp + fp)) if tp + fp else None,
    }


def working_points(stored_labels, trust_scores,
                   fpr_targets=(0.01, 0.05, 0.10)):
    """Metrics at fixed false-alarm rates on correct identifications."""
    out = {}
    for target in fpr_targets:
        threshold = threshold_at_fpr(stored_labels, trust_scores, target)
        metrics = evaluate_at_threshold(
            stored_labels, trust_scores, threshold)
        metrics["target_fpr"] = float(target)
        out[f"fpr_{int(round(target * 100))}"] = metrics
    return out


def fnr_at_fpr5(stored_labels, trust_scores):
    """Missed-error rate when <=5% of correct IDs are falsely flagged."""
    return working_points(
        stored_labels, trust_scores, fpr_targets=(0.05,))["fpr_5"]["fnr"]


def evaluate_oof(stored_labels, trust_scores):
    """Pooled OOF ranking and fixed-FPR metrics under canonical semantics."""
    from sklearn.metrics import average_precision_score, roc_auc_score

    error_truth, error_score = as_error_detection(stored_labels, trust_scores)
    points = working_points(stored_labels, trust_scores)
    return {
        "metric_semantics": METRIC_SEMANTICS_VERSION,
        "positive_class": "incorrect_identification",
        "roc_auc": float(roc_auc_score(error_truth, error_score)),
        "error_pr_auc": float(
            average_precision_score(error_truth, error_score)),
        "fnr_at_fpr5": float(points["fpr_5"]["fnr"]),
        "error_recall_at_fpr10": float(
            points["fpr_10"]["error_recall"]),
        "working_points": points,
    }


def audit_labels(df, trust_scores, label_col="label", threshold=0.9,
                 top_n=200,
                 id_cols=("sequence", "charge", "label_type"),
                 diag_cols=("all_p75", "precursor_pearson",
                            "all_cosine_mean",
                            "all_heavy_shape_irregularity_max")):
    """Incorrect references ranked by erroneous model support.

    These are false-negative candidates under the canonical convention: known
    errors that receive a high trust score and would be accepted as correct.
    """
    work = df.copy()
    work["trust_score"] = np.asarray(trust_scores)
    errors = work[work[label_col] == 0]
    suspects = (errors[errors["trust_score"] >= threshold]
                .sort_values("trust_score", ascending=False)
                .head(top_n))
    keep = ([c for c in id_cols if c in suspects.columns]
            + ["trust_score"]
            + [c for c in diag_cols if c in suspects.columns])
    return suspects[keep].reset_index(drop=True)


def average_proba(proba_list):
    """Mean of per-fold trust-score arrays for an external sample set."""
    arr = np.vstack([np.asarray(p, dtype="f8") for p in proba_list])
    return arr.mean(axis=0)
