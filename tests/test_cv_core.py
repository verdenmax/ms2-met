import os
import sys

import numpy as np
import pandas as pd
import pytest

sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..",
                                "tools", "spec_trainer", "src"))


def test_as_error_detection_inverts_storage_convention():
    from cv_core import as_error_detection
    error_truth, error_score = as_error_detection(
        [1, 0], [0.9, 0.2])
    assert error_truth.tolist() == [0, 1]
    assert np.allclose(error_score, [0.1, 0.8])


def test_make_cv_splits_grouped_no_leak_full_cover():
    from cv_core import make_cv_splits
    groups = np.repeat(np.arange(10), 2)
    y = np.where(groups < 6, 1, 0)
    splits = make_cv_splits(y, groups, n_folds=5, seed=42)
    assert len(splits) == 5
    covered = np.concatenate([te for _, te in splits])
    assert sorted(covered.tolist()) == list(range(len(y)))
    for tr, te in splits:
        assert set(groups[tr]).isdisjoint(set(groups[te]))


def test_make_cv_splits_no_groups_fallback():
    from cv_core import make_cv_splits
    y = np.array([1] * 16 + [0] * 4)
    splits = make_cv_splits(y, groups=None, n_folds=5, seed=42)
    assert len(splits) == 5
    covered = np.concatenate([te for _, te in splits])
    assert sorted(covered.tolist()) == list(range(len(y)))


def test_working_points_perfect_error_detection():
    from cv_core import fnr_at_fpr5, working_points
    # Stored label=1/correct has high trust; label=0/error has low trust.
    y = np.r_[np.ones(100), np.zeros(100)]
    trust = np.r_[np.full(100, 0.9), np.full(100, 0.1)]
    points = working_points(y, trust)
    assert set(points) == {"fpr_1", "fpr_5", "fpr_10"}
    assert points["fpr_5"]["fpr"] <= 0.05
    assert points["fpr_5"]["fnr"] == 0.0
    assert points["fpr_5"]["error_recall"] == 1.0
    assert fnr_at_fpr5(y, trust) == 0.0


def test_threshold_at_fpr_is_error_positive_and_tie_safe():
    from cv_core import evaluate_at_threshold, threshold_at_fpr
    # 20 actual correct IDs. Their error scores have one 0.9 and a tied 0.8
    # boundary; at FPR<=10%, the whole tie must be excluded -> FP=1.
    correct_error_scores = np.array(
        [0.9, 0.8, 0.8] + list(np.linspace(0.7, 0.0, 17)))
    actual_error_scores = np.array([0.95, 0.85, 0.5])
    stored_labels = np.r_[
        np.ones(len(correct_error_scores)),
        np.zeros(len(actual_error_scores)),
    ]
    trust_scores = 1.0 - np.r_[correct_error_scores, actual_error_scores]

    threshold = threshold_at_fpr(
        stored_labels, trust_scores, target_fpr=0.10)
    metrics = evaluate_at_threshold(
        stored_labels, trust_scores, threshold)
    assert metrics["fpr"] <= 0.10
    assert metrics["fp"] == 1
    assert metrics["tp"] == 2
    assert metrics["fn"] == 1
    assert metrics["tn"] == 19
    assert metrics["correct_recall"] == 0.95
    assert metrics["error_recall"] == 2 / 3
    assert metrics["fnr"] == 1 / 3


def test_threshold_at_fpr_requires_correct_ids_and_valid_target():
    from cv_core import threshold_at_fpr
    with pytest.raises(ValueError, match="without correct identifications"):
        threshold_at_fpr([0, 0], [0.2, 0.8], target_fpr=0.10)
    with pytest.raises(ValueError, match="target_fpr"):
        threshold_at_fpr([0, 1], [0.2, 0.8], target_fpr=1.0)


def test_evaluate_oof_perfect_and_reversed():
    from cv_core import evaluate_oof
    y = np.r_[np.ones(50), np.zeros(50)]
    perfect_trust = np.r_[np.full(50, 0.9), np.full(50, 0.1)]
    metrics = evaluate_oof(y, perfect_trust)
    assert metrics["positive_class"] == "incorrect_identification"
    assert metrics["roc_auc"] == 1.0
    assert metrics["error_pr_auc"] == 1.0
    assert metrics["fnr_at_fpr5"] == 0.0
    assert metrics["error_recall_at_fpr10"] == 1.0

    reversed_metrics = evaluate_oof(y, 1.0 - perfect_trust)
    assert reversed_metrics["roc_auc"] == 0.0
    assert 0.0 < reversed_metrics["error_pr_auc"] < 1.0


def test_audit_labels_errors_only_sorted_filtered():
    from cv_core import audit_labels
    df = pd.DataFrame({
        "label": [0, 0, 0, 1, 1],
        "sequence": list("ABCDE"),
        "charge": [2, 2, 3, 2, 2],
        "all_p75": [0.9, 0.8, 0.1, 0.9, 0.9],
    })
    trust = [0.97, 0.92, 0.40, 0.99, 0.10]
    suspects = audit_labels(df, trust, threshold=0.9, top_n=10)
    assert list(suspects["sequence"]) == ["A", "B"]
    assert "trust_score" in suspects.columns
    assert list(suspects["trust_score"]) == [0.97, 0.92]
    assert len(audit_labels(df, trust, threshold=0.9, top_n=1)) == 1


def test_average_proba():
    from cv_core import average_proba
    out = average_proba([np.array([0.0, 1.0]),
                         np.array([1.0, 0.0]),
                         np.array([0.5, 0.5])])
    assert np.allclose(out, [0.5, 0.5])
