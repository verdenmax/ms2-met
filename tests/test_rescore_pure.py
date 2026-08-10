"""Pure-function tests for tools/spec_trainer/rescore.py.

These cover the lightgbm-free logic (infer_data_source mapping, compute_metrics
formulas) that the existing test_rescore_tool.py only exercises indirectly.
"""
import importlib.util
import os

import numpy as np
import pytest

_RESCORE = os.path.join(
    os.path.dirname(os.path.dirname(os.path.abspath(__file__))),
    "tools", "spec_trainer", "rescore.py")
_spec = importlib.util.spec_from_file_location("rescore_mod", _RESCORE)
rescore = importlib.util.module_from_spec(_spec)
_spec.loader.exec_module(rescore)

TMPL = "runs/baseline_{ds}_{fdr}"


# --- infer_data_source ---

def test_infer_in_sample():
    path, mode = rescore.infer_data_source("in_2da_clean", TMPL)
    assert mode == "in_sample"
    assert str(path) == "runs/baseline_2da_clean/features.csv"


def test_infer_cross_test_maps_to_held_dataset():
    # cross_test_<held>_<fdr> -> scored on the HELD-out dataset file
    path, mode = rescore.infer_data_source("cross_test_normal_neg15", TMPL)
    assert mode == "cross_test"
    assert str(path) == "runs/baseline_normal_neg15/features.csv"


@pytest.mark.parametrize("name", [
    "in_2da",            # too few parts
    "in_2da_clean_x",    # too many parts
    "cross_test_2da",    # too few parts
    "cross_test_a_b_c",  # too many parts
    "combined_neg10",    # unsupported family
    "garbage",
])
def test_infer_rejects_malformed_or_unsupported(name):
    with pytest.raises(ValueError):
        rescore.infer_data_source(name, TMPL)


# --- compute_metrics ---

def test_compute_metrics_confusion_and_rates():
    # Stored labels: 1=correct, 0=error. At error threshold 0.5 the canonical
    # confusion matrix has one TP/FP/FN/TN each.
    y = np.array([1, 1, 0, 0])
    p = np.array([0.9, 0.4, 0.6, 0.1])
    m = rescore.compute_metrics(y, p, 0.5)
    assert (m["tp"], m["fn"], m["fp"], m["tn"]) == (1, 1, 1, 1)
    assert m["n_actual_error"] == 2 and m["n_actual_correct"] == 2
    assert m["error_recall"] == 0.5 and m["correct_recall"] == 0.5
    assert m["error_precision"] == 0.5
    assert m["fnr"] == 0.5 and m["fpr"] == 0.5
    assert m["metric_semantics"] == "error_identification_positive_v1"
    assert m["positive_class"] == "incorrect_identification"


def test_compute_metrics_error_threshold_is_inclusive():
    # trust=.5 -> error_score=.5; equality is flagged as an error.
    y = np.array([1, 0])
    p = np.array([0.5, 0.5])
    m = rescore.compute_metrics(y, p, 0.5)
    assert m["tp"] == 1 and m["fp"] == 1


def test_compute_metrics_auc_nan_when_single_class():
    m = rescore.compute_metrics(np.array([1, 1]), np.array([0.9, 0.8]), 0.5)
    assert m["n_actual_error"] == 0
    assert np.isnan(m["roc_auc"])
