import os
import sys
import importlib.util

import numpy as np
import pandas as pd
import pytest

sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", "tools"))
sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..",
                                "tools", "spec_trainer", "src"))
import fdr_consistency as fc

_HAS_LGB = importlib.util.find_spec("lightgbm") is not None
requires_lgb = pytest.mark.skipif(not _HAS_LGB, reason="lightgbm not installed")


def test_bins_constant():
    assert fc.FDR_BINS == [(0, 0.01), (0.01, 0.05), (0.05, 0.10),
                           (0.10, 0.20), (0.20, 0.50)]


def test_plan_example():
    df = pd.DataFrame({"q_value": [.005, .03, .03, .3], "label": [1, 1, 1, 1]})
    folds = [np.array([.9, .9, .1, .1]), np.array([.9, .1, .1, .1])]
    out = fc.bin_recall(df, folds, thr=.5)
    assert out[(0, 0.01)]["n"] == 1
    assert out[(0, 0.01)]["ens_recall"] == pytest.approx(1.0)
    assert out[(0, 0.01)]["fold_mean"] == pytest.approx(1.0)
    assert out[(0.01, 0.05)]["n"] == 2
    assert out[(0.01, 0.05)]["ens_recall"] == pytest.approx(0.5)
    assert out[(0.01, 0.05)]["fold_mean"] == pytest.approx(0.25)
    assert out[(0.01, 0.05)]["fold_std"] == pytest.approx(0.25)
    assert out[(0.20, 0.50)]["n"] == 1
    assert out[(0.20, 0.50)]["ens_recall"] == pytest.approx(0.0)


def test_empty_bin_no_crash():
    df = pd.DataFrame({"q_value": [.005]})
    out = fc.bin_recall(df, [np.array([.9])], thr=.5)
    assert out[(0.05, 0.10)]["n"] == 0
    assert np.isnan(out[(0.05, 0.10)]["ens_recall"])
    assert np.isnan(out[(0.05, 0.10)]["fold_mean"])
    assert np.isnan(out[(0.05, 0.10)]["fold_std"])


def test_boundary_001_goes_bin0():
    df = pd.DataFrame({"q_value": [0.01]})
    out = fc.bin_recall(df, [np.array([.9])], thr=.5)
    assert out[(0, 0.01)]["n"] == 1          # right-inclusive
    assert out[(0.01, 0.05)]["n"] == 0       # left-exclusive


def test_q_zero_in_bin0():
    df = pd.DataFrame({"q_value": [0.0]})
    out = fc.bin_recall(df, [np.array([.9])], thr=.5)
    assert out[(0, 0.01)]["n"] == 1


def test_q_above_max_excluded():
    df = pd.DataFrame({"q_value": [0.7]})
    out = fc.bin_recall(df, [np.array([.9])], thr=.5)
    assert sum(b["n"] for b in out.values()) == 0


def test_single_fold_std_zero():
    df = pd.DataFrame({"q_value": [.005, .005]})
    out = fc.bin_recall(df, [np.array([.9, .1])], thr=.5)
    assert out[(0, 0.01)]["fold_std"] == pytest.approx(0.0)
    assert out[(0, 0.01)]["fold_mean"] == pytest.approx(0.5)


def test_ens_vs_per_fold_differ():
    df = pd.DataFrame({"q_value": [.005]})
    # ens=0.5 -> recall 1; per-fold .4/.6 -> mean .5
    out = fc.bin_recall(df, [np.array([.4]), np.array([.6])], thr=.5)
    assert out[(0, 0.01)]["ens_recall"] == pytest.approx(1.0)
    assert out[(0, 0.01)]["fold_mean"] == pytest.approx(0.5)


def test_threshold_ties_inclusive():
    df = pd.DataFrame({"q_value": [.005]})
    out = fc.bin_recall(df, [np.array([.5])], thr=.5)
    assert out[(0, 0.01)]["ens_recall"] == pytest.approx(1.0)  # >= thr


def test_error_positive_working_point_controls_false_alarm_rate():
    import cv_core
    rng = np.random.default_rng(0)
    y = np.array([0] * 100 + [1] * 100)
    s = np.concatenate([rng.random(100), rng.random(100) + .5])
    point = cv_core.working_points(y, s)["fpr_5"]
    assert point["fpr"] <= 0.05
    assert point["error_threshold"] == cv_core.threshold_at_fpr(
        y, s, target_fpr=0.05)


def test_no_top_level_lightgbm_import():
    import pathlib
    src = pathlib.Path(fc.__file__).read_text()
    head = src.split("def main")[0]
    assert "import lightgbm" not in head


@requires_lgb
def test_main_smoke(tmp_path):
    repo = os.path.join(os.path.dirname(__file__), "..")
    models = [os.path.join(repo, "runs/spec_trainer/models",
                           f"cv_in_2da_clean.fold{k}.txt") for k in range(5)]
    if not all(os.path.exists(m) for m in models):
        pytest.skip("fold models absent")
    fc.main([])
