"""Tests for tools/spec_trainer/rescore.py.

See docs/specs/2026-06-03-rescore-tool-design.md.

Two test classes:
- Pure-logic tests (no LightGBM, no data files) — always run
- Integration tests (need runs/spec_trainer/models/ + runs/baseline_*/features.csv)
  — auto-skip if artifacts missing.
"""
import json
import os
import subprocess
import sys

import numpy as np
import pytest

_PROJECT_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
_TOOL = os.path.join(_PROJECT_ROOT, "tools", "spec_trainer", "rescore.py")

if _PROJECT_ROOT not in sys.path:
    sys.path.insert(0, _PROJECT_ROOT)

sys.path.insert(0, os.path.join(_PROJECT_ROOT, "tools", "spec_trainer"))
import rescore  # noqa: E402


def _python_exec():
    """Return a python interpreter that has the project's runtime deps.

    Prefer the conda env python (CONDA_PREFIX) so subprocess invocations can
    import lightgbm/pandas/sklearn even when pytest itself runs from a
    different interpreter (e.g. system /usr/bin/pytest)."""
    conda_prefix = os.environ.get("CONDA_PREFIX")
    if conda_prefix:
        candidate = os.path.join(conda_prefix, "bin", "python")
        if os.path.exists(candidate):
            return candidate
    return sys.executable


def test_rescore_threshold_monotonicity():
    """As threshold increases, neg_recall must monotonically increase
    (or stay the same), and pos_recall must monotonically decrease (or
    stay the same). Pure-logic test, no LightGBM."""
    rng = np.random.default_rng(42)
    n = 200
    y_true = np.array([1] * 100 + [0] * 100)
    proba = np.concatenate([
        rng.beta(8, 2, size=100),
        rng.beta(2, 8, size=100),
    ])

    thresholds = [0.1, 0.3, 0.5, 0.7, 0.9]
    prev_neg_recall = -np.inf
    prev_pos_recall = np.inf
    for t in thresholds:
        m = rescore.compute_metrics(y_true, proba, t)
        assert m["neg_recall"] >= prev_neg_recall, (
            "neg_recall not monotonic increasing: "
            "prev={}, current={} at t={}".format(prev_neg_recall, m['neg_recall'], t))
        assert m["pos_recall"] <= prev_pos_recall
        prev_neg_recall = m["neg_recall"]
        prev_pos_recall = m["pos_recall"]


def test_rescore_invalid_threshold_rejected():
    """--thresholds 1.5 rejected (must be in (0,1))."""
    result = subprocess.run(
        [_python_exec(), _TOOL, "--thresholds", "1.5"],
        capture_output=True, text=True)
    assert result.returncode != 0
    combined = result.stdout + result.stderr
    assert "0" in combined and "1" in combined


def test_rescore_threshold_zero_rejected():
    """--thresholds 0 rejected (must be > 0)."""
    result = subprocess.run(
        [_python_exec(), _TOOL, "--thresholds", "0"],
        capture_output=True, text=True)
    assert result.returncode != 0


def test_rescore_threshold_one_rejected():
    """--thresholds 1.0 rejected (must be < 1)."""
    result = subprocess.run(
        [_python_exec(), _TOOL, "--thresholds", "1.0"],
        capture_output=True, text=True)
    assert result.returncode != 0


def _have_artifact(rel_path: str) -> bool:
    return os.path.exists(os.path.join(_PROJECT_ROOT, rel_path))


@pytest.mark.skipif(
    not _have_artifact("runs/spec_trainer/models/in_2da_clean.txt"),
    reason="model not trained yet")
@pytest.mark.skipif(
    not _have_artifact("runs/baseline_2da_clean/features.csv"),
    reason="features.csv not generated yet")
def test_rescore_models_filter(tmp_path):
    """--models in_2da_clean produces 1 model × N thresholds rows."""
    output = tmp_path / "out.csv"
    result = subprocess.run(
        [_python_exec(), _TOOL,
         "--thresholds", "0.5", "0.9",
         "--models", "in_2da_clean",
         "--output", str(output)],
        capture_output=True, text=True, cwd=_PROJECT_ROOT)
    assert result.returncode == 0, (
        "rescore failed:\nSTDOUT:\n" + result.stdout +
        "\nSTDERR:\n" + result.stderr)
    import csv as _csv
    with open(output) as f:
        rows = list(_csv.DictReader(f))
    assert len(rows) == 2
    exps = {r["experiment"] for r in rows}
    assert exps == {"in_2da_clean"}


@pytest.mark.skipif(
    not _have_artifact("runs/spec_trainer/models/in_2da_clean.txt"),
    reason="model not trained yet")
@pytest.mark.skipif(
    not _have_artifact("runs/spec_trainer/results/in_2da_clean.json"),
    reason="result JSON not present")
@pytest.mark.skipif(
    not _have_artifact("runs/baseline_2da_clean/features.csv"),
    reason="features.csv not generated yet")
def test_rescore_in_sample_split_matches_training(tmp_path):
    """rescore.py at threshold 0.5 reproduces in_2da_clean.json confusion."""
    output = tmp_path / "out.csv"
    result = subprocess.run(
        [_python_exec(), _TOOL,
         "--thresholds", "0.5",
         "--models", "in_2da_clean",
         "--output", str(output)],
        capture_output=True, text=True, cwd=_PROJECT_ROOT)
    assert result.returncode == 0, "rescore failed: " + result.stderr

    import csv as _csv
    with open(output) as f:
        rows = list(_csv.DictReader(f))
    assert len(rows) == 1
    r = rows[0]

    with open(os.path.join(
            _PROJECT_ROOT, "runs/spec_trainer/results/in_2da_clean.json")) as f:
        ref = json.load(f)
    ref_cm = ref["confusion_matrix"]
    ref_tn, ref_fp = ref_cm[0]
    ref_fn, ref_tp = ref_cm[1]

    assert int(r["tn"]) == ref_tn
    assert int(r["fp"]) == ref_fp
    assert int(r["fn"]) == ref_fn
    assert int(r["tp"]) == ref_tp
    assert abs(float(r["auc"]) - float(ref["auc"])) < 1e-6


@pytest.mark.skipif(
    not _have_artifact("runs/spec_trainer/models/cross_test_2da_clean.txt"),
    reason="cross_test model not trained yet")
@pytest.mark.skipif(
    not _have_artifact("runs/baseline_2da_clean/features.csv"),
    reason="features.csv not generated yet")
def test_rescore_cross_test_uses_full_held_file(tmp_path):
    """cross_test_<X>_<fdr> uses ENTIRE features.csv as test set."""
    output = tmp_path / "out.csv"
    result = subprocess.run(
        [_python_exec(), _TOOL,
         "--thresholds", "0.5",
         "--models", "cross_test_2da_clean",
         "--output", str(output)],
        capture_output=True, text=True, cwd=_PROJECT_ROOT)
    assert result.returncode == 0, "rescore failed: " + result.stderr

    import csv as _csv
    with open(output) as f:
        rows = list(_csv.DictReader(f))
    r = rows[0]
    test_total = int(r["n_pos"]) + int(r["n_neg"])

    full_csv_rows = sum(
        1 for _ in open(os.path.join(
            _PROJECT_ROOT, "runs/baseline_2da_clean/features.csv"))) - 1

    assert test_total == full_csv_rows
