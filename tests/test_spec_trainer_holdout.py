"""Behavioral tests for spec_trainer holdout split resolution.

Regression for review finding I-ST2 (2026-06-03 audit) + rubber-duck N4:
test the actual branching logic with synthetic data, not just source-grep.
"""
import os
import sys
import numpy as np
import pandas as pd
import pytest

_PROJECT_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
_SRC = os.path.join(_PROJECT_ROOT, "tools", "spec_trainer", "src")
if _SRC not in sys.path:
    sys.path.insert(0, _SRC)

from holdout import resolve_holdout  # noqa: E402


def _synthetic_frame(n=100, seed=0):
    rng = np.random.default_rng(seed)
    X = pd.DataFrame({
        "f1": rng.normal(size=n),
        "f2": rng.normal(size=n),
    })
    y = pd.Series(rng.integers(0, 2, size=n), name="label")
    return X, y


def test_resolve_holdout_uses_distinct_test_files(tmp_path):
    """When test_files distinct from train_files, load them as held-out."""
    X_train, y_train = _synthetic_frame(50, seed=1)
    test_csv = tmp_path / "test.csv"
    test_df = _synthetic_frame(30, seed=2)
    pd.concat([test_df[0], test_df[1]], axis=1).to_csv(test_csv, index=False)

    def loader(files, feature_cols, target_col):
        df = pd.concat([pd.read_csv(f) for f in files], ignore_index=True)
        return df[feature_cols], df[target_col]

    Xt, Xe, yt, ye = resolve_holdout(
        X_train, y_train,
        train_files=["train_a.csv"],
        test_files=[str(test_csv)],
        test_size=0.0,
        feature_cols=["f1", "f2"],
        target_col="label",
        loader=loader,
    )
    assert len(Xe) == 30  # held-out came from test_csv
    assert len(Xt) == 50  # train untouched


def test_resolve_holdout_splits_from_train_when_test_files_empty():
    """When test_files empty + test_size>0, stratified-split from train."""
    X_train, y_train = _synthetic_frame(100, seed=3)
    Xt, Xe, yt, ye = resolve_holdout(
        X_train, y_train,
        train_files=["train.csv"],
        test_files=[],
        test_size=0.25,
        feature_cols=["f1", "f2"],
        target_col="label",
        loader=lambda *a, **k: (_ for _ in ()).throw(
            AssertionError("loader should NOT be called when splitting")),
    )
    # Disjointness: indices in Xt and Xe should not overlap
    assert set(Xt.index).isdisjoint(set(Xe.index)), (
        "train and held-out must be disjoint (I-ST2 + rubber-duck N4)")
    assert len(Xe) == 25  # test_size=0.25 of 100
    assert len(Xt) == 75


def test_resolve_holdout_splits_when_test_files_equals_train():
    """If user accidentally sets test_files==train_files, prefer test_size split."""
    X_train, y_train = _synthetic_frame(100, seed=4)
    train_files = ["a.csv", "b.csv"]
    Xt, Xe, yt, ye = resolve_holdout(
        X_train, y_train,
        train_files=train_files,
        test_files=list(train_files),  # same set
        test_size=0.2,
        feature_cols=["f1", "f2"],
        target_col="label",
        loader=lambda *a, **k: (_ for _ in ()).throw(
            AssertionError("loader should NOT be called when test_files==train_files")),
    )
    assert set(Xt.index).isdisjoint(set(Xe.index))
    assert len(Xe) == 20


def test_resolve_holdout_raises_when_neither_option_provided():
    """No test_files (distinct) and no test_size -> ValueError, no silent in-sample."""
    X_train, y_train = _synthetic_frame(50, seed=5)
    with pytest.raises(ValueError, match="in-sample"):
        resolve_holdout(
            X_train, y_train,
            train_files=["a.csv"],
            test_files=[],
            test_size=0.0,
            feature_cols=["f1", "f2"],
            target_col="label",
            loader=lambda *a, **k: None,
        )


def test_resolve_holdout_stratify_preserves_label_balance():
    """Stratified split must preserve class proportions in both halves."""
    n = 1000
    rng = np.random.default_rng(42)
    X = pd.DataFrame({"f1": rng.normal(size=n)})
    # Heavy class imbalance: 80% class 0, 20% class 1
    y = pd.Series((rng.uniform(size=n) < 0.2).astype(int), name="label")
    train_ratio = y.mean()

    Xt, Xe, yt, ye = resolve_holdout(
        X, y,
        train_files=["x.csv"],
        test_files=[],
        test_size=0.3,
        feature_cols=["f1"],
        target_col="label",
        loader=lambda *a, **k: None,
    )
    # Both halves should have ~20% class 1
    assert abs(yt.mean() - train_ratio) < 0.02
    assert abs(ye.mean() - train_ratio) < 0.02


def test_exp_yamls_do_not_have_in_sample_test_files():
    """exp1/exp2 yaml must not set test_files == train_files (I-ST2)."""
    import yaml
    for name in ("exp1.yaml", "exp2.yaml"):
        p = os.path.join(_PROJECT_ROOT, "tools", "spec_trainer", "config", name)
        with open(p) as f:
            cfg = yaml.safe_load(f)
        train_files = cfg["data"].get("train_files", [])
        test_files = cfg["data"].get("test_files", [])
        if test_files and set(test_files) == set(train_files):
            raise AssertionError(
                f"{name}: test_files == train_files — in-sample AUC! (I-ST2)")
        if not test_files:
            assert "test_size" in cfg["data"], (
                f"{name}: when test_files is empty, must set data.test_size (I-ST2)")
            assert 0.0 < cfg["data"]["test_size"] < 1.0, (
                f"{name}: test_size must be in (0, 1)")


def test_both_exp_yamls_set_is_unbalance_for_imbalanced_data():
    """Both exp1 and exp2 must set is_unbalance: True for ~1% positive
    data (P1-4, Pipeline-I3)."""
    import os
    import yaml
    project_root = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
    for name in ("exp1.yaml", "exp2.yaml"):
        p = os.path.join(project_root, "tools", "spec_trainer", "config", name)
        with open(p) as f:
            cfg = yaml.safe_load(f)
        params = cfg.get("model", {}).get("params", {})
        is_unbalance = params.get("is_unbalance", False)
        assert is_unbalance is True, (
            f"{name}: lightgbm is_unbalance must be True for imbalanced "
            f"SILAC data (~1% positives). Got {is_unbalance}. "
            f"See P1-4, Pipeline-I3 (2026-06-03 audit).")
