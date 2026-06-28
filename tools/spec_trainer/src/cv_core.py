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
