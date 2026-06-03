"""Held-out set resolution for spec_trainer.

Extracted from main.py so the branching logic can be unit-tested without
importing lightgbm/sklearn-model-manager (see I-ST2 + rubber-duck N4,
2026-06-03 audit).
"""
from sklearn.model_selection import train_test_split


def resolve_holdout(
    X_train, y_train,
    train_files, test_files, test_size,
    feature_cols, target_col, loader,
):
    """Resolve (X_train, X_test, y_train, y_test) per the I-ST2 contract.

    Priority:
    1. test_files set AND distinct from train_files -> load it via `loader`.
       loader signature: loader(files, feature_cols, target_col) -> (X, y)
    2. else if test_size > 0 -> sklearn train_test_split with stratify=y,
       random_state=42.
    3. else -> raise ValueError (refuse silent in-sample evaluation).

    Returns: (X_train_out, X_test_out, y_train_out, y_test_out)
    """
    if test_files and set(test_files) != set(train_files):
        X_test, y_test = loader(test_files, feature_cols, target_col)
        return X_train, X_test, y_train, y_test
    if test_size and test_size > 0:
        X_tr, X_te, y_tr, y_te = train_test_split(
            X_train, y_train,
            test_size=test_size,
            random_state=42,
            stratify=y_train,
        )
        return X_tr, X_te, y_tr, y_te
    raise ValueError(
        "Neither distinct test_files nor data.test_size>0 provided — "
        "would evaluate in-sample (see I-ST2 audit 2026-06-03)")
