"""Cohort filters for scientifically paired feature-ablation experiments."""
from __future__ import annotations

import pandas as pd


# All formal evidence arms are compared on these same evaluable rows. Keeping
# the rules explicit makes the selection auditable in the result JSON and
# prevents availability flags from becoming classifier inputs.
COHORT_DEFINITIONS = {
    "none": (),
    "evidence_common": (
        ("heavy_in_raw", 1),
        ("heavy_out_of_range", 0),
        ("precursor_xic_empty", 0),
        ("q1a_valid", 1),
        ("has_lib_pred", 1),
        ("isotope_model_valid", 1),
    ),
}


def _class_counts(df, target_col):
    if target_col not in df:
        raise ValueError(f"target column {target_col!r} is missing")
    return {
        "n_rows": int(len(df)),
        "n_correct": int(df[target_col].eq(1).sum()),
        "n_error": int(df[target_col].eq(0).sum()),
    }


def apply_training_cohort(df, cohort_name, *, target_col="label"):
    """Filter a frame and return ``(filtered_frame, JSON-safe audit)``.

    Missing or non-numeric eligibility values fail their rule. The input frame
    is never mutated, and the retained rows preserve their original order.
    """
    name = cohort_name or "none"
    if name not in COHORT_DEFINITIONS:
        raise ValueError(
            f"unknown cohort {name!r}; expected one of "
            f"{sorted(COHORT_DEFINITIONS)}")

    rules = COHORT_DEFINITIONS[name]
    missing = [column for column, _ in rules if column not in df]
    if missing:
        raise ValueError(
            f"cohort {name!r} requires missing columns: {missing}")

    mask = pd.Series(True, index=df.index)
    failed_by_rule = {}
    for column, expected in rules:
        passed = pd.to_numeric(df[column], errors="coerce").eq(expected)
        failed_by_rule[f"{column} == {expected}"] = int((~passed).sum())
        mask &= passed

    filtered = df.loc[mask].reset_index(drop=True)
    audit = {
        "name": name,
        "rules": [
            {"column": column, "equals": expected}
            for column, expected in rules
        ],
        "failed_by_rule": failed_by_rule,
        "before": _class_counts(df, target_col),
        "after": _class_counts(filtered, target_col),
    }
    if (audit["after"]["n_correct"] == 0
            or audit["after"]["n_error"] == 0):
        raise ValueError(
            f"cohort {name!r} removed one class: {audit['after']}")
    return filtered, audit
