"""Post-feature-extraction filters applied to the assembled features table.

These run after the per-PSM XIC features have been computed (so DIA-derived
columns like ``heavy_out_of_range`` already exist), but before the features
CSV is written and consumed by downstream training/eval.

Currently one filter:

  - ``filter_heavy_out_of_range``: drops PSMs whose heavy SILAC precursor m/z
    fell outside the raw's acquisition range (``heavy_out_of_range == 1``).
    Such PSMs have no acquired heavy channel, so they cannot be validated by
    SILAC fragment evidence. They are dropped for BOTH classes (positive and
    negative) — symmetric, to avoid teaching the model a spurious
    "out-of-range => positive" rule (which would happen if only the trap
    negatives were dropped while the out-of-range positives were kept).
"""
from __future__ import annotations

import pandas as pd


def filter_heavy_out_of_range(df: pd.DataFrame):
    """Drop rows with ``heavy_out_of_range == 1`` (both classes).

    Args:
        df: assembled per-PSM features table. May or may not carry the
            ``heavy_out_of_range`` column (e.g. non-speclib feature paths).

    Returns:
        ``(kept_df, n_pos_dropped, n_neg_dropped)``. ``kept_df`` has a reset
        index. If the column is absent the input is returned unchanged with
        zero drop counts. Class is read from ``label_type`` when present,
        else from ``label`` (1 = positive, 0 = negative).
    """
    if "heavy_out_of_range" not in df.columns:
        return df, 0, 0

    # Robust to int/float/bool/str encodings (CSV round-trips may stringify).
    vals = pd.to_numeric(df["heavy_out_of_range"], errors="coerce")
    out_mask = vals == 1

    if "label_type" in df.columns:
        n_pos = int((out_mask & (df["label_type"] == "positive")).sum())
        n_neg = int((out_mask & (df["label_type"] == "negative")).sum())
    elif "label" in df.columns:
        n_pos = int((out_mask & (df["label"] == 1)).sum())
        n_neg = int((out_mask & (df["label"] == 0)).sum())
    else:
        n_pos = int(out_mask.sum())
        n_neg = 0

    kept = df.loc[~out_mask].reset_index(drop=True)
    return kept, n_pos, n_neg
