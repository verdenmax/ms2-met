"""Feature column resolution for spec_trainer.

Extracted from main.py so unit tests can exercise the logic without
importing lightgbm/sklearn (see review finding I-ST1, 2026-06-03 audit).
"""
import logging
import os

import pandas as pd


# META columns that are not features themselves (PSM identification + label).
# 与 tools/eval_baseline.py:37-41 保持一致。
META_COLUMNS = {
    "sequence", "charge", "raw_title1", "raw_title2",
    "protein_names", "label", "label_type",
    "precursor_mz", "sequence_len",
}

# 额外排除的特征列：modification_count 在训练时倾向于过拟合非物理信号
# （负样本 entrapment 大多带修饰），见 PLAN.md 三-2 分析。
EXCLUDED_EXTRA = {"modification_count"}


def resolve_feature_cols(explicit, sample_csv_paths, target_col):
    """Resolve final feature column list.

    Args:
        explicit: yaml-provided list of features, or None/[] to auto-detect.
        sample_csv_paths: list of CSV paths whose headers will be intersected.
            Accepts either a list of paths or a single string path (back-compat).
        target_col: name of the label column (excluded from features).

    Returns:
        List of feature column names. If sample_csv_paths has multiple
        entries, returns the INTERSECTION of all files' columns minus
        META_COLUMNS + EXCLUDED_EXTRA + target_col. Logs a warning if
        the intersection is smaller than any individual file's column set
        (indicating schema drift). Order follows the first file's header.
    """
    if explicit:
        return list(explicit)

    if isinstance(sample_csv_paths, str):
        sample_csv_paths = [sample_csv_paths]

    per_file_cols = []
    for path in sample_csv_paths:
        df = pd.read_csv(path, nrows=0)
        per_file_cols.append(set(df.columns))

    intersection = set.intersection(*per_file_cols) if per_file_cols else set()

    for path, cols in zip(sample_csv_paths, per_file_cols):
        dropped = cols - intersection
        if dropped:
            logging.warning(
                "resolve_feature_cols: %d columns in %s not in intersection "
                "\u2014 dropped from feature set: %s (P1-5, Pipeline-I5)",
                len(dropped), os.path.basename(path), sorted(dropped))

    first_df = pd.read_csv(sample_csv_paths[0], nrows=0) if sample_csv_paths else None
    ordered = list(first_df.columns) if first_df is not None else []

    return [
        c for c in ordered
        if c in intersection
        and c not in META_COLUMNS
        and c not in EXCLUDED_EXTRA
        and c != target_col
    ]
