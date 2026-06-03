"""Feature column resolution for spec_trainer.

Extracted from main.py so unit tests can exercise the logic without
importing lightgbm/sklearn (see review finding I-ST1, 2026-06-03 audit).
"""
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


def resolve_feature_cols(explicit, sample_csv_path, target_col):
    """Resolve final feature column list.

    If explicit is a non-empty list, return it unchanged (yaml took
    care of selection). Otherwise auto-detect from the CSV column
    header, excluding META_COLUMNS + EXCLUDED_EXTRA + target_col.

    The CSV column order determines the feature order (pandas read_csv
    is deterministic for a given file). Cross-runs with the same
    features.csv produce the same feature_cols list.
    """
    if explicit:
        return list(explicit)
    sample_df = pd.read_csv(sample_csv_path, nrows=0)
    all_cols = list(sample_df.columns)
    return [
        c for c in all_cols
        if c not in META_COLUMNS
        and c not in EXCLUDED_EXTRA
        and c != target_col
    ]
