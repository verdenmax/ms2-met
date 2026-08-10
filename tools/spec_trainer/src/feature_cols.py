"""Feature column resolution for spec_trainer.

Extracted from main.py so unit tests can exercise the logic without
importing lightgbm/sklearn (see review finding I-ST1, 2026-06-03 audit).
"""
import logging
import os

import pandas as pd

try:
    # Package import: ``tools.spec_trainer.src.feature_cols``.
    from .feature_groups import (
        FEATURE_GROUPS,
        METADATA_COLUMNS,
        REGISTERED_COLUMNS,
        TRAINING_EXCLUDED_COLUMNS,
        experiment_arm_features,
        resolve_experiment_arm,
    )
except ImportError:  # Script entry points put this ``src`` directory on PATH.
    from feature_groups import (
        FEATURE_GROUPS,
        METADATA_COLUMNS,
        REGISTERED_COLUMNS,
        TRAINING_EXCLUDED_COLUMNS,
        experiment_arm_features,
        resolve_experiment_arm,
    )


# META columns that are not features themselves (PSM identification + label).
# 与 tools/eval_baseline.py:37-41 保持一致。
META_COLUMNS = METADATA_COLUMNS

# 额外排除的特征列。理由分两类:
#
# (A) 非物理过拟合信号:
#   - modification_count: 负样本 entrapment 大多带修饰，模型容易学到
#     "带修饰 → 负例" 这种非物理规则。见 PLAN.md 三-2 分析。
#
# (B) 数据集 ID / 跨数据集泄露 (cross_test 场景必排):
#   - window_width: 在每个数据集内是常量 (2da=2, 5da=5, normal=20)，
#     本质上等价于 "哪个数据集" 标签。cross_test 场景下训练时该列变成
#     dataset ID，测试时取 OOD 值，树模型行为不可控。
#   - fragment_xic_empty_count: 在所有 3 个数据集均为常量 0，无信息。
#   - fragment_same_mass_count: 跨数据集均值 0.097/2.0/8.7，直接被 DIA
#     窗宽决定 (窗越宽，落在同窗内的同质量碎片越多)。cross_test 场景下
#     成为 dataset 代理特征。
#   - fragment_heavy_absent_count: 跨数据集均值 1.8/1.8/0.1，窗变窄后
#     漏检率剧变；同样是 dataset 代理。
#   - q_value: PSM 鉴定置信度 (来自 pFind FDR)，是 META/分析列而非物理特征。
#     写入 features.csv 仅供 FDR-binning 分析使用，绝不能成为训练特征。
EXCLUDED_EXTRA = TRAINING_EXCLUDED_COLUMNS


def prefer_canonical_shift_feature(columns):
    """Drop the deprecated shift alias when the canonical feature exists."""
    result = list(columns)
    if "total_label_shift" in result and "total_silac_shift" in result:
        result.remove("total_silac_shift")
    return result


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

    result = prefer_canonical_shift_feature([
        c for c in ordered
        if c in intersection
        and c not in META_COLUMNS
        and c not in EXCLUDED_EXTRA
        and c != target_col
    ])
    if not result:
        raise ValueError(
            f"resolve_feature_cols returned 0 features from "
            f"{sample_csv_paths}; all columns are in META_COLUMNS / "
            f"EXCLUDED_EXTRA / target_col. Check yaml feature_cols or "
            f"add features to the CSV. (P2-7, Silent-I5)"
        )
    return result


def resolve_configured_feature_cols(data_cfg, sample_csv_paths, target_col):
    """Resolve either a legacy feature list or a formal experiment arm.

    ``feature_arm`` opts into the strict scientific registry. It is mutually
    exclusive with a non-empty ``feature_cols`` list so a configuration cannot
    silently claim one ablation arm while training another set. The optional
    ``drop_features`` list is validated against registered model inputs and is
    then applied to the resolved arm; entries belonging to another arm are
    harmless, which allows one shared pruning list across an ablation matrix.

    Configurations without ``feature_arm`` retain the legacy auto-detection
    behaviour for backward compatibility.
    """
    explicit = data_cfg.get("feature_cols")
    arm = data_cfg.get("feature_arm")
    drop_features = list(data_cfg.get("drop_features") or [])

    if not arm:
        if drop_features:
            raise ValueError(
                "drop_features requires feature_arm so pruning is tied to a "
                "registered scientific feature set")
        return resolve_feature_cols(explicit, sample_csv_paths, target_col)

    if explicit:
        raise ValueError(
            "data.feature_arm and non-empty data.feature_cols are mutually "
            "exclusive")

    if isinstance(sample_csv_paths, str):
        sample_csv_paths = [sample_csv_paths]
    if not sample_csv_paths:
        raise ValueError("feature_arm resolution requires at least one CSV")

    per_file_headers = [
        list(pd.read_csv(path, nrows=0).columns)
        for path in sample_csv_paths
    ]
    common = set(per_file_headers[0])
    for header in per_file_headers[1:]:
        common.intersection_update(header)
    ordered_common = [c for c in per_file_headers[0] if c in common]
    selected = resolve_experiment_arm(arm, ordered_common, strict=True)

    model_features = set().union(*FEATURE_GROUPS.values())
    unknown_drops = set(drop_features) - REGISTERED_COLUMNS
    invalid_drops = set(drop_features) - model_features
    if unknown_drops:
        raise ValueError(
            f"unknown drop_features: {sorted(unknown_drops)}")
    if invalid_drops:
        raise ValueError(
            "drop_features must name registered model inputs, got: "
            f"{sorted(invalid_drops)}")

    drop_set = set(drop_features)
    result = [feature for feature in selected if feature not in drop_set]
    if not result:
        raise ValueError(
            f"feature arm {arm!r} has zero features after drop_features")
    if data_cfg.get("require_complete_arm", False):
        expected = set(experiment_arm_features(arm)) - drop_set
        missing = sorted(expected - set(result))
        unexpected = sorted(set(result) - expected)
        if missing or unexpected:
            raise ValueError(
                f"feature arm {arm!r} schema drift: missing={missing}, "
                f"unexpected={unexpected}; regenerate a complete feature "
                "snapshot or explicitly disable require_complete_arm")
    return result
