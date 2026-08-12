"""Baseline evaluation with incorrect identifications as the positive class.

用法:
    python tools/eval_baseline.py \
        --features runs/baseline_2da/features.csv \
        --output runs/baseline_2da/baseline_metrics.json

输出:
    - 5-fold CV 指标（mean ± std）
    - 全数据训练后的特征重要性（gain）
    - Error-detection ROC-AUC / PR-AUC and fixed-FPR working points
"""
from __future__ import annotations

import argparse
import json
import logging
import os
import sys
from pathlib import Path

import numpy as np
import pandas as pd

_PROJECT_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
if _PROJECT_ROOT not in sys.path:
    sys.path.insert(0, _PROJECT_ROOT)

from spectrum.species_marker import matches_species_marker
from tools.spec_trainer.src.feature_cols import prefer_canonical_shift_feature
from tools.spec_trainer.src.cv_core import (
    METRIC_SEMANTICS_VERSION, evaluate_oof, working_points)

logging.basicConfig(level=logging.INFO,
                    format="%(asctime)s | %(levelname)s | %(message)s")
logger = logging.getLogger(__name__)


META_COLUMNS = {
    "sequence", "charge", "raw_title1", "raw_title2", "labeling",
    "isotope_model",
    "protein_names", "label", "label_type",
    "precursor_mz", "sequence_len", "rt",
    "negative_source", "negative_confidence", "query_id", "parent_id",
    "group_id", "pair_id", "candidate_family_id", "peptide_group_id",
    "generator", "generator_seed", "heavy_confirmed",
}


def derive_binary_label(df: pd.DataFrame, marker: str = "HUMAN") -> pd.Series:
    """从 features.csv 中提取二分类标签。

    优先级:
      1. 新 schema 的 ``label_type`` 列（"positive"/"negative" 字符串）
      2. 老 schema 中 ``label`` 列已经是 0/1（兼容 pair flow 路径）
      3. 从 ``protein_names`` 列派生（``matches_species_marker`` → 正例），
         与 tools/extract_common.py 共享 marker 检查规则（suffix + 排除 decoy）
    """
    if "label_type" in df.columns:
        label_type = df["label_type"]
        non_null = label_type.dropna()
        if len(non_null) > 0:
            mapping = {"positive": 1, "negative": 0}
            mapped = label_type.map(mapping)
            if mapped.notna().sum() > 0:
                logger.info("Using 'label_type' column (positive/negative).")
                return mapped.fillna(-1).astype(int)

    if "label" in df.columns:
        label_col = df["label"]
        try:
            numeric = pd.to_numeric(label_col, errors="raise")
            uniq = set(numeric.dropna().unique().tolist())
            if uniq.issubset({0, 1}):
                logger.info("Using 'label' column (already 0/1).")
                return numeric.astype(int)
        except (ValueError, TypeError):
            pass

    if "protein_names" in df.columns:
        logger.info(
            "Deriving binary label from 'protein_names' via "
            "matches_species_marker(marker=%r) (suffix + decoy-aware).",
            marker)
        return df["protein_names"].fillna("").apply(
            lambda s: int(matches_species_marker(s, marker))
        )

    raise ValueError(
        "无法确定二分类 label: 缺少 label_type / 数值 label / protein_names 任一列")


def load_features(
        path: Path, marker: str = "HUMAN"
) -> tuple[pd.DataFrame, pd.Series, list[str]]:
    df = pd.read_csv(path)
    logger.info("Loaded %d rows, %d columns from %s",
                len(df), df.shape[1], path)

    y = derive_binary_label(df, marker=marker)

    feature_cols = prefer_canonical_shift_feature(
        [c for c in df.columns if c not in META_COLUMNS])
    logger.info("Using %d feature columns", len(feature_cols))

    X = df[feature_cols].copy()
    # HistGradientBoostingClassifier handles NaN natively; do NOT fillna(0.0)
    # which would conflate "no data" with "value=0".
    X = X.replace([np.inf, -np.inf], np.nan)

    pos = int((y == 1).sum())
    neg = int((y == 0).sum())
    other = int(((y != 0) & (y != 1)).sum())
    logger.info("Positive (HUMAN/correct): %d, Negative (trap/wrong): %d, "
                "Unknown: %d, ratio %.1f:1",
                pos, neg, other, pos / max(neg, 1))
    if other > 0:
        logger.warning("有 %d 条 PSM 的 label 既不是 positive 也不是 negative，"
                       "已被过滤", other)
        mask = (y == 0) | (y == 1)
        X = X[mask].reset_index(drop=True)
        y = y[mask].reset_index(drop=True)

    return X, y, feature_cols


def compute_working_points(y_true: np.ndarray, y_score: np.ndarray) -> dict:
    """Canonical error-positive fixed-FPR operating points."""
    return working_points(y_true, y_score)


def cv_evaluate(X: pd.DataFrame, y: pd.Series, n_splits: int = 5,
                random_state: int = 42) -> dict:
    from sklearn.ensemble import HistGradientBoostingClassifier
    from sklearn.metrics import matthews_corrcoef
    from sklearn.model_selection import StratifiedKFold

    skf = StratifiedKFold(
        n_splits=n_splits, shuffle=True, random_state=random_state)

    fold_metrics = []
    all_y_true = []
    all_y_score = []
    for fold_idx, (train_idx, val_idx) in enumerate(skf.split(X, y), 1):
        X_train, X_val = X.iloc[train_idx], X.iloc[val_idx]
        y_train, y_val = y.iloc[train_idx], y.iloc[val_idx]

        clf = HistGradientBoostingClassifier(
            max_iter=300,
            learning_rate=0.05,
            max_depth=6,
            l2_regularization=1.0,
            random_state=random_state,
            class_weight="balanced",
        )
        clf.fit(X_train, y_train)
        y_proba = clf.predict_proba(X_val)[:, 1]
        y_pred = (y_proba >= 0.5).astype(int)

        evaluated = evaluate_oof(y_val, y_proba)
        metrics = {
            "fold": fold_idx,
            "roc_auc": evaluated["roc_auc"],
            "error_pr_auc": evaluated["error_pr_auc"],
            "fnr_at_fpr5": evaluated["fnr_at_fpr5"],
            "mcc": float(matthews_corrcoef(1 - y_val, 1 - y_pred)),
        }
        logger.info(
            "Fold %d: error ROC-AUC=%.4f PR-AUC=%.4f "
            "FNR@FPR5=%.4f MCC=%.4f", fold_idx, metrics["roc_auc"],
            metrics["error_pr_auc"], metrics["fnr_at_fpr5"], metrics["mcc"])
        fold_metrics.append(metrics)
        all_y_true.append(y_val.values)
        all_y_score.append(y_proba)

    y_true_concat = np.concatenate(all_y_true)
    y_score_concat = np.concatenate(all_y_score)

    pooled = evaluate_oof(y_true_concat, y_score_concat)
    summary = {
        **pooled,
        "roc_auc_mean": float(np.mean([m["roc_auc"] for m in fold_metrics])),
        "roc_auc_std": float(np.std([m["roc_auc"] for m in fold_metrics])),
        "error_pr_auc_mean": float(np.mean(
            [m["error_pr_auc"] for m in fold_metrics])),
        "error_pr_auc_std": float(np.std(
            [m["error_pr_auc"] for m in fold_metrics])),
        "mcc_mean": float(np.mean([m["mcc"] for m in fold_metrics])),
        "mcc_std": float(np.std([m["mcc"] for m in fold_metrics])),
        "fold_metrics": fold_metrics,
    }
    return summary


def compute_feature_importance(
        X: pd.DataFrame, y: pd.Series,
        feature_cols: list[str], random_state: int = 42,
        n_repeats: int = 5) -> list[dict]:
    from sklearn.ensemble import HistGradientBoostingClassifier
    from sklearn.inspection import permutation_importance

    clf = HistGradientBoostingClassifier(
        max_iter=300, learning_rate=0.05, max_depth=6,
        l2_regularization=1.0, random_state=random_state,
        class_weight="balanced",
    )
    clf.fit(X, y)

    logger.info("Computing permutation importance (n_repeats=%d)...", n_repeats)
    def error_average_precision(estimator, features, stored_labels):
        from sklearn.metrics import average_precision_score
        trust_scores = estimator.predict_proba(features)[:, 1]
        return average_precision_score(1 - stored_labels, 1.0 - trust_scores)

    perm = permutation_importance(
        clf, X, y, n_repeats=n_repeats,
        random_state=random_state, scoring=error_average_precision, n_jobs=-1)

    ranked = sorted(
        zip(feature_cols, perm.importances_mean, perm.importances_std),
        key=lambda x: x[1], reverse=True,
    )
    return [
        {"feature": name, "importance_mean": float(m), "importance_std": float(s)}
        for name, m, s in ranked
    ]


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--features", required=True, type=Path)
    parser.add_argument("--output", required=True, type=Path)
    parser.add_argument("--skip-importance", action="store_true",
                        help="跳过 permutation importance（速度慢时）")
    parser.add_argument("--positive-marker", default="HUMAN",
                        help="Species marker for positive label derivation "
                             "(only used in protein_names fallback tier)")
    args = parser.parse_args()

    args.output.parent.mkdir(parents=True, exist_ok=True)

    X, y, feature_cols = load_features(args.features, marker=args.positive_marker)

    logger.info("=== 5-fold CV ===")
    cv_summary = cv_evaluate(X, y)

    logger.info(
        "CV Mean: error ROC-AUC=%.4f±%.4f  PR-AUC=%.4f±%.4f  "
        "MCC=%.4f±%.4f",
        cv_summary["roc_auc_mean"], cv_summary["roc_auc_std"],
        cv_summary["error_pr_auc_mean"], cv_summary["error_pr_auc_std"],
        cv_summary["mcc_mean"], cv_summary["mcc_std"],
    )
    logger.info("Working points: %s",
                json.dumps(cv_summary["working_points"], indent=2))

    importance = []
    if not args.skip_importance:
        logger.info("=== Feature importance ===")
        importance = compute_feature_importance(X, y, feature_cols)
        logger.info("Top 15 features by error PR-AUC permutation drop:")
        for item in importance[:15]:
            logger.info("  %-32s  %.4f ± %.4f",
                        item["feature"], item["importance_mean"],
                        item["importance_std"])

    result = {
        "metric_semantics": METRIC_SEMANTICS_VERSION,
        "positive_class": "incorrect_identification",
        "n_samples": int(len(y)),
        "n_actual_correct": int((y == 1).sum()),
        "n_actual_error": int((y == 0).sum()),
        "n_features": len(feature_cols),
        "cv_summary": cv_summary,
        "feature_importance": importance,
    }
    with args.output.open("w", encoding="utf-8") as f:
        json.dump(result, f, indent=2, ensure_ascii=False)
    logger.info("Wrote results to %s", args.output)


if __name__ == "__main__":
    main()
