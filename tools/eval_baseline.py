"""Baseline 评估脚本：用当前 57 个特征跑 5-fold CV，输出 AUC / AUPRC / MCC / negative recall。

用法:
    python tools/eval_baseline.py \
        --features runs/baseline_2da/features.csv \
        --output runs/baseline_2da/baseline_metrics.json

输出:
    - 5-fold CV 指标（mean ± std）
    - 全数据训练后的特征重要性（gain）
    - 在 negative_recall 80%/90%/95% 三个工作点上的阈值与对应 positive precision
"""
from __future__ import annotations

import argparse
import json
import logging
from pathlib import Path

import numpy as np
import pandas as pd
from sklearn.ensemble import HistGradientBoostingClassifier
from sklearn.inspection import permutation_importance
from sklearn.metrics import (
    average_precision_score,
    matthews_corrcoef,
    precision_recall_curve,
    roc_auc_score,
)
from sklearn.model_selection import StratifiedKFold

logging.basicConfig(level=logging.INFO,
                    format="%(asctime)s | %(levelname)s | %(message)s")
logger = logging.getLogger(__name__)


META_COLUMNS = {
    "sequence", "charge", "raw_title1", "protein_names", "label",
    "precursor_mz", "sequence_len",
}


def load_features(path: Path) -> tuple[pd.DataFrame, pd.Series, list[str]]:
    df = pd.read_csv(path)
    logger.info("Loaded %d rows, %d columns from %s",
                len(df), df.shape[1], path)

    if "label" not in df.columns:
        raise ValueError("features.csv 缺少 label 列")

    feature_cols = [c for c in df.columns if c not in META_COLUMNS]
    logger.info("Using %d feature columns", len(feature_cols))

    X = df[feature_cols].copy()
    X = X.replace([np.inf, -np.inf], np.nan).fillna(0.0)

    y = df["label"].astype(int)
    pos = int((y == 1).sum())
    neg = int((y == 0).sum())
    logger.info("Positive (HUMAN/correct): %d, Negative (trap/wrong): %d, ratio %.1f:1",
                pos, neg, pos / max(neg, 1))

    return X, y, feature_cols


def compute_working_points(y_true: np.ndarray, y_score: np.ndarray) -> dict:
    """计算几个工作点：固定 negative recall，看 positive precision/recall。"""
    fpr_targets = [0.05, 0.10, 0.20]  # neg_recall = 1 - fpr = 0.95/0.90/0.80
    result = {}

    pos_scores = y_score[y_true == 1]
    neg_scores = y_score[y_true == 0]
    for fpr in fpr_targets:
        thr = float(np.quantile(neg_scores, 1 - fpr))
        pos_kept = (pos_scores >= thr).sum()
        pos_total = len(pos_scores)
        neg_kept = (neg_scores >= thr).sum()
        result[f"neg_recall_{int((1-fpr)*100)}"] = {
            "threshold": thr,
            "pos_recall": float(pos_kept / max(pos_total, 1)),
            "neg_recall": float(1 - neg_kept / max(len(neg_scores), 1)),
        }
    return result


def cv_evaluate(X: pd.DataFrame, y: pd.Series, n_splits: int = 5,
                random_state: int = 42) -> dict:
    skf = StratifiedKFold(
        n_splits=n_splits, shuffle=True, random_state=random_state)

    pos = int((y == 1).sum())
    neg = int((y == 0).sum())
    sample_weight_pos = neg / max(pos, 1)

    fold_metrics = []
    all_y_true = []
    all_y_score = []
    for fold_idx, (train_idx, val_idx) in enumerate(skf.split(X, y), 1):
        X_train, X_val = X.iloc[train_idx], X.iloc[val_idx]
        y_train, y_val = y.iloc[train_idx], y.iloc[val_idx]

        sample_weight = np.where(y_train == 1, 1.0, sample_weight_pos)

        clf = HistGradientBoostingClassifier(
            max_iter=300,
            learning_rate=0.05,
            max_depth=6,
            l2_regularization=1.0,
            random_state=random_state,
            class_weight="balanced",
        )
        clf.fit(X_train, y_train, sample_weight=sample_weight)
        y_proba = clf.predict_proba(X_val)[:, 1]
        y_pred = (y_proba >= 0.5).astype(int)

        metrics = {
            "fold": fold_idx,
            "auc": float(roc_auc_score(y_val, y_proba)),
            "auprc": float(average_precision_score(y_val, y_proba)),
            "mcc": float(matthews_corrcoef(y_val, y_pred)),
        }
        logger.info("Fold %d: AUC=%.4f AUPRC=%.4f MCC=%.4f",
                    fold_idx, metrics["auc"], metrics["auprc"], metrics["mcc"])
        fold_metrics.append(metrics)
        all_y_true.append(y_val.values)
        all_y_score.append(y_proba)

    y_true_concat = np.concatenate(all_y_true)
    y_score_concat = np.concatenate(all_y_score)

    summary = {
        "auc_mean": float(np.mean([m["auc"] for m in fold_metrics])),
        "auc_std": float(np.std([m["auc"] for m in fold_metrics])),
        "auprc_mean": float(np.mean([m["auprc"] for m in fold_metrics])),
        "auprc_std": float(np.std([m["auprc"] for m in fold_metrics])),
        "mcc_mean": float(np.mean([m["mcc"] for m in fold_metrics])),
        "mcc_std": float(np.std([m["mcc"] for m in fold_metrics])),
        "fold_metrics": fold_metrics,
        "working_points": compute_working_points(
            y_true_concat, y_score_concat),
    }
    return summary


def compute_feature_importance(
        X: pd.DataFrame, y: pd.Series,
        feature_cols: list[str], random_state: int = 42,
        n_repeats: int = 5) -> list[dict]:
    pos = int((y == 1).sum())
    neg = int((y == 0).sum())
    sample_weight = np.where(y == 1, 1.0, neg / max(pos, 1))

    clf = HistGradientBoostingClassifier(
        max_iter=300, learning_rate=0.05, max_depth=6,
        l2_regularization=1.0, random_state=random_state,
        class_weight="balanced",
    )
    clf.fit(X, y, sample_weight=sample_weight)

    logger.info("Computing permutation importance (n_repeats=%d)...", n_repeats)
    perm = permutation_importance(
        clf, X, y, n_repeats=n_repeats,
        random_state=random_state, scoring="average_precision", n_jobs=-1)

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
    args = parser.parse_args()

    args.output.parent.mkdir(parents=True, exist_ok=True)

    X, y, feature_cols = load_features(args.features)

    logger.info("=== 5-fold CV ===")
    cv_summary = cv_evaluate(X, y)

    logger.info(
        "CV Mean: AUC=%.4f±%.4f  AUPRC=%.4f±%.4f  MCC=%.4f±%.4f",
        cv_summary["auc_mean"], cv_summary["auc_std"],
        cv_summary["auprc_mean"], cv_summary["auprc_std"],
        cv_summary["mcc_mean"], cv_summary["mcc_std"],
    )
    logger.info("Working points: %s",
                json.dumps(cv_summary["working_points"], indent=2))

    importance = []
    if not args.skip_importance:
        logger.info("=== Feature importance ===")
        importance = compute_feature_importance(X, y, feature_cols)
        logger.info("Top 15 features by permutation AUPRC drop:")
        for item in importance[:15]:
            logger.info("  %-32s  %.4f ± %.4f",
                        item["feature"], item["importance_mean"],
                        item["importance_std"])

    result = {
        "n_samples": int(len(y)),
        "n_positive": int((y == 1).sum()),
        "n_negative": int((y == 0).sum()),
        "n_features": len(feature_cols),
        "cv_summary": cv_summary,
        "feature_importance": importance,
    }
    with args.output.open("w", encoding="utf-8") as f:
        json.dump(result, f, indent=2, ensure_ascii=False)
    logger.info("Wrote results to %s", args.output)


if __name__ == "__main__":
    main()
