"""特征分组对照实验：用 4 组特征集训练，看代谢轻/重配对证据的贡献。

特征分组:
  - sequence_only: 仅肽段序列属性（modification_count, sequence_kr_count,
    sequence_len,
    valid_fragment_ions_num, total_label_shift, window_width, precursor_centering,
    precursor_mz, charge）
  - label_evidence_only: 仅轻/重配对类（precursor_*, all_*, b_*, y_*, isotope_correlation,
    mass_shift_error, frag_corr_weighted, matched_intensity_percent）
  - all:           全部
  - label_evidence_minus_intensity: 配对证据去掉绝对强度类，
    避免和肽段丰度泄漏

Every metric treats incorrect identifications as the positive class.
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

from tools.eval_baseline import derive_binary_label
from tools.spec_trainer.src.feature_cols import prefer_canonical_shift_feature
from tools.spec_trainer.src.cv_core import evaluate_oof

logging.basicConfig(level=logging.INFO,
                    format="%(asctime)s | %(levelname)s | %(message)s")
logger = logging.getLogger(__name__)


SEQUENCE_FEATURES = {
    "modification_count", "sequence_kr_count", "kr_count", "sequence_len",
    "valid_fragment_ions_num", "total_label_shift", "total_silac_shift",
    "window_width", "precursor_centering",
    "heavy_in_raw",
}

INTENSITY_FEATURES = {
    "precursor_light_max_int", "precursor_heavy_max_int",
    "precursor_snr", "all_snr_mean", "all_snr_p50", "all_snr_std",
}

ID_COLUMNS = {
    "sequence", "charge", "raw_title1", "raw_title2", "labeling",
    "isotope_model",
    "protein_names", "label", "label_type",
    "precursor_mz", "rt",
    "negative_source", "negative_confidence", "query_id", "parent_id",
    "group_id", "pair_id", "candidate_family_id", "peptide_group_id",
    "generator", "generator_seed", "heavy_confirmed",
}


def split_features(all_features: list[str]) -> dict[str, list[str]]:
    label_evidence = [f for f in all_features if f not in SEQUENCE_FEATURES]
    label_evidence_no_intensity = [
        f for f in label_evidence if f not in INTENSITY_FEATURES]
    return {
        "sequence_only": [f for f in all_features if f in SEQUENCE_FEATURES],
        "label_evidence_only": label_evidence,
        "label_evidence_minus_intensity": label_evidence_no_intensity,
        "all": list(all_features),
    }


def cv_one(X: pd.DataFrame, y: pd.Series, name: str,
           n_splits: int = 5) -> dict:
    from sklearn.ensemble import HistGradientBoostingClassifier
    from sklearn.model_selection import StratifiedKFold

    skf = StratifiedKFold(n_splits=n_splits, shuffle=True, random_state=42)
    aucs, error_pr_aucs, fnrs = [], [], []
    all_y_true, all_y_score = [], []
    for fold_idx, (train_idx, val_idx) in enumerate(skf.split(X, y), 1):
        X_train, X_val = X.iloc[train_idx], X.iloc[val_idx]
        y_train, y_val = y.iloc[train_idx], y.iloc[val_idx]

        clf = HistGradientBoostingClassifier(
            max_iter=300, learning_rate=0.05, max_depth=6,
            l2_regularization=1.0, random_state=42,
            class_weight="balanced",
        )
        clf.fit(X_train, y_train)
        proba = clf.predict_proba(X_val)[:, 1]
        metrics = evaluate_oof(y_val, proba)
        aucs.append(metrics["roc_auc"])
        error_pr_aucs.append(metrics["error_pr_auc"])
        fnrs.append(metrics["fnr_at_fpr5"])
        all_y_true.append(y_val.values)
        all_y_score.append(proba)

    y_true = np.concatenate(all_y_true)
    y_score = np.concatenate(all_y_score)

    pooled = evaluate_oof(y_true, y_score)

    return {
        "metric_semantics": pooled["metric_semantics"],
        "positive_class": pooled["positive_class"],
        "name": name,
        "n_features": X.shape[1],
        "roc_auc": pooled["roc_auc"],
        "error_pr_auc": pooled["error_pr_auc"],
        "fnr_at_fpr5": pooled["fnr_at_fpr5"],
        "error_recall_at_fpr10": pooled["error_recall_at_fpr10"],
        "roc_auc_mean": float(np.mean(aucs)),
        "roc_auc_std": float(np.std(aucs)),
        "error_pr_auc_mean": float(np.mean(error_pr_aucs)),
        "error_pr_auc_std": float(np.std(error_pr_aucs)),
        "fnr_at_fpr5_mean": float(np.mean(fnrs)),
        "working_points": pooled["working_points"],
    }


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--features", required=True, type=Path)
    parser.add_argument("--output", required=True, type=Path)
    args = parser.parse_args()

    args.output.parent.mkdir(parents=True, exist_ok=True)

    df = pd.read_csv(args.features)
    y = derive_binary_label(df)
    mask = (y == 0) | (y == 1)
    df = df[mask].reset_index(drop=True)
    y = y[mask].reset_index(drop=True)

    all_feats = prefer_canonical_shift_feature(
        [c for c in df.columns if c not in ID_COLUMNS])
    groups = split_features(all_feats)

    logger.info("=== Feature groups ===")
    for name, feats in groups.items():
        logger.info("  %s: %d features", name, len(feats))

    results = []
    for name, feats in groups.items():
        if not feats:
            continue
        X = df[feats].replace([np.inf, -np.inf], np.nan)
        logger.info("Running CV for group '%s' (%d features)...",
                    name, len(feats))
        res = cv_one(X, y, name)
        results.append(res)
        logger.info(
            "  %s: error ROC-AUC=%.4f±%.4f PR-AUC=%.4f "
            "FNR@FPR5=%.4f error recall@FPR10=%.4f",
            name, res["roc_auc_mean"], res["roc_auc_std"],
            res["error_pr_auc"], res["fnr_at_fpr5"],
            res["error_recall_at_fpr10"])

    with args.output.open("w", encoding="utf-8") as f:
        json.dump(results, f, indent=2, ensure_ascii=False)
    logger.info("Wrote results to %s", args.output)


if __name__ == "__main__":
    main()
