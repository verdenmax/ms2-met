"""特征分组对照实验：用 4 组特征集分别训练，看 SILAC 配对信号的真实贡献。

特征分组:
  - sequence_only: 仅肽段序列属性（modification_count, kr_count, sequence_len,
    valid_fragment_ions_num, total_silac_shift, window_width, precursor_centering,
    precursor_mz, charge）
  - silac_only:    仅 SILAC 配对类（precursor_*, all_*, b_*, y_*, isotope_correlation,
    mass_shift_error, frag_corr_weighted, matched_intensity_percent）
  - all:           全部
  - silac_minus_intensity: SILAC 但去掉绝对强度类（precursor_*_max_int 等），
    避免和肽段丰度泄漏

每组跑 5-fold CV，对比 AUC / pos_recall@neg_recall=95%/90%.
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

logging.basicConfig(level=logging.INFO,
                    format="%(asctime)s | %(levelname)s | %(message)s")
logger = logging.getLogger(__name__)


SEQUENCE_FEATURES = {
    "modification_count", "kr_count", "sequence_len",
    "valid_fragment_ions_num", "total_silac_shift",
    "window_width", "precursor_centering",
    "heavy_in_raw",
}

INTENSITY_FEATURES = {
    "precursor_light_max_int", "precursor_heavy_max_int",
    "precursor_snr", "all_snr_mean", "all_snr_p50", "all_snr_std",
}

ID_COLUMNS = {
    "sequence", "charge", "raw_title1", "raw_title2",
    "protein_names", "label", "label_type",
    "precursor_mz", "rt",
    "negative_source", "negative_confidence", "query_id", "parent_id",
    "group_id", "generator", "generator_seed", "heavy_confirmed",
}


def split_features(all_features: list[str]) -> dict[str, list[str]]:
    silac_all = [f for f in all_features if f not in SEQUENCE_FEATURES]
    silac_no_intensity = [f for f in silac_all if f not in INTENSITY_FEATURES]
    return {
        "sequence_only": [f for f in all_features if f in SEQUENCE_FEATURES],
        "silac_only": silac_all,
        "silac_minus_intensity": silac_no_intensity,
        "all": list(all_features),
    }


def cv_one(X: pd.DataFrame, y: pd.Series, name: str,
           n_splits: int = 5) -> dict:
    from sklearn.ensemble import HistGradientBoostingClassifier
    from sklearn.metrics import average_precision_score, roc_auc_score
    from sklearn.model_selection import StratifiedKFold

    skf = StratifiedKFold(n_splits=n_splits, shuffle=True, random_state=42)
    aucs, auprcs = [], []
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
        aucs.append(float(roc_auc_score(y_val, proba)))
        auprcs.append(float(average_precision_score(y_val, proba)))
        all_y_true.append(y_val.values)
        all_y_score.append(proba)

    y_true = np.concatenate(all_y_true)
    y_score = np.concatenate(all_y_score)

    pos_scores = y_score[y_true == 1]
    neg_scores = y_score[y_true == 0]
    wps = {}
    for fpr in (0.05, 0.10, 0.20):
        thr = float(np.quantile(neg_scores, 1 - fpr))
        wps[f"neg_recall_{int((1-fpr)*100)}"] = float(
            (pos_scores >= thr).sum() / len(pos_scores))

    return {
        "name": name,
        "n_features": X.shape[1],
        "auc_mean": float(np.mean(aucs)),
        "auc_std": float(np.std(aucs)),
        "auprc_mean": float(np.mean(auprcs)),
        "working_points": wps,
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

    all_feats = [c for c in df.columns if c not in ID_COLUMNS]
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
        wp = res["working_points"]
        logger.info(
            "  %s: AUC=%.4f±%.4f  pos_recall@neg95=%.3f  @neg90=%.3f  @neg80=%.3f",
            name, res["auc_mean"], res["auc_std"],
            wp["neg_recall_95"], wp["neg_recall_90"], wp["neg_recall_80"])

    with args.output.open("w", encoding="utf-8") as f:
        json.dump(results, f, indent=2, ensure_ascii=False)
    logger.info("Wrote results to %s", args.output)


if __name__ == "__main__":
    main()
