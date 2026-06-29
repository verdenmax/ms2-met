"""FDR-bin recall decay analysis for the 2Da spec_trainer ensemble.

Splits positive PSMs into pFind q_value bins and measures how well the CV
fold-ensemble recovers them at a fixed working-point threshold. A healthy
model recalls high-confidence (low-q) bins and decays toward the loose bins;
collapse in low-q bins flags a real problem.

The pure helpers (FDR_BINS / bin_recall) are lightgbm-free so they unit-test
without it; lightgbm is lazy-imported inside main() only (mirrors cv_train.py).
"""
import argparse
import os
import sys

import numpy as np

# spec_trainer/src on path for feature_cols + cv_core (mirrors cv_train pattern)
sys.path.insert(0, os.path.join(os.path.dirname(__file__), "spec_trainer", "src"))

FDR_BINS = [(0, 0.01), (0.01, 0.05), (0.05, 0.10), (0.10, 0.20), (0.20, 0.50)]


def bin_recall(df, fold_probas, thr):
    """Per-q-bin recall of the fold ensemble and of each fold individually.

    Args:
        df: DataFrame with a ``q_value`` column (rows aligned to fold_probas).
        fold_probas: list of np.array, one per fold, aligned to df rows.
        thr: keep-threshold; a row is "recalled" when proba >= thr.

    Bins are left-exclusive / right-inclusive (lo<q<=hi); q==0 falls into the
    first bin. Returns {(lo,hi): {n, ens_recall, fold_mean, fold_std}} where
    ens is the across-fold mean proba. Empty bins -> n=0 and NaN recalls.
    """
    q = np.asarray(df["q_value"], dtype="f8")
    ens = np.mean(np.vstack([np.asarray(p, dtype="f8") for p in fold_probas]),
                  axis=0)
    out = {}
    for lo, hi in FDR_BINS:
        mask = ((q > lo) & (q <= hi)) | ((lo == 0) & (q == 0))
        n = int(mask.sum())
        if n == 0:
            out[(lo, hi)] = {"n": 0, "ens_recall": float("nan"),
                             "fold_mean": float("nan"), "fold_std": float("nan")}
            continue
        ens_recall = float((ens[mask] >= thr).mean())
        fold_recalls = [float((np.asarray(p, dtype="f8")[mask] >= thr).mean())
                        for p in fold_probas]
        out[(lo, hi)] = {
            "n": n,
            "ens_recall": ens_recall,
            "fold_mean": float(np.mean(fold_recalls)),
            "fold_std": float(np.std(fold_recalls)),
        }
    return out


def _build_parser():
    p = argparse.ArgumentParser(description="FDR-bin recall decay analysis")
    p.add_argument("--model-prefix",
                   default="runs/spec_trainer/models/cv_in_2da_clean")
    p.add_argument("--folds", type=int, default=5)
    p.add_argument("--clean-csv", default="runs/spec_trainer/2da_clean.features.csv",
                   help="1% clean features.csv (features+label) for the threshold")
    p.add_argument("--pos50-csv", default="runs/spec_trainer/2da_pos50.features.csv",
                   help="pos50 features.csv with a q_value column to bin")
    p.add_argument("--out-csv", default="runs/spec_trainer/results/fdr_consistency_2da.csv")
    p.add_argument("--out-png", default="runs/spec_trainer/results/fdr_consistency_2da.png")
    return p


def main(argv=None):
    import lightgbm as lgb
    import pandas as pd
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    from feature_cols import resolve_feature_cols
    import cv_core

    args = _build_parser().parse_args(argv)
    models = [lgb.Booster(model_file=f"{args.model_prefix}.fold{k}.txt")
              for k in range(args.folds)]

    clean = pd.read_csv(args.clean_csv)
    feature_cols = resolve_feature_cols(None, [args.clean_csv], "label")
    clean_folds = [b.predict(clean[feature_cols].values) for b in models]
    clean_ens = cv_core.average_proba(clean_folds)
    thr = cv_core.working_points(
        clean["label"].values, clean_ens)["neg_recall_95"]["threshold"]

    pos50 = pd.read_csv(args.pos50_csv)
    pos50_folds = [b.predict(pos50[feature_cols].values) for b in models]
    stats = bin_recall(pos50, pos50_folds, thr)

    os.makedirs(os.path.dirname(args.out_csv) or ".", exist_ok=True)
    rows = [{"bin": f"({lo},{hi}]", "n": s["n"], "ens_recall": s["ens_recall"],
             "fold_mean": s["fold_mean"], "fold_std": s["fold_std"]}
            for (lo, hi), s in stats.items()]
    pd.DataFrame(rows).to_csv(args.out_csv, index=False)

    centers = [hi for _, hi in FDR_BINS]
    ens = [stats[b]["ens_recall"] for b in FDR_BINS]
    fmean = [stats[b]["fold_mean"] for b in FDR_BINS]
    fstd = [stats[b]["fold_std"] for b in FDR_BINS]
    plt.figure(figsize=(8, 6))
    plt.plot(centers, ens, "o-", label="ensemble recall")
    plt.errorbar(centers, fmean, yerr=fstd, fmt="s--", capsize=3,
                 label="fold mean +/- std")
    plt.xlabel("q_value bin upper bound")
    plt.ylabel("recall @ thr")
    plt.title("FDR-bin recall decay (2Da pos50)")
    plt.legend()
    plt.tight_layout()
    plt.savefig(args.out_png, dpi=150)
    plt.close()
    return stats


if __name__ == "__main__":
    main()
