"""Re-score trained LightGBM models at multiple thresholds.

See docs/specs/2026-06-03-rescore-tool-design.md.
"""
from __future__ import annotations

import argparse
import csv
import glob
import logging
import os
import sys
from pathlib import Path

import numpy as np
import pandas as pd
from sklearn.metrics import average_precision_score, roc_auc_score

_PROJECT_ROOT = Path(__file__).resolve().parents[2]
_SPEC_TRAINER_SRC = _PROJECT_ROOT / "tools" / "spec_trainer" / "src"
if str(_SPEC_TRAINER_SRC) not in sys.path:
    sys.path.insert(0, str(_SPEC_TRAINER_SRC))

from cv_core import as_error_detection, evaluate_at_threshold

logger = logging.getLogger(__name__)


def compute_metrics(y_true: np.ndarray, y_proba: np.ndarray,
                    threshold: float) -> dict:
    """Metrics at an error-score threshold; actual error is positive."""
    error_truth, error_score = as_error_detection(y_true, y_proba)
    metrics = evaluate_at_threshold(y_true, y_proba, threshold)
    has_both = len(np.unique(error_truth)) == 2
    metrics.update({
        "roc_auc": float(roc_auc_score(error_truth, error_score))
        if has_both else float("nan"),
        "error_pr_auc": float(
            average_precision_score(error_truth, error_score))
        if has_both else float("nan"),
    })
    return metrics


def _threshold_arg(s: str) -> float:
    """Argparse validator: 0 < t < 1."""
    try:
        t = float(s)
    except ValueError:
        raise argparse.ArgumentTypeError(
            "threshold must be a float, got {!r}".format(s))
    if not (0.0 < t < 1.0):
        raise argparse.ArgumentTypeError(
            "threshold must be in (0, 1) exclusive, got {}".format(t))
    return t


def _build_parser() -> argparse.ArgumentParser:
    p = argparse.ArgumentParser(
        description="Re-evaluate trained LightGBM models at multiple thresholds.")
    p.add_argument(
        "--thresholds", type=_threshold_arg, nargs="+", required=True,
        help=("Error-score thresholds (e.g. 0.1 0.2 0.5); "
              "error_score = 1 - model trust score."))
    p.add_argument(
        "--models", nargs="+", default=None,
        help="Model basenames to include. Default: all .txt files.")
    p.add_argument(
        "--output", type=str, default="runs/spec_trainer/rescore_summary.csv",
        help="Output CSV path.")
    p.add_argument(
        "--models-dir", type=str, default="runs/spec_trainer/models",
        help="Directory containing .txt model files.")
    p.add_argument(
        "--features-root", type=str, default="runs/baseline_{ds}_{fdr}",
        help="features.csv parent dir template.")
    return p


def infer_data_source(model_basename: str,
                       features_root_template: str) -> tuple[Path, str]:
    """Map model basename to (csv_path, mode)."""
    if model_basename.startswith("in_"):
        parts = model_basename.split("_")
        if len(parts) != 3:
            raise ValueError(
                "in_* model name must have 3 underscore parts, got {!r}".format(
                    model_basename))
        _, ds, fdr = parts
        csv = features_root_template.format(ds=ds, fdr=fdr) + "/features.csv"
        return Path(csv), "in_sample"

    if model_basename.startswith("cross_test_"):
        parts = model_basename.split("_")
        if len(parts) != 4:
            raise ValueError(
                "cross_test_* model name must have 4 underscore parts, got {!r}".format(
                    model_basename))
        _, _, held, fdr = parts
        csv = features_root_template.format(ds=held, fdr=fdr) + "/features.csv"
        return Path(csv), "cross_test"

    raise ValueError(
        "Model name must start with 'in_' or 'cross_test_'; got {!r}".format(
            model_basename))


def discover_models(models_dir: Path,
                     filter_names) -> list:
    """List .txt files under models_dir."""
    all_paths = sorted(models_dir.glob("*.txt"))
    if filter_names is None:
        return all_paths
    wanted = set(filter_names)
    found = [p for p in all_paths if p.stem in wanted]
    missing = wanted - {p.stem for p in found}
    if missing:
        logger.warning(
            "Requested models not found in %s: %s", models_dir, sorted(missing))
    return found


def score_model(model_path: Path, csv_path: Path, mode: str,
                target_col: str = "label"):
    """Load model, load CSV, return (y_true, y_proba)."""
    import lightgbm as lgb
    from feature_cols import resolve_feature_cols
    from sklearn.model_selection import train_test_split

    booster = lgb.Booster(model_file=str(model_path))
    df = pd.read_csv(csv_path)
    feature_cols = resolve_feature_cols(
        explicit=None,
        sample_csv_paths=[str(csv_path)],
        target_col=target_col,
    )

    if mode == "in_sample":
        y_full = df[target_col].astype(int)
        X_full = df[feature_cols]
        _, X_te, _, y_te = train_test_split(
            X_full, y_full, test_size=0.2, random_state=42,
            stratify=y_full)
    elif mode == "cross_test":
        X_te = df[feature_cols]
        y_te = df[target_col].astype(int)
    else:
        raise ValueError("Unknown mode: {!r}".format(mode))

    y_proba = booster.predict(X_te.values)
    return y_te.values, y_proba


def _print_console_table(rows) -> None:
    """Render rows as a rich.Table."""
    try:
        from rich.console import Console
        from rich.table import Table
    except ImportError:
        for r in rows:
            print(r)
        return

    table = Table(title="Rescore Summary", show_lines=False)
    cols = ["experiment", "error_thr", "n_error", "n_correct",
            "TP", "FP", "FN", "TN", "FPR", "FNR",
            "ErrorRec", "CorrectRec", "ErrorPrec", "ROC-AUC", "ErrorPR"]
    for c in cols:
        table.add_column(c, justify="right" if c != "experiment" else "left")

    last_exp = None
    for r in rows:
        exp = r["experiment"]
        if last_exp is not None and exp != last_exp:
            table.add_section()
        last_exp = exp
        auc_str = ("{:.4f}".format(r["roc_auc"])
                   if not np.isnan(r["roc_auc"]) else "n/a")
        table.add_row(
            exp,
            "{:.3f}".format(r['threshold']),
            str(r["n_actual_error"]), str(r["n_actual_correct"]),
            str(r["tp"]), str(r["fp"]), str(r["fn"]), str(r["tn"]),
            "{:.4f}".format(r['fpr']),
            "{:.4f}".format(r['fnr']),
            "{:.4f}".format(r['error_recall']),
            "{:.4f}".format(r['correct_recall']),
            "{:.4f}".format(r['error_precision']),
            auc_str,
            "{:.4f}".format(r['error_pr_auc']),
        )
    Console().print(table)


def _write_csv(rows, output_path: Path) -> None:
    output_path.parent.mkdir(parents=True, exist_ok=True)
    fieldnames = [
        "experiment", "metric_semantics", "positive_class", "threshold",
        "n_actual_error", "n_actual_correct", "tp", "fp", "fn", "tn",
        "fpr", "fnr", "error_recall", "correct_recall",
        "error_precision", "roc_auc", "error_pr_auc",
    ]
    with open(output_path, "w", newline="") as f:
        w = csv.DictWriter(f, fieldnames=fieldnames)
        w.writeheader()
        for r in rows:
            w.writerow({k: r[k] for k in fieldnames})


def main(argv=None) -> int:
    parser = _build_parser()
    args = parser.parse_args(argv)

    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s | %(levelname)s | %(message)s")

    models_dir = Path(args.models_dir)
    if not models_dir.exists():
        logger.error("Models dir not found: %s", models_dir)
        return 1

    model_paths = discover_models(models_dir, args.models)
    if not model_paths:
        logger.error("No models matched %r in %s", args.models, models_dir)
        return 1

    logger.info("Found %d models", len(model_paths))

    rows = []
    for mp in model_paths:
        try:
            csv_path, mode = infer_data_source(mp.stem, args.features_root)
        except ValueError as e:
            logger.warning("Skipping %s: %s", mp.name, e)
            continue
        if not csv_path.exists():
            logger.warning("Skipping %s: features.csv not found at %s",
                           mp.name, csv_path)
            continue
        logger.info("Scoring %s (mode=%s, data=%s)", mp.stem, mode, csv_path)
        try:
            y_true, y_proba = score_model(mp, csv_path, mode)
        except Exception as e:
            logger.warning("Skipping %s: scoring failed: %s", mp.name, e)
            continue

        for t in args.thresholds:
            m = compute_metrics(y_true, y_proba, t)
            rows.append({"experiment": mp.stem, "threshold": t, **m})

    if not rows:
        logger.error("No rows produced; aborting.")
        return 1

    output_path = Path(args.output)
    _write_csv(rows, output_path)
    logger.info("Wrote %d rows to %s", len(rows), output_path)
    _print_console_table(rows)
    return 0


if __name__ == "__main__":
    sys.exit(main())
