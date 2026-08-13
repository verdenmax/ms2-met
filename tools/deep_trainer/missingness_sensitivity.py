"""Compare matched Phase-1 MLP runs with and without missing indicators."""

from __future__ import annotations

import argparse
import json
from pathlib import Path
import sys

import numpy as np
import pandas as pd

from .comparison import paired_cluster_bootstrap


_PROJECT_ROOT = Path(__file__).resolve().parents[2]
_SPEC_SRC = _PROJECT_ROOT / "tools" / "spec_trainer" / "src"
if str(_SPEC_SRC) not in sys.path:
    sys.path.insert(0, str(_SPEC_SRC))

from cv_core import evaluate_at_threshold, evaluate_ranking  # noqa: E402


_IDENTITY_COLUMNS = [
    "sample_id", "dataset", "sequence", "charge", "precursor_mz", "rt",
    "raw_title1", "label_type", "label", "negative_tier", "__source_row",
]


def _load_predictions(root):
    root = Path(root)
    status = json.loads((root / "bundle_status.json").read_text())
    if status.get("status") != "complete":
        raise ValueError(f"incomplete result bundle: {root}")
    summary = json.loads((root / "summary.json").read_text())
    predictions = pd.read_csv(
        root / "predictions" / "fixed_test_predictions.csv",
        low_memory=False,
    )
    missing = [column for column in _IDENTITY_COLUMNS
               if column not in predictions]
    if missing:
        raise ValueError(f"predictions lack identity columns: {missing}")
    if predictions["sample_id"].duplicated().any():
        raise ValueError(f"duplicate sample_id values in {root}")
    return predictions, summary


def _assert_matched(without, with_indicators, without_summary, with_summary):
    left = without[_IDENTITY_COLUMNS].sort_values(
        "sample_id").reset_index(drop=True)
    right = with_indicators[_IDENTITY_COLUMNS].sort_values(
        "sample_id").reset_index(drop=True)
    pd.testing.assert_frame_equal(
        left, right, check_dtype=False, check_exact=True)
    for key in ("python", "torch", "numpy", "pandas", "git_commit"):
        a = without_summary["provenance"].get(key)
        b = with_summary["provenance"].get(key)
        if a != b:
            raise ValueError(
                f"matched sensitivity runs differ in provenance {key}: "
                f"{a!r} != {b!r}")
    return left


def _values(frame, model):
    return {
        "trust_score": frame[f"{model}_trust_score"].to_numpy(dtype="f8"),
        "fpr_5_vote_fraction": frame[
            f"{model}_fpr_5_error_vote_fraction"].to_numpy(dtype="f8"),
        "fpr_10_vote_fraction": frame[
            f"{model}_fpr_10_error_vote_fraction"].to_numpy(dtype="f8"),
    }


def _ensemble_values(frame, seeds):
    values = [_values(frame, f"MLP_M20_seed{seed}") for seed in seeds]
    return {
        key: np.mean([value[key] for value in values], axis=0)
        for key in values[0]
    }


def _metric_row(labels, values):
    ranking = evaluate_ranking(labels, values["trust_score"])
    fpr5 = evaluate_at_threshold(
        labels, 1.0 - values["fpr_5_vote_fraction"], 0.5)
    fpr10 = evaluate_at_threshold(
        labels, 1.0 - values["fpr_10_vote_fraction"], 0.5)
    return {
        "metric_semantics": "error_identification_positive_v1",
        "positive_class": "incorrect_identification",
        "roc_auc": ranking["roc_auc"],
        "error_pr_auc": ranking["error_pr_auc"],
        "locked_fnr_at_fpr5": fpr5["fnr"],
        "observed_fpr_at_fpr5": fpr5["fpr"],
        "locked_error_recall_at_fpr10": fpr10["error_recall"],
        "observed_fpr_at_fpr10": fpr10["fpr"],
    }


def compare(without_root, with_root, output_dir, *, seeds, reps, seed):
    without, without_summary = _load_predictions(without_root)
    with_indicators, with_summary = _load_predictions(with_root)
    identity = _assert_matched(
        without, with_indicators, without_summary, with_summary)
    without = without.set_index("sample_id").loc[
        identity["sample_id"]].reset_index()
    with_indicators = with_indicators.set_index("sample_id").loc[
        identity["sample_id"]].reset_index()

    model_predictions = {}
    comparisons = []
    for training_seed in seeds:
        for arm, frame in (
            ("without_indicators", without),
            ("with_indicators", with_indicators),
        ):
            model_predictions[f"{arm}_seed{training_seed}"] = _values(
                frame, f"MLP_M20_seed{training_seed}")
        comparisons.append((
            f"without_indicators_seed{training_seed}",
            f"with_indicators_seed{training_seed}",
        ))
    for arm, frame in (
        ("without_indicators", without),
        ("with_indicators", with_indicators),
    ):
        model_predictions[f"{arm}_ensemble{len(seeds) * 5}"] = (
            _ensemble_values(frame, seeds))
    comparisons.append((
        f"without_indicators_ensemble{len(seeds) * 5}",
        f"with_indicators_ensemble{len(seeds) * 5}",
    ))

    rows = []
    labels = identity["label"].to_numpy(dtype=int)
    for model, values in model_predictions.items():
        rows.append({"model": model, **_metric_row(labels, values)})
    metrics = pd.DataFrame(rows)
    bootstrap = paired_cluster_bootstrap(
        identity, model_predictions, comparisons,
        reps=reps, seed=seed, group_col="sequence", target_col="label",
        evaluate_ranking=evaluate_ranking,
        evaluate_at_threshold=evaluate_at_threshold,
    )
    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    metrics.to_csv(output_dir / "fixed_test_summary.csv", index=False)
    bootstrap.to_csv(output_dir / "paired_bootstrap.csv", index=False)
    summary = {
        "metric_semantics": "error_identification_positive_v1",
        "positive_class": "incorrect_identification",
        "comparison": "with_indicators_minus_without_indicators",
        "n_fixed_test": int(len(identity)),
        "n_actual_correct": int((labels == 1).sum()),
        "n_actual_error": int((labels == 0).sum()),
        "n_unique_sequences": int(identity["sequence"].nunique()),
        "training_seeds": list(seeds),
        "ensemble_members": len(seeds) * 5,
        "bootstrap_reps": int(reps),
        "bootstrap_seed": int(seed),
        "matched_provenance": {
            key: with_summary["provenance"].get(key)
            for key in ("python", "torch", "numpy", "pandas", "git_commit")
        },
        "without_indicators_root": str(Path(without_root).resolve()),
        "with_indicators_root": str(Path(with_root).resolve()),
        "artifacts": {
            "fixed_test_summary": "fixed_test_summary.csv",
            "paired_bootstrap": "paired_bootstrap.csv",
        },
    }
    (output_dir / "summary.json").write_text(
        json.dumps(summary, ensure_ascii=False, indent=2) + "\n")
    return summary


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--without-indicators-root", required=True)
    parser.add_argument("--with-indicators-root", required=True)
    parser.add_argument("--output-dir", required=True)
    parser.add_argument("--seeds", nargs="+", type=int, default=[42, 43, 44])
    parser.add_argument("--bootstrap-reps", type=int, default=1000)
    parser.add_argument("--bootstrap-seed", type=int, default=20260812)
    args = parser.parse_args()
    compare(
        args.without_indicators_root,
        args.with_indicators_root,
        args.output_dir,
        seeds=tuple(args.seeds),
        reps=args.bootstrap_reps,
        seed=args.bootstrap_seed,
    )


if __name__ == "__main__":
    main()
