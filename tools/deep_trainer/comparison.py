"""Paired comparison and missingness audits for Phase-1 models."""

from __future__ import annotations

from pathlib import Path

import numpy as np
import pandas as pd


def load_frozen_lightgbm_predictions(
    protocol_root,
    test_frame,
    sample_id_col,
    model_names,
):
    """Load baseline scores only after exact fixed-test identity validation."""
    path = Path(protocol_root) / "predictions" / "fixed_test_predictions.csv"
    if not path.is_file():
        raise FileNotFoundError(
            f"frozen LightGBM predictions are missing: {path}")
    baseline = pd.read_csv(path, low_memory=False)
    if sample_id_col not in baseline:
        raise ValueError("frozen LightGBM predictions lack sample_id")
    if baseline[sample_id_col].duplicated().any():
        raise ValueError("frozen LightGBM predictions have duplicate sample_id")
    expected = test_frame[sample_id_col].astype(str)
    actual = baseline[sample_id_col].astype(str)
    if set(expected) != set(actual):
        raise ValueError(
            "frozen LightGBM predictions do not match the fixed test rows")
    baseline = baseline.assign(
        **{sample_id_col: actual}).set_index(sample_id_col).loc[expected]
    result = {}
    for model_name in model_names:
        prefix = model_name.upper()
        columns = {
            "trust_score": f"{prefix}_trust_score",
            "fpr_5_vote_fraction": (
                f"{prefix}_fpr5_error_vote_fraction"),
            "fpr_10_vote_fraction": (
                f"{prefix}_fpr10_error_vote_fraction"),
        }
        missing = [column for column in columns.values()
                   if column not in baseline]
        if missing:
            raise ValueError(
                f"frozen LightGBM {prefix} predictions lack {missing}")
        result[f"LightGBM_{prefix}"] = {
            key: baseline[column].to_numpy(dtype="f8")
            for key, column in columns.items()
        }
    return result


def _metrics(labels, values, evaluate_ranking, evaluate_at_threshold,
             weights=None):
    ranking = evaluate_ranking(
        labels, values["trust_score"], sample_weight=weights)
    fpr5 = evaluate_at_threshold(
        labels, 1.0 - values["fpr_5_vote_fraction"], 0.5,
        sample_weight=weights)
    fpr10 = evaluate_at_threshold(
        labels, 1.0 - values["fpr_10_vote_fraction"], 0.5,
        sample_weight=weights)
    return {
        "roc_auc": ranking["roc_auc"],
        "error_pr_auc": ranking["error_pr_auc"],
        "locked_fnr_at_fpr5": fpr5["fnr"],
        "locked_error_recall_at_fpr10": fpr10["error_recall"],
    }


def paired_cluster_bootstrap(
    test_frame,
    model_predictions,
    comparisons,
    *,
    reps,
    seed,
    group_col,
    target_col,
    evaluate_ranking,
    evaluate_at_threshold,
):
    """Compare models on identical rows via grouped paired bootstrap."""
    columns = [
        "model_a", "model_b", "metric", "delta_b_minus_a",
        "bootstrap_mean_delta", "ci95_low", "ci95_high",
        "probability_improved", "higher_is_better", "n_bootstrap",
        "resampling_unit",
    ]
    if reps < 1:
        return pd.DataFrame(columns=columns)
    labels = test_frame[target_col].to_numpy(dtype=int)
    codes, groups = pd.factorize(test_frame[group_col], sort=True)
    if len(groups) < 2:
        raise ValueError("paired bootstrap requires at least two groups")
    expected_rows = len(test_frame)
    for model, values in model_predictions.items():
        for key, array in values.items():
            if np.asarray(array).shape != (expected_rows,):
                raise ValueError(
                    f"{model}.{key} is not aligned to the fixed test")
    observed = {
        model: _metrics(
            labels, values, evaluate_ranking, evaluate_at_threshold)
        for model, values in model_predictions.items()
    }
    metric_names = tuple(next(iter(observed.values())))
    samples = {
        pair: {metric: [] for metric in metric_names}
        for pair in comparisons
    }
    rng = np.random.default_rng(seed)
    completed = 0
    for _ in range(reps):
        group_weights = rng.multinomial(
            len(groups), np.full(len(groups), 1.0 / len(groups)))
        weights = group_weights[codes].astype("f8")
        if not weights[labels == 0].sum() or not weights[labels == 1].sum():
            continue
        replicate = {
            model: _metrics(
                labels, values, evaluate_ranking, evaluate_at_threshold,
                weights=weights)
            for model, values in model_predictions.items()
        }
        for pair in comparisons:
            model_a, model_b = pair
            if model_a not in replicate or model_b not in replicate:
                raise ValueError(f"unknown paired comparison {pair}")
            for metric in metric_names:
                samples[pair][metric].append(
                    replicate[model_b][metric]
                    - replicate[model_a][metric])
        completed += 1

    rows = []
    for (model_a, model_b), metrics in samples.items():
        for metric, values in metrics.items():
            array = np.asarray(values, dtype="f8")
            if not len(array):
                raise ValueError("paired bootstrap completed zero valid reps")
            higher_is_better = metric != "locked_fnr_at_fpr5"
            delta = observed[model_b][metric] - observed[model_a][metric]
            rows.append({
                "model_a": model_a,
                "model_b": model_b,
                "metric": metric,
                "delta_b_minus_a": float(delta),
                "bootstrap_mean_delta": float(array.mean()),
                "ci95_low": float(np.quantile(array, 0.025)),
                "ci95_high": float(np.quantile(array, 0.975)),
                "probability_improved": float(
                    (array > 0).mean() if higher_is_better
                    else (array < 0).mean()),
                "higher_is_better": higher_is_better,
                "n_bootstrap": completed,
                "resampling_unit": group_col,
            })
    return pd.DataFrame(rows, columns=columns)


def missingness_audit(frame, feature_cols, *, target_col, split_name,
                      dataset_col=None):
    """Report missing-value shortcuts by label and acquisition domain."""
    keys = [target_col]
    if dataset_col and dataset_col in frame:
        keys.insert(0, dataset_col)
    rows = []
    grouped = frame.groupby(keys, dropna=False, sort=True)
    for group, subset in grouped:
        values = group if isinstance(group, tuple) else (group,)
        identity = dict(zip(keys, values))
        rates = subset[feature_cols].isna().mean()
        for feature, rate in rates.items():
            rows.append({
                "split": split_name,
                **identity,
                "feature": feature,
                "missing_fraction": float(rate),
                "n_rows": int(len(subset)),
            })
    return pd.DataFrame(rows)


def summarize_missingness(audit, *, target_col):
    if audit.empty:
        return {"max_label_missingness_gap": None, "features_gap_ge_0_05": 0}
    index = [column for column in ("split", "dataset", "feature")
             if column in audit]
    pivot = audit.pivot_table(
        index=index, columns=target_col, values="missing_fraction")
    if not {0, 1} <= set(pivot.columns):
        return {"max_label_missingness_gap": None, "features_gap_ge_0_05": 0}
    gap = (pivot[0] - pivot[1]).abs()
    return {
        "max_label_missingness_gap": float(gap.max()),
        "features_gap_ge_0_05": int((gap >= 0.05).sum()),
        "interpretation": (
            "large label-conditional gaps can become missingness shortcuts; "
            "compare the no-missing-indicator sensitivity run"),
    }
