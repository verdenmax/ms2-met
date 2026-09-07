"""Controlled E5/E10/E20 negative-pool experiment on one fixed test set.

The public seam is ``run_fixed_negpool``: callers provide one formal neg20 CV
config plus a feature snapshot/dataset, and receive a complete paired result
bundle.  Internally the module validates nested pool identity and feature
equality, creates one sequence-held-out test manifest and one reusable CV map,
trains M5/M10/M20, evaluates every model on the same E20 test rows, and runs a
paired sequence-cluster bootstrap.

Stored labels and model scores retain the project convention (1/high means a
correct identification).  Every metric delegates to cv_core, where an
incorrect identification is the statistical positive class.
"""
from __future__ import annotations

import argparse
import copy
import datetime as dt
import hashlib
import json
import logging
import os
import platform
import shutil
import subprocess
import sys
import tempfile
import uuid
from dataclasses import dataclass
from pathlib import Path

import numpy as np
import pandas as pd
import yaml

from cohort import COHORT_DEFINITIONS, apply_training_cohort
from cv_core import (METRIC_SEMANTICS_VERSION, evaluate_at_threshold,
                     evaluate_ranking, make_cv_splits)
from cv_train import (_SOURCE_FILE, _SOURCE_ROW, _atomic_csv, _atomic_json,
                      _file_fingerprint, _inner_split, _op_key,
                      _operating_targets, _validate_frame, assemble_oof,
                      evaluate_cross_test)
from feature_cols import resolve_configured_feature_cols
from sample_groups import (
    RELATIONSHIP_COLUMNS,
    assign_leakage_groups as _assign_leakage_groups,
    prepare_cv_groups,
)
from sample_identity import (
    COMBINED_SAMPLE_ID_ALGORITHM,
    LOCAL_SAMPLE_ID_ALGORITHM,
    identity_candidates,
    local_sample_ids,
    namespace_sample_ids,
)


POOL_NAMES = ("neg05", "neg10", "neg20")
ERROR_TIERS = ("t5", "t5_10", "t10_20")
MODEL_TIERS = {
    "M5": ("t5",),
    "M10": ("t5", "t5_10"),
    "M20": ERROR_TIERS,
}
PAIR_COMPARISONS = (("M5", "M10"), ("M10", "M20"), ("M5", "M20"))
_SAMPLE_ID = "sample_id"
_TIER = "negative_tier"
_SPLIT = "fixed_split"
_OUTER_FOLD = "outer_fold"
_DATASET = "dataset"
_SOURCE_SAMPLE_ID = "source_sample_id"
_STRATUM = "dataset_tier_stratum"
_LEAKAGE_GROUP = "leakage_group_id"


@dataclass
class PreparedFixedNegpool:
    """Prepared master cohort and auditable, reusable split assignments."""

    frame: pd.DataFrame
    membership: pd.DataFrame
    group_fold_map: pd.DataFrame
    feature_cols: list[str]
    identity_cols: list[str]
    cohort_audit: dict
    validation: dict
    split_group_col: str


def feature_paths(feature_root, dataset):
    """Return the three required feature CSVs for one dataset."""
    root = Path(feature_root)
    return {
        pool: root / f"baseline_{dataset}_{pool}" / "features.csv"
        for pool in POOL_NAMES
    }


def _load_identity_tables(paths, target_col):
    headers = {
        pool: list(pd.read_csv(path, nrows=0).columns)
        for pool, path in paths.items()
    }
    common = set(headers[POOL_NAMES[0]])
    for pool in POOL_NAMES[1:]:
        common.intersection_update(headers[pool])
    if target_col not in common:
        raise ValueError(f"target column {target_col!r} is not common to pools")

    candidates = identity_candidates(common)
    if not candidates:
        raise ValueError(
            "cannot construct a stable sample identity: query_id/parent_id "
            "and the required sequence composite are unavailable")
    read_columns = sorted(
        set().union(*map(set, candidates), {target_col}))
    tables = {
        pool: pd.read_csv(
            path, usecols=read_columns, dtype=str, keep_default_na=False)
        for pool, path in paths.items()
    }

    identity_cols = None
    diagnostics = []
    for candidate in candidates:
        bad = {}
        for pool, frame in tables.items():
            missing_rows = int(frame[candidate].eq("").any(axis=1).sum())
            duplicate_rows = int(
                frame.duplicated(candidate, keep=False).sum())
            if missing_rows or duplicate_rows:
                bad[pool] = {
                    "rows_with_empty_identity_field": missing_rows,
                    "duplicate_identity_rows": duplicate_rows,
                }
        diagnostics.append({"columns": candidate, "failures": bad})
        if not bad:
            identity_cols = candidate
            break
    if identity_cols is None:
        raise ValueError(
            "no candidate identity is complete and unique in all pools: "
            f"{diagnostics}")

    for pool, frame in tables.items():
        frame[_SAMPLE_ID] = local_sample_ids(frame, identity_cols)
        frame[_SOURCE_ROW] = np.arange(len(frame), dtype=np.int64)
        if frame[_SAMPLE_ID].duplicated().any():
            raise ValueError(
                f"SHA-256 identity collision or duplicate in {pool}")
        labels = pd.to_numeric(frame[target_col], errors="coerce")
        if labels.isna().any() or not set(labels.unique()).issubset({0, 1}):
            raise ValueError(f"{pool} has malformed {target_col} values")
        frame[target_col] = labels.astype(int)
    return tables, identity_cols, headers, diagnostics


def _validate_nested_membership(tables, target_col):
    sets = {
        pool: set(frame[_SAMPLE_ID]) for pool, frame in tables.items()
    }
    if not sets["neg05"] <= sets["neg10"] <= sets["neg20"]:
        raise ValueError(
            "negative-pool files are not nested: require E5 subset E10 "
            "subset E20")

    labels = {
        pool: frame.set_index(_SAMPLE_ID)[target_col]
        for pool, frame in tables.items()
    }
    for pool in ("neg05", "neg10"):
        shared = labels[pool].index
        mismatch = int((labels[pool] != labels["neg20"].loc[shared]).sum())
        if mismatch:
            raise ValueError(
                f"{pool}/neg20 disagree on {mismatch} shared labels")

    correct_sets = {
        pool: set(frame.loc[frame[target_col].eq(1), _SAMPLE_ID])
        for pool, frame in tables.items()
    }
    if not (correct_sets["neg05"] == correct_sets["neg10"]
            == correct_sets["neg20"]):
        raise ValueError(
            "correct-identification rows differ across negative pools")
    error_sets = {
        pool: set(frame.loc[frame[target_col].eq(0), _SAMPLE_ID])
        for pool, frame in tables.items()
    }
    if not error_sets["neg05"] <= error_sets["neg10"] <= error_sets["neg20"]:
        raise ValueError(
            "error rows are not nested: require E5 subset E10 subset E20")

    return {
        "n_rows": {pool: len(frame) for pool, frame in tables.items()},
        "n_correct": {
            pool: int(frame[target_col].eq(1).sum())
            for pool, frame in tables.items()
        },
        "n_error": {
            pool: int(frame[target_col].eq(0).sum())
            for pool, frame in tables.items()
        },
        "nested_all_rows": True,
        "nested_error_rows": True,
        "identical_correct_rows": True,
    }


def _row_hashes(path, columns):
    values = pd.read_csv(
        path, usecols=columns, dtype=str, keep_default_na=False)
    return pd.util.hash_pandas_object(
        values[columns], index=False, hash_key="negpool-check-v1")


def _validate_shared_values(paths, tables, columns):
    """Ensure shared rows have identical formal features and cohort flags."""
    master_hash = pd.Series(
        _row_hashes(paths["neg20"], columns).to_numpy(),
        index=tables["neg20"][_SAMPLE_ID])
    mismatch_counts = {}
    for pool in ("neg05", "neg10"):
        hashes = pd.Series(
            _row_hashes(paths[pool], columns).to_numpy(),
            index=tables[pool][_SAMPLE_ID])
        mismatch = hashes != master_hash.loc[hashes.index]
        mismatch_counts[pool] = int(mismatch.sum())
        if mismatch.any():
            examples = hashes.index[mismatch][:5].tolist()
            raise ValueError(
                f"{pool}/neg20 shared formal feature values differ on "
                f"{int(mismatch.sum())} rows; sample_ids={examples}")
    return {
        "columns_compared": list(columns),
        "n_columns_compared": len(columns),
        "hash_method": "pandas_hash_pandas_object_v1_over_raw_csv_strings",
        "mismatch_counts": mismatch_counts,
    }


def _assign_tiers(master_identity, tables, target_col):
    ids05 = set(tables["neg05"][_SAMPLE_ID])
    ids10 = set(tables["neg10"][_SAMPLE_ID])
    tiers = np.full(len(master_identity), "correct", dtype=object)
    is_error = master_identity[target_col].eq(0).to_numpy()
    sample_ids = master_identity[_SAMPLE_ID]
    tiers[is_error & sample_ids.isin(ids05).to_numpy()] = "t5"
    tiers[is_error & ~sample_ids.isin(ids05).to_numpy()
          & sample_ids.isin(ids10).to_numpy()] = "t5_10"
    tiers[is_error & ~sample_ids.isin(ids10).to_numpy()] = "t10_20"
    return tiers


def _choose_fixed_test(frame, group_col, test_fraction, seed,
                       n_candidates=128, stratum_col=_TIER,
                       min_test_by_stratum=None):
    """Choose the best deterministic grouped split across declared strata."""
    from sklearn.model_selection import GroupShuffleSplit

    if not 0.0 < test_fraction < 1.0:
        raise ValueError("test_fraction must be between 0 and 1")
    if n_candidates < 1:
        raise ValueError("split_candidates must be >= 1")
    if stratum_col not in frame:
        raise ValueError(f"split stratum column {stratum_col!r} is missing")
    strata = frame[stratum_col].astype(str).to_numpy()
    required = set(strata)
    if len(required) < 2:
        raise ValueError(
            f"fixed test requires at least two strata, got {sorted(required)}")
    totals = pd.Series(strata).value_counts()
    minima = dict(min_test_by_stratum or {})
    unknown_minima = set(minima) - required
    if unknown_minima:
        raise ValueError(
            f"minimum test counts name unknown strata: "
            f"{sorted(unknown_minima)}")
    splitter = GroupShuffleSplit(
        n_splits=n_candidates, test_size=test_fraction, random_state=seed)
    best = None
    for candidate, (train_idx, test_idx) in enumerate(splitter.split(
            np.zeros(len(frame)), strata, frame[group_col])):
        test_counts = pd.Series(strata[test_idx]).value_counts()
        train_counts = pd.Series(strata[train_idx]).value_counts()
        if any(test_counts.get(name, 0) == 0
               or train_counts.get(name, 0) == 0 for name in required):
            continue
        if any(test_counts.get(name, 0) < minimum
               for name, minimum in minima.items()):
            continue
        fractions = test_counts.reindex(totals.index, fill_value=0) / totals
        max_error = float((fractions - test_fraction).abs().max())
        row_error = abs(len(test_idx) / len(frame) - test_fraction)
        score = (max_error, row_error, candidate)
        if best is None or score < best[0]:
            best = (score, train_idx, test_idx, fractions)
    if best is None:
        raise ValueError(
            "cannot construct a grouped fixed test containing every tier")
    _, train_idx, test_idx, fractions = best
    split = np.full(len(frame), "train", dtype=object)
    split[test_idx] = "test"
    audit = {
        "method": "best_of_deterministic_group_shuffle_candidates",
        "group_col": group_col,
        "stratum_col": stratum_col,
        "seed": seed,
        "test_fraction_target": test_fraction,
        "n_candidates": n_candidates,
        "minimum_test_rows_by_stratum": {
            name: int(value) for name, value in minima.items()
        },
        "n_train": int(len(train_idx)),
        "n_test": int(len(test_idx)),
        "test_fraction_observed": float(len(test_idx) / len(frame)),
        "test_fraction_by_stratum": {
            name: float(fractions[name]) for name in sorted(fractions.index)
        },
    }
    return split, audit


def _assign_reusable_folds(frame, cfg, group_col, stratum_col=_TIER):
    """Create outer and inner group assignments once on the E20 train set."""
    training = cfg["training"]
    n_folds = int(training.get("cv_folds", 5))
    seed = int(training.get("cv_seed", 42))
    valid_size = float(training.get("valid_size", 0.15))
    train = frame.loc[frame[_SPLIT].eq("train")].copy().reset_index()
    categories = sorted(train[stratum_col].astype(str).unique().tolist())
    stratum_codes = pd.Categorical(
        train[stratum_col].astype(str), categories=categories).codes
    splits = make_cv_splits(
        stratum_codes, train[group_col].to_numpy(), n_folds=n_folds,
        seed=seed)
    outer = np.full(len(train), -1, dtype=int)
    for fold, (_, test_idx) in enumerate(splits):
        outer[test_idx] = fold
    if (outer < 0).any():
        raise AssertionError("reusable outer fold map left rows unassigned")
    train[_OUTER_FOLD] = outer

    dummy = pd.DataFrame({"unused": np.zeros(len(train))})
    y = train[cfg["data"]["target_col"]]
    groups = train[group_col]
    inner_columns = []
    for fold in range(n_folds):
        tr_idx = np.flatnonzero(outer != fold)
        te_idx = np.flatnonzero(outer == fold)
        _, val_idx = _inner_split(
            dummy, y, groups, tr_idx, valid_size, seed + fold)
        mask = np.zeros(len(train), dtype=bool)
        mask[val_idx] = True
        if mask[te_idx].any():
            raise AssertionError("fixed inner validation overlaps outer fold")
        column = f"inner_valid_for_fold_{fold}"
        train[column] = mask
        inner_columns.append(column)

    frame[_OUTER_FOLD] = -1
    for column in inner_columns:
        frame[column] = False
    frame.loc[train["index"], _OUTER_FOLD] = train[_OUTER_FOLD].to_numpy()
    for column in inner_columns:
        frame.loc[train["index"], column] = train[column].to_numpy()
    frame[_OUTER_FOLD] = frame[_OUTER_FOLD].astype(int)

    fold_counts = (
        train.groupby([_OUTER_FOLD, stratum_col], sort=True).size()
        .unstack(fill_value=0).to_dict(orient="index")
    )
    return inner_columns, {
        "method": "stratified_group_kfold_on_E20_training_superset",
        "stratification": stratum_col,
        "n_folds": n_folds,
        "seed": seed,
        "outer_fold_stratum_counts": {
            str(fold): {key: int(value) for key, value in counts.items()}
            for fold, counts in fold_counts.items()
        },
        "inner_validation": (
            "created once per outer fold on the E20 training superset and "
            "reused by M5/M10/M20"),
    }


def _group_fold_manifest(frame, group_col, inner_columns):
    columns = [group_col, _SPLIT, _OUTER_FOLD, *inner_columns]
    grouped = frame[columns].groupby(group_col, sort=True, dropna=False)
    nunique = grouped.nunique(dropna=False)
    if (nunique > 1).any(axis=None):
        bad = nunique.index[(nunique > 1).any(axis=1)][:5].tolist()
        raise ValueError(
            f"group split/fold assignment is inconsistent for {bad}")
    return grouped.first().reset_index()


def prepare_fixed_negpool(paths, cfg, *, test_fraction=0.20,
                          min_test_errors_per_tier=100,
                          split_candidates=128,
                          generate_assignments=True):
    """Validate pools and return the single master cohort/split/fold map."""
    target_col = cfg["data"]["target_col"]
    group_col = cfg["data"].get("group_col")
    if not group_col:
        raise ValueError("fixed-negpool requires data.group_col")
    for pool, path in paths.items():
        if not Path(path).is_file():
            raise FileNotFoundError(f"missing {pool} feature file: {path}")

    logging.info("reading identities and validating E5/E10/E20 nesting")
    tables, identity_cols, headers, identity_diagnostics = (
        _load_identity_tables(paths, target_col))
    nesting = _validate_nested_membership(tables, target_col)
    feature_cols = resolve_configured_feature_cols(
        cfg["data"], [str(paths["neg20"])], target_col)
    cohort_name = cfg["data"].get("cohort") or "none"
    cohort_columns = [
        column for column, _ in COHORT_DEFINITIONS[cohort_name]
    ]
    logging.info(
        "comparing %d formal feature/cohort columns on every shared row",
        len(feature_cols) + len(cohort_columns))
    shared_values = _validate_shared_values(
        paths, tables, list(dict.fromkeys(feature_cols + cohort_columns)))

    logging.info("loading neg20 as the single master feature table")
    master = pd.read_csv(paths["neg20"])
    master_identity = tables["neg20"]
    if len(master) != len(master_identity):
        raise ValueError("neg20 identity/full reads disagree on row count")
    labels = pd.to_numeric(master[target_col], errors="coerce")
    if not np.array_equal(labels.to_numpy(), master_identity[target_col]):
        raise ValueError("neg20 identity/full reads disagree on labels/order")
    master = pd.concat([
        master,
        pd.DataFrame({
            _SAMPLE_ID: master_identity[_SAMPLE_ID].to_numpy(),
            _TIER: _assign_tiers(master_identity, tables, target_col),
            _SOURCE_FILE: str(Path(paths["neg20"]).resolve()),
            _SOURCE_ROW: np.arange(len(master), dtype=np.int64),
        }),
    ], axis=1).copy()
    split_group_col, grouping_audit = prepare_cv_groups(
        master, group_col)
    cohort, cohort_audit = apply_training_cohort(
        master, cohort_name, target_col=target_col)
    # The wide feature table may arrive as many pandas blocks; consolidate it
    # before adding split/fold manifest columns to avoid fragmented writes.
    cohort = cohort.copy()
    _validate_frame(cohort, feature_cols, target_col, group_col)

    if generate_assignments:
        seed = int(cfg["training"].get("cv_seed", 42))
        split, split_audit = _choose_fixed_test(
            cohort, split_group_col, test_fraction, seed,
            n_candidates=split_candidates)
        cohort[_SPLIT] = split
        test_counts = (
            cohort.loc[cohort[_SPLIT].eq("test")]
            .groupby(_TIER, sort=True).size().to_dict()
        )
        insufficient = {
            tier: int(test_counts.get(tier, 0)) for tier in ERROR_TIERS
            if test_counts.get(tier, 0) < min_test_errors_per_tier
        }
        if insufficient:
            raise ValueError(
                "fixed test has too few error rows in tiers "
                f"{insufficient}; require >= {min_test_errors_per_tier}. "
                "Increase the holdout fraction or negative-pool cutoff "
                "before training")
        inner_columns, fold_audit = _assign_reusable_folds(
            cohort, cfg, split_group_col)
        group_map = _group_fold_manifest(
            cohort, split_group_col, inner_columns)
    else:
        # Validation-only callers reconstruct identities and the leakage-group
        # key, then obtain every assignment from an already frozen manifest.
        # No random split/fold generator is invoked in this mode.
        cohort[_SPLIT] = "deferred_to_frozen_manifest"
        cohort[_OUTER_FOLD] = -1
        test_counts = {}
        split_audit = {
            "method": "not_generated_consume_frozen_manifest",
            "assignment_owner": "completed_fixed_negpool_bundle",
        }
        fold_audit = dict(split_audit)
        group_map = cohort[[split_group_col]].drop_duplicates().reset_index(
            drop=True)

    in_cohort = set(cohort[_SAMPLE_ID])
    membership = master_identity[
        [_SAMPLE_ID, *identity_cols, target_col, _SOURCE_ROW]].copy()
    for column in RELATIONSHIP_COLUMNS:
        if column in master and column not in membership:
            membership[column] = master[column].to_numpy()
    membership[_TIER] = master[_TIER].to_numpy()
    membership["in_cohort"] = membership[_SAMPLE_ID].isin(in_cohort)
    split_map = cohort.set_index(_SAMPLE_ID)[_SPLIT]
    membership[_SPLIT] = membership[_SAMPLE_ID].map(split_map).fillna(
        "excluded_by_cohort")

    validation = {
        "protocol_schema": "fixed_negpool_protocol_v2",
        "metric_semantics": METRIC_SEMANTICS_VERSION,
        "positive_class": "incorrect_identification",
        "identity": {
            "columns": identity_cols,
            "serialization": "length_prefixed_utf8_fields_v1",
            "sample_id": "sha256",
            "candidate_diagnostics": identity_diagnostics,
            "split_grouping": grouping_audit,
        },
        "headers_identical": all(
            headers[pool] == headers["neg20"] for pool in POOL_NAMES),
        "nesting": nesting,
        "shared_values": shared_values,
        "feature_schema": {
            "ordered_feature_cols": list(feature_cols),
            "sha256": _feature_schema_sha256(feature_cols),
        },
        "cohort": cohort_audit,
        "fixed_split": split_audit,
        "fixed_folds": fold_audit,
        "fixed_test_tier_counts": {
            key: int(value) for key, value in test_counts.items()
        },
        "min_test_errors_per_tier": min_test_errors_per_tier,
    }
    logging.info(
        "preflight passed: cohort=%s, fixed test tiers=%s",
        cohort_audit["after"], validation["fixed_test_tier_counts"])
    return PreparedFixedNegpool(
        frame=cohort, membership=membership,
        group_fold_map=group_map, feature_cols=feature_cols,
        identity_cols=identity_cols, cohort_audit=cohort_audit,
        validation=validation, split_group_col=split_group_col)


def _sum_class_counts(audits, side):
    keys = ("n_rows", "n_correct", "n_error")
    return {
        key: int(sum(audit[side][key] for audit in audits.values()))
        for key in keys
    }


def prepare_combined_fixed_negpool(feature_root, cfg, *,
                                   test_fraction=0.20,
                                   min_test_errors_per_tier=100,
                                   split_candidates=128,
                                   generate_assignments=True):
    """Prepare one globally grouped 2da+5da+normal fixed E20 experiment.

    Each dataset is independently audited for E5/E10/E20 nesting and shared
    feature equality.  Their neg20 master cohorts are then concatenated and
    split exactly once by global sequence, balancing the twelve
    ``dataset x tier`` strata.  Per-dataset split/fold assignments created by
    the reused preparer are deliberately discarded before the global map is
    made.
    """
    datasets = ("2da", "5da", "normal")
    group_col = cfg["data"].get("group_col")
    target_col = cfg["data"]["target_col"]
    if group_col != "sequence":
        raise ValueError("combined fixed-negpool requires group_col=sequence")

    prepared_by_dataset = {}
    for dataset in datasets:
        logging.info("preparing nested pools for combined dataset=%s", dataset)
        prepared_by_dataset[dataset] = prepare_fixed_negpool(
            feature_paths(feature_root, dataset), cfg,
            test_fraction=test_fraction, min_test_errors_per_tier=1,
            split_candidates=split_candidates,
            generate_assignments=generate_assignments)

    first = prepared_by_dataset[datasets[0]]
    for dataset, prepared in prepared_by_dataset.items():
        if prepared.feature_cols != first.feature_cols:
            raise ValueError(
                f"combined feature schema differs for dataset {dataset}")
        if prepared.identity_cols != first.identity_cols:
            raise ValueError(
                f"combined identity schema differs for dataset {dataset}")

    frames = []
    memberships = []
    for dataset, prepared in prepared_by_dataset.items():
        generated = [
            column for column in prepared.frame
            if column in {_SPLIT, _OUTER_FOLD}
            or column.startswith("inner_valid_for_fold_")
        ]
        frame = namespace_sample_ids(
            prepared.frame.drop(columns=generated), dataset)
        membership = namespace_sample_ids(
            prepared.membership.drop(columns=[_SPLIT]), dataset)
        frames.append(frame)
        memberships.append(membership)

    cohort = pd.concat(frames, ignore_index=True).copy()
    membership = pd.concat(memberships, ignore_index=True)
    if cohort[_SAMPLE_ID].duplicated().any():
        raise ValueError("combined namespaced sample IDs are not unique")
    cohort[_STRATUM] = (
        cohort[_DATASET].astype(str) + "::" + cohort[_TIER].astype(str))

    split_group_col, grouping_audit = _assign_leakage_groups(
        cohort, group_col)

    if generate_assignments:
        seed = int(cfg["training"].get("cv_seed", 42))
        split, split_audit = _choose_fixed_test(
            cohort, split_group_col, test_fraction, seed,
            n_candidates=split_candidates, stratum_col=_STRATUM,
            min_test_by_stratum={
                f"{dataset}::{tier}": min_test_errors_per_tier
                for dataset in datasets for tier in ERROR_TIERS
            })
        cohort[_SPLIT] = split
        test = cohort.loc[cohort[_SPLIT].eq("test")]
        dataset_tier_counts = test.groupby(
            [_DATASET, _TIER], sort=True).size().to_dict()
        insufficient = {
            f"{dataset}::{tier}": int(
                dataset_tier_counts.get((dataset, tier), 0))
            for dataset in datasets for tier in ERROR_TIERS
            if dataset_tier_counts.get((dataset, tier), 0)
            < min_test_errors_per_tier
        }
        if insufficient:
            raise ValueError(
                "combined fixed test has too few error rows in dataset/tier "
                f"strata {insufficient}; require >= "
                f"{min_test_errors_per_tier}")
        inner_columns, fold_audit = _assign_reusable_folds(
            cohort, cfg, split_group_col, stratum_col=_STRATUM)
        group_map = _group_fold_manifest(
            cohort, split_group_col, inner_columns)
    else:
        cohort[_SPLIT] = "deferred_to_frozen_manifest"
        cohort[_OUTER_FOLD] = -1
        dataset_tier_counts = {}
        split_audit = {
            "method": "not_generated_consume_frozen_manifest",
            "assignment_owner": "completed_fixed_negpool_bundle",
        }
        fold_audit = dict(split_audit)
        group_map = cohort[[split_group_col]].drop_duplicates().reset_index(
            drop=True)
    split_map = cohort.set_index(_SAMPLE_ID)[_SPLIT]
    membership[_SPLIT] = membership[_SAMPLE_ID].map(split_map).fillna(
        "excluded_by_cohort")

    pooled_tier_counts = (
        test.groupby(_TIER, sort=True).size().to_dict()
        if generate_assignments else {})
    cohort_audits = {
        dataset: prepared.cohort_audit
        for dataset, prepared in prepared_by_dataset.items()
    }
    cohort_audit = {
        "name": cfg["data"].get("cohort") or "none",
        "scope": "combined_2da_5da_normal",
        "before": _sum_class_counts(cohort_audits, "before"),
        "after": _sum_class_counts(cohort_audits, "after"),
        "by_dataset": cohort_audits,
    }
    dataset_validation = {
        dataset: {
            key: prepared.validation[key]
            for key in (
                "identity", "headers_identical", "nesting",
                "shared_values", "cohort")
        }
        for dataset, prepared in prepared_by_dataset.items()
    }
    validation = {
        "protocol_schema": "fixed_negpool_protocol_v2",
        "metric_semantics": METRIC_SEMANTICS_VERSION,
        "positive_class": "incorrect_identification",
        "mode": "combined_2da_5da_normal",
        "identity": {
            "columns": first.identity_cols,
            "local_sample_id": LOCAL_SAMPLE_ID_ALGORITHM,
            "combined_sample_id": COMBINED_SAMPLE_ID_ALGORITHM,
            "grouping": grouping_audit,
        },
        "datasets": dataset_validation,
        "feature_schema": {
            "ordered_feature_cols": list(first.feature_cols),
            "sha256": _feature_schema_sha256(first.feature_cols),
        },
        "cohort": cohort_audit,
        "fixed_split": split_audit,
        "fixed_folds": fold_audit,
        "fixed_test_tier_counts": {
            key: int(value) for key, value in pooled_tier_counts.items()
        },
        "fixed_test_dataset_tier_counts": {
            f"{dataset}::{tier}": int(value)
            for (dataset, tier), value in dataset_tier_counts.items()
        },
        "min_test_errors_per_dataset_tier": min_test_errors_per_tier,
    }
    logging.info(
        "combined preflight passed: cohort=%s, test dataset/tier=%s",
        cohort_audit["after"],
        validation["fixed_test_dataset_tier_counts"])
    return PreparedFixedNegpool(
        frame=cohort, membership=membership,
        group_fold_map=group_map, feature_cols=first.feature_cols,
        identity_cols=first.identity_cols, cohort_audit=cohort_audit,
        validation=validation, split_group_col=split_group_col)


def _locked_metrics(labels, trust_scores, vote_fractions, weights=None):
    ranking = evaluate_ranking(labels, trust_scores, sample_weight=weights)
    result = dict(ranking)
    for key, votes in vote_fractions.items():
        point = evaluate_at_threshold(
            labels, 1.0 - np.asarray(votes), 0.5,
            sample_weight=weights)
        result[key] = point
    return result


def _flat_result_row(model_name, subset, metrics):
    labels = metrics["labels"]
    fpr5 = metrics["fpr_5"]
    fpr10 = metrics["fpr_10"]
    return {
        "metric_semantics": METRIC_SEMANTICS_VERSION,
        "positive_class": "incorrect_identification",
        "model": model_name,
        "test_subset": subset,
        "n_rows": int(len(labels)),
        "n_actual_correct": int((labels == 1).sum()),
        "n_actual_error": int((labels == 0).sum()),
        "roc_auc": metrics["roc_auc"],
        "error_pr_auc": metrics["error_pr_auc"],
        "locked_fnr_at_fpr5": fpr5["fnr"],
        "observed_fpr_at_fpr5": fpr5["fpr"],
        "locked_error_recall_at_fpr10": fpr10["error_recall"],
        "observed_fpr_at_fpr10": fpr10["fpr"],
    }


def _macro_domain_row(model_name, domain_rows):
    """Equal-dataset-weight macro average; row counts are intentionally blank."""
    metric_columns = (
        "roc_auc", "error_pr_auc", "locked_fnr_at_fpr5",
        "observed_fpr_at_fpr5", "locked_error_recall_at_fpr10",
        "observed_fpr_at_fpr10")
    return {
        "metric_semantics": METRIC_SEMANTICS_VERSION,
        "positive_class": "incorrect_identification",
        "model": model_name,
        "test_subset": "macro_equal_dataset_weight",
        "test_dataset": "macro_equal_weight",
        "n_rows": None,
        "n_actual_correct": None,
        "n_actual_error": None,
        **{
            column: float(np.mean([row[column] for row in domain_rows]))
            for column in metric_columns
        },
    }


def _bootstrap_paired(test_frame, model_predictions, reps, seed, group_col):
    """Paired sequence-cluster bootstrap on the unchanged full E20 test."""
    if reps < 1:
        return pd.DataFrame(columns=[
            "metric_semantics", "positive_class",
            "model_a", "model_b", "metric", "delta_b_minus_a",
            "bootstrap_mean_delta", "ci95_low", "ci95_high",
            "probability_improved", "n_bootstrap"])
    labels = test_frame["label"].to_numpy()
    group_codes, groups = pd.factorize(test_frame[group_col], sort=True)
    n_groups = len(groups)
    rng = np.random.default_rng(seed)
    metric_names = (
        "roc_auc", "error_pr_auc", "locked_fnr_at_fpr5",
        "locked_error_recall_at_fpr10")
    observed = {}
    for model, values in model_predictions.items():
        metrics = _locked_metrics(
            labels, values["trust_score"], {
                "fpr_5": values["fpr_5_vote_fraction"],
                "fpr_10": values["fpr_10_vote_fraction"],
            })
        observed[model] = {
            "roc_auc": metrics["roc_auc"],
            "error_pr_auc": metrics["error_pr_auc"],
            "locked_fnr_at_fpr5": metrics["fpr_5"]["fnr"],
            "locked_error_recall_at_fpr10": metrics["fpr_10"][
                "error_recall"],
        }
    samples = {
        pair: {metric: [] for metric in metric_names}
        for pair in PAIR_COMPARISONS
    }
    completed = 0
    for _ in range(reps):
        group_weight = rng.multinomial(
            n_groups, np.full(n_groups, 1.0 / n_groups))
        weights = group_weight[group_codes].astype("f8")
        if not weights[labels == 0].sum() or not weights[labels == 1].sum():
            continue
        replicate = {}
        for model, values in model_predictions.items():
            metrics = _locked_metrics(
                labels, values["trust_score"], {
                    "fpr_5": values["fpr_5_vote_fraction"],
                    "fpr_10": values["fpr_10_vote_fraction"],
                }, weights=weights)
            replicate[model] = {
                "roc_auc": metrics["roc_auc"],
                "error_pr_auc": metrics["error_pr_auc"],
                "locked_fnr_at_fpr5": metrics["fpr_5"]["fnr"],
                "locked_error_recall_at_fpr10": metrics["fpr_10"][
                    "error_recall"],
            }
        for model_a, model_b in PAIR_COMPARISONS:
            for metric in metric_names:
                samples[(model_a, model_b)][metric].append(
                    replicate[model_b][metric]
                    - replicate[model_a][metric])
        completed += 1

    rows = []
    for (model_a, model_b), metrics in samples.items():
        for metric, values in metrics.items():
            values = np.asarray(values, dtype="f8")
            delta = observed[model_b][metric] - observed[model_a][metric]
            higher_is_better = metric != "locked_fnr_at_fpr5"
            rows.append({
                "metric_semantics": METRIC_SEMANTICS_VERSION,
                "positive_class": "incorrect_identification",
                "model_a": model_a,
                "model_b": model_b,
                "metric": metric,
                "delta_b_minus_a": float(delta),
                "bootstrap_mean_delta": float(values.mean()),
                "ci95_low": float(np.quantile(values, 0.025)),
                "ci95_high": float(np.quantile(values, 0.975)),
                "probability_improved": float(
                    (values > 0).mean() if higher_is_better
                    else (values < 0).mean()),
                "higher_is_better": higher_is_better,
                "n_bootstrap": completed,
                "resampling_unit": group_col,
            })
    return pd.DataFrame(rows)


def _provenance(config_path, cfg, paths, argv):
    try:
        commit = subprocess.run(
            ["git", "rev-parse", "HEAD"], check=True, capture_output=True,
            text=True).stdout.strip()
        dirty = bool(subprocess.run(
            ["git", "status", "--porcelain", "--untracked-files=no"],
            check=True, capture_output=True, text=True).stdout.strip())
    except (OSError, subprocess.CalledProcessError):
        commit, dirty = None, None
    with open(config_path, "rb") as handle:
        config_hash = hashlib.sha256(handle.read()).hexdigest()
    try:
        import lightgbm
        import sklearn
        versions = {
            "python": platform.python_version(), "numpy": np.__version__,
            "pandas": pd.__version__, "scikit_learn": sklearn.__version__,
            "lightgbm": lightgbm.__version__,
        }
    except ImportError:
        versions = {"python": platform.python_version(),
                    "numpy": np.__version__, "pandas": pd.__version__}
    return {
        "created_at_utc": dt.datetime.now(dt.timezone.utc).isoformat(),
        "command": [sys.executable, *argv],
        "config_path": str(Path(config_path).resolve()),
        "config_sha256": config_hash,
        "git_commit": commit,
        "git_tracked_files_dirty": dirty,
        "versions": versions,
        "model_params": cfg.get("model", {}).get("params", {}),
        "training_params": cfg.get("training", {}),
        "inputs": [_file_fingerprint(path) for path in paths.values()],
    }


def _assert_formal_config(cfg):
    data = cfg.get("data", {})
    if data.get("feature_arm") not in {"evidence_all", "evidence_core"}:
        raise ValueError(
            "fixed-negpool requires a formal evidence_all/evidence_core arm")
    if data.get("cohort") != "evidence_common":
        raise ValueError("fixed-negpool requires cohort=evidence_common")
    if data.get("group_col") != "sequence":
        raise ValueError("fixed-negpool requires group_col=sequence")
    params = cfg.get("model", {}).get("params", {})
    forbidden = [key for key in ("is_unbalance", "scale_pos_weight")
                 if key in params]
    if forbidden:
        raise ValueError(
            "class weighting is intentionally excluded from this controlled "
            f"experiment; remove {forbidden}")
    targets, _ = _operating_targets(cfg)
    if not {0.05, 0.10}.issubset(set(targets)):
        raise ValueError("fixed-negpool requires FPR5 and FPR10 targets")


def _known_outputs(output_root):
    root = Path(output_root)
    paths = [
        root / "summary.json", root / "preflight.json",
        root / "manifests" / "membership.csv",
        root / "manifests" / "fixed_test_manifest.csv",
        root / "manifests" / "fold_map.csv",
        root / "predictions" / "fixed_test_predictions.csv",
        root / "fixed_test_summary.csv", root / "tier_summary.csv",
        root / "paired_bootstrap.csv",
        root / "domain_summary.csv", root / "domain_tier_summary.csv",
        root / "paired_bootstrap_by_domain.csv",
    ]
    return paths


def _assert_output_available(output_root, overwrite):
    root = Path(output_root)
    existing = [path for path in _known_outputs(root) if path.exists()]
    if existing and not overwrite:
        raise FileExistsError(
            "refusing to overwrite an existing fixed-negpool bundle; choose "
            "a new output root or pass --overwrite:\n  "
            + "\n  ".join(map(str, existing)))
    if root.exists() and (not root.is_dir() or any(root.iterdir())) \
            and not overwrite:
        raise FileExistsError(
            f"refusing to replace nonempty output path: {root}")


def _publish_bundle(staging, root, overwrite, *, prepare_only, dataset):
    """Publish a validated fixed-negpool bundle and restore on swap failure."""
    staging, root = Path(staging), Path(root)
    required = [
        staging / "bundle_status.json",
        staging / "preflight.json",
        staging / "manifests" / "membership.csv",
        staging / "manifests" / "fixed_test_manifest.csv",
        staging / "manifests" / "fold_map.csv",
    ]
    if not prepare_only:
        required.extend([
            staging / "summary.json",
            staging / "config_used.yaml",
            staging / "fixed_test_summary.csv",
            staging / "tier_summary.csv",
            staging / "paired_bootstrap.csv",
            staging / "predictions" / "fixed_test_predictions.csv",
        ])
        if dataset == "combined":
            required.extend([
                staging / "domain_summary.csv",
                staging / "domain_tier_summary.csv",
                staging / "paired_bootstrap_by_domain.csv",
            ])
    missing = [str(path) for path in required if not path.is_file()]
    if missing:
        raise ValueError(
            "refusing to publish an incomplete fixed-negpool bundle:\n  "
            + "\n  ".join(missing))
    status = json.loads(
        (staging / "bundle_status.json").read_text(encoding="utf-8"))
    expected_status = "prepare_only" if prepare_only else "complete"
    if status.get("status") != expected_status:
        raise ValueError(
            f"fixed-negpool status must be {expected_status!r}, got "
            f"{status.get('status')!r}")

    backup = None
    if root.exists():
        if not overwrite and (not root.is_dir() or any(root.iterdir())):
            raise FileExistsError(f"output path is not empty: {root}")
        backup = root.with_name(f".{root.name}.backup.{uuid.uuid4().hex}")
        os.replace(root, backup)
    try:
        os.replace(staging, root)
    except Exception:
        if backup is not None and backup.exists() and not root.exists():
            os.replace(backup, root)
        raise
    if backup is not None and backup.exists():
        try:
            if backup.is_dir():
                shutil.rmtree(backup)
            else:
                backup.unlink()
        except OSError as exc:
            logging.warning(
                "published bundle but could not remove %s: %s", backup, exc)


def _write_prepared(prepared, output_root):
    root = Path(output_root)
    _atomic_csv(str(root / "manifests" / "membership.csv"),
                prepared.membership)
    identity = list(dict.fromkeys([
        _SAMPLE_ID, _SOURCE_SAMPLE_ID, _DATASET,
        *prepared.identity_cols, "label", _TIER, _STRATUM, _SPLIT,
        _OUTER_FOLD, prepared.split_group_col, *RELATIONSHIP_COLUMNS,
    ]))
    fixed = prepared.frame[
        [column for column in identity if column in prepared.frame]].copy()
    _atomic_csv(str(root / "manifests" / "fixed_test_manifest.csv"), fixed)
    _atomic_csv(str(root / "manifests" / "fold_map.csv"),
                prepared.group_fold_map)
    _atomic_json(str(root / "preflight.json"), prepared.validation)


def _frozen_bundle_hashes(output_root):
    """Anchor every protocol/comparator input consumed by deep trainers."""
    root = Path(output_root)
    paths = {
        "preflight": root / "preflight.json",
        "membership": root / "manifests" / "membership.csv",
        "fixed_test_manifest": (
            root / "manifests" / "fixed_test_manifest.csv"),
        "fold_map": root / "manifests" / "fold_map.csv",
        "fixed_test_predictions": (
            root / "predictions" / "fixed_test_predictions.csv"),
    }
    missing = [str(path) for path in paths.values() if not path.is_file()]
    if missing:
        raise FileNotFoundError(
            "cannot freeze an incomplete fixed-negpool bundle:\n  "
            + "\n  ".join(missing))
    return {
        name: _file_fingerprint(path)["sha256"]
        for name, path in paths.items()
    }


def _feature_schema_sha256(feature_cols):
    payload = json.dumps(
        list(feature_cols), ensure_ascii=False, separators=(",", ":"))
    return hashlib.sha256(payload.encode("utf-8")).hexdigest()


def _run_fixed_negpool_into_root(
        config_path, feature_root, dataset, output_root, *,
        test_fraction=0.20, min_test_errors_per_tier=100,
        split_candidates=128, bootstrap_reps=1000,
        bootstrap_seed=20260810, prepare_only=False, argv=None):
    """Build one result bundle in an empty staging root."""
    with open(config_path, encoding="utf-8") as handle:
        cfg = yaml.safe_load(handle)
    _assert_formal_config(cfg)
    log_path = Path(output_root) / "logs" / "fixed_negpool.log"
    log_path.parent.mkdir(parents=True, exist_ok=True)
    logging.basicConfig(
        level=logging.INFO, force=True,
        format="%(asctime)s %(levelname)s %(message)s",
        handlers=[logging.FileHandler(
            log_path, mode="w", encoding="utf-8"), logging.StreamHandler()])
    logging.info(
        "starting fixed-negpool experiment: dataset=%s, feature_root=%s",
        dataset, feature_root)
    if dataset == "combined":
        paths = {
            f"{source}_{pool}": path
            for source in ("2da", "5da", "normal")
            for pool, path in feature_paths(feature_root, source).items()
        }
        prepared = prepare_combined_fixed_negpool(
            feature_root, cfg, test_fraction=test_fraction,
            min_test_errors_per_tier=min_test_errors_per_tier,
            split_candidates=split_candidates)
    else:
        paths = feature_paths(feature_root, dataset)
        prepared = prepare_fixed_negpool(
            paths, cfg, test_fraction=test_fraction,
            min_test_errors_per_tier=min_test_errors_per_tier,
            split_candidates=split_candidates)
    _write_prepared(prepared, output_root)
    if prepare_only:
        _atomic_json(str(Path(output_root) / "bundle_status.json"), {
            "status": "prepare_only",
            "metric_semantics": METRIC_SEMANTICS_VERSION,
            "positive_class": "incorrect_identification",
        })
        return {"mode": "prepare_only", **prepared.validation}

    root = Path(output_root)
    os.makedirs(root / "models", exist_ok=True)
    os.makedirs(root / "predictions", exist_ok=True)
    config_used = root / "config_used.yaml"
    if dataset == "combined":
        effective_cfg = copy.deepcopy(cfg)
        effective_cfg["data"]["train_files"] = [
            str(feature_paths(feature_root, source)["neg20"])
            for source in ("2da", "5da", "normal")
        ]
        effective_cfg["data"]["test_files"] = []
        effective_cfg["data"]["combined_fixed_test"] = {
            "datasets": ["2da", "5da", "normal"],
            "master_pool": "neg20",
            "group_col": "sequence",
            "stratification": "dataset_x_negative_tier",
            "test_fraction": test_fraction,
        }
        config_tmp = f"{config_used}.tmp.{os.getpid()}"
        with open(config_tmp, "w", encoding="utf-8") as handle:
            yaml.safe_dump(
                effective_cfg, handle, sort_keys=False, allow_unicode=True)
        os.replace(config_tmp, config_used)
    else:
        shutil.copyfile(config_path, config_used)
    target_col = cfg["data"]["target_col"]
    group_col = prepared.split_group_col
    train_master = prepared.frame.loc[
        prepared.frame[_SPLIT].eq("train")].copy()
    test = prepared.frame.loc[
        prepared.frame[_SPLIT].eq("test")].copy().reset_index(drop=True)
    test_labels = test[target_col].to_numpy()
    n_folds = int(cfg["training"].get("cv_folds", 5))
    inner_columns = [f"inner_valid_for_fold_{fold}"
                     for fold in range(n_folds)]
    target_fprs, _ = _operating_targets(cfg)

    prediction_identity = [
        column for column in (
            _SAMPLE_ID, _SOURCE_SAMPLE_ID, _DATASET,
            *prepared.identity_cols, target_col, _TIER, _SOURCE_ROW)
        if column in test
    ]
    test_predictions = test[prediction_identity].copy()
    model_predictions = {}
    model_summaries = {}
    full_rows = []
    tier_rows = []
    domain_rows = []
    domain_tier_rows = []

    for model_name, allowed_tiers in MODEL_TIERS.items():
        use = train_master[target_col].eq(1) | train_master[_TIER].isin(
            allowed_tiers)
        train = train_master.loc[use].copy().reset_index(drop=True)
        logging.info(
            "training %s with error tiers=%s and rows=%d",
            model_name, allowed_tiers, len(train))
        _validate_frame(
            train, prepared.feature_cols, target_col, group_col)
        fixed_inner = {
            fold: train[inner_columns[fold]].to_numpy(dtype=bool)
            for fold in range(n_folds)
        }
        model_prefix = str(root / "models" / model_name.lower())
        oof, fold_metrics, model_paths, oof_folds = assemble_oof(
            train, train[prepared.feature_cols], train[target_col],
            train[group_col], cfg, prepared.feature_cols, model_prefix,
            return_fold_ids=True,
            predefined_fold_ids=train[_OUTER_FOLD].to_numpy(),
            predefined_inner_valid=fixed_inner)
        trust, _, aggregate, details = evaluate_cross_test(
            model_paths, test[prepared.feature_cols], test_labels,
            fold_metrics=fold_metrics, return_details=True)
        votes = details["fold_calibrated_error_vote_fractions"]
        required_keys = {_op_key(target) for target in target_fprs}
        if not {"fpr_5", "fpr_10"}.issubset(required_keys & set(votes)):
            raise ValueError("trained model lacks locked FPR5/FPR10 votes")

        full_metrics = _locked_metrics(test_labels, trust, votes)
        full_metrics["labels"] = test_labels
        full_rows.append(_flat_result_row(model_name, "full_E20", full_metrics))
        tier_metrics = {}
        for tier in ERROR_TIERS:
            mask = test[target_col].eq(1) | test[_TIER].eq(tier)
            indices = np.flatnonzero(mask.to_numpy())
            subset_votes = {key: np.asarray(value)[indices]
                            for key, value in votes.items()}
            metrics = _locked_metrics(
                test_labels[indices], np.asarray(trust)[indices], subset_votes)
            metrics["labels"] = test_labels[indices]
            tier_rows.append(_flat_result_row(
                model_name, f"correct_plus_{tier}", metrics))
            tier_metrics[tier] = {
                key: value for key, value in metrics.items()
                if key != "labels"
            }

        domain_metrics = {}
        if _DATASET in test:
            this_model_domain_rows = []
            for test_dataset in sorted(test[_DATASET].unique()):
                domain_mask = test[_DATASET].eq(test_dataset).to_numpy()
                domain_idx = np.flatnonzero(domain_mask)
                domain_votes = {
                    key: np.asarray(value)[domain_idx]
                    for key, value in votes.items()
                }
                metrics = _locked_metrics(
                    test_labels[domain_idx], np.asarray(trust)[domain_idx],
                    domain_votes)
                metrics["labels"] = test_labels[domain_idx]
                row = _flat_result_row(
                    model_name, "full_E20", metrics)
                row["test_dataset"] = test_dataset
                domain_rows.append(row)
                this_model_domain_rows.append(row)
                domain_metrics[test_dataset] = {
                    key: value for key, value in metrics.items()
                    if key != "labels"
                }

                for tier in ERROR_TIERS:
                    tier_mask = domain_mask & (
                        test[target_col].eq(1)
                        | test[_TIER].eq(tier)).to_numpy()
                    tier_idx = np.flatnonzero(tier_mask)
                    tier_votes = {
                        key: np.asarray(value)[tier_idx]
                        for key, value in votes.items()
                    }
                    tier_domain_metrics = _locked_metrics(
                        test_labels[tier_idx],
                        np.asarray(trust)[tier_idx], tier_votes)
                    tier_domain_metrics["labels"] = test_labels[tier_idx]
                    tier_row = _flat_result_row(
                        model_name, f"correct_plus_{tier}",
                        tier_domain_metrics)
                    tier_row["test_dataset"] = test_dataset
                    domain_tier_rows.append(tier_row)
            domain_rows.append(_macro_domain_row(
                model_name, this_model_domain_rows))

        train_counts = train.groupby(_TIER, sort=True).size().to_dict()
        model_summaries[model_name] = {
            "included_error_tiers": list(allowed_tiers),
            "n_train": int(len(train)),
            "train_counts_by_tier": {
                key: int(value) for key, value in train_counts.items()},
            "n_features": len(prepared.feature_cols),
            "feature_cols": prepared.feature_cols,
            "fold_metrics": fold_metrics,
            "model_paths": model_paths,
            "full_fixed_test": {
                key: value for key, value in full_metrics.items()
                if key != "labels"
            },
            "tier_fixed_test": tier_metrics,
            "domain_fixed_test": domain_metrics,
            "locked_operating_points": aggregate[
                "locked_operating_points"],
        }
        model_predictions[model_name] = {
            "trust_score": np.asarray(trust),
            "fpr_5_vote_fraction": np.asarray(votes["fpr_5"]),
            "fpr_10_vote_fraction": np.asarray(votes["fpr_10"]),
        }
        test_predictions[f"{model_name}_trust_score"] = trust
        test_predictions[f"{model_name}_error_score"] = 1.0 - trust
        test_predictions[f"{model_name}_fpr5_error_vote_fraction"] = (
            votes["fpr_5"])
        test_predictions[f"{model_name}_fpr10_error_vote_fraction"] = (
            votes["fpr_10"])
        oof_identity = [
            column for column in (
                _SAMPLE_ID, _SOURCE_SAMPLE_ID, _DATASET, group_col,
                target_col, _TIER, _OUTER_FOLD)
            if column in train
        ]
        oof_frame = train[oof_identity].copy()
        oof_frame["oof_fold"] = oof_folds
        oof_frame["trust_score"] = oof
        oof_frame["error_score"] = 1.0 - oof
        _atomic_csv(str(root / "predictions" /
                        f"{model_name.lower()}_train_oof.csv"), oof_frame)

    logging.info(
        "running %d paired sequence-cluster bootstrap replicates",
        bootstrap_reps)
    fixed_summary = pd.DataFrame(full_rows)
    tier_summary = pd.DataFrame(tier_rows)
    bootstrap = _bootstrap_paired(
        test, model_predictions, bootstrap_reps, bootstrap_seed, group_col)
    domain_bootstraps = []
    if _DATASET in test:
        for offset, test_dataset in enumerate(sorted(test[_DATASET].unique())):
            mask = test[_DATASET].eq(test_dataset).to_numpy()
            domain_test = test.loc[mask].reset_index(drop=True)
            domain_predictions = {
                model: {
                    key: np.asarray(value)[mask]
                    for key, value in values.items()
                }
                for model, values in model_predictions.items()
            }
            domain_bootstrap = _bootstrap_paired(
                domain_test, domain_predictions, bootstrap_reps,
                bootstrap_seed + offset + 1, group_col)
            domain_bootstrap.insert(0, "test_dataset", test_dataset)
            domain_bootstraps.append(domain_bootstrap)
    domain_bootstrap = (
        pd.concat(domain_bootstraps, ignore_index=True)
        if domain_bootstraps else pd.DataFrame())
    _atomic_csv(str(root / "predictions" / "fixed_test_predictions.csv"),
                test_predictions)
    _atomic_csv(str(root / "fixed_test_summary.csv"), fixed_summary)
    _atomic_csv(str(root / "tier_summary.csv"), tier_summary)
    _atomic_csv(str(root / "paired_bootstrap.csv"), bootstrap)
    if domain_rows:
        _atomic_csv(str(root / "domain_summary.csv"),
                    pd.DataFrame(domain_rows))
        _atomic_csv(str(root / "domain_tier_summary.csv"),
                    pd.DataFrame(domain_tier_rows))
        _atomic_csv(str(root / "paired_bootstrap_by_domain.csv"),
                    domain_bootstrap)

    frozen_bundle = {
        "schema": "fixed_negpool_frozen_bundle_v2",
        "complete": True,
        "feature_cols": list(prepared.feature_cols),
        "feature_cols_sha256": _feature_schema_sha256(
            prepared.feature_cols),
        "artifact_sha256": _frozen_bundle_hashes(root),
    }
    summary = {
        "metric_semantics": METRIC_SEMANTICS_VERSION,
        "positive_class": "incorrect_identification",
        "experiment": (
            "combined_fixed_test_nested_negative_pool_v1"
            if dataset == "combined"
            else "fixed_test_nested_negative_pool_v1"),
        "dataset": dataset,
        "design": {
            "master_feature_table": (
                {
                    source: str(feature_paths(
                        feature_root, source)["neg20"])
                    for source in ("2da", "5da", "normal")
                }
                if dataset == "combined" else str(paths["neg20"])),
            "fixed_test": "correct_test + complete_E20_error_test",
            "training_models": {
                model: list(tiers) for model, tiers in MODEL_TIERS.items()},
            "same_correct_training_rows": True,
            "same_outer_and_inner_group_assignments": True,
            "split_group_col": group_col,
            "split_grouping": prepared.validation["identity"].get(
                "split_grouping", prepared.validation["identity"].get(
                    "grouping")),
            "class_weighting": False,
            "feature_arm": cfg["data"]["feature_arm"],
            "cohort": cfg["data"]["cohort"],
            "interpretation": (
                "models are compared on identical test rows; tier analyses "
                "separate broad discrimination from added-tier coverage"),
            "combined_domain_reporting": (
                "pooled + equal-weight macro + per-dataset"
                if dataset == "combined" else None),
            "tier_pr_auc_note": (
                "PR-AUC is paired/comparable between models within a tier, "
                "not as a raw number across tiers with different prevalence"),
        },
        "validation": prepared.validation,
        "frozen_bundle": frozen_bundle,
        "models": model_summaries,
        "bootstrap": {
            "n_requested": bootstrap_reps,
            "n_completed": int(bootstrap["n_bootstrap"].max())
            if len(bootstrap) else 0,
            "seed": bootstrap_seed,
            "resampling_unit": group_col,
            "primary_hypothesis": (
                "M20 has lower locked FNR@FPR5 than M5 on the fixed E20 test"),
        },
        "artifacts": {
            "membership": str(root / "manifests" / "membership.csv"),
            "fixed_test_manifest": str(
                root / "manifests" / "fixed_test_manifest.csv"),
            "fold_map": str(root / "manifests" / "fold_map.csv"),
            "fixed_test_predictions": str(
                root / "predictions" / "fixed_test_predictions.csv"),
            "fixed_test_summary": str(root / "fixed_test_summary.csv"),
            "tier_summary": str(root / "tier_summary.csv"),
            "paired_bootstrap": str(root / "paired_bootstrap.csv"),
            "domain_summary": (
                str(root / "domain_summary.csv") if domain_rows else None),
            "domain_tier_summary": (
                str(root / "domain_tier_summary.csv")
                if domain_rows else None),
            "paired_bootstrap_by_domain": (
                str(root / "paired_bootstrap_by_domain.csv")
                if domain_rows else None),
            "log": str(log_path),
        },
        "provenance": _provenance(
            config_path, cfg, paths,
            argv or ["tools/spec_trainer/src/fixed_negpool.py"]),
    }
    _atomic_json(str(root / "summary.json"), summary)
    _atomic_json(str(root / "bundle_status.json"), {
        "status": "complete",
        "metric_semantics": METRIC_SEMANTICS_VERSION,
        "positive_class": "incorrect_identification",
    })
    logging.info("fixed-negpool complete: %s", root / "summary.json")
    return summary


def run_fixed_negpool(config_path, feature_root, dataset, output_root, *,
                      test_fraction=0.20, min_test_errors_per_tier=100,
                      split_candidates=128, bootstrap_reps=1000,
                      bootstrap_seed=20260810, overwrite=False,
                      prepare_only=False, argv=None):
    """Build then atomically publish one controlled result bundle."""
    root = Path(output_root)
    _assert_output_available(root, overwrite)
    root.parent.mkdir(parents=True, exist_ok=True)
    staging = Path(tempfile.mkdtemp(
        prefix=f".{root.name}.staging.", dir=root.parent))
    try:
        result = _run_fixed_negpool_into_root(
            config_path, feature_root, dataset, staging,
            test_fraction=test_fraction,
            min_test_errors_per_tier=min_test_errors_per_tier,
            split_candidates=split_candidates,
            bootstrap_reps=bootstrap_reps,
            bootstrap_seed=bootstrap_seed,
            prepare_only=prepare_only,
            argv=argv,
        )
        if not prepare_only:
            result = json.loads(json.dumps(result).replace(
                str(staging), str(root)))
            _atomic_json(str(staging / "summary.json"), result)
        _publish_bundle(
            staging, root, overwrite, prepare_only=prepare_only,
            dataset=dataset)
        return result
    except Exception:
        shutil.rmtree(staging, ignore_errors=True)
        raise


def _parser():
    parser = argparse.ArgumentParser(
        description="Fixed-test controlled E5/E10/E20 training experiment")
    parser.add_argument("--config", required=True,
                        help="formal neg20 CV template YAML")
    parser.add_argument("--feature-root", required=True)
    parser.add_argument("--dataset", required=True,
                        choices=("2da", "5da", "normal", "combined"))
    parser.add_argument("--output-root", required=True)
    parser.add_argument("--test-fraction", type=float, default=0.20)
    parser.add_argument("--min-test-errors-per-tier", type=int, default=100)
    parser.add_argument("--split-candidates", type=int, default=128)
    parser.add_argument("--bootstrap-reps", type=int, default=1000)
    parser.add_argument("--bootstrap-seed", type=int, default=20260810)
    parser.add_argument("--prepare-only", action="store_true")
    parser.add_argument("--overwrite", action="store_true")
    return parser


def main(argv=None):
    args = _parser().parse_args(argv)
    command = ["tools/spec_trainer/src/fixed_negpool.py",
               *(argv if argv is not None else sys.argv[1:])]
    result = run_fixed_negpool(
        args.config, args.feature_root, args.dataset, args.output_root,
        test_fraction=args.test_fraction,
        min_test_errors_per_tier=args.min_test_errors_per_tier,
        split_candidates=args.split_candidates,
        bootstrap_reps=args.bootstrap_reps,
        bootstrap_seed=args.bootstrap_seed,
        overwrite=args.overwrite, prepare_only=args.prepare_only,
        argv=command)
    if args.prepare_only:
        counts = result["fixed_test_tier_counts"]
        print(f"preflight OK: fixed test tier counts={counts}")
    else:
        print(f"fixed-negpool complete: {args.output_root}/summary.json")
    return result


if __name__ == "__main__":
    main()
