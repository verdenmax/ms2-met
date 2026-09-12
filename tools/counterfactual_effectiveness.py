"""Nested grouped evaluation of incremental counterfactual-negative utility.

The outer folds contain only real q<=1% correct/entrapment identifications.
Counterfactual parents participate in the connected-component graph but never
become model rows.  Synthetic candidates enter only the training side of the
outer fold containing their real parent family.  Existing ``cv_train.py`` then
owns inner grouped CV, OOF threshold calibration, and external majority vote.
"""

from __future__ import annotations

import argparse
import copy
import hashlib
import json
import math
import os
import shutil
import tempfile
from dataclasses import dataclass
from pathlib import Path
from typing import Mapping

import numpy as np
import pandas as pd
import yaml

from spectrum.psm_identity import peptide_group_id
from tools.counterfactual_group_holdout import (
    SOURCE_COMPOSITION,
    SOURCE_KR,
    SOURCE_LOCAL,
    SOURCE_POSITIVE,
    SYNTHETIC_SOURCES,
)
from tools.spec_trainer.src.cohort import apply_training_cohort
from tools.spec_trainer.src.cv_core import (
    METRIC_SEMANTICS_VERSION,
    evaluate_at_threshold,
    evaluate_ranking,
    make_cv_splits,
)
from tools.spec_trainer.src.sample_groups import assign_leakage_groups


SOURCE_REAL_CORRECT = "real_correct_q01"
SOURCE_REAL_ERROR = "real_entrapment_q01"
MODEL_SOURCES = {
    "m_real": (),
    "m_real_c": (SOURCE_COMPOSITION,),
    "m_real_k": (SOURCE_KR,),
    "m_real_l": (SOURCE_LOCAL,),
    "m_real_all": SYNTHETIC_SOURCES,
}
SOURCE_FILE_STEMS = {
    SOURCE_COMPOSITION: "composition",
    SOURCE_KR: "kr",
    SOURCE_LOCAL: "local",
}

_GROUP_COL = "leakage_group_id"
_SAMPLE_ID = "experiment_sample_id"
_ORIGIN = "experiment_origin"
_SOURCE_ROW = "experiment_source_row"
_OUTER_FOLD = "experiment_outer_fold"
_ROLE = "experiment_role"
_SELECTED = "experiment_selected_candidate"
_ORPHAN = "experiment_orphan_synthetic"
_INNER_FOLD = "experiment_inner_fold"
_INNER_VALID_PREFIX = "experiment_inner_valid_fold_"
_ROW_IDENTITY_COLUMNS = (_SAMPLE_ID, _OUTER_FOLD, _GROUP_COL, "label")


@dataclass(frozen=True)
class EffectivenessDesign:
    outer_folds: int = 5
    seed: int = 42
    cohort: str = "evidence_observed"
    qvalue_max: float = 0.01
    bootstrap_reps: int = 1000
    bootstrap_seed: int = 20260908
    familywise_alpha: float = 0.05
    max_fpr_increase: float = 0.01
    minimum_recall_gain: float = 0.03

    def __post_init__(self) -> None:
        if self.outer_folds < 2:
            raise ValueError("outer_folds must be at least 2")
        if not 0.0 < self.qvalue_max <= 1.0:
            raise ValueError("qvalue_max must be in (0, 1]")
        if self.bootstrap_reps < 1:
            raise ValueError("bootstrap_reps must be positive")
        if (not math.isfinite(self.familywise_alpha)
                or not 0.0 < self.familywise_alpha < 1.0):
            raise ValueError("familywise_alpha must be finite and in (0, 1)")
        if (not math.isfinite(self.minimum_recall_gain)
                or not 0.0 <= self.minimum_recall_gain <= 1.0):
            raise ValueError("minimum_recall_gain must be finite and in [0, 1]")
        if (not math.isfinite(self.max_fpr_increase)
                or not 0.0 <= self.max_fpr_increase <= 1.0):
            raise ValueError("max_fpr_increase must be finite and in [0, 1]")


@dataclass(frozen=True)
class EffectivenessBundle:
    real_rows: pd.DataFrame
    selected_synthetic: pd.DataFrame
    synthetic_diagnostics: pd.DataFrame
    manifest: pd.DataFrame
    audit: Mapping[str, object]
    design: EffectivenessDesign


def _normalized_text(frame: pd.DataFrame, column: str) -> pd.Series:
    if column not in frame:
        return pd.Series("", index=frame.index, dtype="string")
    return frame[column].astype("string").str.strip().fillna("")


def _digest(*parts: object) -> str:
    payload = "|".join(f"{len(str(part))}:{part}" for part in parts)
    return hashlib.sha256(payload.encode("utf-8")).hexdigest()


def _require(frame: pd.DataFrame, columns: tuple[str, ...], name: str) -> None:
    missing = [column for column in columns if column not in frame]
    if missing:
        raise ValueError(f"{name} is missing required columns: {missing}")


def _validate_binary_labels(frame: pd.DataFrame, name: str) -> pd.Series:
    labels = pd.to_numeric(frame["label"], errors="coerce")
    if labels.isna().any() or not set(labels.unique()).issubset({0, 1}):
        raise ValueError(f"{name} labels must contain only 0 and 1")
    return labels.astype(int)


def _assign_sample_ids(frame: pd.DataFrame) -> None:
    columns = [
        column for column in (
            _ORIGIN, "negative_source", "sequence", "charge", "precursor_mz",
            "rt", "raw_title1", "raw_title2", "label_type", "parent_id",
            "query_id",
        ) if column in frame
    ]
    frame[_SAMPLE_ID] = [
        _digest("counterfactual_effectiveness_sample_v1", *values)
        for values in frame[columns].itertuples(index=False, name=None)
    ]
    duplicated = frame[_SAMPLE_ID].duplicated(keep=False)
    if duplicated.any():
        examples = frame.loc[duplicated, _SAMPLE_ID].drop_duplicates().head(5)
        raise ValueError(f"non-unique effectiveness sample IDs: {examples.tolist()}")


def _prepare_counterfactual(frame: pd.DataFrame) -> tuple[pd.DataFrame, dict]:
    _require(
        frame,
        ("sequence", "label", "negative_source", "parent_id",
         "peptide_group_id"),
        "counterfactual table")
    work = frame.copy()
    work["label"] = _validate_binary_labels(work, "counterfactual")
    work["negative_source"] = _normalized_text(work, "negative_source")
    if _normalized_text(work, "sequence").eq("").any():
        raise ValueError("counterfactual table contains an empty sequence")
    allowed = {SOURCE_POSITIVE, *SYNTHETIC_SOURCES}
    unexpected = sorted(set(work["negative_source"]) - allowed)
    if unexpected:
        raise ValueError(f"unexpected counterfactual sources: {unexpected}")
    invalid = (
        (work["negative_source"].eq(SOURCE_POSITIVE) & work["label"].ne(1))
        | (work["negative_source"].isin(SYNTHETIC_SOURCES)
           & work["label"].ne(0))
    )
    if invalid.any():
        raise ValueError("counterfactual source/label convention is inconsistent")
    positive_parents = set(_normalized_text(
        work.loc[work["negative_source"].eq(SOURCE_POSITIVE)], "parent_id"))
    positive_parents.discard("")
    if not positive_parents:
        raise ValueError("counterfactual table has no gold-positive parent")
    synthetic = work["negative_source"].isin(SYNTHETIC_SOURCES)
    orphan = synthetic & ~_normalized_text(
        work, "parent_id").isin(positive_parents)
    audit = {
        "n_input": int(len(frame)),
        "n_orphan_synthetic_rows": int(orphan.sum()),
        "orphan_by_source": {
            str(source): int(count) for source, count in
            work.loc[orphan, "negative_source"].value_counts().items()
        },
    }
    # Keep invalid/orphan candidates in the relationship graph. Even though
    # they can never become model rows, their sequence/family tokens can bridge
    # two otherwise separate components. They are removed only after the full
    # graph has been assigned and the common cohort has been applied.
    work[_ORPHAN] = orphan
    work[_ORIGIN] = "counterfactual"
    work[_SOURCE_ROW] = work.index.astype(int)
    return work, audit


def _prepare_real(frame: pd.DataFrame,
                  qvalue_max: float) -> tuple[pd.DataFrame, dict]:
    _require(
        frame, ("sequence", "label", "label_type", "q_value"),
        "real q01 table")
    work = frame.copy()
    work["label"] = _validate_binary_labels(work, "real q01")
    label_type = _normalized_text(work, "label_type").str.lower()
    inconsistent_type = (
        (work["label"].eq(1) & label_type.ne("positive"))
        | (work["label"].eq(0) & label_type.ne("negative"))
    )
    if inconsistent_type.any():
        raise ValueError(
            "real q01 label_type must map positive->label 1 and negative->label 0")
    if _normalized_text(work, "sequence").eq("").any():
        raise ValueError("real q01 table contains an empty sequence")
    qvalues = pd.to_numeric(work["q_value"], errors="coerce")
    if (qvalues.isna().any() or (qvalues < 0).any()
            or (qvalues > qvalue_max + 1e-12).any()):
        raise ValueError(
            f"real table must be prefiltered to q_value <= {qvalue_max}")
    work["negative_source"] = work["label"].map({
        1: SOURCE_REAL_CORRECT,
        0: SOURCE_REAL_ERROR,
    })
    computed_peptide_groups = work["sequence"].map(peptide_group_id)
    if "peptide_group_id" in work:
        existing_peptide_groups = _normalized_text(work, "peptide_group_id")
        work["peptide_group_id"] = existing_peptide_groups.mask(
            existing_peptide_groups.eq(""), computed_peptide_groups)
    else:
        work["peptide_group_id"] = computed_peptide_groups
    # Preserve any real relationship metadata until after connected-component
    # assignment. The current baseline has none, but future inputs may carry a
    # legitimate bridge. Missing columns are added solely for a stable schema.
    for column in ("query_id", "parent_id", "group_id", "candidate_family_id"):
        if column not in work:
            work[column] = pd.NA
    work[_ORPHAN] = False
    work[_ORIGIN] = "real_q01"
    work[_SOURCE_ROW] = work.index.astype(int)
    return work, {
        "n_input": int(len(frame)),
        "n_correct": int(work["label"].eq(1).sum()),
        "n_error": int(work["label"].eq(0).sum()),
        "label_type_contract": "positive=correct; negative=entrapment error",
        "qvalue_min": float(qvalues.min()),
        "qvalue_max": float(qvalues.max()),
        "configured_qvalue_max": qvalue_max,
    }


def _select_candidates(frame: pd.DataFrame, seed: int) -> pd.DataFrame:
    pieces = []
    for source in SYNTHETIC_SOURCES:
        candidates = frame.loc[frame["negative_source"].eq(source)].copy()
        candidates["__rank"] = [
            _digest("effectiveness_candidate_v1", seed, source, parent, sample)
            for parent, sample in zip(_normalized_text(candidates, "parent_id"),
                                      candidates[_SAMPLE_ID])
        ]
        selected = (candidates.sort_values(
            ["parent_id", "__rank", _SAMPLE_ID], kind="mergesort")
            .drop_duplicates("parent_id", keep="first")
            .drop(columns="__rank"))
        pieces.append(selected)
    return pd.concat(pieces, ignore_index=True, sort=False)


def _assign_grouped_folds(frame: pd.DataFrame, n_folds: int, seed: int,
                          context: str) -> pd.Series:
    fold_ids = np.full(len(frame), -1, dtype=int)
    splits = make_cv_splits(
        frame["label"].to_numpy(), frame[_GROUP_COL].to_numpy(),
        n_folds=n_folds, seed=seed)
    for fold, (_, test_indices) in enumerate(splits):
        fold_ids[test_indices] = fold
    if (fold_ids < 0).any():
        raise AssertionError(f"a {context} row has no fold")
    assignments = pd.DataFrame({
        "group": frame[_GROUP_COL].to_numpy(), "fold": fold_ids,
    }).groupby("group", sort=False)["fold"].nunique()
    if assignments.gt(1).any():
        raise AssertionError(f"{context} folds split a connected group")
    return pd.Series(fold_ids, index=frame.index, dtype=int)


def _assign_outer_folds(real: pd.DataFrame,
                        design: EffectivenessDesign) -> pd.Series:
    return _assign_grouped_folds(
        real, design.outer_folds, design.seed, "outer-evaluation")


def _inner_valid_column(fold: int) -> str:
    return f"{_INNER_VALID_PREFIX}{fold}"


def _attach_inner_protocol(
        real_train: pd.DataFrame,
        synthetic_train: pd.DataFrame,
        training: Mapping[str, object],
) -> tuple[pd.DataFrame, pd.DataFrame, dict[str, object]]:
    """Freeze member folds/early-stop groups before model augmentation."""
    from tools.spec_trainer.src.cv_train import _inner_split

    n_folds = int(training.get("cv_folds", 5))
    seed = int(training.get("cv_seed", 42))
    valid_size = float(training.get("valid_size", 0.15))
    if not 2 <= n_folds <= 5:
        raise ValueError("effectiveness inner cv_folds must be between 2 and 5")
    if not 0.0 < valid_size < 1.0:
        raise ValueError("effectiveness inner valid_size must be in (0, 1)")
    real_train = real_train.copy()
    real_train[_INNER_FOLD] = _assign_grouped_folds(
        real_train, n_folds, seed, "inner-OOF")
    dummy = pd.DataFrame({"unused": np.zeros(len(real_train))})
    fold_ids = real_train[_INNER_FOLD].to_numpy()
    validation_counts = {}
    for fold in range(n_folds):
        train_indices = np.flatnonzero(fold_ids != fold)
        _, valid_indices = _inner_split(
            dummy, real_train["label"], real_train[_GROUP_COL],
            train_indices, valid_size, seed + fold)
        valid_groups = set(real_train.iloc[valid_indices][_GROUP_COL])
        column = _inner_valid_column(fold)
        real_train[column] = real_train[_GROUP_COL].isin(valid_groups)
        if real_train.loc[real_train[_INNER_FOLD].eq(fold), column].any():
            raise AssertionError(
                f"inner fold {fold} validation overlaps its OOF groups")
        validation_counts[str(fold)] = {
            "n_groups": len(valid_groups),
            "n_rows": int(real_train[column].sum()),
        }

    protocol_columns = [
        _INNER_FOLD, *(_inner_valid_column(fold) for fold in range(n_folds))]
    group_protocol = real_train[[_GROUP_COL, *protocol_columns]].drop_duplicates()
    if group_protocol[_GROUP_COL].duplicated().any():
        raise AssertionError("an inner protocol assigns one group more than once")
    synthetic_train = synthetic_train.drop(
        columns=protocol_columns, errors="ignore").merge(
            group_protocol, on=_GROUP_COL, how="left", validate="many_to_one")
    if synthetic_train[protocol_columns].isna().any().any():
        raise AssertionError(
            "a synthetic training family has no frozen inner protocol")
    synthetic_train[_INNER_FOLD] = synthetic_train[_INNER_FOLD].astype(int)
    for fold in range(n_folds):
        synthetic_train[_inner_valid_column(fold)] = synthetic_train[
            _inner_valid_column(fold)].astype(bool)
    return real_train, synthetic_train, {
        "method": "real_group_protocol_inherited_by_synthetic_family_v1",
        "cv_folds": n_folds,
        "cv_seed": seed,
        "valid_size": valid_size,
        "validation_counts": validation_counts,
    }


def _source_counts(frame: pd.DataFrame) -> dict[str, int]:
    return {
        str(source): int(count) for source, count in
        frame["negative_source"].value_counts(sort=False).items()
    }


def build_effectiveness_bundle(
        counterfactual: pd.DataFrame,
        real_q01: pd.DataFrame,
        design: EffectivenessDesign | None = None,
) -> EffectivenessBundle:
    """Freeze nested outer folds for real-only versus augmented training."""
    design = design or EffectivenessDesign()
    cf, cf_audit = _prepare_counterfactual(counterfactual)
    real_input, real_audit = _prepare_real(real_q01, design.qvalue_max)
    merged = pd.concat([cf, real_input], ignore_index=True, sort=False)
    _assign_sample_ids(merged)
    grouping_frame = merged.copy()
    orphan_graph_bridges = grouping_frame[_ORPHAN].astype(bool)
    # query_id is a row identity, not a family edge. An orphan query cannot
    # resolve its parent by definition, so treating it as a query would make
    # the generic linkage audit fail even though it is excluded from every
    # model. Preserve its sequence and all family tokens as graph bridges.
    if "query_id" in grouping_frame:
        grouping_frame.loc[orphan_graph_bridges, "query_id"] = pd.NA
    group_col, grouping_audit = assign_leakage_groups(
        grouping_frame, "peptide_group_id")
    if group_col != _GROUP_COL:
        raise AssertionError(f"expected {_GROUP_COL}, got {group_col}")
    merged[_GROUP_COL] = grouping_frame[_GROUP_COL]
    grouping_audit["n_orphan_graph_bridge_rows"] = int(
        orphan_graph_bridges.sum())
    grouping_audit["orphan_query_ids_ignored_as_row_identities"] = True
    if not grouping_audit["candidate_family_leakage_protected"]:
        raise ValueError(
            "counterfactual family metadata is incomplete after excluding "
            "orphan query identities")

    cohort, cohort_audit = apply_training_cohort(
        merged, design.cohort, target_col="label")
    real = cohort.loc[cohort[_ORIGIN].eq("real_q01")].copy().reset_index(drop=True)
    synthetic_diagnostics = cohort.loc[
        cohort["negative_source"].isin(SYNTHETIC_SOURCES)].copy()
    orphan_after_cohort = synthetic_diagnostics[_ORPHAN].astype(bool)
    synthetic_pool = synthetic_diagnostics.loc[~orphan_after_cohort].copy()
    if not set(real["label"].unique()) == {0, 1}:
        raise ValueError("real evaluation cohort must contain both classes")
    group_labels = real.groupby(_GROUP_COL)["label"].nunique()
    mixed_groups = set(group_labels[group_labels.gt(1)].index.astype(str))
    if mixed_groups:
        raise ValueError(
            f"real q01 contains {len(mixed_groups)} mixed-label connected groups")

    real_correct_groups = set(real.loc[real["label"].eq(1), _GROUP_COL].astype(str))
    synthetic_linked = synthetic_pool[_GROUP_COL].astype(str).isin(
        real_correct_groups)
    n_unlinked = int((~synthetic_linked).sum())
    synthetic = synthetic_pool.loc[synthetic_linked].copy().reset_index(drop=True)
    missing_sources = sorted(
        set(SYNTHETIC_SOURCES) - set(synthetic["negative_source"]))
    if missing_sources:
        raise ValueError(
            "no eligible synthetic training candidates remain for sources: "
            f"{missing_sources}")
    selected = _select_candidates(synthetic, design.seed)

    real[_OUTER_FOLD] = _assign_outer_folds(real, design)
    group_to_fold = (real[[_GROUP_COL, _OUTER_FOLD]].drop_duplicates()
                     .set_index(_GROUP_COL)[_OUTER_FOLD])
    synthetic[_OUTER_FOLD] = synthetic[_GROUP_COL].map(group_to_fold)
    selected[_OUTER_FOLD] = selected[_GROUP_COL].map(group_to_fold)
    synthetic_diagnostics[_OUTER_FOLD] = synthetic_diagnostics[_GROUP_COL].map(
        group_to_fold)
    if synthetic[_OUTER_FOLD].isna().any() or selected[_OUTER_FOLD].isna().any():
        raise AssertionError("a retained synthetic candidate has no real outer fold")
    synthetic[_OUTER_FOLD] = synthetic[_OUTER_FOLD].astype(int)
    selected[_OUTER_FOLD] = selected[_OUTER_FOLD].astype(int)

    real[_ROLE] = "real_q01"
    real[_SELECTED] = False
    selected_ids = set(selected[_SAMPLE_ID])
    linked_ids = set(synthetic[_SAMPLE_ID])
    synthetic_diagnostics[_ROLE] = "unlinked_graph_bridge"
    synthetic_diagnostics.loc[
        synthetic_diagnostics[_ORPHAN].astype(bool), _ROLE,
    ] = "orphan_graph_bridge"
    synthetic_diagnostics.loc[
        synthetic_diagnostics[_SAMPLE_ID].isin(linked_ids), _ROLE,
    ] = "synthetic_diagnostic"
    synthetic_diagnostics[_SELECTED] = synthetic_diagnostics[
        _SAMPLE_ID].isin(selected_ids)
    synthetic_diagnostics.loc[
        synthetic_diagnostics[_SELECTED], _ROLE,
    ] = "synthetic_training_candidate"
    selected[_ROLE] = "synthetic_training_candidate"
    selected[_SELECTED] = True

    fold_counts = {}
    for fold in range(design.outer_folds):
        test = real.loc[real[_OUTER_FOLD].eq(fold)]
        train = real.loc[real[_OUTER_FOLD].ne(fold)]
        test_groups = set(test[_GROUP_COL])
        if test_groups & set(train[_GROUP_COL]):
            raise AssertionError(f"outer fold {fold} leaks a real group")
        fold_synthetic = selected.loc[selected[_OUTER_FOLD].ne(fold)]
        if test_groups & set(fold_synthetic[_GROUP_COL]):
            raise AssertionError(f"outer fold {fold} leaks a synthetic family")
        if not set(test["label"].unique()) == {0, 1}:
            raise ValueError(f"outer fold {fold} lacks one real test class")
        fold_counts[str(fold)] = {
            "train_real_by_source": _source_counts(train),
            "test_real_by_source": _source_counts(test),
            "train_selected_synthetic_by_source": _source_counts(fold_synthetic),
            "n_train_groups": int(train[_GROUP_COL].nunique()),
            "n_test_groups": int(test[_GROUP_COL].nunique()),
            "n_overlapping_groups": 0,
        }

    manifest = pd.concat(
        [real, synthetic_diagnostics], ignore_index=True, sort=False)[[
        _SAMPLE_ID, _ORIGIN, _SOURCE_ROW, _ROLE, _SELECTED, _OUTER_FOLD,
        _ORPHAN, "label", "negative_source", "sequence", "parent_id",
        "query_id", "peptide_group_id", _GROUP_COL,
    ]].sort_values(_SAMPLE_ID, kind="mergesort").reset_index(drop=True)
    audit = {
        "schema": "counterfactual_real_q01_nested_effectiveness_v1",
        "metric_semantics": METRIC_SEMANTICS_VERSION,
        "positive_class": "incorrect_identification",
        "storage_convention": "label=1 correct identification; label=0 incorrect identification",
        "hypothesis": (
            "adding counterfactual negatives improves detection of held-out "
            "real q<=1% entrapment errors beyond real-only training"),
        "design": {
            "outer_folds": design.outer_folds,
            "outer_training_fraction": 1.0 - 1.0 / design.outer_folds,
            "outer_test_fraction": 1.0 / design.outer_folds,
            "outer_split_unit": "connected peptide/candidate family",
            "outer_test_rows": "real q01 correct and entrapment only",
            "counterfactual_parents_are_graph_bridges_only": True,
            "inner_training": (
                "frozen grouped CV and early-stop partitions with per-member "
                "OOF thresholds"),
            "cohort": design.cohort,
            "seed": design.seed,
            "models": {model: list(sources)
                       for model, sources in MODEL_SOURCES.items()},
        },
        "inputs": {"counterfactual": cf_audit, "real_q01": real_audit},
        "grouping": grouping_audit,
        "cohort": cohort_audit,
        "validation": {
            "n_mixed_label_real_groups": 0,
            "orphan_synthetic_rows_retained_during_graphing": True,
            "n_unlinked_synthetic_rows_dropped": n_unlinked,
            "every_real_row_is_tested_once": True,
            "every_outer_fold_has_zero_train_test_group_overlap": True,
            "test_contains_no_synthetic_rows": True,
        },
        "counts": {
            "real_evaluation_by_source": _source_counts(real),
            "real_groups_by_source": {
                source: int(part[_GROUP_COL].nunique()) for source, part in
                real.groupby("negative_source", sort=True)
            },
            "eligible_synthetic_by_source": _source_counts(synthetic),
            "selected_synthetic_by_source": _source_counts(selected),
            "orphan_after_cohort_by_source": _source_counts(
                synthetic_diagnostics.loc[orphan_after_cohort]),
            "outer_folds": fold_counts,
        },
        "analysis": {
            "primary_target_fpr": 0.05,
            "bootstrap_reps": design.bootstrap_reps,
            "bootstrap_seed": design.bootstrap_seed,
            "familywise_alpha": design.familywise_alpha,
            "resampling_unit": _GROUP_COL,
            "resampling_stratified_by_actual_class": True,
            "minimum_recall_gain": design.minimum_recall_gain,
            "max_fpr_increase": design.max_fpr_increase,
            "success_rule": (
                "Bonferroni familywise recall lower bound > 0, observed recall "
                "gain >= minimum, and familywise FPR upper bound <= maximum"),
        },
    }
    return EffectivenessBundle(
        real_rows=real.sort_values(_SAMPLE_ID, kind="mergesort").reset_index(drop=True),
        selected_synthetic=selected.sort_values(
            _SAMPLE_ID, kind="mergesort").reset_index(drop=True),
        synthetic_diagnostics=synthetic_diagnostics.sort_values(
            _SAMPLE_ID, kind="mergesort").reset_index(drop=True),
        manifest=manifest,
        audit=audit,
        design=design,
    )


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _atomic_json(path: Path, value: object) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    tmp = path.with_name(f"{path.name}.tmp.{os.getpid()}")
    with tmp.open("w", encoding="utf-8") as handle:
        json.dump(value, handle, indent=2, ensure_ascii=False)
    os.replace(tmp, path)


def _atomic_csv(path: Path, frame: pd.DataFrame) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    tmp = path.with_name(f"{path.name}.tmp.{os.getpid()}")
    frame.to_csv(tmp, index=False)
    os.replace(tmp, path)


def _provenance(path: str | os.PathLike[str]) -> dict[str, object]:
    resolved = Path(path).resolve()
    return {
        "path": str(resolved),
        "size_bytes": resolved.stat().st_size,
        "sha256": _sha256(resolved),
    }


def verify_effectiveness_bundle(
        output_root: str | os.PathLike[str]) -> dict[str, object]:
    """Fail closed if a frozen split/config artifact changed after build."""
    root = Path(output_root).resolve()
    status_path = root / "bundle_status.json"
    checksum_path = root / "artifact_checksums.json"
    if not status_path.is_file() or not checksum_path.is_file():
        raise FileNotFoundError(
            f"effectiveness bundle lacks status/checksum files: {root}")
    status = json.loads(status_path.read_text(encoding="utf-8"))
    if status.get("metric_semantics") != METRIC_SEMANTICS_VERSION or status.get(
            "positive_class") != "incorrect_identification":
        raise ValueError("effectiveness bundle has incompatible metric semantics")
    checksums = json.loads(checksum_path.read_text(encoding="utf-8"))
    if checksums.get("algorithm") != "sha256" or not isinstance(
            checksums.get("artifacts"), dict):
        raise ValueError("unsupported effectiveness checksum manifest")
    verified = 0
    for relative, expected in checksums["artifacts"].items():
        path = (root / relative).resolve()
        try:
            path.relative_to(root)
        except ValueError as exc:
            raise ValueError(
                f"checksum artifact escapes the bundle root: {relative}") from exc
        if not path.is_file():
            raise FileNotFoundError(f"frozen effectiveness artifact is missing: {path}")
        actual = _sha256(path)
        if actual != expected:
            raise ValueError(f"frozen effectiveness artifact changed: {path}")
        verified += 1
    return {
        "status": status.get("status"),
        "verified_artifacts": verified,
        "algorithm": "sha256",
    }


def _cv_config(template: Mapping[str, object], final_root: Path,
               fold: int, model: str) -> dict:
    cfg = copy.deepcopy(template)
    if not isinstance(cfg.get("data"), dict) or not isinstance(
            cfg.get("training"), dict):
        raise ValueError("training template requires data and training mappings")
    inner_folds = int(cfg["training"].get("cv_folds", 5))
    if not 2 <= inner_folds <= 5:
        raise ValueError("effectiveness template cv_folds must be between 2 and 5")
    fold_root = final_root / "folds" / f"fold_{fold}"
    train_files = [str(fold_root / "train_real_q01.csv")]
    for source in MODEL_SOURCES[model]:
        train_files.append(str(
            fold_root / f"train_synthetic_{SOURCE_FILE_STEMS[source]}.csv"))
    cfg["data"].update({
        "train_files": train_files,
        "test_files": [str(fold_root / "test_real_q01.csv")],
        "feature_cols": [],
        "target_col": "label",
        "feature_arm": "ms1_ms2_no_prediction",
        "cohort": "evidence_observed",
        "group_col": _GROUP_COL,
        "frozen_group_graph": True,
        "predefined_cv_fold_col": _INNER_FOLD,
        "predefined_inner_valid_cols": {
            fold_id: _inner_valid_column(fold_id)
            for fold_id in range(inner_folds)
        },
        "require_complete_arm": True,
        "nested_effectiveness_contract": {
            "outer_fold": fold,
            "outer_test": "real_q01_only",
            "augmentation_sources": list(MODEL_SOURCES[model]),
        },
    })
    cfg["operating_point"] = {
        "target_fprs": [0.01, 0.05, 0.10],
        "primary_target_fpr": 0.05,
    }
    result_root = final_root / "training" / f"fold_{fold}" / model
    cfg["output"] = {
        "model_path": str(result_root / "models" / "cv.txt"),
        "result_path": str(result_root / "training.cv.json"),
    }
    cfg["evaluation_semantics"] = {
        "metric_semantics": METRIC_SEMANTICS_VERSION,
        "positive_class": "incorrect_identification",
        "stored_label": "1=correct_identification, 0=incorrect_identification",
        "model_score": "trust_score=P(correct_identification)",
        "metric_score": "error_score=1-trust_score",
        "external_threshold_contract": (
            "per-member inner-OOF thresholds followed by majority vote"),
    }
    return cfg


def write_effectiveness_bundle(
        bundle: EffectivenessBundle,
        output_root: str | os.PathLike[str],
        training_template: str | os.PathLike[str],
        *,
        source_files: Mapping[str, str | os.PathLike[str]] | None = None,
        experiment_config: str | os.PathLike[str] | None = None,
) -> Path:
    """Atomically publish all outer-fold inputs and generated CV configs."""
    root = Path(output_root).resolve()
    if root.exists():
        raise FileExistsError(f"refusing to overwrite effectiveness bundle: {root}")
    template_path = Path(training_template).resolve()
    with template_path.open(encoding="utf-8") as handle:
        template = yaml.safe_load(handle)
    if not isinstance(template, dict):
        raise ValueError("training template must be a YAML mapping")
    root.parent.mkdir(parents=True, exist_ok=True)
    staging = Path(tempfile.mkdtemp(prefix=f".{root.name}.staging.", dir=root.parent))
    try:
        bundle.manifest.to_csv(staging / "outer_fold_manifest.csv", index=False)
        bundle.synthetic_diagnostics.to_csv(
            staging / "synthetic_diagnostics.csv", index=False)
        inner_protocols = {}
        for fold in range(bundle.design.outer_folds):
            fold_dir = staging / "folds" / f"fold_{fold}"
            config_dir = staging / "configs" / f"fold_{fold}"
            fold_dir.mkdir(parents=True)
            config_dir.mkdir(parents=True)
            real_train = bundle.real_rows.loc[
                bundle.real_rows[_OUTER_FOLD].ne(fold)].copy()
            real_test = bundle.real_rows.loc[
                bundle.real_rows[_OUTER_FOLD].eq(fold)].copy()
            fold_synthetic = bundle.selected_synthetic.loc[
                bundle.selected_synthetic[_OUTER_FOLD].ne(fold)].copy()
            real_train, fold_synthetic, protocol_audit = _attach_inner_protocol(
                real_train, fold_synthetic, template["training"])
            inner_protocols[str(fold)] = protocol_audit
            real_train.to_csv(fold_dir / "train_real_q01.csv", index=False)
            real_test.to_csv(fold_dir / "test_real_q01.csv", index=False)
            for source, stem in SOURCE_FILE_STEMS.items():
                candidates = fold_synthetic.loc[
                    fold_synthetic["negative_source"].eq(source)]
                candidates.to_csv(
                    fold_dir / f"train_synthetic_{stem}.csv", index=False)
            for model in MODEL_SOURCES:
                config = _cv_config(template, root, fold, model)
                with (config_dir / f"{model}.yaml").open(
                        "w", encoding="utf-8") as handle:
                    yaml.safe_dump(config, handle, sort_keys=False,
                                   allow_unicode=True)

        audit = copy.deepcopy(bundle.audit)
        audit["design"]["inner_protocol"] = {
            "shared_across_models": True,
            "outer_folds": inner_protocols,
        }
        audit["provenance"] = {
            "source_files": {
                name: _provenance(path)
                for name, path in (source_files or {}).items()
            },
            "experiment_config": (
                _provenance(experiment_config)
                if experiment_config is not None else None),
            "training_template": _provenance(template_path),
        }
        _atomic_json(staging / "split_audit.json", audit)
        artifacts = sorted(path for path in staging.rglob("*") if path.is_file())
        _atomic_json(staging / "artifact_checksums.json", {
            "algorithm": "sha256",
            "artifacts": {
                str(path.relative_to(staging)): _sha256(path) for path in artifacts
            },
        })
        _atomic_json(staging / "bundle_status.json", {
            "status": "prepared",
            "schema": bundle.audit["schema"],
            "metric_semantics": METRIC_SEMANTICS_VERSION,
            "positive_class": "incorrect_identification",
        })
        os.replace(staging, root)
    except Exception:
        shutil.rmtree(staging, ignore_errors=True)
        raise
    return root


def _prediction_path(root: Path, fold: int, model: str) -> Path:
    return (root / "training" / f"fold_{fold}" / model /
            "training.cv.test_scores.csv")


def _training_result_path(root: Path, fold: int, model: str) -> Path:
    return root / "training" / f"fold_{fold}" / model / "training.cv.json"


def _locked_vote_metrics(stored_labels, vote_fraction,
                         weights: np.ndarray | None = None) -> dict[str, object]:
    vote = np.asarray(vote_fraction, dtype="f8")
    if (not np.isfinite(vote).all() or (vote < 0.0).any()
            or (vote > 1.0).any()):
        raise ValueError("error vote fractions must be finite and in [0, 1]")
    return evaluate_at_threshold(
        stored_labels, 1.0 - vote, error_threshold=0.5,
        sample_weight=weights)


def _validate_training_result(root: Path, fold: int, model: str,
                              predictions: pd.DataFrame) -> None:
    result_path = _training_result_path(root, fold, model)
    if not result_path.is_file():
        raise FileNotFoundError(f"missing completed training result: {result_path}")
    result = json.loads(result_path.read_text(encoding="utf-8"))
    if (result.get("metric_semantics") != METRIC_SEMANTICS_VERSION
            or result.get("positive_class") != "incorrect_identification"
            or result.get("mode") != "cross_test"):
        raise ValueError(f"incompatible cross-test result: {result_path}")
    config_path = root / "configs" / f"fold_{fold}" / f"{model}.yaml"
    recorded_config = result.get("provenance", {}).get("config_sha256")
    if recorded_config != _sha256(config_path):
        raise ValueError(f"training result/config fingerprint mismatch: {result_path}")
    for target in (1, 5, 10):
        key = f"fpr_{target}"
        try:
            external = result["operating_points"][key]["external_ensemble"]
            recorded = external["test_metrics"]
        except KeyError as exc:
            raise ValueError(
                f"training result lacks locked {key} external metrics: "
                f"{result_path}") from exc
        if external.get("method") != "fold_calibrated_majority_vote":
            raise ValueError(f"training result has an unsafe {key} method")
        observed = _locked_vote_metrics(
            predictions["label"],
            predictions[f"{key}_error_vote_fraction"])
        for metric in (
                "n_actual_correct", "n_actual_error", "tp", "fp", "fn", "tn",
                "fpr", "fnr", "error_recall", "correct_recall"):
            left, right = recorded.get(metric), observed.get(metric)
            if left is None or right is None:
                equal = left is right
            elif isinstance(left, (int, float)) and isinstance(right, (int, float)):
                equal = bool(np.isclose(left, right, rtol=0.0, atol=1e-12))
            else:
                equal = left == right
            if not equal:
                raise ValueError(
                    f"training result/test-score {key}.{metric} mismatch: "
                    f"{result_path}")


def _load_pooled_predictions(root: Path, outer_folds: int, *,
                             models=None) -> pd.DataFrame:
    """Validate and pool matching model outputs; optional arms reuse this contract."""
    models = tuple(MODEL_SOURCES if models is None else models)
    if not models or len(set(models)) != len(models):
        raise ValueError("prediction model names must be nonempty and unique")
    fold_frames = []
    identity = [
        *_ROW_IDENTITY_COLUMNS, "negative_source", "sequence", "charge",
    ]
    for fold in range(outer_folds):
        merged = None
        expected_ids = None
        expected_identity = None
        for model in models:
            path = _prediction_path(root, fold, model)
            if not path.is_file():
                raise FileNotFoundError(f"missing trained test scores: {path}")
            frame = pd.read_csv(path)
            missing = [column for column in (
                *_ROW_IDENTITY_COLUMNS,
                "ensemble_trust_score", "fpr_1_error_vote_fraction",
                "fpr_5_error_vote_fraction", "fpr_10_error_vote_fraction",
            ) if column not in frame]
            if missing:
                raise ValueError(f"{path} lacks effectiveness columns: {missing}")
            if frame[_SAMPLE_ID].duplicated().any():
                raise ValueError(f"{path} contains duplicate effectiveness sample IDs")
            recorded_folds = pd.to_numeric(frame[_OUTER_FOLD], errors="coerce")
            if recorded_folds.isna().any() or not recorded_folds.eq(fold).all():
                raise ValueError(f"{path} contains rows from the wrong outer fold")
            ids = set(frame[_SAMPLE_ID])
            row_identity = set(zip(
                frame[_SAMPLE_ID].astype(str),
                recorded_folds.astype(int),
                frame[_GROUP_COL].astype(str),
                pd.to_numeric(frame["label"], errors="coerce"),
            ))
            if expected_ids is None:
                expected_ids = ids
                expected_identity = row_identity
            elif ids != expected_ids:
                raise ValueError(f"model test membership differs in outer fold {fold}")
            elif row_identity != expected_identity:
                raise ValueError(f"model test identity differs in outer fold {fold}")
            trust = pd.to_numeric(frame["ensemble_trust_score"], errors="coerce")
            if (not np.isfinite(trust).all() or (trust < 0.0).any()
                    or (trust > 1.0).any()):
                raise ValueError(f"{path} contains invalid ensemble trust scores")
            _validate_training_result(root, fold, model, frame)
            score_columns = {
                "ensemble_trust_score": f"{model}_trust_score",
                "fpr_1_error_vote_fraction": f"{model}_fpr_1_vote",
                "fpr_5_error_vote_fraction": f"{model}_fpr_5_vote",
                "fpr_10_error_vote_fraction": f"{model}_fpr_10_vote",
            }
            if merged is None:
                available_identity = [column for column in identity if column in frame]
                merged = frame[available_identity + list(score_columns)].rename(
                    columns=score_columns)
                merged[_OUTER_FOLD] = fold
            else:
                merged = merged.merge(
                    frame[[_SAMPLE_ID, *score_columns]].rename(columns=score_columns),
                    on=_SAMPLE_ID, validate="one_to_one")
        fold_frames.append(merged)
    pooled = pd.concat(fold_frames, ignore_index=True)
    if pooled[_SAMPLE_ID].duplicated().any():
        raise ValueError("a real sample was evaluated in more than one outer fold")
    return pooled.sort_values(_SAMPLE_ID, kind="mergesort").reset_index(drop=True)


def _model_metrics(pooled: pd.DataFrame, model: str,
                   weights: np.ndarray | None = None) -> dict[str, object]:
    labels = pooled["label"].to_numpy()
    trust = pooled[f"{model}_trust_score"].to_numpy()
    ranking = evaluate_ranking(labels, trust, sample_weight=weights)
    result = {
        "roc_auc": ranking["roc_auc"],
        "error_pr_auc": ranking["error_pr_auc"],
        "operating_points": {},
    }
    for target in (1, 5, 10):
        vote = pooled[f"{model}_fpr_{target}_vote"].to_numpy()
        point = _locked_vote_metrics(labels, vote, weights=weights)
        result[f"error_recall_at_fpr{target}"] = point["error_recall"]
        result[f"fnr_at_fpr{target}"] = point["fnr"]
        result[f"observed_fpr_at_fpr{target}"] = point["fpr"]
        result["operating_points"][f"fpr_{target}"] = {
            "metric_semantics": METRIC_SEMANTICS_VERSION,
            "positive_class": "incorrect_identification",
            "target_fpr": target / 100.0,
            "external_ensemble": {
                "method": "fold_calibrated_majority_vote",
                "calibration_source": "each_member_inner_oof_fold",
                "vote_error_threshold": 0.5,
                "test_metrics": point,
            },
        }
    return result


def _primary_locked_metrics(pooled: pd.DataFrame, model: str,
                            weights: np.ndarray | None = None) -> dict[str, float]:
    vote = pooled[f"{model}_fpr_5_vote"].to_numpy()
    point = _locked_vote_metrics(
        pooled["label"].to_numpy(), vote, weights=weights)
    return {
        "error_recall_at_fpr5": point["error_recall"],
        "observed_fpr_at_fpr5": point["fpr"],
    }


def _paired_bootstrap(pooled: pd.DataFrame, design: EffectivenessDesign) -> pd.DataFrame:
    group_codes, group_values = pd.factorize(pooled[_GROUP_COL], sort=True)
    group_table = pd.DataFrame({
        "code": group_codes, "label": pooled["label"].to_numpy(),
    })
    if group_table.groupby("code")["label"].nunique().gt(1).any():
        raise ValueError("paired bootstrap requires class-pure connected groups")
    group_labels = group_table.drop_duplicates("code").set_index("code")["label"]
    group_codes_by_class = {
        label: group_labels[group_labels.eq(label)].index.to_numpy(dtype=int)
        for label in (0, 1)
    }
    comparisons = [("m_real", model) for model in MODEL_SOURCES if model != "m_real"]
    # Each confirmatory comparison makes two one-sided claims: recall gain and
    # FPR non-inferiority. Bonferroni therefore allocates alpha over 2*m bounds.
    adjusted_tail = design.familywise_alpha / (2 * len(comparisons))
    observed = {
        model: _primary_locked_metrics(pooled, model) for model in MODEL_SOURCES
    }
    samples = {
        pair: {"error_recall_at_fpr5": [], "observed_fpr_at_fpr5": []}
        for pair in comparisons
    }
    rng = np.random.default_rng(design.bootstrap_seed)
    completed = 0
    for _ in range(design.bootstrap_reps):
        group_weights = np.zeros(len(group_values), dtype="f8")
        for label in (0, 1):
            codes = group_codes_by_class[label]
            sampled = rng.choice(codes, size=len(codes), replace=True)
            group_weights += np.bincount(
                sampled, minlength=len(group_values)).astype("f8")
        weights = group_weights[group_codes]
        replicate = {
            model: _primary_locked_metrics(pooled, model, weights=weights)
            for model in MODEL_SOURCES
        }
        for pair in comparisons:
            baseline, augmented = pair
            for metric in samples[pair]:
                samples[pair][metric].append(
                    replicate[augmented][metric] - replicate[baseline][metric])
        completed += 1

    rows = []
    for baseline, augmented in comparisons:
        recall_values = np.asarray(
            samples[(baseline, augmented)]["error_recall_at_fpr5"])
        fpr_values = np.asarray(
            samples[(baseline, augmented)]["observed_fpr_at_fpr5"])
        recall_delta = (
            observed[augmented]["error_recall_at_fpr5"]
            - observed[baseline]["error_recall_at_fpr5"])
        fpr_delta = (
            observed[augmented]["observed_fpr_at_fpr5"]
            - observed[baseline]["observed_fpr_at_fpr5"])
        recall_low, recall_high = np.quantile(recall_values, [0.025, 0.975])
        fpr_low, fpr_high = np.quantile(fpr_values, [0.025, 0.975])
        recall_familywise_low = float(np.quantile(
            recall_values, adjusted_tail))
        fpr_familywise_high = float(np.quantile(
            fpr_values, 1.0 - adjusted_tail))
        rows.append({
            "metric_semantics": METRIC_SEMANTICS_VERSION,
            "positive_class": "incorrect_identification",
            "baseline_model": baseline,
            "augmented_model": augmented,
            "error_recall_delta_at_fpr5": float(recall_delta),
            "error_recall_delta_ci95_low": float(recall_low),
            "error_recall_delta_ci95_high": float(recall_high),
            "observed_fpr_delta_at_fpr5": float(fpr_delta),
            "observed_fpr_delta_ci95_low": float(fpr_low),
            "observed_fpr_delta_ci95_high": float(fpr_high),
            "recall_gain_familywise_lower_bound": recall_familywise_low,
            "fpr_increase_familywise_upper_bound": fpr_familywise_high,
            "familywise_alpha": design.familywise_alpha,
            "multiplicity_adjustment": (
                "bonferroni_over_4_comparisons_x_2_one_sided_bounds"),
            "minimum_recall_gain": design.minimum_recall_gain,
            "max_fpr_increase": design.max_fpr_increase,
            "effectiveness_supported": bool(
                recall_delta >= design.minimum_recall_gain
                and recall_familywise_low > 0.0
                and fpr_familywise_high <= design.max_fpr_increase),
            "n_bootstrap": completed,
            "resampling_unit": _GROUP_COL,
        })
    return pd.DataFrame(rows)


def summarize_effectiveness(output_root: str | os.PathLike[str]) -> dict:
    """Pool strict outer-fold predictions and run paired group bootstrap."""
    root = Path(output_root).resolve()
    verification = verify_effectiveness_bundle(root)
    audit = json.loads((root / "split_audit.json").read_text(encoding="utf-8"))
    analysis = audit["analysis"]
    design = EffectivenessDesign(
        outer_folds=int(audit["design"]["outer_folds"]),
        seed=int(audit["design"]["seed"]),
        cohort=str(audit["design"]["cohort"]),
        qvalue_max=float(audit["inputs"]["real_q01"]["configured_qvalue_max"]),
        bootstrap_reps=int(analysis["bootstrap_reps"]),
        bootstrap_seed=int(analysis["bootstrap_seed"]),
        familywise_alpha=float(analysis["familywise_alpha"]),
        max_fpr_increase=float(analysis["max_fpr_increase"]),
        minimum_recall_gain=float(analysis["minimum_recall_gain"]),
    )
    pooled = _load_pooled_predictions(root, design.outer_folds)
    expected_real = pd.read_csv(root / "outer_fold_manifest.csv")
    expected_ids = set(expected_real.loc[
        expected_real[_ROLE].eq("real_q01"), _SAMPLE_ID])
    if set(pooled[_SAMPLE_ID]) != expected_ids:
        raise ValueError("pooled predictions do not cover every real row exactly once")
    manifest_identity = expected_real.loc[
        expected_real[_ROLE].eq("real_q01"),
        list(_ROW_IDENTITY_COLUMNS)]
    observed_identity = pooled[list(_ROW_IDENTITY_COLUMNS)]
    identity_check = manifest_identity.merge(
        observed_identity, on=_SAMPLE_ID, how="outer",
        suffixes=("_manifest", "_prediction"), validate="one_to_one")
    for column in (_OUTER_FOLD, "label"):
        manifest_values = pd.to_numeric(
            identity_check[f"{column}_manifest"], errors="coerce")
        prediction_values = pd.to_numeric(
            identity_check[f"{column}_prediction"], errors="coerce")
        if (manifest_values.isna().any() or prediction_values.isna().any()
                or not np.array_equal(manifest_values.to_numpy(),
                                      prediction_values.to_numpy())):
            raise ValueError(
                f"pooled prediction {column} differs from the frozen manifest")
    if not identity_check[f"{_GROUP_COL}_manifest"].astype(str).equals(
            identity_check[f"{_GROUP_COL}_prediction"].astype(str)):
        raise ValueError(
            f"pooled prediction {_GROUP_COL} differs from the frozen manifest")
    model_metrics = {model: _model_metrics(pooled, model)
                     for model in MODEL_SOURCES}
    summary_rows = []
    for model, metrics in model_metrics.items():
        row = {
            "metric_semantics": METRIC_SEMANTICS_VERSION,
            "positive_class": "incorrect_identification",
            "model": model,
            "n_actual_correct": int(pooled["label"].eq(1).sum()),
            "n_actual_error": int(pooled["label"].eq(0).sum()),
        }
        row.update({key: value for key, value in metrics.items()
                    if key != "operating_points"})
        summary_rows.append(row)
    bootstrap = _paired_bootstrap(pooled, design)
    pooled_output = pooled.copy()
    pooled_output.insert(0, "positive_class", "incorrect_identification")
    pooled_output.insert(0, "metric_semantics", METRIC_SEMANTICS_VERSION)
    _atomic_csv(root / "pooled_real_test_predictions.csv", pooled_output)
    _atomic_csv(root / "effectiveness_summary.csv", pd.DataFrame(summary_rows))
    _atomic_csv(root / "paired_group_bootstrap.csv", bootstrap)
    summary = {
        "schema": "counterfactual_real_q01_effectiveness_summary_v1",
        "metric_semantics": METRIC_SEMANTICS_VERSION,
        "positive_class": "incorrect_identification",
        "n_actual_correct": int(pooled["label"].eq(1).sum()),
        "n_actual_error": int(pooled["label"].eq(0).sum()),
        "bundle_verification": verification,
        "models": model_metrics,
        "comparisons": bootstrap.to_dict(orient="records"),
        "interpretation": (
            "effectiveness_supported is evidence of incremental practical "
            "utility over real-only training under this 2Da q01 domain; it "
            "uses familywise-adjusted inference but does not isolate "
            "augmentation count from candidate diversity"),
    }
    _atomic_json(root / "effectiveness_summary.json", summary)
    _atomic_json(root / "bundle_status.json", {
        "status": "complete",
        "schema": summary["schema"],
        "metric_semantics": METRIC_SEMANTICS_VERSION,
        "positive_class": "incorrect_identification",
    })
    return summary


def _load_config(path: str | os.PathLike[str]) -> tuple[EffectivenessDesign, Path]:
    config_path = Path(path).resolve()
    with config_path.open(encoding="utf-8") as handle:
        config = yaml.safe_load(handle)
    if not isinstance(config, dict) or config.get("schema") != (
            "counterfactual_real_q01_effectiveness_config_v1"):
        raise ValueError("unsupported effectiveness config schema")
    split = config.get("outer_evaluation", {})
    analysis = config.get("analysis", {})
    design = EffectivenessDesign(
        outer_folds=int(split.get("folds", 5)),
        seed=int(split.get("seed", 42)),
        cohort=str(config.get("cohort", "evidence_observed")),
        qvalue_max=float(config.get("real_qvalue_max", 0.01)),
        bootstrap_reps=int(analysis.get("bootstrap_reps", 1000)),
        bootstrap_seed=int(analysis.get("bootstrap_seed", 20260908)),
        familywise_alpha=float(analysis.get("familywise_alpha", 0.05)),
        max_fpr_increase=float(analysis.get("max_fpr_increase", 0.01)),
        minimum_recall_gain=float(analysis.get("minimum_recall_gain", 0.03)),
    )
    template = (config_path.parent / config["training_template"]).resolve()
    return design, template


def _parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description="nested real-q01 counterfactual effectiveness experiment")
    commands = parser.add_subparsers(dest="command", required=True)
    build = commands.add_parser("build")
    build.add_argument("--config", required=True)
    build.add_argument("--counterfactual-features", required=True)
    build.add_argument("--real-features", required=True)
    build.add_argument("--output-root", required=True)
    summarize = commands.add_parser("summarize")
    summarize.add_argument("--output-root", required=True)
    verify = commands.add_parser("verify")
    verify.add_argument("--output-root", required=True)
    return parser


def main(argv: list[str] | None = None):
    args = _parser().parse_args(argv)
    if args.command == "verify":
        result = verify_effectiveness_bundle(args.output_root)
        print(
            "effectiveness bundle verified: "
            f"{result['verified_artifacts']} artifacts")
        return result
    if args.command == "summarize":
        summary = summarize_effectiveness(args.output_root)
        print(f"effectiveness summary complete: {args.output_root}")
        return summary
    design, template = _load_config(args.config)
    counterfactual = pd.read_csv(args.counterfactual_features)
    real_q01 = pd.read_csv(args.real_features)
    bundle = build_effectiveness_bundle(counterfactual, real_q01, design)
    root = write_effectiveness_bundle(
        bundle, args.output_root, template,
        source_files={
            "counterfactual_features": args.counterfactual_features,
            "real_q01_features": args.real_features,
        },
        experiment_config=args.config)
    print(f"effectiveness bundle prepared: {root}")
    return root


if __name__ == "__main__":
    main()
