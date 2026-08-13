"""Compatibility adapter from legacy PSM JSON rows to frozen sample IDs."""

from __future__ import annotations

import hashlib
from collections import defaultdict
from collections.abc import Sequence

import numpy as np
import pandas as pd

from spectrum.psm_info import PSMInfo


_NUMERIC_COLUMNS = frozenset({"charge", "precursor_mz", "rt", "q_value"})
_ABS_TOLERANCE = {
    "charge": 0.0,
    # JSON and feature rows originate from float32 PSM fields.
    "precursor_mz": 5e-5,
    "rt": 5e-5,
    "q_value": 1e-10,
}
_PSM_ATTRIBUTES = {
    "sequence": "_sequence",
    "charge": "_charge",
    "precursor_mz": "_precursor_mz",
    "rt": "_rt",
    "raw_title1": "_raw_title",
    "protein_names": "_protein_names",
    "q_value": "_q_value",
    "label_type": "_label_type",
    "query_id": "_query_id",
    "parent_id": "_parent_id",
    "group_id": "_group_id",
    "pair_id": "_pair_id",
    "candidate_family_id": "_candidate_family_id",
    "peptide_group_id": "_peptide_group_id",
}


def select_pilot_rows(frame: pd.DataFrame, *, correct_per_dataset: int,
                      error_per_dataset: int, seed: int) -> pd.DataFrame:
    """Select a deterministic, row-order-independent integrity pilot."""
    required = {"sample_id", "dataset", "label"}
    missing = required - set(frame)
    if missing:
        raise ValueError(f"protocol frame lacks pilot columns: {sorted(missing)}")
    if correct_per_dataset <= 0 or error_per_dataset <= 0:
        raise ValueError("pilot class counts must be positive")
    selected = []
    for dataset in sorted(frame["dataset"].astype(str).unique()):
        domain = frame[frame["dataset"].astype(str).eq(dataset)]
        for label, count, name in (
            (1, correct_per_dataset, "correct"),
            (0, error_per_dataset, "incorrect"),
        ):
            candidates = domain[domain["label"].eq(label)].copy()
            if len(candidates) < count:
                raise ValueError(
                    f"dataset {dataset} has only {len(candidates)} {name} "
                    f"rows; pilot requires {count}")
            candidates["__pilot_rank"] = candidates["sample_id"].map(
                lambda sample_id: hashlib.sha256(
                    f"{seed}|{sample_id}".encode("utf-8")).hexdigest())
            selected.append(candidates.sort_values(
                ["__pilot_rank", "sample_id"]).head(count))
    result = pd.concat(selected, ignore_index=True).drop(
        columns="__pilot_rank")
    if result["sample_id"].duplicated().any():
        raise ValueError("pilot selection produced duplicate sample IDs")
    return result.sort_values(["dataset", "sample_id"]).reset_index(drop=True)


def _psm_value(psm: PSMInfo, column: str):
    attribute = _PSM_ATTRIBUTES.get(column)
    if attribute is None:
        raise ValueError(
            f"Phase 2 cannot recover identity column {column!r} from PSM JSON")
    return getattr(psm, attribute, None)


def _text(value) -> str:
    if value is None or (isinstance(value, float) and np.isnan(value)):
        return ""
    return str(value)


def _bucket_key(values: dict, columns: Sequence[str]) -> tuple[str, ...]:
    return tuple(_text(values[column]) for column in columns)


def _matches(row, psm: PSMInfo, identity_cols: Sequence[str]) -> bool:
    for column in identity_cols:
        left = row[column]
        right = _psm_value(psm, column)
        if column in _NUMERIC_COLUMNS:
            if pd.isna(left) or right is None or (
                    isinstance(right, float) and np.isnan(right)):
                if pd.isna(left) and (
                        right is None or pd.isna(right)):
                    continue
                return False
            if not np.isclose(
                    float(left), float(right), rtol=0.0,
                    atol=_ABS_TOLERANCE[column]):
                return False
        elif _text(left) != _text(right):
            return False
    return True


def match_psms_to_protocol(
    protocol_rows: pd.DataFrame,
    psms: Sequence[PSMInfo],
    identity_cols: Sequence[str],
) -> tuple[dict[str, PSMInfo], pd.DataFrame]:
    """Match selected frozen rows without relying on lossy float strings."""
    missing = [column for column in identity_cols if column not in protocol_rows]
    if missing:
        raise ValueError(f"protocol rows lack identity columns: {missing}")
    text_cols = [
        column for column in identity_cols if column not in _NUMERIC_COLUMNS
    ]
    buckets: dict[tuple[str, ...], list[int]] = defaultdict(list)
    for index, psm in enumerate(psms):
        values = {column: _psm_value(psm, column) for column in text_cols}
        buckets[_bucket_key(values, text_cols)].append(index)

    matched: dict[str, PSMInfo] = {}
    used: set[int] = set()
    audit = []
    for _, row in protocol_rows.iterrows():
        sample_id = str(row["sample_id"])
        values = {column: row[column] for column in text_cols}
        candidates = buckets.get(_bucket_key(values, text_cols), [])
        exact = [
            index for index in candidates
            if index not in used and _matches(row, psms[index], identity_cols)
        ]
        status = "matched" if len(exact) == 1 else (
            "unmatched" if not exact else "ambiguous")
        audit.append({
            "sample_id": sample_id,
            "dataset": row.get("dataset"),
            "status": status,
            "bucket_candidates": len(candidates),
            "exact_candidates": len(exact),
        })
        if len(exact) == 1:
            index = exact[0]
            used.add(index)
            matched[sample_id] = psms[index]
    return matched, pd.DataFrame(audit)
