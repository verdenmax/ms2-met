"""Stable sample-identity contract shared by tabular and signal datasets.

Identity hashes are schema, not an implementation detail: frozen protocol
bundles and every downstream model must agree byte-for-byte.  Keep the
length-prefixed serialization and dataset namespace in this single module.
"""

from __future__ import annotations

import hashlib
from collections.abc import Iterable, Sequence

import pandas as pd


LOCAL_SAMPLE_ID_ALGORITHM = "sha256_length_prefixed_identity_fields"
COMBINED_SAMPLE_ID_ALGORITHM = "sha256(dataset|local_sample_id)"


def identity_candidates(common_columns: Iterable[str]) -> list[list[str]]:
    """Return supported identity schemas in strict preference order."""
    common = set(common_columns)
    candidates: list[list[str]] = []
    for singleton in ("query_id", "parent_id"):
        if singleton in common:
            candidates.append([singleton])
    base = [
        column for column in (
            "sequence", "charge", "precursor_mz", "rt", "raw_title1",
            "raw_title2", "label_type",
        )
        if column in common
    ]
    if "sequence" in base and "charge" in base:
        candidates.append(base)
        extras = [column for column in ("protein_names", "q_value")
                  if column in common]
        for count in range(1, len(extras) + 1):
            candidates.append(base + extras[:count])
    return candidates


def identity_text(frame: pd.DataFrame,
                  columns: Sequence[str]) -> pd.Series:
    """Serialize identity fields without delimiter ambiguity.

    Values intentionally use their stored textual representation.  Changing
    float formatting here would invalidate every existing frozen sample ID.
    """
    missing = [column for column in columns if column not in frame]
    if missing:
        raise ValueError(f"identity columns are missing: {missing}")
    encoded = pd.Series("", index=frame.index, dtype=object)
    for column in columns:
        values = frame[column].astype(str)
        encoded = encoded + values.str.len().astype(str) + ":" + values + "|"
    return encoded


def local_sample_ids(frame: pd.DataFrame,
                     columns: Sequence[str]) -> pd.Series:
    """Hash rows using the frozen local sample-ID algorithm."""
    return identity_text(frame, columns).map(
        lambda value: hashlib.sha256(value.encode("utf-8")).hexdigest())


def combined_sample_id(dataset: str, local_sample_id: str) -> str:
    """Namespace one local ID for a multi-acquisition protocol."""
    payload = f"{dataset}|{local_sample_id}".encode("utf-8")
    return hashlib.sha256(payload).hexdigest()


def namespace_sample_ids(
    frame: pd.DataFrame,
    dataset: str,
    *,
    sample_id_col: str = "sample_id",
    source_sample_id_col: str = "source_sample_id",
    dataset_col: str = "dataset",
) -> pd.DataFrame:
    """Return a copy whose local IDs are namespaced by acquisition dataset."""
    if sample_id_col not in frame:
        raise ValueError(f"sample ID column {sample_id_col!r} is missing")
    result = frame.copy()
    source_ids = result[sample_id_col].astype(str)
    result[source_sample_id_col] = source_ids
    result[sample_id_col] = source_ids.map(
        lambda value: combined_sample_id(dataset, value))
    result[dataset_col] = dataset
    if result[sample_id_col].duplicated().any():
        raise ValueError(f"namespaced sample IDs are not unique for {dataset}")
    return result
