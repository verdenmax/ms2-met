"""Shared connected grouping for sequences and derived candidate families."""
from __future__ import annotations

import hashlib

import numpy as np
import pandas as pd

_LEAKAGE_GROUP = "leakage_group_id"
RELATIONSHIP_COLUMNS = (
    "query_id", "group_id", "pair_id", "candidate_family_id",
    "peptide_group_id", "parent_id", "leakage_group_id",
)
FAMILY_RELATIONSHIP_COLUMNS = (
    "group_id", "candidate_family_id", "peptide_group_id", "parent_id",
)
ROOT_FAMILY_RELATIONSHIP_COLUMNS = (
    "group_id", "candidate_family_id", "peptide_group_id",
)


def nonempty_relation_columns(frame):
    available = []
    for column in RELATIONSHIP_COLUMNS:
        if column not in frame:
            continue
        values = frame[column].astype("string").str.strip()
        if values.notna().any() and values.fillna("").ne("").any():
            available.append(column)
    return available


def assign_leakage_groups(frame, base_group_col):
    """Group the connected components of sequence and candidate relations.

    A synthetic negative may have a different sequence from its parent.  A
    plain sequence split would therefore leak that family across partitions.
    When upstream relationship IDs are available, unioning their tokens with
    the sequence tokens makes every connected family one indivisible group.
    """
    if base_group_col not in frame:
        raise ValueError(
            f"split group column {base_group_col!r} is missing; "
            "refusing to run ungrouped CV")
    if frame[base_group_col].isna().any():
        raise ValueError(f"split group column {base_group_col!r} has nulls")
    relation_columns = nonempty_relation_columns(frame)
    if not relation_columns:
        return base_group_col, {
            "mode": "sequence_only",
            "base_group_col": base_group_col,
            "relationship_columns_available": [],
            "candidate_family_leakage_protected": False,
            "limitation": (
                "upstream feature rows do not contain pair/family IDs; only "
                "same-sequence leakage is prevented"),
        }

    family_columns = [
        column for column in FAMILY_RELATIONSHIP_COLUMNS
        if column in relation_columns
    ]
    family_values = frame[family_columns].astype("string")
    has_family_id = family_values.apply(
        lambda column: column.str.strip().fillna("").ne(""))
    row_has_family_id = (
        has_family_id.any(axis=1)
        if family_columns else pd.Series(False, index=frame.index)
    )
    complete_family_coverage = bool(row_has_family_id.all())

    # query_id/pair_id are often row-unique identifiers.  They are useful
    # graph tokens, but their mere presence cannot prove that a generated
    # candidate is connected to its parent.  When query rows exist, require
    # every declared parent token to occur on at least one other row in the
    # family-ID namespace.
    query_rows = (
        frame["query_id"].astype("string").str.strip().fillna("").ne("")
        if "query_id" in frame else pd.Series(False, index=frame.index)
    )
    root_columns = [
        column for column in ROOT_FAMILY_RELATIONSHIP_COLUMNS
        if column in family_columns
    ]
    root_token_rows = {}
    for row_position, values in enumerate(
            frame[root_columns].itertuples(index=False, name=None)):
        if query_rows.iloc[row_position]:
            continue
        for value in values:
            if pd.isna(value) or not str(value).strip():
                continue
            root_token_rows.setdefault(str(value).strip(), set()).add(
                row_position)
    unresolved_parent_rows = 0
    if "parent_id" in frame:
        parent_values = frame["parent_id"].astype("string").str.strip()
        for row_position in np.flatnonzero(query_rows.to_numpy()):
            value = parent_values.iloc[row_position]
            if pd.isna(value) or not value:
                unresolved_parent_rows += 1
                continue
            linked_rows = root_token_rows.get(str(value), set())
            if not linked_rows:
                unresolved_parent_rows += 1
    elif query_rows.any():
        unresolved_parent_rows = int(query_rows.sum())
    complete_family_linkage = (
        complete_family_coverage and unresolved_parent_rows == 0)
    n_rows = len(frame)
    parent = np.arange(n_rows, dtype=np.int64)
    rank = np.zeros(n_rows, dtype=np.int8)

    def find(index):
        while parent[index] != index:
            parent[index] = parent[parent[index]]
            index = parent[index]
        return int(index)

    def union(left, right):
        left, right = find(left), find(right)
        if left == right:
            return
        if rank[left] < rank[right]:
            left, right = right, left
        parent[right] = left
        if rank[left] == rank[right]:
            rank[left] += 1

    first_seen = {}
    token_columns = tuple(dict.fromkeys((
        base_group_col, *(('sequence',) if 'sequence' in frame else ()),
        *relation_columns)))
    for row_index, values in enumerate(
            frame[list(token_columns)].itertuples(index=False, name=None)):
        for column, value in zip(token_columns, values):
            if pd.isna(value) or not str(value).strip():
                continue
            # Relationship IDs deliberately share one namespace.  For
            # example, a parent row may expose ``group_id=P1`` while its
            # generated child exposes ``parent_id=P1``; those two rows still
            # belong to one indivisible candidate family.  Sequence remains
            # namespaced so an accidental text collision with an opaque ID
            # cannot join unrelated peptides.
            if column == "sequence":
                namespace = "sequence"
                value = str(value).strip().upper().replace("I", "L")
            elif column in {
                    "group_id", "candidate_family_id", "peptide_group_id",
                    "parent_id"}:
                namespace = "candidate_family"
            else:
                namespace = column
            token = f"{namespace}:{str(value).strip()}"
            previous = first_seen.setdefault(token, row_index)
            union(row_index, previous)

    component_tokens = {}
    for token, row_index in first_seen.items():
        root = find(row_index)
        current = component_tokens.get(root)
        if current is None or token < current:
            component_tokens[root] = token
    identifiers = [
        hashlib.sha256(
            f"connected_candidate_family_v2|{component_tokens[find(i)]}"
            .encode("utf-8")
        ).hexdigest()
        for i in range(n_rows)
    ]
    frame[_LEAKAGE_GROUP] = identifiers
    group_sizes = pd.Series(identifiers).value_counts()
    return _LEAKAGE_GROUP, {
        "mode": (
            "sequence_family_connected_components_v2"
            if complete_family_linkage else
            "sequence_family_connected_components_partial_ids_v2"),
        "base_group_col": base_group_col,
        "relationship_columns_available": relation_columns,
        "family_relationship_columns_available": family_columns,
        "root_family_columns_available": root_columns,
        "relationship_ids_applied": True,
        "n_rows_with_relationship_id": int(row_has_family_id.sum()),
        "relationship_id_coverage_fraction": float(
            row_has_family_id.mean()),
        "n_query_rows": int(query_rows.sum()),
        "n_unresolved_query_parent_rows": unresolved_parent_rows,
        "candidate_family_leakage_protected": complete_family_linkage,
        "limitation": (
            None if complete_family_linkage else
            "family-linking IDs are missing on some rows or a generated "
            "query does not resolve to another row through its parent ID; "
            "available relations are grouped, but global candidate-family "
            "non-leakage cannot be claimed"),
        "n_connected_groups": int(len(group_sizes)),
        "n_multirow_groups": int((group_sizes > 1).sum()),
        "max_group_rows": int(group_sizes.max()),
    }


def synthetic_rows(frame):
    """Identify derived hypotheses even in legacy CSVs without source names."""
    mask = pd.Series(False, index=frame.index)
    if "query_id" in frame:
        mask |= frame["query_id"].astype("string").str.strip().fillna("").ne("")
    if "negative_source" in frame:
        mask |= frame["negative_source"].astype("string").str.startswith(
            ("synthetic_", "silver_synthetic_"), na=False)
    return mask


def _validate_candidate_metadata(frame):
    candidates = synthetic_rows(frame)
    if not candidates.any():
        return
    if "parent_id" not in frame or frame.loc[
            candidates, "parent_id"].astype("string").str.strip().fillna("").eq("").any():
        raise ValueError("synthetic query rows require nonempty parent_id for CV")
    if "negative_source" in frame:
        counterfactual = frame["negative_source"].astype("string").str.startswith(
            "synthetic_", na=False)
        if counterfactual.any() and (
                "peptide_group_id" not in frame or frame.loc[
                    counterfactual, "peptide_group_id"].astype("string")
                .str.strip().fillna("").eq("").any()):
            raise ValueError("counterfactual rows require peptide_group_id for CV")


def prepare_cv_groups(frame, configured_group_col):
    """Apply all family links before cohort filtering can remove a parent."""
    _validate_candidate_metadata(frame)
    if not nonempty_relation_columns(frame) and configured_group_col is None:
        return None, {"mode": "configured_groups", "group_col": None}
    base = configured_group_col or "sequence"
    return assign_leakage_groups(frame, base)


def validate_cv_groups(frame, groups):
    """Reject API callers supplying partitions that break a known family."""
    _validate_candidate_metadata(frame)
    if not nonempty_relation_columns(frame):
        return
    if groups is None or len(groups) != len(frame) or pd.isna(groups).any():
        raise ValueError("candidate-family CV requires complete split groups")
    work = frame.copy()
    work["__provided_group"] = np.asarray(groups)
    group_col, _ = assign_leakage_groups(work, "__provided_group")
    if work.groupby(group_col)["__provided_group"].nunique().gt(1).any():
        raise ValueError("CV groups split a peptide/candidate family across folds")
