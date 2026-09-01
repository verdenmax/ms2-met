"""Stable identity helpers for related peptide-hypothesis families."""

from __future__ import annotations

import hashlib


PEPTIDE_GROUP_ID_SCHEMA = "li_normalized_parent_sequence_sha256_v1"


def li_normalize_sequence(sequence: str) -> str:
    """Return the uppercase sequence under the project's L/I equivalence."""
    return str(sequence).strip().upper().replace("I", "L")


def peptide_group_id(sequence: str) -> str:
    """Identify one parent peptide family across raw files and charge states."""
    normalized = li_normalize_sequence(sequence)
    if not normalized:
        raise ValueError("cannot build peptide_group_id from an empty sequence")
    payload = f"{PEPTIDE_GROUP_ID_SCHEMA}|{normalized}".encode("utf-8")
    return "PG" + hashlib.sha256(payload).hexdigest()[:24]
