"""Centralized species-marker matching for protein-names columns.

The convention follows UniProt's accession suffix (e.g. P12345_HUMAN);
substring matches are unsafe because:
  - "HUMAN" appears in legitimate non-decoy proteins (HUMANIN, HUMANITY)
  - Decoy prefixes (REV_X_HUMAN, DECOY_X_HUMAN) would falsely match a
    target marker

Use matches_species_marker(protein_names, marker) from extract_common,
flow_utils, and eval_baseline for consistent label derivation.
"""
import re
from typing import Optional

_TOKEN_SEP = re.compile(r"[;/]")


def _is_decoy_token(token: str) -> bool:
    """A token whose accession starts with REV_ / DECOY_ / _REV_ / _DECOY_
    (case-insensitive). For UniProt-style tokens like sp|REV_P12345|GENE_HUMAN,
    any pipe-separated segment with that prefix marks the whole token as decoy."""
    if not token:
        return True
    for segment in token.split("|"):
        upper = segment.upper()
        if (upper.startswith("REV_")
                or upper.startswith("DECOY_")
                or upper.startswith("_REV_")
                or upper.startswith("_DECOY_")):
            return True
    return False


def _strip_uniprot_accession(token: str) -> str:
    """If token is sp|P12345|GENE_HUMAN style, return the last segment
    (GENE_HUMAN). Otherwise return token unchanged."""
    if "|" in token:
        return token.split("|")[-1]
    return token


def _token_matches_marker(token: str, marker: str) -> bool:
    """True iff token ends with '_<marker>' (case-sensitive on marker)
    and is not a decoy."""
    if _is_decoy_token(token):
        return False
    suffix_candidate = _strip_uniprot_accession(token)
    return (suffix_candidate.endswith(f"_{marker}")
            or suffix_candidate == marker)


def matches_species_marker(
    protein_names: Optional[str], marker: str
) -> bool:
    """Return True if any non-decoy token in `protein_names` ends
    with `_{marker}`.

    Args:
        protein_names: a string from a search engine's protein-names
            column. May be None or empty. Multi-protein lists separated
            by `;` or `/`. UniProt format `sp|P12345|GENE_HUMAN` is
            supported.
        marker: the species marker (e.g. "HUMAN", "MOUSE").

    Returns:
        True iff at least one token has suffix `_marker` and is not
        a decoy. False for empty/None input.
    """
    if not protein_names or not marker:
        return False
    tokens = [t.strip() for t in _TOKEN_SEP.split(protein_names) if t.strip()]
    return any(_token_matches_marker(t, marker) for t in tokens)
