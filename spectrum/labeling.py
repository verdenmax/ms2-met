"""Canonical metabolic-labeling rules.

This module is the single in-process seam for labeling names and sequence
mass shifts. Configuration readers adapt their strings to ``HeavyType`` once;
the rest of the pipeline works with the enum and cannot maintain a separate
alias table.

Uniform 13C/15N calculations cover atoms contributed by the unmodified
peptide sequence. PTM elemental compositions are intentionally outside this
module's current supported domain.
"""

from __future__ import annotations

from enum import Enum

from pyteomics import mass


MASS_DELTA_C13_C12 = 1.003355
MASS_DELTA_N15_N14 = 0.997035


class HeavyType(Enum):
    SILAC = 1
    CHEAVY = 2
    NHEAVY = 3


_ALIASES = {
    "silac": HeavyType.SILAC,
    "c13": HeavyType.CHEAVY,
    "13c": HeavyType.CHEAVY,
    "cheavy": HeavyType.CHEAVY,
    "n15": HeavyType.NHEAVY,
    "15n": HeavyType.NHEAVY,
    "nheavy": HeavyType.NHEAVY,
}

_CANONICAL_NAMES = {
    HeavyType.SILAC: "silac",
    HeavyType.CHEAVY: "c13",
    HeavyType.NHEAVY: "n15",
}


def parse_heavy_type(value: str | HeavyType) -> HeavyType:
    """Return the canonical enum for a case-insensitive labeling alias."""
    if isinstance(value, HeavyType):
        return value
    key = str(value).strip().lower()
    if key not in _ALIASES:
        raise ValueError(
            f"unknown labeling {value!r}; expected one of "
            f"{sorted(_ALIASES)}")
    return _ALIASES[key]


def canonical_labeling_name(heavy_type: HeavyType) -> str:
    """Return the stable serialized name: ``silac``, ``c13``, or ``n15``."""
    try:
        return _CANONICAL_NAMES[parse_heavy_type(heavy_type)]
    except KeyError as exc:  # defensive if HeavyType is extended incompletely
        raise ValueError(f"unsupported heavy type: {heavy_type!r}") from exc


def get_silac_increase_mass(sequence: str) -> float:
    """Return the K(+8.014204)/R(+10.008275) SILAC mass shift."""
    increase_mass = 0.0
    for amino_acid in str(sequence).upper():
        if amino_acid == "K":
            increase_mass += 8.014204
        elif amino_acid == "R":
            increase_mass += 10.008275
    return increase_mass


def get_heavy_increase_mass(
    sequence: str,
    heavy_type: HeavyType,
) -> float:
    """Return the sequence-only mass shift for the selected labeling."""
    selected = parse_heavy_type(heavy_type)
    if selected is HeavyType.SILAC:
        return get_silac_increase_mass(sequence)

    composition = mass.Composition(str(sequence).upper())
    if selected is HeavyType.CHEAVY:
        return float(composition["C"] * MASS_DELTA_C13_C12)
    return float(composition["N"] * MASS_DELTA_N15_N14)


def has_label_site(
    sequence: str,
    heavy_type: HeavyType = HeavyType.SILAC,
) -> bool:
    """Whether a nonempty peptide acquires a shift under ``heavy_type``."""
    seq = str(sequence).upper()
    if not seq:
        return False
    selected = parse_heavy_type(heavy_type)
    if selected is HeavyType.SILAC:
        return any(amino_acid in "KR" for amino_acid in seq)
    # Every peptide contains carbon and nitrogen in its backbone.
    return True


def supports_modified_peptide(heavy_type: HeavyType) -> bool:
    """Whether PTM atoms are covered by the current labeling mass model."""
    return parse_heavy_type(heavy_type) is HeavyType.SILAC
