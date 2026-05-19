"""Self-contained L0/L1 entrapment classifier for trap PSMs.

Determines whether a trap peptide is mass-spec-indistinguishable from
the target proteome:

  - L0 (razor-error): trap stripped sequence appears verbatim in the
    target FASTA (substring match). Cannot be distinguished from a
    target peptide by mass spec.

  - L1 (LI-isomer): after L↔I normalization, the trap sequence appears
    in the target. Since L (Leu, 113.08406 Da) and I (Ile, 113.08406 Da)
    are mass-identical, these are also indistinguishable.

  - L4 (true trap): neither L0 nor L1 (we don't compute L2/L3 here —
    those require Hamming-distance scans).

Use substring matching instead of in-silico trypsin digestion:
  - Simpler and faster (one linear scan of the proteome text)
  - Marginally more conservative (a substring hit is a superset of
    the digest hits for any reasonable enzyme specificity)
  - For SILAC validation, "conservative" means "drop more potential
    L0/L1", which is the safer direction (we'd rather over-clean
    the negative set than leave noise in it)

The classifier outputs a TSV compatible with
``tools/extract_common.load_entrapment_classifications`` so the
downstream filter pipeline is unchanged.
"""
from __future__ import annotations

import logging
import os
from dataclasses import dataclass


# Sentinel character placed between proteins so a peptide cannot
# match across two protein boundaries. Must not appear in any
# valid amino-acid sequence — '|' is unused in standard FASTA.
_PROTEIN_SEPARATOR = "|"


@dataclass
class TargetIndex:
    """Concatenated target proteome plus its L↔I-normalized view.

    Attributes:
        raw_text: All target protein sequences joined with a separator
            so substring searches don't leak across boundaries.
        li_normalized_text: Same text after replacing all 'I' with 'L'
            (canonical direction; either choice is fine as long as
            it's used consistently with classify_peptide).
        n_proteins: Number of protein records loaded.
    """
    raw_text: str
    li_normalized_text: str
    n_proteins: int


def load_target_fasta(fasta_path: str) -> TargetIndex:
    """Load a FASTA file into a TargetIndex.

    Args:
        fasta_path: Path to a FASTA file with one or more protein
            records.  Multi-line sequences are concatenated. Lowercase
            residues are uppercased.

    Returns:
        TargetIndex with raw_text, li_normalized_text, n_proteins.

    Raises:
        FileNotFoundError: if fasta_path does not exist.
    """
    if not os.path.exists(fasta_path):
        raise FileNotFoundError(
            f"target FASTA 文件不存在: '{fasta_path}'")

    proteins: list[str] = []
    current: list[str] = []
    n_records = 0

    with open(fasta_path, "r", encoding="utf-8") as f:
        for line in f:
            line = line.rstrip("\r\n")
            if not line:
                continue
            if line.startswith(">"):
                if current:
                    proteins.append("".join(current).upper())
                    current = []
                n_records += 1
            else:
                current.append(line.strip())
        if current:
            proteins.append("".join(current).upper())

    raw_text = _PROTEIN_SEPARATOR.join(proteins)
    li_normalized_text = raw_text.replace("I", "L")

    logging.info(
        f"加载 target FASTA: {fasta_path} "
        f"({n_records} 条记录, 拼接后长度 {len(raw_text)} aa)"
    )
    return TargetIndex(
        raw_text=raw_text,
        li_normalized_text=li_normalized_text,
        n_proteins=n_records,
    )


def classify_peptide(peptide: str, target: TargetIndex) -> str:
    """Return L0 / L1 / L4 for a single trap peptide.

    Args:
        peptide: trap peptide stripped sequence. Case-insensitive
            (will be uppercased).
        target: TargetIndex from load_target_fasta.

    Returns:
        "L0" — exact substring of target proteome
        "L1" — substring of target only after L↔I normalization
        "L4" — neither (we don't compute L2/L3 here)

    The empty peptide returns "L4" (cannot be a meaningful match).
    """
    if not peptide:
        return "L4"
    peptide = peptide.upper()

    # L0 takes precedence — exact match
    if peptide in target.raw_text:
        return "L0"

    # L1 — normalize both sides (I → L) and check substring
    peptide_li = peptide.replace("I", "L")
    if peptide_li in target.li_normalized_text:
        return "L1"

    return "L4"


def classify_peptides_batch(
    peptides: list[str], target: TargetIndex,
) -> list[str]:
    """Classify a list of peptides; returns same-order list of levels."""
    return [classify_peptide(p, target) for p in peptides]
