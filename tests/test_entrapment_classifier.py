"""Tests for spectrum/entrapment_classifier.py — self-contained L0/L1 classifier.

L0/L1 are mass-spec-indistinguishable from target peptides:
  - L0: trap stripped sequence exists verbatim in the target proteome
  - L1: trap sequence, after L↔I normalization, exists in the L↔I
    normalized target proteome

Substring match (not in-silico digest) is sufficient — a peptide
that appears as a substring inside any target protein is at least
as severe as a digest hit, and may catch additional edge cases.
"""
import textwrap

import pytest


# ----------------------------------------------------------------------
# load_target_fasta
# ----------------------------------------------------------------------

def test_load_target_fasta_basic(tmp_path):
    from spectrum.entrapment_classifier import load_target_fasta

    fasta = tmp_path / "tiny.fasta"
    fasta.write_text(textwrap.dedent("""\
        >sp|P00001|GENE1_HUMAN Description 1
        PEPTIDEKAAAR
        >sp|P00002|GENE2_HUMAN Description 2
        MKAGAVKLLLR
        """))

    target = load_target_fasta(str(fasta))
    # Returns a TargetIndex object with two views: raw + LI-normalized
    assert "PEPTIDEK" in target.raw_text
    assert "AGAVK" in target.raw_text
    # Sequences are separated by a delimiter so a peptide cannot match
    # across two proteins by accident
    assert "PEPTIDEKAAARMKAGAVKLLLR" not in target.raw_text
    # LI-normalized has all L → I (or all I → L; either convention works)
    assert target.li_normalized_text.count("I") + \
           target.li_normalized_text.count("L") == \
           target.raw_text.count("I") + target.raw_text.count("L")
    # Exactly one of L or I in normalized text (not both)
    assert (target.li_normalized_text.count("I") == 0) \
        or (target.li_normalized_text.count("L") == 0)


def test_load_target_fasta_handles_multiline_sequence(tmp_path):
    from spectrum.entrapment_classifier import load_target_fasta

    fasta = tmp_path / "ml.fasta"
    fasta.write_text(">prot1\nPEPTIDE\nKAAAR\n>prot2\nMKLLLR\n")
    target = load_target_fasta(str(fasta))
    assert "PEPTIDEKAAAR" in target.raw_text


def test_load_target_fasta_skips_empty(tmp_path):
    from spectrum.entrapment_classifier import load_target_fasta

    fasta = tmp_path / "empty_ok.fasta"
    fasta.write_text(">prot1\n\nPEPTIDE\n>prot2\n\n\nMKR\n")
    target = load_target_fasta(str(fasta))
    assert "PEPTIDE" in target.raw_text
    assert "MKR" in target.raw_text


def test_load_target_fasta_missing_file_raises(tmp_path):
    from spectrum.entrapment_classifier import load_target_fasta

    with pytest.raises(FileNotFoundError) as excinfo:
        load_target_fasta(str(tmp_path / "nope.fasta"))
    assert "nope.fasta" in str(excinfo.value)


def test_load_target_fasta_uppercases_sequence(tmp_path):
    """Some FASTA contain lowercase residues; should normalize."""
    from spectrum.entrapment_classifier import load_target_fasta

    fasta = tmp_path / "mixed.fasta"
    fasta.write_text(">prot1\npepTIDEkaaar\n")
    target = load_target_fasta(str(fasta))
    assert "PEPTIDEKAAAR" in target.raw_text


# ----------------------------------------------------------------------
# classify_peptide — single peptide, full algorithm
# ----------------------------------------------------------------------

@pytest.fixture
def small_target(tmp_path):
    """Tiny target with known peptides to test classification edge cases."""
    from spectrum.entrapment_classifier import load_target_fasta
    fasta = tmp_path / "small.fasta"
    fasta.write_text(textwrap.dedent("""\
        >sp|P1|GENE_HUMAN
        MKPEPTIDEKAAARGAVKLLLRLLLSTART
        >sp|P2|FOO_HUMAN
        XCATGGGAR
        """))
    return load_target_fasta(str(fasta))


def test_classify_exact_match_is_L0(small_target):
    from spectrum.entrapment_classifier import classify_peptide
    # PEPTIDEK is a substring of MKPEPTIDEKAAARGAVK...
    assert classify_peptide("PEPTIDEK", small_target) == "L0"
    assert classify_peptide("GAVK", small_target) == "L0"
    assert classify_peptide("LLLR", small_target) == "L0"


def test_classify_li_isomer_is_L1(small_target):
    from spectrum.entrapment_classifier import classify_peptide
    # Target has LLLR — trap "IIIR" is L1 (all L → I, mass identical)
    assert classify_peptide("IIIR", small_target) == "L1"
    # Target has LLLSTART → trap ILLSTART (first L → I) is L1
    assert classify_peptide("ILLSTART", small_target) == "L1"
    # Mixed: target has PEPTIDEK → trap PEPT is L0 (substring)
    assert classify_peptide("PEPT", small_target) == "L0"


def test_classify_not_in_target_is_L4(small_target):
    from spectrum.entrapment_classifier import classify_peptide
    # Random sequence not present
    assert classify_peptide("WWWWWW", small_target) == "L4"
    # Single residue change that's not L↔I (K→R is real mass diff)
    assert classify_peptide("GAVR", small_target) == "L4"


def test_classify_L0_takes_precedence_over_L1(small_target):
    """A peptide that is both an exact match AND an L↔I isomer of
    another target peptide is L0 (more severe)."""
    from spectrum.entrapment_classifier import classify_peptide
    # GAVK is in target; also GAVK has no L/I to normalize, still L0
    assert classify_peptide("GAVK", small_target) == "L0"


def test_classify_empty_peptide_returns_L4(small_target):
    from spectrum.entrapment_classifier import classify_peptide
    assert classify_peptide("", small_target) == "L4"


def test_classify_lowercase_peptide_handled(small_target):
    from spectrum.entrapment_classifier import classify_peptide
    # User input may have lowercase; should be uppercased first
    assert classify_peptide("peptidek", small_target) == "L0"


# ----------------------------------------------------------------------
# classify_peptides_batch — performance / batch API
# ----------------------------------------------------------------------

def test_classify_peptides_batch(small_target):
    from spectrum.entrapment_classifier import classify_peptides_batch
    peptides = ["PEPTIDEK", "IIIR", "WWWWWW", ""]
    levels = classify_peptides_batch(peptides, small_target)
    assert levels == ["L0", "L1", "L4", "L4"]


def test_classify_peptides_batch_no_target_protein_separator_leak(tmp_path):
    """A peptide that spans across two protein boundaries in the FASTA
    must NOT be classified as L0 — the separator should prevent it."""
    from spectrum.entrapment_classifier import (
        load_target_fasta, classify_peptide,
    )
    fasta = tmp_path / "boundary.fasta"
    fasta.write_text(">p1\nAAACCC\n>p2\nDDDEEE\n")
    target = load_target_fasta(str(fasta))
    # AAACCC is L0 (in p1)
    assert classify_peptide("AAACCC", target) == "L0"
    # CCCDDD straddles boundary → must NOT be L0
    assert classify_peptide("CCCDDD", target) == "L4"
