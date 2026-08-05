"""Tests for label column writing in flow_utils and label-derivation helpers."""
import numpy as np
import pandas as pd
import pytest

from spectrum.psm_info import PSMInfo


def _psm(seq, label_type, raw="r1", protein_names="X_HUMAN"):
    p = PSMInfo(
        sequence=seq, charge=2, modify=[],
        rt=np.float32(1.0), precursor_mz=np.float32(500.0),
        raw_title=raw, protein_names=protein_names,
    )
    p._label_type = label_type
    return p


def test_single_flow_writes_numeric_label_from_label_type():
    """`label` column in single-flow result must be 0/1
    (not the protein name string). `None` label_type now raises
    (P1-3, Pipeline-I1, 2026-06-03 audit) — see dedicated raise test."""
    from workflows.flow_utils import _make_result_row_single

    row_pos = _make_result_row_single(
        _psm("HUMAN_PEP", "positive"), features={"feat1": 1.0})
    row_neg = _make_result_row_single(
        _psm("BAD_PEP", "negative"), features={"feat1": 2.0})

    assert row_pos["label"] == 1
    assert row_neg["label"] == 0
    assert row_pos["label_type"] == "positive"
    assert row_neg["label_type"] == "negative"
    # features merged into row
    assert row_pos["feat1"] == 1.0
    # other required keys still present
    assert row_pos["sequence"] == "HUMAN_PEP"
    assert row_pos["rt"] == pytest.approx(1.0)
    assert "raw_title1" in row_pos

    # None label_type must raise (no silent NaN labels)
    with pytest.raises(ValueError, match="label_type"):
        _make_result_row_single(
            _psm("UNKNOWN", None), features={"feat1": 3.0})


def test_single_flow_dict_used_by_process_batch_single():
    """Confirm that process_batch_single uses _make_result_row_single
    (so the fix actually propagates to the CSV)."""
    import inspect
    from workflows import flow_utils
    src = inspect.getsource(flow_utils.process_batch_single)
    assert "_make_result_row_single" in src or "_make_result_row" in src, (
        "process_batch_single must use the shared helper")


def test_matches_species_marker_basic_human_suffix():
    from spectrum.species_marker import matches_species_marker
    assert matches_species_marker("X_HUMAN", "HUMAN") is True
    assert matches_species_marker("X_HUMAN/Y_HUMAN", "HUMAN") is True
    assert matches_species_marker("X_HUMAN;Y_HUMAN", "HUMAN") is True


def test_matches_species_marker_substring_false_positives_rejected():
    from spectrum.species_marker import matches_species_marker
    # These contain "HUMAN" as substring but are not _HUMAN suffixed
    assert matches_species_marker("HUMANIN", "HUMAN") is False
    assert matches_species_marker("HUMANITY", "HUMAN") is False
    assert matches_species_marker("HUMANSP", "HUMAN") is False


def test_matches_species_marker_excludes_decoys():
    from spectrum.species_marker import matches_species_marker
    assert matches_species_marker("REV_X_HUMAN", "HUMAN") is False
    assert matches_species_marker("DECOY_X_HUMAN", "HUMAN") is False
    assert matches_species_marker("rev_X_HUMAN", "HUMAN") is False
    assert matches_species_marker("decoy_X_HUMAN", "HUMAN") is False
    assert matches_species_marker("_REV_X_HUMAN", "HUMAN") is False


def test_matches_species_marker_mixed_multi_protein():
    """Mixed list: at least one non-decoy target → True"""
    from spectrum.species_marker import matches_species_marker
    assert matches_species_marker(
        "REV_X_HUMAN/Y_HUMAN", "HUMAN") is True
    assert matches_species_marker(
        "Y_HUMAN/REV_X_HUMAN", "HUMAN") is True
    # All decoys → False
    assert matches_species_marker(
        "REV_X_HUMAN/DECOY_Y_HUMAN", "HUMAN") is False


def test_matches_species_marker_empty_input():
    from spectrum.species_marker import matches_species_marker
    assert matches_species_marker("", "HUMAN") is False
    assert matches_species_marker(None, "HUMAN") is False
    assert matches_species_marker("X_HUMAN", "") is False


def test_matches_species_marker_other_species():
    from spectrum.species_marker import matches_species_marker
    assert matches_species_marker("X_MOUSE", "MOUSE") is True
    assert matches_species_marker("X_HUMAN", "MOUSE") is False
    assert matches_species_marker("X_YEAST", "YEAST") is True


def test_matches_species_marker_uniprot_format():
    """UniProt-style sp|P12345|GENE_HUMAN should match."""
    from spectrum.species_marker import matches_species_marker
    assert matches_species_marker("sp|P12345|GENE_HUMAN", "HUMAN") is True
    assert matches_species_marker(
        "sp|REV_P12345|GENE_HUMAN", "HUMAN") is False


def test_extract_common_uses_new_marker_matcher():
    """extract_n_engines_from_psms must call matches_species_marker
    instead of substring `in`."""
    import inspect
    from tools import extract_common
    src = inspect.getsource(extract_common.extract_n_engines_from_psms)
    assert "matches_species_marker" in src or "species_marker" in src, (
        "extract_common must use the new helper")
    # The string ` in psm._protein_names` should no longer appear
    assert "positive_marker in psm._protein_names" not in src
    assert "positive_marker not in psm._protein_names" not in src


def test_eval_baseline_uses_new_marker_matcher():
    import inspect
    from tools import eval_baseline
    src = inspect.getsource(eval_baseline.derive_binary_label)
    # The "HUMAN" substring contains check should be gone
    assert ".str.contains(" not in src or "matches_species_marker" in src
