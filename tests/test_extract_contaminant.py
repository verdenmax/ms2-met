"""Tests for extract_common 污染库过滤 (contaminant filter)."""
import numpy as np

from spectrum.entrapment_classifier import load_target_fasta
from spectrum.psm_info import PSMInfo
from tools.extract_common import filter_by_contaminant


def _psm(seq, label):
    return PSMInfo(sequence=seq, charge=2, modify=[], rt=np.float32(10.0),
                   precursor_mz=np.float32(500.0), raw_title="r1",
                   protein_names="X", label_type=label)


def _contaminant_index(tmp_path):
    # one contaminant protein containing PEPTIDEK (L0) and GALLLR (L1 target)
    fasta = tmp_path / "cont.fasta"
    fasta.write_text(">CON_test|demo\nPEPTIDEKGALLLRCONTAMR\n")
    return load_target_fasta(str(fasta))


def test_contaminant_drops_exact_substring_both_classes(tmp_path):
    idx = _contaminant_index(tmp_path)
    psms = [
        _psm("PEPTIDEK", "positive"),   # exact substring (L0) -> DROP
        _psm("PEPTIDEK", "negative"),   # exact substring (L0) -> DROP
        _psm("CONTAMR", "positive"),    # exact substring (L0) -> DROP
        _psm("MISSINGW", "negative"),   # not present -> KEEP
    ]
    kept = filter_by_contaminant(psms, idx, match_li=True)
    assert [p._sequence for p in kept] == ["MISSINGW"]


def test_contaminant_li_isomer_dropped_only_when_match_li(tmp_path):
    idx = _contaminant_index(tmp_path)
    # GALILR -> L↔I normalize -> GALLLR, a substring of the contaminant
    iso = _psm("GALILR", "positive")
    miss = _psm("MISSINGW", "positive")

    # match_li=True -> isomer treated as contaminant -> dropped
    kept_li = filter_by_contaminant([iso, miss], idx, match_li=True)
    assert [p._sequence for p in kept_li] == ["MISSINGW"]

    # match_li=False -> only exact substrings dropped -> isomer kept
    kept_exact = filter_by_contaminant([iso, miss], idx, match_li=False)
    assert [p._sequence for p in kept_exact] == ["GALILR", "MISSINGW"]


def test_contaminant_empty_psms_noop(tmp_path):
    idx = _contaminant_index(tmp_path)
    assert filter_by_contaminant([], idx, match_li=True) == []


def test_load_target_fasta_log_label(tmp_path, caplog):
    """污染库加载日志应用 '污染库' 而非误导性的 'target FASTA'。"""
    import logging
    fa = tmp_path / "c.fasta"
    fa.write_text(">CON\nPEPTIDEK\n")
    with caplog.at_level(logging.INFO):
        load_target_fasta(str(fa), log_label="污染库")
    msgs = "\n".join(r.message for r in caplog.records)
    assert "加载 污染库" in msgs and "target FASTA" not in msgs


def test_load_target_fasta_missing_uses_label():
    import pytest
    with pytest.raises(FileNotFoundError, match="污染库 文件不存在"):
        load_target_fasta("/no/such/file.fasta", log_label="污染库")
