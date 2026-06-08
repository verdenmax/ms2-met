"""Tests for DIA-NN / alphadia loaders in spectrum/light_result.py."""
import numpy as np
import pandas as pd
import pytest

from spectrum.light_result import LightResult


# ---------- DIA-NN ----------

def _build_diann_parquet(tmp_path, rows):
    """Build a parquet that mirrors DIA-NN's schema for testing."""
    df = pd.DataFrame(rows)
    path = tmp_path / "diann_report.parquet"
    df.to_parquet(path)
    return str(path)


_DIANN_BASE = {
    "Run": "raw1",
    "Modified.Sequence": "PEPTIDE",
    "Stripped.Sequence": "PEPTIDE",
    "Precursor.Charge": 2,
    "RT": 30.0,
    "Precursor.Mz": 500.0,
    "Protein.Names": "X_HUMAN",
    "Decoy": 0,
    "Q.Value": 0.001,
}


def _diann_row(**over):
    r = dict(_DIANN_BASE)
    r.update(over)
    return r


def test_diann_loader_filters_qvalue(tmp_path):
    parquet = _build_diann_parquet(tmp_path, [
        _diann_row(Modified_Sequence="PEPTIDE",
                   **{"Modified.Sequence": "PEPTIDE",
                      "Stripped.Sequence": "PEPTIDE",
                      "Q.Value": 0.001}),
        _diann_row(**{"Modified.Sequence": "BADQ",
                      "Stripped.Sequence": "BADQ",
                      "Q.Value": 0.05}),
    ])
    lr = LightResult()
    lr._load_from_dia_nn_input(parquet, qvalue_threshold=0.01)
    seqs = {p._sequence for p in lr.psm_info}
    assert "PEPTIDE" in seqs
    assert "BADQ" not in seqs


def test_diann_loader_filters_decoy_by_protein_prefix(tmp_path):
    parquet = _build_diann_parquet(tmp_path, [
        _diann_row(**{"Modified.Sequence": "GOOD",
                      "Stripped.Sequence": "GOOD",
                      "Protein.Names": "X_HUMAN"}),
        _diann_row(**{"Modified.Sequence": "DEC1",
                      "Stripped.Sequence": "DEC1",
                      "Protein.Names": "REV_X_HUMAN"}),
        _diann_row(**{"Modified.Sequence": "DEC2",
                      "Stripped.Sequence": "DEC2",
                      "Protein.Names": "_REV_Y_HUMAN"}),
    ])
    lr = LightResult()
    lr._load_from_dia_nn_input(parquet, qvalue_threshold=0.01)
    seqs = {p._sequence for p in lr.psm_info}
    assert "GOOD" in seqs
    assert "DEC1" not in seqs
    assert "DEC2" not in seqs


def test_diann_loader_filters_decoy_by_column(tmp_path):
    """DIA-NN's Decoy=1 rows must be dropped even if Protein.Names is benign."""
    parquet = _build_diann_parquet(tmp_path, [
        _diann_row(**{"Modified.Sequence": "GOOD",
                      "Stripped.Sequence": "GOOD",
                      "Decoy": 0}),
        _diann_row(**{"Modified.Sequence": "DCOY",
                      "Stripped.Sequence": "DCOY",
                      "Decoy": 1}),
    ])
    lr = LightResult()
    lr._load_from_dia_nn_input(parquet, qvalue_threshold=0.01)
    seqs = {p._sequence for p in lr.psm_info}
    assert "GOOD" in seqs
    assert "DCOY" not in seqs


def test_diann_loader_resilient_to_extra_columns(tmp_path):
    """Adding an extra column doesn't shift named access."""
    parquet = _build_diann_parquet(tmp_path, [
        {"_extra_padding": "junk",
         **_diann_row(**{"Modified.Sequence": "OK",
                         "Stripped.Sequence": "OK"})},
    ])
    lr = LightResult()
    lr._load_from_dia_nn_input(parquet, qvalue_threshold=0.01)
    assert len(lr.psm_info) == 1
    assert lr.psm_info[0]._sequence == "OK"
    assert lr.psm_info[0]._charge == 2


def test_diann_parse_modify_handles_non_unimod_pattern():
    """A (...) group not matching UniMod:\\d+ must not crash the whole loader."""
    from spectrum.light_result import parse_diann_peptide_modify
    # Pattern that has parens but no UniMod inside
    result = parse_diann_peptide_modify("PEPT(GG@K)IDE")
    assert isinstance(result, list)
    # Should NOT contain a None unimod_id
    for pos, uid in result:
        assert uid is not None


def test_diann_loader_charge_as_float(tmp_path):
    """Precursor.Charge stored as float64 should still produce int charge."""
    parquet = _build_diann_parquet(tmp_path, [
        _diann_row(**{"Modified.Sequence": "OK",
                      "Stripped.Sequence": "OK",
                      "Precursor.Charge": 2.0}),
    ])
    lr = LightResult()
    lr._load_from_dia_nn_input(parquet, qvalue_threshold=0.01)
    assert lr.psm_info[0]._charge == 2
    assert isinstance(lr.psm_info[0]._charge, int)


# ---------- alphadia ----------

def _build_alphadia_parquet(tmp_path, rows):
    df = pd.DataFrame(rows)
    path = tmp_path / "alphadia.parquet"
    df.to_parquet(path)
    return str(path)


_ALPHADIA_BASE = {
    "raw.name": "r1",
    "precursor.sequence": "PEPTIDE",
    "precursor.charge": 2,
    "precursor.rt.observed": 1800.0,  # seconds → 30 min
    "precursor.mz.observed": 500.0,
    "pg.genes": "X_HUMAN",
    "precursor.decoy": 0,
    "precursor.qval": 0.001,
    "precursor.mods": "",
    "precursor.mod_sites": "",
}


def _ad_row(**over):
    r = dict(_ALPHADIA_BASE)
    r.update(over)
    return r


def test_alphadia_loader_filters_qvalue(tmp_path):
    parquet = _build_alphadia_parquet(tmp_path, [
        _ad_row(**{"precursor.sequence": "GOOD",
                   "precursor.qval": 0.001}),
        _ad_row(**{"precursor.sequence": "BAD",
                   "precursor.qval": 0.05}),
    ])
    lr = LightResult()
    lr._load_from_alphadia_input(parquet, qvalue_threshold=0.01)
    seqs = {p._sequence for p in lr.psm_info}
    assert "GOOD" in seqs
    assert "BAD" not in seqs


def test_alphadia_loader_filters_decoy(tmp_path):
    parquet = _build_alphadia_parquet(tmp_path, [
        _ad_row(**{"precursor.sequence": "GOOD", "precursor.decoy": 0}),
        _ad_row(**{"precursor.sequence": "DECOY", "precursor.decoy": 1}),
    ])
    lr = LightResult()
    lr._load_from_alphadia_input(parquet, qvalue_threshold=0.01)
    seqs = {p._sequence for p in lr.psm_info}
    assert "GOOD" in seqs
    assert "DECOY" not in seqs


def test_alphadia_unknown_mod_skipped_not_none():
    """parse_alphadia_peptide_modify must NOT append (pos, None) for
    unknown UniMod names — skip and warn."""
    from spectrum.light_result import parse_alphadia_peptide_modify
    result = parse_alphadia_peptide_modify(
        "Carbamidomethyl@C;UnknownXYZ@K", "3;5")
    for pos, uid in result:
        assert uid is not None, (
            f"unknown mod was appended as None: {result}")
    # Carbamidomethyl should remain
    assert any(uid == 4 for _, uid in result)


def test_diann_modify_positions_are_0based_residue_index():
    """DIA-NN 残基修饰位置应为 0-based 残基下标（与 pFind/alphadia 及消费方一致）。

    C(UniMod:4)PEPK：carbamidomethyl 在 C（残基 0）→ 应 (0,4)，而非 (1,4)。
    物理依据：b1（=该 C）必须带 +57.02146；off-by-one 会把修饰挂到下一个残基。
    """
    from spectrum.light_result import parse_diann_peptide_modify
    assert parse_diann_peptide_modify("C(UniMod:4)PEPK") == [(0, 4)]
    assert parse_diann_peptide_modify("PEM(UniMod:35)K") == [(2, 35)]
    # N 端修饰（前导括号）仍为位置 0
    assert parse_diann_peptide_modify("(UniMod:1)PEPK") == [(0, 1)]


def test_diann_decoy_kept_when_group_contains_a_target(tmp_path):
    """与 pFind 规则一致：仅当所有蛋白 token 都是 decoy 才丢弃。
    decoy-led 但含真 target 的蛋白组应保留（旧实现 startswith 整串会误删）。"""
    parquet = _build_diann_parquet(tmp_path, [
        _diann_row(**{"Modified.Sequence": "PEPTIDEK",
                      "Stripped.Sequence": "PEPTIDEK",
                      "Protein.Names": "REV_P1;P2"}),        # 含 target → 保留
        _diann_row(**{"Modified.Sequence": "PEPTIDER",
                      "Stripped.Sequence": "PEPTIDER",
                      "Protein.Names": "REV_P1;REV_P2"}),    # 全 decoy → 丢弃
    ])
    lr = LightResult()
    lr._load_from_dia_nn_input(parquet, qvalue_threshold=0.01)
    seqs = {p._sequence for p in lr.psm_info}
    assert "PEPTIDEK" in seqs       # decoy-led 但含 target → 保留
    assert "PEPTIDER" not in seqs   # 全 decoy → 丢弃
