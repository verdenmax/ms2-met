"""测试 tools/extract_common.py 的 N 引擎交并集逻辑。"""
import numpy as np
import pytest

from spectrum.psm_info import PSMInfo


def _make_psm(seq, charge, protein_names, rt=10.0, mz=500.0, raw="r"):
    return PSMInfo(
        sequence=seq,
        charge=charge,
        modify=[],
        rt=np.float32(rt),
        precursor_mz=np.float32(mz),
        raw_title=raw,
        protein_names=protein_names,
    )


def test_extract_intersection_no_marker():
    """无 positive_marker → 简单交集。"""
    from tools.extract_common import extract_n_engines_from_psms

    engine_psms = {
        "pfind": [
            _make_psm("AAA", 2, "sp|X|HUMAN/"),
            _make_psm("BBB", 2, "sp|X|HUMAN/"),
            _make_psm("CCC", 2, "sp|X|HUMAN/"),
        ],
        "diann": [
            _make_psm("AAA", 2, "sp|X|HUMAN/"),
            _make_psm("BBB", 2, "sp|X|HUMAN/"),
            _make_psm("DDD", 2, "sp|X|HUMAN/"),
        ],
    }
    result = extract_n_engines_from_psms(
        engine_psms, engine_order=["pfind", "diann"],
        positive_marker=None)
    seqs = sorted([p._sequence for p in result])
    assert seqs == ["AAA", "BBB"]
    assert all(p._label_type is None for p in result)


def test_extract_positive_negative_with_marker():
    """有 positive_marker → 正例（交集 + marker）+ 负例（并集 + 非 marker）。"""
    from tools.extract_common import extract_n_engines_from_psms

    engine_psms = {
        "pfind": [
            _make_psm("HUMAN_PEP1", 2, "sp|X|TEST_HUMAN/"),
            _make_psm("HUMAN_PEP2", 2, "sp|X|TEST_HUMAN/"),
            _make_psm("ECOLI_PEP1", 2, "sp|X|TEST_ECOLI/"),
        ],
        "diann": [
            _make_psm("HUMAN_PEP1", 2, "sp|X|TEST_HUMAN/"),
            _make_psm("ECOLI_PEP2", 2, "sp|X|TEST_ECOLI/"),
        ],
    }
    result = extract_n_engines_from_psms(
        engine_psms, engine_order=["pfind", "diann"],
        positive_marker="HUMAN")

    positives = [p for p in result if p._label_type == "positive"]
    negatives = [p for p in result if p._label_type == "negative"]

    pos_seqs = sorted([p._sequence for p in positives])
    neg_seqs = sorted([p._sequence for p in negatives])

    assert pos_seqs == ["HUMAN_PEP1"]
    assert neg_seqs == ["ECOLI_PEP1", "ECOLI_PEP2"]


def test_extract_three_engines_intersection():
    """N=3 引擎，正例必须三个引擎都识别。"""
    from tools.extract_common import extract_n_engines_from_psms

    engine_psms = {
        "pfind": [
            _make_psm("ALL_THREE", 2, "sp|X|TEST_HUMAN/"),
            _make_psm("ONLY_PFIND", 2, "sp|X|TEST_HUMAN/"),
        ],
        "diann": [
            _make_psm("ALL_THREE", 2, "sp|X|TEST_HUMAN/"),
            _make_psm("PFIND_DIANN", 2, "sp|X|TEST_HUMAN/"),
        ],
        "alphadia": [
            _make_psm("ALL_THREE", 2, "sp|X|TEST_HUMAN/"),
        ],
    }
    result = extract_n_engines_from_psms(
        engine_psms, engine_order=["pfind", "diann", "alphadia"],
        positive_marker="HUMAN")
    positives = [p for p in result if p._label_type == "positive"]
    pos_seqs = [p._sequence for p in positives]
    assert pos_seqs == ["ALL_THREE"]


def test_extract_label_type_attached():
    """所有输出 PSM 都应有明确的 label_type（在 marker 模式下）。"""
    from tools.extract_common import extract_n_engines_from_psms

    engine_psms = {
        "pfind": [_make_psm("AAA", 2, "sp|X|TEST_HUMAN/")],
        "diann": [_make_psm("AAA", 2, "sp|X|TEST_HUMAN/")],
    }
    result = extract_n_engines_from_psms(
        engine_psms, engine_order=["pfind", "diann"],
        positive_marker="HUMAN")
    assert all(p._label_type in ("positive", "negative") for p in result)
