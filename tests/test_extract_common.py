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


def test_diann_priority_for_marker_check():
    """当 diann 在引擎列表中时，diann 的 PSM 优先作为权威源。"""
    from tools.extract_common import extract_n_engines_from_psms

    engine_psms = {
        "pfind": [_make_psm("DISPUTED_PEP", 2, "sp|X|FAKE_HUMAN/")],
        "diann": [_make_psm("DISPUTED_PEP", 2, "sp|X|FAKE_ECOLI/")],
    }
    result = extract_n_engines_from_psms(
        engine_psms, engine_order=["pfind", "diann"],
        positive_marker="HUMAN")

    assert len(result) == 1
    assert result[0]._label_type == "negative"


def test_no_diann_falls_back_to_engine_order():
    """无 diann 时按 engine_order 顺序选权威。"""
    from tools.extract_common import extract_n_engines_from_psms

    engine_psms = {
        "pfind": [_make_psm("AAA", 2, "sp|X|FAKE_HUMAN/")],
        "alphadia": [_make_psm("AAA", 2, "sp|X|FAKE_ECOLI/")],
    }
    result = extract_n_engines_from_psms(
        engine_psms, engine_order=["alphadia", "pfind"],
        positive_marker="HUMAN")

    assert result[0]._label_type == "negative"


def test_no_stale_label_on_repeated_call():
    """重复调用同一 engine_psms 应无 stale state 残留。

    注：因为函数 in-place 修改 PSM 对象，result1/result2 共享同一对象引用，
    所以需要在每次调用后立刻 snapshot label_type 才能验证各自的标签结果。
    """
    from tools.extract_common import extract_n_engines_from_psms

    engine_psms = {
        "pfind": [_make_psm("AAA", 2, "sp|X|FAKE_HUMAN/"),
                  _make_psm("BBB", 2, "sp|X|FAKE_ECOLI/")],
        "diann": [_make_psm("AAA", 2, "sp|X|FAKE_HUMAN/"),
                  _make_psm("CCC", 2, "sp|X|FAKE_HUMAN/")],
    }

    result1 = extract_n_engines_from_psms(
        engine_psms, engine_order=["pfind", "diann"],
        positive_marker="HUMAN")
    snap1 = {p._sequence: p._label_type for p in result1}

    result2 = extract_n_engines_from_psms(
        engine_psms, engine_order=["pfind", "diann"],
        positive_marker="ECOLI")
    snap2 = {p._sequence: p._label_type for p in result2}

    # result1：marker=HUMAN
    #   - AAA 在交集 + 含 HUMAN → positive
    #   - BBB 在并集（pfind only）+ 不含 HUMAN → negative
    #   - CCC 在并集（diann only）+ 含 HUMAN → 不进 negative
    assert snap1.get("AAA") == "positive"
    assert snap1.get("BBB") == "negative"

    # result2：marker=ECOLI
    #   - AAA 在交集 + 不含 ECOLI → 不进 positive
    #   - AAA 在并集 + 不含 ECOLI → negative
    #   - BBB 含 ECOLI → 不进 negative
    #   - CCC 不含 ECOLI → negative
    # 关键：snap2 应该完全反映 result2 的逻辑，
    # 不应有 result1 残留的 "positive"
    assert snap2.get("AAA") == "negative"
    assert snap2.get("CCC") == "negative"
    # BBB 在 result2 中应该不出现（被 reset 清空后 result2 没给它打标签）
    assert "BBB" not in snap2

    # 验证 stale state 防御：所有未被 result2 选中的 PSM 的 label 应为 None
    for psms in engine_psms.values():
        for psm in psms:
            if psm._sequence not in snap2:
                assert psm._label_type is None, (
                    f"stale label on {psm._sequence}: {psm._label_type}")
