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


def test_extract_preserves_cross_raw_observations():
    """同一肽段在不同 raw 出现时应作为独立观测保留（不合并）。"""
    from tools.extract_common import extract_n_engines_from_psms

    engine_psms = {
        "pfind": [
            _make_psm("PEPTIDE_K", 2, "sp|X|TEST_HUMAN/", raw="rep1"),
            _make_psm("PEPTIDE_K", 2, "sp|X|TEST_HUMAN/", raw="rep2"),
            _make_psm("PEPTIDE_K", 2, "sp|X|TEST_HUMAN/", raw="rep3"),
        ],
        "diann": [
            _make_psm("PEPTIDE_K", 2, "sp|X|TEST_HUMAN/", raw="rep1"),
            _make_psm("PEPTIDE_K", 2, "sp|X|TEST_HUMAN/", raw="rep2"),
            _make_psm("PEPTIDE_K", 2, "sp|X|TEST_HUMAN/", raw="rep3"),
        ],
    }
    result = extract_n_engines_from_psms(
        engine_psms, engine_order=["pfind", "diann"],
        positive_marker="HUMAN")

    assert len(result) == 3
    raw_titles = {p._raw_title for p in result}
    assert raw_titles == {"rep1", "rep2", "rep3"}, f"Got: {raw_titles}"
    assert all(p._label_type == "positive" for p in result)


def test_extract_cross_raw_only_in_one_engine_keeps_independently():
    """跨 raw 观测 + 一个引擎单独有 → 各自独立."""
    from tools.extract_common import extract_n_engines_from_psms

    engine_psms = {
        "pfind": [
            _make_psm("PEP", 2, "sp|X|H_HUMAN/", raw="rep1"),
            _make_psm("PEP", 2, "sp|X|H_HUMAN/", raw="rep2"),
        ],
        "diann": [
            _make_psm("PEP", 2, "sp|X|H_HUMAN/", raw="rep1"),
            _make_psm("PEP", 2, "sp|X|H_HUMAN/", raw="rep3"),
        ],
    }
    result = extract_n_engines_from_psms(
        engine_psms, engine_order=["pfind", "diann"],
        positive_marker="HUMAN")

    positives = sorted([p._raw_title for p in result if p._label_type == "positive"])
    assert positives == ["rep1"]
    assert len(result) == 1


# ------------------------------------------------------------------
# Entrapment 过滤相关测试 (L0/L1 negative 剔除)
# ------------------------------------------------------------------


def _write_classified_tsv(path, rows):
    """rows: list of dict with keys peptide/charge/spectrum_file/level (+ optional fields)."""
    header = ["peptide", "charge", "precursor_mz", "retention_time",
              "scan_number", "spectrum_file", "protein_ids", "q_value",
              "group", "level"]
    with open(path, "w", encoding="utf-8") as f:
        f.write("\t".join(header) + "\n")
        for r in rows:
            f.write("\t".join([
                str(r.get("peptide", "")),
                str(r.get("charge", "")),
                str(r.get("precursor_mz", "0")),
                str(r.get("retention_time", "0")),
                str(r.get("scan_number", "")),
                str(r.get("spectrum_file", "")),
                str(r.get("protein_ids", "")),
                str(r.get("q_value", "")),
                str(r.get("group", "trap")),
                str(r.get("level", "")),
            ]) + "\n")


def test_load_entrapment_classifications_basic(tmp_path):
    """读取 classified.tsv 应建立 (sequence, charge, raw_title) -> level 索引。"""
    from tools.extract_common import load_entrapment_classifications

    tsv = tmp_path / "classified.tsv"
    _write_classified_tsv(tsv, [
        {"peptide": "AAAK", "charge": 2, "spectrum_file": "raw1", "level": "L0"},
        {"peptide": "BBBR", "charge": 3, "spectrum_file": "raw1", "level": "L1"},
        {"peptide": "CCCK", "charge": 2, "spectrum_file": "raw2", "level": "L4"},
    ])

    classifications = load_entrapment_classifications(str(tsv))
    assert classifications[("AAAK", 2, "raw1")] == "L0"
    assert classifications[("BBBR", 3, "raw1")] == "L1"
    assert classifications[("CCCK", 2, "raw2")] == "L4"
    assert classifications.get(("ZZZZ", 2, "raw1")) is None


def test_load_entrapment_classifications_handles_empty_lines(tmp_path):
    """空行 / 缺失 level 的行应被跳过，不报错。"""
    from tools.extract_common import load_entrapment_classifications

    tsv = tmp_path / "classified.tsv"
    _write_classified_tsv(tsv, [
        {"peptide": "AAAK", "charge": 2, "spectrum_file": "raw1", "level": "L0"},
        {"peptide": "MISSING", "charge": 2, "spectrum_file": "raw1", "level": ""},
        {"peptide": "BBBR", "charge": 2, "spectrum_file": "raw1", "level": "L4"},
    ])
    classifications = load_entrapment_classifications(str(tsv))
    # 缺 level 的行不计入
    assert ("MISSING", 2, "raw1") not in classifications
    assert classifications[("AAAK", 2, "raw1")] == "L0"
    assert classifications[("BBBR", 2, "raw1")] == "L4"


def test_filter_by_entrapment_removes_L0_L1_negatives_only():
    """L0/L1 negative 被剔除；L4 negative 与 unknown negative 保留；positive 全部不动。"""
    from tools.extract_common import filter_by_entrapment

    # 输入: 2 个 positive (HUMAN) + 4 个 negative (各种 level)
    psm_l0_neg = _make_psm("L0PEP", 2, "ROA1_RAT", raw="raw1")
    psm_l0_neg._label_type = "negative"
    psm_l1_neg = _make_psm("L1PEP", 2, "ACT_DICDI", raw="raw1")
    psm_l1_neg._label_type = "negative"
    psm_l4_neg = _make_psm("L4PEP", 2, "GYP7_YEAST", raw="raw2")
    psm_l4_neg._label_type = "negative"
    psm_unknown_neg = _make_psm("UNKNOWN", 2, "FOO_BAR", raw="raw1")
    psm_unknown_neg._label_type = "negative"
    psm_pos_1 = _make_psm("HUMAN_A", 2, "P12345_HUMAN", raw="raw1")
    psm_pos_1._label_type = "positive"
    psm_pos_2 = _make_psm("HUMAN_B", 3, "P67890_HUMAN", raw="raw1")
    psm_pos_2._label_type = "positive"

    psms = [psm_l0_neg, psm_l1_neg, psm_l4_neg,
            psm_unknown_neg, psm_pos_1, psm_pos_2]

    classifications = {
        ("L0PEP", 2, "raw1"): "L0",
        ("L1PEP", 2, "raw1"): "L1",
        ("L4PEP", 2, "raw2"): "L4",
    }

    result = filter_by_entrapment(
        psms, classifications, drop_levels={"L0", "L1"})

    result_seqs = {p._sequence for p in result}
    assert "L0PEP" not in result_seqs
    assert "L1PEP" not in result_seqs
    assert "L4PEP" in result_seqs
    assert "UNKNOWN" in result_seqs
    assert "HUMAN_A" in result_seqs
    assert "HUMAN_B" in result_seqs
    assert len(result) == 4  # 6 - 2


def test_filter_by_entrapment_only_touches_negatives():
    """即使某 positive 的 (seq, charge, raw) 也在 classified.tsv 中被标 L0/L1，positive 也不动。"""
    from tools.extract_common import filter_by_entrapment

    psm_pos = _make_psm("CONFUSING", 2, "HUMAN_X", raw="raw1")
    psm_pos._label_type = "positive"
    psm_neg = _make_psm("L0PEP", 2, "TRAP_X", raw="raw1")
    psm_neg._label_type = "negative"

    classifications = {
        ("CONFUSING", 2, "raw1"): "L0",  # 错误地把 positive 也标了
        ("L0PEP", 2, "raw1"): "L0",
    }

    result = filter_by_entrapment(
        [psm_pos, psm_neg], classifications, drop_levels={"L0", "L1"})

    result_seqs = {p._sequence for p in result}
    assert "CONFUSING" in result_seqs  # positive 不剔除
    assert "L0PEP" not in result_seqs


def test_filter_by_entrapment_empty_classifications_is_noop():
    """空分类表 → 一个不动。"""
    from tools.extract_common import filter_by_entrapment

    psm = _make_psm("ANY", 2, "X_TRAP", raw="r")
    psm._label_type = "negative"
    result = filter_by_entrapment([psm], {}, drop_levels={"L0", "L1"})
    assert len(result) == 1
    assert result[0]._sequence == "ANY"


def test_filter_by_entrapment_custom_drop_levels():
    """drop_levels 可配置，例如同时剔除 L0/L1/L2。"""
    from tools.extract_common import filter_by_entrapment

    psm_l0 = _make_psm("L0PEP", 2, "X_TRAP", raw="r1")
    psm_l0._label_type = "negative"
    psm_l2 = _make_psm("L2PEP", 2, "X_TRAP", raw="r1")
    psm_l2._label_type = "negative"
    psm_l4 = _make_psm("L4PEP", 2, "X_TRAP", raw="r1")
    psm_l4._label_type = "negative"

    classifications = {
        ("L0PEP", 2, "r1"): "L0",
        ("L2PEP", 2, "r1"): "L2",
        ("L4PEP", 2, "r1"): "L4",
    }

    result = filter_by_entrapment(
        [psm_l0, psm_l2, psm_l4], classifications,
        drop_levels={"L0", "L1", "L2"})

    seqs = {p._sequence for p in result}
    assert seqs == {"L4PEP"}
