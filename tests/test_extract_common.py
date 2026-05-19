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


# ------------------------------------------------------------------
# 加固 / 边界 case 测试（review 后补齐）
# ------------------------------------------------------------------


def _write_classified_tsv_with_group(path, rows):
    """带 group 列 + 任意额外字段的 TSV，rows 是 dict 列表"""
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


def test_load_handles_float_like_charge(tmp_path):
    """charge='2.0' 应被解析为 2（防 pandas 上游 NaN 污染导致的字符串变形）"""
    from tools.extract_common import load_entrapment_classifications

    tsv = tmp_path / "classified.tsv"
    _write_classified_tsv_with_group(tsv, [
        {"peptide": "AAAK", "charge": "2.0", "spectrum_file": "raw1", "level": "L0"},
        {"peptide": "BBBR", "charge": "3.0", "spectrum_file": "raw1", "level": "L1"},
    ])
    cls = load_entrapment_classifications(str(tsv))
    assert cls.get(("AAAK", 2, "raw1")) == "L0"
    assert cls.get(("BBBR", 3, "raw1")) == "L1"


def test_load_logs_skipped_counts(tmp_path, caplog):
    """log 应包含 total_rows / loaded / skipped_empty_level / skipped_bad_charge / collisions"""
    import logging
    from tools.extract_common import load_entrapment_classifications

    tsv = tmp_path / "classified.tsv"
    _write_classified_tsv_with_group(tsv, [
        {"peptide": "AAAK", "charge": "2", "spectrum_file": "raw1", "level": "L0"},
        {"peptide": "EMPTY", "charge": "2", "spectrum_file": "raw1", "level": ""},
        {"peptide": "BADCHG", "charge": "abc", "spectrum_file": "raw1", "level": "L0"},
        {"peptide": "AAAK", "charge": "2", "spectrum_file": "raw1", "level": "L0"},  # dup
    ])

    with caplog.at_level(logging.INFO, logger=""):
        cls = load_entrapment_classifications(str(tsv))

    assert len(cls) == 1  # AAAK 一个 key
    log_text = "\n".join(r.message for r in caplog.records)
    assert "total_rows=4" in log_text
    assert "loaded=1" in log_text
    assert "skipped_empty_level=1" in log_text
    assert "skipped_bad_charge=1" in log_text
    assert "collisions=1" in log_text


def test_load_collision_same_level_ok(tmp_path, caplog):
    """同 (seq, charge, raw) 不同 modify 但 level 一致 → 不 warn"""
    import logging
    from tools.extract_common import load_entrapment_classifications

    tsv = tmp_path / "classified.tsv"
    _write_classified_tsv_with_group(tsv, [
        {"peptide": "AAAK", "charge": "2", "spectrum_file": "raw1", "level": "L0"},
        {"peptide": "AAAK", "charge": "2", "spectrum_file": "raw1", "level": "L0"},
    ])
    with caplog.at_level(logging.WARNING, logger=""):
        cls = load_entrapment_classifications(str(tsv))

    assert cls[("AAAK", 2, "raw1")] == "L0"
    warnings = [r for r in caplog.records if r.levelno >= logging.WARNING]
    # 同 level 合并不应触发 warning
    inconsistency_warnings = [
        w for w in warnings if "conflict" in w.message.lower()
        or "不一致" in w.message]
    assert len(inconsistency_warnings) == 0


def test_load_collision_different_level_warns(tmp_path, caplog):
    """同 key 不同 level → logging.warning，保留 most-severe (L0 > L1 > L2 > L3 > L4)"""
    import logging
    from tools.extract_common import load_entrapment_classifications

    tsv = tmp_path / "classified.tsv"
    _write_classified_tsv_with_group(tsv, [
        {"peptide": "AAAK", "charge": "2", "spectrum_file": "raw1", "level": "L4"},
        {"peptide": "AAAK", "charge": "2", "spectrum_file": "raw1", "level": "L0"},  # 更严重
        {"peptide": "BBBR", "charge": "2", "spectrum_file": "raw1", "level": "L2"},
        {"peptide": "BBBR", "charge": "2", "spectrum_file": "raw1", "level": "L3"},  # 较轻
    ])
    with caplog.at_level(logging.WARNING, logger=""):
        cls = load_entrapment_classifications(str(tsv))

    # L0 是最严重，应保留
    assert cls[("AAAK", 2, "raw1")] == "L0"
    # L2 比 L3 严重
    assert cls[("BBBR", 2, "raw1")] == "L2"

    warnings = [r for r in caplog.records if r.levelno >= logging.WARNING]
    # 至少有一条 warning 提到 conflict / 不一致 / 冲突
    assert any("conflict" in w.message.lower() or "不一致" in w.message
               or "冲突" in w.message
               for w in warnings)


def test_load_filters_non_trap_rows(tmp_path):
    """group='target' 行不应进 dict"""
    from tools.extract_common import load_entrapment_classifications

    tsv = tmp_path / "classified.tsv"
    _write_classified_tsv_with_group(tsv, [
        {"peptide": "TRAP1", "charge": "2", "spectrum_file": "raw1",
         "group": "trap", "level": "L0"},
        {"peptide": "TARGET1", "charge": "2", "spectrum_file": "raw1",
         "group": "target", "level": "L0"},
        {"peptide": "TRAP_CAPS", "charge": "2", "spectrum_file": "raw1",
         "group": "TRAP", "level": "L1"},  # case-insensitive
    ])
    cls = load_entrapment_classifications(str(tsv))

    assert ("TRAP1", 2, "raw1") in cls
    assert ("TARGET1", 2, "raw1") not in cls
    assert ("TRAP_CAPS", 2, "raw1") in cls


def test_load_validates_level_values(tmp_path, caplog):
    """L5 等非法 level 应跳过并 warn；l0 lowercase 应被 normalize 为 L0 (友好)"""
    import logging
    from tools.extract_common import load_entrapment_classifications

    tsv = tmp_path / "classified.tsv"
    _write_classified_tsv_with_group(tsv, [
        {"peptide": "GOOD", "charge": "2", "spectrum_file": "raw1", "level": "L0"},
        {"peptide": "WRONG_L5", "charge": "2", "spectrum_file": "raw1", "level": "L5"},
        {"peptide": "LOWERCASE", "charge": "2", "spectrum_file": "raw1", "level": "l0"},
    ])
    with caplog.at_level(logging.WARNING, logger=""):
        cls = load_entrapment_classifications(str(tsv))

    assert cls[("GOOD", 2, "raw1")] == "L0"
    # L5 是真非法 → 拒绝
    assert ("WRONG_L5", 2, "raw1") not in cls
    # l0 应被 normalize 为 L0（友好处理大小写错误）
    assert cls.get(("LOWERCASE", 2, "raw1")) == "L0"
    # 仍应有 warning 提到 L5
    assert any("L5" in r.message for r in caplog.records
               if r.levelno >= logging.WARNING)


def test_filter_case_insensitive_drop_levels():
    """drop_levels={'l0'} 小写应等价于大写"""
    from tools.extract_common import filter_by_entrapment

    psm = _make_psm("X", 2, "TRAP", raw="r1")
    psm._label_type = "negative"
    classifications = {("X", 2, "r1"): "L0"}

    result = filter_by_entrapment([psm], classifications, drop_levels={"l0"})
    assert len(result) == 0  # 应被剔除


def test_load_missing_file_helpful_error(tmp_path):
    """文件不存在 → 异常 message 含路径"""
    from tools.extract_common import load_entrapment_classifications

    bad_path = str(tmp_path / "nope.tsv")
    with pytest.raises(FileNotFoundError) as excinfo:
        load_entrapment_classifications(bad_path)
    assert bad_path in str(excinfo.value) or "nope.tsv" in str(excinfo.value)


def test_extract_n_engines_with_entrapment_section_e2e(tmp_path, monkeypatch):
    """完整 extract_n_engines() 走 [entrapment] 段"""
    import configparser
    from tools import extract_common

    # 准备：3 个 negative，其中 L0/L1 应被剔除
    psm_l0_neg = _make_psm("L0X", 2, "TRAP", raw="r1")
    psm_l0_neg._label_type = "negative"
    psm_l4_neg = _make_psm("L4X", 2, "TRAP", raw="r1")
    psm_l4_neg._label_type = "negative"
    psm_pos = _make_psm("HUM", 2, "X_HUMAN", raw="r1")
    psm_pos._label_type = "positive"

    # monkey-patch extract_n_engines_from_psms 不执行真引擎加载
    def fake_load_engine_psms(engine_name, config):
        return []  # 不用真引擎
    def fake_extract(engine_psms, engine_order, marker):
        return [psm_l0_neg, psm_l4_neg, psm_pos]

    monkeypatch.setattr(extract_common, "load_engine_psms",
                        fake_load_engine_psms)
    monkeypatch.setattr(extract_common, "extract_n_engines_from_psms",
                        fake_extract)

    # 构造 classified.tsv
    tsv = tmp_path / "classified.tsv"
    _write_classified_tsv_with_group(tsv, [
        {"peptide": "L0X", "charge": "2", "spectrum_file": "r1", "level": "L0"},
        {"peptide": "L4X", "charge": "2", "spectrum_file": "r1", "level": "L4"},
    ])

    cfg = configparser.ConfigParser()
    cfg["extract"] = {
        "engines": "pfind",
        "positive_species_marker": "HUMAN",
    }
    cfg["engine.pfind"] = {"path": "/dev/null"}
    cfg["entrapment"] = {
        "classified_tsv": str(tsv),
        "drop_levels": "L0,L1",
    }

    result = extract_common.extract_n_engines(cfg)
    seqs = {p._sequence for p in result}
    assert "L0X" not in seqs
    assert "L4X" in seqs
    assert "HUM" in seqs


def test_extract_n_engines_no_entrapment_section_skips(tmp_path, monkeypatch):
    """没有 [entrapment] 段 → 不调用 filter"""
    import configparser
    from tools import extract_common

    psm_neg = _make_psm("L0X", 2, "TRAP", raw="r1")
    psm_neg._label_type = "negative"

    monkeypatch.setattr(extract_common, "load_engine_psms",
                        lambda n, c: [])
    monkeypatch.setattr(extract_common, "extract_n_engines_from_psms",
                        lambda ep, eo, m: [psm_neg])

    cfg = configparser.ConfigParser()
    cfg["extract"] = {"engines": "pfind", "positive_species_marker": "HUMAN"}
    cfg["engine.pfind"] = {"path": "/dev/null"}
    # 注意：故意不加 [entrapment]

    result = extract_common.extract_n_engines(cfg)
    assert len(result) == 1


def test_extract_n_engines_empty_classified_tsv_path(tmp_path, monkeypatch):
    """[entrapment] 段存在但 classified_tsv 为空 → 跳过过滤，不报错"""
    import configparser
    from tools import extract_common

    psm_neg = _make_psm("X", 2, "TRAP", raw="r1")
    psm_neg._label_type = "negative"

    monkeypatch.setattr(extract_common, "load_engine_psms",
                        lambda n, c: [])
    monkeypatch.setattr(extract_common, "extract_n_engines_from_psms",
                        lambda ep, eo, m: [psm_neg])

    cfg = configparser.ConfigParser()
    cfg["extract"] = {"engines": "pfind", "positive_species_marker": "HUMAN"}
    cfg["engine.pfind"] = {"path": "/dev/null"}
    cfg["entrapment"] = {"classified_tsv": ""}

    result = extract_common.extract_n_engines(cfg)
    assert len(result) == 1


# ----------------------------------------------------------------------
# Inline FASTA entrapment classification (one-step mode)
# ----------------------------------------------------------------------

def test_extract_with_target_fasta_runs_classifier_inline(tmp_path, monkeypatch):
    """[entrapment] target_fasta → extract_common runs the classifier
    in-memory and applies the L0/L1 filter in one command, without
    requiring a pre-built classified.tsv."""
    import configparser
    from tools import extract_common

    psm_l0_neg = _make_psm("HUMANSEQ", 2, "TRAP", raw="r1")
    psm_l0_neg._label_type = "negative"
    psm_l4_neg = _make_psm("WWWWWW", 2, "TRAP", raw="r1")
    psm_l4_neg._label_type = "negative"
    psm_pos = _make_psm("HUM", 2, "X_HUMAN", raw="r1")
    psm_pos._label_type = "positive"

    monkeypatch.setattr(extract_common, "load_engine_psms",
                        lambda n, c: [])
    monkeypatch.setattr(extract_common, "extract_n_engines_from_psms",
                        lambda ep, eo, m: [psm_l0_neg, psm_l4_neg, psm_pos])

    fasta = tmp_path / "tiny.fasta"
    fasta.write_text(">p1\nMKHUMANSEQAAAR\n")

    cfg = configparser.ConfigParser()
    cfg["extract"] = {"engines": "pfind", "positive_species_marker": "HUMAN"}
    cfg["engine.pfind"] = {"path": "/dev/null"}
    cfg["entrapment"] = {
        "target_fasta": str(fasta),
        "drop_levels": "L0, L1",
    }

    result = extract_common.extract_n_engines(cfg)
    seqs = {p._sequence for p in result}
    assert "HUMANSEQ" not in seqs
    assert "WWWWWW" in seqs
    assert "HUM" in seqs


def test_extract_with_both_classified_tsv_and_fasta_prefers_tsv(tmp_path, monkeypatch):
    """If both classified_tsv and target_fasta are configured, prefer
    classified_tsv (explicit > derived)."""
    import configparser
    from tools import extract_common

    psm_neg = _make_psm("SEQONE", 2, "TRAP", raw="r1")
    psm_neg._label_type = "negative"
    psm_l4 = _make_psm("L4PEP", 2, "TRAP", raw="r1")
    psm_l4._label_type = "negative"

    monkeypatch.setattr(extract_common, "load_engine_psms",
                        lambda n, c: [])
    monkeypatch.setattr(extract_common, "extract_n_engines_from_psms",
                        lambda ep, eo, m: [psm_neg, psm_l4])

    # FASTA says SEQONE is L0; TSV says it's L4 — TSV wins
    fasta = tmp_path / "tiny.fasta"
    fasta.write_text(">p1\nMKSEQONEAAAR\n")
    tsv = tmp_path / "classified.tsv"
    tsv.write_text("peptide\tcharge\tprecursor_mz\tretention_time\tscan_number\t"
                   "spectrum_file\tprotein_ids\tq_value\tgroup\tlevel\n"
                   "SEQONE\t2\t500\t0\t\tr1\tp\t0\ttrap\tL4\n")

    cfg = configparser.ConfigParser()
    cfg["extract"] = {"engines": "pfind", "positive_species_marker": "HUMAN"}
    cfg["engine.pfind"] = {"path": "/dev/null"}
    cfg["entrapment"] = {
        "classified_tsv": str(tsv),
        "target_fasta": str(fasta),
        "drop_levels": "L0, L1",
    }

    result = extract_common.extract_n_engines(cfg)
    seqs = {p._sequence for p in result}
    # TSV says SEQONE is L4 → should be KEPT (not dropped)
    assert "SEQONE" in seqs


def test_extract_with_neither_classified_tsv_nor_fasta_skips_filter(monkeypatch):
    """Empty [entrapment] section → no filter applied (no error)."""
    import configparser
    from tools import extract_common

    psm = _make_psm("ANY", 2, "X", raw="r1")
    psm._label_type = "negative"
    monkeypatch.setattr(extract_common, "load_engine_psms",
                        lambda n, c: [])
    monkeypatch.setattr(extract_common, "extract_n_engines_from_psms",
                        lambda ep, eo, m: [psm])

    cfg = configparser.ConfigParser()
    cfg["extract"] = {"engines": "pfind", "positive_species_marker": "HUMAN"}
    cfg["engine.pfind"] = {"path": "/dev/null"}
    cfg["entrapment"] = {}  # no tsv, no fasta

    result = extract_common.extract_n_engines(cfg)
    assert len(result) == 1
