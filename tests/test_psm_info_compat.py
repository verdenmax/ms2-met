"""测试 PSMInfo 新字段与向后兼容。"""
import numpy as np
from spectrum.psm_info import PSMInfo


def _make_basic_psm(**overrides):
    defaults = dict(
        sequence="AGFAGDDAPK",
        charge=2,
        modify=[],
        rt=np.float32(50.0),
        precursor_mz=np.float32(500.0),
        raw_title="test_raw",
        protein_names="sp|P00000|TEST_HUMAN/",
    )
    defaults.update(overrides)
    return PSMInfo(**defaults)


def test_psminfo_new_fields_default_none():
    """未显式给新字段时应默认为 None。"""
    psm = _make_basic_psm()
    assert psm._q_value is None
    assert psm._score is None
    assert psm._label_type is None
    assert psm._candidate_family_id is None


def test_psminfo_new_fields_set():
    """显式给的新字段应被存储。"""
    psm = _make_basic_psm(q_value=0.001, score=20.5, label_type="positive")
    assert psm._q_value == 0.001
    assert psm._score == 20.5
    assert psm._label_type == "positive"


def test_psminfo_to_dict_omits_none_new_fields():
    """to_dict 在新字段为 None 时不应输出，保持老格式兼容。"""
    psm = _make_basic_psm()
    d = psm.to_dict()
    assert "q_value" not in d
    assert "score" not in d
    assert "label_type" not in d


def test_psminfo_to_dict_includes_new_fields_when_set():
    """to_dict 在新字段非 None 时应输出。"""
    psm = _make_basic_psm(q_value=0.001, score=20.5, label_type="positive")
    d = psm.to_dict()
    assert d["q_value"] == 0.001
    assert d["score"] == 20.5
    assert d["label_type"] == "positive"


def test_psminfo_relationship_metadata_roundtrip():
    psm = _make_basic_psm(
        query_id="Q1", parent_id="P1", group_id="G1",
        pair_id="PAIR1", candidate_family_id="F1", peptide_group_id="PG1")
    restored = PSMInfo.from_dict(psm.to_dict())
    assert restored._query_id == "Q1"
    assert restored._parent_id == "P1"
    assert restored._group_id == "G1"
    assert restored._pair_id == "PAIR1"
    assert restored._candidate_family_id == "F1"
    assert restored._peptide_group_id == "PG1"


def test_psminfo_from_dict_old_json_no_new_fields():
    """老 JSON 没有新字段时，from_dict 应回填 None。"""
    old_data = {
        "sequence": "AGFAGDDAPK",
        "charge": 2,
        "modify": [],
        "rt": 50.0,
        "precursor_mz": 500.0,
        "raw_title": "test_raw",
        "protein_names": "sp|P00000|TEST_HUMAN/",
    }
    psm = PSMInfo.from_dict(old_data)
    assert psm._q_value is None
    assert psm._score is None
    assert psm._label_type is None
    assert psm._sequence == "AGFAGDDAPK"


def test_psminfo_from_dict_new_json_with_fields():
    """新 JSON 带新字段时，from_dict 应正确加载。"""
    new_data = {
        "sequence": "AGFAGDDAPK",
        "charge": 2,
        "modify": [],
        "rt": 50.0,
        "precursor_mz": 500.0,
        "raw_title": "test_raw",
        "protein_names": "sp|P00000|TEST_HUMAN/",
        "q_value": 0.005,
        "score": 18.7,
        "label_type": "negative",
    }
    psm = PSMInfo.from_dict(new_data)
    assert psm._q_value == 0.005
    assert psm._score == 18.7
    assert psm._label_type == "negative"


def test_psminfo_get_key_unchanged_by_new_fields():
    """get_key 不受新字段影响（保持 PSM 等价判定语义）。"""
    psm1 = _make_basic_psm()
    psm2 = _make_basic_psm(q_value=0.01, score=15.0, label_type="positive")
    assert psm1.get_key() == psm2.get_key()


def test_psminfo_get_key_with_raw_includes_raw_title():
    """get_key_with_raw 应包含 raw_title。"""
    psm1 = _make_basic_psm()
    psm2 = _make_basic_psm()
    psm3 = _make_basic_psm(raw_title="different_raw")

    assert psm1.get_key_with_raw() == psm2.get_key_with_raw()
    assert psm1.get_key_with_raw() != psm3.get_key_with_raw()
    assert psm1.get_key() == psm3.get_key()


def test_psminfo_get_key_with_raw_structure():
    """get_key_with_raw 返回 4-tuple: (seq, charge, mod_tuple, raw_title)."""
    psm = _make_basic_psm()
    key = psm.get_key_with_raw()
    assert len(key) == 4
    assert key[0] == "AGFAGDDAPK"
    assert key[1] == 2
    assert key[2] == ()
    assert key[3] == "test_raw"
