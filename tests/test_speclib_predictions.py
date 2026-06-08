"""测试 speclib.predictions：RT 数组 + 流式 MS2（跳过文本尾巴）。"""
import numpy as np
import pytest
from spectrum.speclib.predictions import (
    FragIon, read_rt_pred, iter_ms2_records, iter_ms2_arrays,
    read_chg_max_from_trailer,
)


def test_read_rt_pred(tmp_path, build_rt):
    p = tmp_path / "pepdata.rt.predb"
    p.write_bytes(build_rt([12.5, 33.0, 7.25]))
    assert list(read_rt_pred(str(p))) == pytest.approx([12.5, 33.0, 7.25])


def test_iter_ms2_ion_decode(tmp_path, build_ms2):
    # iontype: 2=b2+, 3=y2+
    p = tmp_path / "pepdata.ms2.predb"
    p.write_bytes(build_ms2([[(2, 2, 0.8), (3, 3, 0.4)]]))  # 无尾巴
    out = list(iter_ms2_records(str(p)))
    assert len(out) == 1
    assert out[0][0] == FragIon("b", 2, 2, pytest.approx(0.8))
    assert out[0][1] == FragIon("y", 3, 2, pytest.approx(0.4))


def test_iter_ms2_stops_at_text_trailer(tmp_path, build_ms2):
    recs = [[(0, 0, 1.0)], [(1, 1, 0.5)], [(0, 0, 0.8)], [(2, 3, 0.3)]]
    p = tmp_path / "pepdata.ms2.predb"
    p.write_bytes(build_ms2(recs, chg_max=2, n_peptides=2))  # 含尾巴
    out = list(iter_ms2_records(str(p)))
    assert len(out) == 4                       # 尾巴被跳过
    assert out[0][0] == FragIon("b", 0, 1, pytest.approx(1.0))
    assert out[3][0] == FragIon("y", 2, 2, pytest.approx(0.3))


def test_iter_ms2_empty_record_present(tmp_path, build_ms2):
    recs = [[(0, 0, 1.0)], [], [(1, 1, 0.5)], []]
    p = tmp_path / "pepdata.ms2.predb"
    p.write_bytes(build_ms2(recs, chg_max=2, n_peptides=2))
    out = list(iter_ms2_records(str(p)))
    assert len(out) == 4
    assert out[1] == []
    assert out[3] == []


def test_read_chg_max_from_trailer(tmp_path, build_ms2):
    p = tmp_path / "pepdata.ms2.predb"
    p.write_bytes(build_ms2([[(0, 0, 1.0)]] * 8, chg_max=4, n_peptides=2))
    assert read_chg_max_from_trailer(str(p)) == 4


def test_iter_ms2_arrays_decode(tmp_path, build_ms2):
    recs = [[(2, 2, 0.8), (3, 3, 0.4)], [(0, 0, 1.0)]]
    p = tmp_path / "pepdata.ms2.predb"
    p.write_bytes(build_ms2(recs, chg_max=2, n_peptides=1))  # 含尾巴
    out = list(iter_ms2_arrays(str(p)))
    assert len(out) == 2                       # 尾巴被跳过
    assert out[0].dtype.names == ("pos", "iontype", "inten")
    assert list(out[0]["pos"]) == [2, 3]
    assert list(out[0]["iontype"]) == [2, 3]
    assert out[0]["inten"][0] == pytest.approx(0.8)


def test_iter_ms2_arrays_empty_record(tmp_path, build_ms2):
    recs = [[(0, 0, 1.0)], [], [(1, 1, 0.5)], []]
    p = tmp_path / "pepdata.ms2.predb"
    p.write_bytes(build_ms2(recs, chg_max=2, n_peptides=2))
    out = list(iter_ms2_arrays(str(p)))
    assert len(out) == 4
    assert len(out[1]) == 0
    assert len(out[3]) == 0


def test_iter_ms2_arrays_copy_is_retainable(tmp_path, build_ms2):
    # copy=True（默认）：数组是独立副本，遍历完仍可安全访问（mmap 已关闭）
    recs = [[(0, 0, 1.0)], [(1, 1, 0.5)]]
    p = tmp_path / "pepdata.ms2.predb"
    p.write_bytes(build_ms2(recs, chg_max=2, n_peptides=1))
    kept = list(iter_ms2_arrays(str(p)))       # 生成器耗尽 → mmap 关闭
    assert kept[0]["inten"][0] == pytest.approx(1.0)   # 仍可读，无 BufferError
    assert int(kept[1]["pos"][0]) == 1


def test_iter_ms2_records_matches_arrays(tmp_path, build_ms2):
    # 对象 API 与数组 API 解码一致
    recs = [[(2, 2, 0.8), (3, 3, 0.4)]]
    p = tmp_path / "pepdata.ms2.predb"
    p.write_bytes(build_ms2(recs))
    objs = list(iter_ms2_records(str(p)))[0]
    arr = list(iter_ms2_arrays(str(p)))[0]
    assert objs[0] == FragIon("b", 2, 2, pytest.approx(0.8))
    assert objs[1] == FragIon("y", 3, 2, pytest.approx(0.4))
    assert [o.frag_pos for o in objs] == list(arr["pos"])
