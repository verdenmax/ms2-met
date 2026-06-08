"""测试 SpecLib 锁步流式 loader 与质量交叉校验。"""
import pytest
from spectrum.speclib import SpecLib


def test_open_dir_and_iter_peptides(lib_files):
    lib = SpecLib.open_dir(str(lib_files),
                           fasta_path=str(lib_files / "db.fasta"),
                           mod_path=str(lib_files / "modification.ini"))
    assert lib.num_peptides == 2
    assert lib.chg_max == 2
    peps = list(lib.iter_peptides())
    assert len(peps) == 2
    # B1: 锁步对齐 —— 变体1(无修饰) 对 rt=20.0，变体2(Carbamidomethyl[C]) 对 rt=21.5
    assert peps[0].mods == []
    assert peps[0].pred_rt == pytest.approx(20.0)
    assert set(peps[0].pred_ms2.keys()) == {1, 2}
    assert peps[0].pred_ms2[1][0].ion_type == "b"
    assert peps[1].mods[0].name == "Carbamidomethyl[C]"
    assert peps[1].pred_rt == pytest.approx(21.5)


def test_iter_peptides_decode_none_skips_ms2(lib_files):
    # decode_ms2='none'：不读 ms2，pred_ms2 留空，但 RT/序列仍正确对齐
    import os
    lib = SpecLib.open_dir(str(lib_files),
                           fasta_path=str(lib_files / "db.fasta"),
                           mod_path=str(lib_files / "modification.ini"))
    # 删掉 ms2 文件证明 'none' 模式根本不读它（open_dir 已读过尾巴拿 chg_max）
    os.remove(lib_files / "pepdata.ms2.predb")
    peps = list(lib.iter_peptides(decode_ms2="none"))
    assert len(peps) == 2
    assert peps[0].pred_ms2 == {}
    assert peps[0].pred_rt == pytest.approx(20.0)
    assert peps[1].sequence == "PEPTIDEKACDM"


def test_iter_peptides_decode_arrays(lib_files):
    import numpy as np
    lib = SpecLib.open_dir(str(lib_files),
                           fasta_path=str(lib_files / "db.fasta"),
                           mod_path=str(lib_files / "modification.ini"))
    peps = list(lib.iter_peptides(decode_ms2="arrays"))
    assert set(peps[0].pred_ms2.keys()) == {1, 2}
    rec = peps[0].pred_ms2[1]
    assert isinstance(rec, np.ndarray)
    assert rec.dtype.names == ("pos", "iontype", "inten")
    assert int(rec["iontype"][0]) == 0          # 第 1 条记录 ion0 = iontype 0


def test_iter_peptides_invalid_decode_mode_raises(lib_files):
    lib = SpecLib.open_dir(str(lib_files),
                           fasta_path=str(lib_files / "db.fasta"),
                           mod_path=str(lib_files / "modification.ini"))
    with pytest.raises(ValueError, match="decode_ms2"):
        list(lib.iter_peptides(decode_ms2="bogus"))


def test_validate_masses_all_pass(lib_files):
    lib = SpecLib.open_dir(str(lib_files),
                           fasta_path=str(lib_files / "db.fasta"),
                           mod_path=str(lib_files / "modification.ini"))
    report = lib.validate_masses(str(lib_files / "element.ini"),
                                 str(lib_files / "aa.ini"), tol=1e-4)
    assert report.total == 2
    assert report.passed == 2
    assert report.failed == 0
    assert report.max_abs_error < 1e-4


def test_validate_masses_flags_wrong_mass(lib_files):
    # 破坏 pdb 中第一个变体的 mass：直接改文件首个 double
    import struct
    pdb = bytearray((lib_files / "pepdata.pdb").read_bytes())
    off = struct.calcsize("<IIbbbbIQ")  # 第一个变体 mass 的起点
    bad = struct.pack("<d", struct.unpack_from("<d", pdb, off)[0] + 5.0)
    pdb[off:off + 8] = bad
    (lib_files / "pepdata.pdb").write_bytes(bytes(pdb))
    lib = SpecLib.open_dir(str(lib_files),
                           fasta_path=str(lib_files / "db.fasta"),
                           mod_path=str(lib_files / "modification.ini"))
    report = lib.validate_masses(str(lib_files / "element.ini"),
                                 str(lib_files / "aa.ini"), tol=1e-4)
    assert report.failed == 1
    assert report.failures[0][0] == 0  # index 0


def test_iter_peptides_rt_count_mismatch_raises(lib_files, build_rt):
    (lib_files / "pepdata.rt.predb").write_bytes(build_rt([1.0]))  # 只有 1 个
    lib = SpecLib.open_dir(str(lib_files),
                           fasta_path=str(lib_files / "db.fasta"),
                           mod_path=str(lib_files / "modification.ini"))
    with pytest.raises(ValueError, match="RT count"):
        list(lib.iter_peptides())


def test_iter_peptides_fewer_peptides_than_rt_raises(lib_files, build_rt):
    # pdb 2 肽段，但 RT 3 个 → 遍历结束 i+1(2) != n_rt(3)
    (lib_files / "pepdata.rt.predb").write_bytes(build_rt([1.0, 2.0, 3.0]))
    lib = SpecLib.open_dir(str(lib_files),
                           fasta_path=str(lib_files / "db.fasta"),
                           mod_path=str(lib_files / "modification.ini"))
    with pytest.raises(ValueError, match="RT count"):
        list(lib.iter_peptides())


def test_iter_peptides_ms2_exhausted_raises(lib_files, build_ms2):
    # 只给 2 条 MS2 记录，但需 2 肽段 × chg_max 2 = 4 → 第 2 个肽段耗尽
    (lib_files / "pepdata.ms2.predb").write_bytes(build_ms2(
        [[(0, 0, 1.0)], [(1, 1, 0.5)]], chg_max=2, n_peptides=2))
    lib = SpecLib.open_dir(str(lib_files),
                           fasta_path=str(lib_files / "db.fasta"),
                           mod_path=str(lib_files / "modification.ini"))
    with pytest.raises(ValueError, match="exhausted"):
        list(lib.iter_peptides())


def test_iter_peptides_ms2_oversupply_raises(lib_files, build_ms2):
    # 给 6 条 MS2 记录，多于 2×2=4 → 过供检测
    (lib_files / "pepdata.ms2.predb").write_bytes(build_ms2(
        [[(0, 0, 1.0)]] * 6, chg_max=2, n_peptides=2))
    lib = SpecLib.open_dir(str(lib_files),
                           fasta_path=str(lib_files / "db.fasta"),
                           mod_path=str(lib_files / "modification.ini"))
    with pytest.raises(ValueError, match="more records"):
        list(lib.iter_peptides())
