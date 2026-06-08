"""测试 SpecLib 锁步流式 loader 与质量交叉校验。"""
import pytest
from spectrum.speclib import SpecLib


@pytest.fixture
def lib_files(tmp_path, build_pdb, build_rt, build_ms2):
    fasta = tmp_path / "db.fasta"
    fasta.write_text(">PROT1 d\nPEPTIDEKACDM\n", encoding="utf-8")
    mod = tmp_path / "modification.ini"
    mod.write_text(
        "@NUMBER_MODIFICATION=2\n"
        "name1=Carbamidomethyl[C] 0\n"
        "Carbamidomethyl[C]=C NORMAL 57.021464 57.05 0 H(3)C(2)N(1)O(1)\n"
        "name2=Oxidation[M] 0\n"
        "Oxidation[M]=M NORMAL 15.994915 16.0 0 O(1)\n",
        encoding="utf-8")
    elem = tmp_path / "element.ini"
    elem.write_text(
        "E1=H|1.00782503207,|1.0,|\nE2=C|12.0,|1.0,|\n"
        "E3=N|14.0030740048,|1.0,|\nE4=O|15.99491461956,|1.0,|\n"
        "E5=S|31.972071,|1.0,|\n", encoding="utf-8")
    aa = tmp_path / "aa.ini"
    aa.write_text(
        "R1=A|C(3)H(5)N(1)O(1)S(0)|\nR2=C|C(3)H(5)N(1)O(1)S(1)|\n"
        "R3=D|C(4)H(5)N(1)O(3)S(0)|\nR4=E|C(5)H(7)N(1)O(3)S(0)|\n"
        "R5=I|C(6)H(11)N(1)O(1)S(0)|\nR6=K|C(6)H(12)N(2)O(1)S(0)|\n"
        "R7=M|C(5)H(9)N(1)O(1)S(1)|\nR8=P|C(5)H(7)N(1)O(1)S(0)|\n"
        "R9=T|C(4)H(7)N(1)O(2)S(0)|\n", encoding="utf-8")

    from spectrum.speclib.config_io import (
        parse_element_masses, parse_residue_masses, water_mass)
    em = parse_element_masses(str(elem))
    res = parse_residue_masses(str(aa), em)
    w = water_mass(em)
    seq = "PEPTIDEKACDM"  # pep_start 0, len 12
    m1 = w + sum(res[a] for a in seq)
    m2 = m1 + 57.021464                       # 变体2: Carbamidomethyl[C] 在第 9 位
    pdb = build_pdb([
        {"pro_id": 0, "pep_start": 0, "pep_len": 12,
         "variants": [(m1, []), (m2, [(9, 1)])]},
    ])
    (tmp_path / "pepdata.pdb").write_bytes(pdb)
    # 2 肽段变体 × chg_max=2 = 4 条 MS2 记录 + 文本尾巴
    (tmp_path / "pepdata.ms2.predb").write_bytes(build_ms2(
        [[(0, 0, 1.0)], [(1, 1, 0.5)], [(0, 0, 0.8)], [(2, 3, 0.3)]],
        chg_max=2, n_peptides=2))
    (tmp_path / "pepdata.rt.predb").write_bytes(build_rt([20.0, 21.5]))
    return tmp_path


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
