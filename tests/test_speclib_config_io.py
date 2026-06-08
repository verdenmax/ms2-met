"""测试 speclib.config_io 文本配置解析。"""
from spectrum.speclib.config_io import (
    Protein, ModEntry,
    parse_fasta, parse_modifications,
    parse_element_masses, parse_residue_masses, water_mass,
)


def _write(p, text):
    p.write_text(text, encoding="utf-8")
    return str(p)


def test_parse_fasta_basic(tmp_path):
    path = _write(tmp_path / "db.fasta",
                  ">PROT1 first protein\nPEPTIDEK\nSAMPLER\n"
                  ">REV_PROT2 decoy\nACDEFGHIK\n")
    pros = parse_fasta(path)
    assert len(pros) == 2
    assert pros[0].ac == "PROT1"
    assert pros[0].sequence == "PEPTIDEKSAMPLER"
    assert pros[0].is_decoy is False
    assert pros[1].ac == "REV_PROT2"
    assert pros[1].is_decoy is True


def test_parse_fasta_uniprot_pipe_rule(tmp_path):
    # 第一个 '|' 在 index 3 (<15) 不作分隔，AC 取到第一个空格
    path = _write(tmp_path / "db.fasta", ">sp|P12345|NAME desc here\nMKMK\n")
    pros = parse_fasta(path)
    assert pros[0].ac == "sp|P12345|NAME"


def test_parse_fasta_pipe_after_15_splits(tmp_path):
    # '|' 在 index 17 (>15) 时作为分隔符（覆盖 |>15 正分支）
    path = _write(tmp_path / "db.fasta", ">ABCDEFGHIJKLMNOP|Q desc\nMKMK\n")
    pros = parse_fasta(path)
    assert pros[0].ac == "ABCDEFGHIJKLMNOP"


def test_parse_modifications_ordering_and_skips(tmp_path):
    path = _write(tmp_path / "modification.ini",
        "@NUMBER_MODIFICATION=3\n"
        "label_name=foo\n"
        "name1=Acetyl[K] 0\n"
        "Acetyl[K]=K NORMAL 42.010565 42.0367 0 H(2)C(2)O(1)\n"
        "name2=Carbamidomethyl[C] 0\n"
        "Carbamidomethyl[C]=C NORMAL 57.021464 57.0513 0 H(3)C(2)N(1)O(1)\n"
        "Label_13C(6)[K]=K NORMAL 6.020129 6.0 0 C(-6)13C(6)\n"
        "Met-loss+Acetyl[ProteinN-termM]=M PRO_N -89.029920 -89.09 0 C(0)\n"
        "name3=Oxidation[M] 0\n"
        "Oxidation[M]=M NORMAL 15.994915 16.0 0 O(1)\n")
    mods = parse_modifications(path)
    # label_name / Label_ / Met-loss+Acetyl 行都被跳过且不占 id
    assert [m.name for m in mods] == ["Acetyl[K]", "Carbamidomethyl[C]", "Oxidation[M]"]
    assert [m.mod_id for m in mods] == [1, 2, 3]
    assert mods[0].mono_mass == 42.010565
    assert mods[1].sites == "C"
    assert mods[2].mod_type == "NORMAL"


def test_parse_element_and_residue_masses(tmp_path):
    elem = _write(tmp_path / "element.ini",
        "@NUMBER_ELEMENT=6\n"
        "E1=H|1.00782503207,|1.0,|\n"
        "E2=C|12.0,|1.0,|\n"
        "E3=N|14.0030740048,|1.0,|\n"
        "E4=O|15.99491461956,|1.0,|\n"
        "E5=S|31.972071,|1.0,|\n"
        # 多同位素：最高丰度在中间（index 1），验证取 max 而非首个，且过滤后索引对齐
        "E6=Z|10.0,20.0,30.0,|0.1,0.7,0.2,|\n")
    aa = _write(tmp_path / "aa.ini",
        "@NUMBER_RESIDUE=2\n"
        "R1=G|C(2)H(3)N(1)O(1)S(0)|\n"
        "R2=K|C(6)H(12)N(2)O(1)S(0)|\n")
    em = parse_element_masses(elem)
    assert abs(em["O"] - 15.99491461956) < 1e-9
    assert em["Z"] == 20.0   # 取最高丰度同位素质量（非首个、非末个）
    assert abs(water_mass(em) - 18.0105646837) < 1e-6
    res = parse_residue_masses(aa, em)
    assert abs(res["G"] - 57.02146372057) < 1e-6
    assert abs(res["K"] - 128.094963014) < 1e-6
