"""Common pytest fixtures for ms2-met tests."""
import os
import pytest

FIXTURES_DIR = os.path.join(os.path.dirname(__file__), "fixtures")


@pytest.fixture
def sample_pfind_file():
    return os.path.join(FIXTURES_DIR, "sample_pfind.qry.res")


@pytest.fixture
def sample_pfind_dir():
    return os.path.join(FIXTURES_DIR, "sample_pfind_dir")


# === speclib 二进制构造 helper ===
import struct as _struct

_PDB_HEADER = _struct.Struct("<IIbbbbIQ")
_PDB_VAR = _struct.Struct("<db")
_PDB_MOD = _struct.Struct("<bi")


def _build_pdb(entries):
    """entries: list of dict(pro_id, pep_start, pep_len, pro_nc?, enz?, miss?, variants)
    variants: list of (mass, [(pos, mod_id), ...])。返回模拟 pepdata.pdb 的 bytes。
    """
    out = b""
    for e in entries:
        block = b""
        for mass, modlist in e["variants"]:
            block += _PDB_VAR.pack(mass, len(modlist))
            for pos, mid in modlist:
                block += _PDB_MOD.pack(pos, mid)
        out += _PDB_HEADER.pack(
            e["pro_id"], e["pep_start"], e["pep_len"],
            e.get("pro_nc", 0), e.get("enz", 0), e.get("miss", 0),
            len(e["variants"]), len(block))
        out += block
    return out


@pytest.fixture
def build_pdb():
    return _build_pdb


def _build_rt(values):
    return _struct.pack(f"<{len(values)}f", *values)


_MS2_HEAD = _struct.Struct("<h")
_MS2_ION = _struct.Struct("<bbf")


def _build_ms2(records, chg_max=None, n_peptides=None):
    """records: list of list of (pos, iontype, inten)。
    若给 chg_max+n_peptides，则在末尾追加 n_peptides 行文本尾巴
    （每行 '1\\t0\\t...\\tchg_max\\t0\\t\\n'），模拟真实文件。"""
    out = b""
    for ions in records:
        out += _MS2_HEAD.pack(len(ions))
        for pos, iontype, inten in ions:
            out += _MS2_ION.pack(pos, iontype, inten)
    if chg_max is not None and n_peptides is not None:
        line = "".join(f"{c}\t0\t" for c in range(1, chg_max + 1)) + "\n"
        out += (line * n_peptides).encode("latin-1")
    return out


@pytest.fixture
def build_rt():
    return _build_rt


@pytest.fixture
def build_ms2():
    return _build_ms2


@pytest.fixture
def lib_files(tmp_path, build_pdb, build_rt, build_ms2):
    """合成一个微型谱库目录（含文本尾巴）+ 解码/校验配置，供 loader 与 CLI 测试复用。"""
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
