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
