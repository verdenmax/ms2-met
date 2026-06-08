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
