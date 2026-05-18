"""测试 pfind 字段解析器。"""
import pytest
from spectrum.pfind_parser import (
    parse_pfind_modify,
    mhp_to_mz,
    resolve_pfind_mod_name,
    extract_raw_title_from_pfind_path,
)


# === parse_pfind_modify ===

def test_parse_pfind_modify_empty():
    """空字符串应返回空列表。"""
    assert parse_pfind_modify("") == []


def test_parse_pfind_modify_whitespace():
    """纯空白应返回空列表。"""
    assert parse_pfind_modify("   ") == []


def test_parse_pfind_modify_single_mod():
    """单个修饰应正确解析为 0-based 位置 + unimod id。"""
    # "17,Carbamidomethyl[C];" → pos 17 (1-based) → 16 (0-based), Carbamidomethyl = unimod 4
    result = parse_pfind_modify("17,Carbamidomethyl[C];")
    assert result == [(16, 4)]


def test_parse_pfind_modify_multiple_mods():
    """多个修饰应都正确解析。"""
    result = parse_pfind_modify("3,Carbamidomethyl[C];10,Carbamidomethyl[C];")
    assert result == [(2, 4), (9, 4)]


def test_parse_pfind_modify_unknown_skip():
    """未知修饰应被跳过（log warning），不抛异常。"""
    # 未知修饰名 + 已知修饰共存：只保留已知的
    result = parse_pfind_modify("3,UnknownMod[X];5,Carbamidomethyl[C];")
    assert result == [(4, 4)]


def test_parse_pfind_modify_oxidation():
    """Oxidation[M] 应解析为 unimod 35。"""
    result = parse_pfind_modify("5,Oxidation[M];")
    assert result == [(4, 35)]


# === mhp_to_mz ===

def test_mhp_to_mz_z1():
    """z=1 时 MH+ 应等于 m/z。"""
    mz = mhp_to_mz(1000.0, 1)
    assert abs(mz - 1000.0) < 1e-9


def test_mhp_to_mz_z2():
    """z=2 验证质子质量正确扣除。"""
    # MH+ = 中性质量 + 1×1.00727646677
    # m/z(z=2) = (中性 + 2×proton) / 2 = (MH+ - proton + 2×proton) / 2 = (MH+ + proton) / 2
    proton = 1.00727646677
    mhp = 2000.0
    expected_mz = (mhp + proton) / 2.0
    mz = mhp_to_mz(mhp, 2)
    assert abs(mz - expected_mz) < 1e-9


def test_mhp_to_mz_z3():
    """z=3 同上。"""
    proton = 1.00727646677
    mhp = 3000.0
    expected_mz = (mhp + 2 * proton) / 3.0
    mz = mhp_to_mz(mhp, 3)
    assert abs(mz - expected_mz) < 1e-9


# === resolve_pfind_mod_name ===

def test_resolve_pfind_mod_name_hardcoded():
    """硬编码字典命中。"""
    assert resolve_pfind_mod_name("Carbamidomethyl[C]") == 4


def test_resolve_pfind_mod_name_unimod_fallback():
    """unimod.xml 兑底查询——给一个硬编码字典里没有但 UniMod 数据库里有的修饰名。"""
    # 选一个不在硬编码字典里但 UniMod 数据库里有的修饰
    # （需要先用 _get_unimod_db().by_title 验证一下，确认存在）
    result = resolve_pfind_mod_name("Biotin")
    assert result is not None
    assert isinstance(result, int)


def test_resolve_pfind_mod_name_unknown_returns_none():
    """完全不存在的修饰名应返回 None。"""
    result = resolve_pfind_mod_name("ThisModificationDoesNotExist_XYZ_12345")
    assert result is None


# === extract_raw_title_from_pfind_path ===

def test_extract_raw_title_basic():
    """从 .qry.res 文件名提取 raw_title。"""
    assert (
        extract_raw_title_from_pfind_path("/path/to/sample.qry.res")
        == "sample"
    )


def test_extract_raw_title_complex_name():
    """复杂文件名也应正确处理。"""
    assert (
        extract_raw_title_from_pfind_path(
            "/path/20190830_HF_ZHW_hela_SILAC_DDIA_500_550_2Da_Rep1.qry.res")
        == "20190830_HF_ZHW_hela_SILAC_DDIA_500_550_2Da_Rep1"
    )
