"""测试 speclib.pepdata 流式二进制解析。"""
import pytest
from spectrum.speclib.config_io import Protein, ModEntry
from spectrum.speclib.pepdata import iter_pepdata, read_pepdata, LibPeptide, ModSite


@pytest.fixture
def proteins():
    return [
        Protein("PROT1", "d", "PEPTIDEKSAMPLER"),
        Protein("REV_PROT2", "d", "ACDEFGHIKLMNPQR"),
    ]


@pytest.fixture
def mods_by_id():
    return {
        2: ModEntry(2, "Acetyl[K]", 42.010565, "K", "NORMAL"),
        9: ModEntry(9, "Carbamidomethyl[C]", 57.021464, "C", "NORMAL"),
    }


def test_read_single_peptide_no_mods(tmp_path, build_pdb, proteins, mods_by_id):
    data = build_pdb([
        {"pro_id": 0, "pep_start": 0, "pep_len": 8, "miss": 1,
         "variants": [(900.45, [])]},
    ])
    p = tmp_path / "pepdata.pdb"
    p.write_bytes(data)
    peps = read_pepdata(str(p), proteins, mods_by_id)
    assert len(peps) == 1
    assert peps[0].sequence == "PEPTIDEK"
    assert peps[0].mods == []
    assert peps[0].neutral_mass == pytest.approx(900.45)
    assert peps[0].protein == "PROT1"
    assert peps[0].is_decoy is False
    assert peps[0].miss == 1


def test_read_peptide_with_mods(tmp_path, build_pdb, proteins, mods_by_id):
    data = build_pdb([
        {"pro_id": 1, "pep_start": 0, "pep_len": 9,
         "variants": [(1100.5, [(1, 9), (9, 2)])]},
    ])
    p = tmp_path / "pepdata.pdb"
    p.write_bytes(data)
    peps = read_pepdata(str(p), proteins, mods_by_id)
    assert peps[0].sequence == "ACDEFGHIK"
    assert peps[0].is_decoy is True
    sites = peps[0].mods
    assert [(m.pos, m.mod_id, m.name) for m in sites] == [
        (1, 9, "Carbamidomethyl[C]"), (9, 2, "Acetyl[K]")]
    assert sites[0].mono_mass == pytest.approx(57.021464)


def test_multiple_variants_and_entries(tmp_path, build_pdb, proteins, mods_by_id):
    data = build_pdb([
        {"pro_id": 0, "pep_start": 0, "pep_len": 8,
         "variants": [(900.4, []), (942.4, [(8, 2)])]},
        {"pro_id": 0, "pep_start": 8, "pep_len": 7, "variants": [(800.3, [])]},
    ])
    p = tmp_path / "pepdata.pdb"
    p.write_bytes(data)
    peps = read_pepdata(str(p), proteins, mods_by_id)
    assert len(peps) == 3            # M = Σ mod_pep_num = 2 + 1
    assert peps[1].sequence == "PEPTIDEK"
    assert peps[1].mods[0].pos == 8
    assert peps[2].sequence == "SAMPLER"


def test_iter_pepdata_is_lazy(tmp_path, build_pdb, proteins, mods_by_id):
    data = build_pdb([
        {"pro_id": 0, "pep_start": 0, "pep_len": 8,
         "variants": [(900.4, []), (942.4, [(8, 2)])]},
    ])
    p = tmp_path / "pepdata.pdb"
    p.write_bytes(data)
    gen = iter_pepdata(str(p), proteins, mods_by_id)
    first = next(gen)
    assert isinstance(first, LibPeptide)
    assert first.neutral_mass == pytest.approx(900.4)


def test_mod_pep_bytes_mismatch_raises(tmp_path, proteins, mods_by_id):
    import struct
    header = struct.pack("<IIbbbbIQ", 0, 0, 8, 0, 0, 0, 1, 999)  # 故意 999
    body = struct.pack("<db", 900.0, 0)
    p = tmp_path / "pepdata.pdb"
    p.write_bytes(header + body)
    with pytest.raises(ValueError, match="mod_pep_bytes"):
        read_pepdata(str(p), proteins, mods_by_id)


def test_zero_variant_entry_consumed_and_skipped(tmp_path, proteins, mods_by_id):
    import struct
    e0 = struct.pack("<IIbbbbIQ", 0, 0, 8, 0, 0, 0, 0, 0)            # 0 变体
    e1 = (struct.pack("<IIbbbbIQ", 0, 0, 7, 0, 0, 0, 1, 9)
          + struct.pack("<db", 800.3, 0))                           # 正常条目
    p = tmp_path / "pepdata.pdb"
    p.write_bytes(e0 + e1)
    peps = read_pepdata(str(p), proteins, mods_by_id)
    assert len(peps) == 1
    assert peps[0].sequence == "PEPTIDE"


def test_truncated_record_raises(tmp_path, proteins, mods_by_id):
    import struct
    # 头声称 1 个变体，但缺少变体字节 → 应 struct.error 大声失败，而非静默产出垃圾
    data = struct.pack("<IIbbbbIQ", 0, 0, 8, 0, 0, 0, 1, 9)  # 无 body
    p = tmp_path / "pepdata.pdb"
    p.write_bytes(data)
    with pytest.raises(struct.error):
        read_pepdata(str(p), proteins, mods_by_id)


def test_early_break_skips_byte_check(tmp_path, proteins, mods_by_id):
    import struct
    # 单条目 mod_pep_bytes 故意错误；只取第一个肽段(早退)不应触发校验
    data = (struct.pack("<IIbbbbIQ", 0, 0, 8, 0, 0, 0, 1, 999)
            + struct.pack("<db", 900.0, 0))
    p = tmp_path / "pepdata.pdb"
    p.write_bytes(data)
    gen = iter_pepdata(str(p), proteins, mods_by_id)
    first = next(gen)                       # 早退：不继续消费
    assert first.sequence == "PEPTIDEK"     # 无异常抛出
