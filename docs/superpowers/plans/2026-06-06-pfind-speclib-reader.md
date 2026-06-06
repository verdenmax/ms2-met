# pFind 谱库读取模块 Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** 在 `spectrum/speclib/` 实现一个独立、自校验的 pFind 谱库二进制读取模块，读入 M 个肽段并生成 M 组预测 RT + MS2。

**Architecture:** 四个单向依赖的小模块：`config_io`（解析 FASTA/modification.ini/element.ini/aa.ini）→ `pepdata`（解析 `pepdata.pdb`）+ `predictions`（解析 `pepdata.rt.predb`/`pepdata.ms2.predb`）→ `speclib`（顶层 loader + 质量交叉校验）。外加 `tools/speclib_inspect.py` CLI 用于真实文件验证。

**Tech Stack:** Python 3.10+（仓库实际 3.14），仅用标准库 `struct`/`dataclasses`/`os`；pytest 测试（`python -m pytest`，repo root）。所有多字节字段小端、x64 位宽。

参考 spec：`docs/specs/2026-06-06-pfind-speclib-reader-design.md`。

**关键格式（已对真实 puku 文件验证）：**
- `pepdata.pdb` 条目头 `struct '<IIbbbbIQ'`（24B）= pro_id,u32; pep_start,u32; pep_len,i8; pro_nc,i8; enz,i8; miss,i8; mod_pep_num,u32; mod_pep_bytes,u64。随后 mod_pep_num 个变体：`'<db'`（9B）= mass,f64; mod_cnt,i8；再 mod_cnt 个 `'<bi'`（5B）= pos,i8; mod_id,i32。
- `pepdata.rt.predb` = M×f32（分钟）。
- `pepdata.ms2.predb` 每记录 `'<h'`（2B）= n_size，再 n_size 个 `'<bbf'`（6B）= pos,i8; iontype,i8; inten,f32。共 M×chg_max 条（肽段外层、电荷内层）。
- 序列 = `proteins[pro_id].sequence[pep_start:pep_start+pep_len]`。
- mod_id = modification.ini 过滤后数据行的 1-based read-order（Carbamidomethyl[C]=9, Oxidation[M]=46 已验证）。
- iontype：偶=b、奇=y；frag_charge=iontype//2+1。
- 中性质量 = Σ残基 + H₂O(18.0105646837) + Σ修饰单同位素质量（残基质量经 element.ini+aa.ini 验证：G=57.02146, K=128.09496）。**已由 C++ 验证**：`_m2mz(m,chg)=(m+chg*proton)/chg`（sdk.h:881）证明 `lfPepMass` 不含质子；y 离子用 `MOLECULE_MASS_H2O`（Instrument.cpp:45/61）证明含水。

**关键边界条件（rubber-duck 复核确认，合成 fixture 必须覆盖）：**
- **M = Σ mod_pep_num**（修饰变体总数），不是 pdb 头条目数；一个头条目 `mod_pep_num≥2` 会 push 多个 `LibPeptide`，RT/MS2 与之逐变体对齐（loader 测试以 1 头条目 2 变体覆盖）。
- **MS2 `n_size==0` 记录照样写出**（每"肽段-电荷"恒有一个 2 字节头），读取须当作"存在的空记录"，否则后续全部错位。
- **`mod_pep_num==0`** 的头条目合法（消耗 24B 头、push 0 肽段）。
- **chg_max ∈ [1,6]**（MAX_CHG=7 ⇒ 最多 6 价，iontype 0..11）；推断后做硬校验。
- 各电荷桶离子不对称：charge-c 记录只含 `frag_charge ≤ c` 的离子；**不要**假设各桶离子相同。

---

### Task 1: config_io — FASTA / 修饰 / 元素 / 残基质量解析

**Files:**
- Create: `spectrum/speclib/__init__.py`
- Create: `spectrum/speclib/config_io.py`
- Test: `tests/test_speclib_config_io.py`

- [ ] **Step 1: 写失败测试**

Create `tests/test_speclib_config_io.py`:

```python
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


def test_parse_modifications_ordering_and_skips(tmp_path):
    path = _write(tmp_path / "modification.ini",
        "@NUMBER_MODIFICATION=3\n"
        "name1=Acetyl[K] 0\n"
        "Acetyl[K]=K NORMAL 42.010565 42.0367 0 H(2)C(2)O(1)\n"
        "name2=Carbamidomethyl[C] 0\n"
        "Carbamidomethyl[C]=C NORMAL 57.021464 57.0513 0 H(3)C(2)N(1)O(1)\n"
        "Label_13C(6)[K]=K NORMAL 6.020129 6.0 0 C(-6)13C(6)\n"
        "name3=Oxidation[M] 0\n"
        "Oxidation[M]=M NORMAL 15.994915 16.0 0 O(1)\n")
    mods = parse_modifications(path)
    # Label_ 行被跳过且不占 id
    assert [m.name for m in mods] == ["Acetyl[K]", "Carbamidomethyl[C]", "Oxidation[M]"]
    assert [m.mod_id for m in mods] == [1, 2, 3]
    assert mods[0].mono_mass == 42.010565
    assert mods[1].sites == "C"
    assert mods[2].mod_type == "NORMAL"


def test_parse_element_and_residue_masses(tmp_path):
    elem = _write(tmp_path / "element.ini",
        "@NUMBER_ELEMENT=5\n"
        "E1=H|1.00782503207,|1.0,|\n"
        "E2=C|12.0,|1.0,|\n"
        "E3=N|14.0030740048,|1.0,|\n"
        "E4=O|15.99491461956,|1.0,|\n"
        "E5=S|31.972071,|1.0,|\n")
    aa = _write(tmp_path / "aa.ini",
        "@NUMBER_RESIDUE=2\n"
        "R1=G|C(2)H(3)N(1)O(1)S(0)|\n"
        "R2=K|C(6)H(12)N(2)O(1)S(0)|\n")
    em = parse_element_masses(elem)
    assert abs(em["O"] - 15.99491461956) < 1e-9
    assert abs(water_mass(em) - 18.0105646837) < 1e-6
    res = parse_residue_masses(aa, em)
    assert abs(res["G"] - 57.02146372057) < 1e-6
    assert abs(res["K"] - 128.094963014) < 1e-6
```

- [ ] **Step 2: 运行测试，确认失败**

Run: `python -m pytest tests/test_speclib_config_io.py -q`
Expected: FAIL（`ModuleNotFoundError: No module named 'spectrum.speclib'`）

- [ ] **Step 3: 创建包并实现 config_io**

Create `spectrum/speclib/__init__.py`:

```python
"""pFind 谱库（spectral library）二进制读取模块。"""
```

Create `spectrum/speclib/config_io.py`:

```python
"""解析 pFind 谱库依赖的文本配置：FASTA、modification.ini、element.ini、aa.ini。

逻辑复刻自 pFindSDK：fastaparser.cpp / Reader.cpp(ReadMod/ReadAA/ReadElementInfo)。
"""
from dataclasses import dataclass


@dataclass
class Protein:
    ac: str
    description: str
    sequence: str

    @property
    def is_decoy(self) -> bool:
        return self.ac.startswith("REV_")


@dataclass
class ModEntry:
    mod_id: int
    name: str
    mono_mass: float
    sites: str
    mod_type: str


def _parse_fasta_header(line: str):
    """复刻 CFastaParser::ReadOnePrteinEntry 的 AC/DE 切分。"""
    s = line.rstrip("\r\n")

    def find(ch: str) -> int:
        i = s.find(ch)
        return i if i != -1 else len(s)

    tpos = find(" ")
    t2 = find("\t")
    if t2 < tpos:
        tpos = t2
    t2 = find("|")
    if t2 < tpos and t2 > 15:
        tpos = t2
    ac = s[1:tpos]
    de = s[tpos + 1:]
    return ac, de


def parse_fasta(path: str) -> list[Protein]:
    """按文件顺序解析蛋白条目；返回的 list 下标即 pdb 中的 pro_id。"""
    proteins: list[Protein] = []
    ac = de = None
    seq_parts: list[str] = []
    with open(path, encoding="latin-1") as fh:
        for raw in fh:
            if raw.startswith(">"):
                if ac is not None:
                    proteins.append(Protein(ac, de, "".join(seq_parts)))
                ac, de = _parse_fasta_header(raw)
                seq_parts = []
            else:
                seq_parts.append(raw.strip())
    if ac is not None:
        proteins.append(Protein(ac, de, "".join(seq_parts)))
    return proteins


def parse_modifications(path: str) -> list[ModEntry]:
    """复刻 CReader::ReadMod 的过滤与 read-order id 赋值。"""
    mods: list[ModEntry] = []
    with open(path, encoding="latin-1") as fh:
        for raw in fh:
            line = raw.rstrip("\r\n")
            if "=" not in line:
                continue
            key, _, val = line.partition("=")
            if key.startswith("name"):
                continue
            if key == "@NUMBER_MODIFICATION":
                continue
            if key == "label_name":
                continue
            if key == "Met-loss+Acetyl[ProteinN-termM]":
                continue
            if key.startswith("Label_"):
                continue
            parts = val.split()
            mods.append(ModEntry(
                mod_id=len(mods) + 1,
                name=key,
                mono_mass=float(parts[2]),
                sites=parts[0],
                mod_type=parts[1],
            ))
    return mods


def parse_element_masses(path: str) -> dict[str, float]:
    """复刻 CReader::ReadElementInfo：取丰度最高同位素的质量。"""
    masses: dict[str, float] = {}
    with open(path, encoding="latin-1") as fh:
        for raw in fh:
            line = raw.rstrip("\r\n")
            if not line.startswith("E"):
                continue
            value = line[line.find("=") + 1:]
            p1 = value.find("|")
            name = value[:p1]
            p2 = value.find("|", p1 + 1)
            mass_str = value[p1 + 1:p2]
            p3 = value.find("|", p2 + 1)
            ab_str = value[p2 + 1:p3]
            ms = [x for x in mass_str.split(",") if x != ""]
            ab = [x for x in ab_str.split(",") if x != ""]
            max_i, max_a = 0, -1.0
            for i, a in enumerate(ab):
                av = float(a)
                if av > max_a:
                    max_a, max_i = av, i
            masses[name] = float(ms[max_i])
    return masses


def parse_residue_masses(path: str, element_masses: dict[str, float]) -> dict[str, float]:
    """复刻 CReader::ReadAA：残基质量 = Σ 元素质量 × 个数。"""
    residues: dict[str, float] = {}
    with open(path, encoding="latin-1") as fh:
        for raw in fh:
            line = raw.rstrip("\r\n")
            if not line.startswith("R"):
                continue
            value = line[line.find("=") + 1:]
            p1 = value.find("|")
            name = value[:p1]
            if not name or not name[0].isupper():
                continue
            p2 = value.find("|", p1 + 1)
            comp = value[p1 + 1:p2]
            total = 0.0
            for elem in comp.split(")"):
                if "(" not in elem:
                    continue
                sym, _, cnt = elem.partition("(")
                if sym == "" or cnt == "":
                    continue
                total += element_masses.get(sym, 0.0) * int(cnt)
            residues[name] = total
    return residues


def water_mass(element_masses: dict[str, float]) -> float:
    return 2 * element_masses["H"] + element_masses["O"]
```

- [ ] **Step 4: 运行测试，确认通过**

Run: `python -m pytest tests/test_speclib_config_io.py -q`
Expected: PASS（5 passed）

- [ ] **Step 5: 提交**

```bash
git add spectrum/speclib/__init__.py spectrum/speclib/config_io.py tests/test_speclib_config_io.py
git commit -m "feat(speclib): config_io — FASTA/modification/element/residue mass parsers

Co-authored-by: Copilot <223556219+Copilot@users.noreply.github.com>"
```

---

### Task 2: pepdata — 解析 pepdata.pdb 为 LibPeptide 列表

**Files:**
- Create: `spectrum/speclib/pepdata.py`
- Modify: `tests/conftest.py`（加 `build_pdb` fixture）
- Test: `tests/test_speclib_pepdata.py`

- [ ] **Step 1: 在 conftest.py 增加二进制构造 helper**

Append to `tests/conftest.py`:

```python
import struct as _struct

_PDB_HEADER = _struct.Struct("<IIbbbbIQ")
_PDB_VAR = _struct.Struct("<db")
_PDB_MOD = _struct.Struct("<bi")


def _build_pdb(entries):
    """entries: list of dict(pro_id, pep_start, pep_len, pro_nc?, enz?, miss?, variants)
    variants: list of (mass, [(pos, mod_id), ...])。
    返回模拟 pepdata.pdb 的 bytes（小端，x64 位宽）。
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
```

- [ ] **Step 2: 写失败测试**

Create `tests/test_speclib_pepdata.py`:

```python
"""测试 speclib.pepdata 二进制解析。"""
import pytest
from spectrum.speclib.config_io import Protein, ModEntry
from spectrum.speclib.pepdata import read_pepdata, LibPeptide, ModSite


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
    assert len(peps) == 3
    assert peps[1].sequence == "PEPTIDEK"
    assert peps[1].mods[0].pos == 8
    assert peps[2].sequence == "SAMPLER"


def test_mod_pep_bytes_mismatch_raises(tmp_path, proteins, mods_by_id):
    # 手工构造一个 mod_pep_bytes 错误的条目
    import struct
    header = struct.pack("<IIbbbbIQ", 0, 0, 8, 0, 0, 0, 1, 999)  # 故意 999
    body = struct.pack("<db", 900.0, 0)
    p = tmp_path / "pepdata.pdb"
    p.write_bytes(header + body)
    with pytest.raises(ValueError, match="mod_pep_bytes"):
        read_pepdata(str(p), proteins, mods_by_id)


def test_zero_variant_entry_consumed_and_skipped(tmp_path, proteins, mods_by_id):
    # B3: mod_pep_num==0 的头条目合法，消耗 24B 头、push 0 肽段
    import struct
    e0 = struct.pack("<IIbbbbIQ", 0, 0, 8, 0, 0, 0, 0, 0)            # 0 变体
    e1 = (struct.pack("<IIbbbbIQ", 0, 0, 7, 0, 0, 0, 1, 9)
          + struct.pack("<db", 800.3, 0))                           # 正常条目
    p = tmp_path / "pepdata.pdb"
    p.write_bytes(e0 + e1)
    peps = read_pepdata(str(p), proteins, mods_by_id)
    assert len(peps) == 1
    assert peps[0].sequence == "PEPTIDE"
```

- [ ] **Step 3: 运行测试，确认失败**

Run: `python -m pytest tests/test_speclib_pepdata.py -q`
Expected: FAIL（`ModuleNotFoundError: No module named 'spectrum.speclib.pepdata'`）

- [ ] **Step 4: 实现 pepdata.py**

Create `spectrum/speclib/pepdata.py`:

```python
"""读取 pepdata.pdb（二进制肽段库），复刻 CReader::ReadPepData。

需要：proteins（parse_fasta 结果，按 pro_id 索引）、mods_by_id（mod_id->ModEntry）。
"""
import struct
from dataclasses import dataclass, field

from .config_io import Protein, ModEntry

_HEADER = struct.Struct("<IIbbbbIQ")  # pro_id,pep_start,pep_len,pro_nc,enz,miss,mod_pep_num,mod_pep_bytes
_VAR = struct.Struct("<db")           # mass(double), mod_cnt(char)
_MOD = struct.Struct("<bi")           # pos(char), mod_id(int)


@dataclass
class ModSite:
    pos: int
    mod_id: int
    name: str = ""
    mono_mass: float = 0.0


@dataclass
class LibPeptide:
    sequence: str
    mods: list  # list[ModSite]
    neutral_mass: float
    protein: str
    is_decoy: bool
    pro_nc: int = 0
    enz: int = 0
    miss: int = 0
    charge_mask: int = 0
    pred_rt: float | None = None
    pred_ms2: dict = field(default_factory=dict)  # charge -> list[FragIon]


def read_pepdata(path: str, proteins: list[Protein],
                 mods_by_id: dict[int, ModEntry],
                 validate_bytes: bool = True) -> list[LibPeptide]:
    with open(path, "rb") as fh:
        data = fh.read()
    peptides: list[LibPeptide] = []
    off = 0
    n = len(data)
    while off < n:
        (pro_id, pep_start, pep_len, pro_nc, enz, miss,
         mod_pep_num, mod_pep_bytes) = _HEADER.unpack_from(data, off)
        off += _HEADER.size
        protein = proteins[pro_id]
        seq = protein.sequence[pep_start:pep_start + pep_len]
        consumed = 0
        for _ in range(mod_pep_num):
            mass, mod_cnt = _VAR.unpack_from(data, off)
            off += _VAR.size
            consumed += _VAR.size
            sites: list[ModSite] = []
            for _ in range(mod_cnt):
                mpos, mid = _MOD.unpack_from(data, off)
                off += _MOD.size
                consumed += _MOD.size
                entry = mods_by_id.get(mid)
                sites.append(ModSite(
                    pos=mpos, mod_id=mid,
                    name=entry.name if entry else "",
                    mono_mass=entry.mono_mass if entry else 0.0))
            peptides.append(LibPeptide(
                sequence=seq, mods=sites, neutral_mass=mass,
                protein=protein.ac, is_decoy=protein.is_decoy,
                pro_nc=pro_nc, enz=enz, miss=miss))
        if validate_bytes and consumed != mod_pep_bytes:
            raise ValueError(
                f"mod_pep_bytes mismatch at pro_id={pro_id}: "
                f"consumed {consumed} != declared {mod_pep_bytes}")
    return peptides
```

- [ ] **Step 5: 运行测试，确认通过**

Run: `python -m pytest tests/test_speclib_pepdata.py -q`
Expected: PASS（5 passed）

- [ ] **Step 6: 提交**

```bash
git add spectrum/speclib/pepdata.py tests/conftest.py tests/test_speclib_pepdata.py
git commit -m "feat(speclib): pepdata.pdb binary reader with mod_pep_bytes self-check

Co-authored-by: Copilot <223556219+Copilot@users.noreply.github.com>"
```

---

### Task 3: predictions — 解析 RT 与 MS2 预测，按 chg_max 分组

**Files:**
- Create: `spectrum/speclib/predictions.py`
- Modify: `tests/conftest.py`（加 `build_rt`/`build_ms2` fixtures）
- Test: `tests/test_speclib_predictions.py`

- [ ] **Step 1: 在 conftest.py 增加 RT/MS2 构造 helper**

Append to `tests/conftest.py`:

```python
def _build_rt(values):
    return _struct.pack(f"<{len(values)}f", *values)


_MS2_HEAD = _struct.Struct("<h")
_MS2_ION = _struct.Struct("<bbf")


def _build_ms2(records):
    """records: list of list of (pos, iontype, inten)。返回 pepdata.ms2.predb bytes。"""
    out = b""
    for ions in records:
        out += _MS2_HEAD.pack(len(ions))
        for pos, iontype, inten in ions:
            out += _MS2_ION.pack(pos, iontype, inten)
    return out


@pytest.fixture
def build_rt():
    return _build_rt


@pytest.fixture
def build_ms2():
    return _build_ms2
```

- [ ] **Step 2: 写失败测试**

Create `tests/test_speclib_predictions.py`:

```python
"""测试 speclib.predictions RT/MS2 二进制解析与分组。"""
import pytest
from spectrum.speclib.predictions import (
    FragIon, read_rt_pred, read_ms2_records, group_ms2_by_peptide,
)


def test_read_rt_pred(tmp_path, build_rt):
    p = tmp_path / "pepdata.rt.predb"
    p.write_bytes(build_rt([12.5, 33.0, 7.25]))
    rts = read_rt_pred(str(p))
    assert rts == pytest.approx([12.5, 33.0, 7.25])


def test_read_ms2_records_ion_decode(tmp_path, build_ms2):
    # iontype: 0=b1+, 1=y1+, 2=b2+, 3=y2+
    recs = [
        [(0, 0, 0.9), (1, 1, 0.5)],   # b(pos0,ch1), y(pos1,ch1)
        [(2, 2, 0.8), (3, 3, 0.4)],   # b(pos2,ch2), y(pos3,ch2)
    ]
    p = tmp_path / "pepdata.ms2.predb"
    p.write_bytes(build_ms2(recs))
    out = read_ms2_records(str(p))
    assert len(out) == 2
    assert out[0][0] == FragIon("b", 0, 1, pytest.approx(0.9))
    assert out[0][1] == FragIon("y", 1, 1, pytest.approx(0.5))
    assert out[1][0] == FragIon("b", 2, 2, pytest.approx(0.8))
    assert out[1][1] == FragIon("y", 3, 2, pytest.approx(0.4))


def test_group_ms2_infers_chg_max(tmp_path, build_ms2):
    # 2 肽段 × chg_max=3 = 6 条记录
    recs = [[(0, 0, 1.0)]] * 6
    p = tmp_path / "pepdata.ms2.predb"
    p.write_bytes(build_ms2(recs))
    records = read_ms2_records(str(p))
    grouped, chg_max = group_ms2_by_peptide(records, num_peptides=2)
    assert chg_max == 3
    assert len(grouped) == 2
    assert set(grouped[0].keys()) == {1, 2, 3}


def test_group_ms2_bad_count_raises(tmp_path, build_ms2):
    recs = [[(0, 0, 1.0)]] * 5  # 5 不能被 2 整除
    p = tmp_path / "pepdata.ms2.predb"
    p.write_bytes(build_ms2(recs))
    records = read_ms2_records(str(p))
    with pytest.raises(ValueError, match="not divisible"):
        group_ms2_by_peptide(records, num_peptides=2)


def test_read_ms2_empty_record_in_middle(tmp_path, build_ms2):
    # B2: n_size==0 的记录照样写出，必须当作"存在的空记录"，否则后续错位
    recs = [[(0, 0, 1.0)], [], [(1, 1, 0.5)], []]
    p = tmp_path / "pepdata.ms2.predb"
    p.write_bytes(build_ms2(recs))
    out = read_ms2_records(str(p))
    assert len(out) == 4
    assert out[1] == []
    assert out[3] == []
    grouped, chg_max = group_ms2_by_peptide(out, num_peptides=2)
    assert chg_max == 2
    assert grouped[0][2] == []  # 肽段0 的 charge2 桶为空


def test_group_ms2_chg_max_out_of_range_raises(tmp_path, build_ms2):
    # B5: 推断出的 chg_max 必须 ∈ [1,6]
    recs = [[(0, 0, 1.0)]] * 7  # 1 肽段 × 7 > 6
    p = tmp_path / "pepdata.ms2.predb"
    p.write_bytes(build_ms2(recs))
    records = read_ms2_records(str(p))
    with pytest.raises(ValueError, match="chg_max"):
        group_ms2_by_peptide(records, num_peptides=1)
```

- [ ] **Step 3: 运行测试，确认失败**

Run: `python -m pytest tests/test_speclib_predictions.py -q`
Expected: FAIL（`ModuleNotFoundError: No module named 'spectrum.speclib.predictions'`）

- [ ] **Step 4: 实现 predictions.py**

Create `spectrum/speclib/predictions.py`:

```python
"""读取 pepdata.rt.predb（M 个 float）与 pepdata.ms2.predb（逐肽段-电荷记录）。

复刻 pPredRT.cpp / pPredMS2.cpp 的二进制写出格式。
"""
import struct
from dataclasses import dataclass

_MS2_HEAD = struct.Struct("<h")   # n_size(short)
_MS2_ION = struct.Struct("<bbf")  # pos(char), iontype(char), inten(float)


@dataclass
class FragIon:
    ion_type: str       # 'b' | 'y'
    frag_pos: int       # 0-indexed 切割位
    frag_charge: int    # 1..6
    intensity: float


def read_rt_pred(path: str) -> list[float]:
    with open(path, "rb") as fh:
        data = fh.read()
    count = len(data) // 4
    return list(struct.unpack(f"<{count}f", data[:count * 4]))


def read_ms2_records(path: str) -> list[list[FragIon]]:
    with open(path, "rb") as fh:
        data = fh.read()
    records: list[list[FragIon]] = []
    off = 0
    n = len(data)
    while off < n:
        (n_size,) = _MS2_HEAD.unpack_from(data, off)
        off += _MS2_HEAD.size
        ions: list[FragIon] = []
        for _ in range(n_size):
            pos, iontype, inten = _MS2_ION.unpack_from(data, off)
            off += _MS2_ION.size
            ions.append(FragIon(
                ion_type="b" if iontype % 2 == 0 else "y",
                frag_pos=pos,
                frag_charge=iontype // 2 + 1,
                intensity=inten))
        records.append(ions)
    return records


def group_ms2_by_peptide(records: list[list[FragIon]], num_peptides: int,
                         chg_max: int | None = None):
    """把扁平 records 按 (肽段外层, 电荷 1..chg_max 内层) 分组。

    返回 (grouped, chg_max)，grouped[i] = {charge: [FragIon]}。
    chg_max 为 None 时由 len(records)/num_peptides 推断。
    """
    total = len(records)
    if chg_max is None:
        if num_peptides <= 0 or total % num_peptides != 0:
            raise ValueError(
                f"ms2 record count {total} not divisible by "
                f"num_peptides {num_peptides}")
        chg_max = total // num_peptides
    if not (1 <= chg_max <= 6):
        raise ValueError(f"inferred chg_max {chg_max} out of range [1,6]")
    grouped: list[dict[int, list[FragIon]]] = []
    idx = 0
    for _ in range(num_peptides):
        d: dict[int, list[FragIon]] = {}
        for chg in range(1, chg_max + 1):
            d[chg] = records[idx]
            idx += 1
        grouped.append(d)
    return grouped, chg_max
```

- [ ] **Step 5: 运行测试，确认通过**

Run: `python -m pytest tests/test_speclib_predictions.py -q`
Expected: PASS（6 passed）

- [ ] **Step 6: 提交**

```bash
git add spectrum/speclib/predictions.py tests/conftest.py tests/test_speclib_predictions.py
git commit -m "feat(speclib): RT/MS2 prediction binary readers + per-peptide grouping

Co-authored-by: Copilot <223556219+Copilot@users.noreply.github.com>"
```

---

### Task 4: speclib — 顶层 loader + 质量交叉校验

**Files:**
- Create: `spectrum/speclib/speclib.py`
- Modify: `spectrum/speclib/__init__.py`（导出 `SpecLib`）
- Test: `tests/test_speclib_loader.py`

- [ ] **Step 1: 写失败测试**

Create `tests/test_speclib_loader.py`:

```python
"""测试 SpecLib 顶层 loader 与质量交叉校验。"""
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

    # 用同一套质量公式算出正确中性质量，写进 pdb
    from spectrum.speclib.config_io import (
        parse_element_masses, parse_residue_masses, water_mass)
    em = parse_element_masses(str(elem))
    res = parse_residue_masses(str(aa), em)
    w = water_mass(em)
    seq = "PEPTIDEKACDM"  # pep_start 0, len 12
    # 变体1：无修饰；变体2：Carbamidomethyl[C] 在第 9 位 (C)
    m1 = w + sum(res[a] for a in seq)
    m2 = m1 + 57.021464
    pdb = build_pdb([
        {"pro_id": 0, "pep_start": 0, "pep_len": 12,
         "variants": [(m1, []), (m2, [(9, 1)])]},
    ])
    (tmp_path / "pepdata.pdb").write_bytes(pdb)
    # 2 肽段变体 × chg_max=2 = 4 条 MS2 记录
    (tmp_path / "pepdata.ms2.predb").write_bytes(build_ms2([
        [(0, 0, 1.0)], [(1, 1, 0.5)], [(0, 0, 0.8)], [(2, 3, 0.3)]]))
    (tmp_path / "pepdata.rt.predb").write_bytes(build_rt([20.0, 21.5]))
    return tmp_path


def test_load_dir_end_to_end(lib_files):
    lib = SpecLib.load_dir(str(lib_files),
                           fasta_path=str(lib_files / "db.fasta"),
                           mod_path=str(lib_files / "modification.ini"))
    assert len(lib.peptides) == 2
    assert lib.chg_max == 2
    assert lib.peptides[0].pred_rt == pytest.approx(20.0)
    assert lib.peptides[1].pred_rt == pytest.approx(21.5)
    # B1: 1 个 pdb 头条目(2 变体) → 2 个 LibPeptide，逐变体与 RT/MS2 对齐
    assert lib.peptides[0].mods == []          # 变体1(无修饰) 对齐 rt=20.0
    assert set(lib.peptides[0].pred_ms2.keys()) == {1, 2}
    assert lib.peptides[0].pred_ms2[1][0].ion_type == "b"
    assert lib.peptides[1].mods[0].name == "Carbamidomethyl[C]"  # 变体2 对齐 rt=21.5


def test_validate_masses_all_pass(lib_files):
    lib = SpecLib.load_dir(str(lib_files),
                           fasta_path=str(lib_files / "db.fasta"),
                           mod_path=str(lib_files / "modification.ini"))
    report = lib.validate_masses(str(lib_files / "element.ini"),
                                 str(lib_files / "aa.ini"), tol=1e-4)
    assert report.total == 2
    assert report.passed == 2
    assert report.failed == 0
    assert report.max_abs_error < 1e-4


def test_validate_masses_flags_wrong_mass(lib_files):
    lib = SpecLib.load_dir(str(lib_files),
                           fasta_path=str(lib_files / "db.fasta"),
                           mod_path=str(lib_files / "modification.ini"))
    lib.peptides[0].neutral_mass += 5.0  # 人为破坏
    report = lib.validate_masses(str(lib_files / "element.ini"),
                                 str(lib_files / "aa.ini"), tol=1e-4)
    assert report.failed == 1
    assert report.failures[0][0] == 0  # index 0


def test_rt_count_mismatch_raises(lib_files, build_rt):
    (lib_files / "pepdata.rt.predb").write_bytes(build_rt([1.0]))  # 只有 1 个
    with pytest.raises(ValueError, match="RT count"):
        SpecLib.load_dir(str(lib_files),
                         fasta_path=str(lib_files / "db.fasta"),
                         mod_path=str(lib_files / "modification.ini"))
```

- [ ] **Step 2: 运行测试，确认失败**

Run: `python -m pytest tests/test_speclib_loader.py -q`
Expected: FAIL（`ImportError: cannot import name 'SpecLib'`）

- [ ] **Step 3: 实现 speclib.py**

Create `spectrum/speclib/speclib.py`:

```python
"""SpecLib 顶层 loader：组装 pepdata + RT + MS2 为 M 个 LibPeptide，并提供质量自校验。"""
import os
from dataclasses import dataclass, field

from .config_io import (parse_fasta, parse_modifications,
                        parse_element_masses, parse_residue_masses, water_mass)
from .pepdata import read_pepdata, LibPeptide
from .predictions import read_rt_pred, read_ms2_records, group_ms2_by_peptide


@dataclass
class MassValidationReport:
    total: int
    passed: int
    failed: int
    max_abs_error: float
    failures: list = field(default_factory=list)  # (index, seq, computed, stored, err)


class SpecLib:
    def __init__(self, peptides: list[LibPeptide], chg_max: int):
        self.peptides = peptides
        self.chg_max = chg_max

    @classmethod
    def load(cls, *, pepdata_path: str, rt_path: str, ms2_path: str,
             fasta_path: str, mod_path: str) -> "SpecLib":
        proteins = parse_fasta(fasta_path)
        mods_by_id = {m.mod_id: m for m in parse_modifications(mod_path)}
        peptides = read_pepdata(pepdata_path, proteins, mods_by_id)

        rts = read_rt_pred(rt_path)
        if len(rts) != len(peptides):
            raise ValueError(
                f"RT count {len(rts)} != peptide count {len(peptides)}")
        for pep, rt in zip(peptides, rts):
            pep.pred_rt = rt

        records = read_ms2_records(ms2_path)
        grouped, chg_max = group_ms2_by_peptide(records, len(peptides))
        for pep, g in zip(peptides, grouped):
            pep.pred_ms2 = g

        return cls(peptides, chg_max)

    @classmethod
    def load_dir(cls, library_dir: str, *, fasta_path: str,
                 mod_path: str) -> "SpecLib":
        return cls.load(
            pepdata_path=os.path.join(library_dir, "pepdata.pdb"),
            rt_path=os.path.join(library_dir, "pepdata.rt.predb"),
            ms2_path=os.path.join(library_dir, "pepdata.ms2.predb"),
            fasta_path=fasta_path, mod_path=mod_path)

    def validate_masses(self, element_path: str, aa_path: str,
                        tol: float = 0.01) -> MassValidationReport:
        em = parse_element_masses(element_path)
        res = parse_residue_masses(aa_path, em)
        water = water_mass(em)
        failures = []
        max_err = 0.0
        passed = 0
        for i, pep in enumerate(self.peptides):
            computed = (water
                        + sum(res.get(a, 0.0) for a in pep.sequence)
                        + sum(m.mono_mass for m in pep.mods))
            err = abs(computed - pep.neutral_mass)
            max_err = max(max_err, err)
            if err <= tol:
                passed += 1
            else:
                failures.append((i, pep.sequence, computed,
                                 pep.neutral_mass, err))
        return MassValidationReport(
            total=len(self.peptides), passed=passed,
            failed=len(self.peptides) - passed,
            max_abs_error=max_err, failures=failures)
```

- [ ] **Step 4: 导出 SpecLib**

Replace `spectrum/speclib/__init__.py` content with:

```python
"""pFind 谱库（spectral library）二进制读取模块。"""
from .speclib import SpecLib, MassValidationReport
from .pepdata import LibPeptide, ModSite
from .predictions import FragIon

__all__ = ["SpecLib", "MassValidationReport", "LibPeptide", "ModSite", "FragIon"]
```

- [ ] **Step 5: 运行测试，确认通过**

Run: `python -m pytest tests/test_speclib_loader.py -q`
Expected: PASS（4 passed）

- [ ] **Step 6: 跑全部 speclib 测试 + 确认未破坏既有测试**

Run: `python -m pytest tests/test_speclib_config_io.py tests/test_speclib_pepdata.py tests/test_speclib_predictions.py tests/test_speclib_loader.py -q`
Expected: PASS（全部通过，19 passed）

- [ ] **Step 7: 提交**

```bash
git add spectrum/speclib/speclib.py spectrum/speclib/__init__.py tests/test_speclib_loader.py
git commit -m "feat(speclib): SpecLib loader + neutral-mass cross-validation

Co-authored-by: Copilot <223556219+Copilot@users.noreply.github.com>"
```

---

### Task 5: CLI — 真实谱库验证工具

> 用户拿到真实谱库文件后，用此 CLI 加载并做质量交叉校验，打印摘要、样例肽段、质量误差分布。这是 spec 中"真实文件验证"步骤的落地。

**Files:**
- Create: `tools/speclib_inspect.py`
- Test: `tests/test_speclib_inspect_cli.py`

- [ ] **Step 1: 写失败测试**

Create `tests/test_speclib_inspect_cli.py`:

```python
"""测试 speclib_inspect CLI 的核心 summarize 逻辑（不依赖真实大文件）。"""
import pytest
from tools.speclib_inspect import summarize


def test_summarize_runs_on_fixture(lib_files):
    # 复用 test_speclib_loader 的 lib_files fixture（在 conftest 之外，需共享）
    text = summarize(
        library_dir=str(lib_files),
        fasta_path=str(lib_files / "db.fasta"),
        mod_path=str(lib_files / "modification.ini"),
        element_path=str(lib_files / "element.ini"),
        aa_path=str(lib_files / "aa.ini"),
        n_samples=1, tol=1e-4)
    assert "peptides: 2" in text
    assert "chg_max: 2" in text
    assert "mass pass: 2/2" in text
```

- [ ] **Step 2: 把 `lib_files` fixture 提到 conftest 以便 CLI 测试复用**

Move the `lib_files` fixture from `tests/test_speclib_loader.py` into `tests/conftest.py` (cut its full definition from the test file and paste at end of conftest, keeping the `import pytest` already present). After moving, `tests/test_speclib_loader.py` keeps using `lib_files` as a fixture argument (no local definition needed).

Append to `tests/conftest.py` (the fixture body, identical to Task 4 Step 1's `lib_files`):

```python
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
    seq = "PEPTIDEKACDM"
    m1 = w + sum(res[a] for a in seq)
    m2 = m1 + 57.021464
    pdb = build_pdb([
        {"pro_id": 0, "pep_start": 0, "pep_len": 12,
         "variants": [(m1, []), (m2, [(9, 1)])]},
    ])
    (tmp_path / "pepdata.pdb").write_bytes(pdb)
    (tmp_path / "pepdata.ms2.predb").write_bytes(build_ms2([
        [(0, 0, 1.0)], [(1, 1, 0.5)], [(0, 0, 0.8)], [(2, 3, 0.3)]]))
    (tmp_path / "pepdata.rt.predb").write_bytes(build_rt([20.0, 21.5]))
    return tmp_path
```

Then delete the duplicate `lib_files` fixture definition from `tests/test_speclib_loader.py` (leave the four test functions intact).

- [ ] **Step 3: 运行测试，确认失败**

Run: `python -m pytest tests/test_speclib_inspect_cli.py -q`
Expected: FAIL（`ModuleNotFoundError: No module named 'tools.speclib_inspect'`）

- [ ] **Step 4: 实现 CLI**

Create `tools/speclib_inspect.py`:

```python
"""加载 pFind 谱库并打印摘要 + 质量交叉校验，用于真实文件验证。

用法:
  python -m tools.speclib_inspect --library-dir DIR \\
      --fasta merge_human_ecoli_yeast.fasta --mod modification.ini \\
      [--element element.ini --aa aa.ini] [--n-samples 5] [--tol 0.01]
"""
import argparse

from spectrum.speclib import SpecLib


def summarize(*, library_dir: str, fasta_path: str, mod_path: str,
              element_path: str | None = None, aa_path: str | None = None,
              n_samples: int = 5, tol: float = 0.01) -> str:
    lib = SpecLib.load_dir(library_dir, fasta_path=fasta_path,
                           mod_path=mod_path)
    lines = []
    lines.append(f"peptides: {len(lib.peptides)}")
    lines.append(f"chg_max: {lib.chg_max}")
    rts = [p.pred_rt for p in lib.peptides if p.pred_rt is not None]
    if rts:
        lines.append(f"rt range (min): {min(rts):.3f} .. {max(rts):.3f}")

    for pep in lib.peptides[:n_samples]:
        modstr = ",".join(f"{m.pos}:{m.name}" for m in pep.mods) or "-"
        top = sorted(
            (ion for ions in pep.pred_ms2.values() for ion in ions),
            key=lambda x: x.intensity, reverse=True)[:3]
        topstr = " ".join(
            f"{i.ion_type}{i.frag_pos}^{i.frag_charge}={i.intensity:.2f}"
            for i in top)
        lines.append(
            f"  {pep.sequence} mods=[{modstr}] mass={pep.neutral_mass:.4f} "
            f"rt={pep.pred_rt} top_ms2=[{topstr}]")

    if element_path and aa_path:
        rep = lib.validate_masses(element_path, aa_path, tol=tol)
        lines.append(f"mass pass: {rep.passed}/{rep.total} "
                     f"(max_abs_err={rep.max_abs_error:.5f}, tol={tol})")
        for idx, seq, computed, stored, err in rep.failures[:5]:
            lines.append(f"  FAIL #{idx} {seq} computed={computed:.4f} "
                         f"stored={stored:.4f} err={err:.4f}")
    else:
        lines.append("mass validation skipped (no --element/--aa)")
    return "\n".join(lines)


def main():
    ap = argparse.ArgumentParser(description="Inspect a pFind spectral library")
    ap.add_argument("--library-dir", required=True)
    ap.add_argument("--fasta", required=True)
    ap.add_argument("--mod", required=True)
    ap.add_argument("--element", default=None)
    ap.add_argument("--aa", default=None)
    ap.add_argument("--n-samples", type=int, default=5)
    ap.add_argument("--tol", type=float, default=0.01)
    args = ap.parse_args()
    print(summarize(
        library_dir=args.library_dir, fasta_path=args.fasta,
        mod_path=args.mod, element_path=args.element, aa_path=args.aa,
        n_samples=args.n_samples, tol=args.tol))


if __name__ == "__main__":
    main()
```

- [ ] **Step 5: 运行测试，确认通过**

Run: `python -m pytest tests/test_speclib_inspect_cli.py tests/test_speclib_loader.py -q`
Expected: PASS（5 passed；确认移动 fixture 后 loader 测试仍通过）

- [ ] **Step 6: 跑全部 speclib 相关测试**

Run: `python -m pytest tests/ -k speclib -q`
Expected: PASS（20 passed）

- [ ] **Step 7: 提交**

```bash
git add tools/speclib_inspect.py tests/test_speclib_inspect_cli.py tests/conftest.py tests/test_speclib_loader.py
git commit -m "feat(speclib): speclib_inspect CLI for real-library validation

Co-authored-by: Copilot <223556219+Copilot@users.noreply.github.com>"
```

---

## 真实文件验证（用户拿到谱库后执行）

谱库到手后，运行（路径按实际情况替换）：

```bash
python -m tools.speclib_inspect \
  --library-dir <谱库目录> \
  --fasta ../puku/merge_human_ecoli_yeast.fasta \
  --mod ../puku/modification.ini \
  --element ../puku/element.ini --aa ../puku/aa.ini \
  --n-samples 10 --tol 0.01
```

**判读：**
- `mass pass` 接近 100% ⇒ 序列+修饰解码正确。
- 若 `max_abs_err` 是一个**恒定常数**（≈1.0073=质子、≈18.0106=水、≈ 某修饰质量），说明 `lfPepMass` 的定义与"中性=Σ残基+H₂O+Σ修饰"假设有系统偏差 → 在 `SpecLib.validate_masses` 的 `computed` 公式按该常数校正，并更新 spec。
- 若"一个文件"实为单文件/压缩包而非目录 ⇒ 视形态加一个适配 loader（如 `SpecLib.load(pepdata_path=..., rt_path=..., ms2_path=...)` 已支持显式三路径；压缩包先解压或加 zip 适配）。

---

## Self-Review

- **Spec coverage**：模块结构（config_io/pepdata/predictions/speclib）✓ Task1–4；二进制格式三表 ✓ Task2–3；mod_id 映射 ✓ Task1；FASTA 解析 ✓ Task1；数据模型（LibPeptide/FragIon/ModEntry/ModSite）✓；chg_max 推断 ✓ Task3；四层自校验中"质量交叉校验"✓ Task4、"mod_pep_bytes"✓ Task2、"计数一致性"✓ Task3/4；测试策略（合成 fixture + 真实文件）✓ Task1–5 + 验证小节；非目标（不接入 pipeline）已遵守。
- **边界条件覆盖（rubber-duck 复核）**：B1 M=Σmod_pep_num 逐变体对齐 ✓ Task4 `test_load_dir_end_to_end`；B2 `n_size==0` 空记录 ✓ Task3 `test_read_ms2_empty_record_in_middle`；B3 `mod_pep_num==0` ✓ Task2 `test_zero_variant_entry_consumed_and_skipped`；B5 `chg_max∈[1,6]` 硬校验 ✓ Task3 `test_group_ms2_chg_max_out_of_range_raises`；B4 各电荷桶离子不对称——reader 不做对称假设，validate 亦不假设。
- **Placeholder scan**：无 TBD/TODO；每个代码步骤含完整代码与精确命令/期望输出。
- **Type/name consistency**：`Protein/ModEntry/ModSite/LibPeptide/FragIon/MassValidationReport/SpecLib` 跨任务一致；`parse_fasta/parse_modifications/parse_element_masses/parse_residue_masses/water_mass/read_pepdata/read_rt_pred/read_ms2_records/group_ms2_by_peptide/SpecLib.load/load_dir/validate_masses` 签名一致；struct 格式串 `'<IIbbbbIQ'/'<db'/'<bi'/'<h'/'<bbf'` 全程统一。
