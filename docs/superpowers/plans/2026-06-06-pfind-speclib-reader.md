# pFind 谱库读取模块 Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** 在 `spectrum/speclib/` 实现一个独立、自校验、**内存安全（流式）**的 pFind 谱库二进制读取模块，逐肽段产出预测 RT + MS2。

**Architecture:** 四个单向依赖的小模块：`config_io`（解析 FASTA/modification.ini/element.ini/aa.ini）→ `pepdata`（流式读 `pepdata.pdb`）+ `predictions`（读 `pepdata.rt.predb` 全量数组、流式读 `pepdata.ms2.predb` 并跳过文本尾巴）→ `speclib`（**锁步流式** loader：pdb+RT+MS2 同序逐肽段 yield + 质量自校验）。外加 `tools/speclib_inspect.py` CLI 做真实文件验证（4.4GB 不 OOM）。

**Tech Stack:** Python 3.10+（仓库实际 3.14），仅用标准库 `struct`/`array`/`dataclasses`/`os`；pytest（`python -m pytest`，repo root）。所有多字节字段小端、x64 位宽。

参考 spec：`docs/specs/2026-06-06-pfind-speclib-reader-design.md`（已对真实 `lib-2th` 验证：mass 100%、RT 数=M、ms2=4×M 二进制+M 行文本尾巴、chg_max=4）。

**关键格式（已对真实文件验证）：**
- `pepdata.pdb` 条目头 `struct '<IIbbbbIQ'`（24B）= pro_id u32, pep_start u32, pep_len i8, pro_nc i8, enz i8, miss i8, mod_pep_num u32, mod_pep_bytes u64。随后 mod_pep_num 个变体：`'<db'`（9B）= mass f64 + mod_cnt i8；再 mod_cnt 个 `'<bi'`（5B）= pos i8 + mod_id i32。
- `pepdata.rt.predb` = M×f32（分钟）。
- `pepdata.ms2.predb` = `[M×chg_max 二进制记录][M 行文本尾巴]`。每记录 `'<h'`（2B）= n_size，再 n_size 个 `'<bbf'`（6B）= pos,i8; iontype,i8; inten,f32。**文本尾巴**每行 `"1\t0\t…\tchg_max\t0\t\n"`；读取遇 `n_size<0 或 >MAX_ION_OUTPUT(1000)` 即停（尾巴首 2 字节 `'1\t'` 作 i16=12553>1000）。
- 序列 = `proteins[pro_id].sequence[pep_start:pep_start+pep_len]`。
- mod_id = modification.ini 过滤后数据行 1-based read-order（Carbamidomethyl[C]=9, Oxidation[M]=46 已验证）。
- iontype：偶=b、奇=y；frag_charge=iontype//2+1。
- 中性质量 = Σ残基 + H₂O(18.0105646837) + Σ修饰单同位素质量。**已由 C++ 验证**（sdk.h:881 不含质子；Instrument.cpp:45/61 含水）。真实库实测 100% 通过（max_err=0）。

**体量 / 边界条件（rubber-duck + 真实文件复核）：**
- ms2 ~4.4GB / ~1250 万记录、pdb ~312 万肽段 → **不全量物化**；核心用**锁步流式生成器**逐肽段 yield，内存 O(1)。RT 小（~12MB）全量进 `array('f')`。随机按肽段查 MS2 的缓存偏移索引**本步不做**。
- **M = Σ mod_pep_num**（变体总数），逐变体与 RT/MS2 对齐。
- MS2 `n_size==0` 记录照样存在（空记录），不可跳过。
- chg_max 从尾巴行解析（实测=4），并约束 ∈ [1,6]。
- charge-c 桶只含 `frag_charge ≤ c` 的离子（各桶不对称），勿假设对称。

---

## 分层文档（L1–L4，边写代码边填，随代码提交）

**目录结构（按组件分目录）：**

```
docs/speclib/
  L1_overview.md                      # 整个模块：目标/架构/数据流/快速上手/关键事实
  parts/
    config_io/{L2_role,L3_details,L4_api}.md
    pepdata/{L2_role,L3_details,L4_api}.md
    predictions/{L2_role,L3_details,L4_api}.md
    speclib/{L2_role,L3_details,L4_api}.md
    speclib_inspect/{L2_role,L3_details,L4_api}.md
```

**各层模板（中文撰写）：**

- **L1_overview.md**：`# speclib — pFind 谱库读取模块`；小节：目标 / 架构图（config_io → pepdata + predictions → speclib → CLI）/ 数据流（谱库目录→解析→锁步流式逐肽段）/ 快速上手（`SpecLib.open_dir` + `iter_peptides` 代码示例、`speclib_inspect` 命令）/ 组件索引（链接 parts/\*/L2_role.md）/ 关键事实（M、chg_max=4、MS2 文本尾巴、质量校验 100%）。
- **L2_role.md**：`# <组件> — 职责与接口`；小节：一句话职责 / 对外接口（函数·类签名 + 一行简述）/ 依赖（依赖谁·被谁依赖）/ 输入·输出。
- **L3_details.md**：`# <组件> — 细节`；小节：解析·算法细节 / 二进制·文本格式（如适用）/ 边界与坑（按组件：尾巴 / 空记录 / mod_pep_bytes / 锁步对齐 / 质量公式）/ 设计取舍 / 复刻自哪段 C++（带文件:行号）。
- **L4_api.md**：`# <组件> — API 参考（<源文件路径>）`；逐 public 函数·类：签名、参数、返回、异常、最小示例。

**规则：** 每个 Task 在代码+测试通过后、提交前，写/更新该组件的 `parts/<组件>/{L2_role,L3_details,L4_api}.md`；Task 1 额外创建 `L1_overview.md` 骨架，Task 5 末尾回填 L1 的"组件索引/快速上手/关键事实"。文档与代码**同一次 commit**。文档无需测试。

---

### Task 1: config_io — FASTA / 修饰 / 元素 / 残基质量解析

**Files:**
- Create: `spectrum/speclib/__init__.py`
- Create: `spectrum/speclib/config_io.py`
- Test: `tests/test_speclib_config_io.py`
- Docs: `docs/speclib/L1_overview.md`、`docs/speclib/parts/config_io/{L2_role,L3_details,L4_api}.md`

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
    """复刻 CReader::ReadAA：残基质量 = Σ 元素质量 × 个数（不含水）。"""
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
Expected: PASS（4 passed）

- [ ] **Step 5: 写分层文档**

按"分层文档"模板创建：
- `docs/speclib/L1_overview.md`（骨架：目标/架构/数据流/快速上手/组件索引/关键事实，后续 Task 持续补全；现可先列出 5 个组件与已知关键事实 M=3,124,520、chg_max=4、MS2 文本尾巴、质量校验 100%）
- `docs/speclib/parts/config_io/L2_role.md`：职责=解析 FASTA/modification.ini/element.ini/aa.ini；接口=`parse_fasta`/`parse_modifications`/`parse_element_masses`/`parse_residue_masses`/`water_mass`；依赖=纯文本，被 pepdata/speclib 依赖。
- `docs/speclib/parts/config_io/L3_details.md`：FASTA AC 切分（空格/Tab/`|`>15 规则）、modification.ini read-order id 与跳过规则（name\*/@NUMBER/label_name/Met-loss/Label_）、element 取最高丰度同位素、residue=Σ元素×个数（不含水）、water=2H+O；复刻 Reader.cpp(ReadMod/ReadAA/ReadElementInfo)、fastaparser.cpp。
- `docs/speclib/parts/config_io/L4_api.md`：`spectrum/speclib/config_io.py` 逐函数与 `Protein`/`ModEntry` 的签名/参数/返回/示例。

- [ ] **Step 6: 提交**

```bash
git add spectrum/speclib/__init__.py spectrum/speclib/config_io.py tests/test_speclib_config_io.py docs/speclib/L1_overview.md docs/speclib/parts/config_io/
git commit -m "feat(speclib): config_io — FASTA/modification/element/residue mass parsers

Co-authored-by: Copilot <223556219+Copilot@users.noreply.github.com>"
```

---

### Task 2: pepdata — 流式读 pepdata.pdb

**Files:**
- Create: `spectrum/speclib/pepdata.py`
- Modify: `tests/conftest.py`（加 `build_pdb` fixture）
- Test: `tests/test_speclib_pepdata.py`
- Docs: `docs/speclib/parts/pepdata/{L2_role,L3_details,L4_api}.md`

- [ ] **Step 1: 在 conftest.py 增加 pdb 构造 helper**

Append to `tests/conftest.py`:

```python
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
```

- [ ] **Step 2: 写失败测试**

Create `tests/test_speclib_pepdata.py`:

```python
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
```

- [ ] **Step 3: 运行测试，确认失败**

Run: `python -m pytest tests/test_speclib_pepdata.py -q`
Expected: FAIL（`ModuleNotFoundError: No module named 'spectrum.speclib.pepdata'`）

- [ ] **Step 4: 实现 pepdata.py**

Create `spectrum/speclib/pepdata.py`:

```python
"""流式读取 pepdata.pdb（二进制肽段库），复刻 CReader::ReadPepData。

需要：proteins（parse_fasta 结果，按 pro_id 索引）、mods_by_id（mod_id->ModEntry）。
体量大（~312 万肽段），核心用生成器 iter_pepdata 逐条产出；read_pepdata 为 list 包装。
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


def iter_pepdata(path: str, proteins: list[Protein],
                 mods_by_id: dict[int, ModEntry],
                 validate_bytes: bool = True):
    """逐肽段变体 yield LibPeptide（内存 O(1)，不含预测值）。"""
    with open(path, "rb") as fh:
        data = fh.read()
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
            yield LibPeptide(
                sequence=seq, mods=sites, neutral_mass=mass,
                protein=protein.ac, is_decoy=protein.is_decoy,
                pro_nc=pro_nc, enz=enz, miss=miss)
        if validate_bytes and consumed != mod_pep_bytes:
            raise ValueError(
                f"mod_pep_bytes mismatch at pro_id={pro_id}: "
                f"consumed {consumed} != declared {mod_pep_bytes}")


def read_pepdata(path: str, proteins: list[Protein],
                 mods_by_id: dict[int, ModEntry],
                 validate_bytes: bool = True) -> list[LibPeptide]:
    """list 包装（小数据 / 测试用）。"""
    return list(iter_pepdata(path, proteins, mods_by_id, validate_bytes))
```

- [ ] **Step 5: 运行测试，确认通过**

Run: `python -m pytest tests/test_speclib_pepdata.py -q`
Expected: PASS（6 passed）

- [ ] **Step 6: 写分层文档**

- `docs/speclib/parts/pepdata/L2_role.md`：职责=流式解析 `pepdata.pdb` → `LibPeptide`；接口=`iter_pepdata`（生成器）/`read_pepdata`（list 包装）/`LibPeptide`/`ModSite`；依赖=config_io，被 speclib 依赖。
- `docs/speclib/parts/pepdata/L3_details.md`：条目头 `'<IIbbbbIQ'`(24B)、变体 `'<db'`、修饰 `'<bi'`；序列还原 `seq[pep_start:pep_start+pep_len]`；M=Σmod_pep_num（一个头条目多变体）；`mod_pep_bytes` 自校验（确认 size_t=8）；`mod_pep_num==0` 合法；为何用生成器（312 万肽段内存安全）；复刻 Reader.cpp:264 ReadPepData。
- `docs/speclib/parts/pepdata/L4_api.md`：`spectrum/speclib/pepdata.py` 的 `iter_pepdata`/`read_pepdata` 签名/参数/返回/异常（`ValueError: mod_pep_bytes mismatch`）、`LibPeptide`/`ModSite` 字段表。

- [ ] **Step 7: 提交**

```bash
git add spectrum/speclib/pepdata.py tests/conftest.py tests/test_speclib_pepdata.py docs/speclib/parts/pepdata/
git commit -m "feat(speclib): streaming pepdata.pdb reader with mod_pep_bytes self-check

Co-authored-by: Copilot <223556219+Copilot@users.noreply.github.com>"
```

---

### Task 3: predictions — RT 数组 + 流式 MS2（跳过文本尾巴）

**Files:**
- Create: `spectrum/speclib/predictions.py`
- Modify: `tests/conftest.py`（加 `build_rt`/`build_ms2` fixtures）
- Test: `tests/test_speclib_predictions.py`
- Docs: `docs/speclib/parts/predictions/{L2_role,L3_details,L4_api}.md`

- [ ] **Step 1: 在 conftest.py 增加 RT/MS2 构造 helper（可带文本尾巴）**

Append to `tests/conftest.py`:

```python
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
```

- [ ] **Step 2: 写失败测试**

Create `tests/test_speclib_predictions.py`:

```python
"""测试 speclib.predictions：RT 数组 + 流式 MS2（跳过文本尾巴）。"""
import pytest
from spectrum.speclib.predictions import (
    FragIon, read_rt_pred, iter_ms2_records, read_chg_max_from_trailer,
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
```

- [ ] **Step 3: 运行测试，确认失败**

Run: `python -m pytest tests/test_speclib_predictions.py -q`
Expected: FAIL（`ModuleNotFoundError: No module named 'spectrum.speclib.predictions'`）

- [ ] **Step 4: 实现 predictions.py**

Create `spectrum/speclib/predictions.py`:

```python
"""读取 pepdata.rt.predb（M 个 float）与 pepdata.ms2.predb（二进制记录 + 文本尾巴）。

复刻 pPredRT.cpp / pPredMS2.cpp 写出格式。MS2 文件结构为
`[M×chg_max 二进制记录][M 行文本尾巴]`，读取遇尾巴即停。
"""
import os
import struct
from array import array
from dataclasses import dataclass

_MS2_HEAD = struct.Struct("<h")   # n_size(short)
_MS2_ION = struct.Struct("<bbf")  # pos(char), iontype(char), inten(float)
MAX_ION_OUTPUT = 1000


@dataclass
class FragIon:
    ion_type: str       # 'b' | 'y'
    frag_pos: int       # 0-indexed 切割位
    frag_charge: int    # 1..6
    intensity: float


def read_rt_pred(path: str) -> "array":
    """全量读取（~12MB），返回 array('f')。"""
    with open(path, "rb") as fh:
        data = fh.read()
    a = array("f")
    a.frombytes(data[:len(data) // 4 * 4])
    return a


def iter_ms2_records(path: str, max_ions: int = MAX_ION_OUTPUT):
    """逐记录 yield list[FragIon]；遇 n_size<0 或 >max_ions（文本尾巴）即停。"""
    with open(path, "rb") as fh:
        data = fh.read()
    off = 0
    n = len(data)
    while off + 2 <= n:
        (n_size,) = _MS2_HEAD.unpack_from(data, off)
        if n_size < 0 or n_size > max_ions:
            break
        off += 2
        ions: list[FragIon] = []
        for _ in range(n_size):
            pos, iontype, inten = _MS2_ION.unpack_from(data, off)
            off += _MS2_ION.size
            ions.append(FragIon(
                ion_type="b" if iontype % 2 == 0 else "y",
                frag_pos=pos,
                frag_charge=iontype // 2 + 1,
                intensity=inten))
        yield ions


def read_chg_max_from_trailer(path: str, tail_bytes: int = 8192) -> int:
    """从文件末尾文本尾巴解析 chg_max。尾巴行形如 '1\\t0\\t2\\t0\\t...\\tC\\t0\\t'。"""
    size = os.path.getsize(path)
    with open(path, "rb") as fh:
        fh.seek(max(0, size - tail_bytes))
        tail = fh.read().decode("latin-1", errors="replace")
    for line in reversed(tail.split("\n")):
        toks = line.split("\t")
        charges = toks[0::2]            # 偶数下标为电荷
        if charges and charges[-1] == "":
            charges = charges[:-1]
        if charges and all(c.isdigit() for c in charges):
            vals = [int(c) for c in charges]
            if vals == list(range(1, len(vals) + 1)):
                return len(vals)
    raise ValueError(f"cannot parse chg_max from trailer of {path}")
```

- [ ] **Step 5: 运行测试，确认通过**

Run: `python -m pytest tests/test_speclib_predictions.py -q`
Expected: PASS（5 passed）

- [ ] **Step 6: 写分层文档**

- `docs/speclib/parts/predictions/L2_role.md`：职责=读 RT（全量数组）+ 流式读 MS2（跳尾巴）+ 解析 chg_max；接口=`read_rt_pred`/`iter_ms2_records`/`read_chg_max_from_trailer`/`FragIon`；依赖=无（纯二进制），被 speclib 依赖。
- `docs/speclib/parts/predictions/L3_details.md`：rt.predb=M×f32（分钟）；ms2.predb=`[M×chg_max 记录][M 行文本尾巴]`；记录 `'<h'`+`'<bbf'`；iontype 偶 b 奇 y、frag_charge=iontype//2+1；**文本尾巴成因**（pPredMS2.cpp:868-873 收尾 fprintf 在 binary 外）+ 停止规则（n_size<0 或 >1000）；`n_size==0` 空记录保留；尾巴行 `"1\t0\t…\tC\t0\t\n"` 解析 chg_max。
- `docs/speclib/parts/predictions/L4_api.md`：`spectrum/speclib/predictions.py` 的 `read_rt_pred`（返回 `array('f')`）/`iter_ms2_records`（生成器，`max_ions` 参数）/`read_chg_max_from_trailer`（`ValueError` 条件）/`FragIon` 字段。

- [ ] **Step 7: 提交**

```bash
git add spectrum/speclib/predictions.py tests/conftest.py tests/test_speclib_predictions.py docs/speclib/parts/predictions/
git commit -m "feat(speclib): RT array + streaming MS2 reader (skips text trailer)

Co-authored-by: Copilot <223556219+Copilot@users.noreply.github.com>"
```

---

### Task 4: speclib — 锁步流式 loader + 质量自校验

**Files:**
- Create: `spectrum/speclib/speclib.py`
- Modify: `spectrum/speclib/__init__.py`（导出 `SpecLib` 等）
- Test: `tests/test_speclib_loader.py`
- Docs: `docs/speclib/parts/speclib/{L2_role,L3_details,L4_api}.md`

- [ ] **Step 1: 写失败测试**

Create `tests/test_speclib_loader.py`:

```python
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
```

- [ ] **Step 2: 运行测试，确认失败**

Run: `python -m pytest tests/test_speclib_loader.py -q`
Expected: FAIL（`ImportError: cannot import name 'SpecLib'`）

- [ ] **Step 3: 实现 speclib.py**

Create `spectrum/speclib/speclib.py`:

```python
"""SpecLib 锁步流式 loader：pdb+RT+MS2 同序逐肽段产出，并提供质量自校验。

体量大（ms2 ~4.4GB），不全量物化：iter_peptides() 流式 yield 已填好
pred_rt/pred_ms2 的 LibPeptide，由调用方即用即弃。
"""
import os
from dataclasses import dataclass, field

from .config_io import (parse_fasta, parse_modifications,
                        parse_element_masses, parse_residue_masses, water_mass)
from .pepdata import iter_pepdata, LibPeptide
from .predictions import read_rt_pred, iter_ms2_records, read_chg_max_from_trailer


@dataclass
class MassValidationReport:
    total: int
    passed: int
    failed: int
    max_abs_error: float
    failures: list = field(default_factory=list)  # (index, seq, computed, stored, err)


class SpecLib:
    def __init__(self, *, pepdata_path, rt_path, ms2_path,
                 proteins, mods_by_id, rt, chg_max):
        self.pepdata_path = pepdata_path
        self.rt_path = rt_path
        self.ms2_path = ms2_path
        self.proteins = proteins
        self.mods_by_id = mods_by_id
        self.rt = rt
        self.chg_max = chg_max

    @property
    def num_peptides(self) -> int:
        return len(self.rt)

    @classmethod
    def open(cls, *, pepdata_path: str, rt_path: str, ms2_path: str,
             fasta_path: str, mod_path: str) -> "SpecLib":
        proteins = parse_fasta(fasta_path)
        mods_by_id = {m.mod_id: m for m in parse_modifications(mod_path)}
        rt = read_rt_pred(rt_path)
        chg_max = read_chg_max_from_trailer(ms2_path)
        if not (1 <= chg_max <= 6):
            raise ValueError(f"chg_max {chg_max} out of range [1,6]")
        return cls(pepdata_path=pepdata_path, rt_path=rt_path,
                   ms2_path=ms2_path, proteins=proteins,
                   mods_by_id=mods_by_id, rt=rt, chg_max=chg_max)

    @classmethod
    def open_dir(cls, library_dir: str, *, fasta_path: str,
                 mod_path: str) -> "SpecLib":
        return cls.open(
            pepdata_path=os.path.join(library_dir, "pepdata.pdb"),
            rt_path=os.path.join(library_dir, "pepdata.rt.predb"),
            ms2_path=os.path.join(library_dir, "pepdata.ms2.predb"),
            fasta_path=fasta_path, mod_path=mod_path)

    def iter_peptides(self):
        """锁步流式：pdb+RT+MS2 同序逐肽段 yield（已填 pred_rt/pred_ms2）。"""
        ms2 = iter_ms2_records(self.ms2_path)
        n_rt = len(self.rt)
        i = -1
        for i, pep in enumerate(iter_pepdata(
                self.pepdata_path, self.proteins, self.mods_by_id)):
            if i >= n_rt:
                raise ValueError(
                    f"peptide count exceeds RT count {n_rt}")
            pep.pred_rt = self.rt[i]
            d = {}
            for chg in range(1, self.chg_max + 1):
                try:
                    d[chg] = next(ms2)
                except StopIteration:
                    raise ValueError(
                        f"ms2 records exhausted at peptide {i} charge {chg}")
            pep.pred_ms2 = d
            yield pep
        if i + 1 != n_rt:
            raise ValueError(
                f"peptide count {i + 1} != RT count {n_rt}")

    def validate_masses(self, element_path: str, aa_path: str,
                        tol: float = 0.01, limit: int | None = None
                        ) -> MassValidationReport:
        em = parse_element_masses(element_path)
        res = parse_residue_masses(aa_path, em)
        water = water_mass(em)
        failures = []
        max_err = 0.0
        passed = total = 0
        for pep in iter_pepdata(self.pepdata_path, self.proteins,
                               self.mods_by_id):
            computed = (water
                        + sum(res.get(a, 0.0) for a in pep.sequence)
                        + sum(m.mono_mass for m in pep.mods))
            err = abs(computed - pep.neutral_mass)
            if err > max_err:
                max_err = err
            if err <= tol:
                passed += 1
            elif len(failures) < 20:
                failures.append((total, pep.sequence, computed,
                                 pep.neutral_mass, err))
            total += 1
            if limit is not None and total >= limit:
                break
        return MassValidationReport(
            total=total, passed=passed, failed=total - passed,
            max_abs_error=max_err, failures=failures)
```

- [ ] **Step 4: 导出符号**

Replace `spectrum/speclib/__init__.py` content with:

```python
"""pFind 谱库（spectral library）二进制读取模块。"""
from .speclib import SpecLib, MassValidationReport
from .pepdata import LibPeptide, ModSite, iter_pepdata, read_pepdata
from .predictions import FragIon, read_rt_pred, iter_ms2_records, read_chg_max_from_trailer

__all__ = [
    "SpecLib", "MassValidationReport", "LibPeptide", "ModSite", "FragIon",
    "iter_pepdata", "read_pepdata", "read_rt_pred", "iter_ms2_records",
    "read_chg_max_from_trailer",
]
```

- [ ] **Step 5: 运行测试，确认通过**

Run: `python -m pytest tests/test_speclib_loader.py -q`
Expected: PASS（4 passed）

- [ ] **Step 6: 跑全部 speclib 测试**

Run: `python -m pytest tests/test_speclib_config_io.py tests/test_speclib_pepdata.py tests/test_speclib_predictions.py tests/test_speclib_loader.py -q`
Expected: PASS（19 passed）

- [ ] **Step 7: 写分层文档**

- `docs/speclib/parts/speclib/L2_role.md`：职责=顶层锁步流式 loader + 质量自校验；接口=`SpecLib.open`/`open_dir`/`iter_peptides`/`validate_masses`/`num_peptides`/`MassValidationReport`；依赖=config_io+pepdata+predictions。
- `docs/speclib/parts/speclib/L3_details.md`：`open_dir` 定位三文件；chg_max 从尾巴解析并约束 [1,6]；`iter_peptides` 锁步逻辑（pdb 生成器 + `rt[i]` + 每肽段取 `chg_max` 条 MS2）；对齐校验（peptide 数 vs RT 数、MS2 耗尽报错）；`validate_masses` 流式 + `limit`；质量公式 = Σ残基+H₂O+Σ修饰（C++ 已验证）；为何不全量物化。
- `docs/speclib/parts/speclib/L4_api.md`：`spectrum/speclib/speclib.py` 的 `SpecLib` 各方法签名/参数/返回/异常、`MassValidationReport` 字段。

- [ ] **Step 8: 提交**

```bash
git add spectrum/speclib/speclib.py spectrum/speclib/__init__.py tests/test_speclib_loader.py docs/speclib/parts/speclib/
git commit -m "feat(speclib): SpecLib lockstep-streaming loader + mass cross-validation

Co-authored-by: Copilot <223556219+Copilot@users.noreply.github.com>"
```

---

### Task 5: CLI — 真实谱库验证工具（流式，不 OOM）

> 用此 CLI 流式加载真实谱库（4.4GB 不全载内存），打印摘要、样例肽段、质量交叉校验。`--mass-limit N` 只校验前 N 条以加速。

**Files:**
- Create: `tools/speclib_inspect.py`
- Modify: `tests/conftest.py`（把 `lib_files` fixture 移入以便复用）
- Modify: `tests/test_speclib_loader.py`（删除本地 `lib_files`，改用 conftest）
- Test: `tests/test_speclib_inspect_cli.py`
- Docs: `docs/speclib/parts/speclib_inspect/{L2_role,L3_details,L4_api}.md`、回填 `docs/speclib/L1_overview.md`

- [ ] **Step 1: 把 `lib_files` fixture 从 test_speclib_loader.py 移入 conftest.py**

Cut the entire `lib_files` fixture definition (the `@pytest.fixture def lib_files(...)` block) from `tests/test_speclib_loader.py`, and append it to `tests/conftest.py` (identical body — `import pytest` is already present in conftest):

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
    (tmp_path / "pepdata.ms2.predb").write_bytes(build_ms2(
        [[(0, 0, 1.0)], [(1, 1, 0.5)], [(0, 0, 0.8)], [(2, 3, 0.3)]],
        chg_max=2, n_peptides=2))
    (tmp_path / "pepdata.rt.predb").write_bytes(build_rt([20.0, 21.5]))
    return tmp_path
```

- [ ] **Step 2: 写失败测试**

Create `tests/test_speclib_inspect_cli.py`:

```python
"""测试 speclib_inspect CLI 的 summarize（用合成 fixture，不依赖大文件）。"""
from tools.speclib_inspect import summarize


def test_summarize_runs_on_fixture(lib_files):
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

- [ ] **Step 3: 运行测试，确认失败**

Run: `python -m pytest tests/test_speclib_inspect_cli.py -q`
Expected: FAIL（`ModuleNotFoundError: No module named 'tools.speclib_inspect'`）

- [ ] **Step 4: 实现 CLI**

Create `tools/speclib_inspect.py`:

```python
"""流式加载 pFind 谱库并打印摘要 + 质量交叉校验，用于真实文件验证。

用法:
  python -m tools.speclib_inspect --library-dir DIR \\
      --fasta merge_human_ecoli_yeast.fasta --mod modification.ini \\
      [--element element.ini --aa aa.ini] [--n-samples 5] \\
      [--tol 0.01] [--mass-limit N]
"""
import argparse

from spectrum.speclib import SpecLib


def summarize(*, library_dir: str, fasta_path: str, mod_path: str,
              element_path: str | None = None, aa_path: str | None = None,
              n_samples: int = 5, tol: float = 0.01,
              mass_limit: int | None = None) -> str:
    lib = SpecLib.open_dir(library_dir, fasta_path=fasta_path,
                           mod_path=mod_path)
    lines = []
    lines.append(f"peptides: {lib.num_peptides}")
    lines.append(f"chg_max: {lib.chg_max}")
    if len(lib.rt):
        lines.append(f"rt range (min): {min(lib.rt):.3f} .. {max(lib.rt):.3f}")

    # 流式取前 n_samples 个肽段（含 RT/MS2），不全载
    samples = []
    for pep in lib.iter_peptides():
        samples.append(pep)
        if len(samples) >= n_samples:
            break
    for pep in samples:
        modstr = ",".join(f"{m.pos}:{m.name}" for m in pep.mods) or "-"
        top = sorted((ion for ions in pep.pred_ms2.values() for ion in ions),
                     key=lambda x: x.intensity, reverse=True)[:3]
        topstr = " ".join(
            f"{i.ion_type}{i.frag_pos}^{i.frag_charge}={i.intensity:.2f}"
            for i in top)
        lines.append(
            f"  {pep.sequence} mods=[{modstr}] mass={pep.neutral_mass:.4f} "
            f"rt={pep.pred_rt:.2f} top_ms2=[{topstr}]")

    if element_path and aa_path:
        rep = lib.validate_masses(element_path, aa_path, tol=tol,
                                  limit=mass_limit)
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
    ap.add_argument("--mass-limit", type=int, default=None,
                    help="只校验前 N 条质量（加速）")
    args = ap.parse_args()
    print(summarize(
        library_dir=args.library_dir, fasta_path=args.fasta,
        mod_path=args.mod, element_path=args.element, aa_path=args.aa,
        n_samples=args.n_samples, tol=args.tol, mass_limit=args.mass_limit))


if __name__ == "__main__":
    main()
```

- [ ] **Step 5: 运行测试，确认通过**

Run: `python -m pytest tests/test_speclib_inspect_cli.py tests/test_speclib_loader.py -q`
Expected: PASS（5 passed；确认移动 fixture 后 loader 测试仍通过）

- [ ] **Step 6: 跑全部 speclib 测试**

Run: `python -m pytest tests/ -k speclib -q`
Expected: PASS（20 passed）

- [ ] **Step 7: 写分层文档（含回填 L1）**

- `docs/speclib/parts/speclib_inspect/L2_role.md`：职责=真实库验证 CLI；接口=`summarize(...)` 函数 + `main()`/命令行参数；依赖=speclib。
- `docs/speclib/parts/speclib_inspect/L3_details.md`：流式取样（前 N 肽段不全载）、质量校验 `--mass-limit` 加速、输出字段含义（peptides/chg_max/rt range/样例/mass pass）、4.4GB 不 OOM 的原因。
- `docs/speclib/parts/speclib_inspect/L4_api.md`：`tools/speclib_inspect.py` 的 `summarize` 签名/参数/返回、CLI 参数表、真实库运行示例命令。
- 回填 `docs/speclib/L1_overview.md`：补全"快速上手"（`SpecLib.open_dir`+`iter_peptides` 示例、`python -m tools.speclib_inspect` 命令）、"组件索引"（5 个 parts 链接）、"关键事实"，确保 L1 与最终代码一致。

- [ ] **Step 8: 提交**

```bash
git add tools/speclib_inspect.py tests/conftest.py tests/test_speclib_inspect_cli.py tests/test_speclib_loader.py docs/speclib/parts/speclib_inspect/ docs/speclib/L1_overview.md
git commit -m "feat(speclib): speclib_inspect CLI for real-library validation (streaming)

Co-authored-by: Copilot <223556219+Copilot@users.noreply.github.com>"
```

---

## 真实文件验证（实现后执行）

谱库已确认在 `~/share/2026_06_07_kongweisa_guangshan_puku/{lib-2th,lib_5th,lib-normal}`（目录含 `pepdata.pdb`/`pepdata.rt.predb`/`pepdata.ms2.predb`）。运行：

```bash
python -m tools.speclib_inspect \
  --library-dir ~/share/2026_06_07_kongweisa_guangshan_puku/lib-2th \
  --fasta ../puku/merge_human_ecoli_yeast.fasta \
  --mod ../puku/modification.ini \
  --element ../puku/element.ini --aa ../puku/aa.ini \
  --n-samples 10 --tol 0.01 --mass-limit 200000
```

**预期（已用一次性脚本预验证）：** `peptides: 3124520`、`chg_max: 4`、`mass pass: N/N`（100%，max_abs_err≈0）。`--mass-limit` 控制质量校验条数（全量 312 万约 30s）；不带 limit 跑全量以最终确认。

---

## Self-Review

- **Spec coverage**：模块结构（config_io/pepdata/predictions/speclib）✓ Task1–4；二进制格式（pdb/rt/ms2+文本尾巴）✓ Task2–3；mod_id 映射 ✓ Task1；FASTA 解析 ✓ Task1；数据模型 ✓；**锁步流式 + 体量安全** ✓ Task4 `iter_peptides`；**MS2 文本尾巴跳过** ✓ Task3 `iter_ms2_records`；**chg_max 从尾巴解析** ✓ Task3 `read_chg_max_from_trailer`；质量交叉校验 ✓ Task4；`mod_pep_bytes` ✓ Task2；计数一致性（RT 数=M）✓ Task4；CLI 流式不 OOM + `--mass-limit` ✓ Task5；真实文件验证小节 ✓；非目标（不接入 pipeline、不做偏移索引）已遵守。
- **边界条件覆盖**：B1 M=Σmod_pep_num 逐变体对齐 ✓ Task2/Task4；B2 `n_size==0` 空记录 ✓ Task3；B3 `mod_pep_num==0` ✓ Task2；尾巴停止 ✓ Task3；chg_max∈[1,6] 硬校验 ✓ Task4；各电荷桶不对称——不做对称假设。
- **Placeholder scan**：无 TBD/TODO；每步含完整代码与精确命令/期望输出。
- **分层文档**：每个 Task 含 L2/L3/L4 文档步骤并随代码同 commit；Task1 建 L1 骨架、Task5 回填 L1。结构 `docs/speclib/L1_overview.md + parts/<组件>/{L2_role,L3_details,L4_api}.md`。
- **Type/name consistency**：`Protein/ModEntry/ModSite/LibPeptide/FragIon/MassValidationReport/SpecLib` 跨任务一致；`parse_fasta/parse_modifications/parse_element_masses/parse_residue_masses/water_mass/iter_pepdata/read_pepdata/read_rt_pred/iter_ms2_records/read_chg_max_from_trailer/SpecLib.open/open_dir/iter_peptides/validate_masses` 签名一致；struct 串 `'<IIbbbbIQ'/'<db'/'<bi'/'<h'/'<bbf'` 全程统一；`read_rt_pred` 返回 `array('f')`（测试用 `list(...)` 比较）。
