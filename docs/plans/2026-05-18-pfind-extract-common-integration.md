# pfind 支持与 extract_com 融合 实施计划

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** 让 ms2-met 原生支持 pfind 搜索结果（含 FDR 与 decoy 过滤），并把 extract_com 融合为项目内通用 N 引擎 CLI 工具，消除代码重复。

**Architecture:** pfind loader 作为 `LightResult._load_from_pfind_input` 方法接入 `search_engine_type = 3`，解析逻辑集中在新文件 `spectrum/pfind_parser.py`。extract_common 作为 `tools/extract_common.py` CLI，直接 import ms2-met 内部模块，无代码重复；通用 N 引擎交集（正例）+ 并集（负例）逻辑，物种 marker 决定标签。`PSMInfo` 新增 `q_value` / `score` / `label_type` 三个可选字段，向后兼容老 JSON。

**Tech Stack:** Python 3.13、pandas、pyteomics（Unimod）、pytest、configparser。

**前置条件：**
- conda 环境名：`jianyan`
- 激活方式：`source /opt/miniconda3/etc/profile.d/conda.sh && conda activate jianyan`
- 不要用真实大 raw 文件做测试——所有测试用小型手工 fixture
- 设计文档：`docs/specs/2026-05-18-pfind-extract-common-integration.md`

**文件结构：**
- 创建 `spectrum/pfind_parser.py`（修饰解析、m/z 转换、文件加载）
- 创建 `tools/__init__.py`、`tools/extract_common.py`（N 引擎工具）
- 创建 `tests/__init__.py`（测试包）
- 创建 `tests/test_pfind_parser.py`（pfind 解析单元测试）
- 创建 `tests/test_psm_info_compat.py`（PSMInfo 向后兼容测试）
- 创建 `tests/test_extract_common.py`（N 引擎工具测试）
- 创建 `tests/fixtures/sample_pfind.qry.res`（小型 pfind 测试数据）
- 创建 `tests/fixtures/sample_pfind_dir/`（目录扫描测试用，含 2 个文件）
- 创建 `extract_common_config.ini.example`（extract_common 配置示例）
- 修改 `spectrum/psm_info.py`（扩展 PSMInfo 字段）
- 修改 `spectrum/light_result.py`（添加 `_load_from_pfind_input` 方法）
- 修改 `manager/light_result_manager.py`（添加 search_engine_type = 3 分支）
- 修改 `constant/keys.py`（添加 pfind 相关常量）
- 修改 `config.ini`（补充 pfind 配置示例与注释）
- 修改 `PROJECT_INFO.md`（更新引擎支持表）
- 创建 `../extract_com/README.md`（迁移说明，标记为 deprecated）

---

## Task 1: 创建测试基础设施

**Files:**
- Create: `tests/__init__.py`
- Create: `tests/fixtures/sample_pfind.qry.res`
- Create: `tests/fixtures/sample_pfind_dir/raw1.qry.res`
- Create: `tests/fixtures/sample_pfind_dir/raw2.qry.res`
- Create: `tests/conftest.py`

- [ ] **Step 1: 创建 tests 包结构**

```bash
mkdir -p tests/fixtures/sample_pfind_dir
touch tests/__init__.py
touch tests/fixtures/__init__.py
```

- [ ] **Step 2: 创建单文件 pfind fixture**

Create `tests/fixtures/sample_pfind.qry.res`（tab 分隔，列必须与真实 pfind 一致）：

```
PeptideSequence	Modifications	PepMass	PredRT	CleavageType	ProNCTerm	Proteins	MH+	Charge	ScanNo	RawScore	DeltaMassPPM	DeltaRT(Min)	FinalScore	QValue
TGVHHYSGNNIELGTACGK	17,Carbamidomethyl[C];	2013.932673	29.033	3	0	sp|P62888|RL30_HUMAN/	2014.941562	2	19856	30.318819	0.801	-0.301	2.74403e-06	0
DENQSINHQMAQEDAQR		2012.860630	26.057	3	0	sp|P20073|ANXA7_HUMAN/	2013.868506	2	18394	26.398899	0.298	0.574	1.99755e-06	0
HIGH_FDR_PEP		1500.000000	30.000	3	0	sp|FAKE|TEST_HUMAN/	1501.007264	2	10000	5.000000	0.100	0.100	1.0e-04	0.05
DECOY_PEP		1600.000000	40.000	3	0	REV_sp|FAKE|REV_TEST_HUMAN/	1601.007264	2	11000	8.000000	0.200	0.150	5.0e-06	0
XYAAAA		1000.000000	20.000	3	0	sp|FAKE|INVALID_HUMAN/	1001.007264	2	5000	10.000000	0.000	0.000	1.0e-06	0
TRYPSINPEP		800.500000	50.000	3	0	sp|P00000|ECOLI_PROT_ECOLI/	801.507264	2	8000	15.000000	0.500	0.200	2.0e-06	0
```

注意：这个 fixture 设计如下：
- 第 1 行：正常人源 PSM，含 Carbamidomethyl 修饰
- 第 2 行：正常人源 PSM，无修饰
- 第 3 行：QValue=0.05，应被 FDR 过滤
- 第 4 行：REV_ 前缀的 decoy
- 第 5 行：序列含 X，PSMInfo.valid() 应拒绝
- 第 6 行：E.coli（用于物种标记测试）

- [ ] **Step 3: 创建目录扫描 fixture**

Create `tests/fixtures/sample_pfind_dir/raw1.qry.res`：

```
PeptideSequence	Modifications	PepMass	PredRT	CleavageType	ProNCTerm	Proteins	MH+	Charge	ScanNo	RawScore	DeltaMassPPM	DeltaRT(Min)	FinalScore	QValue
PEPTIDE1		1100.500000	25.000	3	0	sp|A0001|TEST1_HUMAN/	1101.507264	2	5000	20.000000	0.500	0.100	1.0e-06	0
PEPTIDE2		1200.500000	30.000	3	0	sp|A0002|TEST2_HUMAN/	1201.507264	2	6000	22.000000	0.300	0.150	1.5e-06	0
```

Create `tests/fixtures/sample_pfind_dir/raw2.qry.res`：

```
PeptideSequence	Modifications	PepMass	PredRT	CleavageType	ProNCTerm	Proteins	MH+	Charge	ScanNo	RawScore	DeltaMassPPM	DeltaRT(Min)	FinalScore	QValue
PEPTIDE3		1300.500000	35.000	3	0	sp|A0003|TEST3_HUMAN/	1301.507264	3	7000	24.000000	0.700	0.200	2.0e-06	0
```

- [ ] **Step 4: 创建 conftest.py 提供 fixture 路径**

```python
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
```

- [ ] **Step 5: 提交**

```bash
git add tests/
git commit -m "test: 创建 pfind 测试基础设施（fixtures + conftest）"
```

---

## Task 2: 扩展 PSMInfo 添加 q_value/score/label_type 字段

**Files:**
- Modify: `spectrum/psm_info.py`
- Create: `tests/test_psm_info_compat.py`

- [ ] **Step 1: 写失败的测试**

Create `tests/test_psm_info_compat.py`：

```python
"""测试 PSMInfo 新字段与向后兼容。"""
import numpy as np
from spectrum.psm_info import PSMInfo


def _make_basic_psm(**overrides):
    defaults = dict(
        sequence="AGFAGDDAPK",
        charge=2,
        modify=[],
        rt=np.float32(50.0),
        precursor_mz=np.float32(500.0),
        raw_title="test_raw",
        protein_names="sp|P00000|TEST_HUMAN/",
    )
    defaults.update(overrides)
    return PSMInfo(**defaults)


def test_psminfo_new_fields_default_none():
    """未显式给新字段时应默认为 None。"""
    psm = _make_basic_psm()
    assert psm._q_value is None
    assert psm._score is None
    assert psm._label_type is None


def test_psminfo_new_fields_set():
    """显式给的新字段应被存储。"""
    psm = _make_basic_psm(q_value=0.001, score=20.5, label_type="positive")
    assert psm._q_value == 0.001
    assert psm._score == 20.5
    assert psm._label_type == "positive"


def test_psminfo_to_dict_omits_none_new_fields():
    """to_dict 在新字段为 None 时不应输出，保持老格式兼容。"""
    psm = _make_basic_psm()
    d = psm.to_dict()
    assert "q_value" not in d
    assert "score" not in d
    assert "label_type" not in d


def test_psminfo_to_dict_includes_new_fields_when_set():
    """to_dict 在新字段非 None 时应输出。"""
    psm = _make_basic_psm(q_value=0.001, score=20.5, label_type="positive")
    d = psm.to_dict()
    assert d["q_value"] == 0.001
    assert d["score"] == 20.5
    assert d["label_type"] == "positive"


def test_psminfo_from_dict_old_json_no_new_fields():
    """老 JSON 没有新字段时，from_dict 应回填 None。"""
    old_data = {
        "sequence": "AGFAGDDAPK",
        "charge": 2,
        "modify": [],
        "rt": 50.0,
        "precursor_mz": 500.0,
        "raw_title": "test_raw",
        "protein_names": "sp|P00000|TEST_HUMAN/",
    }
    psm = PSMInfo.from_dict(old_data)
    assert psm._q_value is None
    assert psm._score is None
    assert psm._label_type is None
    assert psm._sequence == "AGFAGDDAPK"


def test_psminfo_from_dict_new_json_with_fields():
    """新 JSON 带新字段时，from_dict 应正确加载。"""
    new_data = {
        "sequence": "AGFAGDDAPK",
        "charge": 2,
        "modify": [],
        "rt": 50.0,
        "precursor_mz": 500.0,
        "raw_title": "test_raw",
        "protein_names": "sp|P00000|TEST_HUMAN/",
        "q_value": 0.005,
        "score": 18.7,
        "label_type": "negative",
    }
    psm = PSMInfo.from_dict(new_data)
    assert psm._q_value == 0.005
    assert psm._score == 18.7
    assert psm._label_type == "negative"


def test_psminfo_get_key_unchanged_by_new_fields():
    """get_key 不受新字段影响（保持 PSM 等价判定语义）。"""
    psm1 = _make_basic_psm()
    psm2 = _make_basic_psm(q_value=0.01, score=15.0, label_type="positive")
    assert psm1.get_key() == psm2.get_key()
```

- [ ] **Step 2: 运行测试，确认失败**

```bash
source /opt/miniconda3/etc/profile.d/conda.sh && conda activate jianyan
cd /home/verden/pfind/2025-fall/code/ms2-met
pytest tests/test_psm_info_compat.py -v
```

Expected: 大部分测试 FAIL（PSMInfo 当前没有 q_value/score/label_type 字段）。

- [ ] **Step 3: 修改 PSMInfo 添加新字段**

Modify `spectrum/psm_info.py` 的 `PSMInfo.__init__`、`to_dict`、`from_dict`：

```python
class PSMInfo:
    """ 记录一个 psm 的主要信息"""

    def __init__(
        self,
        sequence: str,
        charge: int,
        modify: [(int, int)],
        rt: np.float32,
        precursor_mz: np.float32,
        raw_title: str,
        protein_names: str,
        q_value: float | None = None,
        score: float | None = None,
        label_type: str | None = None,
    ):
        self._sequence = sequence
        self._charge = charge
        self._modify = modify
        self._rt = rt
        self._precursor_mz = precursor_mz
        self._raw_title = raw_title
        self._protein_names = protein_names
        self._q_value = q_value
        self._score = score
        self._label_type = label_type

    def to_dict(self):
        """将对象转为 JSON 兼容的字典"""
        d = {
            "sequence": self._sequence,
            "charge": self._charge,
            "modify": [list(pair) for pair in self._modify],
            "rt": float(self._rt),
            "precursor_mz": float(self._precursor_mz),
            "raw_title": self._raw_title,
            "protein_names": self._protein_names,
        }
        if self._q_value is not None:
            d["q_value"] = float(self._q_value)
        if self._score is not None:
            d["score"] = float(self._score)
        if self._label_type is not None:
            d["label_type"] = self._label_type
        return d

    @classmethod
    def from_dict(cls, data: dict):
        """从字典重建 PSMInfo 对象，对新字段做 None 兜底以兼容老 JSON"""
        return cls(
            sequence=data["sequence"],
            charge=data["charge"],
            modify=[(int(pos), int(mod))
                    for pos, mod in data["modify"]],
            rt=np.float32(data["rt"]),
            precursor_mz=np.float32(data["precursor_mz"]),
            raw_title=data["raw_title"],
            protein_names=data["protein_names"],
            q_value=data.get("q_value"),
            score=data.get("score"),
            label_type=data.get("label_type"),
        )
```

注意：原 `__repr__`、`get_key`、`valid`、`get_modify_mass`、`get_fragment_ions`、`get_SILAC_precursor_mz`、`get_C_N_HEAVY_precursor_mz`、`get_heavy_info` 等方法保持不变。

- [ ] **Step 4: 运行测试，确认通过**

```bash
pytest tests/test_psm_info_compat.py -v
```

Expected: 7 个测试全部 PASS。

- [ ] **Step 5: 验证 light_result.py 现有 loader 仍工作**

确保 DIANN/AlphaDIA loader 仍能用（它们没传新字段，应该走默认 None）。

```bash
python3 -c "
from spectrum.psm_info import PSMInfo
import numpy as np
psm = PSMInfo(sequence='AGK', charge=2, modify=[], rt=np.float32(10),
              precursor_mz=np.float32(500), raw_title='x', protein_names='y')
print('repr:', repr(psm))
print('to_dict:', psm.to_dict())
"
```

Expected: 输出正常，to_dict 不含 q_value/score/label_type。

- [ ] **Step 6: 提交**

```bash
git add spectrum/psm_info.py tests/test_psm_info_compat.py
git commit -m "feat: PSMInfo 扩展 q_value/score/label_type 字段（向后兼容）"
```

---

## Task 3: 实现 pfind 修饰解析器

**Files:**
- Create: `spectrum/pfind_parser.py`
- Create: `tests/test_pfind_parser.py`

- [ ] **Step 1: 写失败的测试**

Create `tests/test_pfind_parser.py`：

```python
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
    # "Biotin" 不在硬编码字典里，但 UniMod 数据库里 record_id = 21（实际值由 UniMod 决定）
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
```

- [ ] **Step 2: 运行测试，确认失败**

```bash
pytest tests/test_pfind_parser.py -v
```

Expected: 全部 FAIL（pfind_parser 模块不存在）。

- [ ] **Step 3: 创建 pfind_parser.py**

Create `spectrum/pfind_parser.py`：

```python
"""pfind 搜索引擎结果的解析工具。

提供修饰名称解析、MH+ → m/z 转换、raw_title 提取等工具。
"""
import os
import logging
from functools import lru_cache

from pyteomics import mass


# 质子质量
PROTON_MASS = 1.00727646677


# pfind 修饰名 → UniMod ID 硬编码字典（覆盖最常见的修饰，
# 避免每次都查 unimod.xml）。键含 pfind 风格的氨基酸标注。
PFIND_MOD_TO_UNIMOD: dict[str, int] = {
    # Carbamidomethyl
    "Carbamidomethyl[C]": 4,
    "Carbamidomethyl[AnyN-term]": 4,
    # Oxidation
    "Oxidation[M]": 35,
    "Oxidation[W]": 35,
    "Oxidation[H]": 35,
    # Phospho
    "Phospho[S]": 21,
    "Phospho[T]": 21,
    "Phospho[Y]": 21,
    # Acetyl
    "Acetyl[K]": 1,
    "Acetyl[ProteinN-term]": 1,
    "Acetyl[AnyN-term]": 1,
    # Methyl / Dimethyl / Trimethyl
    "Methyl[K]": 34,
    "Methyl[R]": 34,
    "Dimethyl[K]": 36,
    "Dimethyl[R]": 36,
    "Trimethyl[K]": 37,
    # Deamidated
    "Deamidated[N]": 7,
    "Deamidated[Q]": 7,
    # N 端 pyro 转换
    "Pyro-carbamidomethyl[AnyN-term]": 26,
    "Gln->pyro-Glu[AnyN-termQ]": 28,
    "Glu->pyro-Glu[AnyN-termE]": 27,
}


# 单例 UniMod 数据库（lazy）
_UNIMOD_DB = None


def _get_unimod_db():
    """Lazy 加载 UniMod 数据库（pyteomics 自带 OBO 解析）。"""
    global _UNIMOD_DB
    if _UNIMOD_DB is None:
        unimod_xml_path = os.path.join(
            os.path.dirname(os.path.dirname(os.path.abspath(__file__))),
            "unimod.xml",
        )
        if os.path.exists(unimod_xml_path):
            with open(unimod_xml_path, "rb") as f:
                _UNIMOD_DB = mass.Unimod(source=f)
        else:
            _UNIMOD_DB = mass.Unimod()
    return _UNIMOD_DB


@lru_cache(maxsize=1024)
def resolve_pfind_mod_name(name: str) -> int | None:
    """从 pfind 修饰名称解析出 UniMod ID。

    解析顺序：
      1. 先查硬编码字典（性能优先，覆盖常见修饰）
      2. 兑底用 UniMod 数据库按 base name 查询（取 "[" 之前的部分）
      3. 未命中则返回 None（调用方应 log warning 并跳过该修饰）

    Returns:
        UniMod ID（整数），或 None 表示无法解析。
    """
    if name in PFIND_MOD_TO_UNIMOD:
        return PFIND_MOD_TO_UNIMOD[name]

    # 兑底：取基础名（如 "Carbamidomethyl[C]" → "Carbamidomethyl"）
    base_name = name.split("[")[0] if "[" in name else name
    try:
        db = _get_unimod_db()
        record = db.by_title(base_name)
        if record is not None:
            return int(record.get("record_id"))
    except (KeyError, Exception) as e:
        logging.debug(f"UniMod 查询失败 name={name} base={base_name}: {e}")
    return None


def parse_pfind_modify(modify_str: str) -> list[tuple[int, int]]:
    """解析 pfind Modifications 字段。

    输入格式（pfind 输出）："3,Carbamidomethyl[C];10,Carbamidomethyl[C];"
      - 位置是 1-based
      - 多个修饰用 ";" 分隔，末尾可能有 ";"

    输出：list of (0-based position, unimod_id)。

    未知修饰会被跳过并 log warning。
    """
    if not modify_str or not modify_str.strip():
        return []

    modifications: list[tuple[int, int]] = []
    for entry in modify_str.rstrip(";").split(";"):
        entry = entry.strip()
        if not entry:
            continue

        # 每个修饰应为 "位置,名称"
        try:
            pos_str, name = entry.split(",", 1)
            pos = int(pos_str.strip()) - 1  # 1-based → 0-based
            name = name.strip()
        except (ValueError, IndexError):
            logging.warning(f"pfind 修饰格式无法解析: '{entry}'")
            continue

        unimod_id = resolve_pfind_mod_name(name)
        if unimod_id is None:
            logging.warning(f"pfind 修饰未知，跳过: '{name}'")
            continue

        modifications.append((pos, unimod_id))

    return modifications


def mhp_to_mz(mhp: float, charge: int) -> float:
    """pfind MH+ → 带 charge 的 m/z。

    MH+ 表示 1+ 离子质量 = 中性质量 + 1 × proton_mass。
    m/z(z) = (中性质量 + z × proton_mass) / z
           = (MH+ + (z-1) × proton_mass) / z
    """
    if charge <= 0:
        raise ValueError(f"charge 必须 > 0，得到 {charge}")
    return (mhp + (charge - 1) * PROTON_MASS) / charge


def extract_raw_title_from_pfind_path(path: str) -> str:
    """从 pfind .qry.res 文件路径提取 raw_title（去掉目录和 .qry.res 后缀）。"""
    basename = os.path.basename(path)
    if basename.endswith(".qry.res"):
        return basename[: -len(".qry.res")]
    return basename
```

- [ ] **Step 4: 运行测试，确认通过**

```bash
pytest tests/test_pfind_parser.py -v
```

Expected: 全部 PASS。

如果 `test_resolve_pfind_mod_name_unimod_fallback` 因为 Biotin 不是 UniMod 标准 title 而 fail，可改成实测 UniMod 数据库中存在的 title 名（例如 "Phospho"——但这个在硬编码字典里也有；用 "Carbox" 之类的）。验证方法：

```bash
python3 -c "
from spectrum.pfind_parser import _get_unimod_db
db = _get_unimod_db()
for name in ['Biotin', 'Sulfo', 'Farnesyl', 'Pentose']:
    try:
        r = db.by_title(name)
        print(name, '→', r.get('record_id'))
    except Exception as e:
        print(name, '→ ERROR:', e)
"
```

选一个返回 record_id 的修饰名，更新测试中的修饰名。

- [ ] **Step 5: 提交**

```bash
git add spectrum/pfind_parser.py tests/test_pfind_parser.py
git commit -m "feat: pfind 修饰解析器（硬编码字典 + UniMod 兑底）"
```

---

## Task 4: 实现 pfind 文件加载（单文件）

**Files:**
- Modify: `spectrum/pfind_parser.py`（添加 load_pfind_file 函数）
- Modify: `spectrum/light_result.py`（添加 `_load_from_pfind_input` 方法）
- Modify: `tests/test_pfind_parser.py`（新增加载测试）

- [ ] **Step 1: 写失败的测试**

Append to `tests/test_pfind_parser.py`：

```python
import numpy as np
from spectrum.pfind_parser import load_pfind_file


# === load_pfind_file ===

def test_load_pfind_file_basic(sample_pfind_file):
    """加载单文件，确认 FDR/decoy/合法性过滤都生效。"""
    psms = load_pfind_file(sample_pfind_file, qvalue_threshold=0.01)
    # fixture 中 6 行 PSM：
    #   - 2 个合法 HUMAN（保留）
    #   - 1 个 QValue=0.05（被 FDR 过滤）
    #   - 1 个 REV_（被 decoy 过滤）
    #   - 1 个 X 序列（被 PSMInfo.valid 过滤）
    #   - 1 个 E.coli（保留，FDR/decoy/valid 都通过）
    # → 共保留 3 个
    assert len(psms) == 3


def test_load_pfind_file_qvalue_filter(sample_pfind_file):
    """所有保留的 PSM 的 q_value 应都 <= 阈值。"""
    psms = load_pfind_file(sample_pfind_file, qvalue_threshold=0.01)
    for psm in psms:
        assert psm._q_value <= 0.01


def test_load_pfind_file_no_decoy(sample_pfind_file):
    """保留的 PSM 中不应有 REV_ 前缀。"""
    psms = load_pfind_file(sample_pfind_file, qvalue_threshold=0.01)
    for psm in psms:
        assert not psm._protein_names.startswith("REV_")


def test_load_pfind_file_raw_title_extracted(sample_pfind_file):
    """所有 PSM 的 raw_title 应来自文件名。"""
    psms = load_pfind_file(sample_pfind_file, qvalue_threshold=0.01)
    for psm in psms:
        assert psm._raw_title == "sample_pfind"


def test_load_pfind_file_precursor_mz_computed(sample_pfind_file):
    """PSM 的 precursor_mz 应通过 mhp_to_mz 正确计算。"""
    from spectrum.pfind_parser import mhp_to_mz
    psms = load_pfind_file(sample_pfind_file, qvalue_threshold=0.01)
    # 找到 TGVHHYSGNNIELGTACGK 的 PSM
    target = next(p for p in psms if p._sequence == "TGVHHYSGNNIELGTACGK")
    expected_mz = mhp_to_mz(2014.941562, 2)
    assert abs(float(target._precursor_mz) - expected_mz) < 1e-6


def test_load_pfind_file_modify_parsed(sample_pfind_file):
    """带修饰的 PSM 应被正确解析。"""
    psms = load_pfind_file(sample_pfind_file, qvalue_threshold=0.01)
    target = next(p for p in psms if p._sequence == "TGVHHYSGNNIELGTACGK")
    # "17,Carbamidomethyl[C];" → [(16, 4)]
    assert target._modify == [(16, 4)]


def test_load_pfind_file_qvalue_and_score_set(sample_pfind_file):
    """每个 PSM 的 q_value 和 score 都应被填入。"""
    psms = load_pfind_file(sample_pfind_file, qvalue_threshold=0.01)
    for psm in psms:
        assert psm._q_value is not None
        assert psm._score is not None


def test_load_pfind_file_strict_qvalue(sample_pfind_file):
    """提高 qvalue_threshold 应纳入更多 PSM。"""
    relaxed = load_pfind_file(sample_pfind_file, qvalue_threshold=0.1)
    strict = load_pfind_file(sample_pfind_file, qvalue_threshold=0.01)
    # 放宽后会多出 HIGH_FDR_PEP（QValue=0.05）
    assert len(relaxed) == len(strict) + 1
```

- [ ] **Step 2: 运行测试，确认失败**

```bash
pytest tests/test_pfind_parser.py::test_load_pfind_file_basic -v
```

Expected: FAIL（`load_pfind_file` 不存在）。

- [ ] **Step 3: 在 pfind_parser.py 添加 load_pfind_file 函数**

Append to `spectrum/pfind_parser.py`：

```python
import pandas as pd

from spectrum.psm_info import PSMInfo
import numpy as np


# pfind .qry.res 文件中标识 decoy 的蛋白前缀
PFIND_DECOY_PREFIX = "REV_"


def load_pfind_file(
    file_path: str,
    qvalue_threshold: float = 0.01,
) -> list:
    """加载单个 pfind .qry.res 文件并应用过滤。

    过滤顺序：
      1. QValue > qvalue_threshold → 丢弃（FDR 过滤）
      2. Proteins 以 "REV_" 开头 → 丢弃（decoy 过滤）
      3. PSMInfo.valid() 为 False → 丢弃（含 X 等）

    Returns:
        list[PSMInfo]
    """
    if not os.path.exists(file_path):
        logging.error(f"pfind 文件不存在: {file_path}")
        return []

    logging.info(f"正在加载 pfind 文件: {file_path}")
    df = pd.read_csv(file_path, sep="\t")

    raw_title = extract_raw_title_from_pfind_path(file_path)
    psms: list[PSMInfo] = []

    n_total = len(df)
    n_filtered_fdr = 0
    n_filtered_decoy = 0
    n_filtered_invalid = 0
    n_parse_error = 0

    for row in df.itertuples(index=False):
        # FDR 过滤
        try:
            qvalue = float(getattr(row, "QValue"))
        except Exception:
            n_parse_error += 1
            continue
        if qvalue > qvalue_threshold:
            n_filtered_fdr += 1
            continue

        # Decoy 过滤
        proteins = str(getattr(row, "Proteins"))
        if proteins.startswith(PFIND_DECOY_PREFIX):
            n_filtered_decoy += 1
            continue

        # 字段提取
        try:
            modifications = parse_pfind_modify(str(getattr(row, "Modifications", "") or ""))
            charge = int(getattr(row, "Charge"))
            mhp_value = float(getattr(row, "_8") if hasattr(row, "_8") else getattr(row, "MH+"))
            precursor_mz = mhp_to_mz(mhp_value, charge)
            # 暂定 RT = PredRT + DeltaRT (Min)
            # pandas 处理 'DeltaRT(Min)' 列名时会替换非法字符；使用 _DeltaRT_Min_ 兑底
            pred_rt = float(getattr(row, "PredRT"))
            delta_rt = float(_get_delta_rt(row))
            rt = pred_rt + delta_rt
            score = float(getattr(row, "FinalScore"))
            sequence = str(getattr(row, "PeptideSequence"))
        except Exception as e:
            n_parse_error += 1
            logging.warning(f"pfind 行解析失败 file={raw_title}: {e}")
            continue

        psm = PSMInfo(
            sequence=sequence,
            charge=charge,
            modify=modifications,
            rt=np.float32(rt),
            precursor_mz=np.float32(precursor_mz),
            raw_title=raw_title,
            protein_names=proteins,
            q_value=qvalue,
            score=score,
        )

        if not psm.valid():
            n_filtered_invalid += 1
            continue

        psms.append(psm)

    logging.info(
        f"pfind 加载完成 {raw_title}: total={n_total}, "
        f"kept={len(psms)}, fdr_filtered={n_filtered_fdr}, "
        f"decoy_filtered={n_filtered_decoy}, "
        f"invalid={n_filtered_invalid}, parse_error={n_parse_error}"
    )
    return psms


def _get_delta_rt(row) -> float:
    """提取 DeltaRT(Min) 列。

    pandas itertuples 会把 'DeltaRT(Min)' 列名转换。我们尝试多种可能的名称。
    """
    for attr_name in ("_DeltaRT_Min_", "DeltaRT_Min_", "DeltaRT", "_13"):
        if hasattr(row, attr_name):
            try:
                return float(getattr(row, attr_name))
            except Exception:
                continue
    return 0.0
```

注：`_8` 和 `_13` 是 pandas 给非法 Python 标识符列名的备用属性名。我们使用 `getattr` 加备用名是为了健壮性。

- [ ] **Step 4: 实际测试一次以确认 pandas itertuples 列名映射**

```bash
python3 -c "
import pandas as pd
df = pd.read_csv('tests/fixtures/sample_pfind.qry.res', sep='\t')
print('列名:', list(df.columns))
for row in df.head(1).itertuples(index=False):
    print('row 属性:', [a for a in dir(row) if not a.startswith('_') or a.startswith('_') and a[1:].isdigit()][:30])
    print('字段值 try Proteins:', getattr(row, 'Proteins', 'N/A'))
    print('字段值 try MH+:', getattr(row, 'MH+', 'N/A'))
    print('字段值 _8:', getattr(row, '_8', 'N/A'))
    print('字段值 _13:', getattr(row, '_13', 'N/A'))
"
```

根据输出调整 `load_pfind_file` 中的属性名。重要：pandas 会把 `MH+` 这种含 `+` 的列名规范化，可能变成 `MH_` 或 `_8`（按位置）。

- [ ] **Step 5: 运行测试，确认通过**

```bash
pytest tests/test_pfind_parser.py -v
```

Expected: 全部 PASS。

- [ ] **Step 6: 提交**

```bash
git add spectrum/pfind_parser.py tests/test_pfind_parser.py
git commit -m "feat: 实现 pfind 单文件加载（FDR + decoy + 合法性过滤）"
```

---

## Task 5: 支持 pfind 目录扫描

**Files:**
- Modify: `spectrum/pfind_parser.py`（添加 load_pfind_path 函数）
- Modify: `tests/test_pfind_parser.py`

- [ ] **Step 1: 写失败的测试**

Append to `tests/test_pfind_parser.py`：

```python
from spectrum.pfind_parser import load_pfind_path


def test_load_pfind_path_directory(sample_pfind_dir):
    """目录扫描应加载目录下所有 .qry.res 文件。"""
    psms = load_pfind_path(sample_pfind_dir, qvalue_threshold=0.01)
    # raw1.qry.res 有 2 条；raw2.qry.res 有 1 条 → 共 3 条
    assert len(psms) == 3


def test_load_pfind_path_directory_raw_titles(sample_pfind_dir):
    """目录扫描出的 PSM 应携带正确的 raw_title。"""
    psms = load_pfind_path(sample_pfind_dir, qvalue_threshold=0.01)
    raw_titles = {p._raw_title for p in psms}
    assert raw_titles == {"raw1", "raw2"}


def test_load_pfind_path_single_file(sample_pfind_file):
    """传单文件路径应仅加载该文件。"""
    psms = load_pfind_path(sample_pfind_file, qvalue_threshold=0.01)
    # 与 load_pfind_file 同一结果
    assert len(psms) == 3
    raw_titles = {p._raw_title for p in psms}
    assert raw_titles == {"sample_pfind"}


def test_load_pfind_path_empty_directory(tmp_path):
    """空目录应返回空列表，不报错。"""
    psms = load_pfind_path(str(tmp_path), qvalue_threshold=0.01)
    assert psms == []


def test_load_pfind_path_nonexistent(tmp_path):
    """不存在的路径应返回空列表并 log error。"""
    psms = load_pfind_path(str(tmp_path / "nonexistent"), qvalue_threshold=0.01)
    assert psms == []
```

- [ ] **Step 2: 运行测试，确认失败**

```bash
pytest tests/test_pfind_parser.py::test_load_pfind_path_directory -v
```

Expected: FAIL（`load_pfind_path` 不存在）。

- [ ] **Step 3: 在 pfind_parser.py 添加 load_pfind_path 函数**

Append to `spectrum/pfind_parser.py`：

```python
import glob


def load_pfind_path(
    path: str,
    qvalue_threshold: float = 0.01,
) -> list:
    """加载 pfind 路径——目录则扫描所有 .qry.res 文件，单文件则只加载该文件。

    Args:
        path: 目录或单个 .qry.res 文件路径
        qvalue_threshold: FDR 阈值

    Returns:
        list[PSMInfo]
    """
    if not os.path.exists(path):
        logging.error(f"pfind 路径不存在: {path}")
        return []

    if os.path.isdir(path):
        files = sorted(glob.glob(os.path.join(path, "*.qry.res")))
        logging.info(f"pfind 目录扫描: {path}，找到 {len(files)} 个 .qry.res 文件")
    else:
        files = [path]

    all_psms = []
    for file_path in files:
        all_psms.extend(load_pfind_file(file_path, qvalue_threshold))

    logging.info(f"pfind 路径加载完毕: {path}，共 {len(all_psms)} 条 PSM")
    return all_psms
```

- [ ] **Step 4: 运行测试，确认通过**

```bash
pytest tests/test_pfind_parser.py -v
```

Expected: 全部 PASS。

- [ ] **Step 5: 提交**

```bash
git add spectrum/pfind_parser.py tests/test_pfind_parser.py
git commit -m "feat: pfind 路径加载支持目录扫描与单文件"
```

---

## Task 6: 在 LightResult 中接入 pfind loader

**Files:**
- Modify: `spectrum/light_result.py`
- Modify: `constant/keys.py`
- Modify: `manager/light_result_manager.py`
- Create: `tests/test_light_result_pfind.py`

- [ ] **Step 1: 写失败的测试**

Create `tests/test_light_result_pfind.py`：

```python
"""测试 LightResult 对 pfind 输入的支持。"""
import configparser
import pytest
from spectrum.light_result import LightResult


def test_load_from_pfind_input_file(sample_pfind_file):
    """LightResult._load_from_pfind_input 加载单文件。"""
    lr = LightResult()
    lr._load_from_pfind_input(sample_pfind_file, qvalue_threshold=0.01)
    assert lr.peptide_len == 3
    assert len(lr.psm_info) == 3


def test_load_from_pfind_input_directory(sample_pfind_dir):
    """LightResult._load_from_pfind_input 加载目录。"""
    lr = LightResult()
    lr._load_from_pfind_input(sample_pfind_dir, qvalue_threshold=0.01)
    assert lr.peptide_len == 3


def test_load_from_pfind_input_psm_has_q_value(sample_pfind_file):
    """加载后 PSM 应携带 q_value 字段。"""
    lr = LightResult()
    lr._load_from_pfind_input(sample_pfind_file, qvalue_threshold=0.01)
    for psm in lr.psm_info:
        assert psm._q_value is not None


def test_light_result_manager_dispatch_pfind(sample_pfind_file):
    """LightResultManager 应能根据 search_engine_type=3 分发到 pfind loader。"""
    from manager.light_result_manager import LightResultManager
    from constant.keys import ConfigKeys

    config = configparser.ConfigParser()
    config[ConfigKeys.INPUT] = {
        ConfigKeys.LIGHT_RESULT_PATH: sample_pfind_file,
        ConfigKeys.SEARCH_ENGINE_TYPE: "3",
        ConfigKeys.PFIND_QVALUE_THRESHOLD: "0.01",
    }

    mgr = LightResultManager(config=config, path=None, load_from_file=False)
    lr = mgr.get_light_result_object(sample_pfind_file)
    assert len(lr.psm_info) == 3
```

- [ ] **Step 2: 运行测试，确认失败**

```bash
pytest tests/test_light_result_pfind.py -v
```

Expected: FAIL（`_load_from_pfind_input` 不存在；ConfigKeys 没有 PFIND_QVALUE_THRESHOLD）。

- [ ] **Step 3: 修改 constant/keys.py 添加常量**

Modify `constant/keys.py` 的 `ConfigKeys` 类（在末尾添加）：

```python
class ConfigKeys(metaclass=ConstantsClass):
    """String constants for accessing the config."""

    INPUT = "input"
    RAW_NUM = "raw_num"
    RAW_PATH = "raw_path"
    LIGHT_RESULT_PATH = "light_result_file"
    SEARCH_ENGINE_TYPE = "search_engine_type"

    # pfind 特有配置
    PFIND_QVALUE_THRESHOLD = "pfind_qvalue_threshold"

    GENERAL = "general"
    WORK_DIRECTORY = "work_directory"
    MASS_TOL_PPM = "mass_tol_ppm"
    XIC_CYCLE_WINDOW = "xic_cycle_window"
    RESULT_FILE = "result_file"
    FEATURE_TYPE = "feature_type"
```

- [ ] **Step 4: 修改 LightResult 添加 _load_from_pfind_input 方法**

Modify `spectrum/light_result.py`：

在文件顶部 import 区添加：

```python
from spectrum.pfind_parser import load_pfind_path
```

在 `LightResult` 类中添加新方法（放在 `_load_from_dia_nn_input` 之后）：

```python
    def _load_from_pfind_input(
        self,
        light_result_path: str,
        qvalue_threshold: float = 0.01,
    ):
        """加载 pfind 搜索结果（.qry.res 单文件或目录）。"""
        self.psm_info = load_pfind_path(
            light_result_path, qvalue_threshold=qvalue_threshold)
        self.peptide_len = len(self.psm_info)
```

- [ ] **Step 5: 修改 LightResultManager 分发**

Modify `manager/light_result_manager.py` 的 `get_light_result_object` 方法：

```python
    def get_light_result_object(
        self,
        light_result_path: None | str = None,
    ) -> LightResult:
        """ 从路径中读取搜索引擎结果，根据 search_engine_type 分发 """

        light_result = LightResult()

        search_engine_type = self._config[ConfigKeys.INPUT].getint(
            ConfigKeys.SEARCH_ENGINE_TYPE, fallback=1)

        if search_engine_type == 0:
            light_result._load_from_pkl(light_result_path)
        elif search_engine_type == 1:
            light_result._load_from_dia_nn_input(light_result_path)
        elif search_engine_type == 2:
            light_result._load_from_alphadia_input(light_result_path)
        elif search_engine_type == 3:
            qvalue_threshold = self._config[ConfigKeys.INPUT].getfloat(
                ConfigKeys.PFIND_QVALUE_THRESHOLD, fallback=0.01)
            light_result._load_from_pfind_input(
                light_result_path, qvalue_threshold=qvalue_threshold)
        else:
            logging.error(
                f"错误搜索引擎类型: {search_engine_type}（支持 0/1/2/3）")

        return light_result
```

- [ ] **Step 6: 运行测试，确认通过**

```bash
pytest tests/test_light_result_pfind.py -v
```

Expected: 全部 PASS。

- [ ] **Step 7: 运行所有现有测试，确认无回归**

```bash
pytest tests/ -v
```

Expected: 所有测试 PASS。

- [ ] **Step 8: 提交**

```bash
git add spectrum/light_result.py constant/keys.py manager/light_result_manager.py tests/test_light_result_pfind.py
git commit -m "feat: LightResult 接入 pfind loader（search_engine_type=3）"
```

---

## Task 7: 更新 config.ini 注释与示例

**Files:**
- Modify: `config.ini`

- [ ] **Step 1: 修改 config.ini 增加 pfind 配置示例与注释**

Modify `config.ini`：

```ini

[input]
raw_num = 1
raw_path_1 = ./../20190830_HF_ZHW_hela_SILAC_DIA_350_1000_Rep1.mzML
raw_path_2 = ./20190830_HF_ZHW_hela_SILAC_DDIA_500_550_2Da_Rep2.raw

light_result_file = ../hela.json
# search_engine_type 取值：
#   0 = 自定义 JSON（来自 tools/extract_common.py 的输出）
#   1 = DIA-NN parquet
#   2 = AlphaDIA parquet
#   3 = pfind .qry.res（可传单文件或目录路径）
search_engine_type = 0
# 仅当 search_engine_type = 3 时生效；pfind 的 FDR (q-value) 阈值
pfind_qvalue_threshold = 0.01

[general]
# 生成特征的模式，0 为相同文件之间进行生成
# 1 为 正常的轻重标进行生成
feature_type = 0

work_directory = ./workspace

mass_tol_ppm = 10

xic_cycle_window = 6

result_file = result.csv
```

- [ ] **Step 2: 用本地 pfind 文件做 smoke test（仅加载几条 PSM）**

```bash
python3 -c "
import configparser
from spectrum.light_result import LightResult

lr = LightResult()
lr._load_from_pfind_input('tests/fixtures/sample_pfind.qry.res', qvalue_threshold=0.01)
print(f'加载 {lr.peptide_len} 条 PSM')
for psm in lr.psm_info[:3]:
    print(f'  {psm._sequence} z={psm._charge} mz={float(psm._precursor_mz):.4f} '
          f'rt={float(psm._rt):.3f} q={psm._q_value} score={psm._score}')
"
```

Expected: 输出 3 条 PSM 信息，含 q_value 和 score。

- [ ] **Step 3: 提交**

```bash
git add config.ini
git commit -m "docs(config): 在 config.ini 中标注 search_engine_type 选项与 pfind 配置"
```

---

## Task 8: 创建 tools/ 包与 extract_common 工具骨架

**Files:**
- Create: `tools/__init__.py`
- Create: `tools/extract_common.py`
- Create: `extract_common_config.ini.example`
- Create: `tests/test_extract_common.py`

- [ ] **Step 1: 创建 tools/__init__.py 与配置示例**

```bash
mkdir -p tools
touch tools/__init__.py
```

Create `extract_common_config.ini.example`：

```ini
# extract_common 工具配置示例
#
# 用法：python tools/extract_common.py --configpath extract_common_config.ini

[extract]
# 引擎列表，用逗号分隔，可任意顺序、任意数量
# 支持的引擎：pfind, diann, alphadia
engines = pfind, diann

# 物种 marker（可选）。给出时启用"正负例"模式：
#   正例 = 所有引擎都识别为含 marker 的肽段（交集）
#   负例 = 任一引擎识别为不含 marker 的肽段（并集）
# 留空时仅取所有引擎的交集，不分正负例
positive_species_marker = HUMAN

# 输出 JSON 路径（ms2-met 单引擎模式 search_engine_type=0 可直接读取）
result_file = ./datasets/hela-2da-pfind-diann.json

[engine.pfind]
path = ./pfind-dia/2th/
qvalue_threshold = 0.01

[engine.diann]
path = ./hela-mix-2da_report.parquet

[engine.alphadia]
# 如需启用 alphadia 引擎，把 engines 改为 "pfind, diann, alphadia" 并取消下面注释
# path = ./precursors.parquet
```

- [ ] **Step 2: 写失败的测试（先写工具核心 API 的测试）**

Create `tests/test_extract_common.py`：

```python
"""测试 tools/extract_common.py 的 N 引擎交并集逻辑。"""
import numpy as np
import pytest

from spectrum.psm_info import PSMInfo


def _make_psm(seq, charge, protein_names, rt=10.0, mz=500.0, raw="r"):
    return PSMInfo(
        sequence=seq,
        charge=charge,
        modify=[],
        rt=np.float32(rt),
        precursor_mz=np.float32(mz),
        raw_title=raw,
        protein_names=protein_names,
    )


def test_extract_intersection_no_marker():
    """无 positive_marker → 简单交集。"""
    from tools.extract_common import extract_n_engines_from_psms

    engine_psms = {
        "pfind": [
            _make_psm("AAA", 2, "sp|X|HUMAN/"),
            _make_psm("BBB", 2, "sp|X|HUMAN/"),
            _make_psm("CCC", 2, "sp|X|HUMAN/"),
        ],
        "diann": [
            _make_psm("AAA", 2, "sp|X|HUMAN/"),
            _make_psm("BBB", 2, "sp|X|HUMAN/"),
            _make_psm("DDD", 2, "sp|X|HUMAN/"),
        ],
    }
    result = extract_n_engines_from_psms(
        engine_psms, engine_order=["pfind", "diann"],
        positive_marker=None)
    seqs = sorted([p._sequence for p in result])
    assert seqs == ["AAA", "BBB"]  # 交集
    # label_type 应为 None
    assert all(p._label_type is None for p in result)


def test_extract_positive_negative_with_marker():
    """有 positive_marker → 正例（交集 + marker）+ 负例（并集 + 非 marker）。"""
    from tools.extract_common import extract_n_engines_from_psms

    engine_psms = {
        "pfind": [
            _make_psm("HUMAN_PEP1", 2, "sp|X|TEST_HUMAN/"),  # 正例候选
            _make_psm("HUMAN_PEP2", 2, "sp|X|TEST_HUMAN/"),  # 仅 pfind 有 → 不入正例
            _make_psm("ECOLI_PEP1", 2, "sp|X|TEST_ECOLI/"),  # 负例（pfind 找到）
        ],
        "diann": [
            _make_psm("HUMAN_PEP1", 2, "sp|X|TEST_HUMAN/"),  # 正例（交集）
            _make_psm("ECOLI_PEP2", 2, "sp|X|TEST_ECOLI/"),  # 负例（diann 找到）
        ],
    }
    result = extract_n_engines_from_psms(
        engine_psms, engine_order=["pfind", "diann"],
        positive_marker="HUMAN")

    positives = [p for p in result if p._label_type == "positive"]
    negatives = [p for p in result if p._label_type == "negative"]

    pos_seqs = sorted([p._sequence for p in positives])
    neg_seqs = sorted([p._sequence for p in negatives])

    assert pos_seqs == ["HUMAN_PEP1"]  # 仅交集且含 HUMAN
    assert neg_seqs == ["ECOLI_PEP1", "ECOLI_PEP2"]  # 并集且不含 HUMAN


def test_extract_three_engines_intersection():
    """N=3 引擎，正例必须三个引擎都识别。"""
    from tools.extract_common import extract_n_engines_from_psms

    engine_psms = {
        "pfind": [
            _make_psm("ALL_THREE", 2, "sp|X|TEST_HUMAN/"),
            _make_psm("ONLY_PFIND", 2, "sp|X|TEST_HUMAN/"),
        ],
        "diann": [
            _make_psm("ALL_THREE", 2, "sp|X|TEST_HUMAN/"),
            _make_psm("PFIND_DIANN", 2, "sp|X|TEST_HUMAN/"),
        ],
        "alphadia": [
            _make_psm("ALL_THREE", 2, "sp|X|TEST_HUMAN/"),
        ],
    }
    result = extract_n_engines_from_psms(
        engine_psms, engine_order=["pfind", "diann", "alphadia"],
        positive_marker="HUMAN")
    positives = [p for p in result if p._label_type == "positive"]
    pos_seqs = [p._sequence for p in positives]
    assert pos_seqs == ["ALL_THREE"]


def test_extract_label_type_attached():
    """所有输出 PSM 都应有明确的 label_type（在 marker 模式下）。"""
    from tools.extract_common import extract_n_engines_from_psms

    engine_psms = {
        "pfind": [_make_psm("AAA", 2, "sp|X|TEST_HUMAN/")],
        "diann": [_make_psm("AAA", 2, "sp|X|TEST_HUMAN/")],
    }
    result = extract_n_engines_from_psms(
        engine_psms, engine_order=["pfind", "diann"],
        positive_marker="HUMAN")
    assert all(p._label_type in ("positive", "negative") for p in result)
```

- [ ] **Step 3: 运行测试，确认失败**

```bash
pytest tests/test_extract_common.py -v
```

Expected: FAIL（`tools.extract_common` 模块或 `extract_n_engines_from_psms` 函数不存在）。

- [ ] **Step 4: 实现 extract_common.py 的核心算法**

Create `tools/extract_common.py`：

```python
"""extract_common：通用 N 引擎交并集工具。

从多个搜索引擎的结果中构造正负例数据集：
- 正例：所有引擎都识别为目标物种的 PSM（key 交集 + species marker 匹配）
- 负例：任一引擎识别为非目标物种的 PSM（key 并集 + species marker 不匹配）

支持的引擎：pfind, diann, alphadia
"""
import argparse
import configparser
import json
import logging
import os
import sys
from typing import Optional

from rich.logging import RichHandler

# 把项目根目录加入 sys.path 以便 import ms2-met 模块
_PROJECT_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
if _PROJECT_ROOT not in sys.path:
    sys.path.insert(0, _PROJECT_ROOT)

from spectrum.light_result import LightResult
from spectrum.psm_info import PSMInfo


SUPPORTED_ENGINES = {"pfind", "diann", "alphadia"}


def load_engine_psms(engine_name: str, config: configparser.ConfigParser) -> list:
    """根据引擎名加载对应 PSM 列表。"""
    section = f"engine.{engine_name}"
    if section not in config:
        raise ValueError(f"配置中缺少 [{section}] 段")
    path = config[section].get("path")
    if not path:
        raise ValueError(f"[{section}] 缺少 path 配置")

    lr = LightResult()
    if engine_name == "pfind":
        qvalue = config[section].getfloat("qvalue_threshold", fallback=0.01)
        lr._load_from_pfind_input(path, qvalue_threshold=qvalue)
    elif engine_name == "diann":
        lr._load_from_dia_nn_input(path)
    elif engine_name == "alphadia":
        lr._load_from_alphadia_input(path)
    else:
        raise ValueError(
            f"不支持的引擎: {engine_name}（支持 {SUPPORTED_ENGINES}）")

    return lr.psm_info


def extract_n_engines_from_psms(
    engine_psms: dict,
    engine_order: list,
    positive_marker: Optional[str] = None,
) -> list:
    """从多引擎的 PSM 列表构造正负例数据集（核心算法）。

    Args:
        engine_psms: dict[engine_name -> list[PSMInfo]]
        engine_order: list[engine_name]，决定权威 PSM 来源顺序
        positive_marker: 物种 marker 字符串；为 None 则仅取交集，不打 label

    Returns:
        list[PSMInfo]，每条 PSM 的 label_type 字段已被设置（或保持 None）
    """
    # 1. 构建每个引擎的 key 集合
    key_sets = {name: {p.get_key() for p in psms}
                for name, psms in engine_psms.items()}

    intersection_keys = set.intersection(*key_sets.values()) if key_sets else set()
    union_keys = set.union(*key_sets.values()) if key_sets else set()

    # 2. 构建 key → PSM 映射（按引擎优先级）
    key_to_psm = {}
    for engine_name in engine_order:
        for psm in engine_psms.get(engine_name, []):
            key = psm.get_key()
            if key not in key_to_psm:
                key_to_psm[key] = psm

    result = []

    if not positive_marker:
        # 无 marker：仅交集，不打 label
        for key in intersection_keys:
            psm = key_to_psm.get(key)
            if psm is not None:
                psm._label_type = None
                result.append(psm)
        logging.info(
            f"无 marker 模式：intersection size={len(result)}")
        return result

    # 有 marker：正例（交集 + marker）+ 负例（并集 + 非 marker）
    pos_count = 0
    neg_count = 0
    for key in intersection_keys:
        psm = key_to_psm.get(key)
        if psm is None:
            continue
        if positive_marker in psm._protein_names:
            psm._label_type = "positive"
            result.append(psm)
            pos_count += 1

    for key in union_keys:
        psm = key_to_psm.get(key)
        if psm is None:
            continue
        if positive_marker not in psm._protein_names:
            # 避免重复：交集中也可能有非 marker 的 PSM（如果某个引擎错把非 HUMAN 标成共有）
            # 实际上不应在 intersection_keys 中，但保险起见检查
            if psm._label_type == "positive":
                continue
            psm._label_type = "negative"
            result.append(psm)
            neg_count += 1

    logging.info(
        f"marker='{positive_marker}': positive={pos_count}, negative={neg_count}, "
        f"total={len(result)}"
    )
    return result


def extract_n_engines(config: configparser.ConfigParser) -> list:
    """根据 config 加载各引擎并构造正负例。"""
    engines_str = config["extract"]["engines"]
    engine_order = [e.strip() for e in engines_str.split(",") if e.strip()]

    invalid = [e for e in engine_order if e not in SUPPORTED_ENGINES]
    if invalid:
        raise ValueError(
            f"未知引擎: {invalid}（支持 {SUPPORTED_ENGINES}）")

    positive_marker = config["extract"].get("positive_species_marker", "").strip()
    if not positive_marker:
        positive_marker = None

    # 加载每个引擎
    engine_psms = {}
    for name in engine_order:
        logging.info(f"加载引擎: {name}")
        engine_psms[name] = load_engine_psms(name, config)
        logging.info(f"  → {name} 共 {len(engine_psms[name])} 条 PSM")

    return extract_n_engines_from_psms(
        engine_psms, engine_order, positive_marker)


def write_psms_to_json(psms: list, output_path: str):
    """把 PSMInfo 列表序列化到 JSON。"""
    output_dir = os.path.dirname(output_path)
    if output_dir and not os.path.exists(output_dir):
        os.makedirs(output_dir, exist_ok=True)
    with open(output_path, "w") as f:
        json.dump([p.to_dict() for p in psms], f, indent=2)
    logging.info(f"已写入 {len(psms)} 条 PSM 到 {output_path}")


def main():
    parser = argparse.ArgumentParser(
        description="extract_common: 通用 N 引擎交并集数据集构造工具")
    parser.add_argument(
        "--configpath", default="./extract_common_config.ini",
        help="配置文件路径")
    parser.add_argument(
        "--logpath", default="./extract_common.log", help="日志文件路径")
    args = parser.parse_args()

    file_handler = logging.FileHandler(args.logpath, encoding="utf-8")
    file_handler.setFormatter(logging.Formatter(
        "%(asctime)s | %(levelname)s | %(name)s | %(message)s"))
    logging.basicConfig(
        level=logging.INFO,
        handlers=[RichHandler(), file_handler],
    )

    config = configparser.ConfigParser()
    config.read(args.configpath)

    if "extract" not in config:
        logging.error(f"配置文件 {args.configpath} 缺少 [extract] 段")
        sys.exit(1)

    psms = extract_n_engines(config)
    result_file = config["extract"]["result_file"]
    write_psms_to_json(psms, result_file)


if __name__ == "__main__":
    main()
```

- [ ] **Step 5: 运行测试，确认通过**

```bash
pytest tests/test_extract_common.py -v
```

Expected: 4 个测试全部 PASS。

- [ ] **Step 6: 提交**

```bash
git add tools/ extract_common_config.ini.example tests/test_extract_common.py
git commit -m "feat: tools/extract_common.py 通用 N 引擎交并集工具"
```

---

## Task 9: extract_common 端到端集成验证

**Files:**
- Create: `tests/test_extract_common_integration.py`
- Create: `tests/fixtures/sample_extract_config.ini`

- [ ] **Step 1: 创建集成测试配置**

Create `tests/fixtures/sample_extract_config.ini`：

```ini
[extract]
engines = pfind
positive_species_marker = HUMAN
result_file = /tmp/test_extract_common_output.json

[engine.pfind]
path = tests/fixtures/sample_pfind.qry.res
qvalue_threshold = 0.01
```

注意 result_file 路径用 /tmp（测试时会覆盖）。

- [ ] **Step 2: 写集成测试**

Create `tests/test_extract_common_integration.py`：

```python
"""extract_common 工具端到端集成测试。"""
import configparser
import json
import os
import tempfile
import pytest

from spectrum.psm_info import PSMInfo
from tools.extract_common import (
    extract_n_engines, write_psms_to_json, load_engine_psms,
)


def test_load_engine_pfind(tmp_path, sample_pfind_file):
    """load_engine_psms 应能加载 pfind 引擎。"""
    config = configparser.ConfigParser()
    config["engine.pfind"] = {
        "path": sample_pfind_file,
        "qvalue_threshold": "0.01",
    }
    psms = load_engine_psms("pfind", config)
    assert len(psms) == 3


def test_extract_single_engine_with_marker(tmp_path, sample_pfind_file):
    """单引擎 + marker 模式：自交集 = 自己，按 marker 分正负。"""
    config = configparser.ConfigParser()
    config["extract"] = {
        "engines": "pfind",
        "positive_species_marker": "HUMAN",
        "result_file": str(tmp_path / "out.json"),
    }
    config["engine.pfind"] = {
        "path": sample_pfind_file,
        "qvalue_threshold": "0.01",
    }
    psms = extract_n_engines(config)
    # 3 个保留：2 个 HUMAN（正例）+ 1 个 ECOLI（负例）
    pos = [p for p in psms if p._label_type == "positive"]
    neg = [p for p in psms if p._label_type == "negative"]
    assert len(pos) == 2
    assert len(neg) == 1


def test_extract_to_json_roundtrip(tmp_path, sample_pfind_file):
    """写出 JSON 后再读回，PSM 数应一致且 label_type 保留。"""
    config = configparser.ConfigParser()
    output = str(tmp_path / "out.json")
    config["extract"] = {
        "engines": "pfind",
        "positive_species_marker": "HUMAN",
        "result_file": output,
    }
    config["engine.pfind"] = {
        "path": sample_pfind_file,
        "qvalue_threshold": "0.01",
    }
    psms = extract_n_engines(config)
    write_psms_to_json(psms, output)

    with open(output) as f:
        data = json.load(f)

    reconstructed = [PSMInfo.from_dict(d) for d in data]
    assert len(reconstructed) == len(psms)
    for p in reconstructed:
        assert p._label_type in ("positive", "negative")
        assert p._q_value is not None  # pfind 提供
```

- [ ] **Step 3: 运行测试**

```bash
pytest tests/test_extract_common_integration.py -v
```

Expected: 全部 PASS。

- [ ] **Step 4: 手动运行 CLI 工具做 smoke test**

```bash
mkdir -p /tmp/extract_test
cat > /tmp/extract_test/config.ini <<'EOF'
[extract]
engines = pfind
positive_species_marker = HUMAN
result_file = /tmp/extract_test/output.json

[engine.pfind]
path = tests/fixtures/sample_pfind.qry.res
qvalue_threshold = 0.01
EOF

python3 tools/extract_common.py --configpath /tmp/extract_test/config.ini --logpath /tmp/extract_test/run.log

cat /tmp/extract_test/output.json | python3 -m json.tool | head -30
echo "---"
echo "PSM 数：$(python3 -c "import json; print(len(json.load(open('/tmp/extract_test/output.json'))))")"
```

Expected:
- 输出 JSON 文件
- 含 3 条 PSM（2 个正例 + 1 个负例）
- 每条都有 label_type 字段

- [ ] **Step 5: 验证 ms2-met 主流程能读取 extract_common 输出**

```bash
python3 -c "
import json
from spectrum.psm_info import PSMInfo
data = json.load(open('/tmp/extract_test/output.json'))
psms = [PSMInfo.from_dict(d) for d in data]
print(f'重建 {len(psms)} 条 PSM')
for p in psms:
    print(f'  {p._sequence} label={p._label_type} q={p._q_value}')
"
```

Expected: 重建成功，无报错，label_type 字段正确。

- [ ] **Step 6: 清理临时文件**

```bash
rm -rf /tmp/extract_test
```

- [ ] **Step 7: 提交**

```bash
git add tests/test_extract_common_integration.py tests/fixtures/sample_extract_config.ini
git commit -m "test: extract_common 端到端集成测试"
```

---

## Task 10: 老 extract_com 项目添加 README 占位

**Files:**
- Create: `../extract_com/README.md`

- [ ] **Step 1: 创建迁移说明 README**

Create `/home/verden/pfind/2025-fall/code/extract_com/README.md`：

```markdown
# extract_com（已迁移 / DEPRECATED）

> **本项目已迁移到 [`ms2-met/tools/extract_common.py`](../ms2-met/tools/extract_common.py)。**
> 本目录仅保留 git history 备查，**不再维护**。

## 迁移信息

- **迁移日期**：2026-05-18
- **新位置**：`ms2-met/tools/extract_common.py`
- **新增能力**：
  - 通用 N 引擎模式（不再限定 DIANN + AlphaDIA）
  - 原生 pfind 支持
  - 与 ms2-met 共享 PSMInfo / LightResult / 修饰解析逻辑，无代码重复
  - 输出 JSON 含 `label_type` / `q_value` / `score` 字段

## 如何使用新工具

```bash
cd ../ms2-met
python tools/extract_common.py --configpath extract_common_config.ini
```

配置示例见 `ms2-met/extract_common_config.ini.example`。

## 老配置 → 新配置映射

```ini
# 老 extract_com config.ini
[general]
diann_path = ../hela_report.parquet
alphadia_path = ../precursors.parquet
result_file = hela.json
positive_species_marker = HUMAN
```

变为：

```ini
# 新 ms2-met tools/extract_common.py 配置
[extract]
engines = diann, alphadia
positive_species_marker = HUMAN
result_file = ./hela.json

[engine.diann]
path = ../hela_report.parquet

[engine.alphadia]
path = ../precursors.parquet
```

## 历史

详细变更可查 git log：

```bash
git log --all -- extract_comm.py load_data.py psm_info.py
```
```

- [ ] **Step 2: 提交（在 extract_com 仓库中）**

```bash
cd /home/verden/pfind/2025-fall/code/extract_com
git add README.md
git commit -m "docs: 迁移到 ms2-met/tools/extract_common.py，本项目 deprecated

Co-authored-by: Copilot <223556219+Copilot@users.noreply.github.com>"
```

---

## Task 11: 更新 PROJECT_INFO.md 记录新能力

**Files:**
- Modify: `PROJECT_INFO.md`

- [ ] **Step 1: 修改 PROJECT_INFO.md**

查找现有 `search_engine_type` 相关段落并更新——把表格扩到 3：

定位章节：在 PROJECT_INFO.md 中找到说明 search_engine_type 的部分。如果找不到对应的小节，在合适位置（如"3.1 配置文件"段落附近）添加：

```markdown
### 3.1 搜索引擎支持

ms2-met 支持以下搜索引擎结果格式（通过 `search_engine_type` 配置）：

| search_engine_type | 引擎 | 输入格式 | 备注 |
|--------------------|------|---------|------|
| 0 | 自定义 JSON | `.json`（来自 `tools/extract_common.py`） | 通过 N 引擎交并集构造 |
| 1 | DIA-NN | `.parquet` | 单文件 |
| 2 | AlphaDIA | `.parquet` | 单文件 |
| 3 | pfind | `.qry.res` 文件或目录 | 含 FDR + REV_ decoy 过滤 |

pfind 相关配置项：
- `pfind_qvalue_threshold`（默认 0.01）：FDR 阈值

### 3.2 数据集构造工具

`tools/extract_common.py` 是通用 N 引擎交并集工具：

- 加载多个搜索引擎结果
- 正例 = 所有引擎都识别为目标物种的 PSM（key 交集）
- 负例 = 任一引擎识别为非目标物种的 PSM（key 并集）
- 输出 JSON（兼容 search_engine_type=0）

使用：`python tools/extract_common.py --configpath <config>`
```

如已存在但表格未含 pfind，在表格中添加行 `3 | pfind | ...`，其他段落对应扩充。

- [ ] **Step 2: 提交**

```bash
git add PROJECT_INFO.md
git commit -m "docs(project-info): 更新搜索引擎支持表与 extract_common 工具说明"
```

---

## Task 12: 全面回归与最终验收

**Files:**（无新增）

- [ ] **Step 1: 运行所有测试**

```bash
source /opt/miniconda3/etc/profile.d/conda.sh && conda activate jianyan
cd /home/verden/pfind/2025-fall/code/ms2-met
pytest tests/ -v
```

Expected: 全部 PASS，无 warning（除 OpenSSL legacy 警告，与本任务无关）。

- [ ] **Step 2: 手动验证 ms2-met 主流程加载 pfind 数据**

```bash
# 用 fixture 测试，避免真实大文件
cat > /tmp/test_pfind_main_config.ini <<'EOF'
[input]
raw_num = 0
light_result_file = tests/fixtures/sample_pfind.qry.res
search_engine_type = 3
pfind_qvalue_threshold = 0.01

[general]
feature_type = 0
work_directory = /tmp/ms2met_test_workspace
mass_tol_ppm = 10
xic_cycle_window = 6
result_file = result.csv
EOF

# 仅验证 LightResultManager 加载部分（不跑完整 PairFlow，避免 raw 文件依赖）
python3 -c "
import configparser
from manager.light_result_manager import LightResultManager
from constant.keys import ConfigKeys

config = configparser.ConfigParser()
config.read('/tmp/test_pfind_main_config.ini')

mgr = LightResultManager(config=config, path=None, load_from_file=False)
lr = mgr.get_light_result_object(
    config[ConfigKeys.INPUT][ConfigKeys.LIGHT_RESULT_PATH])
print(f'通过 manager 加载 {len(lr.psm_info)} 条 PSM')
for p in lr.psm_info[:3]:
    print(f'  {p._sequence} z={p._charge} q={p._q_value}')
"
```

Expected: 加载 3 条 PSM，无报错。

- [ ] **Step 3: 验证老 hela.json（如果存在）仍能加载**

```bash
# 在 ms2-met 目录或父目录搜索现有 hela.json 测试兼容性
if [ -f "../hela.json" ]; then
    python3 -c "
import json
from spectrum.psm_info import PSMInfo
data = json.load(open('../hela.json'))
psms = [PSMInfo.from_dict(d) for d in data]
print(f'老 hela.json 加载 {len(psms)} 条 PSM 成功，向后兼容验证通过')
for p in psms[:2]:
    print(f'  {p._sequence} label_type={p._label_type} q={p._q_value} score={p._score}')
"
else
    echo "找不到 ../hela.json，跳过老文件兼容测试"
fi
```

Expected: 如存在，加载成功，新字段为 None。

- [ ] **Step 4: 清理临时文件**

```bash
rm -f /tmp/test_pfind_main_config.ini
rm -rf /tmp/ms2met_test_workspace
```

- [ ] **Step 5: 最终提交（如本任务有任何小修补）**

```bash
git status
# 如有未提交的小改动，最后整理一次
```

- [ ] **Step 6: 总结提交日志**

```bash
git log --oneline -15
```

Expected: 看到本计划的 11 次提交，从 fixture 开始到 PROJECT_INFO 结束。

---

## 验收清单（实施完成时核对）

- [ ] PSMInfo 加 q_value/score/label_type 字段（向后兼容老 JSON）
- [ ] pfind 修饰名称解析（硬编码 + UniMod 兑底）
- [ ] pfind MH+ → m/z 转换正确
- [ ] pfind 加载支持单文件与目录扫描
- [ ] FDR 过滤（QValue ≤ qvalue_threshold）生效
- [ ] decoy 过滤（REV_ 前缀）生效
- [ ] LightResultManager 分发 search_engine_type=3 → pfind
- [ ] tools/extract_common.py 通用 N 引擎交并集
- [ ] label_type = positive/negative 在有 marker 时被打上
- [ ] 无 marker 时仅取交集，label_type = None
- [ ] config.ini 含 pfind 配置示例与注释
- [ ] extract_common_config.ini.example 提供示例
- [ ] 老 extract_com 项目添加 README deprecation 说明
- [ ] PROJECT_INFO.md 反映新能力
- [ ] 所有测试 PASS

## 完成后的能力

完成后用户可以：

1. **跑 pfind 单引擎**：
```bash
# config.ini 设 search_engine_type = 3, light_result_file = <pfind 目录>
python main.py --configpath config.ini
```

2. **构造 pfind + DIANN 数据集**：
```bash
python tools/extract_common.py --configpath extract_common_config.ini
# 然后用 search_engine_type = 0 跑特征提取
```

3. **未来加新引擎**只需：
- 在 `LightResult` 添加新 loader 方法
- 在 `LightResultManager` 加分发分支
- 在 `tools/extract_common.py` 的 `load_engine_psms` 添加引擎名

---

## 未决问题（提醒）

实施完成前，请用户验证：

1. **pfind PredRT 语义**：当前公式 `rt = PredRT + DeltaRT(Min)`。若 DeltaRT 方向相反，需改为 `rt = PredRT - DeltaRT(Min)`。
2. **pfind QValue 是 q-value 还是 PEP**：当前按 q-value 处理（阈值 0.01）。如果是 PEP，含义会不同。

二者均可通过对比已知正确 PSM 的 RT 与 QValue 分布快速验证。建议在 Task 7 的 smoke test 阶段同时验证。
