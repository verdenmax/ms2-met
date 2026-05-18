# pfind 支持与 extract_com 融合——设计文档

> 编写日期：2026-05-18
> 项目：ms2-met
> 文档版本：0.1（初稿）
> 状态：待评审

---

## 一、研究背景与目标

### 1.1 当前状态

ms2-met 当前支持三种搜索引擎结果格式：

| search_engine_type | 引擎 | 加载入口 |
|---|---|---|
| 0 | 自定义 JSON（来自 extract_com） | `_load_from_pkl` |
| 1 | DIA-NN parquet | `_load_from_dia_nn_input` |
| 2 | AlphaDIA parquet | `_load_from_alphadia_input` |

`extract_com/`（位于项目同级目录）是一个独立项目，负责从 DIA-NN 和 AlphaDIA 结果构造"交集正例 + 并集负例"的数据集，输出 JSON 供 ms2-met 消费。但 extract_com 与 ms2-met 之间存在大量重复代码（`psm_info.py`、`util.py`、`load_data.py` 几乎完全相同）。

### 1.2 目标

**目标 A：在 ms2-met 中原生支持 pfind 搜索结果**

pfind 是另一类主流蛋白质组学搜索引擎，输出格式为 TSV `.qry.res`。AlphaDIA 在你们数据上表现差，需用 pfind 替代以提高数据集质量。

**目标 B：将 extract_com 融合为 ms2-met 内部工具**

消除代码重复，让数据集构造工具与特征提取工具共享同一份 PSMInfo / LightResult / 修饰解析逻辑。

### 1.3 业务价值

1. **解锁 pfind + DIA-NN 数据集**——取代 AlphaDIA 不可靠的结果
2. **代码统一**——单一来源的 PSMInfo / Loader，降低维护负担
3. **通用 N 引擎能力**——未来引入新引擎无需改 extract 逻辑

---

## 二、范围与非范围

### 2.1 在范围

- 实现 pfind `.qry.res` 文件加载器（含 FDR 过滤、decoy 过滤、修饰解析、m/z 计算）
- 在 ms2-met 创建 `tools/extract_common.py`，复用项目内部代码
- 通用 N 引擎交并集逻辑（任意数量、任意引擎组合）
- 正负例 `label_type` 字段写入 PSMInfo
- 老 `extract_com/` 项目保留为 README 占位（迁移说明）

### 2.2 不在范围

- 重写 DIA-NN 或 AlphaDIA loader（保持现状）
- 移除 AlphaDIA 支持（保留代码，不主动使用）
- 负例细分（`negative_no_signal` vs `negative_interference`）的判定逻辑实现——该逻辑必须基于特征提取后的结果（如 `q1a_total_count`），属于下游分析；本 spec 仅保留字段空间
- 不平衡类别在训练分类器时的处理（属于分类器训练阶段，非本 spec 范围）
- pfind PredRT 语义验证——本 spec 暂定"实测 RT = PredRT + DeltaRT"，需用户后续 check

---

## 三、关键技术决策

### 3.1 架构：tools/ 子目录独立 CLI

新建 `tools/extract_common.py`，作为与 `main.py` 平级的独立 CLI 入口：

```
ms2-met/
├── main.py                              ← 特征提取（不变）
├── config.ini                           ← 扩展 pfind/extract 字段
├── tools/                                ← 新建
│   ├── __init__.py
│   └── extract_common.py                ← 通用 N 引擎数据集构造工具
├── spectrum/
│   ├── psm_info.py                      ← 扩展 q_value, score, label_type
│   ├── light_result.py                  ← 添加 _load_from_pfind_input
│   └── pfind_parser.py                  ← 新增：pfind 专用解析
├── manager/
│   └── light_result_manager.py          ← 分发支持 pfind = 3
└── constant/
    └── keys.py                          ← 新增 pfind 与 extract 常量
```

`tools/extract_common.py` 直接 import ms2-met 内部模块（`spectrum.psm_info`、`spectrum.light_result`、`manager.light_result_manager`），不存在代码重复。

### 3.2 pfind 加载器

#### 3.2.1 输入格式支持

- **目录扫描**（主用法）：`light_result_file = ./pfind-dia/2th/` → 扫描该目录下所有 `*.qry.res`
- **单文件**（兼容）：`light_result_file = ./xxx.qry.res` → 仅加载该文件

判断逻辑：`os.path.isdir(path)` 是目录则扫描，否则单文件。

#### 3.2.2 字段映射

| pfind 列 | PSMInfo 字段 | 转换 |
|---|---|---|
| PeptideSequence | sequence | 直接 |
| Modifications | modify | `parse_pfind_modify` 转 [(0-based pos, unimod_id), ...] |
| Charge | charge | 直接 |
| PredRT + DeltaRT(Min) | rt | **暂定** rt = PredRT + DeltaRT |
| MH+ + Charge | precursor_mz | `mhp_to_mz(MH+, charge)` |
| Proteins | protein_names | 直接保留全字符串 |
| QValue | q_value | 直接 |
| FinalScore | score | 直接 |
| 文件名（去 `.qry.res`） | raw_title | 自动提取 |

#### 3.2.3 过滤逻辑

加载时依次应用三个过滤：

1. **FDR 过滤**：`QValue > pfind_qvalue_threshold`（默认 0.01）→ 丢弃
2. **Decoy 过滤**：`Proteins.startswith("REV_")`（前缀大小写敏感）→ 丢弃
3. **合法性过滤**：`PSMInfo.valid()`（序列含 X 等）→ 丢弃

按此顺序处理（FDR 先，最便宜）。每个过滤步骤记录被剔除的行数到日志。

#### 3.2.4 m/z 转换

pfind 给出 MH+（即 1+ 离子质量 = 中性质量 + 1 × proton_mass）。实际 m/z 需要根据 Charge 计算：

```python
PROTON_MASS = 1.00727646677

def mhp_to_mz(mhp: float, charge: int) -> float:
    """MH+ → 带 charge 的 m/z"""
    neutral_mass = mhp - PROTON_MASS
    return (neutral_mass + charge * PROTON_MASS) / charge
```

#### 3.2.5 RT 转换

暂定：

```python
rt = pred_rt + delta_rt  # PredRT + DeltaRT(Min)
```

**未决问题**：DeltaRT(Min) 的符号约定（实测-预测 还是 预测-实测）未确认。spec 落实时需要用户验证。如确认为后者，公式改为 `rt = pred_rt - delta_rt`。

### 3.3 修饰解析

pfind 修饰格式：`"3,Carbamidomethyl[C];10,Carbamidomethyl[C];"`

- 位置：1-based，需转 0-based
- 名称：`Carbamidomethyl[C]`、`Phospho[S]` 等

解析策略（硬编码字典 + unimod.xml 兜底）：

```python
PFIND_MOD_TO_UNIMOD = {
    "Carbamidomethyl[C]": 4,
    "Oxidation[M]": 35,
    "Phospho[S]": 21, "Phospho[T]": 21, "Phospho[Y]": 21,
    "Acetyl[K]": 1, "Acetyl[ProteinN-term]": 1,
    "Methyl[K]": 34, "Methyl[R]": 34,
    "Dimethyl[K]": 36, "Dimethyl[R]": 36,
    "Trimethyl[K]": 37,
    "Deamidated[N]": 7, "Deamidated[Q]": 7,
    "Pyro-carbamidomethyl[AnyN-term]": 26,
    "Gln->pyro-Glu[AnyN-termQ]": 28,
    "Glu->pyro-Glu[AnyN-termE]": 27,
    # ...持续扩充
}

@lru_cache(maxsize=1024)
def resolve_pfind_mod_name(name: str) -> int | None:
    # 1. 先查硬编码字典
    if name in PFIND_MOD_TO_UNIMOD:
        return PFIND_MOD_TO_UNIMOD[name]
    # 2. 兜底：unimod.xml 按基础名查询
    base_name = name.split("[")[0] if "[" in name else name
    try:
        record = unimods.by_title(base_name)
        return record["record_id"]
    except Exception:
        return None  # 未知修饰
```

未知修饰 → log warning + 跳过该 PSM。

### 3.4 PSMInfo 扩展

```python
class PSMInfo:
    def __init__(self,
                 sequence: str, charge: int,
                 modify: list[tuple[int, int]],
                 rt: float, precursor_mz: float,
                 raw_title: str, protein_names: str,
                 q_value: float | None = None,      # 新增
                 score: float | None = None,        # 新增
                 label_type: str | None = None):    # 新增
        ...

    def to_dict(self):
        # 只在非 None 时写入新字段，避免污染老格式
        d = {...原有...}
        if self._q_value is not None: d["q_value"] = self._q_value
        if self._score is not None: d["score"] = self._score
        if self._label_type is not None: d["label_type"] = self._label_type
        return d

    @classmethod
    def from_dict(cls, data):
        return cls(
            ...原有...,
            q_value=data.get("q_value"),
            score=data.get("score"),
            label_type=data.get("label_type"),
        )

    def get_key(self):  # 不变，新字段不参与 key
        ...
```

**`label_type` 取值**：
- `"positive"`：通过 extract_common 正例规则的 PSM
- `"negative"`：通过 extract_common 负例规则的 PSM
- `None`：单引擎模式或未经 extract_common 处理（如直接从 pfind .qry.res 单引擎跑）

下游可基于特征值进一步细化（`negative_no_signal` / `negative_interference`），但本 spec 不实现该细化逻辑。

### 3.5 extract_common 通用 N 引擎

#### 3.5.1 配置格式

```ini
[extract]
engines = pfind, diann               ; 任意数量、任意顺序
positive_species_marker = HUMAN      ; 可选；为空则仅做交集，不分正负例
positive_mode = intersection         ; intersection（所有引擎都识别为 marker）
negative_mode = union                ; union（任一引擎识别为非 marker）
result_file = ./datasets/hela-2da-pfind-diann.json

[engine.pfind]
path = ./pfind-dia/2th/
qvalue_threshold = 0.01

[engine.diann]
path = ./hela-mix-2da_report.parquet
```

#### 3.5.2 核心算法

```python
def extract_n_engines(config):
    engine_names = parse_engines(config)
    positive_marker = config.get("extract", "positive_species_marker", None)

    # 1. 加载每个引擎（复用 LightResult 各 loader）
    engine_psms = {}
    for name in engine_names:
        engine_psms[name] = load_engine(name, config)

    # 2. 构建 (sequence, charge, modify) key 集合
    key_sets = {name: {p.get_key() for p in psms}
                for name, psms in engine_psms.items()}

    intersection_keys = set.intersection(*key_sets.values())
    union_keys = set.union(*key_sets.values())

    # 3. 选择"权威源"——所有引擎中第一个含该 key 的，作为 PSM 详情来源
    def find_psm(key):
        for name in engine_names:
            for psm in engine_psms[name]:
                if psm.get_key() == key:
                    return psm
        return None

    result = []
    if positive_marker:
        # 正例：交集 + 物种匹配
        for key in intersection_keys:
            psm = find_psm(key)
            if positive_marker in psm._protein_names:
                psm._label_type = "positive"
                result.append(psm)

        # 负例：并集 + 物种不匹配
        for key in union_keys:
            psm = find_psm(key)
            if positive_marker not in psm._protein_names:
                psm._label_type = "negative"
                result.append(psm)
    else:
        # 无物种标记 → 简单交集（保持 extract_com 原始行为）
        for key in intersection_keys:
            psm = find_psm(key)
            psm._label_type = None
            result.append(psm)

    return result
```

#### 3.5.3 复杂度

- N 引擎，M 个 PSM/引擎：构建 key 集合 O(N×M)；交并集 O(N×M)；权威 PSM 查找 O(N×M)
- 总复杂度 O(N×M)，N≤3、M ≤ 10⁵ → 实际 ~ 10⁵ 操作，秒级完成

### 3.6 配置文件示例

```ini
[input]
raw_num = 1
raw_path_1 = ./xxx.mzML
light_result_file = ./pfind-dia/2th/                ; pfind 目录
search_engine_type = 3                              ; pfind
pfind_qvalue_threshold = 0.01

[general]
feature_type = 0
work_directory = ./workspace
mass_tol_ppm = 10
xic_cycle_window = 6
result_file = result.csv
```

```ini
[extract]
engines = pfind, diann
positive_species_marker = HUMAN
result_file = ./datasets/hela-2da.json

[engine.pfind]
path = ./pfind-dia/2th/
qvalue_threshold = 0.01

[engine.diann]
path = ./hela-mix-2da_report.parquet
```

---

## 四、正负例标签策略——理论依据

### 4.1 为什么保留陷阱库方法

陷阱库（E.coli / yeast 入数据库）方法在本项目中**物理逻辑严密**：

- 样本是纯 HeLa（HUMAN），E.coli/yeast 物理上不存在 → 任何识别为 E.coli/yeast 的 PSM **机械地为假阳性**
- 完全**独立于 TDA**——decoy 与 entrapment 是两套独立的 ground truth 源
- 数据已有的 3.3:1 正负比来自这套方案

### 4.2 关键洞察：5Da 下陷阱负例自然包含干扰场景

PLAN.md 数据揭示：

| 窗口 | 陷阱负例特征 |
|---|---|
| 2Da | 70% 陷阱碎片完全无信号（mechanically wrong, no signal） |
| 5Da | **99% 陷阱碎片能匹配到信号**（来自共流出的真实 HUMAN 肽段） |

**5Da 下陷阱负例正是"被真实干扰肽段污染的错误 PSM"**——这恰好是 §3 共窗口失败模式的物理对应。陷阱库方法在 5Da 下天然地训练分类器识别这种干扰，不是"用错了方法"，而是"恰好对路"。

### 4.3 决定的策略：保留陷阱库 + 三层修订

| 修订 | 内容 |
|---|---|
| 修订 1：引擎组合 | 正例 = HUMAN ∩ pfind ∩ diann（去掉 alphadia） |
| 修订 1：引擎组合 | 负例 = entrapment ∩ (pfind ∪ diann)（并集，提供更多负例） |
| 修订 2：负例分层 | extract_common 仅输出 positive/negative；下游基于 q1a_total_count 等细化 negative_no_signal / negative_interference（本 spec 仅占位） |
| 修订 3：decoy 独立 | TDA decoy（pfind 已有 REV_）**不进训练**；仅作外部 sanity check |

### 4.4 不采用的方案

| 不采用 | 原因 |
|---|---|
| decoy 作训练负例 | 让分类器学 TDA，违背工具与 TDA 独立的核心论点 |
| 正例放宽到任一引擎 | 引入引擎 FP，污染正例集合 |
| 把 alphadia 加回 | 用户已验证不可靠 |

### 4.5 不平衡处理

3.3:1 正负比偏斜在分类器训练阶段处理（class weight 或 sample weight），**不是标签构造问题**。本 spec 不涉及。

---

## 五、数据流总览

### 5.1 单引擎 pfind 模式

```
pfind .qry.res / 目录
  ↓ [LightResult._load_from_pfind_input]
  ↓ [QValue ≤ 0.01 过滤]
  ↓ [REV_ decoy 过滤]
  ↓ [PSMInfo.valid 过滤]
LightResult.psm_info  (label_type = None)
  ↓ [PairFlow]
特征 CSV
```

### 5.2 多引擎 extract 模式

```
pfind 目录 ──┐
            ├→ [tools/extract_common.py]
DIA-NN ─────┤   ↓ [N 引擎交并集 + 物种标记]
            │   ↓ [打 label_type]
alphadia ───┘ JSON  (含 label_type)
（保留支持） ↓ [LightResult._load_from_pkl]
              LightResult.psm_info
                ↓ [PairFlow]
              特征 CSV  (含 label_type 列)
```

---

## 六、测试策略

### 6.1 单元测试

| 测试模块 | 测试点 |
|---|---|
| `pfind_parser.parse_pfind_modify` | 空字符串、单修饰、多修饰、未知修饰、1-based → 0-based 转换 |
| `pfind_parser.mhp_to_mz` | z=1,2,3,4 各电荷态 |
| `pfind_parser.resolve_pfind_mod_name` | 硬编码命中、unimod.xml 兜底、未知名称 |
| `light_result._load_from_pfind_input` | FDR 过滤、REV_ 过滤、合法性过滤、目录扫描、单文件 |
| `PSMInfo.from_dict` | 老 JSON（无新字段）、新 JSON（含新字段） |
| `extract_common.extract_n_engines` | N=2、N=3 交并集；有 / 无 positive_marker |

### 6.2 集成测试

| 测试场景 | 验证点 |
|---|---|
| pfind 单引擎跑 2th/ 一个 raw | PSM 数与 QValue=0.01 过滤后一致；REV_ 数为 0 |
| pfind ∩ diann 跑 2th | JSON 输出含 label_type=positive/negative；正负比合理 |
| 老 hela.json 加载 | PSMInfo.from_dict 无报错；q_value/score/label_type 均为 None |
| pfind 三个目录都能扫描 | normal/、2th/、5th/ 都生成有效 LightResult |

### 6.3 与现有结果对照

- 老 extract_com 在 hela 数据上的输出 JSON（DIANN ∩ AlphaDIA + HUMAN）
- 新 extract_common 在相同数据上跑相同引擎组合（DIANN + AlphaDIA），输出 JSON
- 两份 JSON 的 (sequence, charge, modify) key 集合应当**完全等价**
- 偏差容忍：PSM 顺序可能不同；新 JSON 含 label_type / q_value / score 字段
- 本对照仅验证融合的正确性，与"新方案使用 pfind + DIANN"无关

---

## 七、实施里程碑

### M1: pfind loader 基础设施

- PSMInfo 扩展 q_value / score / label_type
- 新建 `spectrum/pfind_parser.py`：修饰解析 + m/z 转换 + RT 计算
- 在 `light_result.py` 添加 `_load_from_pfind_input`
- `LightResultManager` 分发支持 search_engine_type = 3
- `constant/keys.py` 添加 PFIND_QVALUE_THRESHOLD 等
- 单元测试覆盖所有 pfind_parser 函数与 loader 主路径

**验收**：单文件 .qry.res 能加载、过滤、生成 LightResult，且 PSMInfo 各字段正确。

### M2: pfind 单引擎端到端

- 目录扫描支持
- config.ini 模板更新
- 集成测试：跑 2th/ 任一 raw，与 light_result 加载结果对比

**验收**：`python main.py --config <pfind-config>` 能跑通整个特征提取流程，输出 CSV。

### M3: extract_common 通用 N 引擎

- 新建 `tools/extract_common.py`
- 实现 N 引擎交并集逻辑
- 打 label_type 字段
- CLI 入口 + config 解析
- 单元测试（N=2、N=3）

**验收**：`python tools/extract_common.py --config <extract-config>` 能跑通；输出 JSON 含 label_type 字段；与老 extract_com 输出 key 集合等价。

### M4: 兼容性与清理

- PSMInfo.from_dict 老 JSON 兜底验证
- 老 extract_com 项目添加 README 占位
- 更新 PROJECT_INFO.md 与 config.ini 示例
- 端到端集成测试（pfind ∩ diann → JSON → 特征 CSV）

**验收**：所有现有数据集（含老 JSON）都能加载；新数据集能从 pfind+diann 构造。

### M5（可选/未来）：负例细分

- 编写 `tools/refine_negative_labels.py` 基于特征 CSV 细化 negative_no_signal / negative_interference
- 不在 M1-M4 范围内，作为后续 backlog

---

## 八、风险与未决问题

### 8.1 已识别风险

1. **PredRT 语义未确认**——暂定 `rt = PredRT + DeltaRT`。若错误，所有 RT 相关特征会偏 ~1min。**M1 实施前必须验证**。
2. **pfind QValue 列含义**——暂定为 q-value（FDR 估计）。若实际是 PEP，阈值含义变化。**M1 实施时通过日志验证 QValue 分布**。
3. **大文件性能**——normal/ 单文件 10w+ 行，FDR 过滤后预计 ~5w PSM/raw。Pandas read_csv 应能处理但需性能测试。
4. **shared peptide / 蛋白群**——pfind 用 `/` 分隔多个蛋白。物种 marker 匹配采用 `in` 语义（任一蛋白含 HUMAN 即视为人源）。与 extract_com 现有行为一致。
5. **未知修饰名称**——硬编码字典覆盖有限，unimod.xml 兜底也可能不命中。未知修饰 → 跳过该 PSM 并 log warning。可能丢失数据但不影响正确性。

### 8.2 未决问题（需用户后续确认）

1. **PredRT 语义**：实测 RT 是否真的是 PredRT + DeltaRT？符号方向？
2. **pfind QValue vs PEP**：QValue 列严格为 q-value，还是某些 pfind 配置下输出 PEP？

---

## 九、与现有项目的关系

### 9.1 现有 spec 的相互独立性

本 spec（pfind + extract_common 融合）与 `2026-05-13-silac-validation-framework.md`（SILAC 多维独立证据验证框架）**完全独立**：

- 本 spec 解决"如何获得高质量数据集"问题（数据 pipeline）
- 前者解决"如何从数据中提取鉴别特征"问题（算法层）
- 二者顺序执行（数据先行），但无逻辑依赖

### 9.2 现有代码映射

| 设计章节 | 现有代码位置 | 改动方向 |
|---|---|---|
| pfind loader | 新建 `spectrum/pfind_parser.py` | 全新文件 |
| light_result 加载分发 | `spectrum/light_result.py`（170+ 行） | 添加 `_load_from_pfind_input` 方法 |
| LightResultManager 分发 | `manager/light_result_manager.py`（53 行） | 添加 search_engine_type = 3 分支 |
| PSMInfo 扩展 | `spectrum/psm_info.py`（214 行） | 添加 3 个字段，向后兼容 |
| extract_common | 新建 `tools/extract_common.py` | 全新文件 |
| 配置常量 | `constant/keys.py` | 添加 pfind 相关字段 |

---

## 十、文档维护

- 本文档为 **设计文档（design spec）**，不是实施计划
- 实施计划在本 spec 评审通过后由 writing-plans 流程产出，存放于 `docs/plans/`
- 任何对设计的修改应增量更新版本号
