# pFind 谱库读取模块 — Spec Library Reader

**Date**: 2026-06-06
**Scope**: 谱库（spectral library）二进制读取，独立模块
**Status**: Approved

## 问题

pFind 通过 pPred 生成的谱库以二进制存储，外部读取较复杂：

- **肽段**用二进制方式存（`pepdata.pdb`），只存"蛋白 id + 起止位置 + 修饰 id"，需要配合**数据库 FASTA** 还原序列、配合 **modification.ini** 还原修饰。
- **预测 RT**（`pepdata.rt.predb`）和**预测 MS2**（`pepdata.ms2.predb`）也是二进制，逐肽段（MS2 逐肽段-电荷）顺序排列。
- 读入 M 个肽段后，应对应生成 M 组预测值（每肽段 1 个 RT，1 组按电荷分桶的 MS2 碎片）。

本模块**第一步只做独立读取 + 正确性自校验**，暂不接入现有 SILAC 特征提取 pipeline（后续单独评估接入）。

## 来源线索

逆向自 `../puku/`：

- `pFindSDK/Reader.cpp` `CReader::ReadPepData()` — pepdata.pdb 解析逻辑
- `pFindSDK/Reader.cpp` `CReader::ReadMod()/ReadAA()/ReadElementInfo()` — 配置解析
- `pFindSDK/fastaparser.cpp` `CFastaParser::ReadOnePrteinEntry()` — FASTA 解析
- `pFindSDK/Condition.h:204-217` — 谱库文件命名（`pepdata.pdb` / `pepdata.rt.predb` / `pepdata.ms2.predb`）
- `pFindSDK/Condition.h:212` `gen_ms2data_info()` — MS2 info 串格式 `instru_nce0_nce1_nce2_chgMin_chgMax`
- `pPred/pPredMS2.cpp` `MS2Predictor::predict()/load_ms2_data()` — MS2 二进制写出与电荷展开
- `pPred/pPredRT.cpp` `RTPredictor::predict()` — RT 二进制写出
- `pPred/pPredMS2.h` `MS2Predictor` 默认 `chg_min=1, chg_max=4`

## 决策摘要

| 决策 | 选择 |
|---|---|
| 模块位置 | 新建小包 `spectrum/speclib/` |
| 输入：谱库目录 | 包含 `pepdata.pdb` / `pepdata.rt.predb` / `pepdata.ms2.predb`（实际打包形态待用户下载后确认，loader 以"定位三个组件"为接口，兼容目录或显式三路径）|
| 输入：解码资源 | FASTA（默认 `merge_human_ecoli_yeast.fasta`）、`modification.ini`；自校验另用 `element.ini`+`aa.ini`（可选）|
| 字节序/位宽 | 小端，x64 构建：`UINT`=u32、`int`=i32、`char`=i8、`double`=f64、`float`=f32、`short`=i16、`size_t`=u64 |
| chg_max | **从文件推断** = MS2 记录总数 / M，不依赖外部 config |
| RT 单位 | 分钟（写出时已 `norm2real`，为真实 RT）|
| decoy 判定 | 蛋白 AC 以 `REV_` 开头 |
| 正确性验证 | 三层自校验（见下），无需外部 ground truth |
| 测试 | TDD：合成微型 fixture 做确定性单测；用户真实文件做 mass 交叉校验 |

## 二进制格式契约

所有多字节字段为**小端**。

### pepdata.pdb（逐"蛋白-肽段"条目，循环到 EOF）

| 字段 | 类型 | 字节 | 含义 |
|---|---|---|---|
| pro_id | u32 | 4 | FASTA 蛋白索引（文件顺序）|
| pep_start | u32 | 4 | 在蛋白序列中的起始位置 |
| pep_len | i8 | 1 | 肽段长度 |
| pro_nc | i8 | 1 | 蛋白 N/C 端属性（0 normal,1 pro_c,2 pro_n,3 pro_nc）|
| enz | i8 | 1 | 酶切特异性（cClv）|
| miss | i8 | 1 | 漏切数 |
| mod_pep_num | u32 | 4 | 该肽段的修饰变体数 |
| mod_pep_bytes | u64 | 8 | 后续所有修饰变体块的总字节数 |

随后 `mod_pep_num` 个修饰变体，每个：

| 字段 | 类型 | 字节 | 含义 |
|---|---|---|---|
| mass | f64 | 8 | 中性单同位素肽段质量 |
| mod_cnt | i8 | 1 | 修饰数 |
| 每修饰 × mod_cnt | | | |
| &nbsp;&nbsp;pos | i8 | 1 | 修饰位置 |
| &nbsp;&nbsp;mod_id | i32 | 4 | 修饰 id（见下映射）|

- 序列还原：`fasta[pro_id].SQ[pep_start : pep_start + pep_len]`
- 每条目应满足 `bytes(变体块) == mod_pep_bytes`（自校验 2）

### pepdata.rt.predb

- 紧凑 `M × f32`，顺序与 pdb 中肽段顺序一致（每个修饰变体一个，即按 `vPeps` 展开顺序）。
- 数值为真实 RT（分钟）。

### pepdata.ms2.predb（逐"肽段-电荷"记录）

- 记录顺序：肽段为外层、电荷 `1..chg_max` 为内层（pdb 读取时 `pi.chg==0`，故每肽段全量展开 `chg_max` 个电荷）。
- 总记录数 `= M × chg_max` ⇒ `chg_max = 记录总数 / M`。
- 每条记录：

| 字段 | 类型 | 字节 | 含义 |
|---|---|---|---|
| n_size | i16 | 2 | 该记录的碎片离子数 |
| 每离子 × n_size | | | |
| &nbsp;&nbsp;pos | i8 | 1 | 碎片切割位（0-indexed，有效 `0..len-2`）|
| &nbsp;&nbsp;iontype | i8 | 1 | 离子类型编码（见下）|
| &nbsp;&nbsp;inten | f32 | 4 | 相对强度 |

- iontype 编码：偶=`b`、奇=`y`；碎片电荷 `= iontype // 2 + 1`（0=b⁺,1=y⁺,2=b²⁺,3=y²⁺,…，共 12 类对应 1–6 价）。
- 离子按强度降序写出，且已按 `inten/最大 inten > 1e-3` 截断、上限 `MAX_ION_OUTPUT=1000`。

### 修饰 id 映射（modification.ini）

复刻 `CReader::ReadMod()`：逐行读，仅取含 `=` 的"数据行"，**按文件顺序**赋 1-based id，跳过以下行（不占用 id）：

- `strKey` 以 `name` 开头（如 `name1=...`）
- `strKey == "@NUMBER_MODIFICATION"`
- `strKey == "label_name"`
- `strKey == "Met-loss+Acetyl[ProteinN-termM]"`
- 名称以 `Label_` 开头（注意：该 `continue` 在赋 id 之前，故不占 id）

数据行格式：`名称=位点 类型 单同位素质量 平均质量 NL数 [NL...] 元素组成`，取**第一个浮点**为单同位素质量。

> 注：C++ 中 `m_vID2Mod` 按质量排序但保留 read-order 的 `m_nID` 作为索引，因此二进制里的 `mod_id` 等价于"过滤后数据行的文件顺序 1-based 序号"。

### FASTA 解析（复刻 CFastaParser）

- 条目以 `>` 行起始；AC = `>` 后到第一个空格/Tab/`|`（`|` 仅当位置 > 15 时作为分隔）之间的子串；其余为 DE；SQ = 后续序列行拼接。
- 蛋白按文件顺序入表，`pro_id` 即该顺序索引。

## 数据模型

```
LibPeptide:
    sequence: str
    mods: list[ModSite]            # (pos, mod_id, mod_name, mod_mass)
    neutral_mass: float            # pdb 中存储的 mass
    protein: str                   # 首个蛋白 AC
    is_decoy: bool                 # AC.startswith("REV_")
    charge_mask: int               # pdb 读取恒为 0（全电荷）
    pred_rt: float | None          # 分钟
    pred_ms2: dict[int, list[FragIon]]   # charge -> ions

FragIon:
    ion_type: str                  # 'b' | 'y'
    frag_pos: int                  # 0-indexed 切割位
    frag_charge: int               # 1..6
    intensity: float

ModEntry:                          # modification.ini 一条
    mod_id: int
    name: str
    mono_mass: float
    sites: str
    mod_type: str
```

## 模块职责

| 文件 | 职责 | 依赖 |
|---|---|---|
| `config_io.py` | 解析 element.ini+aa.ini → 残基质量；modification.ini → `list[ModEntry]`/`{id:ModEntry}`；FASTA → `list[Protein]` | 纯文本 |
| `pepdata.py` | 读 `pepdata.pdb` → `list[LibPeptide]`（仅序列/修饰/质量，不含预测）| config_io |
| `predictions.py` | 读 `pepdata.rt.predb` → `list[float]`；读 `pepdata.ms2.predb` → `list[record]`，并按 `chg_max` 分组为 `{charge:[FragIon]}` | — |
| `speclib.py` | `SpecLib` 顶层 loader：定位组件、组装 M 个 `LibPeptide`（含 RT/MS2）、暴露查询与自校验入口 | 以上全部 |

每个单元可独立测试：输入明确（路径/字节）、输出明确（dataclass 列表）、依赖单向。

## 正确性自校验（无需外部 ground truth）

1. **质量交叉校验**（最关键，验证序列+修饰解码）：用 element.ini+aa.ini+modification.ini 独立重算中性质量 = Σ残基 + H₂O + Σ修饰质量，与 pdb 存储 `mass` 比较（容差 ~1e-4 Da）。需要 element.ini/aa.ini 时启用，缺省可跳过并告警。
2. **结构完整性**：每条目变体块消耗字节 == `mod_pep_bytes`（同时确认 `size_t`=8）。
3. **计数一致性**：`len(rt) == M`；`len(ms2_records) % M == 0`，得整数 `chg_max`。
4. **范围检查**：MS2 `frag_pos ∈ [0, len-2]`、`iontype ∈ [0,11]`。

## 测试策略

- **合成 fixture 单测**：手工构造微型 FASTA（1–2 蛋白）、modification.ini 子集、按上表 hand-craft 的 `pepdata.pdb` / `rt.predb` / `ms2.predb` 字节串，断言：序列/修饰/质量、RT 列表、MS2 分组与 ion 解码、chg_max 推断、三层自校验全部通过。
- **真实文件验证**：用户提供真实谱库后，跑 loader + 质量交叉校验，统计通过率与异常条目。
- 复用仓库现有 `tests/` 约定与 pytest。

## 非目标（本步不做）

- 不接入 `workflows/` / `single_work.py` SILAC 特征提取。
- 不做 MS2 碎片 m/z 计算与实测谱比对（接入阶段再做）。
- 不解析 `pepdata.match.predb` 等其他可能的预测分量（仅 RT + MS2）。
- 不处理 decoy 生成（`gen_decoy=false`，库内已有即读）。
