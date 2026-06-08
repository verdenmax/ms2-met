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
| 输入：谱库目录 | **已确认为目录**，含 `pepdata.pdb` / `pepdata.rt.predb` / `pepdata.ms2.predb`（外加 `model_*.pt`，读取时忽略）；用户所谓"一个文件"= 下载的 zip。loader 支持目录或显式三路径 |
| 输入：解码资源 | FASTA（默认 `merge_human_ecoli_yeast.fasta`）、`modification.ini`；自校验另用 `element.ini`+`aa.ini`（可选）|
| 字节序/位宽 | 小端，x64 构建：`UINT`=u32、`int`=i32、`char`=i8、`double`=f64、`float`=f32、`short`=i16、`size_t`=u64 |
| chg_max | **从文件推断** = 有效 MS2 记录数 / M（排除文本尾巴后），真实库实测 = 4；亦可从尾巴行解析交叉验证 |
| 体量策略 | ms2 ~4.4GB / ~1250 万记录、pdb ~312 万肽段 → **不全量物化**。核心 = **锁步流式**（pdb/RT/MS2 同序逐肽段 yield，内存 O(1)）；RT 小（~12MB）全量进数组；**随机按肽段查 MS2 的缓存偏移索引延到接入 pipeline 阶段** |
| RT 单位 | 分钟（写出时已 `norm2real`，为真实 RT）|
| decoy 判定 | 蛋白 AC 以 `REV_` 开头 |
| 正确性验证 | 多层自校验（见下），无需外部 ground truth |
| 测试 | TDD：合成微型 fixture（含文本尾巴）做确定性单测；用户真实文件做 mass 交叉校验 |

## 验证结果（2026-06-06，对真实 `lib-2th` 实测）

- 库形态 = 目录，含 `pepdata.pdb`(91,816,563 B) / `pepdata.rt.predb`(12,498,080 B) / `pepdata.ms2.predb`(4,465,990,502 B) / `model_ms2.pt` / `model_rt.pt`。
- pdb：M = **3,124,520** 肽段变体；序列+修饰**质量交叉校验 100%**（max_err = 0）；每条目 `mod_pep_bytes` 全部自洽；FASTA 59,490 蛋白。
- RT：3,124,520 个 f32 = **恰好 = M**；值 7–77 min。
- MS2：`[4×M=12,498,080 二进制记录][M 行文本尾巴]`；尾巴 = 17×M = 53,116,840 B，内容为每行 `"1\t0\t2\t0\t3\t0\t4\t0\t\n"`（即 chg_max=4）；以 `n_size>1000` 守卫停在尾巴，停点字节恰为尾巴起点。

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

### pepdata.ms2.predb（`[二进制记录区][文本尾巴]`）

文件结构：**先**是 `M × chg_max` 条二进制记录，**后**紧跟一段文本尾巴（见末尾）。

- 记录顺序：肽段为外层、电荷 `1..chg_max` 为内层（pdb 读取时 `pi.chg==0`，故每肽段全量展开 `chg_max` 个电荷）。
- 有效记录数 `= M × chg_max`；`chg_max = 有效记录数 / M`（实测 4）。
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
- **边界**：`n_size==0` 的记录照样写出（每"肽段-电荷"恒有 2 字节头），读取须当作"存在的空记录"；charge-c 记录只含 `frag_charge ≤ c` 的离子（各桶不对称）；M = Σ mod_pep_num（变体总数）。

**文本尾巴（必须跳过）**：二进制记录区之后追加 `M` 行 ASCII 文本，每行 `"1\t0\t2\t0\t…\tchg_max\t0\t\n"`（行长 `4×chg_max+1` 字节）。成因：`MS2Predictor::predict()` 收尾的 `fprintf` 循环在 `if(binary)` 之外，二进制模式下 `curr_pep_id=0/curr_pep_chg=1` 未更新，故对每肽段都写一行文本残留（pPredMS2.cpp:868-873）。pFind 引擎只读前 `M×chg_max` 条二进制记录、忽略尾巴。
**读取规则**：逐记录读，当 `n_size < 0` 或 `n_size > MAX_ION_OUTPUT(1000)` 即判定进入尾巴并停止（尾巴首 2 字节 `'1'(0x31)'\t'(0x09)` 作小端 i16 = 0x0931 = 2353 > 1000）。停止后应满足 `有效记录数 == chg_max × M` 且 `剩余字节 == M × (4×chg_max+1)`。

### 修饰 id 映射（modification.ini）

复刻 `CReader::ReadMod()`：逐行读，仅取含 `=` 的"数据行"，**按文件顺序**赋 1-based id，跳过以下行（不占用 id）：

- `strKey` 以 `name` 开头（如 `name1=...`）
- `strKey == "@NUMBER_MODIFICATION"`
- `strKey == "label_name"`
- `strKey == "Met-loss+Acetyl[ProteinN-termM]"`
- 名称以 `Label_` 开头（注意：该 `continue` 在赋 id 之前，故不占 id）

数据行格式：`名称=位点 类型 单同位素质量 平均质量 NL数 [NL...] 元素组成`，取**第 3 个 token（单同位素质量）**为 `mono_mass`（复刻 `is_in >> sites >> type >> mass`；位点/类型非数值，故等价于"首个浮点"）。

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
    pred_rt: float | None          # 分钟（锁步流式时逐肽段填充）
    pred_ms2: dict[int, list[FragIon]]   # charge -> ions（锁步流式时逐肽段填充）

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

> `LibPeptide` 结构不变，但**不一次性物化全部 312 万个**：核心 API 为锁步流式生成器，逐肽段产出已填好 `pred_rt`/`pred_ms2` 的对象，由调用方即用即弃。

## 模块职责

| 文件 | 职责 | 依赖 |
|---|---|---|
| `config_io.py` | 解析 element.ini+aa.ini → 残基质量；modification.ini → `list[ModEntry]`/`{id:ModEntry}`；FASTA → `list[Protein]` | 纯文本 |
| `pepdata.py` | `iter_pepdata(...)` 生成器逐条 yield `LibPeptide`（仅序列/修饰/质量，不含预测）；`read_pepdata(...)` 薄包装返回 list（小数据/测试用）| config_io |
| `predictions.py` | `read_rt_pred` → `array('f')`（全量，~12MB）；`iter_ms2_records` 生成器逐记录 yield `list[FragIon]`，遇 `n_size<0 或 >1000` 即停（跳过文本尾巴）；`read_chg_max_from_trailer` 从尾巴行解析 chg_max | — |
| `speclib.py` | `SpecLib`：解析 proteins/mods、加载 RT 数组、确定 chg_max；`iter_peptides()` **锁步流式**组装 pdb+RT+MS2 逐肽段 yield；`validate_masses(...)` 流式质量校验 | 以上全部 |

每个单元可独立测试：输入明确（路径/字节）、输出明确（dataclass / 生成器）、依赖单向。**随机按肽段查 MS2 的缓存偏移索引不在本步**（接入 pipeline 阶段再加）。

## 正确性自校验（无需外部 ground truth）

1. **质量交叉校验**（最关键，验证序列+修饰解码）：用 element.ini+aa.ini+modification.ini 独立重算中性质量 = Σ残基 + H₂O + Σ修饰质量，与 pdb 存储 `mass` 比较（容差 ~1e-4 Da）。**真实 `lib-2th` 实测 100% 通过（max_err=0）**。质量公式已由 C++ 验证：`_m2mz(m,chg)=(m+chg*proton)/chg`（sdk.h:881）证明 `lfPepMass` 不含质子；y 离子加 `MOLECULE_MASS_H2O`（Instrument.cpp:45/61）证明含水；残基质量来自 aa.ini 组成×element.ini（不含水，Reader.cpp:209-218）。
2. **结构完整性**：每条目变体块消耗字节 == `mod_pep_bytes`（同时确认 `size_t`=8）。**实测全部自洽**。
3. **计数一致性**：`len(rt) == M`（实测相等）；MS2 流式停在尾巴后 `有效记录数 == chg_max × M`、`剩余字节 == M × (4×chg_max+1)`；`chg_max ∈ [1,6]`（硬校验）。
4. **范围检查**：MS2 `frag_pos ∈ [0, len-2]`、`iontype ∈ [0,11]`。

## 测试策略

- **合成 fixture 单测**：手工构造微型 FASTA（1–2 蛋白）、modification.ini 子集、按上表 hand-craft 的 `pepdata.pdb` / `rt.predb` / `ms2.predb`（**含文本尾巴**）字节串，断言：序列/修饰/质量、RT、MS2 解码、**尾巴停止**、chg_max、锁步对齐、各层自校验。
- **真实文件验证**：`tools/speclib_inspect.py` 流式跑真实库（4.4GB 不 OOM），输出摘要 + 质量通过率 + chg_max + 尾巴检查 + 样例肽段；支持 `--limit N` 只校验前 N 条加速。
- 复用仓库现有 `tests/` 约定与 pytest。

## 文档交付（分层 L1–L4，边写代码边填）

随代码同步产出分层文档，结构按组件分目录：

```
docs/speclib/
  L1_overview.md                      # 整个模块：目标/架构/数据流/快速上手/关键事实
  parts/<组件>/L2_role.md             # 组件职责与对外接口
  parts/<组件>/L3_details.md          # 组件细节：格式/算法/边界/取舍/对应 C++
  parts/<组件>/L4_api.md              # 逐源文件 API 参考
```

组件 = `config_io` / `pepdata` / `predictions` / `speclib` / `speclib_inspect`。每个实现任务在测试通过后、提交前写对应 L2–L4 并随代码同 commit；L1 在首个任务建骨架、末个任务回填。

## 非目标（本步不做）

- 不接入 `workflows/` / `single_work.py` SILAC 特征提取。
- 不做 MS2 碎片 m/z 计算与实测谱比对（接入阶段再做）。
- **不做随机按肽段查 MS2 的缓存偏移索引**（接入阶段再做）。
- 不解析 `pepdata.match.predb` 等其他可能的预测分量（仅 RT + MS2）。
- 不处理 decoy 生成（`gen_decoy=false`，库内已有即读）。
