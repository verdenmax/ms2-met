# 设计：用 pFind 谱库预测强度赋能轻重标验证特征（第一版）

> 文档版本：1.3（2026-06-09，加入 Phase 2c J5 自适应覆盖度 + 更正 §4.7 公式）
> 状态：Phase 1 基础已实现并**实测验证通过**（修正后判据）；Phase 2 待写实现计划
> 关联：
> - `docs/specs/2026-05-13-silac-validation-framework.md`（SILAC 验证框架，类2/3/4、情形 B、Q1a）
> - `docs/specs/2026-06-06-pfind-speclib-reader-design.md` + `docs/speclib/`（谱库 reader，已实现）
> - 集成点：`workflows/single_work.py`（碎片循环）、`workflows/q1a_helpers.py`（Q1a 累加器/信号存在判据）、`workflows/pair_flow.py`（任务编排）、`spectrum/psm_info.py`（`get_heavy_info` 碎片枚举）

---

## 1. 背景与动机

工具的使命：对搜索引擎给出的**轻标**鉴定，用「轻重标必须成对、且有可预测物理关系」去**约束重标**，从而提取轻重关系特征并判定鉴定可信度（不依赖 decoy）。

现状：主流程（`single_work.py`）枚举理论 b/y 碎片，对每个碎片提取轻标/重标 MS2 XIC，算逐碎片 Pearson/cosine/apex_delta 等，并由 `Q1aAccumulator` 统计「同时有可信轻、重信号」的碎片数。**所有可分碎片被等权对待**，且**完全没有用到谱库的预测强度/预测 RT**（已确认：`single_work.py`/`psm_info.py`/`dia_data.py` 不引用 speclib）。

框架明确的主要弱点是 **5Da 下的「情形 B」**：真实共流出、但身份错误的肽段，能伪装时间形状与同位素包络，仅靠相似度判据无法拒绝。

谱库（pPred 生成）**唯一**提供、而管线尚未利用的信息：
1. **预测的 b/y 碎片相对强度模式**（深度模型给出「哪些碎片该强/该弱」）。
2. 预测 RT（每肽段一个浮点；本版**不用**，见非目标）。

核心洞察：**纯轻标的谱图相似度只用轻标、偏题**（等同搜索引擎的谱库重打分）。谱库对本工具的价值在于把「预测强度」**路由进轻重关系**：
- 决定哪些碎片可信（top-K 选择/加权）；
- 由 L 的预测模式**构造预测重标谱**去检验实测重标（情形 B 的克星——干扰肽段复现不了 L 的预测强度模式）。

---

## 2. 目标与非目标

### 2.1 第一版目标（本 spec）

| 编号 | 名称 | 一句话 |
|---|---|---|
| **0** | 前置 sanity gate | 验证库预测在本数据上与实测轻标相符，过阈才继续 |
| **1** | Lookup 基础设施 + 覆盖率 | 一遍流式抽出命中肽段预测进内存字典 + fallback |
| **2** | top-K 碎片选择/加权 | 可分碎片中按预测强度取前 K（默认 6，可配），驱动所有碎片级特征 |
| **3** | I1 强度模式一致性 | 构造预测重标谱 → 与实测重标算谱角/Spearman |
| **4** | I2 H/L 比一致性 | top-K 上每碎片 H/L 比的（预测加权）离散度 |
| **5** | I3 预测覆盖度 | top-K 中实际拿到可信重标信号的比例 |
| **6** | J2 意外峰污染 | 预测弱/≈0 的位置上出现可信重标信号的比例/强度占比 |
| **7** | J5 自适应信号存在判据 | 用「预测强度×轻标apex×全局L:H比」作期望，替代固定 floor=100 |
| **8** | 性能红利 | 只对 top-K 抽 MS2 XIC，减少每 PSM 的 XIC 提取次数 |

### 2.2 非目标（本版明确不做）

- **纯轻标谱图相似度作为主特征**：偏题，最多日后作次要协变量。
- **预测 RT 特征**：RT 轻重共享，是「身份」约束而非「轻重关系」，本版搁置。
- **持久化 offset 索引**：本版用「一遍流式 + 跳过非命中」，不建可复用索引。
- **J6 多电荷碎片证据扩展、J10 预测谱信息量加权**：放 backlog。

---

## 3. 总体数据流与集成点

```
pair_flow 启动阶段（一次）
  收集本次所有被鉴定肽段 key {(seq, mods, charge)}
        │
        ▼  一遍流式扫库 SpecLib.iter_peptides(decode_ms2="arrays")
  命中 key → 抽 pred_ms2 进内存字典 PredStore：key → 预测碎片强度（按 (ion_type, frag_pos, frag_charge)）
        │  （未命中肽段：无预测 → fallback）
        ▼
single_work 碎片循环（每 PSM，**中心化谱图**）
  算 heavy 母离子 m/z → is_split_window 判「分窗/同窗」（§4.1.5）
  由 PredStore 给每个枚举碎片打上 pred_intensity
        │
        ├─ 可分碎片（分窗→全部碎片；同窗→仅 SILAC 位移碎片）中取预测最强 K 个 = 碎片集 F
        ├─ 现有逐碎片特征仅在 F 上算（去噪/提速）
        ├─ I1：构造 pred_H、obs_H → 谱角/Spearman
        ├─ I2：F 上 H/L 比离散度（预测加权）
        ├─ I3：F 中重标「存在」(J5 判据) 的比例
        ├─ J2：预测弱集合上的意外重标信号占比
        └─ J5：替换 Q1a/存在判定的强度门槛
        ▼
  追加新特征列 → result.csv（每列对未命中 PSM 置 NaN，并出 has_lib_pred 标志）
```

设计原则：**库特征是「增量旁路」**——未命中或 sanity gate 不过时，主流程行为回退到现状（新列 NaN），不破坏既有特征与既有测试。

---

## 4. 详细设计

### 4.0 前置 sanity gate（独立分析步，先跑）

- **入口**：扩展 `tools/speclib_inspect.py` 或新增 `tools/speclib_sanity.py`，离线运行，不进主流程热路径。
- **输入**：一批高置信轻标 PSM（如 pFind q<0.01 且 `light_max_intensity` 高）+ 谱库目录 + 对应 raw。
- **算法**：**在中心化谱图上**，对每条 PSM，按 `(ion_type, frag_pos, frag_charge)` 对齐「预测相对强度 `p`」与「实测轻标碎片强度 `l`」（XIC apex），**只取可分碎片**（§4.1.5；同窗时观测到的轻标未位移碎片会被重标污染，须排除），**b、y 分开**计算谱角相似度；汇总分布（中位、p25/p75）。
- **决策门槛**：在**可分碎片**上算中位谱角相似度 `> SANITY_MIN`（**默认 0.6**，可配；见 §11 实测）→ gate 通过；否则停止并报告，重点排查：单位、**b/y 序号 ↔ frag_pos 对齐**、电荷选择、修饰映射。**度量须分 ion-type**（b、y 不混进同一向量，见 §7/§11）。
- **产出**：一份统计报告（stdout + 可选 CSV）。该步**只用轻标**，是工具自检，不产出最终特征。

### 4.1 Lookup 基础设施 + 覆盖率 + 对齐

- **PredStore 构建**（`workflows` 下新模块，如 `workflows/pred_store.py`）：
  - 输入：被鉴定肽段的 key 集合、谱库句柄。
  - 用 `SpecLib.iter_peptides(decode_ms2="arrays")` 流式扫一遍；命中 key 的，把该肽段的预测碎片（结构化数组 `pos,iontype,inten`）整理成 `dict[(ion_type:str, frag_pos:int, frag_charge:int)] -> intensity`，连同 `pred_rt`（暂存不用）放入 `PredStore[key]`。
  - 未命中 key 不入store。内存 O(命中肽段数)。
- **key 规范化**：`(sequence, mods_normalized, charge)`。需统一：修饰表示（与库一致的 mod 编码/顺序）、`charge ≤ chg_max`、I/L 视库约定。规范化函数集中一处，并在 sanity gate 复用以保证一致。
- **b/y 序号 ↔ frag_pos 对齐**：管线碎片来自 `psm.get_heavy_info(SILAC)`，元素 `(ion_type, ion_num, light_mass, heavy_mass)`，`ion_num` 为 1-based b/y 序号；库 `FragIon.frag_pos` 为 0-indexed 切割位。需建立映射（如 b 的 `ion_num=i` ↔ `frag_pos=i-1`，y 同理但方向相反），**该映射的正确性由 4.0 sanity gate 验证**（错位会让相似度系统性变差）。
- **覆盖率/fallback**：每 PSM 记 `has_lib_pred ∈ {0,1}`；未命中 → 所有库特征列 NaN，主流程其余不变。汇总命中率写日志。

### 4.1.5 可分碎片判定（split-window-aware）+ 中心化要求（核心；实测见 §11）

> 这是整套库特征的**地基规则**，被 §4.0 与 §4.2–§4.7 共同依赖。

**中心化硬前提**：speclib 特征的所有 XIC 提取与强度比较**必须在中心化谱图上**进行（`centroid_enabled=True`，§6）。实测 profile→centroid 各切片相似度 **+0.07~0.10**（y 离子受益最大）；**profile（未中心化）数据下库特征不可信，应关闭/置 NaN**。

**可分性规则**：一个碎片是否「可用于轻重标验证」取决于轻、重**母离子**是否落在同一 isolation window —

1. 算重标母离子 m/z：`heavy_pre_mz = light_pre_mz + ΔSILAC / charge`，其中 `ΔSILAC = get_SILAC_increase_mass(seq)`（= 8.014·#K + 10.008·#R）。
2. `split = is_split_window( get_window_info(light_pre_mz), get_window_info(heavy_pre_mz) )`。
3. **分窗（split=True）→ 所有碎片都可用**：轻、重肽段在**不同 MS2** 里被碎裂；轻碎片从轻窗口（用 `light_pre_mz`）提取得**纯轻**，重碎片从重窗口（用 `heavy_pre_mz`）提取得**纯重**，互不污染。
4. **同窗（split=False，co-isolated）→ 只有被 SILAC 改变了 m/z 的碎片可用**：轻、重在**同一 MS2**；含 K/R 的碎片（`heavy_mass≠light_mass`）轻在 light-m/z、重在 heavy-m/z，可分开 → 可用；**未位移碎片**（如不含 K/R 的 b 离子）轻重落**同一 m/z、重叠**，分不开 → **不可用**。
5. **undecided（split=None）**：重标母离子 m/z 查不到窗口（常因落在本采集 50-Da 范围外 → 重标可能在**另一个采集文件**）。保守**按同窗处理**（只用位移碎片），并记 `heavy_out_of_range` 标志，留待跨文件处理。
6. 综合即 `is_separable_fragment(light_mass, heavy_mass, split)`（= 位移 **OR** 分窗）。

**实测（§11，2Da DDIA）**：~93% PSM 分窗（split=1391 / coiso=8 / undecided=101）→ 绝大多数碎片可用（**含 b**）；**不要把可分性简化成「只用 y」**。

### 4.2 top-K 碎片选择 / 加权

- **碎片集 F**：先按 §4.1.5 定出该 PSM 的**可分碎片集**（分窗→全部碎片；同窗→仅 SILAC 位移碎片；undecided→仅位移），再在其中按 `pred_intensity` 降序取前 `TOP_K`（默认 6，可配）；不足 K 取实际数量。**仅在中心化谱图上**算。
- **加权**：聚合碎片级特征到 PSM 级时，按 `pred_intensity` 归一权重加权（如加权均值）。
- **既有特征改造**：现有逐碎片 Pearson/cosine/apex_delta/mz_err/SNR 等的**聚合**改为在 `F` 上（并提供预测加权版）。为不破坏既有列与测试，新列以新名输出（如 `*_topk`、`*_wpred`），**保留旧列**或由 config 开关切换（见 §6）。
- **元特征**：`n_separable_in_predicted_topK`（预测 top 中可分的数量）、`n_fragments_in_F`。
- **⚠️ 可分性是 split-window-aware**（实测，§11）：`is_separable_fragment` = 碎片被 SILAC 位移 **OR** 轻/重母离子分窗（`is_split_window`）。2Da DDIA 实测 ~93% 分窗 → 轻重在不同 MS2 → **几乎所有碎片（含 b）都可分/可用**，**不是只用 y**。b 离子实测预测最准（cos 0.96），**不应剔除**。

### 4.3 I1 · 强度模式一致性（构造预测重标谱）

- **构造 `pred_H`**：对 `F` 中每碎片，取其预测相对强度（即 L 的预测强度，化学等价 → 重标强度模式与轻标一致），形成向量（位置=重标碎片）。
- **`obs_H`**：实测重标各碎片强度（XIC apex 或积分），与 `pred_H` 同序对齐、同一归一化（L2 或和归一）。提取用**重标母离子 m/z** 定位重标窗口（分窗时是另一个窗口，§4.1.5）；中心化谱图。
- **特征**（均以**重标**通道 `obs_H` 为观测侧，分 ion-type 后取均值合并）：
  - `spec_pattern_SA_b` / `spec_pattern_SA_y` = b、y 各自的谱角相似度(`pred`, `obs_H`)；`spec_pattern_SA` = 两者中有限值的均值。
  - `spec_pattern_spearman_b` / `spec_pattern_spearman_y` = b、y 各自的 **Spearman 排序相关**(`pred`, `obs_H`)（每类需 ≥3 根碎片，n=2 退化为 ±1 → NaN）；`spec_pattern_spearman` = 两者均值。**纯排序度量,对「b2 单峰主导 / b:y 整体缩放偏差」鲁棒**(见 §11 实测:Spearman 下 y 反超 b)。
  - `spec_pattern_LH_consistency` = 预测加权的 corr(`obs_L`, `obs_H`)（轻重应同形）。
- **攻击**：情形 B（干扰肽段的碎片强度模式由自身序列决定，复现不了 L 的预测模式）。
- **度量定义**：见 §7。空/单元素向量 → NaN。

### 4.4 I2 · H/L 比一致性

- 在 `F` 中、轻重都有可信信号的碎片上，算 `r_i = I_heavy_i / I_light_i`。
- **特征**：`hl_ratio_cv_weighted`（预测加权变异系数）、`hl_ratio_mad`（中位绝对偏差）。比值越一致越像真对。
- 与框架类4一致，但 top-K 去噪后分母更干净。少于 2 个有效碎片 → NaN。

### 4.5 I3 · 预测覆盖度

- 分母 = `|F|`；分子 = `F` 中重标信号「存在」（用 §4.7 的 J5 判据）的碎片数。
- **特征**：`pred_coverage`（命中/|F|）、`pred_coverage_wpred`（按预测强度加权命中）。
- **双端在场（轻重都有信号）**：`n_both_present` = `F` 中 `light_apex > floor` **且** `heavy_apex > floor` 的碎片**数目**；`pred_both_present_fraction` = 该数目 / `|F|`。与 `pred_coverage`（只看重标在场）互补——真肽两条链都应出现，trap 常只剩轻、重缺。无库覆盖时为 NaN（与「有库但 0 个双端在场」的可疑情形区分）。
- **攻击**：身份错的肽段更难让「该强的碎片」在重标同时出现。

### 4.6 J2 · 意外峰污染（I3 的反面）

- **预测弱集合 W**：在该 PSM 的**可分碎片**（§4.1.5）里，取 `pred_intensity` 最低的若干（或 `pred_intensity ≈ 0` 者），与 `F` 不相交。
  - **实现取舍（更严格、更安全）**：代码里 W 定义为「库**完全没有预测**的可分碎片」（`pred_frags` 查不到键），而非「预测最弱的若干」。这样**被 top-K 截掉的低预测碎片既不进 F 也不进 W**——它们仍是模型预测「该有」的碎片，不应被算作「意外重标」，避免把低预测真碎片误判成污染。
- **特征**：
  - `unexpected_heavy_fraction` = W 中出现可信重标信号（J5 判据或简单 floor）的比例。
  - `unexpected_heavy_intensity_ratio` = W 上重标信号强度和 / `F` 上重标信号强度和。
- **攻击**：情形 B 的反面——干扰肽段常在 L 未预测的碎片上出信号。

### 4.7 J5 · 自适应信号存在判据（Phase 2c 已实现「自适应覆盖度」版）

- 现状：`q1a_helpers.is_signal_present_heavy` 用绝对 `intensity_floor=100`；speclib I3/J2 用固定 `pred_presence_floor`。
- **⚠️ 公式更正**（Phase 2c 实测）：期望重标强度应为 `E_i = light_apex_i × global_LH_ratio`。原稿 `E_i = pred_rel_i × light_apex_i × global_LH_ratio` 里的 `pred_rel_i` 是**重复计量**——观测到的 `light_apex_i` 本身已携带该碎片的逐碎片强度，再乘预测相对强度会双重计数。正确的物理期望就是「该碎片轻标强度 × 全局 L:H 比」。
- **判据**：碎片「存在」⇔ `heavy_apex_i ≥ α · light_apex_i · global_LH_ratio`。
  - `global_LH_ratio`（`global_lh_ratio` 列）：F 中两端 apex 均 >0 碎片的 `heavy/light` **中位数**；估不出 → NaN。
  - `α`：`[speclib] pred_signal_alpha`，缺省 0.2。
- **Phase 2c 落地形态（feature_type=0）**：作为**增量列** `global_lh_ratio` + `pred_coverage_adaptive`（= F 中 light>0 且满足上式的占比）经 `compute_speclib_adaptive` 输出,**不**改既有 I3/J2 的固定-floor 判定(并存,模型可同时用)。
- **未来**：把 `q1a_helpers.is_signal_present_heavy` 也改成可选的自适应判据（config 开关；关闭时行为不变，保护既有 Q1a 测试）——留待后续。


### 4.8 性能红利

- 只对 `F`（top-K）抽 MS2 XIC，而非全部理论 b/y → 每 PSM 的 `xic_ms2_peaks_extract` 调用数从 ~(2·肽长) 降到 ~`K`。需保证 Q1a/既有特征若仍需全碎片，可在 config 下切换「全碎片」与「仅 F」两种模式（默认保守：先全算、只新增 F 上的库特征；提速作为后续开关）。

---

## 5. 新增特征列清单（汇总）

| 列名 | 来源 | 说明 |
|---|---|---|
| `has_lib_pred` | 4.1 | 该 PSM 是否命中谱库预测（0/1） |
| `n_fragments_in_F` / `n_separable_in_predicted_topK` | 4.2 | top-K 碎片集规模/可分计数 |
| `psm_is_split_window` / `heavy_out_of_range` | 4.1.5 | 轻重母离子是否分窗 / 重标母离子是否超出本采集范围 |
| `spec_pattern_SA_b` / `_y` / `spec_pattern_SA` | I1 | 预测重标谱 vs 实测重标 谱角相似度（b、y 各自 + 均值） |
| `spec_pattern_spearman_b` / `_y` / `spec_pattern_spearman` | I1 | 同上，**Spearman 排序相关**（每类 ≥3 根；抗 b2 主导/b:y 缩放偏差） |
| `spec_pattern_LH_consistency` | I1 | 预测加权 corr(obs_L, obs_H) |
| `hl_ratio_cv_weighted` / `hl_ratio_mad` | I2 | H/L 比离散度 |
| `pred_coverage` / `pred_coverage_wpred` | I3 | 预测覆盖度 |
| `n_both_present` / `pred_both_present_fraction` | I3 | top-K 中轻重**都有**信号的碎片数 / 占比 |
| `unexpected_heavy_fraction` / `unexpected_heavy_intensity_ratio` | J2 | 意外峰污染 |
| `global_lh_ratio` / `pred_coverage_adaptive` | J5（Phase 2c） | 全局 H/L 比中位数 / 自适应覆盖度（`heavy≥α·light·glh`） |
| （可选）`*_topk` / `*_wpred` 版既有聚合特征 | 4.2 | 在 F 上/预测加权重算的既有特征 |

未命中 PSM 的上述列统一为 NaN。

---

## 6. 配置参数（`config.ini` `[general]` 或新 `[speclib]` 段）

| 键 | 默认 | 说明 |
|---|---|---|
| `speclib_dir` | （空=关闭） | 谱库目录；空则整套库特征关闭，主流程回退现状 |
| `speclib_fasta` / `speclib_mod` | — | 解码所需 FASTA / modification.ini |
| `pred_top_k` | 6 | top-K |
| `pred_metric` | `spectral_angle` | I1 主度量（`spectral_angle`/`entropy`/`pearson`） |
| `pred_signal_alpha` | 0.2 | J5 期望强度系数 α |
| `pred_use_adaptive_floor` | false（首版保守） | 是否启用 J5 自适应存在判据 |
| `pred_extract_only_topk` | false | 是否只对 F 抽 XIC（性能红利开关） |
| `sanity_min_similarity` | **0.6** | gate 通过阈值（仅 sanity 工具；可分-中心化谱角，见 §11） |

> **中心化为硬前提**（§4.1.5）：speclib 特征要求 `centroid_enabled = true`。若为 `false`（profile），实测库特征不可信（§11），应整套关闭并把库特征列置 NaN。

---

## 7. 度量定义

- **谱角相似度** `SA(a,b) = 1 - (2/π)·arccos( (a·b)/(‖a‖‖b‖) )`，向量先非负化、和/或 L2 归一；对稀疏强度向量比 Pearson 稳。
- **谱熵相似度**（可选）：基于归一化强度的熵，1 − Jensen-Shannon 型；备选度量。
- **Spearman**：秩相关，抗模型绝对强度尺度偏差。
- 约定：长度 < 2 或全零向量 → 度量返回 NaN（不返回 0，避免与「真实低相似」混淆）。
- **谱角 vs 纯 cosine**：`SA = 1 − (2/π)·arccos(cos)`，对高相似区更敏感（cos 在 θ≈0 处平，把「好/极好」压成一窄带）。实测对照（中心化，§11）：b 谱角 0.82 ↔ cos 0.96；y 谱角 0.64 ↔ cos 0.84；ALL 谱角 0.59 ↔ cos 0.80。**主度量用谱角**。
- **⚠️ 分 ion-type**（实测，§11）：预测的 **b:y 整体强度比例**标定有偏，b、y **不要**混进同一向量直接算谱角（会拖低——实测 ALL 0.59 < b 0.82 与 y 0.64 两子集）；应 **b、y 各自算谱角再合并**，或用对 b:y 缩放鲁棒的度量。

---

## 8. 测试策略（TDD）

每块先写失败测试再实现。最小集合：

- **PredStore（4.1）**：构造小 SpecLib（复用 `tests/conftest.py` 的 `_build_pdb/_build_rt/_build_ms2`），断言命中 key 取到预测、未命中 `has_lib_pred=0`；key 规范化/对齐单测（含 b/y↔frag_pos 映射的具体数值用例）。
- **top-K（4.2）**：给定可分/不可分混合 + 预测强度，断言选中的正是「可分中预测最强的 K 个」；不足 K 的边界。
- **I1（4.3）**：构造 `pred_H` 与 `obs_H` 完全一致 → SA≈1、Spearman≈1；打乱强度模式 → 显著下降；长度<2 → NaN。
- **I2（4.4）**：所有碎片同一 H/L 比 → CV≈0；一个离群 → CV 上升。
- **I3（4.5）**：F 中部分重标「存在」→ coverage = 命中/|F|。
- **J2（4.6）**：在预测弱集合放入重标信号 → `unexpected_*` 上升；无则≈0。
- **J5（4.7）**：开关关闭时 `is_signal_present_heavy` 与现状逐位一致（回归保护）；开启时按 `α·E_i` 判定的数值用例。
- **集成回归**：`speclib_dir` 为空时 `result.csv` 列与数值与现状一致（增量旁路不破坏既有 4 项之外的测试）。
- 全量 `python -m pytest tests/ -q`（注意 4 项既有失败与本工作无关）。

---

## 9. 风险与开放问题

1. **gate 不过**：~~若库预测与实测轻标相关性差~~ — **已实测通过**（§11：覆盖 100%，可分-中心化谱角 0.64 / cos 0.84，远离随机）。
2. **b/y↔frag_pos 对齐**：错位会系统性拉低相似度；必须由 sanity gate + 专门单测锁定。**已实测验证**：y 必须反向 `frag_pos=seq_len-ion_num-1`（0.64 vs forward 0.19，§11）。
3. **覆盖率**：库由特定 FASTA+酶+修饰生成；被鉴定肽段未必全在库内；fallback 已设计。**实测**：未修饰子集命中率 100%；修饰肽段的命中率（mod 映射）仍需评估。
4. **修饰映射**：库的修饰编码与 PSM 的 `_modify`（unimod id）需一致映射；不一致会导致命中失败或错配。
5. **全局 L:H 比估计**（J5）：在弱信号 PSM 上不稳；已设回退。
6. **度量选择**：实测（§11）选定 **谱角**为主度量、阈值 ~0.6、**分 ion-type**；谱熵/Spearman 备选。
7. **跨文件重标**（实测，§11）：窄窗 DDIA 下重标母离子可能落在另一个 50-Da 采集文件，Phase 2 找重标证据须跨文件考虑（实测 1500 条里 101 条重标母离子超出本 550–600 采集范围）。

---

## 10. 分阶段落地顺序（供实现计划参考）

1. **P0 · sanity gate（4.0）+ PredStore（4.1）+ 可分性/中心化（4.1.5）**：先验证可用性、b/y 对齐、split-aware 可分性与中心化收益，建好 lookup。**gate 通过是后续前提。**
2. **P1 · top-K（4.2）+ I1（4.3）**：地基 + 最打情形 B 的新维度。
3. **P2 · I2（4.4）+ I3（4.5）+ J2（4.6）**：补齐关系/覆盖/污染三特征。
4. **P3 · J5（4.7）+ 性能红利（4.8）**：判据升级与提速（带 config 开关，保守默认）。**Phase 2c 已落地 J5 的「自适应覆盖度」增量列**（`global_lh_ratio`/`pred_coverage_adaptive`，feature_type=0，公式见 §4.7 更正）；`is_signal_present_heavy` 自适应版、性能红利、`feature_type=1/2` 路径留待后续。

每阶段 TDD + 既有测试回归 + 提交。

---

## 11. Phase 1 实测验证（2026-06-09，lib-2th / HeLa SILAC 2Da DDIA）

**Setup**：库 `lib-2th` + `merge_human_ecoli_yeast.fasta` + `modification.ini`；PSM = `hela_2da.json`（自定义 JSON，取**未修饰**子集）；raw = `..._550_600_2Da_Rep1`（原生中心化缓存；另用「profile npz 内存重中心化」近似，两法结果**逐位一致**）；样本 1500 条 PSM。内存峰值 ~0.3–0.9GB（库与 npz 全程 mmap，4.2GB 库 + 2.8GB npz 不进 RAM）。

**覆盖率**：hit=1500 / miss=0（**100%**）→ 查表 / key 规范化 / 修饰处理 / 库内容都正确。

**预测 vs 实测「轻标」相似度中位数（中心化数据；谱角 | 纯cosine）**：

| 切片 | 谱角 | cosine |
|---|---|---|
| ALL | 0.589 | 0.799 |
| 可分(y/K·R) | 0.637 | ~0.84 |
| Y-only | 0.640 | 0.844 |
| **B-only** | **0.818** | **0.960** |

（profile 数据对应为 ALL 0.519 / sep 0.538 / y 0.544 / b 0.778 → 中心化各 +0.07~0.10，y 受益最大。）

**窗口分组（关键）**：split=1391 / co-isolated=8 / undecided=101。在**分窗子集**（b 离子为**纯轻标、零重标污染**）：B-only 0.823 ≫ Y-only 0.643。

### 结论（修正先前理解）

1. **谱库验证有效**：预测真实（b cos 0.96 / y cos 0.84，远离对齐错误时的随机 ~0.19），100% 覆盖，基础设施正确。**这是 Phase 2 的 go 信号。**
2. **可分性是 split-window-aware**：2Da DDIA ~93% 分窗 → 轻重在不同 MS2 → **几乎所有碎片（含 b）可分/可用，不是只用 y**。`is_separable_fragment`（shifted OR split）已正确编码；先前「separable≈y」的口头简化是错的。
3. **b>y 是真实模型精度差**（在分窗、无污染子集中仍成立）→ **先前「SILAC 把 b 抬高」的归因被推翻**。**不要剔除 b**——它是预测最准且可用的碎片。
4. **ALL < b、y 子集** → 预测的 **b:y 整体比例**标定有偏 → I1/gate 度量应**分 ion-type**（b、y 各自算谱角再合，或用对 b:y 缩放鲁棒的度量），不要把 b、y 混进一条向量直接算。
5. **阈值**：可分-中心化谱角 ~0.64（cos ~0.84） → gate 阈值定 **~0.6（谱角）**；0.70 太严。
6. **中心化必要**：profile→centroid 各切片 +0.07~0.10。
7. **y 对齐 = 反向**（`frag_pos = seq_len − ion_num − 1`）经实测验证正确（0.64 vs forward 0.19）。
8. **跨文件重标**：undecided=101 = 重标母离子落在本 550–600 采集范围外 → Phase 2 找重标证据须考虑「重标可能在另一个 50-Da 采集文件」。

> 复用产物：`workspace/..._550_600_2Da_Rep1.centroid.dia.npz`（有效 v3 中心化缓存）。

### 11.1 b/y × light/heavy × 谱角/Spearman 复核（2026-06-09，600 条 PSM，逐根手验）

在同一批数据上把**完整预测 b/y 系列**与实测 light/heavy apex 并排，逐根核对，得到（检出碎片中位）：

| ion / 通道 | 谱角 | **Spearman** |
|---|---|---|
| b / light | 0.804 | 0.714 |
| b / heavy | 0.771 | 0.700 |
| y / light | 0.657 | **0.762** |
| y / heavy | 0.612 | **0.771** |

单峰主导度（max_pred / Σpred 中位）：**b 0.563 vs y 0.266**。

**关键发现（驱动接入 Spearman 到 I1）**：
1. **b 的高 cosine/谱角大半是「b2 单峰主导」的数学虚高**——b 系列被 N 端 b2 一根占据(主导度 0.56)，两个向量「共享一根对齐的主导分量」时 cosine 机械地≈1，与小峰预测准不准无关；y 强度分散(主导度 0.27)，没有这根「免费分」，每根都要对上 → 谱角更低。
2. **按纯排序(Spearman)，y(0.76–0.77)反超 b(0.71)**：模型对 **y 的次序**预测得更准；b 的小尾巴(b2 之后又小又近)排序噪声大，反而拉低 b 的 Spearman。**「b 比 y 准」是磁量级度量造的假象**。
3. **Spearman 对 heavy 噪声更鲁棒**：谱角从 light→heavy 明显下滑(y 0.657→0.612)，Spearman 几乎不降(y 0.762→0.771)。I1 用的正是 heavy 通道 → 接入 Spearman 价值更大。
4. **落地**：`compute_speclib_i1` 已增量输出 `spec_pattern_spearman_b/_y/_`（与谱角并存；每类 ≥3 根，n=2 退化 → NaN；合并=各类均值）。真实 target/trap 区分力仍待 per-feature AUC 实测确认。

---

## 12. Trap 集卫生 / 工具能力边界（2026-06-10）

**原则**：SILAC 轻重标验证有一个**能力边界**。落在边界外的 PSM，本工具**根本无法判别真假**，必须在评估区分力**之前**从 trap 集里**剔除**——否则它们会同时压低 AUC 上限、并污染指标。这不是 speclib 的失败，而是问题定义。

### 12.1 三类"超出工具能力"的 trap（都应剔除）

| 类 | 判据 | 为什么工具无法判别 | 实现 |
|---|---|---|---|
| **1. L0/L1 人源同源** | trap 序列（含 L↔I 同分异构）**出现在完整人源蛋白库**中 | 质谱上和真人源肽**不可分**；很可能是搜索引擎**物种误判** | `spectrum/entrapment_classifier.py`（自带，L0=精确子串 / L1=L↔I 子串）**【已做】** |
| **2. 真污染蛋白** | 蛋白身份属于已知污染物 / spike-in（cRAP、E.coli β-gal 等） | 它是**真实存在并被标记的肽**，工具正确判"真"，本就不该当假阳 | 外部污染物名单按**蛋白身份**剔 **【待做】** |
| **3. 重标出采集窗** | `heavy_out_of_range==1`（重标母离子 m/z 超出本 raw 的采集范围） | 重标通道**没采到** → 轻重标方法**直接失效** | `heavy_out_of_range` flag **【已做】** |
| **4. 无标记位点** | 序列**不含 K/R** | 没有重标位点 → 重版 = 轻版（质量不变）→ 轻重标验证**无定义**（一切平凡地"完美匹配"） | `has_label_site`（序列查 K/R）**【已做】** |

> **第 3 类的多文件注意**：全实验 m/z 分段采（500–550 / 550–600 / 600–650），某肽重标在**本文件**出窗，可能在**隔壁文件**在窗。严格判据应是"重标在**所有 raw**里都没采到"才剔；单 raw 评估时按本 raw 的窗口剔（更保守）。

### 12.2 关键澄清：「真配对」是诊断提示，**不是**过滤判据

- **「真配对」**（轻、重母离子都在且同峰共洗脱，且碎片轻重同峰）= 这条肽**真实存在且被标记**（真肽）。它**只说明"是真肽"，不直接说明"是 target 还是 trap"**：一个被标成 trap 的真肽 = **标注错误**（污染物/同源），而不是"工具识破的假阳"。
- **不能用工具自己的"真配对"信号去过滤 trap**——这是**循环论证**（用工具判"像真肽"去删 trap，会删掉最难的样本、把指标刷虚），且弱配对/部分干扰也可能伪造半个配对。
- 所以**类2 必须用外部客观判据**（污染物名单 / 蛋白身份），不能用 `precursor_pearson` 等工具内信号。

### 12.3 实测分布（pilot, 550_600_Rep1, 155 条 trap）

| 类 | 条数 |
|---|---|
| 类1 人源同源（is_L0） | 52 |
| 类3 重标出窗 | 14 |
| 类2 真配对（仅诊断，未剔） | 45 |
| **踢掉（任一类，有重叠）** | **~77（50%）** |
| **剩"工具该管"的真 trap** | **~78** |

→ **原始 entrapment 集约一半超出工具能力**。剔除类1/3 后，speclib 在干净 trap 上的区分力才有意义（@99% target 留存：speclib ~59% vs 既有 ~46%，见 §11/分析）。

### 12.4 当前实现范围

**本期做类1（L0/L1）+ 类3（重标出窗）+ 类4（无 K/R 标记位点）**，用现成的 `entrapment_classifier` + `heavy_out_of_range` flag + `has_label_site`（序列查 K/R）；**类2（污染物名单）留待后续**。剔除优先级：类1（不可分）> 类4（无标记位点）> 类3（出窗）——越靠前越本质，但三者都剔。

> **类4 的诚实标注**：无 K/R 肽多为蛋白 C 端肽，很少（pilot 干净 trap 里仅 1 条；target 里约 0.4%）；本期只对 trap 侧过滤（target 侧的无 K/R 肽同样无法被本工具验证，属评估范围外，留作后续说明）。
