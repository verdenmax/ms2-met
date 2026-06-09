# workflows — 细节

## 总体数据流（`pair_flow` → `single_work`）

`main.py` → `PairFlow.run()`：

1. `load()`：用 `LightResultManager` 读出 `psm_info` 列表，用 `DataManager` 准备 raw 数据访问。
2. `distribute()` 第一阶段：对每个 `raw_path_N` 用进程池调 `data_to_npz`，把 DIA 数据写成 `<name>.dia.npz`，得到 `name → shared_path` 映射。各 worker 用 `DIAData.load_from_file(..., use_mmap=True)` 内存映射加载，物理内存共享、零拷贝。
3. `distribute()` 第二阶段：把 `psm_info` 按 `PSMInfo.get_key` 分组（同序列/电荷/…的重复样本聚到一起），按 `feature_type` 生成任务，按 `shared_path` 分桶，每桶切成 `BATCH_SIZE=5000` 的 chunk 提交进程池。
4. 各批 worker 调 `single_pair_work` 或 `multi_batch_work` 算出特征 dict，拼上元数据与 `label`，`as_completed` 收集，`pd.DataFrame` 落盘 `result_file`。

### feature_type 三种模式（`distribute` / flow_utils 批函数）

| type | 配对方式 | 批函数 | 负例构造 |
|---|---|---|---|
| 0 | 单文件，按 `shared1` 分组，每 PSM 自身推 heavy | `process_batch_single` → `single_pair_work` | label 来自 `psm._label_type`（上游打标，**现行：陷阱库 entrapment 负例**） |
| 1 | 双文件，重复样本两两组合 | `process_batch_pair` → `multi_batch_work` | 负例把 `psm2._rt += 10`（人为错位，**已弃用**） |
| 2 | 双文件，同上 | `process_batch_pair_shuffle` → `multi_batch_work` | 负例对序列做 `sequence_controlled_shuffle`（in-silico） |

- **⚠️ DEPRECATED（type 1，RT+10 in-silico 负例）**：固定 +10 偏移到空 XIC 区域时，全零特征会成为 label proxy（模型学到“全零⇒负例”而非真实信号）。现行流程**直接使用陷阱库(entrapment)作为负例**（type 0，label 由 `entrapment_classifier` 上游打标）。RT+10 路径（`process_batch_pair` / `process_psm_pair_shared` / `PairFlow._process_group` 的负例半段）仅为兼容旧配置保留，不应用于新实验。
- type 0：`_make_result_row_single` 把 `_label_type`（"positive"/"negative"）映射成 1/0；若为 `None` 直接抛 `ValueError`（避免 NaN 标签让 LightGBM 训练崩溃）。
- type 2 的 shuffle 用 `random_seed`（缺省 42）+ `zlib.crc32(sequence)` 生成每条 PSM 唯一种子；用 crc32 而非内置 `hash()`，因为 `hash()` 受 `PYTHONHASHSEED` 随机化会破坏可复现性。
- `PairFlow.multi_handle` / `_process_group` 是单进程内的等价路径（负例用 `copy.copy(psm)` 偏移 `_rt`，避免就地累加 +10/+20；同属已弃用的 RT+10 路径）。

## DIA 缓存校验（`flow_utils.data_to_npz`）

- 已存在 `.dia.npz` 时先 `DIAData.validate_cache_params` 校验 centroid 参数（轻量 mmap 标量读，不加载数组）；与当前 config 不匹配则删除重建。否则改 centroid 参数将无效。

## 特征提取流程（`single_pair_work` / `multi_batch_work`）

两函数结构对称，输出**同一 schema**（schema parity 是硬约束，type 0/1/2 必须列对齐）。差异：单文件版从 `psm.get_heavy_info(HeavyType.SILAC)` 在同一 `DIAData` 内推出 heavy 母离子与碎片；双文件版直接用两个 PSM、两个 DIAData。

每条 PSM 的特征分几大块：

1. **母离子 XIC**：`xic_peaks_extreact` 取轻/重标母离子 XIC → `calc_xic_score` 给 19 个 `precursor_*` 特征；空对（任一 XIC 空或全零）走默认全零分支并置 `precursor_xic_empty=1`。
2. **同位素 + 质量校验**：在 heavy M0 apex RT 处插值取 M0/M1/M2 强度，与 `get_theoretical_isotope_ratios` 做 cosine → `isotope_correlation`；apex 处 ppm → `mass_shift_error`。同位素间距 `1.003355 / charge`。
3. **碎片离子循环**：遍历 b/y 理论碎片，逐对取 MS2 XIC（`xic_ms2_peaks_extract`），`calc_xic_score` 打分后累计到 `pearsons_map`（b/y/all）与十余个碎片级列表；同时喂给 `Q1aAccumulator`。
4. **碎片聚合**：`extract_ion_pearson_features`（分位数/均值/std/high_ratio）、`extract_ion_numeric_features`（mean/p50/std/max）汇总各列表；强度加权 `frag_corr_weighted`；H/L 比一致性 `_calc_hl_ratio_consistency`。
5. **序列/窗口/Q1a**：`kr_count`、`modification_count`、`total_silac_shift`、`window_width`、`precursor_centering`、`heavy_in_raw`，最后 merge `q1a_acc.compute_features()`。

### 碎片跳过语义（边界设计）

- 双文件版（`multi_batch_work`）：只对"两 XIC 任一为空"计 `fragment_xic_empty_count`；`fragment_heavy_absent_count`/`fragment_same_mass_count` 恒为 0（仅为 schema parity 占位）。
- 单文件版（`single_pair_work`）：三类**正交**跳过计数 —— heavy 不在 raw 窗（`fragment_heavy_absent_count`）、同 MS2 窗且无 SILAC 位移（`fragment_same_mass_count`）、抽出后 XIC 空（`fragment_xic_empty_count`），给模型无歧义信号。
- XIC 空时对**所有**碎片级列表补 0（`fragment_hl_ratios` 例外，只保留真实 heavy>0 且 light>0 的比值），保证聚合分母与 `valid_fragment_ions_num` 一致。
- `matched_intensity_percent` 分母（`intensitys_map["all"] = last_light_all + last_heavy_all`）是 per-PSM 常量，在循环外赋值；早期在循环内累加会被乘以碎片数。

## `calc_xic_score` 关键算法 / 边界

- 入口先按 rt 排序（`np.interp` 要求 xp 单调；多路复用 DIA 扫描序不保证）。
- 空对 / 全零 → `_default_xic_score()`（全零）。这避免 `np.argmax` 在全零上返回 0 而伪装成"完美共洗脱"。
- Pearson：统一到 100 点公共 RT 网格插值；任一近零、std<1e-10、或 scipy 返回 NaN（`ConstantInputWarning`）一律置 0。
- 还产出 cosine、`apex_delta`（含 signed，因负例总把 heavy +10，无符号无法区分方向）、`intensity_ratio=light/heavy`、SNR（p25 噪声底，封顶 1000）、FWHM 宽度比、对称性、`base_to_apex_ratio`、`apex_monotonicity`、`n_peaks`（prominence≥0.3·apex）、`smoothness`（按二阶差分项数归一，跨 window 可比）、`*_apex_cycle_offset`（abs/signed，apex 相对 center_rt 的周期偏移）。

## Q1a 碎片配对召回（`q1a_helpers`）

- 衡量**可分离**理论碎片中、同时有可信 light 与 heavy 信号的比例（spec §4.2）。每个 TP 是一份"light 鉴定正确"的独立物理证据。
- **可分离** `is_separable_fragment`：碎片带 K/R 被 SILAC 位移（light≠heavy），或 DIA 窗已知被分裂（`is_split_window` 返回 `True`）；窗口未知（`None`）不当作分裂。
- **light present**：XIC 峰强 > `intensity_floor`(100)。**heavy present** 需三条件全真：强度过阈、apex RT 差 < `0.3 * light_peak_width`、共网格 Pearson > 0.5。
- `Q1aAccumulator` 按 `(机制, 离子型, TP/FN)` 分桶；`compute_features` 给 11 个特征。`q1a_recall` 等整体/分机制召回在桶 < `MIN_VALID_TOTAL=3` 时为 NaN；共隔离（非 split）时 `q1a_recall_unshifted_separable` 恒 NaN（该桶按构造为空）。

## 鲁棒性 / 坑

- 批函数捕获每条 PSM 异常计入 `n_errors`，不中断整批；`distribute` 汇总错误率，>1% 告警。
- 进程池 `BrokenProcessPool`（常见 OOM）时提前 break，仍写出已完成部分并落 `*.PARTIAL_INCOMPLETE` 标记，供下游识别 CSV 不完整。
- 进程池 `max_tasks_per_child=4` 限制单 worker 任务数以控内存；out-of-window XIC 请求按 worker 汇总记日志（不再逐次告警）。

## 谱库预测强度特征（Phase 1 基础，未接入主流程）

把 pFind 谱库的**预测碎片强度**路由进轻重关系特征（设计见 `docs/specs/2026-06-08-speclib-predicted-intensity-features-design.md`）。Phase 1 只落地可复用基础 + 前置 gate，**不**改 `result.csv`：

- **度量**（`pred_features`）：谱角（对稀疏强度向量比 Pearson 稳）、Spearman（抗模型绝对强度偏差）、预测加权 Pearson。所有度量对退化/**非有限**输入返回 NaN（与各兄弟函数一致），避免把“未定义”混成“真实低相似”。
- **top-K**（`select_topk_separable`）：在**可分**碎片里取预测最强 K 个，驱动后续碎片级特征；丢弃非有限预测强度。
- **I1**（`i1_pattern_features`）：以 L 的预测相对强度构造“预测重标谱”，与实测重标比（谱角/Spearman），并出预测加权 corr(obs_L,obs_H)。直击情形 B（干扰肽段复现不了 L 的预测强度模式）。
- **lookup**（`build_pred_store`）：一遍流式扫库、只留被鉴定肽段；`(seq,mods,charge)` 与碎片 `(ion_type,frag_pos,frag_charge)` 经规范化键对齐；覆盖率 hit/miss 记日志。
- **前置 sanity gate**（`tools/speclib_sanity.py`）：在高置信轻标 PSM 上比“预测 vs 实测轻标”相似度分布；中位过阈才放行 Phase 2。b/y↔frag_pos 对齐约定为 `frag_pos = ion_num-1`，由该 gate 验证。

## 谱库 I1 特征接入 feature_type=0（Phase 2a）

把 Phase 1 地基接入主流程的**单流程路径**（设计见 `docs/specs/2026-06-08-...-design.md` v1.2 §4.1–§4.3/§7）：

- `PairFlow.distribute()` 启动时 `_build_pred_store()` 一遍流式扫库建 `PredStore`（仅当 `[speclib] speclib_dir` 配置；记覆盖率 hit/miss）。`_build_raw_tasks` 给 `feature_type=0` 的每个任务 dict 附该 PSM 的 `pred_frags`（小字典，随任务 pickle 到 worker；**不**把整个 PredStore 传进 worker）。
- `single_pair_work` 在既有碎片循环里收集**可分**碎片记录（heavy 不在 raw、或同窗且无 SILAC 位移的碎片在更早处已 `continue`，故记录的天然是可分碎片；空 heavy 的可分碎片也记 `heavy_apex=0`）。`return` 前经 `compute_speclib_i1` 产出 `spec_pattern_SA_b/_y/_SA/_LH_consistency` + `has_lib_pred` + `psm_is_split_window`（`check_in_same_ms2` 取反）+ `heavy_out_of_range`（`check_in_raw` 取反）。
- **增量旁路**：`speclib_dir` 空 → 不建 PredStore → 任务 dict 无 `pred_frags` 键 → `single_pair_work` 不出任何新列（schema 与现状一致）。
- 度量**按 ion-type 分开**（§7 实测：预测 b:y 整体比例标定有偏，混算会拖低）。
- Phase 2b：I2/I3/J2/J5 + `feature_type=1/2` 路径 + 提速开关。

### I2/I3/J2 接入（Phase 2b）

复用 Phase 2a 收集的 `speclib_frag_records`（无需新增 XIC），`single_pair_work` 在 I1 之后再合并 `compute_speclib_i2_i3_j2`：
- **划分**：`cands`=可分且库有预测的碎片，`F`=cands 按预测强度 top-K；`W`=可分但库**未**预测的碎片；`present` = `heavy_apex > pred_presence_floor`。
- **I2**（H/L 比一致性）`pred_hl_ratio_cv`/`pred_hl_ratio_mad`：F 上 log10(H/L) 的预测加权 std 与 MAD（与既有 `*_log_lh_ratio_*` 区分：后者在全碎片，前者在预测可靠的 F）。
- **I3**（预测覆盖度）`pred_coverage`/`pred_coverage_wpred`：F 中重标「存在」的（加权）占比；另出 `n_both_present`/`pred_both_present_fraction`：F 中轻重**都有**信号（两端 apex 均 > floor）的碎片数与占比，捕捉「真肽两条链都该在」。
- **J2**（意外峰污染）`unexpected_heavy_fraction`/`unexpected_heavy_intensity_ratio`：库未预测的碎片上冒出重标信号的占比/强度比——情形 B 的反面证据。
- 仍是增量旁路（speclib 关闭则不出列）；Phase 2c：J5 自适应判据、`feature_type=1/2`、提速。

### J5 自适应覆盖度接入（Phase 2c）

`single_pair_work` 在 I1 / I2-I3-J2 之后再合并 `compute_speclib_adaptive`，复用同一批可分碎片记录，新增 `global_lh_ratio`（F 上 `heavy/light` 中位数）与 `pred_coverage_adaptive`（F 中 light>0 且 `heavy ≥ α·light·global_lh` 的占比；`α`=`pred_signal_alpha`，缺省 0.2）。这是比固定 floor 更物理的「该碎片是否如预期出现重标」判据，作**增量列**与既有固定-floor 的 I3/J2 并存。公式按 spec §4.7 更正（期望=`light·glh`，不重复乘预测相对强度）。
