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
