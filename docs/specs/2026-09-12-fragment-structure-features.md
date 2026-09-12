# 单候选逐电荷、峰去重和共同峰内序列结构特征（Q/D/S v1）

状态：特征提取代码、训练特征组注册、真实 PFB 案例验算和回归测试已完成。后续合并/训练/汇总入口也已实现，见 [五臂验证流水线](2026-09-12-fragment-structure-validation.md)。真实全量提取和五臂训练尚未运行；目前没有新的真实 FPR/FNR 结果。

## 目的和兼容性

保留同一候选的旧特征，额外提供 28 个数值字段：Q 13 个、D 3 个、S 12 个。只使用当前 PSM 的序列、修饰、前体电荷、RT、标记规则和对应谱图；不使用竞争候选、搜索分数、物种、蛋白名、FN 编号或标签计算这些字段。

旧 `y_count` 仍是有有限相关系数的 y 离子数量，旧 Q1A 和相关性仍使用 1+/2+ 合并轨迹。不能把这些旧字段解释为同电荷通过数量或独立实测峰数量。三个新组单独注册，原 `ms1_ms2_no_prediction`、`evidence_all`、`full` 等明确命名的训练臂不自动增加 Q/D/S。

## 数据流和接口

1. `spectrum/spectrum_utils.py::match_peak_targets_ppm` 可选返回匹配到的正强度 centroid 原始下标。输入谱图若重排 m/z，下标仍对应重排前的谱图。零强度峰不构成支持。
2. `spectrum/dia_data.py::xic_ms2_fragment_panel_extract` 每个通道一次提取完整离子面板，按扫描逐次加载谱图，保留 1+/2+、RT、cycle、强度、ppm 和 `(scan_index, centroid_indices)`。身份仅在同一 DIAData/raw 内有效。
3. `pool_fragment_charges` 复用旧合并算术，生成原有特征所需的轨迹；新增字段由 `workflows/fragment_structure.py` 计算。
4. `workflows/single_work.py::single_pair_work` 将旧字段与新字段写入同一特征行。逐离子数组在计算期间驻留内存，本版本不落盘全量逐离子轨迹。

新增功能默认开启，可在 `[general]` 设置 `fragment_structure_features = false` 关闭。关闭时旧流程仍工作，新数值列为缺失并标记 `disabled`。旧第三方 DIA 适配器没有 panel 方法时标记 `unsupported_extractor`，不能从合并轨迹推造逐电荷信息。

本版支持同一 raw 内单候选轻重配对，即 `feature_type=0`。跨 run 配对仍计算旧特征，新字段标记 `cross_run`；不同 raw 的 cycle 和 centroid 下标不能直接比较。

## 通用定义

- 理论可分辨离子沿用 `is_separable_fragment`：轻重位于同一隔离窗时，未发生质量位移的离子不能提供独立轻重证据。隔离窗未知则标记不可用。
- 新证据仅考虑 1+/2+ 且不超过前体电荷。旧合并字段继续沿用原定义。
- 一个 **target** 是 `(b/y, ordinal, charge)`；一个 **ion** 是 `(b/y, ordinal)`，忽略电荷但保留离子类型和位置。
- 同电荷通过条件沿用 Q1A：轻、重最大强度均严格大于 100；apex 相差不超过 1 个采集 cycle；共同 RT 插值后的 Pearson 严格大于 0.5。未定义的相关性不通过。
- 强度面积在这里指提取窗口内采样强度之和，不是 RT 梯形积分。
- 空分母为 NaN。没有采集到 MS2 与已经采集但没有匹配信号是不同状态。峰是否通过不是最终真假判定。

## Q：逐电荷证据（13 列）

以下字段前缀均为 `ms2_charge_`。

| 字段后缀 | 定义 |
| --- | --- |
| `paired_target_count` | 同电荷通过的 target 数 |
| `paired_ion_count` | 至少一个允许电荷通过的 ion 数 |
| `b_paired_ion_count` / `y_paired_ion_count` | 分别统计 b/y 的通过 ion 数 |
| `shifted_paired_ion_count` | 通过且理论轻重中性质量差绝对值 ≥0.001 Da 的 ion 数 |
| `paired_fraction` | 通过 target 数 / 轻侧强度超过门槛的允许 target 数 |
| `pooled_only_fraction` | 合并后通过、但无允许同电荷通过的 ion 数 / 合并后通过的 ion 数 |
| `dominant_mismatch_fraction` | 两侧主导电荷不同的 ion 比例；两侧允许电荷总面积均需为正。平手取较小电荷 |
| `z1_pearson_median` / `z2_pearson_median` | 两侧均超过强度门槛的同电荷 Pearson 中位数；包含未通过 apex/相关门槛者的有限相关系数 |
| `z1_log_lh_mad` / `z2_log_lh_mad` | 通过 target 的 `log2(轻面积/重面积)` 的中位绝对偏差，电荷分别计算；至少 3 个有限值，否则 NaN |
| `paired_effective_points_median` | 每个通过 target 取两侧有效点数的较小值，再取中位数；有效点数为 `(ΣI)²/ΣI²` |

同电荷通过与合并通过不完全嵌套；合并可能增加干扰，所以不能用一个计数简单相减推导另一个计数。

## D：独立实测轨迹（3 列）

对已经同电荷通过的 target，分别构建完整轻、重轨迹的实测峰身份签名。只有两侧在整段提取窗口使用的全部正强度 scan/centroid 集合完全一致，才合为一个等价类。质量相同、容差窗重叠、强度相同，均不足以判定为同一实测峰。

以下字段前缀均为 `ms2_dedup_`。

| 字段后缀 | 定义 |
| --- | --- |
| `paired_trace_count` | 去重后的两侧轨迹等价类数量 |
| `reused_target_fraction` | `(通过 target 数−等价类数)/通过 target 数` |
| `reused_intensity_fraction` | `(去重前面积总和−每类计一次的面积总和)/去重前面积总和`；面积为轻重之和 |

这是严格完全复用的下界描述：只有部分扫描重叠、只共享轻侧、一个 target 匹配峰集合包含另一个集合，都不会被此版合并。不能将该计数解读为已消除所有相关或部分共享的信号。

## S：共同峰内序列结构（12 列）

先完成 D 去重，再保留含 `ordinal≥2` 解释的轨迹类。顺序不能颠倒，例如 `b1+ / b2++` 共用峰的歧义不能因删掉 b1 而消失。只有 b1/y1 的支持不贡献本组序列定位。

每侧以最强采样点为中心取连续半高区间，遇到缺失 cycle 停止，两端各扩半个 cycle。找出轻侧区间共享一个点、重侧区间也共享一个点的最大轨迹集合；每类一票。不会用相邻关系传递连接多个峰组。数量平手时选两个通道内更接近候选 RT 所在 cycle 的交点，再选较早交点。宽峰保留实际宽度；平背景也可能产生宽区间，因此这仍是描述特征。

切点坐标：b_i 对应 i，y_j 对应 `肽长−j`。一个等价类有多个可能切点时，不定位到任一切点。同一实测峰同时可解释为 b/y 且指向同一切点时，可贡献切点覆盖，但不能算成独立 b/y 互补支持，也不贡献各自连续序列长度。

以下字段前缀均为 `ms2_structure_`；“主组”指上述共同峰组。

| 字段后缀 | 定义 |
| --- | --- |
| `main_trace_count` | 主组轨迹类数量 |
| `main_trace_fraction` | 主组类数 / 含 ordinal≥2 解释的类数 |
| `main_shifted_trace_count` | 主组内全部候选解释均有理论轻重质量位移的类数 |
| `main_cut_fraction` | 无歧义切点数 / `(肽长−1)` |
| `main_b_run_fraction` / `main_y_run_fraction` | b/y 各自最长连续支持切点数 / `(肽长−1)` |
| `main_complementary_cut_fraction` | 由不同轨迹类独立 b 和 y 支持的切点数 / `(肽长−1)` |
| `main_longest_gap_fraction` | 无歧义切点与两端 `{0,肽长}` 之间的最长距离 / 肽长；无切点为 1 |
| `main_internal_kr_bracket_fraction` | 内部 K/R 两侧相邻切点均获支持的比例；排除序列首尾位点，只适用于 SILAC；无内部 K/R 为 NaN |
| `main_anchor_offset_cycles` | 主组共同交点到候选 RT 最近采样 cycle 的绝对偏移，轻重两侧取平均；不是 apex 差 |
| `outside_main_intensity_fraction` | 含 ordinal≥2 解释的类中，主组外面积占比，每类计一次 |
| `main_ambiguous_cut_fraction` | 主组中存在多个可能切点的类所占比例 |

S 的计算依赖 Q/D 底层表示，但 S 单组训练只输入 S 汇总列。

## 缺失与状态

每行还有 3 列：`fragment_structure_valid`、`fragment_structure_version=qds_v1`、`fragment_structure_status`。valid 是可计算性标志，status 是原因，二者都不进入正式 Q/D/S 模型特征；version 也是元数据。

不可用原因包括 `disabled`、`cross_run`、`unsupported_extractor`、`missing_window`、`no_fragment_targets`、`no_separable_targets`、`no_ms2_scans`、`missing_peak_identity`、`invalid_acquisition_rows`、`inconsistent_peak_identity`。不可用时 28 列全部 NaN。不可用不删除样本。

正常采集但空信号时 status=`ok`、valid=1，支持数量为 0，无定义的比例/相关性/离散度为 NaN。不能统一把 NaN 替换成“好”或“坏”的特征值；后续训练沿用训练折内预处理。

## 全量提取命令

在仓库根目录执行：

```bash
make 2da-structure-features
```

默认读取 `runs/baseline_2da_clean/config.ini`，保留该配置的输入 JSON、原始谱图、标记、质量容差、提取窗口及其他设置；使用独立 workspace，输出到 `runs/baseline_2da_structure/features.csv`。这要求源 config 指向的完整输入已经存在；该 target 不重新运行 extract_common。

可明确指定来源和输出：

```bash
make 2da-structure-features \
  FRAGMENT_STRUCTURE_CONFIG=/path/to/baseline_2da_clean/config.ini \
  FRAGMENT_STRUCTURE_OUTPUT=runs/baseline_2da_structure_v1
```

源配置的相对输入路径按仓库根目录解释，与原 `main.py` 的根目录启动方式一致。从服务器复制的配置若含服务器绝对路径，需先提供当前机器可用的路径。`PY` 可按原 Makefile 用法指定 Python 环境。

输出还有有效运行 `config.ini`、`extract.log`、`structure_extraction.json` 和 `workspace/`。审计 JSON 记录源配置指纹、运行状态、行数和可计算状态数量。已有 `features.csv` 时拒绝覆盖；失败后若已有部分 CSV，也需选择新的输出目录。

只生成配置而暂不运行：

```bash
python -m tools.extract_fragment_structure --prepare-only \
  --config runs/baseline_2da_clean/config.ini \
  --output-dir runs/baseline_2da_structure
```

该入口只提取特征，没有执行五臂训练，也没有按特征质量筛掉记录。它沿用源配置的完整输入和既有提取后处理；产出的原始行数未必等于冻结 cohort。完成后运行 `make 2da-structure-validation`，按稳定样本身份合并原 manifest，核对冻结真实样本是否齐全，冻结内部协议，再训练和汇总五臂。未匹配的冻结行会报错，不会静默丢弃；新特征不可用的行保留。

## 训练接入与统计边界

| 臂 | `data.feature_arm` |
| --- | --- |
| B | `ms1_ms2_no_prediction` |
| B+Q | `ms1_ms2_charge` |
| B+D | `ms1_ms2_dedup` |
| B+S | `ms1_ms2_structure` |
| B+Q+D+S | `ms1_ms2_qds` |
| 原观察＋预测谱＋Q/D/S | `evidence_all_qds` |

正式训练配置使用 `data.require_complete_arm: true`，防止旧 CSV 缺列时只取列交集。未指定 feature_arm 的自动选列流程可能纳入新增数值列，因此复现实验必须明确 feature_arm。

完整设计见 [开发验证协议](../../analysis/single_peptide_fn_audit_2026_09_12/deep_review/feature_validation_plan.md)。继续使用冻结 `leakage_group_id`、`experiment_outer_fold`；不能跨 Rep 重划。成员阈值由训练侧 OOF 正确 IDs 校准，再按成员多数投票，不对平均 ensemble 分数套用单成员阈值。

存储仍为 `label=1` 正确、`label=0` 错误，模型输出 trust。评估复用 `cv_core.py` 的 `error_truth=1-label`、`error_score=1-trust`；FPR 是正确鉴定被误报比例，FNR 是错误鉴定被漏放比例。未来结果必须标注 `metric_semantics=error_identification_positive_v1`、`positive_class=incorrect_identification`。本实现没有修改评估代码。

## 已完成验证

- 相关回归测试 242 项通过，覆盖原 XIC/Q1A、单候选流程、特征注册、预测谱流程、Phase2 提取接口、配对调度及配置。
- 新增 16 项测试包含：仅跨电荷形成好相关、超过前体电荷、实测峰完全复用、相同数值但不同峰、伪 b/y 互补、内部标记位点、共同峰组不传递连接、宽峰、仅 ordinal1、缺采集与空信号、原 centroid 身份、生产流程旧值相等和输出保护。
- 对 10 条实际 PFB 案例切换新功能开关，共 1,520 个旧字段值完全一致，且原保存的 y_count/Q1A 计数吻合。该检查在直接 `single_pair_work` 默认预测上下文下进行；预测谱集成另由回归测试覆盖。

| 案例 | 原 Q1A 配对 ion 数 | 新同电荷 target 数 | 去重 trace 数 | 主组切点覆盖 |
| --- | ---: | ---: | ---: | ---: |
| FN007 | 4 | 3 | 3 | 1/15 |
| FN018 | 12 | 12 | 11 | 7/9 |
| FN091 | 7 | 7 | 6 | 4/7 |
| FN097 | 20 | 23 | 23 | 15/17 |
| FN102 | 2 | 2 | 2 | 0/22 |

表中 ion/target/trace 单位不同，不能直接用三列相减推算同一计数。FN097 仍保留强结构支持，表明这些字段不会按 FN 标签强制否定候选。FN102 的 b1/y1 两个通过配对不贡献 ordinal≥2 结构支持。

可重跑案例脚本 `analysis/single_peptide_fn_audit_2026_09_12/verify_structure_implementation.py`；它依赖本地 PFB 和既有审计案例缓存。产物为该目录下 `structure_implementation_check/cases.csv` 与 `validation.json`，后者记录代码指纹。它是实现验算，不是训练效果评估，也不证明参考条目逐条生物学正确。
