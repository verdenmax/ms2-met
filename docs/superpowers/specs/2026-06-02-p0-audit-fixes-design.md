# P0 审计修复设计

**日期**：2026-06-02
**背景**：2026-06-02 对仓库进行了 5 路并行 sub-agent 深度审查，发现 9 项 P0 级问题。本次仅修复 7 项明确的 bug（#1-#7），#8（负样本设计）和 #9（CHEAVY/NHEAVY 修饰）属设计/特性工作，留待 P1 单独处理。
**审计原始报告**：`~/.copilot/session-state/313ffd92-883d-43bd-bda5-27d53504e24c/files/audit-*.txt`

## 范围

本次只动以下文件，且只修复审计明确列出的 bug，不做额外重构、不引入新特性。

| 修复 | 主要文件 | 性质 |
|---|---|---|
| #1 | `PROJECT_INFO.md` | 文档常量错误 |
| #2 | `spectrum/dia_data.py` | API 替换（`logging.warn` → `logging.warning`） |
| #3 | `spectrum/dia_data.py` | 缺陷修复 + 缓存版本 bump |
| #4 | `workflows/single_work.py` | 数值/算法修复 |
| #5+#6 | `spectrum/dia_data.py` | 数值/算法修复 |
| #7 | `tools/eval_baseline.py`, `tools/eval_feature_ablation.py` | 评估指标防泄漏 + 一致化（含 P1 #26） |

## 各项修复设计

### #1 — PROJECT_INFO.md 文档常量错误

- L281、L350：把 `1.00728` 改为 `1.003355`（¹³C-¹²C 中性质量差）。`1.00728` 是质子质量，不是 ¹³C-¹²C delta。
- L282、L351：把 `0.997036` 统一为 `0.997035`，与 `spectrum/psm_info.py:12` `MASS_DELTA_N15_N14 = 0.997035` 一致。

**影响**：仅本地 scratch 文档（`.gitignore` 中），无代码影响。`PROJECT_INFO.md` 不会进 git。

### #2 — `logging.warn` → `logging.warning`

`spectrum/dia_data.py` 第 524、525、703、704、711 行使用 `logging.warn`，在 Python 3.12+ 已被移除（运行时 `AttributeError`）。当前环境用 Python 3.14。

- 改动：5 处 `logging.warn` → `logging.warning`，参数不变。
- 防回归测试：新增 `tests/test_no_deprecated_logging.py`，在仓库内 `grep` 任何 `logging\.warn[^i]` 不应有命中。

### #3 — `_scan_id_to_index` sentinel 修复

**问题**：当前 `_scan_id_to_index` 用 `np.zeros` 初始化。当 MS2 引用一个被 pParse / ProteoWizard 过滤掉的 MS1 scan_id 时，查找返回 0，被错误归到 0 号谱图（通常是第一个 MS1）。`_ms2_cycle_idx` 和 `get_spectrum` 都受此影响，导致下游 `cycle_idx` 错误归类。

**改动**：

1. `spectrum/dia_data.py:292`：
   ```python
   self._scan_id_to_index = np.full(scan_id_table_size, -1, dtype=np.int64)
   ```
2. `get_spectrum(scan_id)` (`dia_data.py:619`)：保留现有 OOB 检查，新增 unmapped 检查：
   ```python
   index = int(self._scan_id_to_index[scan_id])
   if index < 0:
       raise KeyError(f"scan_id {scan_id} 在该 mzML 中未映射")
   return self.get_spectrum_by_index(index)
   ```
3. `_ms2_cycle_idx` (`dia_data.py:612`)：在 `searchsorted` 之前增加：
   ```python
   ms1_global_idx = int(self._scan_id_to_index[ms1_scan_id])
   if ms1_global_idx < 0:
       return -1
   ```
4. **缓存格式 bump**：`_format_version=2` → `=3`，沿用 centroiding spec 已建立的拒绝旧缓存模式：
   - `dia_data.py:168`：`'_format_version': np.int32(3)`
   - `_check_format_version` 仅接受 v=3；v=2 或缺字段→`ValueError`，错误信息提示用户删除 `.dia.npz` 重生成。
   - 同步更新 `docs/specs/2026-06-01-mzml-centroiding-on-load.md` 末尾追加一句说明 v3 的语义差异（_scan_id_to_index 用 -1 sentinel）。

**新增测试** `tests/test_dia_data_scan_id_sentinel.py`：
- `test_unmapped_scan_id_raises_key_error`：合成一个 DIAData，访问从未赋值的 scan_id 抛 `KeyError`。
- `test_orphan_ms2_cycle_idx_returns_minus_one`：合成 precursor_scan_ids 指向 unmapped scan_id → `_ms2_cycle_idx` 返回 -1。
- `test_v2_cache_rejected`：写一个 v=2 的 npz，`load_from_file` 抛 `ValueError`。

**影响**：所有现有 `workspace/*.dia.npz` 缓存失效。需在 commit message 显著标注；并向 PROJECT_INFO/README 加一行清理指引。

### #4 — `calc_xic_score` 100-point 过采样

**问题**：`single_work.py:1169` 固定 `np.linspace(rt_start, rt_end, 100)`，对典型 7-13 cycle 的稀疏 SILAC XIC 过采样 ~10×。大量插值点落在零尾区域 → Pearson/cosine 被系统性抬高。`q1a_helpers.py:110` 已用 cap 公式，但本路径未跟进。

**改动**：
```python
n_points = min(100, max(len(light_xic), len(heavy_xic), 10))
common_rt = np.linspace(rt_start, rt_end, n_points)
```
（与 `q1a_helpers.py:110` 完全一致的公式。）

**新增/扩展测试**：在 `tests/test_single_work_numerics.py` 增加：

- `test_sparse_xic_pearson_not_inflated_by_grid`：构造两路 7 点稀疏 XIC（同高斯型，apex 对齐），运行 `calc_xic_score`，断言返回的 pearson **比同一输入用旧 100-point linspace 复跑的 pearson 严格小**（消除上偏的方向性测试）。
- `test_grid_size_caps_at_actual_sample_count`：mock `numpy.linspace`，断言 `n_points` 参数等于 `min(100, max(len_l, len_h, 10))`，不再固定 100。

**影响**：`precursor_pearson`、`b/y/all_pearson_*` 以及 `precursor_cosine`、`all_cosine_*` 数值会下降（消除偏差）。所有依赖此特征的训练模型需重训。本次只动算法，不动特征列名。

### #5 + #6 — `xic_ms2_peaks_extract` ppm_error 语义统一

**问题**：
- (#5) 未匹配时 `ppm_error = 0.0`，与 MS1 路径的 NaN 约定不一致，导致下游 `nanmean` 把 0 当作真实测量值参与平均，`all_mz_err_*` 系统性下偏。
- (#6) 当 charge=1 和 charge=2 都匹配时，`ppm_error += ...` 是求和而非平均，物理无意义。

**改动**（合并修，因二者在同一段循环内）：`dia_data.py:751-768` 改写为：
```python
ppm_errors: list[float] = []
ppm_weights: list[float] = []
for charge in range(1, 3):
    theo_mz = (ions_mass + charge * protonmass) / charge
    err, intens = match_peak_ppm(mz_arr, intensity_arr, theo_mz, mass_tol_ppm)
    if not np.isnan(err) and intens > 0:
        ppm_errors.append(float(err))
        ppm_weights.append(float(intens))
    match_intensity += intens

ppm_error = (float(np.average(ppm_errors, weights=ppm_weights))
             if ppm_errors else float("nan"))
```
- `match_intensity` 仍跨 charge 求和（语义不变：总匹配强度）。
- `ppm_error` 改为对成功匹配的电荷做强度加权平均；全无匹配时为 NaN（与 MS1 路径一致）。

**下游兼容性确认**：
- `single_work.calc_xic_score` 已用 `nanmean` 并对 `all(isnan)` 兜底为 0（line 956-959）。无需改动。
- `apex_idx` 索引处的 `mass_shift_error = heavy_xic["ppm_error"][apex_idx]`（line 132/443）：如果 apex 这个 cycle 没匹配到，原本是 0.0，现在是 NaN。需要在 `mass_shift_error` 落库前用 `float(np.nan_to_num(..., nan=0.0))` 或显式 NaN 落库——保留 NaN 更诚实但下游 ML 流水线要支持。**决策**：保留 NaN 落库（HistGradientBoosting 原生支持 NaN），与 `calc_xic_score` 内部约定一致；不做 nan_to_num。

**新增测试** `tests/test_xic_ppm_error_semantics.py`：
- `test_no_match_returns_nan_ppm_error`：合成 MS2 谱图无任何峰命中目标 m/z → 结构化数组的 `ppm_error` 字段为 NaN。
- `test_both_charges_match_returns_weighted_average`：合成 charge=1 与 charge=2 都各有一个匹配峰，验证返回值是强度加权平均，不是和。
- `test_one_charge_match_returns_that_value`：只 charge=1 匹配 → 返回 charge=1 的 ppm 值。

**影响**：`all_mz_err_*` 数值会变（消除 0 偏），模型需重训。

### #7 — Eval CV 防泄漏 + META 一致化

**问题**：
- `tools/eval_baseline.py:147` 和 `tools/eval_feature_ablation.py:74` 都用 `StratifiedKFold(shuffle=True)` 做行级 CV。同一 sequence 的多个 PSM 可同时落入 train 和 val 折 → 模型记忆肽段身份特征（`sequence_len`、`kr_count`、`total_silac_shift` 等）→ AUC 高估。
- `eval_baseline.py:36-40` 的 `META_COLUMNS` 含 `sequence_len`（被当 meta 排除），但 `eval_feature_ablation.py:50-54` 的 `ID_COLUMNS` 不含 `sequence_len`（被当特征）→ 两个评估器"全特征"基线不可比。

**改动**：
1. 两个文件的 K-fold：`from sklearn.model_selection import StratifiedGroupKFold`；CV 对象用 `StratifiedGroupKFold(n_splits=5, shuffle=True, random_state=...)`；`.split(X, y, groups=df["sequence"].values)`。
2. 抽公共常量到 `tools/_eval_common.py`（新文件）：
   ```python
   META_COLUMNS = {"sequence", "charge", "raw_title1", "raw_title2",
                   "protein_names", "label", "label_type",
                   "precursor_mz", "sequence_len"}
   ```
   两脚本都 `from tools._eval_common import META_COLUMNS`。`eval_feature_ablation.ID_COLUMNS` 删除，直接使用 `META_COLUMNS`。

**新增/扩展测试**：
- `tests/test_eval_baseline_internals.py` 增加：
  - `test_stratified_group_kfold_no_sequence_leakage`：构造小数据集（同 sequence 出现 3 次，不同 label），断言所有 fold 的 train/val sequence 集合不相交。
  - `test_meta_columns_consistent_between_evaluators`：导入两个模块的 META，断言相等。

**影响**：所有历史 AUC/AUPRC/MCC 数值会下降（消除高估）。这是修正偏差，应在 commit message 显著注明"先前评估指标乐观"。

## 全局执行约束

- **TDD**：每项先写失败测试（红）→ 实现（绿）→ refactor。
- **每项独立 commit**：commit message 引用本文件路径与 P0 编号；commit trailer 含 Copilot 协作者。
- **文档同步**：
  - `PROJECT_INFO.md` §11 "Bug 记录" 追加本次修复的 6 个 entry（#1 文档自修；#2-#7 共 6 项代码修复）。
  - 缓存清理指引追加到 PROJECT_INFO.md 顶部"运行须知"段：v3 缓存需清空 `workspace/*.dia.npz`。
- **不做额外重构**：所有已知 P1/P2 项除 #26（已纳入 #7）外暂不动。
- **不引入新依赖**：`StratifiedGroupKFold` 在 sklearn ≥1.0 已稳定，requirements.txt 不动。

## 风险与不在范围

- **特征值变化**：#4、#5、#6 都会改变历史输出 CSV 的数值。**不**做兼容性 shim 或双轨开关——这是 bug 修复，不是 feature flag。
- **缓存失效**：#3 强制 v3。**不**做向后兼容（参考已有 centroiding spec 同样做法）。
- **#8 负样本 RT shift 设计** 与 **#9 CHEAVY/NHEAVY 修饰支持** 留待 P1 单独立项。

## 验收标准

1. `pytest tests/` 全绿。
2. 7 个 commit 干净分离，每个 commit 单测覆盖其改动。
3. 删除 `workspace/*.dia.npz` 后端到端跑通 `make run`（如有可执行的小型 fixture）。
4. 新增/修改的测试至少包含本设计中列出的全部新测试用例。
