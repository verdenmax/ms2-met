# 设计文档：H/L 强度比一致性 + apex cycle offset 两类正交特征

> 编写日期：2026-05-26 | 目标：第三轮特征工程，补充对 5Da 宽窗口干扰场景的区分力

---

## 一、动机

现有特征（pearson、apex_delta、cosine、SNR 等）大多衡量轻重标 XIC 之间的形状相似度。
在 5Da 窗口下，干扰肽段的碎片 XIC 也具有真实的洗脱峰形，"形状相似"无法区分真实共洗脱与碰撞匹配。

新增两类与现有维度**正交**的特征：

| 维度 | 物理含义 | 真 PSM 表现 | 假 PSM 表现 |
|------|---------|------------|------------|
| **H/L 强度比一致性** | 同一肽段所有碎片的轻/重比应一致 | 各碎片 log(ratio) 离散度低 | log(ratio) 散乱 |
| **apex 偏离 PSM RT 的 cycle 数** | 真 PSM 强度峰应位于 PSM 报告 RT 附近 | offset 接近 0 | offset 远离 0 |

二者均不依赖轻重标之间的"形状对齐"，而是单边/分布层面的特征。

---

## 二、特征定义

### 2.1 H/L 强度比一致性

#### 收集

在 `single_pair_work` / `multi_batch_work` 的碎片循环中，对每个有效碎片（已通过 `calc_xic_score` 算出 `intensity_ratio = light_total / heavy_total`），收集到按离子类型分组的字典：

```python
fragment_hl_ratios = {"all": [], "b": [], "y": []}
# 当 ion_score["intensity_ratio"] > 0 时:
fragment_hl_ratios[ions_type].append(ion_score["intensity_ratio"])
fragment_hl_ratios["all"].append(ion_score["intensity_ratio"])
```

> 注：`intensity_ratio == 0` 表示某一标缺失或全零强度，不应纳入一致性统计。

#### 计算（对每个 ion_type ∈ {all, b, y}）

```python
log_ratios = np.log10([r for r in ratios if r > 0])
count = len(log_ratios)

if count >= 2:
    std_v = float(np.std(log_ratios))
elif count == 1:
    std_v = float("nan")        # 与 extract_ion_pearson_features 的 Bug #21 处理一致
else:
    std_v = 0.0

if count >= 1:
    med = np.median(log_ratios)
    mad_v = float(np.median(np.abs(log_ratios - med)))
else:
    mad_v = 0.0
```

#### 6 个新特征

| 列名 | 定义 |
|------|------|
| `all_log_hl_ratio_std` | 所有有效碎片 log10(intensity_ratio) 的标准差 |
| `b_log_hl_ratio_std` | b 离子碎片 log10(ratio) 标准差 |
| `y_log_hl_ratio_std` | y 离子碎片 log10(ratio) 标准差 |
| `all_log_hl_ratio_mad` | 所有有效碎片 log10(ratio) 中位绝对偏差 |
| `b_log_hl_ratio_mad` | b 离子 log10(ratio) MAD |
| `y_log_hl_ratio_mad` | y 离子 log10(ratio) MAD |

### 2.2 apex cycle offset

#### Cycle 编号定义

DIA 一个 cycle = 1 个 MS1 + N 个 MS2。Cycle 编号 = **该 MS1 在 `ms1_indexs` 中的位置**。
- MS1 谱图：cycle_idx = ms1_indexs 中的位置（trivial）。
- MS2 谱图：通过 `precursor_scan_ids[ms2_global_idx]` 反查所属 MS1 的 scan_id，再经 `_scan_id_to_index` 拿到该 MS1 的全局 spectrum index，最后 `np.searchsorted(ms1_indexs, ms1_global_idx)` 得到 cycle 编号。

#### XIC 数据结构扩展

`xic_peaks_extreact`（MS1）和 `xic_ms2_peaks_extract`（MS2）返回的结构化数组 dtype 增加一列：

```python
dtype = [("rt", "f8"), ("ppm_error", "f8"), ("intensity", "f8"),
         ("cycle_idx", "i4")]
```

向后兼容：现有代码按 `xic["rt"]` / `xic["intensity"]` / `xic["ppm_error"]` 访问，不受新字段影响。

#### offset 计算

```python
def _calc_cycle_offset(xic, center_rt):
    if len(xic) == 0:
        return 0, 0
    valid_mask = xic["cycle_idx"] >= 0
    if not np.any(valid_mask):
        return 0, 0
    valid_xic = xic[valid_mask]
    # 用 RT 定位中心采样点，然后取它的 cycle_idx
    center_local_idx = int(np.argmin(np.abs(valid_xic["rt"] - center_rt)))
    center_cycle = int(valid_xic["cycle_idx"][center_local_idx])
    # apex 取最大强度处的 cycle_idx（在全 xic 上）
    apex_global_idx = int(np.argmax(xic["intensity"]))
    apex_cycle = int(xic["cycle_idx"][apex_global_idx])
    if apex_cycle < 0:
        return 0, 0
    signed = apex_cycle - center_cycle
    return abs(signed), signed
```

#### `calc_xic_score` 接口扩展

```python
def calc_xic_score(light_xic, heavy_xic, center_rt=None, intensity_threshold=1e-10) -> dict:
    ...
    # 新增字段
    if center_rt is not None:
        l_abs, l_signed = _calc_cycle_offset(light_xic, center_rt)
        h_abs, h_signed = _calc_cycle_offset(heavy_xic, center_rt)
    else:
        l_abs = l_signed = h_abs = h_signed = 0
    result["light_apex_cycle_offset"] = l_abs
    result["light_apex_cycle_offset_signed"] = l_signed
    result["heavy_apex_cycle_offset"] = h_abs
    result["heavy_apex_cycle_offset_signed"] = h_signed
```

`_default_xic_score()` 同步补上四个零字段。

#### 4 个前体级特征

| 列名 | 定义 |
|------|------|
| `precursor_light_apex_cycle_offset` | 轻标前体 apex 偏离 PSM RT 的 cycle 数（绝对值） |
| `precursor_light_apex_cycle_offset_signed` | 同上带符号（负=apex 早于 PSM RT） |
| `precursor_heavy_apex_cycle_offset` | 重标前体 apex 偏离的 cycle 数（绝对值） |
| `precursor_heavy_apex_cycle_offset_signed` | 同上带符号 |

#### 16 个碎片级汇总特征

在碎片循环中收集 4 个列表：
```python
fragment_light_cycle_offsets = []
fragment_light_cycle_offsets_signed = []
fragment_heavy_cycle_offsets = []
fragment_heavy_cycle_offsets_signed = []
```

循环后用扩展的 `extract_ion_numeric_features` 汇总（新增 `_max` 字段），每个列表得到 mean/p50/std/max 共 4 个特征：

| 列名前缀 | 4 统计量 |
|---------|---------|
| `all_light_apex_cycle_offset_{mean,p50,std,max}` | abs offset |
| `all_light_apex_cycle_offset_signed_{mean,p50,std,max}` | signed offset |
| `all_heavy_apex_cycle_offset_{mean,p50,std,max}` | abs offset |
| `all_heavy_apex_cycle_offset_signed_{mean,p50,std,max}` | signed offset |

### 2.3 新增特征总计

**6 + 4 + 16 = 26 个新 CSV 列**

---

## 三、实现改动清单

### 3.1 `spectrum/dia_data.py`

1. 新增私有方法 `_ms2_cycle_idx(self, global_ms2_idx) -> int`：
   ```python
   ms1_scan_id = self.precursor_scan_ids[global_ms2_idx]
   ms1_global_idx = self._scan_id_to_index[ms1_scan_id]
   pos = int(np.searchsorted(self.ms1_indexs, ms1_global_idx))
   if pos < len(self.ms1_indexs) and self.ms1_indexs[pos] == ms1_global_idx:
       return pos
   return -1
   ```

2. 修改 `xic_peaks_extreact`：
   - dtype 增加 `("cycle_idx", "i4")` 字段
   - 收集时填充 `cycle_idx = start_index + i`

3. 修改 `xic_ms2_peaks_extract`：
   - dtype 同上扩展
   - 收集时为每个 selected_global_idx 调用 `_ms2_cycle_idx` 填充

4. 失败/早返回路径（return `np.array([], dtype=dtype)`）同步使用新 dtype。

### 3.2 `workflows/single_work.py`

1. 新增 `_calc_cycle_offset(xic, center_rt)` 辅助函数（见 2.2）
2. `calc_xic_score` 加可选 `center_rt=None` 参数，返回字典加 4 个 `*_apex_cycle_offset*` 字段
3. `_default_xic_score()` 加 4 个零字段
4. `extract_ion_numeric_features` 增加 `_max` 字段（空列表时返回 0.0）
5. `single_pair_work` 和 `multi_batch_work` 同步改动：
   - 前体处调用 `calc_xic_score(light_xic, heavy_xic, center_rt=psm._rt)`
   - 前体特征字典中添加 4 个 `precursor_*_apex_cycle_offset*` 字段（空 XIC 路径同样填 0）
   - 新增 4 个 fragment 收集列表 + 1 个 fragment_hl_ratios 分组字典
   - 碎片循环中调用 `calc_xic_score(light_xic, heavy_xic, center_rt=psm._rt)`，收集每个值
   - 循环后：
     - 用 `extract_ion_numeric_features` 汇总 4 个 cycle_offset 列表 → 16 个特征
     - 用新增辅助函数 `_calc_hl_ratio_consistency(ratios)` 计算 6 个 H/L 一致性特征

### 3.3 `tests/test_single_work_numerics.py`

1. `test_xic_dtype_has_cycle_idx`
2. `test_ms1_xic_cycle_idx_is_position_in_ms1_indexs`
3. `test_ms2_xic_cycle_idx_matches_owning_ms1`（fake DIAData，含两个相邻 MS2 同一 cycle 的场景）
4. `test_calc_cycle_offset_handles_empty_xic`
5. `test_calc_cycle_offset_signed_direction`
6. `test_calc_xic_score_emits_cycle_offset_when_center_rt_provided`
7. `test_calc_xic_score_omits_cycle_offset_default_zero`（不传 center_rt 时新字段返回 0，向后兼容）
8. `test_hl_ratio_std_excludes_zero_ratios`
9. `test_hl_ratio_std_returns_nan_for_single_element`
10. `test_extract_ion_numeric_features_emits_max`

---

## 四、向后兼容性

| 接口 | 兼容性说明 |
|------|----------|
| `calc_xic_score(light, heavy)` 旧签名 | 完全兼容；不传 center_rt 时新字段返回 0 |
| XIC 结构化数组按字段访问 | 完全兼容；新增的 `cycle_idx` 字段对现有代码透明 |
| `extract_ion_numeric_features` 旧调用 | 完全兼容；只新增一个 key，旧消费者不读 `_max` 也无影响 |
| `_default_xic_score()` 增字段 | 完全兼容；只是返回字典多了 4 个零项 |
| DIAData npz 缓存格式 | 不变（cycle_idx 不持久化，XIC 时动态计算） |

---

## 五、风险

| 风险 | 评估 | 应对 |
|------|------|------|
| 碎片级 cycle_offset signed 的 `max` 物理意义弱（取的是数值最大而非绝对最大） | 低 | 仍然有信息（"最延迟"的碎片是谁），LightGBM 可自动处理冗余 |
| MS2 cycle_idx 反查在窗口重叠/异常数据下返回 -1 | 极少发生 | `_calc_cycle_offset` 跳过 -1 项；若全无效则返回 (0, 0) |
| `precursor_scan_ids` 在某些 mzML 解析后可能为空 | 低 | 现有代码在 MS2 提取中已隐式依赖该字段；若为空则现有 XIC 提取本身就会失败 |
| 26 个新特征可能与既有特征共线（如 mean ↔ p50） | 中 | LightGBM 鲁棒；后续可用 SHAP 分析裁剪冗余特征 |
| H/L 一致性对 ratio≈0 的边界敏感 | 低 | 已用 `> 0` 严格过滤，log10 不会出现 -inf |

---

## 六、不在本次范围内

- 不改 `extract_ion_pearson_features` 的现有字段
- 不引入新的 fragment 分组（继续沿用 all/b/y）
- 不改 q1a_helpers
- 不改 npz 缓存格式
- 不为 H/L 一致性的"前体↔碎片差异"额外建特征（之前已经评估过：碎片间离散度信息已足够）
