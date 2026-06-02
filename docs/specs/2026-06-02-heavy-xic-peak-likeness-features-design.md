# 设计文档：重标 XIC 峰形质量（peak-likeness）特征

> 编写日期：2026-06-02 | 目标：补齐 SILAC 验证的"峰形质量"维度，针对 5Da 宽窗口干扰场景

---

## 一、动机

现有重标 XIC 峰形特征只覆盖 3 个维度：

| 已有 | 衡量的事 |
|------|---------|
| `precursor_snr` | 峰是否突出 |
| `precursor_peak_symmetry` | 左右面积是否平衡 |
| `precursor_peak_width_ratio` | H/L 峰宽是否一致 |

**缺失的核心维度**：

| 缺失维度 | 物理意义 |
|---------|---------|
| 边缘是否衰减 | 真山峰两边趋于 0，平台/噪声两边仍很高 |
| apex 两侧是否单调 | 真山峰两侧严格上升/下降，干扰是锯齿 |
| 是否单峰 | 真山峰只有一个 apex，共洗脱叠加产生多峰 |
| 曲线是否平滑 | 真信号连续过渡，噪声/单点尖刺不平滑 |

在 5Da 宽窗口下，干扰肽段的真实碎片峰常常被错误匹配，但其峰形质量与真 SILAC 对的峰形质量在以上 4 个维度上有可观差异。这是与现有 H/L 共洗脱、apex_cycle_offset、pearson/cosine 等维度**正交**的新信号源。

## 二、特征定义

### 设计哲学：只算 heavy XIC

与现有 `_calc_snr` / `_calc_symmetry` 一致，所有 4 个新特征只在 **heavy XIC** 上计算。理由：

1. light 的位置是 PSM 报告的 RT，light 峰几乎总是真信号
2. heavy 是被预测出来的；假 PSM 的 heavy 是巧合匹配的
3. heavy 的峰形质量 = 这次匹配是否真的最直接信号

### 2.1 `_calc_base_to_apex_ratio(intensity)`

```python
def _calc_base_to_apex_ratio(intensity):
    """边缘平均强度 / apex 强度。

    真山峰两侧衰减到接近 0 → 接近 0。
    平台 / 背景 / 多峰叠加 → 接近 1。
    """
    if len(intensity) < 3:
        return 0.0
    apex = float(np.max(intensity))
    if apex <= 0:
        return 0.0
    base = (float(intensity[0]) + float(intensity[-1])) / 2
    return base / apex
```

- **范围**：`[0, 1]`（apex = max 保证 ≤ 1）
- **真 PSM**：< 0.2
- **假 PSM / 平台**：0.5+

### 2.2 `_calc_apex_monotonicity(intensity)`

```python
def _calc_apex_monotonicity(intensity):
    """apex 两侧"应该单调"的比例。

    左半应严格上升；右半应严格下降。
    返回 1 - 违反比例 ∈ [0, 1]。
    """
    if len(intensity) < 3:
        return 0.0
    apex_idx = int(np.argmax(intensity))
    left = intensity[:apex_idx + 1]
    right = intensity[apex_idx:]
    if len(left) < 2 and len(right) < 2:
        return 0.0
    left_viol = int(np.sum(np.diff(left) < 0))
    right_viol = int(np.sum(np.diff(right) > 0))
    total_pairs = max(len(intensity) - 1, 1)
    return 1.0 - (left_viol + right_viol) / total_pairs
```

- **范围**：`[0, 1]`
- **真 PSM**：> 0.8
- **假 PSM / 锯齿**：0.3~0.7
- **`right = intensity[apex_idx:]` 包含 apex** —— apex 在边缘时确保两半至少一侧有 2 元素可计算

### 2.3 `_calc_n_peaks(intensity, prominence_frac=0.3)`

```python
def _calc_n_peaks(intensity, prominence_frac=0.3):
    """检测局部极大值数量。

    真山峰 = 1；共洗脱叠加 / 多肽段干扰 = 2+。
    用 scipy.signal.find_peaks 的 prominence 过滤抑制小波动。
    """
    if len(intensity) < 3:
        return 0
    from scipy.signal import find_peaks
    max_int = float(np.max(intensity))
    if max_int <= 0:
        return 0
    peaks, _ = find_peaks(intensity, prominence=max_int * prominence_frac)
    return int(len(peaks))
```

- **范围**：整数 ≥ 0
- **真 PSM**：1
- **共洗脱**：2+
- **`prominence=max*0.3`** 阈值：过滤 apex 30% 以下的小波动。DIA XIC 仅 5~9 点，阈值过严会漏检双峰，过松会算入噪声小峰。0.3 是平衡值。
- **端点不计入 peak**（scipy 的默认行为）；5 点单峰 XIC `[1,5,100,5,1]` 返回 1

### 2.4 `_calc_smoothness(intensity)`

```python
def _calc_smoothness(intensity):
    """二阶差分平方和 / 总强度^2。

    平滑山峰 → 接近 0；抖动 / 单点尖刺 → 大值。
    """
    if len(intensity) < 3:
        return 0.0
    total = float(np.sum(intensity))
    if total <= 0:
        return 0.0
    second_diff = np.diff(intensity, n=2)
    return float(np.sum(second_diff ** 2) / (total ** 2 + 1e-12))
```

- **范围**：`[0, ∞)`；用 `total^2` 归一化使不同绝对强度可比
- **真 PSM**：< 0.01
- **噪声 / 尖刺**：> 0.05
- **局限**：对 XIC 点数不归一，5 点和 9 点 XIC 的 smoothness 量纲不严格可比；但因为同一项目 `xic_cycle_window` 配置固定，实际跨样本可比

## 三、新增特征清单（20 列）

### 前体级（4 列，heavy XIC）

| 列名 | 来源 |
|------|------|
| `precursor_base_to_apex_ratio` | `_calc_base_to_apex_ratio(heavy_xic["intensity"])` |
| `precursor_apex_monotonicity` | `_calc_apex_monotonicity(heavy_xic["intensity"])` |
| `precursor_n_peaks` | `_calc_n_peaks(heavy_xic["intensity"])` |
| `precursor_smoothness` | `_calc_smoothness(heavy_xic["intensity"])` |

### 碎片级汇总（16 列）

通过 `extract_ion_numeric_features` 把每个碎片的 4 个 peak-likeness 值汇总成 mean/p50/std/max：

| 列名前缀 | 4 统计量 |
|---------|---------|
| `all_base_to_apex_ratio_{mean,p50,std,max}` | 全部有效碎片的 base/apex |
| `all_apex_monotonicity_{mean,p50,std,max}` | 全部有效碎片的单调性 |
| `all_n_peaks_{mean,p50,std,max}` | 全部有效碎片的峰数 |
| `all_smoothness_{mean,p50,std,max}` | 全部有效碎片的平滑度 |

**总计**：4 + 16 = **20 个新 CSV 列**

## 四、实现改动清单

### 4.1 `workflows/single_work.py`

1. **新增 4 个 `_calc_*` helper**（位置：紧跟 `_calc_snr` 之后，~line 862）
2. **修改 `_default_xic_score()`**：新增 4 个 0.0 默认字段
3. **修改 `calc_xic_score()`**：在已有 `peak_symmetry` 计算之后加 4 行 helper 调用
4. **修改 `calc_xic_score()` 早返回路径**（`rt_start >= rt_end`）：新增 4 字段到 result
5. **修改 `calc_xic_score()` 正常返回 dict**：新增 4 字段
6. **修改 `single_pair_work()` 前体块**：
   - 空 XIC 路径：4 个 0.0 默认值
   - 正常路径：4 个从 `precursor_score` 提取
7. **修改 `single_pair_work()` 碎片循环**：
   - 循环前：4 个新 `fragment_*` 列表
   - 循环内：4 个 `.append`
   - 循环后：4 个 `extract_ion_numeric_features` 调用（前缀 `all_*`）
8. **修改 `multi_batch_work()`**：镜像 6~7 的全部改动

### 4.2 `tests/test_single_work_numerics.py`

新增 ~12 个 TDD 测试（每个 helper 3 个：典型 + 边界 + 反例）。

## 五、向后兼容性

| 接口 | 兼容性 |
|------|-------|
| `calc_xic_score(...)` 现有调用 | 完全兼容；新字段总是被设置 |
| `_default_xic_score()` 返回 dict | 多 4 字段；旧消费者按 key 访问不受影响 |
| 现有 26 + 11 个 CSV 列 | 不变；新 20 列追加到现有列之后 |
| npz cache 格式 | 不影响 |
| `extract_ion_numeric_features` 接口 | 不变 |

## 六、风险

| 风险 | 评估 | 应对 |
|------|------|------|
| `_calc_n_peaks` 的 `prominence=0.3` 阈值在不同窗口宽度下表现不一 | 中 | 与 PLAN.md 已有 `high_corr_frag_ratio=0.5` 阈值思路一致；如效果不佳，加 `prominence=0.2/0.5` 消融 |
| `_calc_smoothness` 对 XIC 点数不严格归一 | 低 | 项目 `xic_cycle_window` 配置固定，实际可比 |
| `scipy.signal.find_peaks` 在 XIC 极短（3-4 点）时返回不稳定 | 低 | 长度 < 3 提前返回 0；3 点情况下 find_peaks 至多识别 1 个中间点 |
| 20 个新特征与现有 `snr` / `peak_symmetry` 高度共线 | 中 | 4 个新特征覆盖不同峰形维度；LightGBM 自动处理冗余；可用 SHAP 验证 |
| `smoothness` 在全零或常值 XIC 上数值为 0，与正常值不可分 | 低 | 长度/total 检查已覆盖；LightGBM 可借助 `valid_fragment_ions_num` 等 covariate 区分 |

## 七、不在本次范围

- **`_calc_gaussian_r2`**：高斯拟合特征（用户选择推迟，待 4 个简单特征验证后再考虑）
- **light XIC 上的同类特征**：light 的位置是 PSM 报告 RT，几乎总是真信号，加入会增加共线性而非区分力
- **`peak_width_ratio` 的扩展**：4 个新特征都是 heavy 单边的山峰倾向，与 H/L 一致性的 `peak_width_ratio` 设计哲学相反，不混入
