# XIC apex/峰形 异常的完整惩罚设计（方案 A+）

**日期**：2026-06-20
**背景**：分析 `runs_new` 固定 clean hard-test 上的强 decoy（如 2da `QQITDLER` z2、`EQQEIAER` z2）时发现，许多负例的母离子/碎片 XIC 并非"山峰"，而是**信号从最高单调到最低的斜坡**或**信号簇**，apex 不在窗口中心。现有峰形特征 `apex_monotonicity` 对这种"边缘斜坡"给出满分 1.0（与居中真峰无异），即**漏惩罚**。进一步核对发现：(a) 峰形质量（`base_to_apex`/`monotonicity`/`n_peaks`/`smoothness`/`symmetry`）只在**重标**通道计算，轻标斜坡只有一个弱 `light_cycle_offset` 能反映；(b) 单 cycle 尖刺、平台/flat-top 也未被充分惩罚；(c) 碎片聚合只有 `mean/p50/std/max`，对"高=好"的 `monotonicity` 没有"最差碎片"统计量。

**目标**：系统性地让**每一类坏 XIC apex/峰形**都至少被一个特征惩罚到，且**轻、重双通道 + 母离子 + 碎片**全覆盖。

**关键代码**：
- 共享计算 `workflows/single_work.py:calc_xic_score()`（母离子与每个碎片对都调用它）。
- 峰形 helper：`_calc_apex_monotonicity`、`_calc_base_to_apex_ratio`、`_calc_n_peaks`、`_calc_smoothness`、`_calc_symmetry`、`_calc_snr`、`_calc_cycle_offset`、`_calc_fwhm`（均在 `single_work.py:1091-1260`）。
- 4 个消费点：`multi_batch_work`（母离子块 + 碎片循环）、`single_pair_work`（母离子块 + 碎片循环）。
- 碎片聚合：`extract_ion_numeric_features`（mean/p50/std/max）。
- 训练选列：`tools/spec_trainer/src/feature_cols.py:resolve_feature_cols`，配置 `feature_cols: []` 即**自动取全部非 META/EXCLUDED 列**——新增列自动进模型。

---

## 1. 坏形态 taxonomy 与惩罚覆盖

| 坏形态 | 现有惩罚（重标） | 现有（轻标） | 本设计后 |
|---|---|---|---|
| 边缘斜坡（apex 贴边，高→低/低→高） | `monotonicity`=**1.0(bug)**、`base_to_apex`~0.6、`n_peaks`=0、`symmetry`高 | 仅 `cycle_offset` | `centering_defect`≈1 + `base_to_apex`~0.6（轻+重镜像）。注：纯斜坡本身单调，故由"apex 贴边"而非"形状不规则"来抓 |
| 平台 / flat-top（宽平顶） | `monotonicity`~1.0（持平不计违例）、`base_to_apex`（仅两端也高时） | 无 | `shape_irregularity`（严格升降）+ `base_to_apex` |
| 多峰簇 / 共流出 | `n_peaks`≥2、`monotonicity`低 | 无 | `n_peaks`（轻+重）+ `shape_irregularity` |
| 锯齿 / 噪声 | `monotonicity`低、`smoothness`高 | 无 | `shape_irregularity` + `smoothness`（轻+重） |
| 单 cycle 尖刺 | `monotonicity`=1.0、`base_to_apex`≈0、`snr`=1000 均给好评；仅 `smoothness` 微弱反对 | 无 | `narrow_defect`≈1 + `smoothness`（轻+重） |
| apex 偏离预期 RT | `cycle_offset`（原始整数） | `cycle_offset` | `centering_defect`（归一化，轻+重） |
| 某通道好 / 另一通道坏 | 只测重标 | — | 轻标镜像后可比 |
| 少数碎片坏、多数好 | 聚合 `mean/p50/max/std`（`max`=最好碎片，非最差） | — | 新缺陷指标"高=坏"→ 现有 `_max`=最差碎片 |

---

## 2. 设计原则

1. **集中实现**：所有改动落在 `calc_xic_score` + helper，母离子与碎片自动同时受益。
2. **统一"高 = 坏"方向**：所有新/改的峰形缺陷指标数值越大越可疑。如此碎片现有 `_max` 聚合即"最差碎片"，无需新增聚合方向（解决"少数坏碎片被均值淹没"）。
3. **鲁棒 apex 定位**：取近峰顶区（≥0.95·max 的点）的**中位索引**，避免 `np.argmax` 在平顶/并列时落到左缘导致误判。
4. **轻重双通道**：峰形在 light 与 heavy 上各算一份。

---

## 3. 特征目录

所有"缺陷类"指标方向统一为 **高 = 坏**，取值范围标注于公式。

| 指标 | 通道 | 母/碎 | 状态 | 主要抓取 |
|---|---|---|---|---|
| `apex_centering_defect` | 轻 + 重 | 母 + 碎 | 新增 | 偏离 RT + 边缘斜坡 |
| `shape_irregularity` | 轻 + 重 | 母 + 碎 | 改（替代 `apex_monotonicity`） | 平台/flat-top + 锯齿/多峰 |
| `base_to_apex_ratio` | 轻 + 重 | 母 + 碎 | 重已有，补轻标 | 斜坡 / 平台 |
| `n_peaks` | 轻 + 重 | 母 + 碎 | 重已有，补轻标 | 多峰簇 / 共流出 |
| `smoothness` | 轻 + 重 | 母 + 碎 | 重已有，补轻标 | 尖刺 / 锯齿 |
| `narrow_defect` | 轻 + 重 | 母 + 碎 | 新增 | 单 cycle 尖刺 |

**保留不变**（覆盖轻重配对类错误，与上表互补）：`*_apex_cycle_offset(_signed)`（原始有符号偏移）、`apex_delta(_signed)`、`pearson`、`cosine`、`intensity_ratio`、`peak_width_ratio`、`peak_symmetry`、`snr`、`mz_avg_err`。

---

## 4. 公式

记 `I` = XIC 强度数组（已按 rt 排序），`n=len(I)`，`max=np.max(I)`。

### 4.1 鲁棒 apex 索引
```
near = where(I >= 0.95 * max)
apex_idx = int(median(near))          # 平顶并列时取中点
```
空 / 全零 / 短（n<3）XIC → 退化处理（见 §6 默认值）。

### 4.2 `apex_centering_defect`（高=坏，[0,1]）
基于现有 `_calc_cycle_offset` 的整数偏移，按窗口半宽归一化：
```
half = max(1, (valid_cycles - 1) / 2)          # valid_cycles = cycle_idx>=0 的点数
centering_defect = min(1.0, abs(cycle_offset) / half)
```
apex 正中→0；apex 贴窗边（含边缘斜坡）→≈1。参照 RT：light 用 `center_rt`（轻标 ID RT），heavy 用 `heavy_center_rt`。

### 4.3 `shape_irregularity`（高=坏，[0,1]，替代 monotonicity）
用鲁棒 `apex_idx`，**严格**升/降（持平 `diff==0` 也计违例）：
```
left  = I[:apex_idx + 1];  right = I[apex_idx:]
lv = count(diff(left)  <= 0)      # 左侧应严格上升
rv = count(diff(right) >= 0)      # 右侧应严格下降
shape_irregularity = (lv + rv) / max(1, n - 1)
```
居中真峰→≈0；平台（持平）→高；锯齿/多峰→高。

### 4.4 `base_to_apex_ratio`（高=坏，沿用）
```
base = (I[0] + I[-1]) / 2;  ratio = base / max
```
仅新增在 **light** 通道也计算（heavy 已有）。

### 4.5 `n_peaks`（高=坏，沿用）
`scipy.signal.find_peaks(I, prominence=0.3*max)` 的峰数。仅新增 light 通道。

### 4.6 `smoothness`（高=坏，沿用）
归一化二阶差分均方（现有 `_calc_smoothness`）。仅新增 light 通道。

### 4.7 `narrow_defect`（高=坏，[0,1]）
```
support = count(I >= 0.5 * max)        # 近似 FWHM 的 cycle 数
narrow_defect = 1.0 / max(1, support)
```
单点尖刺(support=1)→1.0；宽峰(support≥3)→≤0.33。轻、重各算。

---

## 5. 集成与数据流

1. **`calc_xic_score`**：新增返回键（每个指标 light/heavy 各一份）：
   `light_centering_defect`、`heavy_centering_defect`、`light_shape_irregularity`、`heavy_shape_irregularity`、`light_base_to_apex_ratio`、`light_n_peaks`、`light_smoothness`、`light_narrow_defect`、`heavy_narrow_defect`。其中 `shape_irregularity` 在 heavy 上替代原 `apex_monotonicity`。
2. **母离子（2 个消费点）**：在 `multi_batch_work` 与 `single_pair_work` 的母离子块写出对应 `precursor_*` 列；空 XIC 分支补默认值。
3. **碎片（2 个消费点）**：每个碎片把新指标 append 到对应列表，循环后用聚合函数汇总为 `all_*` 列。
4. **碎片聚合**：新"缺陷类"指标（`*_centering_defect`、`*_shape_irregularity`、`*_narrow_defect`、`light_*`）**只输出 `mean` + `max` 两个聚合**（`max`=最差碎片），以控制列数；沿用指标（heavy `base_to_apex`/`n_peaks`/`smoothness`）保持现有 4 聚合不变。为此在 `extract_ion_numeric_features` 旁新增一个轻量聚合 helper（或加 `stats=("mean","max")` 参数），不改动现有调用。
5. **训练**：`feature_cols: []` 自动纳入新列，无需改任何训练/cross_test 配置。

**新增列**：母离子 9 列；碎片 18 列（新缺陷指标 ×2 聚合）；移除 5 列（旧 `apex_monotonicity`）。特征总数 142 → 164。早前消融表明模型对多特征鲁棒、树会自动取舍，可接受。

### 5.1 轻标碎片形状开关
轻标**碎片**形状是本批最贵、置信度最低的一块（点名失败案例为母离子轻标）。新增 config 布尔项 `[general] light_fragment_shape`（默认 `true`，与现有 `xic_cycle_window` 等同属 `[general]` 段，经 `config[ConfigKeys.GENERAL].getboolean(...)` 读取）；为 `false` 时跳过轻标碎片形状指标的计算与写出，便于消融。母离子轻标形状不受此开关影响（始终计算）。

---

## 6. 边界 / 缺失值处理

- 空 XIC（`xic_empty`）或 `n<3`：`centering_defect=0`、`shape_irregularity=0`、`narrow_defect=0`、`base_to_apex=0`、`n_peaks=0`、`smoothness=0`（与现有"空→0"约定一致，避免引入 NaN）。
  - 说明：缺失信号本身已由既有 `*_xic_empty`、`heavy_in_raw` 等标志位表达，缺陷指标取 0（"无证据"）而非 1（"确诊坏"），防止把"没数据"误当"坏形状"。
- 非有限值（NaN/Inf）：沿用现有 helper 的 `np.all(np.isfinite)` 守卫，返回 0。
- 鲁棒 apex 的 `near` 为空（全零）：退化为 `argmax`，并在全零时按上一条返回默认。

---

## 7. 兼容性 / 重命名

- `apex_monotonicity` → `shape_irregularity`（语义反向；用户已确认采用"原地修复、改变语义"方案）。需同步更新：
  - `tools/eval_feature_ablation.py`（硬编码特征名列表）。
  - 任何引用 `apex_monotonicity` / `all_apex_monotonicity` 的分析脚本与文档。
- `feature_cols.py` 的 `META_COLUMNS` / `EXCLUDED_EXTRA` 不含上述任何指标，无需改动；新列默认参与训练。
- 重标已有 `base_to_apex_ratio`/`n_peaks`/`smoothness` 列名**保持不变**（隐含=重标），轻标版加 `light_` 前缀；不为减小改动面而重命名重标列。

---

## 8. 测试

### 8.1 单元测试（合成轨迹，断言方向）
对以下每种 1D 强度构造断言新指标方向正确：

| 轨迹 | 期望 |
|---|---|
| 居中高斯真峰 | 所有缺陷≈0 |
| 单调降斜坡（apex idx0） | `centering_defect`≈1、`base_to_apex`~0.6；`shape_irregularity`≈0（纯斜坡本就单调，不靠它抓） |
| 单调升斜坡（apex idx n-1） | `centering_defect`≈1、`base_to_apex`~0.6 |
| 宽平顶 | `shape_irregularity` 中（持平计违例）、`base_to_apex` 偏低 |
| 双峰簇 | `n_peaks`≥2、`shape_irregularity` 高 |
| 锯齿噪声 | `shape_irregularity` 高、`smoothness` 高 |
| 单点尖刺（居中） | `narrow_defect`≈1、`shape_irregularity`~0.5（两端持平零）、`centering_defect`≈0 |
| 居中但偏移峰（apex 离 center_rt 远） | `centering_defect` 高、`shape_irregularity` 低 |
| 空 / 全零 / n<3 | 全部默认值，无 NaN |

鲁棒 apex 单测：平顶并列 → `apex_idx` 落在平顶中点而非左缘。

### 8.2 集成 / 回归
- 跑 filter → 重抽特征 make 流水线，确认新列出现、无 NaN 泄漏、`light_fragment_shape=false` 时对应列消失。
- 现有 `single_work` / `calc_xic_score` 测试全绿（除被替换的 `apex_monotonicity` 断言改名）。

### 8.3 判别力验证（固定 clean hard-test）
- 对 2da / 5da / normal，比较改前/改后在固定 clean hard-test 上的 **FNR@FPR≤5%**（既有评估协议）。
- 查新列的 gain importance。
- 对 `QQITDLER` z2、`EQQEIAER` z2 复核其 `centering_defect`/`shape_irregularity`/`narrow_defect` 确实升高（apex 不在中心被惩罚）。

---

## 9. 预期收益与风险

- **预期**：母离子家族判别力最弱（单变量 AUC 上限 ~0.32），单独收益有限；**主要杠杆在碎片**（81 维、更强），`shape_irregularity` 边界修复 + `_max` 暴露最差碎片有望带来可测量但温和的 FNR 改善。对 EQQEIAER 类轻标斜坡有针对性补强。
- **风险**：
  - 列数增长 ~33；早前消融显示模型对此鲁棒，但需在固定 clean hard-test 上确认未引入过拟合/退化。
  - 鲁棒 apex 的 0.95 阈值、`narrow_defect` 的 0.5 阈值为经验默认；若验证不佳，作为后续可调项（暂硬编码，不提前进 config）。
  - `shape_irregularity` 严格升降会对真实平肩峰略有惩罚，属可接受噪声。

## 10. 不在范围

- 不引入高斯拟合 / 模板匹配等重型峰形描述（方案 C，过拟合与性能风险）。
- 不改负样本生成、不改 speclib/q1a 特征。
- 不为历史 run 回填新列。
