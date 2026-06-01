# mzML 加载时谱图中心化（centroiding）设计

> 状态：Approved  
> 日期：2026-06-01  
> 范围：`spectrum/dia_data.py`、`spectrum/spectrum_utils.py`、
> `workflows/flow_utils.py`、`config.ini`、`tests/`  
> **不包含**：pf2/pf1 格式支持、DIAData 的多 spectrum-source 抽象重构。

---

## 1. 背景与动机

`DIAData._load_from_mzml` 当前直接吃 mzML 的 `m/z array` / `intensity array`。
经验证，现有测试 mzML（`20190830_HF_ZHW_hela_SILAC_DIA_350_1000_Rep1.mzML`）
是 **profile 模式**：MS1 单张约 31 929 点、MS2 单张 16 000–22 000 点。

profile 数据带来两个问题：

1. **内存/磁盘膨胀** — 现有路径会把所有 profile 点全量塞进
   `_mz_values` / `_intensity_values`；DIA 全 run 几十万张谱图，单
   `.dia.npz` 缓存常达数 GiB。
2. **匹配精度反而下降** — `spectrum/spectrum_utils.py::match_peak_ppm`
   依赖"一峰一点"语义；profile 模式下同一物理峰被多次计入，导致
   `np.sum(matched_intensities)` 重复累加、`np.average` 的 ppm 漂移到
   峰肩位置。

引入 **加载时 centroiding**：把每张 profile 谱图就地压成"一峰 = 一个
(m/z, intensity)"的中心化表示，下游所有逻辑无须改动。

## 2. 用户/调用方接触面

| 接触面 | 变化 |
|--------|------|
| `config.ini` `[general]` | 新增 2 个键（见 §6），默认值保持现有行为兼容 |
| `_load_from_mzml(...)` 行为 | 加载结果中 `_mz_values` 长度变小、峰值定位更准；接口签名不变 |
| `.dia.npz` 缓存 | 加入 `_format_version` 字段，老缓存将被视为非法、自动重建（见 §7） |
| 下游 `xic_*` / `match_peak_ppm` | 无需改动 |

## 3. 总体架构

```
   mzML (profile)
        │
        ▼  (pyteomics.mzml.read)
   spectrum dict
        │
        ▼  ← 新增：centroid_spectrum(...)（spectrum_utils.py）
   (mz_centroided, int_centroided)        ← 即下游唯一感知到的数据
        │
        ▼
   DIAData 数组 (_mz_values 等)
        │
        ▼
   .dia.npz (_format_version=2)
```

## 4. centroid 算法（已确认：B 方案）

放在 `spectrum/spectrum_utils.py`：

```python
def centroid_spectrum(
    mz: np.ndarray,
    intensity: np.ndarray,
    rel_threshold: float = 1e-3,
) -> tuple[np.ndarray, np.ndarray]:
    """对单张 profile 谱图做局部极大值 + 抛物线重心 centroiding。

    Args:
        mz, intensity: 同长度的 1D 数组 (float32 / float64)。
            假定按 mz 单调递增（mzML 原生保证）。
        rel_threshold: 单张谱图内，intensity 阈值 = max(intensity)*rel_threshold；
            低于阈值的局部极大值丢弃。默认 1e-3。

    Returns:
        (mz_out, intensity_out)，dtype 与输入相同（float32 优先）。
        mz_out 是抛物线精修后的浮点 m/z；intensity_out 是峰顶 intensity。
    """
```

**算法步骤（全程 numpy 向量化）：**

1. **找极大值**：用切片比较 `mz[1:-1]` 处满足
   `int[i-1] < int[i] and int[i] >= int[i+1]` 的索引集合 `idx`。  
   （等号放右边以稳定处理平顶峰；输入是单调 mz，左闭右闭等号策略对实际
   profile MS 不会产生 systematic bias。）
2. **阈值过滤**：保留 `intensity[idx] >= max(intensity) * rel_threshold` 的 idx。
3. **抛物线精修**（每个保留 idx 上）：

   ```
   y0, y1, y2 = int[idx-1], int[idx], int[idx+1]
   dx = 0.5 * (y0 - y2) / (y0 - 2*y1 + y2)           # 抛物线顶 offset
   refined_mz[k] = mz[idx] + dx * (mz[idx+1] - mz[idx-1]) / 2
   refined_int[k] = y1                                # 峰顶高度
   ```

   分母接近 0（噪声平台）时直接用 `mz[idx]`，避免除零。
4. **边界处理**：第 0 个和最后一个点不参与极大值判定。
5. **空输入 / 长度 < 3**：直接返回 `(empty, empty)`，长度 0。

**Already-centroid 短路**：见 §5.2。

**复杂度**：单张谱图 O(N) 一次扫描，纯 numpy。预计在 31 929 点上 < 1 ms。

## 5. `DIAData._load_from_mzml` 改造

### 5.1 取消"按 total_peaks 预分配 peak 数组"

当前实现：第一遍扫描统计 `total_spectra` 与 `total_peaks`，按这两个
精确数预分配所有定长数组；第二遍填充。

centroid 后 `total_peaks` **未知**且会大幅缩小，"先 centroid 一次统计
长度"等于把代价最重的步骤跑两次。改造方案分两类数组处理：

- **按谱图数 (total_spectra) 预分配的数组**——`precursor_scan_ids`、
  `rt_values`、`_peak_start_idx_list`、`_peak_stop_idx_list`、
  `_precursor_lower_mz`、`_precursor_upper_mz`、`_scan_id_to_index` ——
  **保留第一遍扫描**统计 `total_spectra`。这一遍**不读 `m/z array` /
  `intensity array`**（pyteomics 按需懒解码，访问哪个键才解码哪个），
  代价远小于 centroid。
- **按峰数 (total_peaks) 预分配的数组**——`_mz_values`、
  `_intensity_values` ——**改成 chunk + concat**：第二遍每张谱图
  centroid 后把 `(mz_chunk, int_chunk)` 追加进 Python list；同时把
  `current_peak_index` 推进至 `+ len(mz_chunk)`，写入
  `_peak_start_idx_list[i]` / `_peak_stop_idx_list[i]`；二遍末尾
  `np.concatenate(mz_chunks)` 一次拼成最终数组，原 list 立即释放。

### 5.2 centroid/profile 检测与短路

`_process_single_spectrum` 内：

```python
mz_array = spectrum['m/z array']
intensity_array = spectrum['intensity array']

if self._centroid_enabled and not _is_already_centroid(spectrum):
    mz_array, intensity_array = centroid_spectrum(
        mz_array, intensity_array,
        rel_threshold=self._centroid_rel_threshold,
    )
```

其中：

```python
def _is_already_centroid(spectrum) -> bool:
    # pyteomics 把 cv term 解析为字典键，存在即为 True
    return 'centroid spectrum' in spectrum
```

### 5.3 配置读取

`DIAData.__init__` 暴露 2 个新字段：

```python
self._centroid_enabled: bool = True
self._centroid_rel_threshold: float = 1e-3
```

`manager/data_manager.py::DataManager.get_dia_data_object` 在 `dia_data =
DIAData()` 后从 `self._config` 读两键写入（fallback 即默认值），再调
`_load_from_mzml`。

## 6. config.ini 新增键

```ini
[general]
# ... 现有键 ...

# 加载 mzML 时是否对 profile 谱图做 centroiding。
# 设为 false 退回旧行为（保留 profile，所有点）。
centroid_enabled = true

# centroid 阈值：单张谱图内 intensity < max * 该比值 的局部极大值丢弃。
# 典型范围 1e-4 ~ 1e-2；推荐 1e-3。
centroid_rel_threshold = 0.001
```

`constant/keys.py::ConfigKeys` 同步加 `CENTROID_ENABLED`、
`CENTROID_REL_THRESHOLD`。

## 7. npz 缓存版本兼容

`spectrum/dia_data.py::save_to_file` / `load_from_file` 改造：

- `save_to_file`：写入 `_format_version = np.int32(2)`。
- `load_from_file`：读出 version；若键不存在（=老缓存，profile peaks）
  或 ≠ 2，**抛 `ValueError`** 并在 message 中提示 "请删除
  `<path>` 并重新生成"。
- 不做自动迁移：老缓存里的 profile 数据无法从 npz 反推 centroid 参数，
  迁移没意义。
- `workflows/flow_utils.py::data_to_npz` 现行"`if not exists: 生成`"
  逻辑无须改动——用户删掉老 `.dia.npz` 之后会自然走重建路径。

> **注意**：项目 `.gitignore` 已忽略 `*.dia.npz`，老缓存只在 workspace
> 里，删除无副作用。

## 8. 测试（放 `tests/`）

新文件 `tests/test_centroid_spectrum.py`，**至少**覆盖：

1. **合成高斯峰** — 5 个 sigma=0.005 Da 的孤立高斯峰，每个 11 个采样点；
   断言 `len(out_mz) == 5`、每个峰 `|out_mz[k] - true_center| < 0.001`。
2. **噪声过滤** — 在峰之间加 `intensity = base_peak * 1e-4` 的噪声；
   阈值 1e-3 下断言噪声峰被过滤。
3. **空 / 极短输入** — `len < 3` 时返回空数组、不抛异常。
4. **already-centroid 检测**（unit test, mock spectrum dict）— 含
   `'centroid spectrum'` 键时 `_process_single_spectrum` 不调
   centroid_spectrum（用 monkeypatch / spy 验证）。
5. **抛物线分母接近 0** — 三点完全相等的平顶峰，断言不抛 ZeroDivisionError，
   并返回 `mz[idx]`。

不强制做端到端 mzML 集成测试（依赖大文件），但保留人工 smoke 步骤：
跑一遍 `main.py`，对比 result.csv 关键列。

## 9. 风险与缓解

| 风险 | 缓解 |
|------|------|
| centroid 改变 m/z 后影响下游 ppm 匹配的相对结果 | 默认阈值 1e-3 是保守值；用户可在 config 中下调（如 1e-4）保留更多弱峰 |
| 抛物线在峰肩饱和（detector saturation）失真 | profile 峰顶平坦时分母→0，已 fallback 到 `mz[idx]` |
| 老 `.dia.npz` 缓存被错误复用 | `_format_version` 校验，加载即抛错并指明删除路径 |
| 单遍 list-concat 大批量谱图时 GC 压力 | concat 前不持有原 list 引用；list 元素是 numpy 视图，本身轻量 |
| `_is_already_centroid` 检测漏掉某些 vendor 标记 | 默认行为是"按 profile 处理"；最坏情况是把已 centroid 数据再 centroid 一次（局部极大值还是它自己，幂等） |

## 10. 不在本次范围（搁置项）

- pf2/pf1 二进制格式输入支持 — 因 PF2 不携带 RT/隔离窗口/父 MS1 等
  关键 metadata，单独的格式支持无法满足 DIA-SILAC 流程。需要先有
  sidecar metadata 方案才能推进。
- `DIAData` 解耦为多 spectrum-source 抽象（mzML / pf2 / Bruker tdf 等）—
  当前 schema 与 mzML 强耦合，重构成本不在本次预算内。
