# XIC apex/峰形 异常完整惩罚 实现计划

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** 让每一类坏 XIC apex/峰形（边缘斜坡、平台、多峰簇、锯齿、单 cycle 尖刺、apex 偏离 RT）都至少被一个特征惩罚到，且轻+重双通道、母离子+碎片全覆盖。

**Architecture:** 全部集中在共享函数 `workflows/single_work.py:calc_xic_score()` 及其 helper —— 母离子与每个碎片对都调用它，因此一处实现两处受益。新增/改动的峰形缺陷特征统一为"高=坏"方向，使碎片现有 `_max` 聚合自动等于"最差碎片"。轻标碎片形状由 config 开关 `[general] light_fragment_shape`（默认 on）控制。训练侧 `feature_cols: []` 自动纳入新列。

**Tech Stack:** Python 3.14, numpy, scipy.signal, pytest 9。测试命令前缀 `python -m pytest`（`python` = /usr/bin/python）。

**Spec:** `docs/superpowers/specs/2026-06-20-xic-apex-shape-penalty-design.md`

---

## 文件结构

| 文件 | 职责 | 改动 |
|---|---|---|
| `workflows/single_work.py` | XIC 打分 + 峰形 helper + 母离子/碎片特征装配 | 新增 4 个 helper、改 `calc_xic_score`、改 2 个消费函数、加装配 helper |
| `constant/keys.py` | 配置键常量 | 新增 `LIGHT_FRAGMENT_SHAPE` |
| `tools/eval_feature_ablation.py` | 消融脚本（硬编码特征名） | `apex_monotonicity` → `shape_irregularity` 改名 |
| `tests/test_single_work_numerics.py` | 数值/峰形单测 | 改名相关断言 + 新 helper 测试 |
| `tests/test_deep_audit_p1.py` | 含 `all_apex_monotonicity_mean` 断言 | 改名 |
| `tests/test_feature_cols_contract.py` | 列契约测试（==142） | 更新计数 |
| `tests/fixtures/features_header_155.csv` | 真实表头固定件 | 重新生成 |

## 命名约定（最终列名）

`calc_xic_score` 返回 dict 新增键（每通道）：`light_centering_defect`、`heavy_centering_defect`、`light_shape_irregularity`、`heavy_shape_irregularity`、`light_base_to_apex_ratio`、`light_n_peaks`、`light_smoothness`、`light_narrow_defect`、`heavy_narrow_defect`。移除 `apex_monotonicity`（由 `heavy_shape_irregularity` 取代）。重标已有 `base_to_apex_ratio`/`n_peaks`/`smoothness` 键不变。

母离子列：`precursor_light_centering_defect`、`precursor_heavy_centering_defect`、`precursor_light_shape_irregularity`、`precursor_heavy_shape_irregularity`、`precursor_light_base_to_apex_ratio`、`precursor_light_n_peaks`、`precursor_light_smoothness`、`precursor_light_narrow_defect`、`precursor_heavy_narrow_defect`。移除 `precursor_apex_monotonicity`。重标 `precursor_base_to_apex_ratio`/`precursor_n_peaks`/`precursor_smoothness` 不变。

碎片聚合列（仅 `mean`+`max` 两个聚合）：`all_<key>_mean`、`all_<key>_max`，其中 `<key>` ∈ heavy/light 各缺陷。移除 `all_apex_monotonicity_*`。

---

### Task 1: `_robust_apex_idx` 鲁棒峰顶定位 helper

**Files:**
- Modify: `workflows/single_work.py`（在 `_calc_apex_monotonicity` 定义之前插入，约 line 1145 前）
- Test: `tests/test_single_work_numerics.py`（文件末尾追加）

- [ ] **Step 1: 写失败测试**

在 `tests/test_single_work_numerics.py` 末尾追加：

```python
def test_robust_apex_idx_single_peak():
    from workflows.single_work import _robust_apex_idx
    assert _robust_apex_idx(np.array([1, 3, 9, 3, 1], dtype="f8")) == 2


def test_robust_apex_idx_flat_top_returns_center_not_left_edge():
    """平顶并列时取近峰顶区中位索引，而非 argmax 的最左点。"""
    from workflows.single_work import _robust_apex_idx
    # 三点等高平顶在 idx 2,3,4 -> 期望 apex_idx == 3
    assert _robust_apex_idx(
        np.array([1, 3, 5, 5, 5, 3, 1], dtype="f8")) == 3


def test_robust_apex_idx_empty_and_allzero():
    from workflows.single_work import _robust_apex_idx
    assert _robust_apex_idx(np.array([], dtype="f8")) == 0
    assert _robust_apex_idx(np.array([0, 0, 0], dtype="f8")) == 0


def test_robust_apex_idx_nonfinite_falls_back_to_argmax():
    from workflows.single_work import _robust_apex_idx
    arr = np.array([1, np.nan, 9, 1], dtype="f8")
    # argmax on NaN-containing array is implementation-defined but must not raise
    idx = _robust_apex_idx(arr)
    assert isinstance(idx, int)
```

- [ ] **Step 2: 运行确认失败**

Run: `python -m pytest tests/test_single_work_numerics.py -q -k robust_apex_idx`
Expected: FAIL，`ImportError: cannot import name '_robust_apex_idx'`

- [ ] **Step 3: 实现**

在 `workflows/single_work.py` 中，`def _calc_apex_monotonicity` 之前插入：

```python
def _robust_apex_idx(intensity: np.ndarray, flat_tol: float = 0.05) -> int:
    """Median index of the near-max region (>= (1-flat_tol)*max).

    Robust to flat tops / ties where np.argmax silently returns the
    leftmost max index — which would make a centered plateau look like a
    left-edge ramp. Returns 0 for empty input; falls back to argmax for
    all-zero / non-finite input (defensive, never raises).
    """
    if len(intensity) == 0:
        return 0
    if not np.all(np.isfinite(intensity)):
        return int(np.argmax(intensity))
    max_v = float(np.max(intensity))
    if max_v <= 0:
        return int(np.argmax(intensity))
    near = np.where(intensity >= (1.0 - flat_tol) * max_v)[0]
    if len(near) == 0:
        return int(np.argmax(intensity))
    return int(round(float(np.median(near))))
```

- [ ] **Step 4: 运行确认通过**

Run: `python -m pytest tests/test_single_work_numerics.py -q -k robust_apex_idx`
Expected: PASS（4 passed）

- [ ] **Step 5: 提交**

```bash
git add workflows/single_work.py tests/test_single_work_numerics.py
git commit -m "feat: add _robust_apex_idx (flat-top-aware apex location)

Co-authored-by: Copilot <223556219+Copilot@users.noreply.github.com>"
```

---

### Task 2: `_calc_shape_irregularity` 严格升降违例（高=坏）

**Files:**
- Modify: `workflows/single_work.py`（紧接 `_robust_apex_idx` 之后插入）
- Test: `tests/test_single_work_numerics.py`（末尾追加）

- [ ] **Step 1: 写失败测试**

```python
def test_shape_irregularity_clean_peak_is_zero():
    from workflows.single_work import _calc_shape_irregularity
    assert _calc_shape_irregularity(
        np.array([1, 3, 9, 3, 1], dtype="f8")) == 0.0


def test_shape_irregularity_monotone_ramp_is_zero():
    """纯单调斜坡本身不算'形状不规则'（由 centering_defect 来抓）。"""
    from workflows.single_work import _calc_shape_irregularity
    assert _calc_shape_irregularity(
        np.array([9, 7, 5, 3, 1], dtype="f8")) == 0.0


def test_shape_irregularity_flat_top_penalized():
    """平顶持平 (diff==0) 计违例 -> 非零。"""
    from workflows.single_work import _calc_shape_irregularity
    r = _calc_shape_irregularity(np.array([1, 3, 5, 5, 5, 3, 1], dtype="f8"))
    assert r > 0.0


def test_shape_irregularity_zigzag_high():
    from workflows.single_work import _calc_shape_irregularity
    r = _calc_shape_irregularity(np.array([1, 5, 2, 6, 1], dtype="f8"))
    assert r >= 0.5


def test_shape_irregularity_short_or_nonfinite_zero():
    from workflows.single_work import _calc_shape_irregularity
    assert _calc_shape_irregularity(np.array([1, 2], dtype="f8")) == 0.0
    assert _calc_shape_irregularity(
        np.array([1, np.nan, 3], dtype="f8")) == 0.0
```

- [ ] **Step 2: 运行确认失败**

Run: `python -m pytest tests/test_single_work_numerics.py -q -k shape_irregularity`
Expected: FAIL，`ImportError: cannot import name '_calc_shape_irregularity'`

- [ ] **Step 3: 实现**

```python
def _calc_shape_irregularity(intensity: np.ndarray) -> float:
    """Strict rise-to-apex / fall-after-apex violation fraction. higher=worse.

    Uses _robust_apex_idx. Left of apex must STRICTLY rise, right must
    STRICTLY fall; flat steps (diff==0) count as violations so plateaus /
    flat-tops are penalized (unlike the old monotonicity which treated
    flat as OK). A pure monotone ramp scores ~0 here BY DESIGN — its
    "badness" is that the apex sits at the window edge, which is captured
    separately by _calc_centering_defect.

    Returns 0.0 for empty / short (n<3) / non-finite XIC.
    """
    if len(intensity) < 3:
        return 0.0
    if not np.all(np.isfinite(intensity)):
        return 0.0
    apex_idx = _robust_apex_idx(intensity)
    left = intensity[:apex_idx + 1]
    right = intensity[apex_idx:]
    lv = int(np.sum(np.diff(left) <= 0)) if len(left) >= 2 else 0
    rv = int(np.sum(np.diff(right) >= 0)) if len(right) >= 2 else 0
    return (lv + rv) / max(1, len(intensity) - 1)
```

- [ ] **Step 4: 运行确认通过**

Run: `python -m pytest tests/test_single_work_numerics.py -q -k shape_irregularity`
Expected: PASS（5 passed）

- [ ] **Step 5: 提交**

```bash
git add workflows/single_work.py tests/test_single_work_numerics.py
git commit -m "feat: add _calc_shape_irregularity (strict rise/fall, plateau-aware)

Co-authored-by: Copilot <223556219+Copilot@users.noreply.github.com>"
```

---

### Task 3: `_calc_narrow_defect` 尖刺守卫（高=坏）

**Files:**
- Modify: `workflows/single_work.py`（紧接 `_calc_shape_irregularity` 之后插入）
- Test: `tests/test_single_work_numerics.py`（末尾追加）

- [ ] **Step 1: 写失败测试**

```python
def test_narrow_defect_single_cycle_spike_is_one():
    from workflows.single_work import _calc_narrow_defect
    assert _calc_narrow_defect(np.array([0, 0, 9, 0, 0], dtype="f8")) == 1.0


def test_narrow_defect_broad_peak_low():
    from workflows.single_work import _calc_narrow_defect
    # support (>=0.5*max) = 4 cycles -> 0.25
    r = _calc_narrow_defect(np.array([2, 5, 9, 9, 8, 5, 1], dtype="f8"))
    assert r <= 0.34


def test_narrow_defect_short_zero_nonfinite():
    from workflows.single_work import _calc_narrow_defect
    assert _calc_narrow_defect(np.array([1, 2], dtype="f8")) == 0.0
    assert _calc_narrow_defect(np.array([0, 0, 0], dtype="f8")) == 0.0
    assert _calc_narrow_defect(np.array([1, np.inf, 1], dtype="f8")) == 0.0
```

- [ ] **Step 2: 运行确认失败**

Run: `python -m pytest tests/test_single_work_numerics.py -q -k narrow_defect`
Expected: FAIL，`ImportError: cannot import name '_calc_narrow_defect'`

- [ ] **Step 3: 实现**

```python
def _calc_narrow_defect(intensity: np.ndarray) -> float:
    """1 / support, where support = #cycles with intensity >= 0.5*max.

    Approximates inverse FWHM-in-cycles. higher=worse: a single-cycle
    noise spike (support=1) -> 1.0; a broad chromatographic peak -> low.
    A spike scores ~0 on base_to_apex and 1.0 on the old monotonicity, so
    this is the dedicated penalty that catches it.

    Returns 0.0 for empty / short (n<3) / non-finite / all-zero XIC.
    """
    if len(intensity) < 3:
        return 0.0
    if not np.all(np.isfinite(intensity)):
        return 0.0
    max_v = float(np.max(intensity))
    if max_v <= 0:
        return 0.0
    support = int(np.sum(intensity >= 0.5 * max_v))
    return 1.0 / max(1, support)
```

- [ ] **Step 4: 运行确认通过**

Run: `python -m pytest tests/test_single_work_numerics.py -q -k narrow_defect`
Expected: PASS（3 passed）

- [ ] **Step 5: 提交**

```bash
git add workflows/single_work.py tests/test_single_work_numerics.py
git commit -m "feat: add _calc_narrow_defect (single-cycle spike guard)

Co-authored-by: Copilot <223556219+Copilot@users.noreply.github.com>"
```

---

### Task 4: `_calc_centering_defect` 归一化 apex 偏移（高=坏）

**Files:**
- Modify: `workflows/single_work.py`（紧接 `_calc_cycle_offset` 之后插入，约 line 1257 后）
- Test: `tests/test_single_work_numerics.py`（末尾追加）

依赖既有 helper `_calc_cycle_offset(xic, center_rt)` 与测试辅助 `_make_xic(cycles, rts, intensities)`（已存在于该测试文件，dtype 含 `cycle_idx`）。

- [ ] **Step 1: 写失败测试**

```python
def test_centering_defect_apex_at_center_is_zero():
    from workflows.single_work import _calc_centering_defect
    xic = _make_xic(cycles=[5, 6, 7, 8, 9], rts=[10, 11, 12, 13, 14],
                    intensities=[1, 5, 100, 5, 1])  # apex at center rt=12
    assert _calc_centering_defect(xic, center_rt=12.0) == 0.0


def test_centering_defect_apex_at_edge_near_one():
    from workflows.single_work import _calc_centering_defect
    # apex at last cycle (rt=14); center rt=12 -> offset 2, half=(5-1)/2=2 -> 1.0
    xic = _make_xic(cycles=[5, 6, 7, 8, 9], rts=[10, 11, 12, 13, 14],
                    intensities=[1, 1, 5, 1, 100])
    assert _calc_centering_defect(xic, center_rt=12.0) == 1.0


def test_centering_defect_empty_is_zero():
    from workflows.single_work import _calc_centering_defect
    xic = _make_xic(cycles=[], rts=[], intensities=[])
    assert _calc_centering_defect(xic, center_rt=0.0) == 0.0
```

- [ ] **Step 2: 运行确认失败**

Run: `python -m pytest tests/test_single_work_numerics.py -q -k centering_defect`
Expected: FAIL，`ImportError: cannot import name '_calc_centering_defect'`

- [ ] **Step 3: 实现**

```python
def _calc_centering_defect(xic: np.ndarray, center_rt: float) -> float:
    """|apex cycle offset| normalized by half the valid-cycle span.

    higher=worse, in [0, 1]: 0 = apex sits at the expected (center_rt)
    cycle; ~1 = apex sits at the window edge (e.g. an edge ramp whose true
    peak is clipped / outside the window). Reuses _calc_cycle_offset so the
    apex/center definition stays consistent with the existing offset cols.

    Returns 0.0 for empty XIC or zero offset.
    """
    if len(xic) == 0:
        return 0.0
    abs_off, _ = _calc_cycle_offset(xic, center_rt)
    if abs_off == 0:
        return 0.0
    valid = int(np.sum(xic["cycle_idx"] >= 0))
    half = max(1.0, (valid - 1) / 2.0)
    return float(min(1.0, abs_off / half))
```

- [ ] **Step 4: 运行确认通过**

Run: `python -m pytest tests/test_single_work_numerics.py -q -k centering_defect`
Expected: PASS（3 passed）

- [ ] **Step 5: 提交**

```bash
git add workflows/single_work.py tests/test_single_work_numerics.py
git commit -m "feat: add _calc_centering_defect (normalized apex offset)

Co-authored-by: Copilot <223556219+Copilot@users.noreply.github.com>"
```

---

### Task 5: `extract_ion_numeric_features` 增加 `stats` 参数

为新缺陷指标只产出 `mean`+`max`（控制列数；`max`=最差碎片）。默认保持现有 4 聚合，向后兼容。

**Files:**
- Modify: `workflows/single_work.py:1055-1073`（`extract_ion_numeric_features`）
- Test: `tests/test_single_work_numerics.py`（末尾追加）

- [ ] **Step 1: 写失败测试**

```python
def test_extract_ion_numeric_features_stats_subset():
    from workflows.single_work import extract_ion_numeric_features
    out = extract_ion_numeric_features(
        [0.1, 0.5, 0.3, 0.9], "demo", stats=("mean", "max"))
    assert set(out.keys()) == {"demo_mean", "demo_max"}
    assert abs(out["demo_max"] - 0.9) < 1e-9


def test_extract_ion_numeric_features_stats_subset_empty():
    from workflows.single_work import extract_ion_numeric_features
    out = extract_ion_numeric_features([], "demo", stats=("mean", "max"))
    assert out == {"demo_mean": 0.0, "demo_max": 0.0}


def test_extract_ion_numeric_features_default_still_four():
    from workflows.single_work import extract_ion_numeric_features
    out = extract_ion_numeric_features([1.0, 2.0, 3.0], "demo")
    assert set(out.keys()) == {
        "demo_mean", "demo_p50", "demo_std", "demo_max"}
```

- [ ] **Step 2: 运行确认失败**

Run: `python -m pytest tests/test_single_work_numerics.py -q -k "stats_subset or default_still_four"`
Expected: FAIL（`stats` 参数不存在 / `TypeError`）

- [ ] **Step 3: 实现**

把 `extract_ion_numeric_features` 整体替换为：

```python
def extract_ion_numeric_features(
    values: list, prefix: str, stats: tuple = ("mean", "p50", "std", "max")
) -> dict:
    """
    对碎片级数值列表（如 apex_delta、mz_err、cycle_offset）计算指定统计量。
    清除 NaN/Inf 值后统计。stats 默认四项 (mean/p50/std/max)；缺陷类指标可传
    ("mean","max") 只取均值与最差碎片，以控制列数。
    """
    clean_vals = [v for v in values if not np.isnan(v) and np.isfinite(v)]
    funcs = {
        "mean": lambda a: float(np.mean(a)),
        "p50": lambda a: float(np.median(a)),
        "std": lambda a: float(np.std(a)),
        "max": lambda a: float(np.max(a)),
    }
    if len(clean_vals) == 0:
        return {f"{prefix}_{s}": 0.0 for s in stats}
    arr = np.asarray(clean_vals, dtype="f8")
    return {f"{prefix}_{s}": funcs[s](arr) for s in stats}
```

- [ ] **Step 4: 运行确认通过**

Run: `python -m pytest tests/test_single_work_numerics.py -q -k "extract_ion_numeric_features"`
Expected: PASS（含既有 `_max` 测试，全部通过）

- [ ] **Step 5: 提交**

```bash
git add workflows/single_work.py tests/test_single_work_numerics.py
git commit -m "feat: add stats= param to extract_ion_numeric_features

Co-authored-by: Copilot <223556219+Copilot@users.noreply.github.com>"
```

---

### Task 6: `calc_xic_score` 计算并返回新缺陷键（轻+重）

新键暂与旧 `apex_monotonicity` 并存（旧键在 Task 12 移除），保证每步测试绿。

**Files:**
- Modify: `workflows/single_work.py`：`_default_xic_score`（1285-1307）、`calc_xic_score` 峰形块（1364-1367 后）、`rt_start>=rt_end` 分支（1388 后）、主返回 dict（1445-1465）
- Test: `tests/test_single_work_numerics.py`（末尾追加）

- [ ] **Step 1: 写失败测试**

```python
def _peak_xic(intensities, cycles=None, rts=None):
    dt = [("rt", "f8"), ("ppm_error", "f8"),
          ("intensity", "f8"), ("cycle_idx", "i4")]
    n = len(intensities)
    arr = np.zeros(n, dtype=dt)
    arr["intensity"] = intensities
    arr["rt"] = rts if rts is not None else np.arange(n, dtype="f8") + 10.0
    arr["cycle_idx"] = cycles if cycles is not None else np.arange(n)
    return arr


def test_calc_xic_score_emits_new_shape_defect_keys():
    from workflows.single_work import calc_xic_score
    light = _peak_xic([1, 5, 50, 100, 50, 5, 1])
    heavy = light.copy()
    r = calc_xic_score(light, heavy, center_rt=13.0)
    for k in ("light_centering_defect", "heavy_centering_defect",
              "light_shape_irregularity", "heavy_shape_irregularity",
              "light_base_to_apex_ratio", "light_n_peaks",
              "light_smoothness", "light_narrow_defect",
              "heavy_narrow_defect"):
        assert k in r, f"missing {k}"
    # clean centered peak -> defects all low
    assert r["heavy_shape_irregularity"] == 0.0
    assert r["heavy_centering_defect"] == 0.0


def test_calc_xic_score_default_has_new_shape_defect_keys():
    from workflows.single_work import _default_xic_score
    d = _default_xic_score()
    for k in ("light_centering_defect", "heavy_centering_defect",
              "light_shape_irregularity", "heavy_shape_irregularity",
              "light_base_to_apex_ratio", "light_n_peaks",
              "light_smoothness", "light_narrow_defect",
              "heavy_narrow_defect"):
        assert d[k] == 0 or d[k] == 0.0, f"{k} default not zero"
```

- [ ] **Step 2: 运行确认失败**

Run: `python -m pytest tests/test_single_work_numerics.py -q -k "new_shape_defect"`
Expected: FAIL（缺键 / KeyError）

- [ ] **Step 3a: 扩充 `_default_xic_score`**

在 `workflows/single_work.py` 的 `_default_xic_score` 返回 dict 中，`"smoothness": 0.0,` 之后插入：

```python
        "light_centering_defect": 0.0,
        "heavy_centering_defect": 0.0,
        "light_shape_irregularity": 0.0,
        "heavy_shape_irregularity": 0.0,
        "light_base_to_apex_ratio": 0.0,
        "light_n_peaks": 0,
        "light_smoothness": 0.0,
        "light_narrow_defect": 0.0,
        "heavy_narrow_defect": 0.0,
```

- [ ] **Step 3b: 计算新指标 + 组装 `_shape_defects`**

在 `calc_xic_score` 中，紧接 `smoothness = _calc_smoothness(heavy_xic["intensity"])`（约 line 1367）之后插入：

```python
    # P-AS1: light-channel peak shape + edge/centering/spike defects.
    # All "higher = worse" so the fragment _max aggregate surfaces the
    # worst fragment. apex_monotonicity (heavy, good=high) is kept for now
    # and removed in a later cleanup once consumers migrate.
    heavy_shape_irregularity = _calc_shape_irregularity(heavy_xic["intensity"])
    light_shape_irregularity = _calc_shape_irregularity(light_xic["intensity"])
    light_base_to_apex_ratio = _calc_base_to_apex_ratio(light_xic["intensity"])
    light_n_peaks = _calc_n_peaks(light_xic["intensity"])
    light_smoothness = _calc_smoothness(light_xic["intensity"])
    heavy_narrow_defect = _calc_narrow_defect(heavy_xic["intensity"])
    light_narrow_defect = _calc_narrow_defect(light_xic["intensity"])
    if center_rt is not None:
        _h_center = (heavy_center_rt
                     if heavy_center_rt is not None else center_rt)
        light_centering_defect = _calc_centering_defect(light_xic, center_rt)
        heavy_centering_defect = _calc_centering_defect(heavy_xic, _h_center)
    else:
        light_centering_defect = 0.0
        heavy_centering_defect = 0.0
    _shape_defects = {
        "light_centering_defect": light_centering_defect,
        "heavy_centering_defect": heavy_centering_defect,
        "light_shape_irregularity": light_shape_irregularity,
        "heavy_shape_irregularity": heavy_shape_irregularity,
        "light_base_to_apex_ratio": light_base_to_apex_ratio,
        "light_n_peaks": light_n_peaks,
        "light_smoothness": light_smoothness,
        "light_narrow_defect": light_narrow_defect,
        "heavy_narrow_defect": heavy_narrow_defect,
    }
```

- [ ] **Step 3c: `rt_start>=rt_end` 分支合入**

在该分支中，`result["smoothness"] = smoothness` 之后、`if center_rt is not None:` 之前插入：

```python
        result.update(_shape_defects)
```

- [ ] **Step 3d: 主返回合入**

把主返回的 `return {` 改为 `result = {`，并在该 dict 的闭合 `}`（`"smoothness": smoothness,` 的下一行 `}`）之后替换为：

```python
    }
    result.update(_shape_defects)
    return result
```

- [ ] **Step 4: 运行确认通过**

Run: `python -m pytest tests/test_single_work_numerics.py -q -k "new_shape_defect or calc_xic_score or peak_likeness"`
Expected: PASS（新键测试通过；既有 calc_xic_score 测试仍绿）

- [ ] **Step 5: 提交**

```bash
git add workflows/single_work.py tests/test_single_work_numerics.py
git commit -m "feat: calc_xic_score emits light+heavy shape-defect features

Co-authored-by: Copilot <223556219+Copilot@users.noreply.github.com>"
```

---

### Task 7: 母离子装配 helper

把"从 `precursor_score` 取新列"封装为一处，供 `multi_batch_work` 与 `single_pair_work` 共用。

**Files:**
- Modify: `workflows/single_work.py`（在 `calc_xic_score` 之后、`plot_light_heavy_xic` 之前插入；模块级常量 + 两个函数）
- Test: `tests/test_single_work_numerics.py`（末尾追加）

- [ ] **Step 1: 写失败测试**

```python
def test_precursor_shape_cols_keys_and_values():
    from workflows.single_work import (
        _precursor_shape_cols, _empty_precursor_shape_cols)
    score = {
        "light_centering_defect": 0.2, "heavy_centering_defect": 0.3,
        "light_shape_irregularity": 0.1, "heavy_shape_irregularity": 0.4,
        "light_base_to_apex_ratio": 0.5, "light_n_peaks": 2,
        "light_smoothness": 0.6, "light_narrow_defect": 0.7,
        "heavy_narrow_defect": 0.8,
    }
    cols = _precursor_shape_cols(score)
    assert cols["precursor_heavy_shape_irregularity"] == 0.4
    assert cols["precursor_light_n_peaks"] == 2
    assert cols["precursor_heavy_narrow_defect"] == 0.8
    empty = _empty_precursor_shape_cols()
    assert set(empty.keys()) == set(cols.keys())
    assert all(v == 0 or v == 0.0 for v in empty.values())
    assert "precursor_apex_monotonicity" not in cols
```

- [ ] **Step 2: 运行确认失败**

Run: `python -m pytest tests/test_single_work_numerics.py -q -k precursor_shape_cols`
Expected: FAIL，`ImportError`

- [ ] **Step 3: 实现**

```python
# col_name -> calc_xic_score result key. New precursor shape-defect cols.
# Heavy base_to_apex/n_peaks/smoothness stay as their existing unprefixed
# precursor_* lines and are NOT included here.
_PRECURSOR_SHAPE_MAP = {
    "precursor_light_centering_defect": "light_centering_defect",
    "precursor_heavy_centering_defect": "heavy_centering_defect",
    "precursor_light_shape_irregularity": "light_shape_irregularity",
    "precursor_heavy_shape_irregularity": "heavy_shape_irregularity",
    "precursor_light_base_to_apex_ratio": "light_base_to_apex_ratio",
    "precursor_light_n_peaks": "light_n_peaks",
    "precursor_light_smoothness": "light_smoothness",
    "precursor_light_narrow_defect": "light_narrow_defect",
    "precursor_heavy_narrow_defect": "heavy_narrow_defect",
}


def _precursor_shape_cols(score: dict) -> dict:
    """Map calc_xic_score result -> precursor_* shape-defect feature cols."""
    return {col: score[key] for col, key in _PRECURSOR_SHAPE_MAP.items()}


def _empty_precursor_shape_cols() -> dict:
    """Zero defaults for the empty-XIC branch (schema parity)."""
    return {col: 0.0 for col in _PRECURSOR_SHAPE_MAP}
```

- [ ] **Step 4: 运行确认通过**

Run: `python -m pytest tests/test_single_work_numerics.py -q -k precursor_shape_cols`
Expected: PASS

- [ ] **Step 5: 提交**

```bash
git add workflows/single_work.py tests/test_single_work_numerics.py
git commit -m "feat: add precursor shape-defect column assembler helpers

Co-authored-by: Copilot <223556219+Copilot@users.noreply.github.com>"
```

---

### Task 8: 碎片峰形累加器 helper

把碎片新缺陷指标的"收集 → 聚合"封装为累加器，避免在 4 个循环点散落大量 append。轻标键受 `light_fragment_shape` 开关控制。

**Files:**
- Modify: `workflows/single_work.py`（紧接 Task 7 的母离子 helper 之后插入）
- Test: `tests/test_single_work_numerics.py`（末尾追加）

- [ ] **Step 1: 写失败测试**

```python
def test_fragment_shape_acc_roundtrip_with_light():
    from workflows.single_work import (
        _new_fragment_shape_acc, _append_fragment_shape,
        _append_empty_fragment_shape, _fragment_shape_aggregates)
    acc = _new_fragment_shape_acc(light_fragment_shape=True)
    score = {
        "heavy_centering_defect": 0.9, "heavy_shape_irregularity": 0.1,
        "heavy_narrow_defect": 0.2, "light_centering_defect": 0.3,
        "light_shape_irregularity": 0.4, "light_base_to_apex_ratio": 0.5,
        "light_n_peaks": 1, "light_smoothness": 0.6, "light_narrow_defect": 0.7,
    }
    _append_fragment_shape(acc, score)
    _append_empty_fragment_shape(acc)  # second "fragment" all-zero
    out = _fragment_shape_aggregates(acc)
    assert out["all_heavy_centering_defect_max"] == 0.9   # worst fragment
    assert abs(out["all_heavy_centering_defect_mean"] - 0.45) < 1e-9
    assert "all_light_n_peaks_max" in out
    # only mean + max emitted
    assert "all_heavy_centering_defect_p50" not in out
    assert "all_heavy_centering_defect_std" not in out


def test_fragment_shape_acc_without_light_omits_light_keys():
    from workflows.single_work import (
        _new_fragment_shape_acc, _fragment_shape_aggregates)
    acc = _new_fragment_shape_acc(light_fragment_shape=False)
    out = _fragment_shape_aggregates(acc)
    assert "all_heavy_centering_defect_mean" in out
    assert not any(k.startswith("all_light_") for k in out)
```

- [ ] **Step 2: 运行确认失败**

Run: `python -m pytest tests/test_single_work_numerics.py -q -k fragment_shape_acc`
Expected: FAIL，`ImportError`

- [ ] **Step 3: 实现**

```python
_FRAG_SHAPE_HEAVY = (
    "heavy_centering_defect", "heavy_shape_irregularity", "heavy_narrow_defect")
_FRAG_SHAPE_LIGHT = (
    "light_centering_defect", "light_shape_irregularity",
    "light_base_to_apex_ratio", "light_n_peaks", "light_smoothness",
    "light_narrow_defect")


def _new_fragment_shape_acc(light_fragment_shape: bool) -> dict:
    """Empty accumulator: one list per per-fragment shape-defect metric.
    Light-channel keys are included only when light_fragment_shape is on."""
    keys = list(_FRAG_SHAPE_HEAVY)
    if light_fragment_shape:
        keys += list(_FRAG_SHAPE_LIGHT)
    return {k: [] for k in keys}


def _append_fragment_shape(acc: dict, ion_score: dict) -> None:
    """Append this fragment's metrics from a calc_xic_score result."""
    for k in acc:
        acc[k].append(ion_score[k])


def _append_empty_fragment_shape(acc: dict) -> None:
    """Append zeros for a fragment whose XIC pair was empty."""
    for k in acc:
        acc[k].append(0.0)


def _fragment_shape_aggregates(acc: dict) -> dict:
    """Aggregate each accumulated list to all_<key>_{mean,max}."""
    out = {}
    for k, vals in acc.items():
        out.update(extract_ion_numeric_features(
            vals, f"all_{k}", stats=("mean", "max")))
    return out
```

- [ ] **Step 4: 运行确认通过**

Run: `python -m pytest tests/test_single_work_numerics.py -q -k fragment_shape_acc`
Expected: PASS（2 passed）

- [ ] **Step 5: 提交**

```bash
git add workflows/single_work.py tests/test_single_work_numerics.py
git commit -m "feat: add fragment shape-defect accumulator helpers

Co-authored-by: Copilot <223556219+Copilot@users.noreply.github.com>"
```

---

### Task 9: ConfigKeys 增加 `LIGHT_FRAGMENT_SHAPE`

**Files:**
- Modify: `constant/keys.py`（`GENERAL` 段，`RANDOM_SEED = "random_seed"` 之后）

无独立单测（纯常量）；由 Task 10/11 的行为测试覆盖。

- [ ] **Step 1: 实现**

在 `constant/keys.py` 的 `RANDOM_SEED = "random_seed"` 行之后插入：

```python
    # 轻标【碎片】峰形特征开关（默认 on）。off 时跳过 all_light_* 碎片缺陷列，
    # 便于消融。母离子轻标形状不受此开关影响。见
    # docs/superpowers/specs/2026-06-20-xic-apex-shape-penalty-design.md
    LIGHT_FRAGMENT_SHAPE = "light_fragment_shape"
```

- [ ] **Step 2: 验证导入**

Run: `python -c "from constant.keys import ConfigKeys; print(ConfigKeys.LIGHT_FRAGMENT_SHAPE)"`
Expected: 输出 `light_fragment_shape`

- [ ] **Step 3: 提交**

```bash
git add constant/keys.py
git commit -m "feat: add LIGHT_FRAGMENT_SHAPE config key

Co-authored-by: Copilot <223556219+Copilot@users.noreply.github.com>"
```

---

### Task 10: 接线 `multi_batch_work`

**Files:**
- Modify: `workflows/single_work.py`（`multi_batch_work`：约 57、97、127-128、198、265、312-313、401-409）
- Test: `tests/test_single_work_numerics.py`（更新 R4 测试 + 加新键测试）

- [ ] **Step 1: 更新/新增测试**

把 `test_multi_batch_work_emits_R4_precursor_keys_in_empty_xic_branch` 里的 `R4_PRECURSOR_KEYS` 集合替换为：

```python
    R4_PRECURSOR_KEYS = {
        "precursor_base_to_apex_ratio",
        "precursor_heavy_shape_irregularity",
        "precursor_n_peaks",
        "precursor_smoothness",
        "precursor_light_centering_defect",
        "precursor_heavy_narrow_defect",
    }
```

并在该测试函数末尾追加断言（旧键已移除）：

```python
    assert "precursor_apex_monotonicity" not in features
```

- [ ] **Step 2: 运行确认失败**

Run: `python -m pytest tests/test_single_work_numerics.py -q -k multi_batch_work_emits_R4`
Expected: FAIL（当前仍输出 `precursor_apex_monotonicity`，且缺新键）

- [ ] **Step 3a: 读开关**

在 `multi_batch_work` 内，`xic_cycle_window = config[...].getint(...)` 之后插入：

```python
    light_fragment_shape = config[ConfigKeys.GENERAL].getboolean(
        ConfigKeys.LIGHT_FRAGMENT_SHAPE, fallback=True)
```

- [ ] **Step 3b: 母离子空分支**

将该行：
```python
        features["precursor_apex_monotonicity"] = 0.0
```
替换为：
```python
        features.update(_empty_precursor_shape_cols())
```

- [ ] **Step 3c: 母离子 else 分支**

将这两行：
```python
        features["precursor_apex_monotonicity"] = (
            precursor_score["apex_monotonicity"])
```
替换为：
```python
        features.update(_precursor_shape_cols(precursor_score))
```

- [ ] **Step 3d: 碎片列表声明**

将该行：
```python
    fragment_apex_monotonicities = []
```
替换为：
```python
    frag_shape = _new_fragment_shape_acc(light_fragment_shape)
```

- [ ] **Step 3e: 碎片空 append 分支**

将该行：
```python
            fragment_apex_monotonicities.append(0.0)
```
替换为：
```python
            _append_empty_fragment_shape(frag_shape)
```

- [ ] **Step 3f: 碎片打分 append 分支**

将这两行：
```python
        fragment_apex_monotonicities.append(
            ion_score["apex_monotonicity"])
```
替换为：
```python
        _append_fragment_shape(frag_shape, ion_score)
```

- [ ] **Step 3g: 碎片聚合**

将这两行：
```python
    features.update(extract_ion_numeric_features(
        fragment_apex_monotonicities, "all_apex_monotonicity"))
```
替换为：
```python
    features.update(_fragment_shape_aggregates(frag_shape))
```

- [ ] **Step 4: 运行确认通过**

Run: `python -m pytest tests/test_single_work_numerics.py -q -k "multi_batch_work"`
Expected: PASS

- [ ] **Step 5: 提交**

```bash
git add workflows/single_work.py tests/test_single_work_numerics.py
git commit -m "feat: wire shape-defect features into multi_batch_work

Co-authored-by: Copilot <223556219+Copilot@users.noreply.github.com>"
```

---

### Task 11: 接线 `single_pair_work`

**Files:**
- Modify: `workflows/single_work.py`（`single_pair_work`：约 478、518、546-547、622、714、758-759、843-845）
- Test: `tests/test_deep_audit_p0.py`（复用 `_FakePSM`/`_FakeDIA`/`_minimal_config`，加 parity 测试）
- Test: `tests/test_deep_audit_p1.py:133`（改名聚合断言）

- [ ] **Step 1: 写失败测试**

在 `tests/test_deep_audit_p0.py` 末尾追加：

```python
def test_single_and_multi_emit_same_precursor_shape_defect_keys():
    """single_pair_work 与 multi_batch_work 的母离子缺陷列必须同名同集，
    否则 concat 会产生 NaN 列。"""
    from workflows.single_work import single_pair_work, multi_batch_work
    psm = _FakePSM()
    dia = _FakeDIA(force_empty=False)
    cfg = _minimal_config()
    sf = single_pair_work(psm, dia, cfg)
    mf = multi_batch_work(psm, dia, psm, dia, cfg)
    new_keys = {
        "precursor_light_centering_defect", "precursor_heavy_centering_defect",
        "precursor_light_shape_irregularity",
        "precursor_heavy_shape_irregularity",
        "precursor_light_base_to_apex_ratio", "precursor_light_n_peaks",
        "precursor_light_smoothness", "precursor_light_narrow_defect",
        "precursor_heavy_narrow_defect",
    }
    assert new_keys <= set(sf.keys()), new_keys - set(sf.keys())
    assert new_keys <= set(mf.keys()), new_keys - set(mf.keys())
    assert "precursor_apex_monotonicity" not in sf
    # fragment aggregates present (mean+max), monotonicity aggregate gone
    assert "all_heavy_centering_defect_max" in sf
    assert "all_apex_monotonicity_mean" not in sf
```

- [ ] **Step 2: 运行确认失败**

Run: `python -m pytest tests/test_deep_audit_p0.py -q -k same_precursor_shape_defect`
Expected: FAIL（single_pair_work 尚未接线）

- [ ] **Step 3a: 读开关**

在 `single_pair_work` 内，`xic_cycle_window = config[...].getint(...)` 之后插入：

```python
    light_fragment_shape = config[ConfigKeys.GENERAL].getboolean(
        ConfigKeys.LIGHT_FRAGMENT_SHAPE, fallback=True)
```

- [ ] **Step 3b: 母离子空分支** — 将 `single_pair_work` 中的
```python
        features["precursor_apex_monotonicity"] = 0.0
```
替换为：
```python
        features.update(_empty_precursor_shape_cols())
```

- [ ] **Step 3c: 母离子 else 分支** — 将
```python
        features["precursor_apex_monotonicity"] = (
            precursor_score["apex_monotonicity"])
```
替换为：
```python
        features.update(_precursor_shape_cols(precursor_score))
```

- [ ] **Step 3d: 碎片列表声明** — 将 `single_pair_work` 中的
```python
    fragment_apex_monotonicities = []
```
替换为：
```python
    frag_shape = _new_fragment_shape_acc(light_fragment_shape)
```

- [ ] **Step 3e: 碎片空 append** — 将
```python
            fragment_apex_monotonicities.append(0.0)
```
替换为：
```python
            _append_empty_fragment_shape(frag_shape)
```

- [ ] **Step 3f: 碎片打分 append** — 将
```python
        fragment_apex_monotonicities.append(
            ion_score["apex_monotonicity"])
```
替换为：
```python
        _append_fragment_shape(frag_shape, ion_score)
```

- [ ] **Step 3g: 碎片聚合** — 将 `single_pair_work` 中的
```python
    features.update(extract_ion_numeric_features(
        fragment_apex_monotonicities, "all_apex_monotonicity"))
```
替换为：
```python
    features.update(_fragment_shape_aggregates(frag_shape))
```

> 注：`single_pair_work` 调用 `calc_xic_score(light_ions_xic, heavy_ions_xic, center_rt=float(psm._rt))`（无 `heavy_center_rt`），新键照常返回，无需改该调用。

- [ ] **Step 3h: 更新 `test_deep_audit_p1.py` 聚合断言**

在 `tests/test_deep_audit_p1.py`（约 line 133），将 key 列表里的
```python
        "all_apex_monotonicity_mean",
```
替换为：
```python
        "all_heavy_shape_irregularity_mean",
```

- [ ] **Step 4: 运行确认通过**

Run: `python -m pytest tests/test_deep_audit_p0.py tests/test_deep_audit_p1.py -q -k "same_precursor_shape_defect or single_pair or fragment_empty_branch_aggregates"`
Expected: PASS

- [ ] **Step 5: 提交**

```bash
git add workflows/single_work.py tests/test_deep_audit_p0.py tests/test_deep_audit_p1.py
git commit -m "feat: wire shape-defect features into single_pair_work

Co-authored-by: Copilot <223556219+Copilot@users.noreply.github.com>"
```

---

### Task 12: 移除 `apex_monotonicity`（helper + 键 + 旧测试）

此时所有消费方已迁移到 `shape_irregularity`，可安全删除旧实现。

**Files:**
- Modify: `workflows/single_work.py`（删 3 处 `apex_monotonicity` 赋值、`_default` 键、`_calc_apex_monotonicity` 定义）
- Test: `tests/test_single_work_numerics.py`（删旧 helper 测试、改 2 处 peak-likeness 断言）

- [ ] **Step 1: 改测试（先改测试，红→绿）**

(a) 删除三个旧 helper 测试函数：`test_calc_apex_monotonicity_perfect_peak_returns_one`、`test_calc_apex_monotonicity_zigzag_returns_low`、`test_calc_apex_monotonicity_apex_at_edge`、`test_calc_apex_monotonicity_nan_returns_zero`（约 line 469-524，连同各自函数体整段删除）。

(b) `test_calc_xic_score_emits_peak_likeness_fields`：将
```python
    assert "apex_monotonicity" in result
    assert result["apex_monotonicity"] == 1.0
```
替换为：
```python
    assert "heavy_shape_irregularity" in result
    assert result["heavy_shape_irregularity"] == 0.0
```

(c) `test_default_xic_score_has_peak_likeness_zero_fields`：将
```python
    assert d["apex_monotonicity"] == 0.0
```
替换为：
```python
    assert d["heavy_shape_irregularity"] == 0.0
```

- [ ] **Step 2: 删源码引用**

在 `workflows/single_work.py`：

(a) 删 `_default_xic_score` 中的 `"apex_monotonicity": 0.0,` 行。

(b) 删 `calc_xic_score` 中的 `apex_monotonicity = _calc_apex_monotonicity(heavy_xic["intensity"])` 行。

(c) 删 `rt_start>=rt_end` 分支中的 `result["apex_monotonicity"] = apex_monotonicity` 行。

(d) 删主返回 dict 中的 `"apex_monotonicity": apex_monotonicity,` 行。

(e) 删整个 `_calc_apex_monotonicity` 函数定义（约 line 1145-1171）。

- [ ] **Step 3: 运行确认通过**

Run: `python -m pytest tests/test_single_work_numerics.py -q`
Expected: PASS（无 `apex_monotonicity` 残留引用；`grep -rn "apex_monotonicity" workflows/ tests/` 仅余 0 处）

校验：
```bash
grep -rn "apex_monotonicity\|_calc_apex_monotonicity" workflows/ tests/ --include=*.py | grep -v __pycache__
```
Expected: 无输出

- [ ] **Step 4: 提交**

```bash
git add workflows/single_work.py tests/test_single_work_numerics.py
git commit -m "refactor: remove apex_monotonicity (replaced by shape_irregularity)

Co-authored-by: Copilot <223556219+Copilot@users.noreply.github.com>"
```

---

### Task 13: 更新列契约固定件 + 计数

新 schema：移除 5 个旧特征列（`precursor_apex_monotonicity` + `all_apex_monotonicity_{mean,p50,std,max}`），新增 27 个（母离子 9 + 碎片 18）。特征数 142 → 164，总列数 155 → 177。契约测试只读表头，列顺序无关。

> 备注：`tools/eval_feature_ablation.py` 按前缀（`precursor_*`/`all_*`）分组、不硬编码 `apex_monotonicity`，故无需改动。

**Files:**
- Create: `tests/fixtures/features_header_177.csv`（由旧固定件确定性变换得到，仅表头一行）
- Delete: `tests/fixtures/features_header_155.csv`
- Modify: `tests/test_feature_cols_contract.py`（`_FIXTURE` 路径 + `==142`→`==164` + docstring）

- [ ] **Step 1: 生成新固定件（确定性脚本）**

```bash
python - <<'PY'
OLD = "tests/fixtures/features_header_155.csv"
NEW = "tests/fixtures/features_header_177.csv"
with open(OLD) as f:
    header = f.readline().strip().split(",")
remove = {
    "precursor_apex_monotonicity",
    "all_apex_monotonicity_mean", "all_apex_monotonicity_p50",
    "all_apex_monotonicity_std", "all_apex_monotonicity_max",
}
add = [
    "precursor_light_centering_defect", "precursor_heavy_centering_defect",
    "precursor_light_shape_irregularity", "precursor_heavy_shape_irregularity",
    "precursor_light_base_to_apex_ratio", "precursor_light_n_peaks",
    "precursor_light_smoothness", "precursor_light_narrow_defect",
    "precursor_heavy_narrow_defect",
    "all_heavy_centering_defect_mean", "all_heavy_centering_defect_max",
    "all_heavy_shape_irregularity_mean", "all_heavy_shape_irregularity_max",
    "all_heavy_narrow_defect_mean", "all_heavy_narrow_defect_max",
    "all_light_centering_defect_mean", "all_light_centering_defect_max",
    "all_light_shape_irregularity_mean", "all_light_shape_irregularity_max",
    "all_light_base_to_apex_ratio_mean", "all_light_base_to_apex_ratio_max",
    "all_light_n_peaks_mean", "all_light_n_peaks_max",
    "all_light_smoothness_mean", "all_light_smoothness_max",
    "all_light_narrow_defect_mean", "all_light_narrow_defect_max",
]
new_header = [c for c in header if c not in remove] + add
assert len(new_header) == 177, len(new_header)
with open(NEW, "w") as f:
    f.write(",".join(new_header) + "\n")
print("wrote", NEW, "cols", len(new_header))
PY
git rm tests/fixtures/features_header_155.csv
```

Expected: 打印 `wrote tests/fixtures/features_header_177.csv cols 177`

- [ ] **Step 2: 更新契约测试**

在 `tests/test_feature_cols_contract.py`：

(a) docstring 中 `155 cols -> 142 features` 改为 `177 cols -> 164 features`。

(b) 将
```python
_FIXTURE = os.path.join(os.path.dirname(os.path.abspath(__file__)),
                        "fixtures", "features_header_155.csv")
```
改为
```python
_FIXTURE = os.path.join(os.path.dirname(os.path.abspath(__file__)),
                        "fixtures", "features_header_177.csv")
```

(c) 将 `def test_real_header_resolves_to_142_features` 函数名改为 `test_real_header_resolves_to_164_features`，断言 `assert len(feats) == 164`。

- [ ] **Step 3: 运行确认通过**

Run: `python -m pytest tests/test_feature_cols_contract.py -q`
Expected: PASS（5 passed）

- [ ] **Step 4: 提交**

```bash
git add tests/fixtures/features_header_177.csv tests/test_feature_cols_contract.py
git commit -m "test: update feature-cols contract fixture for shape-defect schema (142->164)

Co-authored-by: Copilot <223556219+Copilot@users.noreply.github.com>"
```

---

### Task 14: 集成 — no-NaN 行为测试 + 全量套件

**Files:**
- Test: `tests/test_deep_audit_p0.py`（追加 no-NaN 断言，复用 `_FakePSM`/`_FakeDIA`）

- [ ] **Step 1: 写测试**

在 `tests/test_deep_audit_p0.py` 末尾追加：

```python
def test_shape_defect_cols_are_finite_not_nan():
    """所有新缺陷列在真实 stub 路径下必须是有限值（非 NaN/Inf）。"""
    import numpy as np
    from workflows.single_work import single_pair_work, multi_batch_work
    psm = _FakePSM()
    dia = _FakeDIA(force_empty=False)
    cfg = _minimal_config()
    new_cols = [
        "precursor_light_centering_defect", "precursor_heavy_centering_defect",
        "precursor_light_shape_irregularity",
        "precursor_heavy_shape_irregularity",
        "precursor_light_base_to_apex_ratio", "precursor_light_n_peaks",
        "precursor_light_smoothness", "precursor_light_narrow_defect",
        "precursor_heavy_narrow_defect",
        "all_heavy_centering_defect_max", "all_heavy_shape_irregularity_max",
        "all_heavy_narrow_defect_max", "all_light_centering_defect_max",
        "all_light_shape_irregularity_max", "all_light_base_to_apex_ratio_max",
        "all_light_n_peaks_max", "all_light_smoothness_max",
        "all_light_narrow_defect_max",
    ]
    for feats in (single_pair_work(psm, dia, cfg),
                  multi_batch_work(psm, dia, psm, dia, cfg)):
        for c in new_cols:
            assert c in feats, f"missing {c}"
            v = feats[c]
            assert np.isfinite(v), f"{c} non-finite: {v}"


def test_light_fragment_shape_off_drops_light_fragment_cols():
    """开关 off 时 all_light_* 碎片列消失，母离子轻标列仍在。"""
    import configparser
    from workflows.single_work import single_pair_work
    psm = _FakePSM()
    dia = _FakeDIA(force_empty=False)
    cfg = configparser.ConfigParser()
    cfg.read_dict({"general": {
        "mass_tol_ppm": "20", "xic_cycle_window": "5",
        "light_fragment_shape": "false"}})
    feats = single_pair_work(psm, dia, cfg)
    assert not any(k.startswith("all_light_") and
                   ("centering_defect" in k or "shape_irregularity" in k or
                    "narrow_defect" in k or "base_to_apex" in k or
                    "_n_peaks_" in k or "smoothness" in k)
                   for k in feats), "light fragment shape cols should be gone"
    assert "precursor_light_centering_defect" in feats  # precursor unaffected
    assert "all_heavy_centering_defect_max" in feats     # heavy unaffected
```

- [ ] **Step 2: 运行新测试**

Run: `python -m pytest tests/test_deep_audit_p0.py -q -k "shape_defect_cols_are_finite or light_fragment_shape_off"`
Expected: PASS

- [ ] **Step 3: 全量套件**

Run: `python -m pytest tests/ -q`
Expected: 全绿。若有失败，多半是漏改的 `apex_monotonicity` 引用——按报错定位修正后再跑。

- [ ] **Step 4: 提交**

```bash
git add tests/test_deep_audit_p0.py
git commit -m "test: assert shape-defect cols finite + light_fragment_shape toggle

Co-authored-by: Copilot <223556219+Copilot@users.noreply.github.com>"
```

---

### Task 15: 端到端验证（固定 clean hard-test）— 需真实数据

此任务在拿到/具备 DIA 原始数据后执行，用于确认新特征带来可测量改善而非退化。非阻塞 CI，属人工 checkpoint。

**Files:** 无源码改动（仅运行 + 记录结论）

- [ ] **Step 1: 重抽特征（提取已内置过滤）**

```bash
# 三个数据集 × clean + neg 变体（按需）。提取在 main.py 内已自动过滤
# heavy_out_of_range，无需单独 make filter。
make all              # 2th/5th/normal clean -> runs/baseline_*_clean/features.csv
make all-neg10        # 如需 neg 变体
```
Expected: 各 `runs/baseline_*/features.csv` 重新生成，含新列。

- [ ] **Step 2: 校验新列存在且无 NaN**

```bash
python - <<'PY'
import pandas as pd, glob, numpy as np
new = ["precursor_heavy_shape_irregularity", "precursor_light_centering_defect",
       "all_heavy_centering_defect_max", "all_light_narrow_defect_max"]
for p in glob.glob("runs/baseline_*/features.csv"):
    df = pd.read_csv(p, nrows=2000)
    miss = [c for c in new if c not in df.columns]
    nan = [c for c in new if c in df.columns and df[c].isna().any()]
    print(p, "missing", miss, "nan", nan)
PY
```
Expected: 每个文件 `missing [] nan []`。

- [ ] **Step 3: 训练（in-distribution + cross_test）**

```bash
make train-clean-all      # 或对应 neg 变体的训练目标（见 make help）
```
Expected: `runs_new/spec_trainer/models/in_*.txt`、`results/in_*.json` 重新生成。

- [ ] **Step 4: 固定 clean hard-test FNR@FPR≤5% 对比**

用既有评估协议（训练于某 neg 层、仅在 clean q≤0.01 负例上测）比较改前/改后。改前基线见会话此前记录（2da/5da/normal 的 FNR@FPR5%）。

```bash
# 复用既有离线评估脚本（/usr/bin/python3 + 纯 numpy LGB 文本模型预测器），
# 对每个 in_*_clean 模型在其 held-out clean 负例上算 FNR@FPR=5%。
# 记录三个数据集改前→改后的 FNR 值。
```
Expected（成功判据）：2da/5da/normal 至少一个数据集 FNR@FPR5% 下降，且无数据集明显上升（>2 个点）。预期主杠杆在碎片（`all_*_shape_irregularity`/`all_*_centering_defect` 的 `_max`）。

- [ ] **Step 5: 新特征重要性 + decoy 复核**

```bash
# (a) 看新列 gain importance（从 in_*.txt 解析），确认非零、合理。
# (b) 复核点名 decoy：2da QQITDLER z2、EQQEIAER z2 在新 features.csv 中
#     precursor/all 的 centering_defect / shape_irregularity / narrow_defect
#     应升高（apex 不在中心被惩罚）。
python - <<'PY'
import pandas as pd
df = pd.read_csv("runs/baseline_2da_clean/features.csv")
cols = ["sequence", "precursor_heavy_centering_defect",
        "precursor_light_centering_defect", "precursor_heavy_narrow_defect",
        "precursor_light_narrow_defect"]
print(df[df["sequence"].isin(["QQITDLER", "EQQEIAER"])][cols].to_string())
PY
```
Expected: 两个 decoy 的 centering/narrow 缺陷值显著高于正例中位数。

- [ ] **Step 6: 记录结论 + 提交（若有阈值微调）**

将 FNR 改前/改后表、重要性、decoy 复核结论记入 checkpoint。若 `flat_tol`(0.95)/`narrow_defect`(0.5 阈) 等需微调，在此迭代后再 commit；否则本任务无 commit。

```bash
# 如有阈值微调：
git add workflows/single_work.py
git commit -m "tune: adjust shape-defect thresholds per hard-test validation

Co-authored-by: Copilot <223556219+Copilot@users.noreply.github.com>"
```

---

## 自查（Self-Review）

- **Spec 覆盖**：六类坏形态（边缘斜坡/平台/多峰簇/锯齿/尖刺/偏移）→ Task 2(irregularity)/3(narrow)/4(centering) + Task 6 装配；轻重双通道 + 母离子(Task 7/10/11) + 碎片(Task 8/10/11)；"高=坏"+`_max` 暴露最差碎片(Task 8)；config 开关(Task 9)；鲁棒 apex(Task 1)；改名清理(Task 12)；契约(Task 13)；验证(Task 15)。✅ 全覆盖。
- **占位符**：无 TBD/TODO；每个代码步给出完整代码与命令。✅
- **类型/命名一致**：`_shape_defects` 键 ↔ `_PRECURSOR_SHAPE_MAP` 值 ↔ `_FRAG_SHAPE_HEAVY/LIGHT` ↔ 固定件 27 新列名，逐一核对一致。`stats=("mean","max")` 统一。✅
- **测试绿色链**：每个改名/删除点都在引发断裂的同一 Task 内更新对应测试（Task 10→688/R4；Task 11→deep_audit_p1:133；Task 12→helper/peak-likeness；Task 13→契约计数）。✅

