# Heavy XIC Peak-Likeness Features Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Implement 4 heavy-XIC peak-likeness features (`base_to_apex_ratio` / `apex_monotonicity` / `n_peaks` / `smoothness`) per `docs/specs/2026-06-02-heavy-xic-peak-likeness-features-design.md`, producing 20 new CSV columns (4 precursor + 16 fragment aggregates) to give 5Da-window negative-PSM detection a non-shape-similarity discriminator.

**Architecture:** Add 4 new private `_calc_*` helpers in `workflows/single_work.py` following the same pattern as the existing `_calc_fwhm` / `_calc_symmetry` / `_calc_snr`. Extend `calc_xic_score` to call them on `heavy_xic` (same design as existing `snr`/`peak_symmetry`). Wire 4 new feature outputs symmetrically into `single_pair_work` precursor block + fragment loop + `multi_batch_work` mirror. Reuse `extract_ion_numeric_features` for fragment aggregation (auto-emits mean/p50/std/max). Backward compatible: defaults to 0 in all early-return paths.

**Tech Stack:** Python 3.13, numpy, scipy.signal.find_peaks (already used via scipy.stats.pearsonr), pytest, conda env `jianyan` at `/home/verden/.conda/envs/jianyan`.

---

## File Structure

**Modified files:**
- `workflows/single_work.py` — Add 4 `_calc_*` helpers; extend `_default_xic_score`, `calc_xic_score`; wire into `single_pair_work` and `multi_batch_work`.
- `tests/test_single_work_numerics.py` — Add 12 TDD tests (3 per helper) + 1 roster check.

No new files. No npz cache changes. No new dependencies.

---

## Task 1: `_calc_base_to_apex_ratio` helper

**Files:**
- Modify: `workflows/single_work.py` (add helper after `_calc_snr`, around line 862)
- Test: `tests/test_single_work_numerics.py`

- [ ] **Step 1: Write the failing tests**

Append to `tests/test_single_work_numerics.py`:

```python
def test_calc_base_to_apex_ratio_real_peak_returns_low_value():
    """Real chromatographic peak: edges decay to near-zero -> ratio close to 0."""
    from workflows.single_work import _calc_base_to_apex_ratio
    intensity = np.array([1, 5, 50, 100, 50, 5, 1], dtype="f8")
    ratio = _calc_base_to_apex_ratio(intensity)
    # base = (1+1)/2 = 1, apex = 100, ratio = 0.01
    assert ratio < 0.05


def test_calc_base_to_apex_ratio_plateau_returns_high():
    """Plateau / continuous background: edges are nearly as high as apex."""
    from workflows.single_work import _calc_base_to_apex_ratio
    intensity = np.array([80, 90, 100, 100, 90, 80, 80], dtype="f8")
    ratio = _calc_base_to_apex_ratio(intensity)
    # base = (80+80)/2 = 80, apex = 100, ratio = 0.8
    assert ratio > 0.7


def test_calc_base_to_apex_ratio_edge_cases():
    """Empty / short / all-zero XIC returns 0.0."""
    from workflows.single_work import _calc_base_to_apex_ratio
    assert _calc_base_to_apex_ratio(np.array([], dtype="f8")) == 0.0
    assert _calc_base_to_apex_ratio(np.array([1, 2], dtype="f8")) == 0.0
    assert _calc_base_to_apex_ratio(np.array([0, 0, 0], dtype="f8")) == 0.0
```

- [ ] **Step 2: Run tests to verify they fail**

```bash
conda run -n jianyan pytest tests/test_single_work_numerics.py::test_calc_base_to_apex_ratio_real_peak_returns_low_value -v
```

Expected: FAIL with `ImportError: cannot import name '_calc_base_to_apex_ratio'`.

- [ ] **Step 3: Implement `_calc_base_to_apex_ratio`**

In `workflows/single_work.py`, add this helper right after `_calc_snr` (around line 862, just before `_calc_cycle_offset`):

```python
def _calc_base_to_apex_ratio(intensity: np.ndarray) -> float:
    """Edge average / apex intensity.

    True peaks: edges decay to near 0 -> ratio close to 0.
    Plateau / background / multi-peak stacks -> ratio close to 1.
    """
    if len(intensity) < 3:
        return 0.0
    apex = float(np.max(intensity))
    if apex <= 0:
        return 0.0
    base = (float(intensity[0]) + float(intensity[-1])) / 2
    return base / apex
```

- [ ] **Step 4: Run tests to verify they pass**

```bash
conda run -n jianyan pytest tests/test_single_work_numerics.py -v -k base_to_apex
```

Expected: All 3 PASS.

- [ ] **Step 5: Commit**

```bash
git add workflows/single_work.py tests/test_single_work_numerics.py
git commit -m "feat(single_work): _calc_base_to_apex_ratio helper

True chromatographic peaks decay to near-zero at edges:
ratio = (intensity[0] + intensity[-1]) / 2 / max(intensity)
Real peak -> ~0; plateau/background -> ~1.

Co-authored-by: Copilot <223556219+Copilot@users.noreply.github.com>"
```

---

## Task 2: `_calc_apex_monotonicity` helper

**Files:**
- Modify: `workflows/single_work.py` (add helper after `_calc_base_to_apex_ratio`)
- Test: `tests/test_single_work_numerics.py`

- [ ] **Step 1: Write the failing tests**

Append to `tests/test_single_work_numerics.py`:

```python
def test_calc_apex_monotonicity_perfect_peak_returns_one():
    """Strictly monotonic up to apex, then strictly down: monotonicity = 1.0."""
    from workflows.single_work import _calc_apex_monotonicity
    intensity = np.array([1, 5, 50, 100, 50, 5, 1], dtype="f8")
    # All 6 diffs go in the "right" direction -> 0 violations -> 1.0
    assert _calc_apex_monotonicity(intensity) == 1.0


def test_calc_apex_monotonicity_zigzag_returns_low():
    """Zigzag (multiple direction reversals) returns low monotonicity."""
    from workflows.single_work import _calc_apex_monotonicity
    intensity = np.array([10, 50, 20, 100, 30, 60, 5], dtype="f8")
    # apex at idx 3, left=[10,50,20,100] diffs [40,-30,80] -> 1 viol (50->20)
    # right=[100,30,60,5] diffs [-70,30,-55] -> 1 viol (30->60)
    # total_pairs = 6, violations = 2, monotonicity = 1 - 2/6 ~= 0.667
    result = _calc_apex_monotonicity(intensity)
    assert 0.5 < result < 0.75


def test_calc_apex_monotonicity_apex_at_edge():
    """Apex at index 0 or last index: function handles boundary safely."""
    from workflows.single_work import _calc_apex_monotonicity
    # Apex at idx 0: left=[100], right=[100,50,10] diffs [-50,-40] all <=0 -> 0 viol
    result_left = _calc_apex_monotonicity(
        np.array([100, 50, 10], dtype="f8"))
    assert result_left == 1.0
    # Apex at last idx: left=[10,50,100] diffs [40,50] -> 0 viol, right=[100] -> 0 pairs
    result_right = _calc_apex_monotonicity(
        np.array([10, 50, 100], dtype="f8"))
    assert result_right == 1.0
    # Edge: empty / short
    assert _calc_apex_monotonicity(np.array([], dtype="f8")) == 0.0
    assert _calc_apex_monotonicity(np.array([1, 2], dtype="f8")) == 0.0
```

- [ ] **Step 2: Run to verify failure**

```bash
conda run -n jianyan pytest tests/test_single_work_numerics.py -v -k apex_monotonicity
```

Expected: FAIL with `ImportError`.

- [ ] **Step 3: Implement `_calc_apex_monotonicity`**

Add right after `_calc_base_to_apex_ratio` in `workflows/single_work.py`:

```python
def _calc_apex_monotonicity(intensity: np.ndarray) -> float:
    """Fraction of pairs that monotonically rise to apex and fall after.

    Left of apex should be non-decreasing; right of apex should be
    non-increasing. Return = 1 - (violations / total_pairs) in [0, 1].
    True peaks -> ~1; zigzag / noise -> low.

    Note: right slice includes apex (intensity[apex_idx:]) so when apex
    is at the leftmost index there is still a meaningful right slice.
    """
    if len(intensity) < 3:
        return 0.0
    apex_idx = int(np.argmax(intensity))
    left = intensity[:apex_idx + 1]
    right = intensity[apex_idx:]
    if len(left) < 2 and len(right) < 2:
        return 0.0
    left_viol = int(np.sum(np.diff(left) < 0)) if len(left) >= 2 else 0
    right_viol = int(np.sum(np.diff(right) > 0)) if len(right) >= 2 else 0
    total_pairs = max(len(intensity) - 1, 1)
    return 1.0 - (left_viol + right_viol) / total_pairs
```

- [ ] **Step 4: Run tests**

```bash
conda run -n jianyan pytest tests/test_single_work_numerics.py -v -k apex_monotonicity
```

Expected: All 3 PASS.

- [ ] **Step 5: Commit**

```bash
git add workflows/single_work.py tests/test_single_work_numerics.py
git commit -m "feat(single_work): _calc_apex_monotonicity helper

Fraction of pairs monotonically rising to apex and falling after.
True peak -> 1.0; zigzag / noise -> low. Handles apex-at-edge safely.

Co-authored-by: Copilot <223556219+Copilot@users.noreply.github.com>"
```

---

## Task 3: `_calc_n_peaks` helper

**Files:**
- Modify: `workflows/single_work.py` (add helper after `_calc_apex_monotonicity`)
- Test: `tests/test_single_work_numerics.py`

- [ ] **Step 1: Write the failing tests**

Append to `tests/test_single_work_numerics.py`:

```python
def test_calc_n_peaks_single_peak_returns_one():
    """Classic unimodal peak: find_peaks returns 1 local maximum."""
    from workflows.single_work import _calc_n_peaks
    intensity = np.array([1, 5, 50, 100, 50, 5, 1], dtype="f8")
    assert _calc_n_peaks(intensity) == 1


def test_calc_n_peaks_bimodal_returns_two():
    """Two well-separated peaks both with prominence > 0.3 * apex."""
    from workflows.single_work import _calc_n_peaks
    intensity = np.array([1, 100, 1, 1, 80, 1, 1], dtype="f8")
    # Two peaks both >= 80 (>= 0.3 * 100 = 30) and dipping back to 1 between
    assert _calc_n_peaks(intensity) == 2


def test_calc_n_peaks_small_noise_suppressed_by_prominence():
    """Small noise bump << 30% of apex should be filtered out."""
    from workflows.single_work import _calc_n_peaks
    # Main peak at idx 3 with apex=100; small bump at idx 6 with height=20 (20%)
    intensity = np.array([1, 50, 100, 50, 1, 1, 20, 1], dtype="f8")
    # Only the main peak should count
    assert _calc_n_peaks(intensity) == 1


def test_calc_n_peaks_edge_cases():
    """Empty / short / all-zero XIC returns 0."""
    from workflows.single_work import _calc_n_peaks
    assert _calc_n_peaks(np.array([], dtype="f8")) == 0
    assert _calc_n_peaks(np.array([1, 2], dtype="f8")) == 0
    assert _calc_n_peaks(np.array([0, 0, 0, 0, 0], dtype="f8")) == 0
```

- [ ] **Step 2: Run to verify failure**

```bash
conda run -n jianyan pytest tests/test_single_work_numerics.py -v -k n_peaks
```

Expected: FAIL with `ImportError`.

- [ ] **Step 3: Implement `_calc_n_peaks`**

Add right after `_calc_apex_monotonicity` in `workflows/single_work.py`:

```python
def _calc_n_peaks(
    intensity: np.ndarray, prominence_frac: float = 0.3
) -> int:
    """Count local maxima with prominence >= prominence_frac * apex.

    True chromatographic peak -> 1; co-elution / interference -> 2+.
    The prominence threshold filters out small bumps that are likely
    noise rather than separate peaks. Endpoints are not counted.
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

- [ ] **Step 4: Run tests**

```bash
conda run -n jianyan pytest tests/test_single_work_numerics.py -v -k n_peaks
```

Expected: All 4 PASS.

- [ ] **Step 5: Commit**

```bash
git add workflows/single_work.py tests/test_single_work_numerics.py
git commit -m "feat(single_work): _calc_n_peaks helper

Counts local maxima using scipy.signal.find_peaks with prominence
threshold = 0.3 * apex to filter noise bumps. True peak -> 1;
co-elution stacks -> 2+.

Co-authored-by: Copilot <223556219+Copilot@users.noreply.github.com>"
```

---

## Task 4: `_calc_smoothness` helper

**Files:**
- Modify: `workflows/single_work.py` (add helper after `_calc_n_peaks`)
- Test: `tests/test_single_work_numerics.py`

- [ ] **Step 1: Write the failing tests**

Append to `tests/test_single_work_numerics.py`:

```python
def test_calc_smoothness_smooth_curve_low_value():
    """Smooth Gaussian-like peak has low smoothness value."""
    from workflows.single_work import _calc_smoothness
    intensity = np.array([1, 5, 50, 100, 50, 5, 1], dtype="f8")
    # Compute: sum of (second diff)^2 / total^2 - should be small
    s = _calc_smoothness(intensity)
    assert 0 < s < 0.05


def test_calc_smoothness_zigzag_high_value():
    """Sharp zigzag / single-point spike has high smoothness value."""
    from workflows.single_work import _calc_smoothness
    smooth = _calc_smoothness(
        np.array([1, 5, 50, 100, 50, 5, 1], dtype="f8"))
    zigzag = _calc_smoothness(
        np.array([1, 100, 1, 100, 1, 100, 1], dtype="f8"))
    assert zigzag > 10 * smooth  # zigzag should be at least 10x smoother peak


def test_calc_smoothness_edge_cases():
    """Empty / short / all-zero XIC returns 0.0."""
    from workflows.single_work import _calc_smoothness
    assert _calc_smoothness(np.array([], dtype="f8")) == 0.0
    assert _calc_smoothness(np.array([1, 2], dtype="f8")) == 0.0
    assert _calc_smoothness(np.array([0, 0, 0, 0, 0], dtype="f8")) == 0.0
```

- [ ] **Step 2: Run to verify failure**

```bash
conda run -n jianyan pytest tests/test_single_work_numerics.py -v -k smoothness
```

Expected: FAIL with `ImportError`.

- [ ] **Step 3: Implement `_calc_smoothness`**

Add right after `_calc_n_peaks` in `workflows/single_work.py`:

```python
def _calc_smoothness(intensity: np.ndarray) -> float:
    """Sum of squared second differences / total^2.

    Smooth Gaussian-like peaks -> close to 0.
    Sharp zigzag / single-point spikes -> large value.
    Normalized by total^2 to make cross-sample comparable; note this
    is NOT normalized by length, so different xic_cycle_window settings
    produce different absolute values.
    """
    if len(intensity) < 3:
        return 0.0
    total = float(np.sum(intensity))
    if total <= 0:
        return 0.0
    second_diff = np.diff(intensity, n=2)
    return float(np.sum(second_diff ** 2) / (total ** 2 + 1e-12))
```

- [ ] **Step 4: Run tests**

```bash
conda run -n jianyan pytest tests/test_single_work_numerics.py -v -k smoothness
```

Expected: All 3 PASS.

- [ ] **Step 5: Commit**

```bash
git add workflows/single_work.py tests/test_single_work_numerics.py
git commit -m "feat(single_work): _calc_smoothness helper

Sum of squared second differences / total^2. Smooth peaks -> ~0;
zigzag / sharp spikes -> large value. Total^2 normalization makes
cross-sample comparable.

Co-authored-by: Copilot <223556219+Copilot@users.noreply.github.com>"
```

---

## Task 5: Extend `_default_xic_score` and `calc_xic_score`

**Files:**
- Modify: `workflows/single_work.py` `_default_xic_score` (~line 918) and `calc_xic_score` (~line 939)
- Test: `tests/test_single_work_numerics.py`

- [ ] **Step 1: Write the failing test**

Append to `tests/test_single_work_numerics.py`:

```python
def test_calc_xic_score_emits_peak_likeness_fields():
    """calc_xic_score returns 4 new peak-likeness fields, all computed
    on the heavy XIC (consistent with existing snr/peak_symmetry)."""
    from workflows.single_work import calc_xic_score
    dt = [("rt", "f8"), ("ppm_error", "f8"),
          ("intensity", "f8"), ("cycle_idx", "i4")]
    n = 7
    light = np.zeros(n, dtype=dt)
    light["rt"] = [10, 11, 12, 13, 14, 15, 16]
    light["cycle_idx"] = [0, 1, 2, 3, 4, 5, 6]
    light["intensity"] = [1, 5, 50, 100, 50, 5, 1]
    heavy = light.copy()
    # Heavy is a clean peak - all 4 metrics should be peak-like
    result = calc_xic_score(light, heavy)
    assert "base_to_apex_ratio" in result
    assert result["base_to_apex_ratio"] < 0.05
    assert "apex_monotonicity" in result
    assert result["apex_monotonicity"] == 1.0
    assert "n_peaks" in result
    assert result["n_peaks"] == 1
    assert "smoothness" in result
    assert 0 < result["smoothness"] < 0.05


def test_default_xic_score_has_peak_likeness_zero_fields():
    """The default-zero dict must include all 4 new peak-likeness keys."""
    from workflows.single_work import _default_xic_score
    d = _default_xic_score()
    assert d["base_to_apex_ratio"] == 0.0
    assert d["apex_monotonicity"] == 0.0
    assert d["n_peaks"] == 0
    assert d["smoothness"] == 0.0
```

- [ ] **Step 2: Run to verify failure**

```bash
conda run -n jianyan pytest tests/test_single_work_numerics.py -v -k "peak_likeness or default_xic_score_has_peak"
```

Expected: FAIL (`base_to_apex_ratio` not in result).

- [ ] **Step 3: Update `_default_xic_score`**

In `workflows/single_work.py`, find `_default_xic_score` (~line 918-936). It currently returns 15 fields. Replace the return dict to add 4 new fields at the END (after `heavy_apex_cycle_offset_signed`):

OLD (around line 920-936):
```python
def _default_xic_score() -> dict:
    """calc_xic_score 的全零默认返回值"""
    return {
        "pearson": np.float32(0.0),
        "mz_avg_err": 0.0,
        "apex_delta": 0.0,
        "apex_delta_signed": 0.0,
        "light_max_int": 0.0,
        "heavy_max_int": 0.0,
        "intensity_ratio": 0.0,
        "cosine": 0.0,
        "snr": 0.0,
        "peak_width_ratio": 0.0,
        "peak_symmetry": 0.0,
        "light_apex_cycle_offset": 0,
        "light_apex_cycle_offset_signed": 0,
        "heavy_apex_cycle_offset": 0,
        "heavy_apex_cycle_offset_signed": 0,
    }
```

NEW:
```python
def _default_xic_score() -> dict:
    """calc_xic_score 的全零默认返回值"""
    return {
        "pearson": np.float32(0.0),
        "mz_avg_err": 0.0,
        "apex_delta": 0.0,
        "apex_delta_signed": 0.0,
        "light_max_int": 0.0,
        "heavy_max_int": 0.0,
        "intensity_ratio": 0.0,
        "cosine": 0.0,
        "snr": 0.0,
        "peak_width_ratio": 0.0,
        "peak_symmetry": 0.0,
        "light_apex_cycle_offset": 0,
        "light_apex_cycle_offset_signed": 0,
        "heavy_apex_cycle_offset": 0,
        "heavy_apex_cycle_offset_signed": 0,
        "base_to_apex_ratio": 0.0,
        "apex_monotonicity": 0.0,
        "n_peaks": 0,
        "smoothness": 0.0,
    }
```

- [ ] **Step 4: Update `calc_xic_score` to compute the 4 new fields**

In `workflows/single_work.py`, find the block in `calc_xic_score` (around line 978-984):

OLD:
```python
    # 峰形特征（在原始 XIC 上计算，不依赖插值）
    snr = _calc_snr(heavy_xic["intensity"])
    peak_symmetry = _calc_symmetry(heavy_xic["intensity"])
    light_fwhm = _calc_fwhm(light_xic["rt"], light_xic["intensity"])
    heavy_fwhm = _calc_fwhm(heavy_xic["rt"], heavy_xic["intensity"])
    peak_width_ratio = (heavy_fwhm / light_fwhm
                        if light_fwhm > 0 else 0.0)
```

NEW (append 4 new lines):
```python
    # 峰形特征（在原始 XIC 上计算，不依赖插值）
    snr = _calc_snr(heavy_xic["intensity"])
    peak_symmetry = _calc_symmetry(heavy_xic["intensity"])
    light_fwhm = _calc_fwhm(light_xic["rt"], light_xic["intensity"])
    heavy_fwhm = _calc_fwhm(heavy_xic["rt"], heavy_xic["intensity"])
    peak_width_ratio = (heavy_fwhm / light_fwhm
                        if light_fwhm > 0 else 0.0)
    base_to_apex_ratio = _calc_base_to_apex_ratio(heavy_xic["intensity"])
    apex_monotonicity = _calc_apex_monotonicity(heavy_xic["intensity"])
    n_peaks = _calc_n_peaks(heavy_xic["intensity"])
    smoothness = _calc_smoothness(heavy_xic["intensity"])
```

- [ ] **Step 5: Update `calc_xic_score` early-return path (rt_start >= rt_end)**

Find the early-return block in `calc_xic_score` (around line 991-1008):

OLD:
```python
    if rt_start >= rt_end:
        result = _default_xic_score()
        result["mz_avg_err"] = mz_avg_err
        result["apex_delta"] = apex_delta
        result["apex_delta_signed"] = apex_delta_signed
        result["light_max_int"] = light_max_int
        result["heavy_max_int"] = heavy_max_int
        result["intensity_ratio"] = intensity_ratio
        if center_rt is not None:
            l_abs, l_sig = _calc_cycle_offset(light_xic, center_rt)
            h_center = (heavy_center_rt
                        if heavy_center_rt is not None else center_rt)
            h_abs, h_sig = _calc_cycle_offset(heavy_xic, h_center)
            result["light_apex_cycle_offset"] = l_abs
            result["light_apex_cycle_offset_signed"] = l_sig
            result["heavy_apex_cycle_offset"] = h_abs
            result["heavy_apex_cycle_offset_signed"] = h_sig
        return result
```

NEW (add 4 lines populating peak-likeness from already-computed locals):
```python
    if rt_start >= rt_end:
        result = _default_xic_score()
        result["mz_avg_err"] = mz_avg_err
        result["apex_delta"] = apex_delta
        result["apex_delta_signed"] = apex_delta_signed
        result["light_max_int"] = light_max_int
        result["heavy_max_int"] = heavy_max_int
        result["intensity_ratio"] = intensity_ratio
        result["snr"] = snr
        result["peak_width_ratio"] = peak_width_ratio
        result["peak_symmetry"] = peak_symmetry
        result["base_to_apex_ratio"] = base_to_apex_ratio
        result["apex_monotonicity"] = apex_monotonicity
        result["n_peaks"] = n_peaks
        result["smoothness"] = smoothness
        if center_rt is not None:
            l_abs, l_sig = _calc_cycle_offset(light_xic, center_rt)
            h_center = (heavy_center_rt
                        if heavy_center_rt is not None else center_rt)
            h_abs, h_sig = _calc_cycle_offset(heavy_xic, h_center)
            result["light_apex_cycle_offset"] = l_abs
            result["light_apex_cycle_offset_signed"] = l_sig
            result["heavy_apex_cycle_offset"] = h_abs
            result["heavy_apex_cycle_offset_signed"] = h_sig
        return result
```

Note: The early-return path previously did NOT populate snr / peak_width_ratio / peak_symmetry. This fix also adds them, making the early-return schema consistent with the normal-return schema. This is a side-effect bug fix worth noting in the commit message.

- [ ] **Step 6: Update `calc_xic_score` final return**

Find the final return dict (around line 1055-1071):

OLD:
```python
    return {
        "pearson": np.float32(corr),
        "mz_avg_err": mz_avg_err,
        "apex_delta": apex_delta,
        "apex_delta_signed": apex_delta_signed,
        "light_max_int": light_max_int,
        "heavy_max_int": heavy_max_int,
        "intensity_ratio": intensity_ratio,
        "cosine": cosine,
        "snr": snr,
        "peak_width_ratio": peak_width_ratio,
        "peak_symmetry": peak_symmetry,
        "light_apex_cycle_offset": l_abs,
        "light_apex_cycle_offset_signed": l_sig,
        "heavy_apex_cycle_offset": h_abs,
        "heavy_apex_cycle_offset_signed": h_sig,
    }
```

NEW (add 4 fields at END):
```python
    return {
        "pearson": np.float32(corr),
        "mz_avg_err": mz_avg_err,
        "apex_delta": apex_delta,
        "apex_delta_signed": apex_delta_signed,
        "light_max_int": light_max_int,
        "heavy_max_int": heavy_max_int,
        "intensity_ratio": intensity_ratio,
        "cosine": cosine,
        "snr": snr,
        "peak_width_ratio": peak_width_ratio,
        "peak_symmetry": peak_symmetry,
        "light_apex_cycle_offset": l_abs,
        "light_apex_cycle_offset_signed": l_sig,
        "heavy_apex_cycle_offset": h_abs,
        "heavy_apex_cycle_offset_signed": h_sig,
        "base_to_apex_ratio": base_to_apex_ratio,
        "apex_monotonicity": apex_monotonicity,
        "n_peaks": n_peaks,
        "smoothness": smoothness,
    }
```

- [ ] **Step 7: Run tests**

```bash
conda run -n jianyan pytest tests/test_single_work_numerics.py -v
```

Expected: All tests pass (the 2 new tests plus all existing tests).

- [ ] **Step 8: Commit**

```bash
git add workflows/single_work.py tests/test_single_work_numerics.py
git commit -m "feat(single_work): calc_xic_score emits 4 peak-likeness fields

Add base_to_apex_ratio / apex_monotonicity / n_peaks / smoothness to
calc_xic_score returned dict and _default_xic_score. All four computed
on heavy_xic (consistent with existing snr/peak_symmetry).

Side-effect fix: rt_start>=rt_end early-return path now also populates
snr/peak_width_ratio/peak_symmetry (previously zeroed via
_default_xic_score), making it schema-consistent with the normal path.

Co-authored-by: Copilot <223556219+Copilot@users.noreply.github.com>"
```

---

## Task 6: Wire peak-likeness into `single_pair_work` precursor block

**Files:**
- Modify: `workflows/single_work.py` `single_pair_work` (~line 367-405)

- [ ] **Step 1: Modify precursor block**

In `workflows/single_work.py`, find the precursor block in `single_pair_work` (around lines 367-405). Update BOTH branches.

For the empty-XIC branch, find:
```python
        features["precursor_heavy_apex_cycle_offset"] = 0
        features["precursor_heavy_apex_cycle_offset_signed"] = 0
```

Append immediately after these lines:
```python
        features["precursor_base_to_apex_ratio"] = 0.0
        features["precursor_apex_monotonicity"] = 0.0
        features["precursor_n_peaks"] = 0
        features["precursor_smoothness"] = 0.0
```

For the normal-path else-branch, find:
```python
        features["precursor_heavy_apex_cycle_offset_signed"] = (
            precursor_score["heavy_apex_cycle_offset_signed"])
```

Append immediately after this line:
```python
        features["precursor_base_to_apex_ratio"] = (
            precursor_score["base_to_apex_ratio"])
        features["precursor_apex_monotonicity"] = (
            precursor_score["apex_monotonicity"])
        features["precursor_n_peaks"] = precursor_score["n_peaks"]
        features["precursor_smoothness"] = precursor_score["smoothness"]
```

- [ ] **Step 2: Smoke-test that import succeeds**

```bash
conda run -n jianyan python -c "from workflows.single_work import single_pair_work; print('OK')"
```

Expected: `OK`.

- [ ] **Step 3: Run full test suite**

```bash
conda run -n jianyan pytest tests/ -v -x
```

Expected: All tests pass.

- [ ] **Step 4: Commit**

```bash
git add workflows/single_work.py
git commit -m "feat(single_work): precursor peak-likeness in single_pair_work

Wire 4 new precursor_* peak-likeness features through single_pair_work
precursor block (empty-path + normal-path both updated).

Co-authored-by: Copilot <223556219+Copilot@users.noreply.github.com>"
```

---

## Task 7: Wire fragment-level peak-likeness into `single_pair_work`

**Files:**
- Modify: `workflows/single_work.py` `single_pair_work` fragment loop (~line 466-545) and aggregation (~line 597-615)

- [ ] **Step 1: Add fragment-level collection lists**

In `workflows/single_work.py`, find the existing collection lists (~line 466-475):
```python
    fragment_apex_deltas = []
    fragment_mz_errs = []
    fragment_intensities = []  # per-ion max intensity for weighted correlation
    fragment_cosines = []
    fragment_snrs = []
    fragment_light_cycle_offsets = []
    fragment_light_cycle_offsets_signed = []
    fragment_heavy_cycle_offsets = []
    fragment_heavy_cycle_offsets_signed = []
    fragment_hl_ratios = {"all": [], "b": [], "y": []}
```

Append 4 new lists (at end of this block, before `ion_data = []`):
```python
    fragment_base_to_apex_ratios = []
    fragment_apex_monotonicities = []
    fragment_n_peaks_list = []
    fragment_smoothnesses = []
```

- [ ] **Step 2: Add per-fragment .append statements**

Find the existing fragment append block (~lines 528-549, inside the `for ions_type, ions_num, light_mass, heavy_mass in fragment_ions:` loop, after `if ion_score["intensity_ratio"] > 0:` block).

Find:
```python
        if ion_score["intensity_ratio"] > 0:
            fragment_hl_ratios[ions_type].append(
                float(ion_score["intensity_ratio"]))
            fragment_hl_ratios["all"].append(
                float(ion_score["intensity_ratio"]))
```

Append immediately after (before `ion_data.append(...)`):
```python
        fragment_base_to_apex_ratios.append(
            ion_score["base_to_apex_ratio"])
        fragment_apex_monotonicities.append(
            ion_score["apex_monotonicity"])
        fragment_n_peaks_list.append(ion_score["n_peaks"])
        fragment_smoothnesses.append(ion_score["smoothness"])
```

- [ ] **Step 3: Add post-loop aggregation**

In `workflows/single_work.py`, find the existing H/L ratio aggregation block in `single_pair_work` (around lines 611-615):
```python
    # H/L 强度比一致性（按 all/b/y 分组的 log10-ratio std/mad）
    for ion_type, ratios in fragment_hl_ratios.items():
        std_v, mad_v = _calc_hl_ratio_consistency(ratios)
        features[f"{ion_type}_log_hl_ratio_std"] = std_v
        features[f"{ion_type}_log_hl_ratio_mad"] = mad_v
```

Append immediately after (before `# 序列级特征`):
```python
    # 碎片级 peak-likeness 汇总（heavy XIC × {mean,p50,std,max}）
    features.update(extract_ion_numeric_features(
        fragment_base_to_apex_ratios, "all_base_to_apex_ratio"))
    features.update(extract_ion_numeric_features(
        fragment_apex_monotonicities, "all_apex_monotonicity"))
    features.update(extract_ion_numeric_features(
        fragment_n_peaks_list, "all_n_peaks"))
    features.update(extract_ion_numeric_features(
        fragment_smoothnesses, "all_smoothness"))
```

- [ ] **Step 4: Smoke-test**

```bash
conda run -n jianyan python -c "from workflows.single_work import single_pair_work; print('OK')"
```

Expected: `OK`.

- [ ] **Step 5: Run full test suite**

```bash
conda run -n jianyan pytest tests/ -v -x
```

Expected: All tests pass.

- [ ] **Step 6: Commit**

```bash
git add workflows/single_work.py
git commit -m "feat(single_work): fragment peak-likeness in single_pair_work

Collect per-fragment 4 peak-likeness values inside the fragment loop,
aggregate via extract_ion_numeric_features after the loop:
- 16 new columns: all_{base_to_apex_ratio,apex_monotonicity,n_peaks,smoothness}_{mean,p50,std,max}

Co-authored-by: Copilot <223556219+Copilot@users.noreply.github.com>"
```

---

## Task 8: Mirror changes into `multi_batch_work`

**Files:**
- Modify: `workflows/single_work.py` `multi_batch_work` (precursor ~line 54-94, fragment loop ~line 135-320)

`multi_batch_work` is structurally identical to `single_pair_work` but operates on two PSMs / two DIAData. Mirror all changes from Tasks 6 + 7.

- [ ] **Step 1: Update precursor block in `multi_batch_work`**

In `workflows/single_work.py`, find the precursor block (around lines 54-94). Mirror Task 6's changes:

For the empty-XIC branch, find:
```python
        features["precursor_heavy_apex_cycle_offset"] = 0
        features["precursor_heavy_apex_cycle_offset_signed"] = 0
```

Append immediately after:
```python
        features["precursor_base_to_apex_ratio"] = 0.0
        features["precursor_apex_monotonicity"] = 0.0
        features["precursor_n_peaks"] = 0
        features["precursor_smoothness"] = 0.0
```

For the normal-path else-branch, find:
```python
        features["precursor_heavy_apex_cycle_offset_signed"] = (
            precursor_score["heavy_apex_cycle_offset_signed"])
```

Append immediately after:
```python
        features["precursor_base_to_apex_ratio"] = (
            precursor_score["base_to_apex_ratio"])
        features["precursor_apex_monotonicity"] = (
            precursor_score["apex_monotonicity"])
        features["precursor_n_peaks"] = precursor_score["n_peaks"]
        features["precursor_smoothness"] = precursor_score["smoothness"]
```

- [ ] **Step 2: Add fragment collection lists in `multi_batch_work`**

Find the collection-list block in `multi_batch_work` (look for `fragment_hl_ratios = {"all": [], "b": [], "y": []}` line in `multi_batch_work`, around line 135-160). Right after `fragment_hl_ratios = ...`, append:

```python
    fragment_base_to_apex_ratios = []
    fragment_apex_monotonicities = []
    fragment_n_peaks_list = []
    fragment_smoothnesses = []
```

- [ ] **Step 3: Add per-fragment .append in `multi_batch_work` fragment loop**

Find the per-fragment append block inside the `multi_batch_work` fragment loop (search for the SECOND occurrence of `if ion_score["intensity_ratio"] > 0:` and the H/L ratio appends). After the H/L ratio block in `multi_batch_work`, append:

```python
        fragment_base_to_apex_ratios.append(
            ion_score["base_to_apex_ratio"])
        fragment_apex_monotonicities.append(
            ion_score["apex_monotonicity"])
        fragment_n_peaks_list.append(ion_score["n_peaks"])
        fragment_smoothnesses.append(ion_score["smoothness"])
```

- [ ] **Step 4: Add post-loop aggregation in `multi_batch_work`**

Find the H/L consistency aggregation block in `multi_batch_work` (search for the FIRST occurrence of `for ion_type, ratios in fragment_hl_ratios.items():` — there are two, one in each function). After the H/L ratio loop in `multi_batch_work`, before `# 序列级特征`, append:

```python
    # 碎片级 peak-likeness 汇总（heavy XIC × {mean,p50,std,max}）
    features.update(extract_ion_numeric_features(
        fragment_base_to_apex_ratios, "all_base_to_apex_ratio"))
    features.update(extract_ion_numeric_features(
        fragment_apex_monotonicities, "all_apex_monotonicity"))
    features.update(extract_ion_numeric_features(
        fragment_n_peaks_list, "all_n_peaks"))
    features.update(extract_ion_numeric_features(
        fragment_smoothnesses, "all_smoothness"))
```

- [ ] **Step 5: Smoke-test**

```bash
conda run -n jianyan python -c "from workflows.single_work import multi_batch_work, single_pair_work; print('imports ok')"
```

Expected: `imports ok`.

- [ ] **Step 6: Run full test suite**

```bash
conda run -n jianyan pytest tests/ -v -x
```

Expected: All tests pass.

- [ ] **Step 7: Commit**

```bash
git add workflows/single_work.py
git commit -m "feat(single_work): mirror peak-likeness into multi_batch_work

Both single_pair_work and multi_batch_work now emit the full 20 new
peak-likeness columns symmetrically (4 precursor + 16 fragment aggregates).

Co-authored-by: Copilot <223556219+Copilot@users.noreply.github.com>"
```

---

## Task 9: End-to-end smoke test + 20-key roster check

**Files:**
- No code changes; run repo's existing tests and inspect output schema.

- [ ] **Step 1: Run the entire pytest suite**

```bash
conda run -n jianyan pytest tests/ -v
```

Expected: All tests pass. Note baseline pass count.

- [ ] **Step 2: Verify the 20 new feature keys are emitted by `single_pair_work`**

```bash
conda run -n jianyan python -c "
import numpy as np
import configparser
from unittest.mock import MagicMock
from spectrum.psm_info import PSMInfo

psm = PSMInfo(
    sequence='PEPTIDEK', charge=2, modify=[],
    rt=np.float32(50.0),
    precursor_mz=np.float32(450.0),
    raw_title='r1', protein_names='X_HUMAN')

dia = MagicMock()
dt = [('rt', 'f8'), ('ppm_error', 'f8'),
      ('intensity', 'f8'), ('cycle_idx', 'i4')]
dia.xic_peaks_extreact.return_value = np.array([], dtype=dt)
dia.xic_ms2_peaks_extract.return_value = (np.array([], dtype=dt), 0.0)
dia.get_window_info.return_value = {'lower': 0.0, 'upper': 1.0,
                                     'width': 1.0, 'centering': 0.5}
dia.check_in_same_ms2.return_value = True
dia.check_in_raw.return_value = False

config = configparser.ConfigParser()
config['general'] = {'mass_tol_ppm': '10', 'xic_cycle_window': '3'}

from workflows.single_work import single_pair_work
feats = single_pair_work(psm, dia, config)

EXPECTED_NEW = [
    'precursor_base_to_apex_ratio',
    'precursor_apex_monotonicity',
    'precursor_n_peaks',
    'precursor_smoothness',
]
for metric in ['base_to_apex_ratio', 'apex_monotonicity', 'n_peaks', 'smoothness']:
    for stat in ['mean', 'p50', 'std', 'max']:
        EXPECTED_NEW.append(f'all_{metric}_{stat}')

print(f'Total new keys expected: {len(EXPECTED_NEW)}')
missing = [k for k in EXPECTED_NEW if k not in feats]
if missing:
    print(f'MISSING: {missing}')
    raise SystemExit(1)
print('All 20 new peak-likeness keys present.')
"
```

Expected output:
```
Total new keys expected: 20
All 20 new peak-likeness keys present.
```

- [ ] **Step 3: Confirm clean working tree**

```bash
git status
```

Expected: `nothing to commit, working tree clean` (or only `cross_domain_analysis/` untracked from baseline).

- [ ] **Step 4: Print summary commit graph**

```bash
git --no-pager log --oneline 2ac0674..HEAD
```

Expected: 8 implementation commits (Tasks 1-8). No commit needed for Task 9.

---

## Summary

After all 9 tasks:

- 1 spec doc (committed in `2ac0674` during brainstorming).
- 8 implementation commits (Tasks 1-8).
- 1 source file modified: `workflows/single_work.py` (4 new helpers + extended calc_xic_score / _default_xic_score + wired into both single_pair_work and multi_batch_work).
- 1 test file modified: `tests/test_single_work_numerics.py` (~14 new tests).
- **20 new CSV columns** emitted by both `single_pair_work` and `multi_batch_work`.
- Backward compatible: existing 197+ tests pass; existing callers of `calc_xic_score` see new fields default to 0.
