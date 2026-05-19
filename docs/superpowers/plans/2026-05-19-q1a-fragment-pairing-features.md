# Q1a Fragment Pairing Features Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Add 11 Q1a "separable-fragment SILAC pairing" features (q1a_recall family) to the feature-extraction pipeline, implementing the design from `docs/specs/2026-05-13-silac-validation-framework.md` §4.2.

**Architecture:** A new helper module `workflows/q1a_helpers.py` exposes a stateful `Q1aAccumulator` and pure helper functions for the three-condition signal-present check. The accumulator is instantiated per-PSM inside the existing `multi_batch_work` and `single_pair_work` fragment loops, fed each fragment's (`light_xic`, `heavy_xic`, mass info), and finalized into the result `features` dict. `spectrum/dia_data.get_window_info` is extended to return `lower`/`upper` so co-isolation can be detected by exact-bound match (not just equal width).

**Tech Stack:** Python 3.13, numpy, scipy.stats.pearsonr (already used), pytest, conda env `jianyan` at `/home/verden/.conda/envs/jianyan`.

---

## Spec Coverage Map

The locked design (spec §4.2 + §4.2 implementation decisions) requires:

| Spec requirement | Implemented in |
|---|---|
| 11 features (q1a_recall, *_shifted, *_unshifted_separable, *_y_recall, *_b_recall, counts, valid) | T3 (compute_features) |
| Three-condition heavy_present (intensity floor 100, apex_delta < 0.3·peak_width, pearson > 0.5) | T2 (is_signal_present_heavy) |
| light_present = intensity > 100 only | T2 (is_signal_present_light) |
| Co/split window via (lower, upper) tuple match | T1 (get_window_info ext) + T2 (Accumulator init) |
| q1a_valid=1 iff total_count >= 3, recall=NaN otherwise | T2 (Q1aAccumulator.compute_features) |
| q1a_recall_unshifted_separable always NaN in co-iso | T2 (Q1aAccumulator) |
| Accumulator-style integration, reuse existing XIC | T3 (multi_batch_work / single_pair_work edits) |
| Output features flow through to CSV unchanged | T4 (e2e test against features.csv schema) |
| q1a_* features classified as SILAC, not sequence | T5 (ablation feature-group update) |

---

## File Structure

**New file:**
- `workflows/q1a_helpers.py` — pure functions + `Q1aAccumulator` class. 1 responsibility: Q1a math. ~150 lines.
- `tests/test_q1a_helpers.py` — unit tests for helpers + accumulator. ~250 lines.

**Modified:**
- `spectrum/dia_data.py:441-469` — `get_window_info` extended return.
- `tests/test_dia_data_window.py` — NEW small test file for `get_window_info` extension. ~40 lines.
- `workflows/single_work.py:22-242` (`multi_batch_work`) and `:245-481` (`single_pair_work`) — insert accumulator wiring into existing fragment loop.
- `tests/test_q1a_integration.py` — NEW e2e test against `multi_batch_work` using minimal mocked DIAData. ~100 lines.
- `tools/eval_feature_ablation.py:33-46` — extend `SEQUENCE_FEATURES` set to NOT contain any `q1a_*` (defensive — they're SILAC features).

---

## Task 1: Extend `get_window_info` to return (lower, upper)

**Why:** Q1a's co/split-window judgment compares exact (lower, upper) bound pairs, not widths. Two windows can have identical width but different positions.

**Files:**
- Modify: `spectrum/dia_data.py:441-469`
- Test: `tests/test_dia_data_window.py` (new)

- [ ] **Step 1: Write the failing test**

Create `tests/test_dia_data_window.py`:

```python
"""Tests for DIAData.get_window_info — extended (lower, upper) return."""
import numpy as np
import pytest

from spectrum.dia_data import DIAData


def _make_minimal_dia(windows):
    """Build a minimal DIAData with N MS2 windows.

    windows: list of (lower_mz, upper_mz) — defines one DIA cycle.
    """
    d = DIAData.__new__(DIAData)
    n = len(windows)
    d._precursor_lower_mz = np.array(
        [np.nan] + [lo for lo, _ in windows], dtype=np.float64)
    d._precursor_upper_mz = np.array(
        [np.nan] + [hi for _, hi in windows], dtype=np.float64)
    # ms2_indexs points into the global arrays (skip the MS1 at idx 0)
    d.ms2_indexs = np.arange(1, n + 1)
    d._cycle_left_precursor = np.array([lo for lo, _ in windows])
    return d


def test_get_window_info_returns_lower_upper():
    """get_window_info must return lower and upper bounds for caller
    to detect co-isolation by exact-pair match."""
    dia = _make_minimal_dia([(500.0, 502.0), (502.0, 504.0)])
    info = dia.get_window_info(501.0)
    assert info["lower"] == 500.0
    assert info["upper"] == 502.0
    assert info["width"] == 2.0


def test_get_window_info_no_match_returns_zero_width_nans():
    """When no window contains the precursor, width=0 and bounds=NaN."""
    dia = _make_minimal_dia([(500.0, 502.0)])
    info = dia.get_window_info(900.0)
    assert info["width"] == 0.0
    assert np.isnan(info["lower"])
    assert np.isnan(info["upper"])


def test_get_window_info_boundary_inclusive_with_tolerance():
    """A precursor at the upper boundary should still match (existing
    code uses 0.1 Da tolerance)."""
    dia = _make_minimal_dia([(500.0, 502.0)])
    info = dia.get_window_info(502.0)
    assert info["lower"] == 500.0
    assert info["upper"] == 502.0
```

- [ ] **Step 2: Run test to verify it fails**

```bash
cd /home/verden/pfind/2025-fall/code/ms2-met
source /opt/miniconda3/etc/profile.d/conda.sh && conda activate jianyan
python -m pytest tests/test_dia_data_window.py -v
```
Expected: 3 FAILED (lower/upper not in return dict; current code returns {"width", "centering"} only).

- [ ] **Step 3: Apply the edit**

In `spectrum/dia_data.py`, modify `get_window_info` (lines 441-469). Replace the existing function body with:

```python
    def get_window_info(self, precursor_mz: float) -> dict:
        """获取包含该 precursor_mz 的 DIA 窗口信息。
        返回 {
            "width": 窗口宽度Da,
            "centering": 前体在窗口中的相对位置 0-1,
            "lower": 窗口下边界 m/z (NaN if not found),
            "upper": 窗口上边界 m/z (NaN if not found),
        }
        """
        default = {"width": 0.0, "centering": 0.5,
                   "lower": float("nan"), "upper": float("nan")}
        if (self._precursor_lower_mz is None or
                self._precursor_upper_mz is None or
                self.ms2_indexs is None or
                len(self.ms2_indexs) == 0):
            return default

        cycle_len = (len(self._cycle_left_precursor)
                     if self._cycle_left_precursor is not None
                     else len(self.ms2_indexs))
        search_range = min(len(self.ms2_indexs), max(cycle_len, 50))

        for i in range(search_range):
            gidx = self.ms2_indexs[i]
            lower = self._precursor_lower_mz[gidx]
            upper = self._precursor_upper_mz[gidx]
            if np.isnan(lower) or np.isnan(upper):
                continue
            if lower - 0.1 <= precursor_mz <= upper + 0.1:
                width = float(upper - lower)
                centering = (float(precursor_mz - lower) / width
                             if width > 0 else 0.5)
                return {"width": width, "centering": centering,
                        "lower": float(lower), "upper": float(upper)}

        return default
```

- [ ] **Step 4: Run test to verify it passes**

```bash
python -m pytest tests/test_dia_data_window.py -v
```
Expected: 3 PASSED. Also run regression:
```bash
python -m pytest tests/ --no-header 2>&1 | tail -3
```
Expected: previous tests still pass (no callers depend on the dict missing those keys; they use `info["width"]` and `info["centering"]`).

- [ ] **Step 5: Commit**

```bash
git add spectrum/dia_data.py tests/test_dia_data_window.py
git commit -m "feat(dia_data): get_window_info returns lower/upper for Q1a co-iso detection

Q1a 'separable fragment' classification requires knowing whether a
peptide's light and heavy precursor m/z fall into the *same* DIA
isolation window (co-isolation) or different windows (split-iso).
Equal window WIDTH is insufficient — two distinct windows can have
identical widths.

Extend the existing get_window_info return dict with 'lower' and
'upper' bound fields. NaN when no matching window is found.

Co-authored-by: Copilot <223556219+Copilot@users.noreply.github.com>"
```

---

## Task 2: Build `workflows/q1a_helpers.py` (pure helpers + accumulator)

**Files:**
- Create: `workflows/q1a_helpers.py`
- Create: `tests/test_q1a_helpers.py`

This task delivers all Q1a math in one file. Subsequent tasks just wire it.

- [ ] **Step 1: Write the failing test for `is_signal_present_light`**

Create `tests/test_q1a_helpers.py`:

```python
"""Tests for workflows/q1a_helpers.py — Q1a fragment-pairing math."""
import numpy as np
import pytest


XIC_DTYPE = [
    ("rt", "f8"), ("intensity", "f8"),
    ("ppm_error", "f8"), ("mz", "f8"),
]


def _xic(rts, intensities):
    """Build a minimal XIC numpy record array for tests."""
    n = len(rts)
    arr = np.zeros(n, dtype=XIC_DTYPE)
    arr["rt"] = rts
    arr["intensity"] = intensities
    return arr


# ----------------------------------------------------------------------
# is_signal_present_light
# ----------------------------------------------------------------------

def test_light_present_when_max_intensity_above_floor():
    from workflows.q1a_helpers import is_signal_present_light
    xic = _xic([10, 11, 12, 13, 14], [50, 200, 500, 200, 50])
    assert is_signal_present_light(xic, intensity_floor=100) is True


def test_light_absent_when_max_intensity_below_floor():
    from workflows.q1a_helpers import is_signal_present_light
    xic = _xic([10, 11, 12], [10, 50, 80])
    assert is_signal_present_light(xic, intensity_floor=100) is False


def test_light_absent_when_xic_empty():
    from workflows.q1a_helpers import is_signal_present_light
    xic = _xic([], [])
    assert is_signal_present_light(xic, intensity_floor=100) is False


def test_light_absent_when_all_nan_intensity():
    from workflows.q1a_helpers import is_signal_present_light
    xic = _xic([10, 11, 12], [np.nan, np.nan, np.nan])
    assert is_signal_present_light(xic, intensity_floor=100) is False
```

- [ ] **Step 2: Run test to verify it fails**

```bash
python -m pytest tests/test_q1a_helpers.py -v
```
Expected: 4 FAILED with ImportError (module doesn't exist).

- [ ] **Step 3: Create `workflows/q1a_helpers.py` skeleton + `is_signal_present_light`**

Create `workflows/q1a_helpers.py`:

```python
"""Q1a fragment-pairing helpers.

Implements §4.2 of docs/specs/2026-05-13-silac-validation-framework.md.

Q1a measures, over the *separable* theoretical b/y fragments of a PSM,
how many fragments have BOTH a credible light signal AND a credible
heavy signal at the predicted m/z and rt. Each TP is one piece of
independent physical evidence that the light search-engine call is
correct; each FN is the inverse.

Public surface:
    - is_signal_present_light(xic, intensity_floor) -> bool
    - is_signal_present_heavy(light_xic, heavy_xic, intensity_floor) -> bool
    - is_split_window(light_window_info, heavy_window_info) -> bool
    - is_separable_fragment(light_mass, heavy_mass, split_window) -> bool
    - Q1aAccumulator: stateful per-PSM accumulator producing 11 features
"""
from __future__ import annotations

import numpy as np
from scipy.stats import pearsonr


# Three-condition signal-present thresholds (spec §4.2 implementation
# decisions, locked 2026-05-19; defaults may be tuned later via config).
DEFAULT_INTENSITY_FLOOR = 100.0
DEFAULT_APEX_DELTA_FRACTION = 0.3   # heavy apex within 0.3 * light_peak_width
DEFAULT_PEARSON_MIN = 0.5

# Below this absolute m/z difference (Da), light and heavy fragment
# masses are considered equal (i.e. the fragment carries no K/R, so
# SILAC does not shift it). Smaller than the smallest SILAC delta
# (R = +10.008 Da, K = +8.014 Da) by many orders.
SHIFT_EPSILON = 0.001


def is_signal_present_light(xic, intensity_floor: float = DEFAULT_INTENSITY_FLOOR) -> bool:
    """Light signal is 'present' iff the XIC has a peak above the floor.

    Only the intensity criterion is applied here; light is the
    reference signal, so we don't compare it against itself for shape
    or coelution.
    """
    if xic is None or len(xic) == 0:
        return False
    max_int = float(np.nanmax(xic["intensity"])) if xic.size > 0 else 0.0
    if not np.isfinite(max_int):
        return False
    return max_int > intensity_floor
```

- [ ] **Step 4: Run test, verify pass**

```bash
python -m pytest tests/test_q1a_helpers.py -v
```
Expected: 4 PASSED.

- [ ] **Step 5: Commit**

```bash
git add workflows/q1a_helpers.py tests/test_q1a_helpers.py
git commit -m "feat(q1a_helpers): add is_signal_present_light intensity-floor check"
```

---

- [ ] **Step 6: Write the failing test for `is_signal_present_heavy`**

Append to `tests/test_q1a_helpers.py`:

```python
# ----------------------------------------------------------------------
# is_signal_present_heavy (three-condition AND)
# ----------------------------------------------------------------------

def test_heavy_present_perfect_pair():
    """Heavy XIC matches light shape + intensity + apex → present."""
    from workflows.q1a_helpers import is_signal_present_heavy
    light = _xic([10, 11, 12, 13, 14], [50, 200, 500, 200, 50])
    heavy = _xic([10, 11, 12, 13, 14], [40, 160, 400, 160, 40])  # same shape
    assert is_signal_present_heavy(
        light, heavy, intensity_floor=100,
        apex_delta_fraction=0.3, pearson_min=0.5) is True


def test_heavy_absent_when_intensity_below_floor():
    from workflows.q1a_helpers import is_signal_present_heavy
    light = _xic([10, 11, 12, 13, 14], [50, 200, 500, 200, 50])
    heavy = _xic([10, 11, 12, 13, 14], [5, 20, 50, 20, 5])  # all < 100
    assert is_signal_present_heavy(
        light, heavy, intensity_floor=100) is False


def test_heavy_absent_when_apex_delta_too_large():
    """Heavy apex far from light apex → not coeluting → absent."""
    from workflows.q1a_helpers import is_signal_present_heavy
    light = _xic([10, 11, 12, 13, 14], [50, 200, 500, 200, 50])  # apex at 12
    heavy = _xic([10, 11, 12, 13, 14], [500, 200, 50, 20, 10])    # apex at 10
    # peak width ≈ 4; apex delta = 2; 2 > 0.3*4 = 1.2 → absent
    assert is_signal_present_heavy(
        light, heavy, intensity_floor=100,
        apex_delta_fraction=0.3) is False


def test_heavy_absent_when_pearson_below_threshold():
    """Same intensity + coeluting but anti-correlated shape → absent."""
    from workflows.q1a_helpers import is_signal_present_heavy
    light = _xic([10, 11, 12, 13, 14], [50, 200, 500, 200, 50])
    # Inverted: same total intensity but anti-correlated
    heavy = _xic([10, 11, 12, 13, 14], [500, 200, 500, 200, 500])
    # pearson ≈ ?  This shape gives a poor correlation with the Gaussian-ish light
    # Force it absent by checking the function's pearson gate.
    assert is_signal_present_heavy(
        light, heavy, intensity_floor=100,
        apex_delta_fraction=1.0,  # disable apex check to isolate pearson
        pearson_min=0.9) is False


def test_heavy_absent_when_empty_xic():
    from workflows.q1a_helpers import is_signal_present_heavy
    light = _xic([10, 11, 12], [50, 200, 500])
    heavy = _xic([], [])
    assert is_signal_present_heavy(light, heavy, intensity_floor=100) is False
```

- [ ] **Step 7: Run test, fail**

```bash
python -m pytest tests/test_q1a_helpers.py::test_heavy_present_perfect_pair -v
```
Expected: FAILED with AttributeError or ImportError.

- [ ] **Step 8: Implement `is_signal_present_heavy`**

Append to `workflows/q1a_helpers.py`:

```python
def _peak_width(xic) -> float:
    """Span of rt values in the XIC (used as a denominator for
    apex_delta normalization). Returns 0 for empty/single-point XICs."""
    if xic is None or len(xic) < 2:
        return 0.0
    rts = np.asarray(xic["rt"], dtype="f8")
    return float(rts.max() - rts.min())


def is_signal_present_heavy(
    light_xic,
    heavy_xic,
    intensity_floor: float = DEFAULT_INTENSITY_FLOOR,
    apex_delta_fraction: float = DEFAULT_APEX_DELTA_FRACTION,
    pearson_min: float = DEFAULT_PEARSON_MIN,
) -> bool:
    """Heavy 'present' iff three conditions all hold:
      1. heavy max intensity > intensity_floor
      2. |heavy_apex_rt - light_apex_rt| < apex_delta_fraction * light_peak_width
      3. pearsonr(aligned_light, aligned_heavy) > pearson_min
    """
    if heavy_xic is None or len(heavy_xic) == 0:
        return False
    if light_xic is None or len(light_xic) == 0:
        return False

    heavy_max = float(np.nanmax(heavy_xic["intensity"]))
    if not np.isfinite(heavy_max) or heavy_max <= intensity_floor:
        return False

    light_apex_rt = float(light_xic["rt"][np.nanargmax(light_xic["intensity"])])
    heavy_apex_rt = float(heavy_xic["rt"][np.nanargmax(heavy_xic["intensity"])])
    apex_delta = abs(heavy_apex_rt - light_apex_rt)
    light_pw = _peak_width(light_xic)
    if light_pw > 0 and apex_delta >= apex_delta_fraction * light_pw:
        return False

    # Pearson correlation on shared rt grid (defensive sort first, mirrors calc_xic_score)
    light_sorted = light_xic[np.argsort(light_xic["rt"])]
    heavy_sorted = heavy_xic[np.argsort(heavy_xic["rt"])]
    rt_start = max(light_sorted["rt"].min(), heavy_sorted["rt"].min())
    rt_end = min(light_sorted["rt"].max(), heavy_sorted["rt"].max())
    if rt_start >= rt_end:
        return False
    common_rt = np.linspace(rt_start, rt_end, 100)
    l_int = np.interp(common_rt, light_sorted["rt"], light_sorted["intensity"])
    h_int = np.interp(common_rt, heavy_sorted["rt"], heavy_sorted["intensity"])
    if np.std(l_int) < 1e-10 or np.std(h_int) < 1e-10:
        return False
    try:
        corr, _ = pearsonr(l_int, h_int)
    except (ValueError, RuntimeWarning):
        return False
    if not np.isfinite(corr):
        return False
    return corr > pearson_min
```

- [ ] **Step 9: Run tests, all pass**

```bash
python -m pytest tests/test_q1a_helpers.py -v
```
Expected: 9 PASSED (4 light + 5 heavy).

- [ ] **Step 10: Commit**

```bash
git add workflows/q1a_helpers.py tests/test_q1a_helpers.py
git commit -m "feat(q1a_helpers): three-condition heavy_present (intensity+apex+pearson)"
```

---

- [ ] **Step 11: Write the failing test for `is_split_window` + `is_separable_fragment`**

Append to `tests/test_q1a_helpers.py`:

```python
# ----------------------------------------------------------------------
# Window separability
# ----------------------------------------------------------------------

def test_is_split_window_same_bounds_returns_false():
    from workflows.q1a_helpers import is_split_window
    w_L = {"width": 2.0, "centering": 0.5, "lower": 500.0, "upper": 502.0}
    w_H = {"width": 2.0, "centering": 0.7, "lower": 500.0, "upper": 502.0}
    assert is_split_window(w_L, w_H) is False


def test_is_split_window_different_bounds_returns_true():
    from workflows.q1a_helpers import is_split_window
    w_L = {"width": 2.0, "centering": 0.5, "lower": 500.0, "upper": 502.0}
    w_H = {"width": 2.0, "centering": 0.5, "lower": 504.0, "upper": 506.0}
    assert is_split_window(w_L, w_H) is True


def test_is_split_window_nan_treated_as_split():
    """If either window lookup fails (NaN bounds), be conservative
    and treat as split (more inclusion in q1a)."""
    from workflows.q1a_helpers import is_split_window
    w_L = {"width": 2.0, "centering": 0.5, "lower": 500.0, "upper": 502.0}
    w_H = {"width": 0.0, "centering": 0.5,
           "lower": float("nan"), "upper": float("nan")}
    assert is_split_window(w_L, w_H) is True


# ----------------------------------------------------------------------
# Fragment separability
# ----------------------------------------------------------------------

def test_shifted_fragment_always_separable():
    """A fragment with K or R is shifted by SILAC; always separable
    by m/z regardless of window configuration."""
    from workflows.q1a_helpers import is_separable_fragment
    # light=300, heavy=310 → shifted
    assert is_separable_fragment(
        light_mass=300.0, heavy_mass=310.0, split_window=True) is True
    assert is_separable_fragment(
        light_mass=300.0, heavy_mass=310.0, split_window=False) is True


def test_unshifted_fragment_separable_only_in_split_window():
    """A fragment with no K/R has equal light/heavy mass. It can only
    be separated by the DIA window (its precursor isolation differs)."""
    from workflows.q1a_helpers import is_separable_fragment
    assert is_separable_fragment(
        light_mass=300.0, heavy_mass=300.0, split_window=True) is True
    assert is_separable_fragment(
        light_mass=300.0, heavy_mass=300.0, split_window=False) is False
```

- [ ] **Step 12: Fail, then implement**

```bash
python -m pytest tests/test_q1a_helpers.py::test_is_split_window_same_bounds_returns_false -v
```
Expected: FAILED with ImportError.

Append to `workflows/q1a_helpers.py`:

```python
def is_split_window(w_light: dict, w_heavy: dict) -> bool:
    """True iff the light and heavy precursor m/z fall into DIFFERENT
    DIA isolation windows.

    Compares (lower, upper) tuples exactly. If either bound is NaN
    (window-lookup failed), conservatively treats as split — this lets
    Q1a still include unshifted fragments when window info is missing,
    which is the safer direction.
    """
    l_lo, l_hi = w_light.get("lower"), w_light.get("upper")
    h_lo, h_hi = w_heavy.get("lower"), w_heavy.get("upper")
    if any(v is None or (isinstance(v, float) and np.isnan(v))
           for v in (l_lo, l_hi, h_lo, h_hi)):
        return True
    return (l_lo != h_lo) or (l_hi != h_hi)


def is_separable_fragment(
    light_mass: float, heavy_mass: float, split_window: bool,
    shift_epsilon: float = SHIFT_EPSILON,
) -> bool:
    """A fragment is separable iff either:
      (a) It carries K or R and is shifted by SILAC (light_mass != heavy_mass), OR
      (b) The DIA windows for light/heavy precursors differ.

    Unshifted fragments under co-isolation cannot be separated:
    light_xic and heavy_xic are extracted from the same MS2 spectra
    at the same m/z, so they are identical by construction.
    """
    is_shifted = (heavy_mass - light_mass) > shift_epsilon
    return bool(is_shifted or split_window)
```

- [ ] **Step 13: Run tests, all 14 pass**

```bash
python -m pytest tests/test_q1a_helpers.py -v 2>&1 | tail -10
```
Expected: 14 PASSED.

- [ ] **Step 14: Commit**

```bash
git add workflows/q1a_helpers.py tests/test_q1a_helpers.py
git commit -m "feat(q1a_helpers): add is_split_window + is_separable_fragment"
```

---

- [ ] **Step 15: Write the failing test for `Q1aAccumulator`**

Append to `tests/test_q1a_helpers.py`:

```python
# ----------------------------------------------------------------------
# Q1aAccumulator (per-PSM, builds the 11 output features)
# ----------------------------------------------------------------------

def _silac_pair(light_int, heavy_int, n=5):
    """Build a (light_xic, heavy_xic) pair with given peak intensities,
    aligned apex, gaussian-ish shape."""
    rts = np.linspace(10, 14, n)
    factor_l = light_int / 500.0
    factor_h = heavy_int / 500.0
    light = _xic(rts, [50 * factor_l, 200 * factor_l, 500 * factor_l,
                       200 * factor_l, 50 * factor_l])
    heavy = _xic(rts, [50 * factor_h, 200 * factor_h, 500 * factor_h,
                       200 * factor_h, 50 * factor_h])
    return light, heavy


def _empty_xic():
    return _xic([], [])


def test_q1a_accumulator_perfect_silac_recall_1():
    """5 shifted fragments all paired → q1a_recall = 1, valid = 1."""
    from workflows.q1a_helpers import Q1aAccumulator
    acc = Q1aAccumulator(split_window=False)
    for ion_type in ("y", "y", "y", "y", "b"):
        light, heavy = _silac_pair(500, 400)
        acc.add(ion_type=ion_type,
                light_mass=300.0, heavy_mass=310.0,
                light_xic=light, heavy_xic=heavy)
    feats = acc.compute_features()
    assert feats["q1a_TP_count"] == 5
    assert feats["q1a_FN_count"] == 0
    assert feats["q1a_recall"] == 1.0
    assert feats["q1a_valid"] == 1
    assert feats["q1a_total_count"] == 5
    assert feats["q1a_recall_shifted"] == 1.0
    # No unshifted_separable contributions
    assert np.isnan(feats["q1a_recall_unshifted_separable"])


def test_q1a_accumulator_trap_no_heavy_recall_0():
    """5 shifted fragments where heavy XIC is empty → FN, recall=0."""
    from workflows.q1a_helpers import Q1aAccumulator
    acc = Q1aAccumulator(split_window=False)
    for _ in range(5):
        light, _ = _silac_pair(500, 0)
        acc.add(ion_type="y",
                light_mass=300.0, heavy_mass=310.0,
                light_xic=light, heavy_xic=_empty_xic())
    feats = acc.compute_features()
    assert feats["q1a_TP_count"] == 0
    assert feats["q1a_FN_count"] == 5
    assert feats["q1a_recall"] == 0.0
    assert feats["q1a_valid"] == 1


def test_q1a_accumulator_total_lt_3_recall_nan_valid_0():
    """Only 2 separable fragments → q1a_valid=0, q1a_recall=NaN."""
    from workflows.q1a_helpers import Q1aAccumulator
    acc = Q1aAccumulator(split_window=False)
    for _ in range(2):
        light, heavy = _silac_pair(500, 400)
        acc.add(ion_type="y",
                light_mass=300.0, heavy_mass=310.0,
                light_xic=light, heavy_xic=heavy)
    feats = acc.compute_features()
    assert feats["q1a_total_count"] == 2
    assert feats["q1a_valid"] == 0
    assert np.isnan(feats["q1a_recall"])
    assert np.isnan(feats["q1a_recall_shifted"])


def test_q1a_accumulator_unshifted_skipped_under_co_iso():
    """Co-isolation + unshifted (b ion no K/R) fragments → not added."""
    from workflows.q1a_helpers import Q1aAccumulator
    acc = Q1aAccumulator(split_window=False)
    # Try to add 5 unshifted fragments under co-iso; should all be
    # silently skipped (not separable).
    for _ in range(5):
        light, heavy = _silac_pair(500, 400)
        acc.add(ion_type="b",
                light_mass=300.0, heavy_mass=300.0,
                light_xic=light, heavy_xic=heavy)
    feats = acc.compute_features()
    assert feats["q1a_total_count"] == 0
    assert feats["q1a_valid"] == 0
    assert np.isnan(feats["q1a_recall"])


def test_q1a_accumulator_unshifted_separable_under_split_iso():
    """Split window + unshifted fragment → counts under
    q1a_recall_unshifted_separable."""
    from workflows.q1a_helpers import Q1aAccumulator
    acc = Q1aAccumulator(split_window=True)
    for _ in range(5):
        light, heavy = _silac_pair(500, 400)
        acc.add(ion_type="b",
                light_mass=300.0, heavy_mass=300.0,
                light_xic=light, heavy_xic=heavy)
    feats = acc.compute_features()
    assert feats["q1a_total_count"] == 5
    assert feats["q1a_TP_unshifted_separable"] == 5
    assert feats["q1a_recall_unshifted_separable"] == 1.0
    # And the *_shifted slice is empty here → NaN
    assert np.isnan(feats["q1a_recall_shifted"])


def test_q1a_accumulator_light_invalid_excluded():
    """Fragments where light signal is below floor → neither TP nor FN."""
    from workflows.q1a_helpers import Q1aAccumulator
    acc = Q1aAccumulator(split_window=False, intensity_floor=100)
    # Light intensity 50 < floor 100 → fragment excluded
    light = _xic([10, 11, 12], [10, 30, 50])
    heavy = _xic([10, 11, 12], [10, 30, 50])
    for _ in range(5):
        acc.add(ion_type="y",
                light_mass=300.0, heavy_mass=310.0,
                light_xic=light, heavy_xic=heavy)
    feats = acc.compute_features()
    assert feats["q1a_total_count"] == 0
    assert np.isnan(feats["q1a_recall"])


def test_q1a_accumulator_y_b_split():
    """y and b counts are tracked separately."""
    from workflows.q1a_helpers import Q1aAccumulator
    acc = Q1aAccumulator(split_window=False)
    # 3 y TP, 2 b TP
    for _ in range(3):
        light, heavy = _silac_pair(500, 400)
        acc.add(ion_type="y",
                light_mass=300.0, heavy_mass=310.0,
                light_xic=light, heavy_xic=heavy)
    for _ in range(2):
        light, heavy = _silac_pair(500, 400)
        acc.add(ion_type="b",
                light_mass=300.0, heavy_mass=310.0,
                light_xic=light, heavy_xic=heavy)
    feats = acc.compute_features()
    assert feats["q1a_y_recall"] == 1.0
    assert feats["q1a_b_recall"] == 1.0
    assert feats["q1a_TP_count"] == 5
    assert feats["q1a_total_count"] == 5
```

- [ ] **Step 16: Implement `Q1aAccumulator`**

Append to `workflows/q1a_helpers.py`:

```python
class Q1aAccumulator:
    """Per-PSM accumulator for Q1a features.

    Usage:
        acc = Q1aAccumulator(split_window=is_split_window(w_L, w_H))
        for ion_type, position, light_mass, heavy_mass in fragments:
            light_xic = dia.xic_ms2_peaks_extract(...)
            heavy_xic = dia.xic_ms2_peaks_extract(...)
            acc.add(ion_type, light_mass, heavy_mass, light_xic, heavy_xic)
        features.update(acc.compute_features())

    All 11 output features are produced by compute_features().
    """

    MIN_VALID_TOTAL = 3  # spec §4.2: q1a_valid = (total >= 3)

    def __init__(
        self,
        split_window: bool,
        intensity_floor: float = DEFAULT_INTENSITY_FLOOR,
        apex_delta_fraction: float = DEFAULT_APEX_DELTA_FRACTION,
        pearson_min: float = DEFAULT_PEARSON_MIN,
    ):
        self.split_window = split_window
        self.intensity_floor = intensity_floor
        self.apex_delta_fraction = apex_delta_fraction
        self.pearson_min = pearson_min
        # Bucket counters: keys are (mechanism, ion_type, outcome)
        # mechanism ∈ {"shifted", "unshifted_separable"}
        # ion_type ∈ {"b", "y"}
        # outcome ∈ {"TP", "FN"}
        self._counts: dict[tuple, int] = {}

    def add(self, ion_type: str, light_mass: float, heavy_mass: float,
            light_xic, heavy_xic) -> None:
        """Process one theoretical fragment.

        Side effects: increments internal counters. Has no return value.
        Fragments that are not separable OR have no light signal are
        silently dropped from Q1a statistics.
        """
        # Separability
        if not is_separable_fragment(light_mass, heavy_mass, self.split_window):
            return
        is_shifted = (heavy_mass - light_mass) > SHIFT_EPSILON
        mechanism = "shifted" if is_shifted else "unshifted_separable"

        # light_present check: cannot judge anything if light absent
        if not is_signal_present_light(light_xic, self.intensity_floor):
            return

        heavy_present = is_signal_present_heavy(
            light_xic, heavy_xic,
            intensity_floor=self.intensity_floor,
            apex_delta_fraction=self.apex_delta_fraction,
            pearson_min=self.pearson_min,
        )
        outcome = "TP" if heavy_present else "FN"

        key = (mechanism, ion_type, outcome)
        self._counts[key] = self._counts.get(key, 0) + 1

    def _sum(self, mechanism=None, ion_type=None, outcome=None) -> int:
        """Sum counters filtered by mechanism/ion_type/outcome (None means wildcard)."""
        total = 0
        for (m, i, o), n in self._counts.items():
            if mechanism is not None and m != mechanism:
                continue
            if ion_type is not None and i != ion_type:
                continue
            if outcome is not None and o != outcome:
                continue
            total += n
        return total

    def _recall(self, mechanism=None, ion_type=None) -> float:
        tp = self._sum(mechanism=mechanism, ion_type=ion_type, outcome="TP")
        fn = self._sum(mechanism=mechanism, ion_type=ion_type, outcome="FN")
        total = tp + fn
        if total < self.MIN_VALID_TOTAL:
            return float("nan")
        return tp / total

    def compute_features(self) -> dict:
        """Return the 11-field Q1a feature dict.

        Conventions:
          - recall is NaN when its bucket has < MIN_VALID_TOTAL (3) entries.
          - q1a_recall_unshifted_separable is additionally NaN under
            co-isolation (where unshifted_separable count is always 0).
          - count features are always integers.
        """
        tp_total = self._sum(outcome="TP")
        fn_total = self._sum(outcome="FN")
        total = tp_total + fn_total
        tp_shifted = self._sum(mechanism="shifted", outcome="TP")
        tp_unsh = self._sum(mechanism="unshifted_separable", outcome="TP")

        # In co-iso, unshifted_separable bucket is by construction empty.
        if not self.split_window:
            recall_unsh = float("nan")
        else:
            recall_unsh = self._recall(mechanism="unshifted_separable")

        return {
            "q1a_recall": self._recall(),
            "q1a_recall_shifted": self._recall(mechanism="shifted"),
            "q1a_recall_unshifted_separable": recall_unsh,
            "q1a_y_recall": self._recall(ion_type="y"),
            "q1a_b_recall": self._recall(ion_type="b"),
            "q1a_TP_count": tp_total,
            "q1a_FN_count": fn_total,
            "q1a_TP_shifted": tp_shifted,
            "q1a_TP_unshifted_separable": tp_unsh,
            "q1a_total_count": total,
            "q1a_valid": 1 if total >= self.MIN_VALID_TOTAL else 0,
        }
```

- [ ] **Step 17: Run all q1a_helpers tests**

```bash
python -m pytest tests/test_q1a_helpers.py -v 2>&1 | tail -15
```
Expected: 21 PASSED (4 light + 5 heavy + 5 window + 7 accumulator).

- [ ] **Step 18: Run full test suite for regression**

```bash
python -m pytest tests/ --no-header 2>&1 | tail -3
```
Expected: All previously-passing tests still pass (Phase-1-5 totals + new Q1a).

- [ ] **Step 19: Commit**

```bash
git add workflows/q1a_helpers.py tests/test_q1a_helpers.py
git commit -m "feat(q1a_helpers): Q1aAccumulator produces 11-feature dict per PSM

Stateful per-PSM accumulator that processes fragments through:
  1. is_separable_fragment gate (shift OR split window)
  2. is_signal_present_light gate (excludes light-invalid fragments)
  3. is_signal_present_heavy three-condition AND (TP vs FN)

compute_features() returns the 11 q1a_* features specified in
docs/specs/2026-05-13-silac-validation-framework.md §4.2.

NaN handled per spec: recall is NaN when bucket has fewer than 3
entries; q1a_recall_unshifted_separable is NaN under co-isolation
by construction.

Co-authored-by: Copilot <223556219+Copilot@users.noreply.github.com>"
```

---

## Task 3: Integrate `Q1aAccumulator` into `multi_batch_work` and `single_pair_work`

**Files:**
- Modify: `workflows/single_work.py` (two functions: `multi_batch_work` and `single_pair_work`)

These two functions have nearly identical fragment loops — both need the same accumulator wiring.

- [ ] **Step 1: Read the current state of `multi_batch_work`**

```bash
grep -n "def multi_batch_work\|def single_pair_work\|^    _, fragment_ions = psm" workflows/single_work.py
```
Note line numbers — the fragment loop is around lines 140-188 (multi_batch_work) and 380-430 (single_pair_work).

- [ ] **Step 2: Write the integration test**

Create `tests/test_q1a_integration.py`:

```python
"""End-to-end test: multi_batch_work emits q1a_* features."""
import configparser

import numpy as np
import pytest


def _minimal_config():
    cfg = configparser.ConfigParser()
    cfg["general"] = {
        "mass_tol_ppm": "10",
        "xic_cycle_window": "3",
    }
    return cfg


def test_multi_batch_work_emits_q1a_features(monkeypatch):
    """multi_batch_work must add 11 q1a_* keys to its features dict."""
    from spectrum.psm_info import PSMInfo
    from workflows import single_work

    psm = PSMInfo(
        sequence="PEPTIDEK", charge=2, modify=[],
        rt=np.float32(10.0), precursor_mz=np.float32(500.0),
        raw_title="r1", protein_names="X_HUMAN",
    )

    # Stub DIAData behavior: return empty XICs (so all fragments are
    # excluded by is_signal_present_light; q1a_total_count == 0).
    class StubDIA:
        def xic_peaks_extreact(self, *args, **kwargs):
            dtype = [("rt", "f8"), ("ppm_error", "f8"),
                     ("intensity", "f8"), ("mz", "f8")]
            return np.array([], dtype=dtype)

        def xic_ms2_peaks_extract(self, *args, **kwargs):
            dtype = [("rt", "f8"), ("ppm_error", "f8"),
                     ("intensity", "f8"), ("mz", "f8")]
            return np.array([], dtype=dtype), 0.0

        def get_window_info(self, mz):
            return {"width": 2.0, "centering": 0.5,
                    "lower": 499.0, "upper": 501.0}

    dia = StubDIA()
    features = single_work.multi_batch_work(
        psm1=psm, dia_data1=dia,
        psm2=psm, dia_data2=dia,
        config=_minimal_config(),
    )

    expected_keys = {
        "q1a_recall", "q1a_recall_shifted", "q1a_recall_unshifted_separable",
        "q1a_y_recall", "q1a_b_recall",
        "q1a_TP_count", "q1a_FN_count",
        "q1a_TP_shifted", "q1a_TP_unshifted_separable",
        "q1a_total_count", "q1a_valid",
    }
    missing = expected_keys - set(features.keys())
    assert not missing, f"missing q1a keys: {missing}"
    # With empty XICs, no fragment passed light_present → counts=0, valid=0
    assert features["q1a_total_count"] == 0
    assert features["q1a_valid"] == 0
    assert np.isnan(features["q1a_recall"])
```

- [ ] **Step 3: Run test, expect fail**

```bash
python -m pytest tests/test_q1a_integration.py -v
```
Expected: FAIL with KeyError or AssertionError (q1a_* not in features dict).

- [ ] **Step 4: Wire `Q1aAccumulator` into `multi_batch_work`**

In `workflows/single_work.py`, at the top of the file alongside the existing imports, add:

```python
from workflows.q1a_helpers import Q1aAccumulator, is_split_window
```

Then in `multi_batch_work` (currently lines 22-242), modify the fragment-loop region as follows. Locate the line that reads `_, fragment_ions = psm1.get_heavy_info(HeavyType.SILAC)` (around line 139). **Just before that line**, insert:

```python
    # --- Q1a setup: classify co/split-isolation for accumulator ---
    w_light_for_q1a = dia_data1.get_window_info(psm1._precursor_mz)
    # heavy precursor mz is the second element of get_heavy_info, fetched below
    heavy_precursor_mz_q1a, _ = psm1.get_heavy_info(HeavyType.SILAC)
    w_heavy_for_q1a = dia_data2.get_window_info(heavy_precursor_mz_q1a)
    q1a_acc = Q1aAccumulator(
        split_window=is_split_window(w_light_for_q1a, w_heavy_for_q1a))
```

Then, **inside the existing fragment loop**, AFTER both `light_ions_xic` and `heavy_ions_xic` have been extracted (around line 162, but BEFORE the existing `if len(light_ions_xic) == 0...` early-return), add:

```python
        # --- Q1a: accumulate fragment evidence for SILAC pairing recall ---
        q1a_acc.add(
            ion_type=ions_type,
            light_mass=light_mass, heavy_mass=heavy_mass,
            light_xic=light_ions_xic, heavy_xic=heavy_ions_xic,
        )
```

Note: the existing loop unpacks `for ions_type, ions_num, light_mass, _ in fragment_ions:` — discarding `heavy_mass`. Change the unpack to keep heavy_mass:

```python
    for ions_type, ions_num, light_mass, heavy_mass in fragment_ions:
```

Finally, AFTER the fragment loop ends (around line 242, before `return features`), add:

```python
    # --- Q1a: finalize and merge features ---
    features.update(q1a_acc.compute_features())
```

- [ ] **Step 5: Run integration test for multi_batch_work**

```bash
python -m pytest tests/test_q1a_integration.py -v
```
Expected: PASS.

- [ ] **Step 6: Replicate the same wiring in `single_pair_work`**

`single_pair_work` (currently lines 245-481) has the same shape with a single DIAData. Apply identical edits:

(a) Find the line `_, fragment_ions = psm.get_heavy_info(HeavyType.SILAC)` (~line 380). Just before it, insert:

```python
    # --- Q1a setup ---
    w_light_for_q1a = dia_data.get_window_info(psm._precursor_mz)
    heavy_precursor_mz_q1a, _ = psm.get_heavy_info(HeavyType.SILAC)
    w_heavy_for_q1a = dia_data.get_window_info(heavy_precursor_mz_q1a)
    q1a_acc = Q1aAccumulator(
        split_window=is_split_window(w_light_for_q1a, w_heavy_for_q1a))
```

(b) Inside the fragment loop, AFTER both `light_ions_xic` and `heavy_ions_xic` are extracted but BEFORE the early-return, add:

```python
        q1a_acc.add(
            ion_type=ions_type,
            light_mass=light_mass, heavy_mass=heavy_mass,
            light_xic=light_ions_xic, heavy_xic=heavy_ions_xic,
        )
```

(c) Change the loop unpacking to keep heavy_mass:

```python
    for ions_type, ions_num, light_mass, heavy_mass in fragment_ions:
```

(d) AFTER the loop, before `return features`, add:

```python
    features.update(q1a_acc.compute_features())
```

- [ ] **Step 7: Add a single_pair_work integration test**

Append to `tests/test_q1a_integration.py`:

```python
def test_single_pair_work_emits_q1a_features():
    """single_pair_work must also add q1a_* keys."""
    from spectrum.psm_info import PSMInfo
    from workflows import single_work

    psm = PSMInfo(
        sequence="PEPTIDEK", charge=2, modify=[],
        rt=np.float32(10.0), precursor_mz=np.float32(500.0),
        raw_title="r1", protein_names="X_HUMAN",
    )

    class StubDIA:
        def xic_peaks_extreact(self, *args, **kwargs):
            dtype = [("rt", "f8"), ("ppm_error", "f8"),
                     ("intensity", "f8"), ("mz", "f8")]
            return np.array([], dtype=dtype)

        def xic_ms2_peaks_extract(self, *args, **kwargs):
            dtype = [("rt", "f8"), ("ppm_error", "f8"),
                     ("intensity", "f8"), ("mz", "f8")]
            return np.array([], dtype=dtype), 0.0

        def get_window_info(self, mz):
            return {"width": 2.0, "centering": 0.5,
                    "lower": 499.0, "upper": 501.0}

    dia = StubDIA()
    features = single_work.single_pair_work(
        psm=psm, dia_data=dia, config=_minimal_config(),
    )

    expected_keys = {
        "q1a_recall", "q1a_recall_shifted", "q1a_recall_unshifted_separable",
        "q1a_y_recall", "q1a_b_recall",
        "q1a_TP_count", "q1a_FN_count",
        "q1a_TP_shifted", "q1a_TP_unshifted_separable",
        "q1a_total_count", "q1a_valid",
    }
    missing = expected_keys - set(features.keys())
    assert not missing, f"missing q1a keys: {missing}"
```

- [ ] **Step 8: Run pytest, all pass**

```bash
python -m pytest tests/test_q1a_integration.py tests/test_q1a_helpers.py -v 2>&1 | tail -10
```
Expected: all PASS.

- [ ] **Step 9: Full regression**

```bash
python -m pytest tests/ --no-header 2>&1 | tail -3
```
Expected: all previously-passing tests still pass; total = 117 (prior) + 22 (q1a) + 3 (window) + 2 (integration) ≈ 144.

- [ ] **Step 10: Commit**

```bash
git add workflows/single_work.py tests/test_q1a_integration.py
git commit -m "feat(single_work): emit 11 q1a_* features per PSM

Wire Q1aAccumulator into multi_batch_work and single_pair_work
fragment loops, reusing the already-extracted light_xic/heavy_xic
to avoid redundant DIA reads. The accumulator finalization runs
once per PSM and merges 11 q1a_* fields into the existing features
dict; downstream CSV emission and pair_flow code paths pick up the
new columns automatically with no further changes.

Implements §4.2 of docs/specs/2026-05-13-silac-validation-framework.md.

Co-authored-by: Copilot <223556219+Copilot@users.noreply.github.com>"
```

---

## Task 4: Update `eval_feature_ablation.py` so q1a_* is grouped as SILAC, not sequence

**Files:**
- Modify: `tools/eval_feature_ablation.py` (around lines 33-46)

`q1a_*` features are SILAC-pairing measurements; they must end up in `silac_only` not `sequence_only` when ablation runs.

- [ ] **Step 1: Inspect current SEQUENCE_FEATURES set**

```bash
grep -n "SEQUENCE_FEATURES\|INTENSITY_FEATURES" tools/eval_feature_ablation.py
```

- [ ] **Step 2: Write a defensive test**

Append to `tests/test_q1a_helpers.py`:

```python
# ----------------------------------------------------------------------
# Ablation feature grouping guard
# ----------------------------------------------------------------------

def test_q1a_features_are_not_in_sequence_only_group():
    """q1a_* are SILAC-pairing features. They must NEVER end up in
    sequence_only when split_features() runs the ablation grouping."""
    from tools.eval_feature_ablation import SEQUENCE_FEATURES, split_features
    # Synthesize a feature column list that includes q1a_*
    all_features = list(SEQUENCE_FEATURES) + [
        "precursor_pearson", "q1a_recall", "q1a_y_recall", "q1a_valid",
        "q1a_TP_count",
    ]
    groups = split_features(all_features)
    for q1a_feat in ("q1a_recall", "q1a_y_recall", "q1a_valid", "q1a_TP_count"):
        assert q1a_feat not in groups["sequence_only"], (
            f"{q1a_feat} accidentally classified as sequence_only")
        assert q1a_feat in groups["silac_only"], (
            f"{q1a_feat} missing from silac_only")
```

- [ ] **Step 3: Run test, see pass-or-fail**

```bash
python -m pytest tests/test_q1a_helpers.py::test_q1a_features_are_not_in_sequence_only_group -v
```

Since `SEQUENCE_FEATURES` is a hardcoded set in `eval_feature_ablation.py` that does NOT contain any `q1a_*` strings, this test should **already pass without modification**. Run it to confirm.

If it passes: no code change needed; the test is purely a regression guard. **Commit only the test**:

```bash
git add tests/test_q1a_helpers.py
git commit -m "test: pin q1a_* as silac_only, not sequence_only, in ablation grouping"
```

If it fails (shouldn't, but defensively): in `tools/eval_feature_ablation.py`, confirm that the `split_features()` function uses `silac_all = [f for f in all_features if f not in SEQUENCE_FEATURES]` pattern. Since `q1a_*` are not in `SEQUENCE_FEATURES`, they fall into `silac_all` automatically. No fix needed.

---

## Task 5: Sanity check Q1a on real baseline_2da_clean data

**Files:**
- Read: `baseline_2da_clean/features.csv` (existing local file — should NOT have q1a_* columns yet since it predates Task 3)
- Create: (ad-hoc) `runs/q1a_smoke_test/config.ini`

This task is **manual / advisory** — it cannot be a pytest unit test because it requires a real DIA `.dia.npz` cache and the conda env. Document the steps so the user can rerun on the remote.

- [ ] **Step 1: Verify columns of existing features.csv**

```bash
head -1 baseline_2da_clean/features.csv | tr ',' '\n' | grep -i q1a
```
Expected: no output (q1a columns absent — this CSV was generated before this implementation).

- [ ] **Step 2: Document the remote rerun command (in plan.md, no execution required by the plan)**

After all tasks merge, the user needs to rerun feature extraction on the remote to populate q1a_* columns:

```bash
# On remote (jianyan env):
cd /home/wskong/jianyan/ms2-met
git pull origin feature_extraction
python main.py \
  --configpath runs/baseline_2da_clean/config.ini \
  --logpath runs/baseline_2da_clean/extract_q1a.log
```

The resulting `features.csv` will have 11 additional columns (`q1a_*`).

Then re-evaluate:
```bash
python tools/eval_baseline.py \
  --features runs/baseline_2da_clean/features.csv \
  --output runs/baseline_2da_clean/eval/baseline_metrics_q1a.json
python tools/eval_feature_ablation.py \
  --features runs/baseline_2da_clean/features.csv \
  --output runs/baseline_2da_clean/eval/ablation_q1a.json
```

Expected outcome (success criterion): `silac_only` AUC should rise from 0.893 (current baseline_2da_clean number) to ≥ 0.92, AND `pos_recall@neg_recall=95%` should rise from 55% to ≥ 65%. The exact numbers depend on the data, but the direction is the contract.

- [ ] **Step 3: No commit needed for this advisory task.**

---

## Self-Review

**Spec coverage:**

| Spec requirement (§4.2 + decisions) | Implemented |
|---|---|
| 11 features named per spec | T2 Step 16 (compute_features) ✓ |
| q1a_recall = TP/(TP+FN) | T2 Step 16 (_recall) ✓ |
| q1a_recall_shifted, *_unshifted_separable, *_y_recall, *_b_recall split | T2 Step 16 (filter wildcards) ✓ |
| q1a_TP_count / q1a_FN_count / q1a_TP_shifted / q1a_TP_unshifted_separable / q1a_total_count / q1a_valid | T2 Step 16 ✓ |
| three-condition AND for heavy_present | T2 Step 8 (is_signal_present_heavy) ✓ |
| intensity_floor=100 default | T2 Step 3 (DEFAULT_INTENSITY_FLOOR) ✓ |
| apex_delta < 0.3*peak_width | T2 Step 8 ✓ |
| pearson > 0.5 | T2 Step 8 ✓ |
| light_present = intensity > floor only | T2 Step 3 (is_signal_present_light) ✓ |
| Co/split window via (lower, upper) tuple match | T1 + T2 Step 12 ✓ |
| q1a_valid = 1 iff total ≥ 3 | T2 Step 16 (MIN_VALID_TOTAL = 3) ✓ |
| q1a_recall_unshifted_separable always NaN under co-iso | T2 Step 16 (explicit branch) ✓ |
| Reuse existing XIC extraction | T3 Step 4 (insert after existing extraction) ✓ |
| Integrate in multi_batch_work AND single_pair_work | T3 Step 4 + Step 6 ✓ |
| Output flows through to CSV unchanged | T3 (features.update preserves dict shape) — confirmed by integration test |
| q1a_* classified as silac_only in ablation | T4 ✓ |

**Placeholder scan:** None — every step has concrete code, exact paths, exact commands.

**Type consistency:**
- `is_signal_present_light(xic, intensity_floor)` — consistent across uses
- `is_signal_present_heavy(light_xic, heavy_xic, intensity_floor=, apex_delta_fraction=, pearson_min=)` — consistent
- `is_split_window(w_light: dict, w_heavy: dict) -> bool` — consistent
- `is_separable_fragment(light_mass, heavy_mass, split_window)` — consistent
- `Q1aAccumulator(split_window, intensity_floor=, apex_delta_fraction=, pearson_min=)` + `.add(ion_type, light_mass, heavy_mass, light_xic, heavy_xic)` + `.compute_features() -> dict` — consistent across T2 tests and T3 integration

**Open issue resolved during plan write:**

In Task 3 Step 4, the existing loop unpacking `for ions_type, ions_num, light_mass, _ in fragment_ions:` discards the heavy_mass. I changed it to `for ions_type, ions_num, light_mass, heavy_mass in fragment_ions:`. This rename is **safe** because the discarded position was `_` and there's no shadowing of `heavy_mass` elsewhere in the loop body (I verified by reading the function source — `heavy_precursor_mz` is the only similar variable and it's set outside the loop).
