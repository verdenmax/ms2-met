# mzML On-Load Centroiding Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Implement profile-mzML on-load centroiding per `docs/specs/2026-06-01-mzml-centroiding-on-load.md`, compressing each profile spectrum to "one peak = one (m/z, intensity)" using local-maxima + parabolic refinement, gated by config with default-on; bump `.dia.npz` cache to `_format_version=2`.

**Architecture:** Add a pure `centroid_spectrum(mz, intensity, rel_threshold)` to `spectrum/spectrum_utils.py`. In `spectrum/dia_data.py` add a `_is_already_centroid` helper and two config fields (`_centroid_enabled`, `_centroid_rel_threshold`); plug centroiding into `_process_single_spectrum`; refactor `_load_from_mzml` so the peak arrays (`_mz_values`, `_intensity_values`) are built by chunk+concat (per-spectrum arrays still preallocated by `total_spectra`). `save_to_file` writes `_format_version=2`; `load_from_file` rejects any other version. `manager/data_manager.py` reads the two new config keys and passes them to `DIAData` before loading. `config.ini` gains the two defaults. All other downstream code is unchanged.

**Tech Stack:** Python 3.13, numpy 2.4, scipy 1.17 (already required), pyteomics, pytest 9.0.

---

## File Structure

**New files:**
- `tests/test_centroid_spectrum.py` — unit tests for `centroid_spectrum` and `_is_already_centroid`.
- `tests/test_dia_data_load_mzml.py` — integration tests for refactored `_load_from_mzml` (uses monkeypatched `pyteomics.mzml.read`) and npz round-trip / version validation.

**Modified files:**
- `spectrum/spectrum_utils.py` — append `centroid_spectrum` function (single responsibility: peak picking on one spectrum).
- `spectrum/dia_data.py` — add 2 config fields + `_is_already_centroid` helper; modify `_process_single_spectrum`, `_load_from_mzml`, `save_to_file`, `load_from_file`, `_load_attrs`.
- `manager/data_manager.py` — read config and set centroid fields on `DIAData` before `_load_from_mzml`.
- `constant/keys.py` — add `CENTROID_ENABLED`, `CENTROID_REL_THRESHOLD` constants.
- `config.ini` — add the 2 new keys with defaults under `[general]`.

No file split / restructure; the changes follow existing patterns in each file.

---

## Task 1: `centroid_spectrum` core function

**Files:**
- Modify: `spectrum/spectrum_utils.py` (append at end of file)
- Test: `tests/test_centroid_spectrum.py` (create)

- [ ] **Step 1: Write the failing tests**

Create `tests/test_centroid_spectrum.py`:

```python
"""Unit tests for spectrum.spectrum_utils.centroid_spectrum."""
import numpy as np
import pytest

from spectrum.spectrum_utils import centroid_spectrum


def _gaussian_profile(centers, heights, sigma=0.005, n_per_peak=11,
                      span_sigmas=3.0, dtype=np.float32):
    """Build a synthetic profile spectrum: isolated Gaussian peaks.

    Returns (mz, intensity) with peaks well-separated so no overlap.
    """
    mz_chunks = []
    int_chunks = []
    for c, h in zip(centers, heights):
        # n_per_peak points across +/- span_sigmas around the center
        rel = np.linspace(-span_sigmas, span_sigmas, n_per_peak)
        mz_chunks.append(c + rel * sigma)
        int_chunks.append(h * np.exp(-0.5 * rel ** 2))
    mz = np.concatenate(mz_chunks).astype(dtype)
    intensity = np.concatenate(int_chunks).astype(dtype)
    # Sort by mz (sanity; should already be sorted)
    order = np.argsort(mz)
    return mz[order], intensity[order]


def test_isolated_gaussian_peaks_recovered():
    """5 well-separated Gaussian peaks → 5 centroids close to true centers."""
    true_centers = [400.0, 500.0, 600.0, 700.0, 800.0]
    heights = [1000.0, 800.0, 1200.0, 600.0, 1500.0]
    mz, intensity = _gaussian_profile(true_centers, heights, sigma=0.005,
                                      n_per_peak=11)

    out_mz, out_int = centroid_spectrum(mz, intensity, rel_threshold=1e-3)

    assert len(out_mz) == 5, f"expected 5 centroids, got {len(out_mz)}"
    assert len(out_int) == 5
    # Each output should match a true center within 0.001 Da.
    for c in true_centers:
        diffs = np.abs(out_mz - c)
        assert diffs.min() < 0.001, (
            f"no centroid within 0.001 of {c}; min diff = {diffs.min()}")
    # Intensities should be near the input peak heights (parabolic-apex is
    # picked as peak-top sample intensity = approx height since center sample
    # sits at the true center).
    for h in heights:
        diffs = np.abs(out_int - h)
        assert diffs.min() < h * 0.05, (
            f"no intensity within 5% of {h}; min diff = {diffs.min()}")


def test_relative_threshold_filters_low_peaks():
    """Peaks below max*rel_threshold must be dropped."""
    true_centers = [500.0, 510.0, 520.0]
    # 520 is at 5e-4 of base peak — below default 1e-3 threshold
    heights = [1000.0, 800.0, 0.5]
    mz, intensity = _gaussian_profile(true_centers, heights, sigma=0.005,
                                      n_per_peak=11)

    out_mz, out_int = centroid_spectrum(mz, intensity, rel_threshold=1e-3)

    assert len(out_mz) == 2, f"expected 2 centroids (3rd below threshold), got {len(out_mz)}"
    assert all(abs(out_mz - 520.0) > 0.5), "520 peak should be filtered"


def test_empty_input_returns_empty():
    out_mz, out_int = centroid_spectrum(
        np.array([], dtype=np.float32),
        np.array([], dtype=np.float32),
    )
    assert out_mz.shape == (0,)
    assert out_int.shape == (0,)


def test_short_input_returns_empty():
    """Length < 3 cannot have an interior local maximum."""
    out_mz, out_int = centroid_spectrum(
        np.array([100.0, 101.0], dtype=np.float32),
        np.array([5.0, 6.0], dtype=np.float32),
    )
    assert out_mz.shape == (0,)
    assert out_int.shape == (0,)


def test_flat_top_no_zero_division():
    """Three equal samples at the apex → parabola denominator = 0, must fall
    back to bin-center m/z without raising ZeroDivisionError."""
    mz = np.array([100.0, 100.01, 100.02, 100.03, 100.04],
                  dtype=np.float32)
    intensity = np.array([1.0, 10.0, 10.0, 10.0, 1.0], dtype=np.float32)
    # Should not raise; should produce at least one centroid.
    out_mz, out_int = centroid_spectrum(mz, intensity, rel_threshold=1e-3)
    assert len(out_mz) >= 1
    # The centroid m/z should fall within the flat-top region.
    assert 100.005 <= out_mz[0] <= 100.035


def test_dtype_preserved_float32():
    mz, intensity = _gaussian_profile([500.0], [1000.0], dtype=np.float32)
    out_mz, out_int = centroid_spectrum(mz, intensity)
    assert out_mz.dtype == np.float32
    assert out_int.dtype == np.float32


def test_strictly_monotonic_mz_in_output():
    """Output m/z must be strictly increasing (no duplicates)."""
    true_centers = [400.0, 500.0, 600.0]
    heights = [1000.0, 1000.0, 1000.0]
    mz, intensity = _gaussian_profile(true_centers, heights)
    out_mz, _ = centroid_spectrum(mz, intensity)
    assert np.all(np.diff(out_mz) > 0)
```

- [ ] **Step 2: Run tests to verify they fail**

```bash
python -m pytest tests/test_centroid_spectrum.py -v
```

Expected: All FAIL with `ImportError: cannot import name 'centroid_spectrum' from 'spectrum.spectrum_utils'`.

- [ ] **Step 3: Implement `centroid_spectrum` in `spectrum/spectrum_utils.py`**

Append to `spectrum/spectrum_utils.py`:

```python
def centroid_spectrum(
    mz: np.ndarray,
    intensity: np.ndarray,
    rel_threshold: float = 1e-3,
) -> tuple[np.ndarray, np.ndarray]:
    """Centroid a single profile-mode spectrum.

    Picks local-maxima with `intensity[i] >= max(intensity) * rel_threshold`,
    then refines the m/z location by 3-point parabolic interpolation.
    Intensity is reported as the apex sample height. Output dtype matches
    input.

    Args:
        mz: 1D array of m/z values, assumed monotonically increasing.
        intensity: 1D array of intensities, same length as `mz`.
        rel_threshold: drop maxima with intensity below
            `max(intensity) * rel_threshold`. Default 1e-3.

    Returns:
        (mz_out, intensity_out): two 1D arrays of equal length (= number of
        accepted peaks). Empty arrays when input length < 3 or no peak
        survives the threshold.
    """
    n = len(mz)
    if n < 3 or len(intensity) != n:
        empty_mz = np.empty(0, dtype=mz.dtype if n > 0 else np.float32)
        empty_int = np.empty(0, dtype=intensity.dtype if n > 0 else np.float32)
        return empty_mz, empty_int

    # 1) Local-maxima index set (interior only).
    # An interior point i (1 <= i <= n-2) is a local max iff
    # intensity[i-1] < intensity[i] >= intensity[i+1].
    interior = intensity[1:-1]
    left = intensity[:-2]
    right = intensity[2:]
    is_peak = (interior > left) & (interior >= right)
    # Indices are relative to the full array.
    peak_idx = np.where(is_peak)[0] + 1
    if peak_idx.size == 0:
        return (np.empty(0, dtype=mz.dtype),
                np.empty(0, dtype=intensity.dtype))

    # 2) Threshold filter.
    max_intensity = float(intensity.max())
    if max_intensity <= 0.0:
        return (np.empty(0, dtype=mz.dtype),
                np.empty(0, dtype=intensity.dtype))
    cutoff = max_intensity * rel_threshold
    peak_idx = peak_idx[intensity[peak_idx] >= cutoff]
    if peak_idx.size == 0:
        return (np.empty(0, dtype=mz.dtype),
                np.empty(0, dtype=intensity.dtype))

    # 3) Parabolic refinement.
    # y0 = intensity[i-1], y1 = intensity[i], y2 = intensity[i+1]
    # apex offset (in *bin* units): dx = 0.5*(y0 - y2) / (y0 - 2*y1 + y2)
    # apex m/z: mz[i] + dx * local_half_step
    # where local_half_step = (mz[i+1] - mz[i-1]) / 2  (avg bin width).
    y0 = intensity[peak_idx - 1].astype(np.float64)
    y1 = intensity[peak_idx].astype(np.float64)
    y2 = intensity[peak_idx + 1].astype(np.float64)

    denom = (y0 - 2.0 * y1 + y2)
    # Fall back to bin-center m/z where denom is ~0 (flat top / saturation).
    safe = np.abs(denom) > 1e-12
    dx = np.zeros_like(denom)
    np.divide(0.5 * (y0 - y2), denom, out=dx, where=safe)
    # Clip dx to [-1, 1] just in case noisy fits push offset outside bin.
    np.clip(dx, -1.0, 1.0, out=dx)

    mz_center = mz[peak_idx].astype(np.float64)
    mz_prev = mz[peak_idx - 1].astype(np.float64)
    mz_next = mz[peak_idx + 1].astype(np.float64)
    half_step = (mz_next - mz_prev) * 0.5
    refined_mz = mz_center + dx * half_step

    out_mz = refined_mz.astype(mz.dtype)
    out_int = y1.astype(intensity.dtype)
    return out_mz, out_int
```

- [ ] **Step 4: Run tests to verify they pass**

```bash
python -m pytest tests/test_centroid_spectrum.py -v
```

Expected: 7 passed.

- [ ] **Step 5: Commit**

```bash
git add spectrum/spectrum_utils.py tests/test_centroid_spectrum.py
git commit -m "feat(spectrum): add centroid_spectrum local-max + parabolic peak picker

Pure function in spectrum_utils. Implements the algorithm described
in docs/specs/2026-06-01-mzml-centroiding-on-load.md §4:

- Local-maxima detection (strict-left / non-strict-right for flat-top
  stability)
- Relative-threshold filter (default max * 1e-3)
- 3-point parabolic refinement with safe fallback when denominator ~0
- dtype preserved; numpy-vectorised

Co-authored-by: Copilot <223556219+Copilot@users.noreply.github.com>"
```

---

## Task 2: `_is_already_centroid` helper

**Files:**
- Modify: `spectrum/dia_data.py` (add helper near other private static helpers)
- Test: `tests/test_centroid_spectrum.py` (append)

- [ ] **Step 1: Write the failing tests**

Append to `tests/test_centroid_spectrum.py`:

```python
from spectrum.dia_data import _is_already_centroid


def test_is_already_centroid_true_when_cv_term_present():
    spectrum = {
        'm/z array': np.array([100.0, 200.0]),
        'intensity array': np.array([1.0, 2.0]),
        'centroid spectrum': '',  # pyteomics stores cv terms as keys
    }
    assert _is_already_centroid(spectrum) is True


def test_is_already_centroid_false_when_profile():
    spectrum = {
        'm/z array': np.array([100.0, 200.0]),
        'intensity array': np.array([1.0, 2.0]),
        'profile spectrum': '',
    }
    assert _is_already_centroid(spectrum) is False


def test_is_already_centroid_false_when_neither_term():
    spectrum = {
        'm/z array': np.array([100.0, 200.0]),
        'intensity array': np.array([1.0, 2.0]),
    }
    assert _is_already_centroid(spectrum) is False
```

- [ ] **Step 2: Run tests to verify they fail**

```bash
python -m pytest tests/test_centroid_spectrum.py -v -k is_already_centroid
```

Expected: 3 FAIL with `ImportError: cannot import name '_is_already_centroid' from 'spectrum.dia_data'`.

- [ ] **Step 3: Implement `_is_already_centroid` in `spectrum/dia_data.py`**

Find the module-level helpers near the top of `spectrum/dia_data.py`
(around line 14 where `deduplicate_with_tolerance` lives). Insert
**immediately after** `def deduplicate_with_tolerance(...)` and **before**
`def _load_attrs(...)`:

```python
def _is_already_centroid(spectrum) -> bool:
    """Return True if the pyteomics spectrum dict carries the MS controlled-
    vocabulary term `MS:1000127 centroid spectrum`.

    pyteomics flattens cv terms into dict keys with the term name as the key
    (and the value as the term's value, often empty string). Presence of the
    key alone is sufficient.
    """
    return 'centroid spectrum' in spectrum
```

- [ ] **Step 4: Run tests to verify they pass**

```bash
python -m pytest tests/test_centroid_spectrum.py -v -k is_already_centroid
```

Expected: 3 passed.

- [ ] **Step 5: Commit**

```bash
git add spectrum/dia_data.py tests/test_centroid_spectrum.py
git commit -m "feat(dia_data): add _is_already_centroid cv-term detector

Module-level helper. Returns True iff the pyteomics spectrum dict
carries the 'centroid spectrum' MS cv term key. Used in a follow-up
commit to short-circuit centroiding for already-centroid input.

Co-authored-by: Copilot <223556219+Copilot@users.noreply.github.com>"
```

---

## Task 3: ConfigKeys constants

**Files:**
- Modify: `constant/keys.py`

No test needed — this is a constants-only change. The downstream task that
reads the keys (Task 7) provides coverage.

- [ ] **Step 1: Add the two constants**

Edit `constant/keys.py`. Locate the existing block under `GENERAL = "general"`:

```python
    GENERAL = "general"
    WORK_DIRECTORY = "work_directory"
    MASS_TOL_PPM = "mass_tol_ppm"
    XIC_CYCLE_WINDOW = "xic_cycle_window"
    RESULT_FILE = "result_file"
    FEATURE_TYPE = "feature_type"
```

Replace it with (append the two new constants):

```python
    GENERAL = "general"
    WORK_DIRECTORY = "work_directory"
    MASS_TOL_PPM = "mass_tol_ppm"
    XIC_CYCLE_WINDOW = "xic_cycle_window"
    RESULT_FILE = "result_file"
    FEATURE_TYPE = "feature_type"

    # mzML centroiding (loaded by manager/data_manager.py)
    CENTROID_ENABLED = "centroid_enabled"
    CENTROID_REL_THRESHOLD = "centroid_rel_threshold"
```

- [ ] **Step 2: Sanity-import**

```bash
python -c "from constant.keys import ConfigKeys; print(ConfigKeys.CENTROID_ENABLED, ConfigKeys.CENTROID_REL_THRESHOLD)"
```

Expected output: `centroid_enabled centroid_rel_threshold`

- [ ] **Step 3: Commit**

```bash
git add constant/keys.py
git commit -m "feat(constants): add CENTROID_ENABLED / CENTROID_REL_THRESHOLD keys

New [general] section keys consumed by manager/data_manager.py to
configure on-load centroiding.

Co-authored-by: Copilot <223556219+Copilot@users.noreply.github.com>"
```

---

## Task 4: DIAData centroid config fields

**Files:**
- Modify: `spectrum/dia_data.py:80-141` (the `__init__` method)
- Test: `tests/test_centroid_spectrum.py` (append)

- [ ] **Step 1: Write the failing test**

Append to `tests/test_centroid_spectrum.py`:

```python
from spectrum.dia_data import DIAData


def test_dia_data_defaults_have_centroid_fields():
    """DIAData() must expose centroid config fields with documented
    defaults: enabled=True, rel_threshold=1e-3."""
    d = DIAData()
    assert d._centroid_enabled is True
    assert d._centroid_rel_threshold == pytest.approx(1e-3)
```

- [ ] **Step 2: Run the test to verify it fails**

```bash
python -m pytest tests/test_centroid_spectrum.py -v -k centroid_fields
```

Expected: FAIL with `AttributeError: 'DIAData' object has no attribute '_centroid_enabled'`.

- [ ] **Step 3: Add the two fields to `DIAData.__init__`**

Edit `spectrum/dia_data.py`. Locate the end of `__init__`:

```python
        self._zeroth_frame: int = 0
        self._scan_max_index: int = 1
        self.frame_max_index: int | None = None
```

Replace with (append the two new fields):

```python
        self._zeroth_frame: int = 0
        self._scan_max_index: int = 1
        self.frame_max_index: int | None = None

        """ 加载时 centroiding 配置 (由 DataManager 从 config 注入；这里给默认值) """
        self._centroid_enabled: bool = True
        self._centroid_rel_threshold: float = 1e-3
```

- [ ] **Step 4: Run the test to verify it passes**

```bash
python -m pytest tests/test_centroid_spectrum.py -v -k centroid_fields
```

Expected: 1 passed.

- [ ] **Step 5: Commit**

```bash
git add spectrum/dia_data.py tests/test_centroid_spectrum.py
git commit -m "feat(dia_data): add _centroid_enabled / _centroid_rel_threshold fields

Defaults: enabled=True, rel_threshold=1e-3 (per spec §6). Fields are
not yet wired into the load pipeline; a follow-up commit integrates
centroiding in _process_single_spectrum.

Co-authored-by: Copilot <223556219+Copilot@users.noreply.github.com>"
```

---

## Task 5: npz `_format_version` save/load

**Files:**
- Modify: `spectrum/dia_data.py:43-77` (`_load_attrs`), `spectrum/dia_data.py:144-196` (`save_to_file`, `load_from_file`)
- Test: `tests/test_dia_data_load_mzml.py` (create)

- [ ] **Step 1: Write the failing tests**

Create `tests/test_dia_data_load_mzml.py`:

```python
"""Tests for DIAData npz format-version handling and refactored
_load_from_mzml peak storage."""
import os

import numpy as np
import pytest

from spectrum.dia_data import DIAData


# ---- npz format version round-trip ----

def _make_minimal_dia_for_save():
    """Build a tiny DIAData that's valid to save_to_file."""
    d = DIAData.__new__(DIAData)
    d.has_mobility = False
    d.has_ms1 = True
    d._max_mz_value = 1000.0
    d._min_mz_value = 400.0
    d._zeroth_frame = 0
    d._scan_max_index = 1
    d.frame_max_index = 2

    d.ms1_indexs = np.array([0], dtype=np.int32)
    d.ms1_indexs_rt = np.array([1.0], dtype=np.float32)
    d.ms2_indexs = np.array([1, 2], dtype=np.int32)
    d.ms2_indexs_rt = np.array([1.1, 1.2], dtype=np.float32)
    d.precursor_scan_ids = np.array([-1, 100, 100], dtype=np.int64)
    d._mz_values = np.array([100.0, 200.0, 300.0], dtype=np.float32)
    d.rt_values = np.array([1.0, 1.1, 1.2], dtype=np.float32)
    d._intensity_values = np.array([10.0, 20.0, 30.0], dtype=np.float32)
    d.mobility_values = np.array([1e-6, 0.0], dtype=np.float32)
    d._cycle_left_precursor = np.array([400.0, 500.0], dtype=np.float32)
    d._scan_id_to_index = np.array([0, 1, 2], dtype=np.int64)
    d._peak_start_idx_list = np.array([0, 1, 2], dtype=np.int64)
    d._peak_stop_idx_list = np.array([1, 2, 3], dtype=np.int64)
    d._precursor_lower_mz = np.array([np.nan, 400.0, 500.0], dtype=np.float32)
    d._precursor_upper_mz = np.array([np.nan, 500.0, 600.0], dtype=np.float32)
    # Optional fields - None
    d._quad_max_mz_value = None
    d._quad_min_mz_value = None
    return d


def test_save_writes_format_version_2(tmp_path):
    """save_to_file persists _format_version=2."""
    d = _make_minimal_dia_for_save()
    out = tmp_path / "x.dia.npz"
    d.save_to_file(str(out))

    with np.load(str(out)) as data:
        assert '_format_version' in data, \
            "expected '_format_version' key in saved npz"
        assert int(data['_format_version']) == 2


def test_load_roundtrip_succeeds(tmp_path):
    """save_to_file → load_from_file recovers the same arrays."""
    d = _make_minimal_dia_for_save()
    out = tmp_path / "y.dia.npz"
    d.save_to_file(str(out))

    d2 = DIAData.load_from_file(str(out), use_mmap=False)
    np.testing.assert_array_equal(d2._mz_values, d._mz_values)
    np.testing.assert_array_equal(d2._intensity_values, d._intensity_values)
    np.testing.assert_array_equal(d2.ms1_indexs, d.ms1_indexs)
    assert d2._max_mz_value == d._max_mz_value


def test_load_rejects_missing_format_version(tmp_path):
    """Old npz without _format_version must be rejected with a clear message
    naming the path so the user knows what to delete."""
    d = _make_minimal_dia_for_save()
    out = tmp_path / "old.dia.npz"
    # Manually build a "legacy" npz with no version key.
    payload = {
        'has_mobility': d.has_mobility,
        'has_ms1': d.has_ms1,
        '_max_mz_value': d._max_mz_value,
        '_min_mz_value': d._min_mz_value,
        '_zeroth_frame': d._zeroth_frame,
        '_scan_max_index': d._scan_max_index,
        'frame_max_index': d.frame_max_index,
        'ms1_indexs': d.ms1_indexs,
        'ms1_indexs_rt': d.ms1_indexs_rt,
        'ms2_indexs': d.ms2_indexs,
        'ms2_indexs_rt': d.ms2_indexs_rt,
        'precursor_scan_ids': d.precursor_scan_ids,
        '_mz_values': d._mz_values,
        'rt_values': d.rt_values,
        '_intensity_values': d._intensity_values,
        'mobility_values': d.mobility_values,
        '_cycle_left_precursor': d._cycle_left_precursor,
        '_scan_id_to_index': d._scan_id_to_index,
        '_peak_start_idx_list': d._peak_start_idx_list,
        '_peak_stop_idx_list': d._peak_stop_idx_list,
        '_precursor_lower_mz': d._precursor_lower_mz,
        '_precursor_upper_mz': d._precursor_upper_mz,
    }
    np.savez_compressed(str(out), **payload)

    with pytest.raises(ValueError, match=r"_format_version"):
        DIAData.load_from_file(str(out), use_mmap=False)


def test_load_rejects_wrong_format_version(tmp_path):
    """npz with _format_version != 2 must be rejected."""
    d = _make_minimal_dia_for_save()
    out = tmp_path / "wrong.dia.npz"
    payload = {
        '_format_version': np.int32(99),
        'has_mobility': d.has_mobility,
        'has_ms1': d.has_ms1,
        '_max_mz_value': d._max_mz_value,
        '_min_mz_value': d._min_mz_value,
        '_zeroth_frame': d._zeroth_frame,
        '_scan_max_index': d._scan_max_index,
        'frame_max_index': d.frame_max_index,
        'ms1_indexs': d.ms1_indexs,
        'ms1_indexs_rt': d.ms1_indexs_rt,
        'ms2_indexs': d.ms2_indexs,
        'ms2_indexs_rt': d.ms2_indexs_rt,
        'precursor_scan_ids': d.precursor_scan_ids,
        '_mz_values': d._mz_values,
        'rt_values': d.rt_values,
        '_intensity_values': d._intensity_values,
        'mobility_values': d.mobility_values,
        '_cycle_left_precursor': d._cycle_left_precursor,
        '_scan_id_to_index': d._scan_id_to_index,
        '_peak_start_idx_list': d._peak_start_idx_list,
        '_peak_stop_idx_list': d._peak_stop_idx_list,
        '_precursor_lower_mz': d._precursor_lower_mz,
        '_precursor_upper_mz': d._precursor_upper_mz,
    }
    np.savez_compressed(str(out), **payload)

    with pytest.raises(ValueError, match=r"_format_version"):
        DIAData.load_from_file(str(out), use_mmap=False)
```

- [ ] **Step 2: Run tests to verify they fail**

```bash
python -m pytest tests/test_dia_data_load_mzml.py -v
```

Expected: `test_save_writes_format_version_2` FAILS (`'_format_version' not in saved npz`); other tests FAIL or ERROR.

- [ ] **Step 3: Modify `save_to_file` to write `_format_version=2`**

Edit `spectrum/dia_data.py`. Locate `save_to_file`:

```python
    def save_to_file(self, filepath: str):
        """将所有 NumPy 数组和标量保存到 .npz 文件"""

        data = {
            # 标量属性
            'has_mobility': self.has_mobility,
            'has_ms1': self.has_ms1,
```

Insert `_format_version` immediately after the docstring (as the first key):

```python
    def save_to_file(self, filepath: str):
        """将所有 NumPy 数组和标量保存到 .npz 文件"""

        data = {
            # 格式版本号 (2 = centroided peaks; 见 docs/specs/2026-06-01-...)
            '_format_version': np.int32(2),
            # 标量属性
            'has_mobility': self.has_mobility,
            'has_ms1': self.has_ms1,
```

- [ ] **Step 4: Modify `load_from_file` to validate version**

Edit `spectrum/dia_data.py`. Locate `load_from_file`:

```python
    @classmethod
    def load_from_file(cls, filepath: str, use_mmap: bool = True):
        """从 .npz 文件加载 DIAData，支持内存映射（只读）"""
        obj = cls()

        if use_mmap:
            # 使用 mmap_mode='r' 实现零拷贝共享
            with np.load(filepath, mmap_mode='r') as data:
                _load_attrs(obj, data)
        else:
            # 普通加载（用于主进程预处理）
            data = np.load(filepath)
            _load_attrs(obj, data)

        return obj
```

Replace with:

```python
    @classmethod
    def load_from_file(cls, filepath: str, use_mmap: bool = True):
        """从 .npz 文件加载 DIAData，支持内存映射（只读）

        Raises:
            ValueError: 若 npz 没有 `_format_version` 字段或版本号 != 2。
                这通常意味着旧版本生成的 profile-peaks 缓存。请删除文件
                让 workflows/flow_utils.py:data_to_npz 重新生成。
        """
        obj = cls()

        if use_mmap:
            with np.load(filepath, mmap_mode='r') as data:
                cls._check_format_version(filepath, data)
                _load_attrs(obj, data)
        else:
            data = np.load(filepath)
            cls._check_format_version(filepath, data)
            _load_attrs(obj, data)

        return obj

    @staticmethod
    def _check_format_version(filepath: str, data) -> None:
        """Reject npz files without `_format_version=2`."""
        if '_format_version' not in data:
            raise ValueError(
                f"npz 缓存 {filepath} 没有 _format_version 字段——这是 "
                f"旧版本（profile peaks）生成的缓存。请删除该文件后重新"
                f"运行以生成 centroided 缓存。"
            )
        version = int(data['_format_version'])
        if version != 2:
            raise ValueError(
                f"npz 缓存 {filepath} 的 _format_version={version}，"
                f"当前代码只支持 version=2。请删除该文件后重新运行。"
            )
```

- [ ] **Step 5: Run tests to verify they pass**

```bash
python -m pytest tests/test_dia_data_load_mzml.py -v
```

Expected: 4 passed.

- [ ] **Step 6: Run full test suite to confirm no regressions**

```bash
python -m pytest -q
```

Expected: all previously-passing tests still pass.

- [ ] **Step 7: Commit**

```bash
git add spectrum/dia_data.py tests/test_dia_data_load_mzml.py
git commit -m "feat(dia_data): add _format_version=2 to npz cache; reject old caches

save_to_file now writes _format_version=2; load_from_file raises
ValueError with a clear remediation message when the field is missing
or != 2. Old profile-peak caches must be deleted (already gitignored
via *.dia.npz pattern).

Co-authored-by: Copilot <223556219+Copilot@users.noreply.github.com>"
```

---

## Task 6: Refactor `_load_from_mzml` peak storage (chunk + concat)

**Files:**
- Modify: `spectrum/dia_data.py:222-237` (`_preallocate_arrays`), `spectrum/dia_data.py:239-351` (`_process_single_spectrum`), `spectrum/dia_data.py:353-420` (`_load_from_mzml`)
- Test: `tests/test_dia_data_load_mzml.py` (append integration test)

This task does the structural change **only** — no centroiding wired yet
(Task 7 adds the centroid call). After this task, behavior must be
identical to before for profile-mode input.

- [ ] **Step 1: Write the failing integration test**

Append to `tests/test_dia_data_load_mzml.py`:

```python
# ---- _load_from_mzml refactor: chunk + concat peak storage ----

class _FakeMzmlReader:
    """Mimic the context manager that pyteomics.mzml.read returns."""
    def __init__(self, spectra):
        self._spectra = spectra
    def __enter__(self):
        return iter(self._spectra)
    def __exit__(self, exc_type, exc_val, exc_tb):
        return False


def _make_spectrum(scan_num, ms_level, rt, mz_arr, int_arr,
                   precursor_scan_num=None, precursor_mz=None,
                   iso_lower_off=0.0, iso_upper_off=0.0):
    """Build a dict matching pyteomics.mzml output shape."""
    spectrum = {
        'id': f'controllerType=0 controllerNumber=1 scan={scan_num}',
        'spectrum title': f'spec{scan_num} dummy',
        'ms level': ms_level,
        'm/z array': np.asarray(mz_arr, dtype=np.float32),
        'intensity array': np.asarray(int_arr, dtype=np.float32),
        'scanList': {
            'scan': [{'scan start time': float(rt)}],
        },
    }
    if ms_level > 1 and precursor_scan_num is not None:
        spectrum['precursorList'] = {
            'precursor': [{
                'spectrumRef': (
                    f'controllerType=0 controllerNumber=1 '
                    f'scan={precursor_scan_num}'),
                'selectedIonList': {
                    'selectedIon': [{
                        'selected ion m/z': precursor_mz,
                        'charge state': 2,
                    }],
                },
                'isolationWindow': {
                    'isolation window lower offset': iso_lower_off,
                    'isolation window upper offset': iso_upper_off,
                },
            }],
        }
    return spectrum


def test_load_from_mzml_chunk_concat_preserves_arrays(monkeypatch):
    """With centroid disabled, refactored _load_from_mzml must produce
    arrays identical to feeding raw peaks in directly."""
    # Build 3 spectra: 1 MS1 + 2 MS2 in a single DIA cycle.
    spectra = [
        _make_spectrum(
            scan_num=100, ms_level=1, rt=1.0,
            mz_arr=[400.0, 401.0, 402.0],
            int_arr=[10.0, 20.0, 30.0],
        ),
        _make_spectrum(
            scan_num=101, ms_level=2, rt=1.05,
            mz_arr=[200.0, 201.0],
            int_arr=[5.0, 15.0],
            precursor_scan_num=100, precursor_mz=500.0,
            iso_lower_off=1.0, iso_upper_off=1.0,
        ),
        _make_spectrum(
            scan_num=102, ms_level=2, rt=1.10,
            mz_arr=[300.0, 301.0, 302.0, 303.0],
            int_arr=[1.0, 2.0, 3.0, 4.0],
            precursor_scan_num=100, precursor_mz=600.0,
            iso_lower_off=2.0, iso_upper_off=2.0,
        ),
    ]

    # Patch pyteomics.mzml.read to return our fake reader.
    from spectrum import dia_data as dd
    monkeypatch.setattr(dd.mzml, 'read',
                        lambda p: _FakeMzmlReader(spectra))

    d = DIAData()
    d._centroid_enabled = False  # Refactor must be behavior-preserving when off
    d._load_from_mzml('fake.mzML')

    # Total peaks = 3 + 2 + 4 = 9
    assert d._mz_values.shape == (9,), \
        f"expected 9 concatenated peaks, got {d._mz_values.shape}"
    assert d._intensity_values.shape == (9,)

    # Peak index bookkeeping
    assert list(d._peak_start_idx_list) == [0, 3, 5]
    assert list(d._peak_stop_idx_list) == [3, 5, 9]

    # Per-spectrum slices match input
    np.testing.assert_array_equal(
        d._mz_values[0:3], np.array([400.0, 401.0, 402.0], dtype=np.float32))
    np.testing.assert_array_equal(
        d._intensity_values[0:3], np.array([10.0, 20.0, 30.0], dtype=np.float32))
    np.testing.assert_array_equal(
        d._mz_values[5:9],
        np.array([300.0, 301.0, 302.0, 303.0], dtype=np.float32))

    # MS1/MS2 partitioning
    np.testing.assert_array_equal(d.ms1_indexs, np.array([0], dtype=np.int32))
    np.testing.assert_array_equal(d.ms2_indexs, np.array([1, 2], dtype=np.int32))
```

- [ ] **Step 2: Run the test to verify it fails**

```bash
python -m pytest tests/test_dia_data_load_mzml.py -v -k chunk_concat
```

Expected: FAIL — current `_process_single_spectrum` writes into the
preallocated `_mz_values` array sized by the first-pass `total_peaks`
estimate. With centroid disabled, the count itself doesn't change, so
the test may actually pass on current code... **verify it fails** by
examining the index logic. Specifically, peak indices must still be
right; if they are, this test mainly defends against regression in the
refactor.

> If the test passes against unrefactored code, that's fine — keep it
> as a regression guard. Continue with Step 3.

- [ ] **Step 3: Refactor `_load_from_mzml`**

Edit `spectrum/dia_data.py`. Locate `_load_from_mzml`:

```python
    def _load_from_mzml(
        self,
        mzml_file_path: None | str = None
    ):
        """从 mzML 文件加载数据"""
        logging.info(f"Loading DIA data from {mzml_file_path} ...")

        # 第一遍：统计数据量
        total_spectra = 0
        total_peaks = 0
        with mzml.read(mzml_file_path) as reader:
            for spectrum in reader:
                total_spectra += 1
                total_peaks += len(spectrum['m/z array'])

        logging.info(f"{mzml_file_path} Total spectra: {
                     total_spectra}, total peaks: {total_peaks}")

        # 预先分配数组
        self._preallocate_arrays(total_spectra=total_spectra,
                                 total_peaks=total_peaks)

        # 第二遍：填充数据
        current_spectrum_idx = 0
        current_peak_idx = 0
        # 开始处理信息
        with mzml.read(mzml_file_path) as reader:
            for spectrum in reader:

                self._process_single_spectrum(
                    spectrum, current_spectrum_idx, current_peak_idx)

                # 更新索引
                num_peaks = len(spectrum['m/z array'])
                current_peak_idx += num_peaks
                current_spectrum_idx += 1
```

Replace with:

```python
    def _load_from_mzml(
        self,
        mzml_file_path: None | str = None
    ):
        """从 mzML 文件加载数据。

        改造说明（spec 2026-06-01-mzml-centroiding-on-load §5.1）：
        第一遍只统计 total_spectra（不访问 peaks 数组，pyteomics 按需懒
        解码）。第二遍 centroid 后通过 chunk + concat 构建
        `_mz_values` / `_intensity_values`；其余按谱图数预分配的数组保持
        现状。
        """
        logging.info(f"Loading DIA data from {mzml_file_path} ...")

        # 第一遍：只统计谱图数（不读 peaks）
        total_spectra = 0
        with mzml.read(mzml_file_path) as reader:
            for _spectrum in reader:
                total_spectra += 1

        logging.info(
            f"{mzml_file_path} Total spectra: {total_spectra}")

        # 按谱图数预分配定长数组（不再预分配 peak 数组）
        self._preallocate_arrays(total_spectra=total_spectra)

        # 第二遍：填充。peak 数组通过 chunk list 收集后 concat。
        mz_chunks: list[np.ndarray] = []
        int_chunks: list[np.ndarray] = []
        current_spectrum_idx = 0
        current_peak_idx = 0

        with mzml.read(mzml_file_path) as reader:
            for spectrum in reader:
                mz_chunk, int_chunk = self._process_single_spectrum(
                    spectrum, current_spectrum_idx, current_peak_idx)
                mz_chunks.append(mz_chunk)
                int_chunks.append(int_chunk)

                current_peak_idx += len(mz_chunk)
                current_spectrum_idx += 1

        # Concat peak arrays (一次性, 然后立即释放 chunk list 节省内存)
        if mz_chunks:
            self._mz_values = np.concatenate(mz_chunks).astype(np.float32)
            self._intensity_values = np.concatenate(int_chunks).astype(
                np.float32)
        else:
            self._mz_values = np.empty(0, dtype=np.float32)
            self._intensity_values = np.empty(0, dtype=np.float32)
        del mz_chunks, int_chunks
```

- [ ] **Step 4: Refactor `_preallocate_arrays` to drop peak arrays**

Locate `_preallocate_arrays`:

```python
    def _preallocate_arrays(self, total_spectra: int, total_peaks: int):
        """ 预先分配数组信息 """
        # 谱图信息数组
        self.precursor_scan_ids = np.zeros(total_spectra, dtype=np.int64)
        self.rt_values = np.zeros(total_spectra, dtype=np.float32)
        self._peak_start_idx_list = np.zeros(total_spectra, dtype=np.int64)
        self._peak_stop_idx_list = np.zeros(total_spectra, dtype=np.int64)
        self._precursor_lower_mz = np.zeros(total_spectra, dtype=np.float32)
        self._precursor_upper_mz = np.zeros(total_spectra, dtype=np.float32)

        # 峰数据数组
        self._mz_values = np.zeros(total_peaks, dtype=np.float32)
        self._intensity_values = np.zeros(total_peaks, dtype=np.float32)

        # 其他数组
        self._scan_id_to_index = np.zeros(total_spectra + 10, dtype=np.int64)
```

Replace with:

```python
    def _preallocate_arrays(self, total_spectra: int):
        """预先分配按谱图数定长的数组。

        Peak 数组 (_mz_values / _intensity_values) 不再预分配，由
        _load_from_mzml 通过 chunk + concat 构建。
        """
        # 谱图信息数组
        self.precursor_scan_ids = np.zeros(total_spectra, dtype=np.int64)
        self.rt_values = np.zeros(total_spectra, dtype=np.float32)
        self._peak_start_idx_list = np.zeros(total_spectra, dtype=np.int64)
        self._peak_stop_idx_list = np.zeros(total_spectra, dtype=np.int64)
        self._precursor_lower_mz = np.zeros(total_spectra, dtype=np.float32)
        self._precursor_upper_mz = np.zeros(total_spectra, dtype=np.float32)

        # scan_id 反查表 (留 +10 余量, 防止极端 scan_id)
        self._scan_id_to_index = np.zeros(total_spectra + 10, dtype=np.int64)
```

- [ ] **Step 5: Refactor `_process_single_spectrum` to return per-spectrum peak chunks**

Locate `_process_single_spectrum`. The current method writes into the
preallocated `_mz_values` / `_intensity_values` slices. Change it to
**return** `(mz_chunk, int_chunk)` for the caller to accumulate.

Find this block in `_process_single_spectrum`:

```python
        """
        记录的原始数据关键数组, mz_value、rt_value、intensity_value、mobility_values。
        """
        peak_stop_idx = current_peak_index + len(mz_array)
        self.precursor_scan_ids[spectrum_idx] = precursor_scan_id
        self._mz_values[current_peak_index:peak_stop_idx] = mz_array
        self._intensity_values[current_peak_index:peak_stop_idx] = intensity_array

        # 提取 RT 值
        self.rt_values[spectrum_idx] = rt
```

Replace with:

```python
        """
        记录的原始数据关键数组, mz_value、rt_value、intensity_value、mobility_values。

        改造（spec 2026-06-01）：mz/intensity 不再写入预分配数组，而是
        作为 chunk 返回给 _load_from_mzml 累积后 concat。
        """
        peak_stop_idx = current_peak_index + len(mz_array)
        self.precursor_scan_ids[spectrum_idx] = precursor_scan_id

        # 提取 RT 值
        self.rt_values[spectrum_idx] = rt
```

Then find the end of `_process_single_spectrum`:

```python
        # 提取这个谱图 mz 范围
        self._precursor_lower_mz[spectrum_idx] = isolation_lower
        self._precursor_upper_mz[spectrum_idx] = isolation_upper
```

Replace with:

```python
        # 提取这个谱图 mz 范围
        self._precursor_lower_mz[spectrum_idx] = isolation_lower
        self._precursor_upper_mz[spectrum_idx] = isolation_upper

        return mz_array, intensity_array
```

> Note: at this point `_process_single_spectrum` returns the raw mz/intensity arrays
> (no centroiding yet). Task 7 wraps that return value with `centroid_spectrum`.

- [ ] **Step 6: Run the refactor regression test**

```bash
python -m pytest tests/test_dia_data_load_mzml.py -v -k chunk_concat
```

Expected: PASS.

- [ ] **Step 7: Run full test suite**

```bash
python -m pytest -q
```

Expected: all previously-passing tests still pass. The npz round-trip
test from Task 5 must also still pass.

- [ ] **Step 8: Commit**

```bash
git add spectrum/dia_data.py tests/test_dia_data_load_mzml.py
git commit -m "refactor(dia_data): chunk+concat peak storage in _load_from_mzml

Per spec 2026-06-01-mzml-centroiding-on-load §5.1:
- First pass now only counts total_spectra (does not touch peaks arrays).
- _preallocate_arrays no longer takes total_peaks; _mz_values /
  _intensity_values are built by collecting per-spectrum chunks into a
  Python list and concatenating once at the end.
- _process_single_spectrum returns (mz_chunk, int_chunk) for caller
  accumulation.

Behavior preserved when centroiding is off; integration test guards
the refactor.

Co-authored-by: Copilot <223556219+Copilot@users.noreply.github.com>"
```

---

## Task 7: Integrate centroid into `_process_single_spectrum`

**Files:**
- Modify: `spectrum/dia_data.py` (`_process_single_spectrum` only)
- Test: `tests/test_dia_data_load_mzml.py` (append)

- [ ] **Step 1: Write the failing integration test**

Append to `tests/test_dia_data_load_mzml.py`:

```python
# ---- centroid integration into _load_from_mzml ----

def _profile_gaussian(centers, heights, sigma=0.005, n_per_peak=11,
                      span_sigmas=3.0):
    """Mirror of helper in test_centroid_spectrum: synthesize a profile
    spectrum from isolated Gaussian peaks."""
    mz_chunks = []
    int_chunks = []
    for c, h in zip(centers, heights):
        rel = np.linspace(-span_sigmas, span_sigmas, n_per_peak)
        mz_chunks.append(c + rel * sigma)
        int_chunks.append(h * np.exp(-0.5 * rel ** 2))
    mz = np.concatenate(mz_chunks).astype(np.float32)
    intensity = np.concatenate(int_chunks).astype(np.float32)
    order = np.argsort(mz)
    return mz[order], intensity[order]


def test_load_from_mzml_with_centroid_enabled_compresses_peaks(monkeypatch):
    """With _centroid_enabled=True, profile spectra are compressed to
    one peak per Gaussian; _mz_values length drops from 22 (2 spectra
    × 11 samples) to N peaks."""
    mz1, int1 = _profile_gaussian([400.0], [1000.0])  # 11 points → 1 peak
    mz2, int2 = _profile_gaussian([500.0, 600.0], [800.0, 1200.0])
    # 22 points → 2 peaks

    spectra = [
        _make_spectrum(
            scan_num=100, ms_level=1, rt=1.0,
            mz_arr=mz1, int_arr=int1,
        ),
        _make_spectrum(
            scan_num=101, ms_level=2, rt=1.05,
            mz_arr=mz2, int_arr=int2,
            precursor_scan_num=100, precursor_mz=500.0,
            iso_lower_off=1.0, iso_upper_off=1.0,
        ),
    ]

    from spectrum import dia_data as dd
    monkeypatch.setattr(dd.mzml, 'read',
                        lambda p: _FakeMzmlReader(spectra))

    d = DIAData()
    d._centroid_enabled = True
    d._centroid_rel_threshold = 1e-3
    d._load_from_mzml('fake.mzML')

    # 1 peak from spectrum 0, 2 peaks from spectrum 1
    assert d._mz_values.shape == (3,), \
        f"centroid expected 3 total peaks, got {d._mz_values.shape}"
    assert list(d._peak_start_idx_list) == [0, 1]
    assert list(d._peak_stop_idx_list) == [1, 3]
    # Recovered m/z within 0.001 Da of true centers.
    assert abs(d._mz_values[0] - 400.0) < 0.001
    assert abs(d._mz_values[1] - 500.0) < 0.001
    assert abs(d._mz_values[2] - 600.0) < 0.001


def test_load_from_mzml_skips_centroid_for_already_centroid(monkeypatch):
    """A spectrum carrying 'centroid spectrum' cv term must be
    passed through verbatim, even with _centroid_enabled=True."""
    spectrum = _make_spectrum(
        scan_num=100, ms_level=1, rt=1.0,
        mz_arr=[400.0, 500.0, 600.0],
        int_arr=[10.0, 20.0, 30.0],
    )
    spectrum['centroid spectrum'] = ''  # mark as already centroid

    from spectrum import dia_data as dd
    monkeypatch.setattr(dd.mzml, 'read',
                        lambda p: _FakeMzmlReader([spectrum]))

    d = DIAData()
    d._centroid_enabled = True
    d._load_from_mzml('fake.mzML')

    # Verbatim pass-through
    assert d._mz_values.shape == (3,)
    np.testing.assert_array_equal(
        d._mz_values, np.array([400.0, 500.0, 600.0], dtype=np.float32))
    np.testing.assert_array_equal(
        d._intensity_values, np.array([10.0, 20.0, 30.0], dtype=np.float32))
```

- [ ] **Step 2: Run tests to verify they fail**

```bash
python -m pytest tests/test_dia_data_load_mzml.py -v -k centroid
```

Expected: `test_load_from_mzml_with_centroid_enabled_compresses_peaks`
FAILS (asserts 3 peaks but gets 22 because centroid is not wired in
yet).

- [ ] **Step 3: Wire `centroid_spectrum` into `_process_single_spectrum`**

Edit `spectrum/dia_data.py`. Add the import near the top of the file
(after the existing `from spectrum.spectrum_utils import match_peak_ppm`):

```python
from spectrum.spectrum_utils import match_peak_ppm, centroid_spectrum
```

Then in `_process_single_spectrum`, locate this block:

```python
        # 获取 m/z 和强度数组
        mz_array = spectrum['m/z array']
        intensity_array = spectrum['intensity array']
```

Replace with:

```python
        # 获取 m/z 和强度数组
        mz_array = spectrum['m/z array']
        intensity_array = spectrum['intensity array']

        # On-load centroiding (spec 2026-06-01-mzml-centroiding-on-load §5.2).
        # Skip if input already carries the centroid cv term, or if disabled.
        if self._centroid_enabled and not _is_already_centroid(spectrum):
            mz_array, intensity_array = centroid_spectrum(
                mz_array, intensity_array,
                rel_threshold=self._centroid_rel_threshold,
            )
```

> The variable rebind means **all downstream code in this method (peak
> count, return value) uses the centroided arrays automatically**.

- [ ] **Step 4: Run integration tests to verify they pass**

```bash
python -m pytest tests/test_dia_data_load_mzml.py -v -k centroid
```

Expected: 2 passed.

- [ ] **Step 5: Run full test suite**

```bash
python -m pytest -q
```

Expected: all tests pass.

- [ ] **Step 6: Commit**

```bash
git add spectrum/dia_data.py tests/test_dia_data_load_mzml.py
git commit -m "feat(dia_data): centroid profile spectra on load

_process_single_spectrum now calls centroid_spectrum(...) when
self._centroid_enabled is True and the input spectrum does not
already carry the 'centroid spectrum' cv term. The centroided arrays
flow naturally through the chunk+concat path introduced in the
previous commit, so peak indices and downstream arrays stay
consistent.

Co-authored-by: Copilot <223556219+Copilot@users.noreply.github.com>"
```

---

## Task 8: Wire config from `DataManager` and update `config.ini`

**Files:**
- Modify: `manager/data_manager.py:30-39` (`get_dia_data_object`)
- Modify: `config.ini` (`[general]` section)
- Test: `tests/test_dia_data_load_mzml.py` (append)

- [ ] **Step 1: Write the failing test**

Append to `tests/test_dia_data_load_mzml.py`:

```python
# ---- DataManager wires config ----

def test_data_manager_passes_centroid_config_to_dia_data(monkeypatch, tmp_path):
    """DataManager.get_dia_data_object reads CENTROID_ENABLED and
    CENTROID_REL_THRESHOLD from [general] and sets them on DIAData
    before _load_from_mzml runs."""
    import configparser
    from manager.data_manager import DataManager

    cfg = configparser.ConfigParser()
    cfg['general'] = {
        'centroid_enabled': 'false',
        'centroid_rel_threshold': '0.005',
    }

    captured = {}

    def fake_load(self, mzml_file_path):
        # Capture the centroid fields at load-time.
        captured['enabled'] = self._centroid_enabled
        captured['threshold'] = self._centroid_rel_threshold
        # Skip actual loading work.
        return None

    monkeypatch.setattr(
        'spectrum.dia_data.DIAData._load_from_mzml', fake_load)

    dm = DataManager(config=cfg, path=str(tmp_path / 'mgr.pkl'))
    dm.get_dia_data_object('does_not_exist.mzML')

    assert captured['enabled'] is False
    assert captured['threshold'] == pytest.approx(0.005)


def test_data_manager_defaults_when_keys_missing(monkeypatch, tmp_path):
    """Missing keys must fall back to DIAData defaults (True / 1e-3)."""
    import configparser
    from manager.data_manager import DataManager

    cfg = configparser.ConfigParser()
    cfg['general'] = {}

    captured = {}

    def fake_load(self, mzml_file_path):
        captured['enabled'] = self._centroid_enabled
        captured['threshold'] = self._centroid_rel_threshold
        return None

    monkeypatch.setattr(
        'spectrum.dia_data.DIAData._load_from_mzml', fake_load)

    dm = DataManager(config=cfg, path=str(tmp_path / 'mgr.pkl'))
    dm.get_dia_data_object('does_not_exist.mzML')

    assert captured['enabled'] is True
    assert captured['threshold'] == pytest.approx(1e-3)
```

- [ ] **Step 2: Run tests to verify they fail**

```bash
python -m pytest tests/test_dia_data_load_mzml.py -v -k data_manager
```

Expected: `test_data_manager_passes_centroid_config_to_dia_data` FAILS
because `get_dia_data_object` doesn't read those keys yet.

- [ ] **Step 3: Update `DataManager.get_dia_data_object`**

Edit `manager/data_manager.py`. Add the import near the top (after
the existing `from spectrum.dia_data import DIAData`):

```python
from constant.keys import ConfigKeys
```

Replace `get_dia_data_object` with:

```python
    def get_dia_data_object(self, tot_raw_path: None | str = None) -> DIAData:
        """ 从路径中读取 dia 数据 """

        dia_data = DIAData()

        # 注入 centroid 配置（spec 2026-06-01-mzml-centroiding-on-load §5.3）
        # 不存在时退回 DIAData 默认值 (True / 1e-3)
        if self._config is not None and self._config.has_section(
                ConfigKeys.GENERAL):
            dia_data._centroid_enabled = self._config.getboolean(
                ConfigKeys.GENERAL, ConfigKeys.CENTROID_ENABLED,
                fallback=dia_data._centroid_enabled)
            dia_data._centroid_rel_threshold = self._config.getfloat(
                ConfigKeys.GENERAL, ConfigKeys.CENTROID_REL_THRESHOLD,
                fallback=dia_data._centroid_rel_threshold)

        dia_data._load_from_mzml(tot_raw_path)

        return dia_data
```

- [ ] **Step 4: Run tests to verify they pass**

```bash
python -m pytest tests/test_dia_data_load_mzml.py -v -k data_manager
```

Expected: 2 passed.

- [ ] **Step 5: Update `config.ini`**

Edit `config.ini`. Locate `[general]`:

```ini
[general]
# 生成特征的模式，0 为相同文件之间进行生成
# 1 为 正常的轻重标进行生成
feature_type = 0

work_directory = ./workspace

mass_tol_ppm = 10

xic_cycle_window = 6

result_file = result.csv
```

Replace with:

```ini
[general]
# 生成特征的模式，0 为相同文件之间进行生成
# 1 为 正常的轻重标进行生成
feature_type = 0

work_directory = ./workspace

mass_tol_ppm = 10

xic_cycle_window = 6

result_file = result.csv

# 加载 mzML 时是否对 profile 谱图做 centroiding。
# 设为 false 退回旧行为（保留 profile，所有点）。
centroid_enabled = true

# centroid 阈值：单张谱图内 intensity < max * 该比值 的局部极大值丢弃。
# 典型范围 1e-4 ~ 1e-2；推荐 1e-3。
centroid_rel_threshold = 0.001
```

> Note: `config.ini` is in `.gitignore` ("User-specific configs"); the
> edit is for your local environment only.

- [ ] **Step 6: Run full test suite**

```bash
python -m pytest -q
```

Expected: all tests pass.

- [ ] **Step 7: Commit**

```bash
git add manager/data_manager.py tests/test_dia_data_load_mzml.py
git commit -m "feat(data_manager): wire centroid config from [general] to DIAData

DataManager.get_dia_data_object now reads CENTROID_ENABLED and
CENTROID_REL_THRESHOLD from the [general] section (with safe
fallbacks to DIAData defaults) and assigns them to the DIAData
instance before _load_from_mzml. This completes the end-to-end
plumbing for on-load centroiding.

Note: config.ini is .gitignore-d; users will need to add the two
new keys manually to enable/configure (defaults are 'on' with 1e-3
threshold).

Co-authored-by: Copilot <223556219+Copilot@users.noreply.github.com>"
```

---

## Final Verification

- [ ] **Run the full test suite one more time**

```bash
python -m pytest -q
```

Expected: all tests pass — 20 new tests added (11 in
`test_centroid_spectrum.py`, 9 in `test_dia_data_load_mzml.py`),
plus all previously-passing tests still pass.

- [ ] **Smoke-check the import graph**

```bash
python -c "
from spectrum.spectrum_utils import centroid_spectrum
from spectrum.dia_data import DIAData, _is_already_centroid
from manager.data_manager import DataManager
from constant.keys import ConfigKeys
print('CENTROID_ENABLED =', ConfigKeys.CENTROID_ENABLED)
print('CENTROID_REL_THRESHOLD =', ConfigKeys.CENTROID_REL_THRESHOLD)
print('DIAData default centroid_enabled =', DIAData()._centroid_enabled)
print('OK')
"
```

Expected output ends with `OK`.

- [ ] **Tell the user**

Report:
- Number of tests added (20)
- Number of commits made (8)
- Reminder to delete any pre-existing `*.dia.npz` cache files in
  `workspace/` before next production run (they will now fail
  `_format_version` check with a clear error message).
- Reminder to add the two new keys to `config.ini` if it predates
  this change (defaults are sensible if keys are absent — both fall
  back to enabled=True, threshold=1e-3).
