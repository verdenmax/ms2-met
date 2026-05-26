# H/L Ratio Consistency + Apex Cycle Offset Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Implement 26 new orthogonal features (6 H/L-ratio consistency + 4 precursor cycle-offset + 16 fragment cycle-offset aggregates) per `docs/specs/2026-05-26-hl-ratio-consistency-and-apex-cycle-offset-design.md`, to give 5Da-window negative-PSM detection a non-shape-similarity discriminator.

**Architecture:** Extend XIC return structs in `spectrum/dia_data.py` with a `cycle_idx` int32 column (computed dynamically, not persisted). Extend `calc_xic_score` in `workflows/single_work.py` with optional `center_rt` / `heavy_center_rt` parameters so both precursor and per-fragment calls produce cycle-offset fields. Wire collection + aggregation into both `single_pair_work` and `multi_batch_work` symmetrically. Backward compatible: existing callers that ignore the new dtype column and don't pass center_rt are unaffected.

**Tech Stack:** Python 3.13, numpy, pytest, conda env `jianyan` at `/home/verden/.conda/envs/jianyan`.

---

## File Structure

**Modified files:**
- `spectrum/dia_data.py` — Add `_ms2_cycle_idx` private method; extend `xic_peaks_extreact` and `xic_ms2_peaks_extract` XIC dtype with `cycle_idx`.
- `workflows/single_work.py` — Add `_calc_cycle_offset` and `_calc_hl_ratio_consistency` helpers; extend `calc_xic_score` signature with `center_rt`/`heavy_center_rt`; extend `extract_ion_numeric_features` with `_max` aggregate; extend `_default_xic_score`; wire features into `single_pair_work` and `multi_batch_work`.

**Modified test files:**
- `tests/test_dia_data_window.py` — Add cycle_idx XIC tests.
- `tests/test_single_work_numerics.py` — Add cycle_offset / H-L-ratio consistency tests, plus a `_max` aggregate test.

No new files are created. No npz cache format changes.

---

## Task 1: DIAData `_ms2_cycle_idx` helper

**Files:**
- Modify: `spectrum/dia_data.py` (add a new method on `DIAData`)
- Test: `tests/test_dia_data_window.py`

- [ ] **Step 1: Write the failing test**

Append to `tests/test_dia_data_window.py`:

```python
def test_ms2_cycle_idx_maps_to_owning_ms1_position():
    """MS2 cycle_idx = position of owning MS1 in ms1_indexs."""
    d = DIAData.__new__(DIAData)
    # Suppose global spectrum indexing:
    #   idx 0,1: MS1 of cycle 0, cycle 1
    #   idx 2,3: MS2 of cycle 0 (precursor_scan_id = ms1 scan_id at idx 0)
    #   idx 4:   MS2 of cycle 1 (precursor_scan_id = ms1 scan_id at idx 1)
    d.ms1_indexs = np.array([0, 1], dtype=np.int32)
    d.ms2_indexs = np.array([2, 3, 4], dtype=np.int32)
    # precursor_scan_ids is keyed by global spectrum index;
    # MS1 entries get -1 (no precursor); MS2 entries get owning MS1 scan_id.
    d.precursor_scan_ids = np.array([-1, -1, 100, 100, 101], dtype=np.int32)
    # _scan_id_to_index maps scan_id -> global index
    d._scan_id_to_index = np.zeros(200, dtype=np.int32)
    d._scan_id_to_index[100] = 0
    d._scan_id_to_index[101] = 1

    # MS2 at ms2_indexs[0]==2 belongs to MS1 at global idx 0 == ms1_indexs[0]
    assert d._ms2_cycle_idx(2) == 0
    # MS2 at ms2_indexs[1]==3 belongs to MS1 at global idx 0 (cycle 0)
    assert d._ms2_cycle_idx(3) == 0
    # MS2 at ms2_indexs[2]==4 belongs to MS1 at global idx 1 (cycle 1)
    assert d._ms2_cycle_idx(4) == 1


def test_ms2_cycle_idx_returns_minus_one_when_owning_ms1_missing():
    """If the owning MS1 isn't in ms1_indexs (shouldn't happen but be safe),
    return -1 rather than a wrong cycle number."""
    d = DIAData.__new__(DIAData)
    d.ms1_indexs = np.array([0, 5], dtype=np.int32)
    d.precursor_scan_ids = np.array([-1, 7, -1, 7, -1, -1], dtype=np.int32)
    d._scan_id_to_index = np.zeros(20, dtype=np.int32)
    d._scan_id_to_index[7] = 3  # but 3 isn't in ms1_indexs

    # MS2 at global idx 1: owning MS1 = scan_id 7 -> global idx 3, not in ms1_indexs
    assert d._ms2_cycle_idx(1) == -1
```

- [ ] **Step 2: Run test to verify it fails**

```bash
conda run -n jianyan pytest tests/test_dia_data_window.py::test_ms2_cycle_idx_maps_to_owning_ms1_position -v
```

Expected: FAIL with `AttributeError: 'DIAData' object has no attribute '_ms2_cycle_idx'`

- [ ] **Step 3: Implement `_ms2_cycle_idx`**

Add this method to the `DIAData` class in `spectrum/dia_data.py` (insert after `get_spectrum_by_index`, around line 500):

```python
    def _ms2_cycle_idx(self, global_ms2_idx: int) -> int:
        """Return the cycle index (= position in ms1_indexs) that owns this MS2.

        DIA cycle = one MS1 followed by N MS2. The owning MS1 is identified by
        precursor_scan_ids[global_ms2_idx]. Return -1 if the owning MS1 isn't
        found in ms1_indexs (defensive; shouldn't happen on well-formed data).
        """
        if (self.precursor_scan_ids is None or
                self._scan_id_to_index is None or
                self.ms1_indexs is None):
            return -1
        ms1_scan_id = int(self.precursor_scan_ids[global_ms2_idx])
        if ms1_scan_id < 0:
            return -1
        ms1_global_idx = int(self._scan_id_to_index[ms1_scan_id])
        pos = int(np.searchsorted(self.ms1_indexs, ms1_global_idx))
        if (pos < len(self.ms1_indexs) and
                int(self.ms1_indexs[pos]) == ms1_global_idx):
            return pos
        return -1
```

- [ ] **Step 4: Run tests to verify they pass**

```bash
conda run -n jianyan pytest tests/test_dia_data_window.py -v
```

Expected: All tests pass, including the two new ones.

- [ ] **Step 5: Commit**

```bash
git add spectrum/dia_data.py tests/test_dia_data_window.py
git commit -m "feat(dia_data): add _ms2_cycle_idx helper

Maps a global MS2 spectrum index to its owning MS1's position in
ms1_indexs (the cycle number). Returns -1 defensively when the owning
MS1 isn't tracked in ms1_indexs.

Foundation for apex_cycle_offset features (spec 2026-05-26).

Co-authored-by: Copilot <223556219+Copilot@users.noreply.github.com>"
```

---

## Task 2: MS1 XIC dtype with cycle_idx

**Files:**
- Modify: `spectrum/dia_data.py` `xic_peaks_extreact` (~line 672)
- Test: `tests/test_dia_data_window.py`

- [ ] **Step 1: Write the failing test**

Append to `tests/test_dia_data_window.py`:

```python
def test_ms1_xic_returns_cycle_idx_field():
    """xic_peaks_extreact dtype includes cycle_idx = ms1_indexs position."""
    d = DIAData.__new__(DIAData)
    # 5 MS1 spectra at global indices 0..4, equally spaced RT
    d.ms1_indexs = np.array([0, 1, 2, 3, 4], dtype=np.int32)
    d.ms1_indexs_rt = np.array([10.0, 20.0, 30.0, 40.0, 50.0], dtype=np.float32)
    d.rt_values = np.array([10.0, 20.0, 30.0, 40.0, 50.0], dtype=np.float32)
    # Empty peak lists so match_peak_ppm returns (nan, 0) safely
    d._peak_start_idx_list = np.zeros(5, dtype=np.int64)
    d._peak_stop_idx_list = np.zeros(5, dtype=np.int64)
    d._mz_values = np.array([], dtype=np.float32)
    d._intensity_values = np.array([], dtype=np.float32)

    xic = d.xic_peaks_extreact(
        rt=np.float32(30.0), xic_cycle_window=2,
        precursor_mz=np.float32(500.0), mass_tol_ppm=np.float32(10.0))

    assert "cycle_idx" in xic.dtype.names
    # Center = ms1_indexs[2] (RT=30); window=2 -> indices 0..4 of ms1_indexs
    assert list(xic["cycle_idx"]) == [0, 1, 2, 3, 4]
```

- [ ] **Step 2: Run to verify it fails**

```bash
conda run -n jianyan pytest tests/test_dia_data_window.py::test_ms1_xic_returns_cycle_idx_field -v
```

Expected: FAIL with `'cycle_idx' not in dtype.names`

- [ ] **Step 3: Modify `xic_peaks_extreact`**

In `spectrum/dia_data.py`, replace the body of `xic_peaks_extreact` (currently lines ~672-707). Find this section:

```python
        # 遍历所有 index
        for index in self.ms1_indexs[start_index:end_index]:

            # 当是 ms1 谱图的时候，取出这个precursor_mz 对应的信息
            (mz_arr, intensity_arr) = self.get_spectrum_by_index(index)

            (ppm_error, match_intensity) = match_peak_ppm(
                mz_arr, intensity_arr, precursor_mz, mass_tol_ppm)

            ans.append(
                {"rt": self.rt_values[index],
                 "ppm_error": ppm_error,
                 "intensity": match_intensity})

        dtype = [("rt", "f8"), ("ppm_error", "f8"), ("intensity", "f8")]

        # 把 list[dict] 转成结构化 ndarray
        arr = np.array([tuple(d.values()) for d in ans], dtype=dtype)

        return arr
```

Replace with:

```python
        # 遍历所有 index, 并记录 cycle_idx (= position in ms1_indexs)
        for local_pos, index in enumerate(
                self.ms1_indexs[start_index:end_index]):
            cycle_idx = start_index + local_pos

            # 当是 ms1 谱图的时候，取出这个precursor_mz 对应的信息
            (mz_arr, intensity_arr) = self.get_spectrum_by_index(index)

            (ppm_error, match_intensity) = match_peak_ppm(
                mz_arr, intensity_arr, precursor_mz, mass_tol_ppm)

            ans.append(
                {"rt": self.rt_values[index],
                 "ppm_error": ppm_error,
                 "intensity": match_intensity,
                 "cycle_idx": cycle_idx})

        dtype = [("rt", "f8"), ("ppm_error", "f8"),
                 ("intensity", "f8"), ("cycle_idx", "i4")]

        # 把 list[dict] 转成结构化 ndarray
        arr = np.array([tuple(d.values()) for d in ans], dtype=dtype)

        return arr
```

- [ ] **Step 4: Run test to verify it passes**

```bash
conda run -n jianyan pytest tests/test_dia_data_window.py::test_ms1_xic_returns_cycle_idx_field -v
```

Expected: PASS.

- [ ] **Step 5: Sanity check existing tests still pass**

```bash
conda run -n jianyan pytest tests/test_dia_data_window.py -v
```

Expected: All tests in this file pass.

- [ ] **Step 6: Commit**

```bash
git add spectrum/dia_data.py tests/test_dia_data_window.py
git commit -m "feat(dia_data): MS1 XIC dtype carries cycle_idx field

xic_peaks_extreact now returns a structured array with a cycle_idx
int32 column equal to the entry's position in ms1_indexs. Existing
field-name accessors (xic[\"rt\"] etc.) unaffected.

Co-authored-by: Copilot <223556219+Copilot@users.noreply.github.com>"
```

---

## Task 3: MS2 XIC dtype with cycle_idx

**Files:**
- Modify: `spectrum/dia_data.py` `xic_ms2_peaks_extract` (~line 534)
- Test: `tests/test_dia_data_window.py`

- [ ] **Step 1: Write the failing test**

Append to `tests/test_dia_data_window.py`:

```python
def test_ms2_xic_returns_cycle_idx_field():
    """xic_ms2_peaks_extract dtype includes cycle_idx that maps
    each entry to its owning MS1's position in ms1_indexs."""
    d = DIAData.__new__(DIAData)
    # Layout (global indices):
    #   0: MS1 (cycle 0), 1: MS2 of cycle 0
    #   2: MS1 (cycle 1), 3: MS2 of cycle 1
    #   4: MS1 (cycle 2), 5: MS2 of cycle 2
    d.ms1_indexs = np.array([0, 2, 4], dtype=np.int32)
    d.ms1_indexs_rt = np.array([10.0, 30.0, 50.0], dtype=np.float32)
    d.ms2_indexs = np.array([1, 3, 5], dtype=np.int32)
    d.ms2_indexs_rt = np.array([15.0, 35.0, 55.0], dtype=np.float32)
    d.rt_values = np.array(
        [10.0, 15.0, 30.0, 35.0, 50.0, 55.0], dtype=np.float32)
    d.precursor_scan_ids = np.array(
        [-1, 100, -1, 101, -1, 102], dtype=np.int32)
    d._scan_id_to_index = np.zeros(200, dtype=np.int32)
    d._scan_id_to_index[100] = 0
    d._scan_id_to_index[101] = 2
    d._scan_id_to_index[102] = 4
    # All MS2 windows contain precursor_mz=500.0
    d._precursor_lower_mz = np.array(
        [np.nan, 499.0, np.nan, 499.0, np.nan, 499.0], dtype=np.float64)
    d._precursor_upper_mz = np.array(
        [np.nan, 501.0, np.nan, 501.0, np.nan, 501.0], dtype=np.float64)
    # Empty peak lists so match_peak_ppm returns harmlessly
    d._peak_start_idx_list = np.zeros(6, dtype=np.int64)
    d._peak_stop_idx_list = np.zeros(6, dtype=np.int64)
    d._mz_values = np.array([], dtype=np.float32)
    d._intensity_values = np.array([], dtype=np.float32)
    d._cycle_left_precursor = np.array([499.0], dtype=np.float32)

    xic, _ = d.xic_ms2_peaks_extract(
        rt=np.float32(35.0), xic_cycle_window=1,
        precursor_mz=np.float32(500.0), ions_mass=np.float32(200.0),
        mass_tol_ppm=np.float32(10.0))

    assert "cycle_idx" in xic.dtype.names
    # Center is MS2 at rt=35 (global idx 3, cycle 1).
    # Window=1 -> 1 left + center + 1 right = 3 entries; cycles [0,1,2].
    assert list(xic["cycle_idx"]) == [0, 1, 2]


def test_ms2_xic_empty_path_still_has_cycle_idx_in_dtype():
    """Early-return path (no matching window) must still emit cycle_idx."""
    d = DIAData.__new__(DIAData)
    d.ms2_indexs = np.array([], dtype=np.int32)
    d.ms2_indexs_rt = np.array([], dtype=np.float32)

    xic, _ = d.xic_ms2_peaks_extract(
        rt=np.float32(10.0), xic_cycle_window=3,
        precursor_mz=np.float32(500.0), ions_mass=np.float32(200.0),
        mass_tol_ppm=np.float32(10.0))
    assert "cycle_idx" in xic.dtype.names
    assert len(xic) == 0
```

- [ ] **Step 2: Run to verify failure**

```bash
conda run -n jianyan pytest tests/test_dia_data_window.py::test_ms2_xic_returns_cycle_idx_field tests/test_dia_data_window.py::test_ms2_xic_empty_path_still_has_cycle_idx_in_dtype -v
```

Expected: FAIL.

- [ ] **Step 3: Modify `xic_ms2_peaks_extract`**

In `spectrum/dia_data.py`, find the **three** dtype definitions inside `xic_ms2_peaks_extract` (lines ~547, ~583, ~650) and update each to include `cycle_idx`:

Replace (at line ~547):
```python
        if self.ms2_indexs is None or len(self.ms2_indexs) == 0:
            dtype = [("rt", "f8"), ("ppm_error", "f8"), ("intensity", "f8")]
            return np.array([], dtype=dtype), 0.0
```
With:
```python
        if self.ms2_indexs is None or len(self.ms2_indexs) == 0:
            dtype = [("rt", "f8"), ("ppm_error", "f8"),
                     ("intensity", "f8"), ("cycle_idx", "i4")]
            return np.array([], dtype=dtype), 0.0
```

Replace (at line ~583):
```python
        if center_idx is None:
            # 没有找到任何窗口匹配的 MS2 谱图
            dtype = [("rt", "f8"), ("ppm_error", "f8"), ("intensity", "f8")]
            logging.warn("没有找到任何匹配ms2 窗口，可能是重标超出当前 raw 的范围了")
```
With:
```python
        if center_idx is None:
            # 没有找到任何窗口匹配的 MS2 谱图
            dtype = [("rt", "f8"), ("ppm_error", "f8"),
                     ("intensity", "f8"), ("cycle_idx", "i4")]
            logging.warn("没有找到任何匹配ms2 窗口，可能是重标超出当前 raw 的范围了")
```

Find the Step 5 collection loop (line ~628-651):

```python
        # Step 5: 处理每个谱图
        for global_idx in selected_global_indices:
            mz_arr, intensity_arr = self.get_spectrum_by_index(global_idx)
            total_intensity += np.sum(intensity_arr)

            ppm_error = 0.0
            match_intensity = 0.0

            for charge in range(1, 3):
                theo_mz = (ions_mass + charge * protonmass) / charge
                tot_ppm_error, tot_match_intensity = match_peak_ppm(
                    mz_arr, intensity_arr, theo_mz, mass_tol_ppm
                )
                if not np.isnan(tot_ppm_error):
                    ppm_error += tot_ppm_error
                match_intensity += tot_match_intensity

            ans.append({
                "rt": self.rt_values[global_idx],
                "ppm_error": ppm_error,
                "intensity": match_intensity
            })

        dtype = [("rt", "f8"), ("ppm_error", "f8"), ("intensity", "f8")]
        arr = np.array([tuple(d.values()) for d in ans], dtype=dtype)

        return arr, total_intensity
```

Replace with:

```python
        # Step 5: 处理每个谱图
        for global_idx in selected_global_indices:
            mz_arr, intensity_arr = self.get_spectrum_by_index(global_idx)
            total_intensity += np.sum(intensity_arr)

            ppm_error = 0.0
            match_intensity = 0.0

            for charge in range(1, 3):
                theo_mz = (ions_mass + charge * protonmass) / charge
                tot_ppm_error, tot_match_intensity = match_peak_ppm(
                    mz_arr, intensity_arr, theo_mz, mass_tol_ppm
                )
                if not np.isnan(tot_ppm_error):
                    ppm_error += tot_ppm_error
                match_intensity += tot_match_intensity

            ans.append({
                "rt": self.rt_values[global_idx],
                "ppm_error": ppm_error,
                "intensity": match_intensity,
                "cycle_idx": self._ms2_cycle_idx(int(global_idx)),
            })

        dtype = [("rt", "f8"), ("ppm_error", "f8"),
                 ("intensity", "f8"), ("cycle_idx", "i4")]
        arr = np.array([tuple(d.values()) for d in ans], dtype=dtype)

        return arr, total_intensity
```

- [ ] **Step 4: Run tests to verify they pass**

```bash
conda run -n jianyan pytest tests/test_dia_data_window.py -v
```

Expected: All tests pass.

- [ ] **Step 5: Commit**

```bash
git add spectrum/dia_data.py tests/test_dia_data_window.py
git commit -m "feat(dia_data): MS2 XIC dtype carries cycle_idx field

xic_ms2_peaks_extract now returns a structured array with a cycle_idx
int32 column. For each collected MS2 spectrum cycle_idx is computed via
_ms2_cycle_idx (reverse-lookup of owning MS1 position in ms1_indexs).
All three dtype declarations (empty/early-return/normal) updated.

Co-authored-by: Copilot <223556219+Copilot@users.noreply.github.com>"
```

---

## Task 4: Extend `extract_ion_numeric_features` with `_max`

**Files:**
- Modify: `workflows/single_work.py` `extract_ion_numeric_features` (~line 691)
- Test: `tests/test_single_work_numerics.py`

- [ ] **Step 1: Write the failing test**

Append to `tests/test_single_work_numerics.py`:

```python
def test_extract_ion_numeric_features_emits_max_field():
    """extract_ion_numeric_features must include `_max` so caller can
    track the worst-offending fragment (e.g. largest apex_cycle_offset)."""
    from workflows.single_work import extract_ion_numeric_features
    out = extract_ion_numeric_features([0.1, 0.5, 0.3, 0.9, 0.2], "demo")
    assert "demo_max" in out
    assert abs(out["demo_max"] - 0.9) < 1e-9
    assert "demo_mean" in out  # existing fields preserved
    assert "demo_p50" in out
    assert "demo_std" in out


def test_extract_ion_numeric_features_max_empty_list_is_zero():
    """Empty list -> demo_max = 0.0 (consistent with other defaults)."""
    from workflows.single_work import extract_ion_numeric_features
    out = extract_ion_numeric_features([], "demo")
    assert out["demo_max"] == 0.0
```

- [ ] **Step 2: Run to verify failure**

```bash
conda run -n jianyan pytest tests/test_single_work_numerics.py::test_extract_ion_numeric_features_emits_max_field -v
```

Expected: FAIL with `'demo_max' not in out`.

- [ ] **Step 3: Modify `extract_ion_numeric_features`**

In `workflows/single_work.py`, replace the function (currently lines ~691-707):

```python
def extract_ion_numeric_features(values: list, prefix: str) -> dict:
    """
    对碎片级数值列表（如 apex_delta、mz_err）计算均值、中位数和标准差。
    清除 NaN/Inf 值后统计。
    """
    clean_vals = [v for v in values if not np.isnan(v) and np.isfinite(v)]
    if len(clean_vals) == 0:
        return {
            f"{prefix}_mean": 0.0,
            f"{prefix}_p50": 0.0,
            f"{prefix}_std": 0.0,
        }
    return {
        f"{prefix}_mean": float(np.mean(clean_vals)),
        f"{prefix}_p50": float(np.median(clean_vals)),
        f"{prefix}_std": float(np.std(clean_vals)),
    }
```

Replace with:

```python
def extract_ion_numeric_features(values: list, prefix: str) -> dict:
    """
    对碎片级数值列表（如 apex_delta、mz_err、cycle_offset）计算均值、
    中位数、标准差和最大值。清除 NaN/Inf 值后统计。
    """
    clean_vals = [v for v in values if not np.isnan(v) and np.isfinite(v)]
    if len(clean_vals) == 0:
        return {
            f"{prefix}_mean": 0.0,
            f"{prefix}_p50": 0.0,
            f"{prefix}_std": 0.0,
            f"{prefix}_max": 0.0,
        }
    return {
        f"{prefix}_mean": float(np.mean(clean_vals)),
        f"{prefix}_p50": float(np.median(clean_vals)),
        f"{prefix}_std": float(np.std(clean_vals)),
        f"{prefix}_max": float(np.max(clean_vals)),
    }
```

- [ ] **Step 4: Run tests**

```bash
conda run -n jianyan pytest tests/test_single_work_numerics.py -v
```

Expected: All tests pass.

- [ ] **Step 5: Commit**

```bash
git add workflows/single_work.py tests/test_single_work_numerics.py
git commit -m "feat(single_work): extract_ion_numeric_features emits _max

Add max aggregate alongside mean/p50/std. Lets feature consumers track
the worst-offending fragment for any numeric attribute. Existing
all_apex_delta / all_mz_err / all_cosine / all_snr aggregates
automatically gain a _max companion column.

Co-authored-by: Copilot <223556219+Copilot@users.noreply.github.com>"
```

---

## Task 5: `_calc_cycle_offset` helper

**Files:**
- Modify: `workflows/single_work.py` (add helper near other `_calc_*` private helpers, around line 710)
- Test: `tests/test_single_work_numerics.py`

- [ ] **Step 1: Write the failing test**

Append to `tests/test_single_work_numerics.py`:

```python
def _make_xic(cycles, rts, intensities):
    dt = [("rt", "f8"), ("ppm_error", "f8"),
          ("intensity", "f8"), ("cycle_idx", "i4")]
    arr = np.zeros(len(cycles), dtype=dt)
    arr["cycle_idx"] = cycles
    arr["rt"] = rts
    arr["intensity"] = intensities
    return arr


def test_calc_cycle_offset_apex_at_center_returns_zero():
    """When apex aligns with the center RT entry, offset is (0, 0)."""
    from workflows.single_work import _calc_cycle_offset
    xic = _make_xic(
        cycles=[5, 6, 7, 8, 9], rts=[10, 11, 12, 13, 14],
        intensities=[1, 5, 100, 5, 1])
    abs_off, signed = _calc_cycle_offset(xic, center_rt=12.0)
    assert abs_off == 0
    assert signed == 0


def test_calc_cycle_offset_apex_before_center_is_negative():
    """Apex one cycle earlier than center -> signed = -1, abs = 1."""
    from workflows.single_work import _calc_cycle_offset
    xic = _make_xic(
        cycles=[5, 6, 7, 8, 9], rts=[10, 11, 12, 13, 14],
        intensities=[1, 100, 5, 1, 1])  # apex at cycle 6
    abs_off, signed = _calc_cycle_offset(xic, center_rt=12.0)
    assert signed == -1
    assert abs_off == 1


def test_calc_cycle_offset_apex_after_center_is_positive():
    """Apex two cycles after center -> signed = +2, abs = 2."""
    from workflows.single_work import _calc_cycle_offset
    xic = _make_xic(
        cycles=[5, 6, 7, 8, 9], rts=[10, 11, 12, 13, 14],
        intensities=[1, 1, 5, 1, 100])  # apex at cycle 9
    abs_off, signed = _calc_cycle_offset(xic, center_rt=12.0)
    assert signed == 2
    assert abs_off == 2


def test_calc_cycle_offset_empty_xic_returns_zero():
    """Empty XIC -> (0, 0)."""
    from workflows.single_work import _calc_cycle_offset
    xic = _make_xic(cycles=[], rts=[], intensities=[])
    abs_off, signed = _calc_cycle_offset(xic, center_rt=0.0)
    assert (abs_off, signed) == (0, 0)


def test_calc_cycle_offset_skips_invalid_cycle_idx():
    """cycle_idx == -1 entries (defensive) are excluded from center search,
    and an apex with cycle_idx == -1 returns (0, 0)."""
    from workflows.single_work import _calc_cycle_offset
    # Center RT entry has cycle_idx == -1 -> picks next-closest valid entry
    xic = _make_xic(
        cycles=[5, -1, 7, 8, 9], rts=[10, 11, 12, 13, 14],
        intensities=[1, 5, 100, 5, 1])  # apex at idx 2 (cycle 7)
    abs_off, signed = _calc_cycle_offset(xic, center_rt=11.0)
    # All cycle_idx>=0 entries: cycles [5,7,8,9] at rts [10,12,13,14]
    # Closest to 11 is rt=10 (cycle 5) or rt=12 (cycle 7), tie -> argmin picks rt=10
    # apex cycle = 7, center cycle = 5 -> signed = 2
    assert signed == 2
    assert abs_off == 2

    # If apex itself has cycle_idx -1 -> return (0, 0)
    xic2 = _make_xic(
        cycles=[5, 6, -1, 8, 9], rts=[10, 11, 12, 13, 14],
        intensities=[1, 1, 100, 1, 1])  # apex at idx 2 (cycle_idx -1)
    abs_off2, signed2 = _calc_cycle_offset(xic2, center_rt=11.0)
    assert (abs_off2, signed2) == (0, 0)
```

- [ ] **Step 2: Run to verify failure**

```bash
conda run -n jianyan pytest tests/test_single_work_numerics.py::test_calc_cycle_offset_apex_at_center_returns_zero -v
```

Expected: FAIL with `ImportError: cannot import name '_calc_cycle_offset'`.

- [ ] **Step 3: Implement `_calc_cycle_offset`**

In `workflows/single_work.py`, add this helper near the other `_calc_*` helpers (right after `_calc_snr`, around line 756):

```python
def _calc_cycle_offset(xic: np.ndarray, center_rt: float) -> tuple[int, int]:
    """Compute how far the intensity apex is from the center RT, in cycles.

    Returns (abs_offset, signed_offset). signed < 0 means apex is at an
    earlier cycle than center_rt; > 0 means later.

    The "center" is the cycle whose RT is closest to center_rt (among
    entries with valid cycle_idx >= 0). The "apex" is the cycle at
    argmax(intensity). Both returned values are integer cycle counts.

    Returns (0, 0) for empty XIC, all-invalid cycle_idx, or apex with
    cycle_idx == -1 (defensive — shouldn't happen on well-formed data).
    """
    if len(xic) == 0:
        return 0, 0
    valid_mask = xic["cycle_idx"] >= 0
    if not np.any(valid_mask):
        return 0, 0
    valid_xic = xic[valid_mask]
    # Center: cycle whose RT is closest to center_rt (only valid entries)
    center_local_idx = int(np.argmin(np.abs(valid_xic["rt"] - center_rt)))
    center_cycle = int(valid_xic["cycle_idx"][center_local_idx])
    # Apex: argmax on the full XIC (intensity is the source of truth);
    # if the apex entry has cycle_idx == -1, we have no useful offset.
    apex_global_idx = int(np.argmax(xic["intensity"]))
    apex_cycle = int(xic["cycle_idx"][apex_global_idx])
    if apex_cycle < 0:
        return 0, 0
    signed = apex_cycle - center_cycle
    return abs(signed), signed
```

- [ ] **Step 4: Run tests**

```bash
conda run -n jianyan pytest tests/test_single_work_numerics.py -v
```

Expected: All tests pass.

- [ ] **Step 5: Commit**

```bash
git add workflows/single_work.py tests/test_single_work_numerics.py
git commit -m "feat(single_work): add _calc_cycle_offset helper

Computes (abs, signed) cycle offset between intensity apex and the
XIC entry whose RT is closest to a caller-supplied center_rt. Uses
the cycle_idx field added to XIC dtype in the prior commits.

Co-authored-by: Copilot <223556219+Copilot@users.noreply.github.com>"
```

---

## Task 6: Extend `calc_xic_score` and `_default_xic_score`

**Files:**
- Modify: `workflows/single_work.py` `_default_xic_score` (~line 758) and `calc_xic_score` (~line 775)
- Test: `tests/test_single_work_numerics.py`

- [ ] **Step 1: Write the failing test**

Append to `tests/test_single_work_numerics.py`:

```python
def test_calc_xic_score_emits_cycle_offset_when_center_rt_provided():
    """calc_xic_score(light, heavy, center_rt=...) returns 4 new fields:
    light_apex_cycle_offset, light_apex_cycle_offset_signed,
    heavy_apex_cycle_offset, heavy_apex_cycle_offset_signed."""
    from workflows.single_work import calc_xic_score
    dt = [("rt", "f8"), ("ppm_error", "f8"),
          ("intensity", "f8"), ("cycle_idx", "i4")]
    n = 5
    light = np.zeros(n, dtype=dt)
    light["rt"] = [10, 11, 12, 13, 14]
    light["cycle_idx"] = [0, 1, 2, 3, 4]
    light["intensity"] = [1, 5, 100, 5, 1]  # apex at cycle 2 (center)
    heavy = light.copy()
    heavy["intensity"] = [1, 100, 5, 1, 1]  # apex at cycle 1 (one early)

    result = calc_xic_score(light, heavy, center_rt=12.0)
    assert result["light_apex_cycle_offset"] == 0
    assert result["light_apex_cycle_offset_signed"] == 0
    assert result["heavy_apex_cycle_offset"] == 1
    assert result["heavy_apex_cycle_offset_signed"] == -1


def test_calc_xic_score_supports_separate_heavy_center_rt():
    """When light and heavy come from different DIAData (multi_batch_work),
    heavy may have a different RT origin."""
    from workflows.single_work import calc_xic_score
    dt = [("rt", "f8"), ("ppm_error", "f8"),
          ("intensity", "f8"), ("cycle_idx", "i4")]
    n = 5
    light = np.zeros(n, dtype=dt)
    light["rt"] = [10, 11, 12, 13, 14]
    light["cycle_idx"] = [0, 1, 2, 3, 4]
    light["intensity"] = [1, 5, 100, 5, 1]

    # Heavy XIC at a different RT range (different raw)
    heavy = np.zeros(n, dtype=dt)
    heavy["rt"] = [20, 21, 22, 23, 24]
    heavy["cycle_idx"] = [10, 11, 12, 13, 14]
    heavy["intensity"] = [1, 5, 100, 5, 1]  # apex at cycle 12

    result = calc_xic_score(
        light, heavy, center_rt=12.0, heavy_center_rt=22.0)
    assert result["light_apex_cycle_offset"] == 0
    assert result["heavy_apex_cycle_offset"] == 0


def test_calc_xic_score_omits_cycle_offset_default_zero():
    """Backwards compat: not passing center_rt returns zero for new fields."""
    from workflows.single_work import calc_xic_score
    dt = [("rt", "f8"), ("ppm_error", "f8"),
          ("intensity", "f8"), ("cycle_idx", "i4")]
    n = 3
    light = np.zeros(n, dtype=dt)
    light["rt"] = [10, 11, 12]
    light["cycle_idx"] = [0, 1, 2]
    light["intensity"] = [1, 100, 1]
    heavy = light.copy()
    result = calc_xic_score(light, heavy)
    assert result["light_apex_cycle_offset"] == 0
    assert result["light_apex_cycle_offset_signed"] == 0
    assert result["heavy_apex_cycle_offset"] == 0
    assert result["heavy_apex_cycle_offset_signed"] == 0


def test_default_xic_score_has_cycle_offset_zero_fields():
    """The early-return default dict must include all 4 cycle offset keys."""
    from workflows.single_work import _default_xic_score
    d = _default_xic_score()
    assert d["light_apex_cycle_offset"] == 0
    assert d["light_apex_cycle_offset_signed"] == 0
    assert d["heavy_apex_cycle_offset"] == 0
    assert d["heavy_apex_cycle_offset_signed"] == 0
```

- [ ] **Step 2: Run to verify failure**

```bash
conda run -n jianyan pytest tests/test_single_work_numerics.py::test_calc_xic_score_emits_cycle_offset_when_center_rt_provided -v
```

Expected: FAIL (`light_apex_cycle_offset` not in result).

- [ ] **Step 3: Update `_default_xic_score`**

In `workflows/single_work.py`, replace `_default_xic_score` (currently lines ~758-772):

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
    }
```

With:

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

- [ ] **Step 4: Update `calc_xic_score` signature and body**

In `workflows/single_work.py`, change the signature line (~line 775):

```python
def calc_xic_score(
    light_xic: np.array, heavy_xic: np.array,
    intensity_threshold: float = 1e-10
) -> dict:
```

To:

```python
def calc_xic_score(
    light_xic: np.array, heavy_xic: np.array,
    center_rt: float | None = None,
    heavy_center_rt: float | None = None,
    intensity_threshold: float = 1e-10
) -> dict:
```

Then find the `result["intensity_ratio"] = intensity_ratio` line in the `rt_start >= rt_end` early-return block (~line 832), and **right above** the `return result` add cycle offset handling. Find this block:

```python
    if rt_start >= rt_end:
        result = _default_xic_score()
        result["mz_avg_err"] = mz_avg_err
        result["apex_delta"] = apex_delta
        result["apex_delta_signed"] = apex_delta_signed
        result["light_max_int"] = light_max_int
        result["heavy_max_int"] = heavy_max_int
        result["intensity_ratio"] = intensity_ratio
        return result
```

Replace with:

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

Then find the final return block (~line 872-884):

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
    }
```

Replace with:

```python
    if center_rt is not None:
        l_abs, l_sig = _calc_cycle_offset(light_xic, center_rt)
        h_center = (heavy_center_rt
                    if heavy_center_rt is not None else center_rt)
        h_abs, h_sig = _calc_cycle_offset(heavy_xic, h_center)
    else:
        l_abs = l_sig = h_abs = h_sig = 0

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

- [ ] **Step 5: Run tests**

```bash
conda run -n jianyan pytest tests/test_single_work_numerics.py -v
```

Expected: All tests pass (including all 4 new ones and existing `test_calc_xic_score_sorted_unsorted_give_same_pearson`, `test_apex_delta_signed_emitted_alongside_unsigned`).

- [ ] **Step 6: Commit**

```bash
git add workflows/single_work.py tests/test_single_work_numerics.py
git commit -m "feat(single_work): calc_xic_score emits apex_cycle_offset

Add center_rt / heavy_center_rt optional params (None = backward
compatible). When set, calc_xic_score returns 4 new fields:
light_apex_cycle_offset[_signed], heavy_apex_cycle_offset[_signed].
heavy_center_rt is independent so multi_batch_work (different DIAData
for light vs heavy) can pass two RT origins.

_default_xic_score updated to include the 4 zero fields so the
rt_start >= rt_end early-return path is consistent.

Co-authored-by: Copilot <223556219+Copilot@users.noreply.github.com>"
```

---

## Task 7: `_calc_hl_ratio_consistency` helper

**Files:**
- Modify: `workflows/single_work.py` (add helper near other `_calc_*` helpers, after `_calc_cycle_offset`)
- Test: `tests/test_single_work_numerics.py`

- [ ] **Step 1: Write the failing test**

Append to `tests/test_single_work_numerics.py`:

```python
def test_calc_hl_ratio_consistency_basic_std_and_mad():
    """Returns (std, mad) of log10(ratios > 0)."""
    from workflows.single_work import _calc_hl_ratio_consistency
    # ratios: 1, 10, 100 -> log10 = 0, 1, 2 -> mean=1, std=sqrt(2/3)
    std_v, mad_v = _calc_hl_ratio_consistency([1.0, 10.0, 100.0])
    assert abs(std_v - np.std([0.0, 1.0, 2.0])) < 1e-9
    # median = 1.0 -> |log10-median| = [1, 0, 1] -> mad = median = 1.0
    assert abs(mad_v - 1.0) < 1e-9


def test_calc_hl_ratio_consistency_drops_non_positive():
    """ratios <= 0 are excluded from log10."""
    from workflows.single_work import _calc_hl_ratio_consistency
    std_v, mad_v = _calc_hl_ratio_consistency([1.0, 0.0, -5.0, 100.0])
    # Only [1, 100] survive -> log10 = [0, 2] -> std=1, median=1, mad=1
    assert abs(std_v - 1.0) < 1e-9
    assert abs(mad_v - 1.0) < 1e-9


def test_calc_hl_ratio_consistency_empty_list_returns_zero():
    """Empty list -> (0.0, 0.0)."""
    from workflows.single_work import _calc_hl_ratio_consistency
    assert _calc_hl_ratio_consistency([]) == (0.0, 0.0)
    assert _calc_hl_ratio_consistency([0.0, -1.0]) == (0.0, 0.0)


def test_calc_hl_ratio_consistency_single_element_std_is_nan():
    """count=1 -> std is NaN (consistent with Bug #21 convention); mad=0."""
    import math
    from workflows.single_work import _calc_hl_ratio_consistency
    std_v, mad_v = _calc_hl_ratio_consistency([5.0])
    assert math.isnan(std_v)
    assert mad_v == 0.0
```

- [ ] **Step 2: Run to verify failure**

```bash
conda run -n jianyan pytest tests/test_single_work_numerics.py::test_calc_hl_ratio_consistency_basic_std_and_mad -v
```

Expected: FAIL with `ImportError`.

- [ ] **Step 3: Implement `_calc_hl_ratio_consistency`**

In `workflows/single_work.py`, add this helper right after `_calc_cycle_offset`:

```python
def _calc_hl_ratio_consistency(ratios: list) -> tuple[float, float]:
    """Compute consistency of light/heavy intensity ratios across fragments.

    Returns (std, mad) of log10(ratio) over the input list. Non-positive
    ratios are dropped (cannot take log). std uses NaN for count==1 to
    match the existing single-element convention (see Bug #21 in
    extract_ion_pearson_features). mad is 0 for empty input, otherwise
    median absolute deviation from the median.
    """
    log_ratios = [float(np.log10(r)) for r in ratios if r > 0]
    count = len(log_ratios)
    if count == 0:
        return 0.0, 0.0
    if count == 1:
        return float("nan"), 0.0
    arr = np.asarray(log_ratios, dtype="f8")
    std_v = float(np.std(arr))
    med = float(np.median(arr))
    mad_v = float(np.median(np.abs(arr - med)))
    return std_v, mad_v
```

- [ ] **Step 4: Run tests**

```bash
conda run -n jianyan pytest tests/test_single_work_numerics.py -v
```

Expected: All tests pass.

- [ ] **Step 5: Commit**

```bash
git add workflows/single_work.py tests/test_single_work_numerics.py
git commit -m "feat(single_work): _calc_hl_ratio_consistency helper

Compute (std, mad) of log10(intensity_ratio) across a list of
fragments. Non-positive ratios are dropped. Returns NaN std for
count==1 to match the single-element convention used by
extract_ion_pearson_features (Bug #21).

Co-authored-by: Copilot <223556219+Copilot@users.noreply.github.com>"
```

---

## Task 8: Wire precursor cycle offset features into `single_pair_work`

**Files:**
- Modify: `workflows/single_work.py` `single_pair_work` (~line 274)

Note: This task introduces the new feature keys into `single_pair_work`'s precursor block. No new test file is required here because Tasks 5-7 already test the underlying helpers; the integration is exercised by the existing end-to-end smoke tests in Task 11.

- [ ] **Step 1: Modify the precursor block in `single_pair_work`**

In `workflows/single_work.py`, find the precursor block (lines ~313-338):

```python
    features = {}
    if len(light_xic) == 0 or len(heavy_xic) == 0:
        features["precursor_pearson"] = 0
        features["precursor_apex_delta"] = 0.0
        features["precursor_apex_delta_signed"] = 0.0
        features["precursor_mz_avg_err"] = 0.0
        features["precursor_light_max_int"] = 0.0
        features["precursor_heavy_max_int"] = 0.0
        features["precursor_intensity_ratio"] = 0.0
        features["precursor_cosine"] = 0.0
        features["precursor_snr"] = 0.0
        features["precursor_peak_width_ratio"] = 0.0
        features["precursor_peak_symmetry"] = 0.0
    else:
        precursor_score = calc_xic_score(light_xic, heavy_xic)
        features["precursor_pearson"] = precursor_score["pearson"]
        features["precursor_apex_delta"] = precursor_score["apex_delta"]
        features["precursor_apex_delta_signed"] = precursor_score["apex_delta_signed"]
        features["precursor_mz_avg_err"] = precursor_score["mz_avg_err"]
        features["precursor_light_max_int"] = precursor_score["light_max_int"]
        features["precursor_heavy_max_int"] = precursor_score["heavy_max_int"]
        features["precursor_intensity_ratio"] = precursor_score["intensity_ratio"]
        features["precursor_cosine"] = precursor_score["cosine"]
        features["precursor_snr"] = precursor_score["snr"]
        features["precursor_peak_width_ratio"] = precursor_score["peak_width_ratio"]
        features["precursor_peak_symmetry"] = precursor_score["peak_symmetry"]
```

Replace with:

```python
    features = {}
    if len(light_xic) == 0 or len(heavy_xic) == 0:
        features["precursor_pearson"] = 0
        features["precursor_apex_delta"] = 0.0
        features["precursor_apex_delta_signed"] = 0.0
        features["precursor_mz_avg_err"] = 0.0
        features["precursor_light_max_int"] = 0.0
        features["precursor_heavy_max_int"] = 0.0
        features["precursor_intensity_ratio"] = 0.0
        features["precursor_cosine"] = 0.0
        features["precursor_snr"] = 0.0
        features["precursor_peak_width_ratio"] = 0.0
        features["precursor_peak_symmetry"] = 0.0
        features["precursor_light_apex_cycle_offset"] = 0
        features["precursor_light_apex_cycle_offset_signed"] = 0
        features["precursor_heavy_apex_cycle_offset"] = 0
        features["precursor_heavy_apex_cycle_offset_signed"] = 0
    else:
        precursor_score = calc_xic_score(
            light_xic, heavy_xic, center_rt=float(psm._rt))
        features["precursor_pearson"] = precursor_score["pearson"]
        features["precursor_apex_delta"] = precursor_score["apex_delta"]
        features["precursor_apex_delta_signed"] = precursor_score["apex_delta_signed"]
        features["precursor_mz_avg_err"] = precursor_score["mz_avg_err"]
        features["precursor_light_max_int"] = precursor_score["light_max_int"]
        features["precursor_heavy_max_int"] = precursor_score["heavy_max_int"]
        features["precursor_intensity_ratio"] = precursor_score["intensity_ratio"]
        features["precursor_cosine"] = precursor_score["cosine"]
        features["precursor_snr"] = precursor_score["snr"]
        features["precursor_peak_width_ratio"] = precursor_score["peak_width_ratio"]
        features["precursor_peak_symmetry"] = precursor_score["peak_symmetry"]
        features["precursor_light_apex_cycle_offset"] = (
            precursor_score["light_apex_cycle_offset"])
        features["precursor_light_apex_cycle_offset_signed"] = (
            precursor_score["light_apex_cycle_offset_signed"])
        features["precursor_heavy_apex_cycle_offset"] = (
            precursor_score["heavy_apex_cycle_offset"])
        features["precursor_heavy_apex_cycle_offset_signed"] = (
            precursor_score["heavy_apex_cycle_offset_signed"])
```

- [ ] **Step 2: Smoke-test that import succeeds**

```bash
conda run -n jianyan python -c "from workflows.single_work import single_pair_work; print('OK')"
```

Expected: `OK`.

- [ ] **Step 3: Run existing test suite to ensure nothing regressed**

```bash
conda run -n jianyan pytest tests/ -v -x
```

Expected: All tests pass.

- [ ] **Step 4: Commit**

```bash
git add workflows/single_work.py
git commit -m "feat(single_work): precursor apex_cycle_offset in single_pair_work

Wire the 4 new precursor_*_apex_cycle_offset[_signed] features through
single_pair_work. center_rt = psm._rt is passed to calc_xic_score so
the offset is measured relative to the PSM-reported RT (= XIC window
center).

Co-authored-by: Copilot <223556219+Copilot@users.noreply.github.com>"
```

---

## Task 9: Wire fragment-level cycle offset + H/L consistency into `single_pair_work`

**Files:**
- Modify: `workflows/single_work.py` `single_pair_work` (lines ~399-473 fragment loop + ~503-511 aggregation)

- [ ] **Step 1: Add fragment-level collection lists (before the loop)**

In `workflows/single_work.py`, find the lines (~399-403):

```python
    fragment_apex_deltas = []
    fragment_mz_errs = []
    fragment_intensities = []  # per-ion max intensity for weighted correlation
    fragment_cosines = []
    fragment_snrs = []
```

Replace with:

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

- [ ] **Step 2: Update the `calc_xic_score` call inside the fragment loop**

Find this line in the fragment loop (~line 453):

```python
        ion_score = calc_xic_score(light_ions_xic, heavy_ions_xic)
```

Replace with:

```python
        ion_score = calc_xic_score(
            light_ions_xic, heavy_ions_xic, center_rt=float(psm._rt))
```

- [ ] **Step 3: Collect per-fragment values right after the existing `pearsons_map[...].append(...)` lines**

In `workflows/single_work.py`, find this block (~lines 455-462):

```python
        pearsons_map[ions_type].append(ion_score["pearson"])
        pearsons_map["all"].append(ion_score["pearson"])
        fragment_apex_deltas.append(ion_score["apex_delta"])
        fragment_mz_errs.append(ion_score["mz_avg_err"])
        fragment_intensities.append(
            max(ion_score["light_max_int"], ion_score["heavy_max_int"]))
        fragment_cosines.append(ion_score["cosine"])
        fragment_snrs.append(ion_score["snr"])
```

Replace with:

```python
        pearsons_map[ions_type].append(ion_score["pearson"])
        pearsons_map["all"].append(ion_score["pearson"])
        fragment_apex_deltas.append(ion_score["apex_delta"])
        fragment_mz_errs.append(ion_score["mz_avg_err"])
        fragment_intensities.append(
            max(ion_score["light_max_int"], ion_score["heavy_max_int"]))
        fragment_cosines.append(ion_score["cosine"])
        fragment_snrs.append(ion_score["snr"])
        fragment_light_cycle_offsets.append(
            ion_score["light_apex_cycle_offset"])
        fragment_light_cycle_offsets_signed.append(
            ion_score["light_apex_cycle_offset_signed"])
        fragment_heavy_cycle_offsets.append(
            ion_score["heavy_apex_cycle_offset"])
        fragment_heavy_cycle_offsets_signed.append(
            ion_score["heavy_apex_cycle_offset_signed"])
        if ion_score["intensity_ratio"] > 0:
            fragment_hl_ratios[ions_type].append(
                float(ion_score["intensity_ratio"]))
            fragment_hl_ratios["all"].append(
                float(ion_score["intensity_ratio"]))
```

- [ ] **Step 4: Add aggregation after the existing fragment aggregations**

In `workflows/single_work.py`, find this block (~lines 503-511):

```python
    # 碎片级 apex_delta / mz_err / cosine / snr 汇总
    features.update(extract_ion_numeric_features(
        fragment_apex_deltas, "all_apex_delta"))
    features.update(extract_ion_numeric_features(
        fragment_mz_errs, "all_mz_err"))
    features.update(extract_ion_numeric_features(
        fragment_cosines, "all_cosine"))
    features.update(extract_ion_numeric_features(
        fragment_snrs, "all_snr"))
```

Insert the following block right after it (before the `# 序列级特征` comment):

```python
    # 碎片级 apex_cycle_offset 汇总（light/heavy × abs/signed × {mean,p50,std,max}）
    features.update(extract_ion_numeric_features(
        fragment_light_cycle_offsets, "all_light_apex_cycle_offset"))
    features.update(extract_ion_numeric_features(
        fragment_light_cycle_offsets_signed,
        "all_light_apex_cycle_offset_signed"))
    features.update(extract_ion_numeric_features(
        fragment_heavy_cycle_offsets, "all_heavy_apex_cycle_offset"))
    features.update(extract_ion_numeric_features(
        fragment_heavy_cycle_offsets_signed,
        "all_heavy_apex_cycle_offset_signed"))

    # H/L 强度比一致性（按 all/b/y 分组的 log10-ratio std/mad）
    for ion_type, ratios in fragment_hl_ratios.items():
        std_v, mad_v = _calc_hl_ratio_consistency(ratios)
        features[f"{ion_type}_log_hl_ratio_std"] = std_v
        features[f"{ion_type}_log_hl_ratio_mad"] = mad_v
```

- [ ] **Step 5: Smoke-test that import + a synthetic call still works**

```bash
conda run -n jianyan python -c "
from workflows.single_work import single_pair_work
print('import ok')
"
```

Expected: `import ok`.

- [ ] **Step 6: Run the full test suite**

```bash
conda run -n jianyan pytest tests/ -v -x
```

Expected: All tests pass.

- [ ] **Step 7: Commit**

```bash
git add workflows/single_work.py
git commit -m "feat(single_work): fragment cycle_offset + H/L consistency in single_pair_work

Collect per-fragment light/heavy apex_cycle_offset (abs and signed)
and intensity_ratio (grouped by b/y/all) in the fragment loop, then
aggregate after the loop:

- 16 new columns: all_{light,heavy}_apex_cycle_offset[_signed]_{mean,p50,std,max}
- 6 new columns: {all,b,y}_log_hl_ratio_{std,mad}

Co-authored-by: Copilot <223556219+Copilot@users.noreply.github.com>"
```

---

## Task 10: Mirror changes into `multi_batch_work`

**Files:**
- Modify: `workflows/single_work.py` `multi_batch_work` (lines ~24-271)

`multi_batch_work` is structurally identical to `single_pair_work` but operates on two PSMs / two DIAData (`psm1`/`dia_data1` for light, `psm2`/`dia_data2` for heavy). All changes from Tasks 8 & 9 must be mirrored, using `heavy_center_rt=float(psm2._rt)` where applicable.

- [ ] **Step 1: Update precursor block in `multi_batch_work`**

Find the precursor block (lines ~54-79):

```python
    features = {}
    if len(light_xic) == 0 or len(heavy_xic) == 0:
        features["precursor_pearson"] = 0
        features["precursor_apex_delta"] = 0.0
        features["precursor_apex_delta_signed"] = 0.0
        features["precursor_mz_avg_err"] = 0.0
        features["precursor_light_max_int"] = 0.0
        features["precursor_heavy_max_int"] = 0.0
        features["precursor_intensity_ratio"] = 0.0
        features["precursor_cosine"] = 0.0
        features["precursor_snr"] = 0.0
        features["precursor_peak_width_ratio"] = 0.0
        features["precursor_peak_symmetry"] = 0.0
    else:
        precursor_score = calc_xic_score(light_xic, heavy_xic)
        features["precursor_pearson"] = precursor_score["pearson"]
        features["precursor_apex_delta"] = precursor_score["apex_delta"]
        features["precursor_apex_delta_signed"] = precursor_score["apex_delta_signed"]
        features["precursor_mz_avg_err"] = precursor_score["mz_avg_err"]
        features["precursor_light_max_int"] = precursor_score["light_max_int"]
        features["precursor_heavy_max_int"] = precursor_score["heavy_max_int"]
        features["precursor_intensity_ratio"] = precursor_score["intensity_ratio"]
        features["precursor_cosine"] = precursor_score["cosine"]
        features["precursor_snr"] = precursor_score["snr"]
        features["precursor_peak_width_ratio"] = precursor_score["peak_width_ratio"]
        features["precursor_peak_symmetry"] = precursor_score["peak_symmetry"]
```

Replace with:

```python
    features = {}
    if len(light_xic) == 0 or len(heavy_xic) == 0:
        features["precursor_pearson"] = 0
        features["precursor_apex_delta"] = 0.0
        features["precursor_apex_delta_signed"] = 0.0
        features["precursor_mz_avg_err"] = 0.0
        features["precursor_light_max_int"] = 0.0
        features["precursor_heavy_max_int"] = 0.0
        features["precursor_intensity_ratio"] = 0.0
        features["precursor_cosine"] = 0.0
        features["precursor_snr"] = 0.0
        features["precursor_peak_width_ratio"] = 0.0
        features["precursor_peak_symmetry"] = 0.0
        features["precursor_light_apex_cycle_offset"] = 0
        features["precursor_light_apex_cycle_offset_signed"] = 0
        features["precursor_heavy_apex_cycle_offset"] = 0
        features["precursor_heavy_apex_cycle_offset_signed"] = 0
    else:
        precursor_score = calc_xic_score(
            light_xic, heavy_xic,
            center_rt=float(psm1._rt),
            heavy_center_rt=float(psm2._rt))
        features["precursor_pearson"] = precursor_score["pearson"]
        features["precursor_apex_delta"] = precursor_score["apex_delta"]
        features["precursor_apex_delta_signed"] = precursor_score["apex_delta_signed"]
        features["precursor_mz_avg_err"] = precursor_score["mz_avg_err"]
        features["precursor_light_max_int"] = precursor_score["light_max_int"]
        features["precursor_heavy_max_int"] = precursor_score["heavy_max_int"]
        features["precursor_intensity_ratio"] = precursor_score["intensity_ratio"]
        features["precursor_cosine"] = precursor_score["cosine"]
        features["precursor_snr"] = precursor_score["snr"]
        features["precursor_peak_width_ratio"] = precursor_score["peak_width_ratio"]
        features["precursor_peak_symmetry"] = precursor_score["peak_symmetry"]
        features["precursor_light_apex_cycle_offset"] = (
            precursor_score["light_apex_cycle_offset"])
        features["precursor_light_apex_cycle_offset_signed"] = (
            precursor_score["light_apex_cycle_offset_signed"])
        features["precursor_heavy_apex_cycle_offset"] = (
            precursor_score["heavy_apex_cycle_offset"])
        features["precursor_heavy_apex_cycle_offset_signed"] = (
            precursor_score["heavy_apex_cycle_offset_signed"])
```

- [ ] **Step 2: Add fragment collection lists**

Find (lines ~135-139):

```python
    fragment_apex_deltas = []
    fragment_mz_errs = []
    fragment_intensities = []  # per-ion max intensity for weighted correlation
    fragment_cosines = []
    fragment_snrs = []
```

Replace with:

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

- [ ] **Step 3: Update `calc_xic_score` call in fragment loop**

Find (~line 194):

```python
        ion_score = calc_xic_score(light_ions_xic, heavy_ions_xic)
```

Replace with:

```python
        ion_score = calc_xic_score(
            light_ions_xic, heavy_ions_xic,
            center_rt=float(psm1._rt),
            heavy_center_rt=float(psm2._rt))
```

- [ ] **Step 4: Extend the per-fragment collection block**

Find (~lines 196-203):

```python
        pearsons_map[ions_type].append(ion_score["pearson"])
        pearsons_map["all"].append(ion_score["pearson"])
        fragment_apex_deltas.append(ion_score["apex_delta"])
        fragment_mz_errs.append(ion_score["mz_avg_err"])
        fragment_intensities.append(
            max(ion_score["light_max_int"], ion_score["heavy_max_int"]))
        fragment_cosines.append(ion_score["cosine"])
        fragment_snrs.append(ion_score["snr"])
```

Replace with:

```python
        pearsons_map[ions_type].append(ion_score["pearson"])
        pearsons_map["all"].append(ion_score["pearson"])
        fragment_apex_deltas.append(ion_score["apex_delta"])
        fragment_mz_errs.append(ion_score["mz_avg_err"])
        fragment_intensities.append(
            max(ion_score["light_max_int"], ion_score["heavy_max_int"]))
        fragment_cosines.append(ion_score["cosine"])
        fragment_snrs.append(ion_score["snr"])
        fragment_light_cycle_offsets.append(
            ion_score["light_apex_cycle_offset"])
        fragment_light_cycle_offsets_signed.append(
            ion_score["light_apex_cycle_offset_signed"])
        fragment_heavy_cycle_offsets.append(
            ion_score["heavy_apex_cycle_offset"])
        fragment_heavy_cycle_offsets_signed.append(
            ion_score["heavy_apex_cycle_offset_signed"])
        if ion_score["intensity_ratio"] > 0:
            fragment_hl_ratios[ions_type].append(
                float(ion_score["intensity_ratio"]))
            fragment_hl_ratios["all"].append(
                float(ion_score["intensity_ratio"]))
```

- [ ] **Step 5: Add aggregation after the existing fragment aggregations**

Find (~lines 248-255):

```python
    features.update(extract_ion_numeric_features(
        fragment_apex_deltas, "all_apex_delta"))
    features.update(extract_ion_numeric_features(
        fragment_mz_errs, "all_mz_err"))
    features.update(extract_ion_numeric_features(
        fragment_cosines, "all_cosine"))
    features.update(extract_ion_numeric_features(
        fragment_snrs, "all_snr"))
```

Insert this **immediately after** (before the `# 序列级特征` comment):

```python
    # 碎片级 apex_cycle_offset 汇总（light/heavy × abs/signed × {mean,p50,std,max}）
    features.update(extract_ion_numeric_features(
        fragment_light_cycle_offsets, "all_light_apex_cycle_offset"))
    features.update(extract_ion_numeric_features(
        fragment_light_cycle_offsets_signed,
        "all_light_apex_cycle_offset_signed"))
    features.update(extract_ion_numeric_features(
        fragment_heavy_cycle_offsets, "all_heavy_apex_cycle_offset"))
    features.update(extract_ion_numeric_features(
        fragment_heavy_cycle_offsets_signed,
        "all_heavy_apex_cycle_offset_signed"))

    # H/L 强度比一致性（按 all/b/y 分组的 log10-ratio std/mad）
    for ion_type, ratios in fragment_hl_ratios.items():
        std_v, mad_v = _calc_hl_ratio_consistency(ratios)
        features[f"{ion_type}_log_hl_ratio_std"] = std_v
        features[f"{ion_type}_log_hl_ratio_mad"] = mad_v
```

- [ ] **Step 6: Smoke-test**

```bash
conda run -n jianyan python -c "
from workflows.single_work import multi_batch_work, single_pair_work
print('imports ok')
"
```

Expected: `imports ok`.

- [ ] **Step 7: Run full test suite**

```bash
conda run -n jianyan pytest tests/ -v -x
```

Expected: All tests pass.

- [ ] **Step 8: Commit**

```bash
git add workflows/single_work.py
git commit -m "feat(single_work): mirror cycle_offset + H/L consistency into multi_batch_work

multi_batch_work uses dia_data1/psm1 for light and dia_data2/psm2 for
heavy. center_rt=psm1._rt and heavy_center_rt=psm2._rt are passed to
calc_xic_score so light/heavy cycle offsets are measured in each
DIAData's own RT system.

Both single_pair_work and multi_batch_work now emit the full 26 new
columns symmetrically.

Co-authored-by: Copilot <223556219+Copilot@users.noreply.github.com>"
```

---

## Task 11: End-to-end smoke test + feature roster sanity check

**Files:**
- No code changes; run repo's existing tests and inspect output schema.

- [ ] **Step 1: Run the entire pytest suite**

```bash
conda run -n jianyan pytest tests/ -v
```

Expected: All tests pass. Note baseline pass count for comparison.

- [ ] **Step 2: Verify the 26 new feature keys are emitted by `single_pair_work`**

Run this one-liner check, which constructs a stub PSM and DIAData and asserts the feature roster contains all 26 new keys:

```bash
conda run -n jianyan python -c "
import numpy as np
import configparser
from unittest.mock import MagicMock

# Build a minimal PSM + DIAData stub
from spectrum.psm_info import PSMInfo
psm = PSMInfo(
    sequence='PEPTIDEK', charge=2, modify=[],
    rt=np.float32(50.0),
    precursor_mz=np.float32(450.0),
    raw_title='r1', protein_names='X_HUMAN')

dia = MagicMock()
# Force the 'len(light_xic)==0 or len(heavy_xic)==0' path so we can
# inspect just the default-zero feature roster without needing real data.
dt = [('rt', 'f8'), ('ppm_error', 'f8'),
      ('intensity', 'f8'), ('cycle_idx', 'i4')]
dia.xic_peaks_extreact.return_value = np.array([], dtype=dt)
dia.xic_ms2_peaks_extract.return_value = (np.array([], dtype=dt), 0.0)
dia.get_window_info.return_value = {'lower': 0.0, 'upper': 1.0,
                                     'width': 1.0, 'centering': 0.5}
dia.check_in_same_ms2.return_value = True
dia.check_in_raw.return_value = False  # skip fragment loop quickly

config = configparser.ConfigParser()
config['general'] = {'mass_tol_ppm': '10', 'xic_cycle_window': '3'}

from workflows.single_work import single_pair_work
feats = single_pair_work(psm, dia, config)

EXPECTED_NEW = [
    'all_log_hl_ratio_std', 'b_log_hl_ratio_std', 'y_log_hl_ratio_std',
    'all_log_hl_ratio_mad', 'b_log_hl_ratio_mad', 'y_log_hl_ratio_mad',
    'precursor_light_apex_cycle_offset',
    'precursor_light_apex_cycle_offset_signed',
    'precursor_heavy_apex_cycle_offset',
    'precursor_heavy_apex_cycle_offset_signed',
]
for stat in ['mean', 'p50', 'std', 'max']:
    for side in ['light', 'heavy']:
        EXPECTED_NEW.append(f'all_{side}_apex_cycle_offset_{stat}')
        EXPECTED_NEW.append(f'all_{side}_apex_cycle_offset_signed_{stat}')

print(f'Total new keys expected: {len(EXPECTED_NEW)}')
missing = [k for k in EXPECTED_NEW if k not in feats]
if missing:
    print(f'MISSING: {missing}')
    raise SystemExit(1)
print('All 26 new feature keys present.')
"
```

Expected output:
```
Total new keys expected: 26
All 26 new feature keys present.
```

- [ ] **Step 3: Diff feature roster between this branch and the previous commit**

```bash
git --no-pager log --oneline -1
```

If you want to count new columns in a real CSV output, run a tiny extraction on a sample mzML in `workspace/`. This is only needed if a real CSV smoke test is available; otherwise the Step 2 stub is sufficient. Skip this step if no sample data is at hand.

- [ ] **Step 4: Update PLAN.md with the new features (optional but recommended)**

If `PLAN.md` has a "已完成" / "下一轮" section, append a note for these 26 columns. Open `PLAN.md`, scroll to the "已完成的代码改进" or similar section, and append (only if maintainer wants the doc kept in sync):

```markdown
### 第三轮特征工程（2026-05-26）

按 `docs/specs/2026-05-26-hl-ratio-consistency-and-apex-cycle-offset-design.md` 实施：

| 维度 | 新增列数 |
|------|---------|
| H/L 强度比一致性（log10-ratio std/mad × {all,b,y}） | 6 |
| 前体 apex_cycle_offset（light/heavy × abs/signed） | 4 |
| 碎片 apex_cycle_offset 汇总（light/heavy × abs/signed × {mean,p50,std,max}） | 16 |
| **合计** | **26** |

实现要点：`spectrum/dia_data.py` 的 XIC dtype 增加 `cycle_idx` 列（动态计算，不影响 npz 缓存）。`workflows/single_work.py` 的 `calc_xic_score` 加可选 `center_rt`/`heavy_center_rt`。`extract_ion_numeric_features` 顺带加 `_max` 字段，受益于 `all_apex_delta` / `all_mz_err` / `all_cosine` / `all_snr` 等已有汇总。
```

- [ ] **Step 5: Final commit (if PLAN.md was updated)**

```bash
git add PLAN.md
git commit -m "docs(plan): add H/L consistency + apex_cycle_offset to roadmap

Co-authored-by: Copilot <223556219+Copilot@users.noreply.github.com>"
```

If PLAN.md was not updated, skip this step.

- [ ] **Step 6: Confirm clean working tree**

```bash
git status
```

Expected: `nothing to commit, working tree clean` (or only the `cross_domain_analysis/` untracked dir from baseline — that is pre-existing and unrelated).

---

## Summary

After all 11 tasks:

- 1 spec doc committed (already done in brainstorming phase).
- 11 implementation commits.
- 2 source files modified: `spectrum/dia_data.py` (+1 method, +XIC dtype), `workflows/single_work.py` (+2 helpers, calc_xic_score signature, extract_ion_numeric_features, both work functions).
- 2 test files modified with ~17 new tests.
- 26 new CSV columns emitted by both `single_pair_work` and `multi_batch_work`.
- Backward compatible: existing tests pass; existing callers of `calc_xic_score(light, heavy)` see new fields default to 0; existing consumers ignoring `xic["cycle_idx"]` are unaffected.
