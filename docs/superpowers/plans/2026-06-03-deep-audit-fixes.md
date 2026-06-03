# Deep Audit Fixes Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Resolve 4 Critical + 15 Important findings from the 2026-06-03 deep audit (units & semantics / pipeline coherence / silent failures) in 19 surgical tasks, then regenerate features.csv from clean state to produce schema-consistent, semantically correct training data.

**Architecture:** 3 phases of independent task fixes (P0 Critical, P1 Active Important, P2 Dormant Important). Each task is independently testable and committable. After all 19 tasks land, the user manually runs `make clean-all && make all` to regenerate features.csv (slow, off-plan).

**Tech Stack:** Python 3, NumPy, pandas, scikit-learn, scipy, pyteomics, lightgbm, pytest, configparser, GNU Make.

---

## Background

See `docs/specs/2026-06-03-deep-audit-fixes-design.md` for full audit findings and design decisions. Key facts:

- All 3 baseline features.csv files on disk are stale (pre R3/R4) and schema-inconsistent (2da has 66 cols, 5da/normal have 77).
- 4 Critical bugs (3 Active + 1 Dormant): empty-XIC indistinguishable from real data, all-zero XIC produces "perfect coelution", cache ignores centroid params, multi_batch_work uses wrong mass.
- User decisions: marker column for empty-XIC (not NaN); version bump for cache; rename only for log_hl_ratio; single 19-task plan in 3 phases.

## Test environments

- `silac_ml` conda env has sklearn 1.8.0 + lightgbm 4.6.0 + pyteomics + scipy + pandas — use for ANY test importing those.
- `jianyan` conda env has pytest + scipy + pandas + numpy but NO sklearn/lightgbm — use for fast tests that don't touch sklearn.
- Default: run all new tests under `silac_ml`. The full regression suite (`pytest tests/ -q`) has 278 passing in `silac_ml`.

## File Structure

### Files modified

| File | Phase | Tasks |
|---|---|---|
| `workflows/single_work.py` | P0, P1, P2 | P0-1, P0-2, P0-4, P1-1, P1-2, P2-1, P2-3, P2-5 |
| `spectrum/dia_data.py` | P0, P1, P2 | P0-3, P1-6, P2-2 |
| `spectrum/spectrum_utils.py` | P1 | P1-7 |
| `workflows/flow_utils.py` | P1, P2 | P1-3, P2-4 |
| `workflows/pair_flow.py` | P1, P2 | P1-6, P2-4 |
| `spectrum/psm_info.py` | P2 | P2-4 |
| `constant/keys.py` | P2 | P2-4 |
| `tools/spec_trainer/src/main.py` | P2 | P2-6, P2-8 |
| `tools/spec_trainer/src/feature_cols.py` | P1, P2 | P1-5, P2-7 |
| `tools/spec_trainer/config/exp1.yaml` | P2 | P2-6 |
| `tools/spec_trainer/config/exp2.yaml` | P1, P2 | P1-4, P2-6 |
| `docs/specs/2026-05-26-hl-ratio-consistency-and-apex-cycle-offset-design.md` | P2 | P2-1 (doc alignment) |

### Files created

- `tests/test_deep_audit_p0.py` — Phase 0 behavioral tests (P0-1..P0-4)
- `tests/test_deep_audit_p1.py` — Phase 1 behavioral tests (P1-1..P1-7)
- `tests/test_deep_audit_p2.py` — Phase 2 unit/source tests (P2-1..P2-8)

---

## Phase 0 — Critical bugs (4 tasks)

### Task P0-1: Silent-C1 — `precursor_xic_empty` marker columns

**Why:** When XIC retrieval fails, `single_pair_work` (lines 380-417) and `multi_batch_work` (lines 55-74) write all 20 precursor features as `0/0.0` with NO marker. LightGBM cannot distinguish "extraction failed" from "computed 0". Adds two boolean columns: `precursor_xic_empty: 0/1` and `fragment_xic_empty_count: int`.

**Files:**
- Modify: `workflows/single_work.py` (new helper + both `single_pair_work` empty/computed branches + both `multi_batch_work` empty/computed branches + fragment loops in both functions)
- Test: `tests/test_deep_audit_p0.py` (new file)

**Coordination with P0-2:** The same `_is_empty_xic_pair` helper introduced here is reused by P0-2 to short-circuit `calc_xic_score`. Implement the helper first; P0-2 just adds the early-return call site.

- [ ] **Step 1: Create test file with empty-XIC marker tests**

Create `tests/test_deep_audit_p0.py` with:

```python
"""Phase 0 (Critical) behavioral tests for deep audit fixes.

See docs/specs/2026-06-03-deep-audit-fixes-design.md for design rationale.
"""
import configparser
import numpy as np
import pytest


def _empty_xic():
    """Return an empty XIC structured array matching dia_data dtype."""
    dtype = [("rt", "f8"), ("ppm_error", "f8"),
             ("intensity", "f8"), ("cycle_idx", "i4")]
    return np.array([], dtype=dtype)


def _real_xic(rts, intensities, cycle_idxs=None):
    """Build a non-empty XIC for tests."""
    n = len(rts)
    if cycle_idxs is None:
        cycle_idxs = list(range(n))
    dtype = [("rt", "f8"), ("ppm_error", "f8"),
             ("intensity", "f8"), ("cycle_idx", "i4")]
    arr = np.zeros(n, dtype=dtype)
    arr["rt"] = rts
    arr["ppm_error"] = 0.0
    arr["intensity"] = intensities
    arr["cycle_idx"] = cycle_idxs
    return arr


class _FakePSM:
    """Minimal PSM stub for triggering single_pair_work / multi_batch_work."""
    def __init__(self, mz=500.0, rt=10.0, seq="AAAAK", charge=2):
        self._precursor_mz = mz
        self._rt = rt
        self._sequence = seq
        self._charge = charge
        self._raw_title = "fake.mzML"
        self._protein_names = "HUMAN"
        self._label_type = "positive"
        self._modify = []


class _FakeDIA:
    """Minimal DIA stub. Returns empty XIC for any precursor_mz query
    when force_empty=True; otherwise returns a small synthetic XIC."""
    def __init__(self, force_empty=False, xic_intensity=None):
        self._force_empty = force_empty
        self._xic_intensity = xic_intensity  # if set, used by xic_*_extract
        self._min_mz_value = 0.0
        self._max_mz_value = 10000.0

    def xic_peaks_extreact(self, rt, window, mz, mass_tol_ppm):
        if self._force_empty:
            return _empty_xic()
        intensity = self._xic_intensity if self._xic_intensity is not None \
            else [100.0, 200.0, 500.0, 300.0, 150.0]
        return _real_xic([9.5, 9.7, 10.0, 10.3, 10.5], intensity)

    def xic_ms2_peaks_extract(self, rt, window, precursor_mz, ions_mass,
                              mass_tol_ppm):
        if self._force_empty:
            return _empty_xic(), 0.0
        intensity = self._xic_intensity if self._xic_intensity is not None \
            else [10.0, 20.0, 50.0, 30.0, 15.0]
        return _real_xic([9.5, 9.7, 10.0, 10.3, 10.5], intensity), 100.0

    def check_in_raw(self, mz):
        return True

    def check_in_same_ms2(self, p1, p2):
        return False

    def get_heavy_info(self, psm):
        return psm._precursor_mz + 4.0, []  # empty fragment_ions

    def get_window_info(self, mz):
        return {"width": 2.0, "lower": mz - 1.0, "upper": mz + 1.0,
                "split_window": False}


def _minimal_config():
    cfg = configparser.ConfigParser()
    cfg.read_dict({
        "general": {
            "mass_tol_ppm": "20",
            "xic_cycle_window": "5",
        },
    })
    return cfg


def test_single_pair_work_marks_precursor_xic_empty_when_empty():
    """single_pair_work empty-XIC branch must set precursor_xic_empty=1."""
    from workflows.single_work import single_pair_work
    psm = _FakePSM()
    dia = _FakeDIA(force_empty=True)
    features = single_pair_work(psm, dia, _minimal_config())
    assert features["precursor_xic_empty"] == 1, (
        "P0-1: empty-XIC must set marker=1 in single_pair_work")


def test_multi_batch_work_marks_precursor_xic_empty_when_empty():
    """multi_batch_work empty-XIC branch must set precursor_xic_empty=1."""
    from workflows.single_work import multi_batch_work
    psm = _FakePSM()
    dia = _FakeDIA(force_empty=True)
    features = multi_batch_work(psm, dia, psm, dia, _minimal_config())
    assert features["precursor_xic_empty"] == 1, (
        "P0-1: empty-XIC must set marker=1 in multi_batch_work")


def test_single_pair_work_marks_precursor_xic_empty_zero_when_valid():
    """Non-empty valid XIC must set precursor_xic_empty=0 (single_pair)."""
    from workflows.single_work import single_pair_work
    psm = _FakePSM()
    dia = _FakeDIA(force_empty=False)
    features = single_pair_work(psm, dia, _minimal_config())
    assert features["precursor_xic_empty"] == 0, (
        "P0-1: valid XIC must set marker=0 in single_pair_work")


def test_multi_batch_work_marks_precursor_xic_empty_zero_when_valid():
    """Non-empty valid XIC must set precursor_xic_empty=0 (multi_batch)."""
    from workflows.single_work import multi_batch_work
    psm = _FakePSM()
    dia = _FakeDIA(force_empty=False)
    features = multi_batch_work(psm, dia, psm, dia, _minimal_config())
    assert features["precursor_xic_empty"] == 0, (
        "P0-1: valid XIC must set marker=0 in multi_batch_work")


def test_fragment_xic_empty_count_present_in_both_paths():
    """Both code paths must emit fragment_xic_empty_count column."""
    from workflows.single_work import single_pair_work, multi_batch_work
    psm = _FakePSM()
    dia_empty = _FakeDIA(force_empty=True)
    f1 = single_pair_work(psm, dia_empty, _minimal_config())
    f2 = multi_batch_work(psm, dia_empty, psm, dia_empty, _minimal_config())
    assert "fragment_xic_empty_count" in f1, (
        "P0-1: single_pair_work missing fragment_xic_empty_count")
    assert "fragment_xic_empty_count" in f2, (
        "P0-1: multi_batch_work missing fragment_xic_empty_count")
```

- [ ] **Step 2: Run tests to verify they fail**

Run: `conda run -n silac_ml pytest tests/test_deep_audit_p0.py -v -k precursor_xic_empty -x`

Expected: 4 FAILs (KeyError on `features["precursor_xic_empty"]`).

If a test errors at `single_pair_work(...)` construction (e.g., AttributeError on PSM), patch the `_FakePSM` / `_FakeDIA` to match the actual attribute access — debug by reading `workflows/single_work.py:348-528` first to see what attributes are touched.

- [ ] **Step 3: Add `_is_empty_xic_pair` helper to `workflows/single_work.py`**

Add this helper near the top of the file (after existing imports, before `multi_batch_work`):

```python
def _is_empty_xic_pair(light_xic: np.ndarray, heavy_xic: np.ndarray) -> bool:
    """Return True if either XIC is empty OR has all-zero intensity.

    Used by both single_pair_work / multi_batch_work to set the
    `precursor_xic_empty` marker, AND by calc_xic_score to short-circuit
    to default features. Consistent definition prevents the marker from
    diverging from the data path (see P0-1 and P0-2 in
    docs/specs/2026-06-03-deep-audit-fixes-design.md).
    """
    if len(light_xic) == 0 or len(heavy_xic) == 0:
        return True
    if not np.any(light_xic["intensity"] > 0):
        return True
    if not np.any(heavy_xic["intensity"] > 0):
        return True
    return False
```

- [ ] **Step 4: Patch `multi_batch_work` empty-XIC branch (lines 55-74)**

In `workflows/single_work.py`, find the empty-XIC branch in `multi_batch_work` (the block starting with `if len(light_xic) == 0 or len(heavy_xic) == 0:` around line 55-74).

Replace the condition with `if _is_empty_xic_pair(light_xic, heavy_xic):` and add the marker assignment AT THE END of the branch:

```python
        features["precursor_xic_empty"] = 1
```

Mirror for the `else:` (computed) branch — add at the end:

```python
        features["precursor_xic_empty"] = 0
```

- [ ] **Step 5: Patch `single_pair_work` empty-XIC branch (lines 380-435)**

Same pattern in `single_pair_work`. Find the `if len(light_xic) == 0 or len(heavy_xic) == 0:` block (around line 380-407). Replace condition with `_is_empty_xic_pair`. Add `features["precursor_xic_empty"] = 1` to empty branch and `features["precursor_xic_empty"] = 0` to computed branch.

- [ ] **Step 6: Add `fragment_xic_empty_count` to fragment loops in both functions**

In `multi_batch_work` fragment loop (around line 183-280):
- Initialize `fragment_xic_empty_count = 0` before the loop
- In the empty-XIC fragment branch (where `len(light_ions_xic) == 0 or len(heavy_ions_xic) == 0`), add `fragment_xic_empty_count += 1` after the existing appends
- After the loop ends, before the per-ion-type feature extraction block, add: `features["fragment_xic_empty_count"] = fragment_xic_empty_count`

Same pattern in `single_pair_work` fragment loop (around line 523-598). Also count when the `continue`s at lines 525-526 (heavy_in_raw==False) and 529-530 (same_ms2 + same mass) fire — these are "fragment skipped, no XIC computed" cases — count them in `fragment_xic_empty_count` too. Add a comment that the count is the union of {empty XIC, no heavy_in_raw, identical mass shift}.

- [ ] **Step 7: Run tests, verify all 5 PASS**

Run: `conda run -n silac_ml pytest tests/test_deep_audit_p0.py -v -k "precursor_xic_empty or fragment_xic_empty"`

Expected: 5 PASSed.

- [ ] **Step 8: Full regression check**

Run: `conda run -n silac_ml pytest tests/ -q 2>&1 | tail -5`

Expected: ≥278 passed (baseline) + new tests; no NEW failures.

- [ ] **Step 9: Commit**

```bash
git add workflows/single_work.py tests/test_deep_audit_p0.py
git commit -m "fix(single_work): precursor_xic_empty marker columns (P0-1, Silent-C1)

Audit finding Silent-C1 (2026-06-03 deep audit): empty-XIC branch in
single_pair_work + multi_batch_work wrote 20 features as 0/0.0 with no
marker. LightGBM could not distinguish 'extraction failed' from
'computed 0' — model learns from garbage rows.

Add _is_empty_xic_pair helper (also used by P0-2 calc_xic_score
short-circuit) returning True when either XIC is empty OR has all-zero
intensity. Both single_pair_work and multi_batch_work now emit:
- precursor_xic_empty: 0/1 (1 = imputed zeros)
- fragment_xic_empty_count: int (union of empty-XIC, no heavy_in_raw,
  and same-mass skip conditions)

5 behavioral tests cover both code paths × {empty, valid}."
```

---

### Task P0-2: Silent-C2 — `calc_xic_score` all-zero non-empty guard

**Why:** `calc_xic_score` (lines 1107-1146) guards `len==0` but not all-zero non-empty. `np.argmax([0,0,0,0])` returns `0` → `apex_delta=0` looks like perfect coelution. Combined with P0-1's marker logic, the all-zero case must also be marked `precursor_xic_empty=1`.

**Files:**
- Modify: `workflows/single_work.py:calc_xic_score`
- Test: append to `tests/test_deep_audit_p0.py`

**Dependency:** Must run AFTER P0-1 (uses the `_is_empty_xic_pair` helper P0-1 introduced).

- [ ] **Step 1: Append failing test**

Append to `tests/test_deep_audit_p0.py`:

```python
def test_calc_xic_score_short_circuits_on_all_zero_intensity():
    """All-zero non-empty XIC must return _default_xic_score (P0-2, Silent-C2)."""
    from workflows.single_work import calc_xic_score, _default_xic_score
    light = _real_xic([9.5, 9.7, 10.0, 10.3, 10.5], [0, 0, 0, 0, 0])
    heavy = _real_xic([9.5, 9.7, 10.0, 10.3, 10.5], [0, 0, 0, 0, 0])
    result = calc_xic_score(light, heavy)
    default = _default_xic_score()
    # All keys present in default must equal default values in result.
    for k, v in default.items():
        assert result[k] == v, (
            f"P0-2: all-zero XIC must produce default for {k}: got {result[k]}, expected {v}")
    # In particular, apex_delta must NOT be 0 from argmax-on-zeros
    # (which is what _default_xic_score returns; that is now the correct
    # value because the guard fires before the argmax). The point of
    # this test is the WHOLE dict is the default, not a mix of
    # default + spurious-computed-from-zeros values.


def test_calc_xic_score_unchanged_on_valid_input():
    """Valid non-empty XIC should still produce computed (non-default) features."""
    from workflows.single_work import calc_xic_score
    light = _real_xic([9.5, 9.7, 10.0, 10.3, 10.5], [10, 20, 100, 30, 15])
    heavy = _real_xic([9.5, 9.7, 10.0, 10.3, 10.5], [5, 10, 50, 15, 8])
    result = calc_xic_score(light, heavy)
    # apex_delta should be 0 (both peaks at rt=10.0) - this IS the real
    # computed value, not a guard-default. Sanity check pearson > 0.
    assert result["pearson"] > 0.9, (
        f"P0-2: valid XIC should compute real pearson, got {result['pearson']}")
    assert result["light_max_int"] == 100.0
    assert result["heavy_max_int"] == 50.0
```

- [ ] **Step 2: Run tests, verify they fail**

Run: `conda run -n silac_ml pytest tests/test_deep_audit_p0.py::test_calc_xic_score_short_circuits_on_all_zero_intensity -v`

Expected: FAIL — the all-zero test fails because current code computes spurious values like `intensity_ratio = 0/0` → `0.0` (matches default) but `apex_delta_signed`, `pearson` (probably NaN-coerced) may differ.

- [ ] **Step 3: Add the guard in `calc_xic_score`**

In `workflows/single_work.py:calc_xic_score`, immediately after the defensive `np.argsort` sort lines (around line 1110, before line 1112 `# 计算重标平均误差`), add:

```python
    # Short-circuit if either XIC is empty or all-zero intensity.
    # Without this, argmax-on-zeros silently returns index 0 and produces
    # apex_delta=0 — indistinguishable from perfect coelution
    # (P0-2, Silent-C2 in 2026-06-03 deep audit).
    if _is_empty_xic_pair(light_xic, heavy_xic):
        return _default_xic_score()
```

- [ ] **Step 4: Run tests, verify both PASS**

Run: `conda run -n silac_ml pytest tests/test_deep_audit_p0.py::test_calc_xic_score_short_circuits_on_all_zero_intensity tests/test_deep_audit_p0.py::test_calc_xic_score_unchanged_on_valid_input -v`

Expected: 2 PASSed.

- [ ] **Step 5: Verify P0-1 tests also still PASS (interaction check)**

Run: `conda run -n silac_ml pytest tests/test_deep_audit_p0.py -v`

Expected: 7 PASSed (5 from P0-1 + 2 from P0-2).

- [ ] **Step 6: Update P0-1 marker behavior to also flag all-zero case**

After P0-2's guard lands, the `single_pair_work` / `multi_batch_work` empty-branch entry condition was `if len(light_xic)==0 or len(heavy_xic)==0`. After replacing with `_is_empty_xic_pair`, all-zero non-empty XICs ALSO route to the empty branch → marker correctly set to 1. Confirm by adding one more test:

```python
def test_single_pair_work_marks_precursor_xic_empty_on_all_zero_xic():
    """All-zero non-empty XIC should also trigger precursor_xic_empty=1 (P0-1+P0-2 interaction)."""
    from workflows.single_work import single_pair_work
    psm = _FakePSM()
    dia = _FakeDIA(force_empty=False, xic_intensity=[0, 0, 0, 0, 0])
    features = single_pair_work(psm, dia, _minimal_config())
    assert features["precursor_xic_empty"] == 1, (
        "All-zero XIC should be treated as empty for marker purposes")
```

Run: `conda run -n silac_ml pytest tests/test_deep_audit_p0.py::test_single_pair_work_marks_precursor_xic_empty_on_all_zero_xic -v`

Expected: PASS.

- [ ] **Step 7: Full regression**

Run: `conda run -n silac_ml pytest tests/ -q 2>&1 | tail -5`

Expected: no NEW failures.

- [ ] **Step 8: Commit**

```bash
git add workflows/single_work.py tests/test_deep_audit_p0.py
git commit -m "fix(single_work): calc_xic_score short-circuit on all-zero XIC (P0-2, Silent-C2)

Audit finding Silent-C2 (2026-06-03 deep audit): calc_xic_score guarded
len(xic)==0 but not all-zero intensity. np.argmax on zeros returns
index 0 → apex_delta=0 indistinguishable from perfect coelution. Common
when DIA scans exist in window but no fragment peak matches ppm tol.

Use _is_empty_xic_pair helper (from P0-1) to short-circuit to
_default_xic_score(). Synergy with P0-1: all-zero XIC now also triggers
precursor_xic_empty=1 marker, so the model sees a consistent signal."
```

---

### Task P0-3: Silent-C3 — Cache `_format_version=3` with centroid params

**Why:** `.dia.npz` cache stores `_format_version=2` but doesn't include `_centroid_enabled` / `_centroid_rel_threshold`. `data_to_npz` only checks file existence. Changing centroid params has no effect — stale cache silently used.

**Files:**
- Modify: `spectrum/dia_data.py` (save_to_file, _check_format_version, _load_attrs)
- Test: append to `tests/test_deep_audit_p0.py`

- [ ] **Step 1: Write failing test for cache rejection on param mismatch**

Append to `tests/test_deep_audit_p0.py`:

```python
def test_cache_load_rejects_mismatched_centroid_params(tmp_path):
    """Cache saved with one set of centroid params must be rejected when
    loaded by a DIAData configured with different params (P0-3, Silent-C3)."""
    from spectrum.dia_data import DIAData
    src = DIAData()
    src._centroid_enabled = True
    src._centroid_rel_threshold = 1e-3
    # Populate minimal required attributes for save to succeed.
    src.has_mobility = False
    src.has_ms1 = True
    src._max_mz_value = 1000.0
    src._min_mz_value = 100.0
    src.ms1_indexs = np.array([0], dtype=np.int64)
    src.ms1_indexs_rt = np.array([0.0])
    src.ms2_indexs = np.array([1], dtype=np.int64)
    src.ms2_indexs_rt = np.array([0.1])
    src.precursor_scan_ids = np.array([0], dtype=np.int64)
    src._mz_values = np.array([500.0])
    src.rt_values = np.array([0.0, 0.1])
    src._intensity_values = np.array([100.0])
    src.mobility_values = np.array([])
    src._cycle_left_precursor = np.array([400.0])
    src._quad_max_mz_value = np.array([600.0])
    src._quad_min_mz_value = np.array([400.0])
    src._scan_id_to_index = np.array([0, 1], dtype=np.int64)
    src._peak_start_idx_list = np.array([0], dtype=np.int64)
    src._peak_stop_idx_list = np.array([1], dtype=np.int64)
    src._precursor_lower_mz = np.array([400.0])
    src._precursor_upper_mz = np.array([600.0])
    cache_path = str(tmp_path / "test_cache.npz")
    src.save_to_file(cache_path)

    # Load with different centroid_rel_threshold — must raise.
    with pytest.raises(ValueError, match="centroid"):
        DIAData.load_from_file(cache_path, expected_centroid_enabled=True,
                                expected_centroid_rel_threshold=1e-2)


def test_cache_load_accepts_matching_centroid_params(tmp_path):
    """Cache with matching params loads successfully (P0-3)."""
    from spectrum.dia_data import DIAData
    src = DIAData()
    src._centroid_enabled = True
    src._centroid_rel_threshold = 1e-3
    # Minimal required state (same as previous test, factor out helper)
    src.has_mobility = False
    src.has_ms1 = True
    src._max_mz_value = 1000.0
    src._min_mz_value = 100.0
    src.ms1_indexs = np.array([0], dtype=np.int64)
    src.ms1_indexs_rt = np.array([0.0])
    src.ms2_indexs = np.array([1], dtype=np.int64)
    src.ms2_indexs_rt = np.array([0.1])
    src.precursor_scan_ids = np.array([0], dtype=np.int64)
    src._mz_values = np.array([500.0])
    src.rt_values = np.array([0.0, 0.1])
    src._intensity_values = np.array([100.0])
    src.mobility_values = np.array([])
    src._cycle_left_precursor = np.array([400.0])
    src._quad_max_mz_value = np.array([600.0])
    src._quad_min_mz_value = np.array([400.0])
    src._scan_id_to_index = np.array([0, 1], dtype=np.int64)
    src._peak_start_idx_list = np.array([0], dtype=np.int64)
    src._peak_stop_idx_list = np.array([1], dtype=np.int64)
    src._precursor_lower_mz = np.array([400.0])
    src._precursor_upper_mz = np.array([600.0])
    cache_path = str(tmp_path / "test_cache2.npz")
    src.save_to_file(cache_path)

    loaded = DIAData.load_from_file(cache_path,
                                     expected_centroid_enabled=True,
                                     expected_centroid_rel_threshold=1e-3)
    assert loaded._centroid_enabled is True
    assert loaded._centroid_rel_threshold == 1e-3
```

- [ ] **Step 2: Run tests, verify they fail**

Run: `conda run -n silac_ml pytest tests/test_deep_audit_p0.py -v -k cache_load`

Expected: 2 FAILs (`load_from_file` doesn't accept `expected_centroid_*` kwargs yet).

- [ ] **Step 3: Patch `spectrum/dia_data.py:save_to_file` to bump version + include centroid params**

In `spectrum/dia_data.py`, find the `data = {...}` dict inside `save_to_file` (around lines 166-196). Change `'_format_version': np.int32(2)` to `np.int32(3)` and add two keys:

```python
            '_format_version': np.int32(3),
            ...
            '_centroid_enabled': np.bool_(self._centroid_enabled),
            '_centroid_rel_threshold': np.float64(self._centroid_rel_threshold),
```

- [ ] **Step 4: Patch `load_from_file` to accept + check centroid params**

Change the signature of `load_from_file`:

```python
    @classmethod
    def load_from_file(cls, filepath: str, use_mmap: bool = True,
                       expected_centroid_enabled: bool | None = None,
                       expected_centroid_rel_threshold: float | None = None):
        """从 .npz 文件加载 DIAData，支持内存映射（只读）

        Args:
            filepath: npz cache path.
            use_mmap: zero-copy mmap mode.
            expected_centroid_enabled: if provided, reject cache if mismatched
                (P0-3, Silent-C3 in 2026-06-03 deep audit).
            expected_centroid_rel_threshold: if provided, reject cache if
                |delta| > 1e-12.

        Raises:
            ValueError: if _format_version != 3 OR centroid params mismatch
                expected values.
        """
        obj = cls()

        if use_mmap:
            with np.load(filepath, mmap_mode='r') as data:
                cls._check_format_version(filepath, data,
                                          expected_centroid_enabled,
                                          expected_centroid_rel_threshold)
                _load_attrs(obj, data)
        else:
            data = np.load(filepath)
            cls._check_format_version(filepath, data,
                                      expected_centroid_enabled,
                                      expected_centroid_rel_threshold)
            _load_attrs(obj, data)

        return obj
```

- [ ] **Step 5: Update `_check_format_version` to validate centroid params**

Replace the existing `_check_format_version` (lines 227-241):

```python
    @staticmethod
    def _check_format_version(filepath: str, data,
                              expected_centroid_enabled: bool | None = None,
                              expected_centroid_rel_threshold: float | None = None) -> None:
        """Reject npz files without `_format_version=3` or mismatched centroid params.

        Bumped from 2 -> 3 in P0-3 (Silent-C3) to embed centroid params
        in the cache. Caller passes the currently-configured centroid
        params; mismatch raises (forcing rebuild).
        """
        if '_format_version' not in data:
            raise ValueError(
                f"npz 缓存 {filepath} 没有 _format_version 字段——这是 "
                f"旧版本（profile peaks）生成的缓存。请删除该文件后重新"
                f"运行以生成 centroided 缓存。"
            )
        version = int(data['_format_version'])
        if version != 3:
            raise ValueError(
                f"npz 缓存 {filepath} 的 _format_version={version}，"
                f"当前代码只支持 version=3。请删除该文件后重新运行。"
            )
        # Centroid param validation (P0-3, Silent-C3).
        if expected_centroid_enabled is not None:
            stored_enabled = bool(data['_centroid_enabled']) \
                if '_centroid_enabled' in data else None
            if stored_enabled != expected_centroid_enabled:
                raise ValueError(
                    f"npz 缓存 {filepath} 的 _centroid_enabled={stored_enabled}, "
                    f"配置要求 {expected_centroid_enabled}。请删除该文件后重新运行。"
                )
        if expected_centroid_rel_threshold is not None:
            stored_threshold = float(data['_centroid_rel_threshold']) \
                if '_centroid_rel_threshold' in data else None
            if (stored_threshold is None
                    or abs(stored_threshold - expected_centroid_rel_threshold) > 1e-12):
                raise ValueError(
                    f"npz 缓存 {filepath} 的 _centroid_rel_threshold={stored_threshold}, "
                    f"配置要求 {expected_centroid_rel_threshold}。请删除该文件后重新运行。"
                )
```

- [ ] **Step 6: Update `_load_attrs` to load centroid params from npz**

In `_load_attrs` (around lines 54-78), add at the end (before the function returns implicitly):

```python
    # Centroid params (P0-3, added in _format_version=3).
    if '_centroid_enabled' in data:
        obj._centroid_enabled = bool(data['_centroid_enabled'])
    if '_centroid_rel_threshold' in data:
        obj._centroid_rel_threshold = float(data['_centroid_rel_threshold'])
```

- [ ] **Step 7: Find all callers of `load_from_file` and pass centroid params**

Run: `grep -rn "DIAData.load_from_file\|load_from_file(" --include="*.py"`

Update each caller to pass the current centroid params from the DataManager or config. Typical caller `workflows/flow_utils.py:DIAData.load_from_file(shared_file, use_mmap=True)` — but the worker doesn't have the config easily. Two-step plan:

  (a) In `manager/data_manager.py` (or wherever DIAData is constructed pre-save), pass the params through.
  (b) For mmap workers (in `flow_utils.py`, `process_batch_single` etc.), keep the call as `load_from_file(shared_file, use_mmap=True)` WITHOUT the `expected_*` args. The `expected_*` validation happens once at the pre-save check; workers trust the cache.

Concretely:
- `manager/data_manager.py` (or main.py path) — pass `expected_centroid_enabled` and `expected_centroid_rel_threshold` from config to the `load_from_file` call that decides whether to use the cache vs rebuild.
- Workers in `flow_utils.py` — leave unchanged.

If you can't easily locate the cache-decision call site, use this rule: any `load_from_file(...)` call inside `workflows/flow_utils.py:data_to_npz` or `manager/data_manager.py:load_or_build_dia_data` needs the kwargs. Inspect both files; update if present.

- [ ] **Step 8: Run tests, verify they pass**

Run: `conda run -n silac_ml pytest tests/test_deep_audit_p0.py -v -k cache_load`

Expected: 2 PASSed.

- [ ] **Step 9: Verify older version=2 caches are still rejected**

Run a quick interactive check:

```bash
conda run -n silac_ml python -c "
import numpy as np
import os
from spectrum.dia_data import DIAData
tmp = '/tmp/v2_cache.npz'
np.savez(tmp, _format_version=np.int32(2))
try:
    DIAData.load_from_file(tmp)
    print('FAIL: v2 cache should have been rejected')
except ValueError as e:
    print(f'OK: v2 rejected with: {e}')
os.remove(tmp)
"
```

Expected: prints `OK: v2 rejected with: npz 缓存 /tmp/v2_cache.npz 的 _format_version=2, ...`

- [ ] **Step 10: Full regression**

Run: `conda run -n silac_ml pytest tests/ -q 2>&1 | tail -5`

Expected: tests that load existing cached npz files via load_from_file may now break IF those tests use v2 caches. Check: are there fixture-cached npz files in `tests/data/` or similar? Run grep `find tests -name "*.npz"`. If yes, regenerate them OR adjust the test to use a freshly-built `tmp_path` cache.

If any existing test now fails because of v2 cache, document the breakage and either:
  (a) regenerate the fixture by deleting it and running the smoke test (preferred)
  (b) keep v2 read-only support (NOT recommended — defeats the purpose)

- [ ] **Step 11: Commit**

```bash
git add spectrum/dia_data.py tests/test_deep_audit_p0.py
# Add other modified caller files (data_manager.py, flow_utils.py) only if updated
git commit -m "fix(dia_data): cache _format_version=3 embeds centroid params (P0-3, Silent-C3)

Audit finding Silent-C3 (2026-06-03 deep audit): .dia.npz cache stored
_format_version=2 but didn't include _centroid_enabled or
_centroid_rel_threshold. data_to_npz only checked file existence to
decide rebuild. User changing centroid params had no effect — stale
cache silently used.

Bump _format_version to 3. Save both centroid params into npz.
load_from_file gains optional expected_centroid_* kwargs; mismatch
raises with clear error pointing at the cache file to delete.

Old v2 caches automatically rejected on next load. User must run
'make clean-all && make all' once after upgrade (or manually rm
*.dia.npz) to regenerate caches."
```

---

### Task P0-4: Units-C1 — `multi_batch_work` heavy_mass fix

**Why:** `workflows/single_work.py:202` passes `ions_mass=light_mass` to `dia_data2.xic_ms2_peaks_extract` for the heavy XIC. SILAC heavy y-ions are +8 (K) or +10 (R) Da heavier — well outside ppm tol. All cross-run heavy fragment XICs return empty. Dormant bug (current `feature_type=0` never calls multi_batch_work) but masks Phase 0 correctness if `feature_type=1/2` is ever used.

**Files:**
- Modify: `workflows/single_work.py:202` (single line change)
- Test: append to `tests/test_deep_audit_p0.py`

- [ ] **Step 1: Append failing test**

Append to `tests/test_deep_audit_p0.py`:

```python
def test_multi_batch_work_passes_heavy_mass_for_heavy_xic():
    """multi_batch_work must pass heavy_mass (not light_mass) when extracting
    heavy MS2 XIC from dia_data2 (P0-4, Units-C1)."""
    from workflows.single_work import multi_batch_work

    captured = {"light_calls": [], "heavy_calls": []}

    class _RecordingDIA(_FakeDIA):
        def __init__(self, name, force_empty=False):
            super().__init__(force_empty=force_empty)
            self._name = name

        def xic_ms2_peaks_extract(self, rt, window, precursor_mz, ions_mass,
                                   mass_tol_ppm):
            # Record which DIA was queried and what mass was passed.
            captured[f"{self._name}_calls"].append(
                {"precursor_mz": precursor_mz, "ions_mass": ions_mass})
            return super().xic_ms2_peaks_extract(rt, window, precursor_mz,
                                                  ions_mass, mass_tol_ppm)

        def get_heavy_info(self, psm):
            # Return one fragment with distinct light vs heavy mass.
            return psm._precursor_mz + 4.0, [("y", 1, 100.0, 110.0)]

    dia_light = _RecordingDIA("light")
    dia_heavy = _RecordingDIA("heavy")
    psm1 = _FakePSM(mz=500.0, rt=10.0)
    psm2 = _FakePSM(mz=504.0, rt=10.1)  # heavy precursor

    multi_batch_work(psm1, dia_light, psm2, dia_heavy, _minimal_config())

    # dia_light (light channel) should be queried with light_mass=100.0
    light_masses = [c["ions_mass"] for c in captured["light_calls"]]
    assert 100.0 in light_masses, (
        f"P0-4: light DIA should be queried with light_mass; got {light_masses}")

    # dia_heavy (heavy channel) should be queried with heavy_mass=110.0
    # (NOT light_mass=100.0 — that was the bug)
    heavy_masses = [c["ions_mass"] for c in captured["heavy_calls"]]
    assert 110.0 in heavy_masses, (
        f"P0-4: heavy DIA must be queried with heavy_mass=110.0; got {heavy_masses}")
    assert 100.0 not in heavy_masses, (
        f"P0-4: heavy DIA should NOT receive light_mass=100.0 (Units-C1 bug); "
        f"got {heavy_masses}")
```

- [ ] **Step 2: Run test, verify it fails**

Run: `conda run -n silac_ml pytest tests/test_deep_audit_p0.py::test_multi_batch_work_passes_heavy_mass_for_heavy_xic -v`

Expected: FAIL — assertion `110.0 in heavy_masses` fails (the buggy code passes `light_mass=100.0` to heavy).

- [ ] **Step 3: Apply one-line fix**

In `workflows/single_work.py`, find line 202:

```python
            ions_mass=light_mass,
```

(inside the `heavy_ions_xic, heavy_all_intensity = dia_data2.xic_ms2_peaks_extract(...)` call.)

Replace with:

```python
            ions_mass=heavy_mass,
```

- [ ] **Step 4: Run test, verify PASS**

Run: `conda run -n silac_ml pytest tests/test_deep_audit_p0.py::test_multi_batch_work_passes_heavy_mass_for_heavy_xic -v`

Expected: PASS.

- [ ] **Step 5: Full regression**

Run: `conda run -n silac_ml pytest tests/ -q 2>&1 | tail -5`

Expected: no NEW failures.

- [ ] **Step 6: Commit**

```bash
git add workflows/single_work.py tests/test_deep_audit_p0.py
git commit -m "fix(single_work): multi_batch_work uses heavy_mass for heavy XIC (P0-4, Units-C1)

Audit finding Units-C1 (2026-06-03 deep audit): multi_batch_work:202
passed ions_mass=light_mass to dia_data2.xic_ms2_peaks_extract — the
heavy channel. SILAC y-ions are +8 (K) or +10 (R) Da heavier than
light, far outside any ppm tolerance. Effect: all cross-run heavy
fragment XICs returned empty, biasing the entire fragment-level
feature block to the empty-XIC branch.

Dormant bug (current 3 baselines all use feature_type=0 which calls
single_pair_work, not multi_batch_work). Fixing it before any
feature_type=1/2 run is attempted.

Single-line fix: light_mass -> heavy_mass at line 202. Behavioral
test records the ions_mass arg per channel to lock in the contract."
```
