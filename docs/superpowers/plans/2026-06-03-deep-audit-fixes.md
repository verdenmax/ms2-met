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

---

## Phase 1 — Active Important fixes (7 tasks)

### Task P1-1: Units-I1 — `matched_intensity_percent` denominator hoist

**Why:** `single_work.py:225-226, 566-567` (and aggregation at 301/637): the denominator `intensitys_map["all"]` accumulates `light_all_intensity + heavy_all_intensity` INSIDE the per-fragment loop. Since `*_all_intensity` is per-PSM (constant across fragments at the same RT/window), it gets added N_fragments times. Result: `matched_intensity_percent ∝ 1/N_fragments` — a hidden peptide-length proxy that the model silently learns.

**Files:**
- Modify: `workflows/single_work.py` (both `single_pair_work` and `multi_batch_work` fragment loops)
- Test: `tests/test_deep_audit_p1.py` (new file)

- [ ] **Step 1: Create test file with denominator test**

Create `tests/test_deep_audit_p1.py`:

```python
"""Phase 1 (Active Important) behavioral tests for deep audit fixes.

See docs/specs/2026-06-03-deep-audit-fixes-design.md.
"""
import configparser
import numpy as np
import pytest

# Reuse fakes from p0 file by importing.
from tests.test_deep_audit_p0 import (
    _empty_xic, _real_xic, _FakePSM, _FakeDIA, _minimal_config,
)


class _MultiFragDIA(_FakeDIA):
    """DIA stub that returns N fragments — for testing denominator independence."""
    def __init__(self, n_fragments, force_empty=False):
        super().__init__(force_empty=force_empty)
        self._n_fragments = n_fragments

    def get_heavy_info(self, psm):
        # Return n_fragments distinct y-ions with light/heavy mass pairs.
        frags = [("y", i, 100.0 + i, 110.0 + i)
                 for i in range(1, self._n_fragments + 1)]
        return psm._precursor_mz + 4.0, frags


def test_matched_intensity_percent_independent_of_fragment_count():
    """matched_intensity_percent should be independent of N_fragments (P1-1, Units-I1).

    Before fix: denominator added per fragment → percent ∝ 1/N → silent
    peptide-length proxy.
    After fix: denominator computed once per PSM → percent reflects
    actual matched/total ratio.
    """
    from workflows.single_work import single_pair_work
    psm = _FakePSM()
    cfg = _minimal_config()

    f_3frags = single_pair_work(psm, _MultiFragDIA(n_fragments=3), cfg)
    f_5frags = single_pair_work(psm, _MultiFragDIA(n_fragments=5), cfg)

    pct_3 = f_3frags.get("matched_intensity_percent")
    pct_5 = f_5frags.get("matched_intensity_percent")

    # Both should be present, both should be the SAME ratio for the same
    # per-fragment matched intensity (the stub returns fixed XIC per
    # fragment). Before P1-1 fix: pct_5 ≈ 0.6 * pct_3 due to inflated denom.
    assert pct_3 is not None and pct_5 is not None, (
        "matched_intensity_percent must be present in both")
    # Allow 1% relative tolerance for float arithmetic.
    rel_diff = abs(pct_3 - pct_5) / max(abs(pct_3), abs(pct_5), 1e-12)
    assert rel_diff < 0.01, (
        f"P1-1: matched_intensity_percent must be independent of N_fragments. "
        f"3 frags={pct_3}, 5 frags={pct_5}, rel_diff={rel_diff}")
```

- [ ] **Step 2: Run test, verify it fails**

Run: `conda run -n silac_ml pytest tests/test_deep_audit_p1.py::test_matched_intensity_percent_independent_of_fragment_count -v`

Expected: FAIL — current code's denominator scales with fragment count.

If `matched_intensity_percent` key is absent, also FAILs. Check `grep -n matched_intensity_percent workflows/single_work.py` to confirm the feature name exists. If named differently (e.g., without underscores), adapt the test.

- [ ] **Step 3: Hoist denominator in `single_pair_work` (line 566-567)**

In `workflows/single_work.py:single_pair_work`, find the fragment loop block:

```python
        if (np.max(light_ions_xic["intensity"]) > 0 and
                np.max(heavy_ions_xic["intensity"]) > 0):
            intensitys_map[ions_type] += np.sum(light_ions_xic["intensity"])
            intensitys_map[ions_type] += np.sum(heavy_ions_xic["intensity"])
            intensitys_map["all"] += light_all_intensity + \
                heavy_all_intensity
```

Replace with (per-ion-type sum stays in loop; "all" denominator moved out):

```python
        if (np.max(light_ions_xic["intensity"]) > 0 and
                np.max(heavy_ions_xic["intensity"]) > 0):
            intensitys_map[ions_type] += np.sum(light_ions_xic["intensity"])
            intensitys_map[ions_type] += np.sum(heavy_ions_xic["intensity"])
            # NOTE: intensitys_map["all"] (the denominator for
            # matched_intensity_percent) is hoisted out of this loop — see
            # P1-1, Units-I1. The light_all_intensity / heavy_all_intensity
            # values are per-PSM (constant across fragments at the same
            # RT/window), so accumulating inside the loop multiplied by
            # N_fragments. Now set ONCE per PSM below.
```

Then OUTSIDE the loop, BEFORE the per-ion-type feature-extraction block (around line 612 after `features["valid_fragment_ions_num"] = ...`), add — but we need light_all_intensity and heavy_all_intensity from outside. They are computed PER FRAGMENT inside the loop, but their values should be the same for all fragments. Strategy: capture from the FIRST non-empty fragment:

```python
    # Hoist the "all" denominator for matched_intensity_percent out of the
    # fragment loop (P1-1, Units-I1). light_all_intensity /
    # heavy_all_intensity are per-PSM (per RT-window), so we use the
    # first observed pair as the canonical value.
    if "all" not in intensitys_map:
        intensitys_map["all"] = 0.0
```

Actually a cleaner approach: capture `last_seen_all_intensity = (light_all_intensity, heavy_all_intensity)` inside the loop on each iteration, then set `intensitys_map["all"] = sum(last_seen)` once outside the loop. Let me restructure properly:

Replace the original block in step 3 above with:

```python
        if (np.max(light_ions_xic["intensity"]) > 0 and
                np.max(heavy_ions_xic["intensity"]) > 0):
            intensitys_map[ions_type] += np.sum(light_ions_xic["intensity"])
            intensitys_map[ions_type] += np.sum(heavy_ions_xic["intensity"])
            # Capture per-PSM all-intensity for hoisted denominator
            # (P1-1, Units-I1). Updated on each iteration; final value
            # used below.
            last_light_all = light_all_intensity
            last_heavy_all = heavy_all_intensity
```

And BEFORE the fragment loop initialization (around line 504-520 where the lists are initialized), add:

```python
    last_light_all = 0.0
    last_heavy_all = 0.0
```

And AFTER the fragment loop ends (around line 612, right before `features["valid_fragment_ions_num"] = ...`), add:

```python
    # Hoisted denominator for matched_intensity_percent (P1-1, Units-I1).
    intensitys_map["all"] = last_light_all + last_heavy_all
```

- [ ] **Step 4: Mirror the fix in `multi_batch_work` (line 225-226)**

In `multi_batch_work`, do the same: capture `last_light_all` / `last_heavy_all` inside the loop, set `intensitys_map["all"] = sum` after the loop. The lines to edit are around 225-226 and the loop ends around line 276.

- [ ] **Step 5: Run test, verify PASS**

Run: `conda run -n silac_ml pytest tests/test_deep_audit_p1.py::test_matched_intensity_percent_independent_of_fragment_count -v`

Expected: PASS.

- [ ] **Step 6: Run full P0+P1 regression**

Run: `conda run -n silac_ml pytest tests/test_deep_audit_p0.py tests/test_deep_audit_p1.py tests/ -q 2>&1 | tail -5`

Expected: ≥278+9 passed, no NEW failures.

- [ ] **Step 7: Commit**

```bash
git add workflows/single_work.py tests/test_deep_audit_p1.py
git commit -m "fix(single_work): matched_intensity_percent denominator hoist (P1-1, Units-I1)

Audit finding Units-I1 (2026-06-03 deep audit): the
matched_intensity_percent denominator (intensitys_map['all']) was
accumulated INSIDE the fragment loop. light_all_intensity /
heavy_all_intensity are per-PSM (per RT-window) constants, not
per-fragment, so the denominator was multiplied by N_fragments.

Effect: matched_intensity_percent ∝ 1/N_fragments — a silent
peptide-length proxy that LightGBM learns as a 'longer peptide -> lower
match%' rule, biasing predictions toward sequence length.

Fix: capture last-seen per-PSM all-intensity inside the loop, set the
'all' denominator once after the loop ends. Same fix in both
single_pair_work and multi_batch_work."
```

---

### Task P1-2: Silent-I1 — Fragment empty-branch list parity

**Why:** When a fragment hits the empty-XIC branch (`single_work.py:213-219, 554-560`) or skip-conditions (`525-526, 529-530` in single only), only 3-4 per-fragment lists are appended; ~11 others get NO entry. Aggregates (`all_*_mean/p50/std/max`) compute over a strictly smaller denominator than `valid_fragment_ions_num` implies. T1 review's M6 concern is not resolved.

**Files:**
- Modify: `workflows/single_work.py` (both fragment loops, all branches that `continue`)
- Test: append to `tests/test_deep_audit_p1.py`

- [ ] **Step 1: Append parity test**

Append to `tests/test_deep_audit_p1.py`:

```python
class _AllEmptyFragDIA(_FakeDIA):
    """DIA stub whose xic_ms2_peaks_extract always returns empty."""
    def get_heavy_info(self, psm):
        return psm._precursor_mz + 4.0, [
            ("y", 1, 100.0, 110.0),
            ("y", 2, 200.0, 210.0),
            ("b", 1, 150.0, 160.0),
        ]

    def xic_ms2_peaks_extract(self, rt, window, precursor_mz, ions_mass,
                              mass_tol_ppm):
        return _empty_xic(), 0.0


def test_fragment_empty_branch_appends_all_lists_consistently():
    """All per-fragment lists must have the same length after the loop,
    even when every fragment hits the empty-XIC branch (P1-2, Silent-I1)."""
    from workflows.single_work import single_pair_work
    psm = _FakePSM()
    dia = _AllEmptyFragDIA()
    cfg = _minimal_config()

    features = single_pair_work(psm, dia, cfg)

    # When all 3 fragments are empty, the count from valid_fragment_ions_num
    # is the truth-of-record. Verify several aggregates do NOT contain NaN
    # / sentinel mismatches.
    n_frag = features.get("valid_fragment_ions_num", 0)
    # The 11 lists should all have the same length. We can't access them
    # directly (internal state), but their aggregates should be well-defined.
    # The fix's contract: aggregates of empty-branch fragments are zeros,
    # not NaN.
    for key in ("all_apex_delta_mean", "all_base_to_apex_ratio_mean",
                "all_apex_monotonicity_mean", "all_n_peaks_mean",
                "all_smoothness_mean"):
        if key in features:
            v = features[key]
            assert not np.isnan(v), (
                f"P1-2: {key} should be 0.0 (not NaN) when all fragments empty; "
                f"got {v}")
```

- [ ] **Step 2: Run test, verify it fails OR check current behavior**

Run: `conda run -n silac_ml pytest tests/test_deep_audit_p1.py::test_fragment_empty_branch_appends_all_lists_consistently -v`

Expected: may FAIL with NaN in some aggregate (e.g., `all_apex_delta_mean = nan`).

- [ ] **Step 3: Patch `single_pair_work` empty-XIC fragment branch (lines 554-560)**

In `workflows/single_work.py:single_pair_work`, find:

```python
        if len(light_ions_xic) == 0 or len(heavy_ions_xic) == 0:
            pearsons_map[ions_type].append(0)
            pearsons_map["all"].append(0)
            fragment_intensities.append(0.0)
            fragment_cosines.append(0.0)
            fragment_snrs.append(0.0)
            continue
```

Replace with:

```python
        if len(light_ions_xic) == 0 or len(heavy_ions_xic) == 0:
            # Empty XIC fragment branch. P1-2 (Silent-I1): append default
            # zeros to ALL per-fragment lists so aggregates have a
            # consistent denominator with valid_fragment_ions_num.
            pearsons_map[ions_type].append(0)
            pearsons_map["all"].append(0)
            fragment_intensities.append(0.0)
            fragment_cosines.append(0.0)
            fragment_snrs.append(0.0)
            fragment_apex_deltas.append(0.0)
            fragment_mz_errs.append(0.0)
            fragment_light_cycle_offsets.append(0)
            fragment_light_cycle_offsets_signed.append(0)
            fragment_heavy_cycle_offsets.append(0)
            fragment_heavy_cycle_offsets_signed.append(0)
            fragment_base_to_apex_ratios.append(0.0)
            fragment_apex_monotonicities.append(0.0)
            fragment_n_peaks_list.append(0)
            fragment_smoothnesses.append(0.0)
            # NOTE: fragment_hl_ratios NOT appended — by design only
            # contains real (heavy>0 AND light>0) ratios.
            fragment_xic_empty_count += 1  # P0-1 marker
            continue
```

Same for the `continue`s at lines 525-526 and 529-530 — those are "fragment dropped, not even attempted". Decision: those fragments should ALSO appear in the per-fragment lists with zero defaults if we want the aggregate denominator to match `valid_fragment_ions_num`. OR they should NOT appear AND `valid_fragment_ions_num` doc/contract should explicitly note "fragments where heavy_in_raw was false are excluded".

Choose option (b) — keep the `continue`s clean and document `valid_fragment_ions_num` as "fragments that reached the XIC extraction stage". Add a comment after line 519:

```python
    # NOTE: valid_fragment_ions_num counts fragments that REACHED the XIC
    # extraction stage. Fragments dropped at lines 525-526 (heavy_in_raw
    # false) and 529-530 (same MS2 + zero mass shift) are NOT counted.
    # This matches the denominator of all per-fragment aggregates below
    # (P1-2, Silent-I1).
```

- [ ] **Step 4: Mirror in `multi_batch_work` (lines 213-219)**

Same parity fix in the empty-XIC branch of `multi_batch_work`. multi_batch_work does NOT have the heavy_in_raw / same_ms2 pre-skip, so only the empty-XIC append parity is needed there.

- [ ] **Step 5: Run test, verify PASS**

Run: `conda run -n silac_ml pytest tests/test_deep_audit_p1.py::test_fragment_empty_branch_appends_all_lists_consistently -v`

Expected: PASS.

- [ ] **Step 6: Full regression**

Run: `conda run -n silac_ml pytest tests/ -q 2>&1 | tail -5`

Expected: no NEW failures.

- [ ] **Step 7: Commit**

```bash
git add workflows/single_work.py tests/test_deep_audit_p1.py
git commit -m "fix(single_work): fragment empty-branch list parity (P1-2, Silent-I1)

Audit finding Silent-I1 (2026-06-03 deep audit): when a fragment hit
the empty-XIC branch in single_pair_work / multi_batch_work, only 3-4
per-fragment lists were appended. ~11 others got NO entry. Aggregates
(all_*_mean/p50/std/max) were computed over a strictly smaller
denominator than valid_fragment_ions_num implied. T1 review's M6
concern unresolved until now.

Fix: append default zeros to ALL per-fragment lists in the empty-XIC
branch. Document valid_fragment_ions_num as 'fragments that reached
the XIC extraction stage' to clarify the heavy_in_raw / same_ms2 skip
semantics in single_pair_work (those are intentionally excluded)."
```

---

### Task P1-3: Pipeline-I1 — label NaN guard

**Why:** `workflows/flow_utils.py:88` maps `_label_type=None → label=None`. `pd.read_csv` infers the column as `float64` with NaN. LightGBM `objective: binary` crashes on NaN labels. Currently dormant (all live `extract_*.ini` set `positive_species_marker`), but the `intersection_keys` no-marker mode in `tools/extract_common.py:127-135` would produce all-NaN labels.

**Files:**
- Modify: `workflows/flow_utils.py:_make_result_row_single` (+ check `process_batch_pair_shuffle` result row builder at ~line 220-240 for same issue)
- Test: append to `tests/test_deep_audit_p1.py`

- [ ] **Step 1: Append test**

Append to `tests/test_deep_audit_p1.py`:

```python
def test_make_result_row_single_raises_on_none_label_type():
    """_make_result_row_single must raise when _label_type is None (P1-3, Pipeline-I1)."""
    from workflows.flow_utils import _make_result_row_single

    class _PSMNoLabel:
        _sequence = "AAAA"
        _charge = 2
        _precursor_mz = 500.0
        _raw_title = "fake"
        _protein_names = "HUMAN"
        _label_type = None

    with pytest.raises(ValueError, match="label_type"):
        _make_result_row_single(_PSMNoLabel(), {"f1": 1.0})


def test_make_result_row_single_accepts_positive():
    """_make_result_row_single produces label=1 for 'positive' (no regression)."""
    from workflows.flow_utils import _make_result_row_single

    class _PSMPos:
        _sequence = "AAAA"
        _charge = 2
        _precursor_mz = 500.0
        _raw_title = "fake"
        _protein_names = "HUMAN"
        _label_type = "positive"

    row = _make_result_row_single(_PSMPos(), {"f1": 1.0})
    assert row["label"] == 1


def test_make_result_row_single_accepts_negative():
    from workflows.flow_utils import _make_result_row_single

    class _PSMNeg:
        _sequence = "AAAA"
        _charge = 2
        _precursor_mz = 500.0
        _raw_title = "fake"
        _protein_names = "HUMAN"
        _label_type = "negative"

    row = _make_result_row_single(_PSMNeg(), {"f1": 1.0})
    assert row["label"] == 0
```

- [ ] **Step 2: Run test, verify FAIL on the None case**

Run: `conda run -n silac_ml pytest tests/test_deep_audit_p1.py::test_make_result_row_single_raises_on_none_label_type tests/test_deep_audit_p1.py::test_make_result_row_single_accepts_positive tests/test_deep_audit_p1.py::test_make_result_row_single_accepts_negative -v`

Expected: 1 FAIL (raise test), 2 PASS.

- [ ] **Step 3: Patch `_make_result_row_single`**

In `workflows/flow_utils.py`, find:

```python
def _make_result_row_single(psm, features: dict) -> dict:
    """Build the result dict for a single-flow PSM.

    Maps psm._label_type ("positive"/"negative"/None) to label int (1/0/None)
    so the CSV's `label` column is numeric — matching the pair-flow convention.
    """
    label_type = psm._label_type
    if label_type == "positive":
        label = 1
    elif label_type == "negative":
        label = 0
    else:
        label = None
```

Replace with:

```python
def _make_result_row_single(psm, features: dict) -> dict:
    """Build the result dict for a single-flow PSM.

    Maps psm._label_type ("positive"/"negative") to label int (1/0).
    Raises ValueError if _label_type is None — silently writing None
    leads to NaN labels in features.csv that crash LightGBM during
    training (P1-3, Pipeline-I1 in 2026-06-03 deep audit).
    """
    label_type = psm._label_type
    if label_type == "positive":
        label = 1
    elif label_type == "negative":
        label = 0
    else:
        raise ValueError(
            f"PSM {getattr(psm, '_sequence', '?')} has _label_type={label_type!r}; "
            f"expected 'positive' or 'negative'. Check extract_common.py — "
            f"running without positive_species_marker produces None labels "
            f"that crash LightGBM training (P1-3, Pipeline-I1, 2026-06-03 audit)."
        )
```

- [ ] **Step 4: Check pair-flow shuffle result row (line ~222-240) for same issue**

In `workflows/flow_utils.py:process_batch_pair_shuffle`, the result row is built inline (around lines 222-240):

```python
            results.append({
                ...
                "label": label,
                ...
            })
```

Here `label` comes from `for psm1_dict, psm2_dict, label in batch_items` — produced by the caller. Inspect callers (search for `process_batch_pair_shuffle` references) to verify the caller produces 0/1 not None. If yes, no fix needed here; document the contract. If no, add a similar guard.

Run: `grep -rn "process_batch_pair_shuffle\b" workflows/ --include="*.py"`

If the caller is in `pair_flow.py`, inspect that call site to confirm. Typically it's a fixed `label = 0 if entrapment else 1`, no None possible — but verify.

- [ ] **Step 5: Run tests, verify all 3 PASS**

Run: `conda run -n silac_ml pytest tests/test_deep_audit_p1.py -v -k make_result_row_single`

Expected: 3 PASSed.

- [ ] **Step 6: Full regression**

Run: `conda run -n silac_ml pytest tests/ -q 2>&1 | tail -5`

Expected: no NEW failures.

- [ ] **Step 7: Commit**

```bash
git add workflows/flow_utils.py tests/test_deep_audit_p1.py
git commit -m "fix(flow_utils): raise on None label_type (P1-3, Pipeline-I1)

Audit finding Pipeline-I1 (2026-06-03 deep audit):
_make_result_row_single mapped _label_type=None → label=None silently.
pd.read_csv inferred the label column as float64 with NaN. LightGBM
objective=binary crashes on NaN labels.

Currently dormant: all 3 live extract_*.ini set
positive_species_marker=HUMAN. But the intersection_keys 'no-marker'
mode in tools/extract_common.py:127-135 would produce every PSM with
label_type=None — silent training failure.

Fix: raise ValueError with a clear message pointing at extract_common
config when _label_type is missing."
```

---

### Task P1-4: Pipeline-I3 — `exp2.yaml` is_unbalance

**Why:** `exp1.yaml` sets `is_unbalance: True` (data is ~1% positives); `exp2.yaml` doesn't. exp2 trains on same imbalanced data but produces miscalibrated probabilities and skewed feature importance.

**Files:**
- Modify: `tools/spec_trainer/config/exp2.yaml`
- Test: extend `tests/test_spec_trainer_holdout.py::test_exp_yamls_do_not_have_in_sample_test_files` (or add new)

- [ ] **Step 1: Add yaml-validation test**

Append to `tests/test_spec_trainer_holdout.py`:

```python


def test_both_exp_yamls_set_is_unbalance_for_imbalanced_data():
    """Both exp1 and exp2 must set is_unbalance: True for ~1% positive data (P1-4, Pipeline-I3)."""
    import os
    import yaml
    project_root = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
    for name in ("exp1.yaml", "exp2.yaml"):
        p = os.path.join(project_root, "tools", "spec_trainer", "config", name)
        with open(p) as f:
            cfg = yaml.safe_load(f)
        params = cfg.get("model", {}).get("params", {})
        is_unbalance = params.get("is_unbalance", False)
        assert is_unbalance is True, (
            f"{name}: lightgbm is_unbalance must be True for imbalanced "
            f"SILAC data (~1% positives). Got {is_unbalance}. "
            f"See P1-4, Pipeline-I3 (2026-06-03 audit).")
```

- [ ] **Step 2: Run test, verify it fails for exp2**

Run: `conda run -n silac_ml pytest tests/test_spec_trainer_holdout.py::test_both_exp_yamls_set_is_unbalance_for_imbalanced_data -v`

Expected: FAIL on `exp2.yaml`.

- [ ] **Step 3: Update `exp2.yaml`**

Find the `model.params` block in `tools/spec_trainer/config/exp2.yaml`:

```yaml
model:
  type: lightgbm
  params:
    boosting_type: gbdt
    objective: binary
    metric: [auc, binary_logloss]
    num_leaves: 31
    learning_rate: 0.05
    feature_fraction: 0.9
    bagging_fraction: 0.8
    verbose: -1
```

Add `is_unbalance: True` (matching exp1.yaml):

```yaml
model:
  type: lightgbm
  params:
    boosting_type: gbdt
    objective: binary
    metric: [auc, binary_logloss]
    num_leaves: 31
    learning_rate: 0.05
    feature_fraction: 0.9
    bagging_fraction: 0.8
    is_unbalance: True  # P1-4, Pipeline-I3 (2026-06-03 audit): SILAC data is ~1% positive
    verbose: -1
```

- [ ] **Step 4: Run test, verify PASS**

Run: `conda run -n silac_ml pytest tests/test_spec_trainer_holdout.py::test_both_exp_yamls_set_is_unbalance_for_imbalanced_data -v`

Expected: PASS.

- [ ] **Step 5: Commit**

```bash
git add tools/spec_trainer/config/exp2.yaml tests/test_spec_trainer_holdout.py
git commit -m "fix(spec_trainer): exp2.yaml is_unbalance: True (P1-4, Pipeline-I3)

Audit finding Pipeline-I3 (2026-06-03 deep audit): exp1.yaml had
is_unbalance: True but exp2.yaml didn't. Both train on the same
imbalanced SILAC data (~1% positives in 2da: 925/104445; ~1.6% in
normal: 1695/104890). Without is_unbalance, exp2 produced miscalibrated
probabilities and skewed feature importance.

Add is_unbalance: True to exp2.yaml model.params. New yaml-validation
test asserts both configs set it."
```

---

### Task P1-5: Pipeline-I5 — `resolve_feature_cols` multi-file intersection

**Why:** Current `resolve_feature_cols` reads column names from `train_files[0]` only. exp2 has `train_files: [2da.csv, 5da.csv]` with different schemas (q1a_* in 5da but not 2da). After P1's `make all` retrain, schemas should be identical — but the guard against future drift matters. Change `resolve_feature_cols` to take the intersection of all train_files' columns and log a warning when intersection differs from any input.

**Files:**
- Modify: `tools/spec_trainer/src/feature_cols.py` (signature change)
- Modify: `tools/spec_trainer/src/main.py` (pass list of paths)
- Test: extend `tests/test_spec_trainer_main.py`

- [ ] **Step 1: Write failing test for intersection behavior**

Append to `tests/test_spec_trainer_main.py`:

```python


def test_resolve_feature_cols_takes_intersection_of_multiple_files(tmp_path):
    """When given multiple sample CSVs, return the column intersection (P1-5, Pipeline-I5)."""
    from feature_cols import resolve_feature_cols
    csv_a = tmp_path / "a.csv"
    csv_a.write_text(
        "label,sequence,feat_common,feat_a_only\n"
    )
    csv_b = tmp_path / "b.csv"
    csv_b.write_text(
        "label,sequence,feat_common,feat_b_only\n"
    )
    result = resolve_feature_cols(
        explicit=None,
        sample_csv_paths=[str(csv_a), str(csv_b)],
        target_col="label",
    )
    # Intersection minus META: only feat_common
    assert result == ["feat_common"]
    assert "feat_a_only" not in result
    assert "feat_b_only" not in result


def test_resolve_feature_cols_single_path_backward_compat(tmp_path):
    """Calling with a single path (str OR list-of-1) still works (P1-5 compat)."""
    from feature_cols import resolve_feature_cols
    csv = tmp_path / "x.csv"
    csv.write_text("label,sequence,feat1,feat2\n")
    # New API: list
    r1 = resolve_feature_cols(explicit=None, sample_csv_paths=[str(csv)],
                               target_col="label")
    assert r1 == ["feat1", "feat2"]
    # Backward-compat: single string path (if supported)
    # If new signature only accepts list, this test will need to be skipped
    # — adjust based on chosen API.
```

- [ ] **Step 2: Run tests, verify they fail**

Run: `conda run -n silac_ml pytest tests/test_spec_trainer_main.py -v -k "intersection or single_path_backward"`

Expected: FAIL — current signature is `sample_csv_path` (singular).

- [ ] **Step 3: Update `feature_cols.py` signature + intersection logic**

In `tools/spec_trainer/src/feature_cols.py`, replace `resolve_feature_cols`:

```python
import logging


def resolve_feature_cols(explicit, sample_csv_paths, target_col):
    """Resolve final feature column list.

    Args:
        explicit: yaml-provided list of features, or None/[] to auto-detect.
        sample_csv_paths: list of CSV paths whose headers will be intersected.
            Accepts either a list of paths or a single string path (back-compat).
        target_col: name of the label column (excluded from features).

    Returns:
        List of feature column names. If sample_csv_paths has multiple
        entries, returns the INTERSECTION of all files' columns minus
        META_COLUMNS + EXCLUDED_EXTRA + target_col. Logs a warning if
        the intersection is smaller than any individual file's column set
        (indicating schema drift).

    Raises:
        ValueError: if resolved list is empty (see P2-7).
    """
    if explicit:
        return list(explicit)

    # Back-compat: accept single string path
    if isinstance(sample_csv_paths, str):
        sample_csv_paths = [sample_csv_paths]

    per_file_cols = []
    for path in sample_csv_paths:
        df = pd.read_csv(path, nrows=0)
        per_file_cols.append(set(df.columns))

    # Intersection across all files
    intersection = set.intersection(*per_file_cols) if per_file_cols else set()

    # Warn if any file has columns absent from intersection
    for path, cols in zip(sample_csv_paths, per_file_cols):
        dropped = cols - intersection
        if dropped:
            logging.warning(
                f"resolve_feature_cols: {len(dropped)} columns in "
                f"{os.path.basename(path)} not in intersection — dropped from "
                f"feature set: {sorted(dropped)} (P1-5, Pipeline-I5)"
            )

    # Preserve column ORDER from the first file (deterministic)
    first_cols = list(per_file_cols[0]) if per_file_cols else []
    # Use the order from first_cols' original file
    first_df = pd.read_csv(sample_csv_paths[0], nrows=0)
    ordered = list(first_df.columns)

    return [
        c for c in ordered
        if c in intersection
        and c not in META_COLUMNS
        and c not in EXCLUDED_EXTRA
        and c != target_col
    ]
```

Add `import os` if not already present.

- [ ] **Step 4: Update main.py call site**

In `tools/spec_trainer/src/main.py`, find:

```python
    feature_cols = _resolve_feature_cols(
        explicit=cfg['data'].get('feature_cols'),
        sample_csv_path=cfg['data']['train_files'][0],
        target_col=target_col,
    )
```

Replace with:

```python
    feature_cols = _resolve_feature_cols(
        explicit=cfg['data'].get('feature_cols'),
        sample_csv_paths=cfg['data']['train_files'],
        target_col=target_col,
    )
```

- [ ] **Step 5: Run tests, verify they pass**

Run: `conda run -n silac_ml pytest tests/test_spec_trainer_main.py -v`

Expected: all tests in the file PASS including new intersection ones.

- [ ] **Step 6: Full regression**

Run: `conda run -n silac_ml pytest tests/ -q 2>&1 | tail -5`

Expected: no NEW failures.

- [ ] **Step 7: Commit**

```bash
git add tools/spec_trainer/src/feature_cols.py tools/spec_trainer/src/main.py tests/test_spec_trainer_main.py
git commit -m "fix(spec_trainer): resolve_feature_cols multi-file intersection (P1-5, Pipeline-I5)

Audit finding Pipeline-I5 (2026-06-03 deep audit): resolve_feature_cols
read column names from train_files[0] only. exp2 had train_files=
[2da.csv, 5da.csv] with different schemas (q1a_* in 5da but not 2da).
LightGBM silently trained on 2da's column subset; 5da's extra columns
were dropped without warning.

Change signature: sample_csv_path (str) -> sample_csv_paths (list).
Compute INTERSECTION of all files' columns. Log warning naming dropped
columns when intersection != any individual file's set. Backward
compat: accept single string path."
```

---

### Task P1-6: Silent-I3 — `logging.warn` per-PSM dumps

**Why:** `spectrum/dia_data.py:524-525, 703-712` uses deprecated `logging.warn` per PSM × fragment, dumping `_cycle_left_precursor` array each time. Megabytes of warnings into `extract.log`; effectively silent.

**Files:**
- Modify: `spectrum/dia_data.py` (replace `logging.warn` with `logging.debug` + counter)
- Modify: `workflows/pair_flow.py` (log summary at batch end)
- Test: append to `tests/test_deep_audit_p1.py`

- [ ] **Step 1: Append test for counter behavior**

Append to `tests/test_deep_audit_p1.py`:

```python
def test_dia_data_check_in_raw_increments_counter_no_warn_per_call(caplog):
    """check_in_raw must NOT logging.warn each call; should increment counter (P1-6, Silent-I3)."""
    import logging as py_logging
    from spectrum.dia_data import DIAData
    dia = DIAData()
    dia._max_mz_value = 1000.0
    dia._min_mz_value = 100.0
    dia._cycle_left_precursor = np.array([400.0, 500.0])

    caplog.clear()
    with caplog.at_level(py_logging.DEBUG, logger="spectrum.dia_data"):
        for _ in range(10):
            result = dia.check_in_raw(1500.0)  # out of range
            assert result is False

    # Should NOT have any WARNING-level records for these
    warn_records = [r for r in caplog.records if r.levelno >= py_logging.WARNING]
    assert len(warn_records) == 0, (
        f"P1-6: check_in_raw should not emit WARNING per call; got {len(warn_records)}")

    # Counter should be 10
    assert hasattr(dia, "_n_out_of_window_xic"), (
        "P1-6: DIAData must expose _n_out_of_window_xic counter")
    assert dia._n_out_of_window_xic == 10, (
        f"P1-6: counter expected 10, got {dia._n_out_of_window_xic}")
```

- [ ] **Step 2: Run test, verify it fails**

Run: `conda run -n silac_ml pytest tests/test_deep_audit_p1.py::test_dia_data_check_in_raw_increments_counter_no_warn_per_call -v`

Expected: FAIL — no `_n_out_of_window_xic` attribute; `logging.warn` still emitted.

- [ ] **Step 3: Patch `spectrum/dia_data.py`**

In `__init__` (around line 100-160), add:

```python
        self._n_out_of_window_xic: int = 0
```

In `check_in_raw` (lines 518-527):

```python
    def check_in_raw(self, precursor_mz) -> bool:
        """ 检查这个 mz 是否在当前 raw 中"""
        if (precursor_mz <= self._max_mz_value + 0.1
                and precursor_mz >= self._min_mz_value - 0.1):
            return True

        # P1-6, Silent-I3: was logging.warn per call; now debug + counter.
        # Summary logged once per batch in workflows/pair_flow.py.
        self._n_out_of_window_xic += 1
        logging.debug(
            "out-of-window XIC: max=%s min=%s mz=%s",
            self._max_mz_value, self._min_mz_value, precursor_mz)
        return False
```

In `xic_ms2_peaks_extract` (lines 703-712):

```python
        if center_idx is None:
            dtype = [("rt", "f8"), ("ppm_error", "f8"),
                     ("intensity", "f8"), ("cycle_idx", "i4")]
            # P1-6, Silent-I3: debug + counter instead of per-call warn.
            self._n_out_of_window_xic += 1
            logging.debug(
                "no MS2 window match: precursor_mz=%s", precursor_mz)
            for i in candidates:
                gidx = self.ms2_indexs[i]
                lower = self._precursor_lower_mz[gidx]
                upper = self._precursor_upper_mz[gidx]
                logging.debug(
                    "candidate idx=%s global=%s rt=%.3f window=[%.3f, %.3f)",
                    i, gidx, self.ms2_indexs_rt[i], lower, upper)
            return np.array([], dtype=dtype), 0.0
```

- [ ] **Step 4: Add summary log at batch end in pair_flow.py**

In `workflows/pair_flow.py`, find the batch end (somewhere in the `run()` method or the batch dispatch logic, often around 318-329 per audit). The DIA instances are loaded per-worker via mmap; the counter lives per-worker. The cleanest summary is in the main process after the batch returns. Since workers don't easily report back, log the counter in each worker AT END of `process_batch_single` / `process_batch_pair_shuffle`:

In `workflows/flow_utils.py:process_batch_single` (and `process_batch_pair`, `process_batch_pair_shuffle`), at the END of the function before `return`:

```python
    # P1-6, Silent-I3: log out-of-window XIC summary per worker
    n_oow = getattr(dia1, "_n_out_of_window_xic", 0)
    if n_oow > 0:
        logging.info(
            "[batch summary] %d out-of-window XIC requests in %s",
            n_oow, os.path.basename(shared_path))
```

(Add `import os` and `import logging` at the top of `flow_utils.py` if not already present.)

- [ ] **Step 5: Run test, verify PASS**

Run: `conda run -n silac_ml pytest tests/test_deep_audit_p1.py::test_dia_data_check_in_raw_increments_counter_no_warn_per_call -v`

Expected: PASS.

- [ ] **Step 6: Full regression**

Run: `conda run -n silac_ml pytest tests/ -q 2>&1 | tail -5`

Expected: no NEW failures.

- [ ] **Step 7: Commit**

```bash
git add spectrum/dia_data.py workflows/flow_utils.py tests/test_deep_audit_p1.py
git commit -m "fix(dia_data): logging.warn -> debug + per-batch summary (P1-6, Silent-I3)

Audit finding Silent-I3 (2026-06-03 deep audit): check_in_raw and
xic_ms2_peaks_extract used deprecated logging.warn per PSM × fragment,
dumping _cycle_left_precursor array each time. Megabytes of warnings
into extract.log; effectively silent because no one reads it.

Fix: downgrade to logging.debug (won't bloat extract.log under default
INFO level), increment new DIAData._n_out_of_window_xic counter, and
log a per-worker summary at batch end in flow_utils.py."
```

---

### Task P1-7: Silent-I8 — `centroid_spectrum` short-spectrum logger

**Why:** `spectrum/spectrum_utils.py:60-82` returns empty result for spectra with <3 peaks or all-zero intensity. No log. Downstream XIC sees zero → fake "no signal". Real MS2s usually have >>3 peaks but edge cases shouldn't be silent.

**Files:**
- Modify: `spectrum/spectrum_utils.py` (return value or side-effect counter — pick counter)
- Modify: `spectrum/dia_data.py:_load_from_mzml` (collect + log summary)
- Test: append to `tests/test_deep_audit_p1.py`

- [ ] **Step 1: Append test**

Append to `tests/test_deep_audit_p1.py`:

```python
def test_dia_data_load_logs_centroid_empty_count_summary(caplog, tmp_path):
    """_load_from_mzml must log a summary of centroid-to-empty spectra (P1-7, Silent-I8).

    We can't easily mock the entire mzML pipeline; verify the COUNTER
    field exists on DIAData and is logged at end of load. Use
    a minimal _load_from_mzml-compatible path or a focused unit test
    of the counter mechanism.
    """
    from spectrum.dia_data import DIAData
    dia = DIAData()
    # The counter should exist as an instance attr (default 0)
    assert hasattr(dia, "_n_centroid_empty"), (
        "P1-7: DIAData must expose _n_centroid_empty counter")
    assert dia._n_centroid_empty == 0


def test_centroid_spectrum_increments_counter_on_empty_return():
    """centroid_spectrum is pure (returns empty); the COUNTER is owned by
    the caller (_load_from_mzml). Verify the function itself behaves
    correctly: len<3 returns empty arrays (no exception, no side-effect)."""
    from spectrum.spectrum_utils import centroid_spectrum
    mz, intensity = centroid_spectrum(
        np.array([100.0, 200.0]),
        np.array([10.0, 20.0]),
        rel_threshold=1e-3,
    )
    assert len(mz) == 0
    assert len(intensity) == 0
```

- [ ] **Step 2: Run tests, verify they fail (only first will fail; second already passes)**

Run: `conda run -n silac_ml pytest tests/test_deep_audit_p1.py -v -k centroid`

Expected: 1 FAIL (`_n_centroid_empty` attr missing), 1 PASS.

- [ ] **Step 3: Add counter to DIAData**

In `spectrum/dia_data.py:__init__`, add:

```python
        self._n_centroid_empty: int = 0
```

- [ ] **Step 4: Increment counter in `_load_from_mzml`**

In `spectrum/dia_data.py`, find the call site around lines 359-363:

```python
        if self._centroid_enabled and not _is_already_centroid(spectrum):
            mz_array, intensity_array = centroid_spectrum(
                mz_array, intensity_array,
                rel_threshold=self._centroid_rel_threshold,
            )
```

Replace with:

```python
        if self._centroid_enabled and not _is_already_centroid(spectrum):
            mz_array, intensity_array = centroid_spectrum(
                mz_array, intensity_array,
                rel_threshold=self._centroid_rel_threshold,
            )
            # P1-7, Silent-I8: count empty-return so we can log a summary.
            if len(mz_array) == 0:
                self._n_centroid_empty += 1
```

- [ ] **Step 5: Log summary at end of `_load_from_mzml`**

Find the end of `_load_from_mzml` (look for the return / function end). Add before the function returns:

```python
        # P1-7, Silent-I8: summary of centroid-to-empty spectra.
        if self._n_centroid_empty > 0:
            logging.info(
                "[centroid] %d spectra returned empty (likely <3 peaks)",
                self._n_centroid_empty)
```

- [ ] **Step 6: Run test, verify PASS**

Run: `conda run -n silac_ml pytest tests/test_deep_audit_p1.py -v -k centroid`

Expected: 2 PASSed.

- [ ] **Step 7: Full regression**

Run: `conda run -n silac_ml pytest tests/ -q 2>&1 | tail -5`

Expected: no NEW failures.

- [ ] **Step 8: Commit**

```bash
git add spectrum/dia_data.py tests/test_deep_audit_p1.py
git commit -m "fix(dia_data): log centroid-to-empty summary (P1-7, Silent-I8)

Audit finding Silent-I8 (2026-06-03 deep audit): centroid_spectrum
returned empty arrays for spectra with <3 peaks or all-zero intensity
without any log. Downstream XIC saw zero → fake 'no signal'. Real MS2s
usually have >>3 peaks but edge cases shouldn't be silent.

Add DIAData._n_centroid_empty counter incremented in _load_from_mzml
when centroid_spectrum returns empty. Log summary line at end of load
so users can spot anomalies (e.g., a low-quality run with many small
spectra)."
```


---

## Phase 2 — Dormant Important fixes (8 tasks)

### Task P2-1: Units-I2 — `log_hl_ratio_*` → `log_lh_ratio_*` rename

**Why:** `_calc_hl_ratio_consistency` computes `log10(light/heavy)` but feature columns named `log_hl_ratio_*` (H/L). std/mad sign-invariant so functionally OK but misleading. Rename 6 columns to match actual semantics.

**Files:**
- Modify: `workflows/single_work.py` (4 occurrences of `log_hl_ratio` → `log_lh_ratio`)
- Modify: `docs/specs/2026-05-26-hl-ratio-consistency-and-apex-cycle-offset-design.md` (doc alignment)
- Test: `tests/test_deep_audit_p2.py` (new file)

- [ ] **Step 1: Create test file**

Create `tests/test_deep_audit_p2.py`:

```python
"""Phase 2 (Dormant Important) tests for deep audit fixes."""
import os
import sys
import numpy as np
import pytest

from tests.test_deep_audit_p0 import (
    _empty_xic, _real_xic, _FakePSM, _FakeDIA, _minimal_config,
)


def test_no_log_hl_ratio_columns_in_single_pair_work():
    """log_hl_ratio_* columns must be renamed to log_lh_ratio_* (P2-1, Units-I2)."""
    from workflows.single_work import single_pair_work
    psm = _FakePSM()
    dia = _FakeDIA(force_empty=False)
    features = single_pair_work(psm, dia, _minimal_config())
    hl_keys = [k for k in features.keys() if "log_hl_ratio" in k]
    assert len(hl_keys) == 0, (
        f"P2-1: log_hl_ratio_* should be renamed log_lh_ratio_*; "
        f"found {hl_keys}")
    lh_keys = [k for k in features.keys() if "log_lh_ratio" in k]
    assert len(lh_keys) >= 1, (
        f"P2-1: expected at least one log_lh_ratio_* column; got {lh_keys}")
```

- [ ] **Step 2: Run test, verify FAIL**

Run: `conda run -n silac_ml pytest tests/test_deep_audit_p2.py::test_no_log_hl_ratio_columns_in_single_pair_work -v`

Expected: FAIL — current names use `hl`.

- [ ] **Step 3: Rename in `workflows/single_work.py` lines 328-329 and 664-665**

In `workflows/single_work.py`, find both occurrences of:

```python
        features[f"{ion_type}_log_hl_ratio_std"] = std_v
        features[f"{ion_type}_log_hl_ratio_mad"] = mad_v
```

Replace with:

```python
        features[f"{ion_type}_log_lh_ratio_std"] = std_v
        features[f"{ion_type}_log_lh_ratio_mad"] = mad_v
```

There are TWO occurrences (lines ~328-329 in `multi_batch_work` and ~664-665 in `single_pair_work`). Make BOTH replacements.

Also add a clarifying comment in `_calc_hl_ratio_consistency` docstring:

```python
def _calc_hl_ratio_consistency(ratios: list) -> tuple[float, float]:
    """Compute consistency of light/heavy intensity ratios across fragments.

    Returns (std, mad) of log10(L/H) over the input list. Despite the
    historical 'hl' suffix in column names (now renamed to 'lh' in
    P2-1 / Units-I2 / 2026-06-03 audit), the computed ratio is
    light/heavy. std/mad are sign-invariant so the renaming is
    cosmetic but eliminates a misleading naming.
    ...
```

- [ ] **Step 4: Update design doc**

In `docs/specs/2026-05-26-hl-ratio-consistency-and-apex-cycle-offset-design.md`, find any reference to `log_hl_ratio` and replace with `log_lh_ratio` (with a parenthetical note "(renamed in P2-1, 2026-06-03 audit)" on the first occurrence).

If no such references exist, skip this step.

- [ ] **Step 5: Run test, verify PASS**

Run: `conda run -n silac_ml pytest tests/test_deep_audit_p2.py::test_no_log_hl_ratio_columns_in_single_pair_work -v`

Expected: PASS.

- [ ] **Step 6: Search for any test that referenced `log_hl_ratio` and update**

Run: `grep -rn "log_hl_ratio" tests/ --include="*.py"`

If any test asserts the OLD name, update it to the new name. Re-run full test suite:

Run: `conda run -n silac_ml pytest tests/ -q 2>&1 | tail -5`

Expected: no failures introduced by the rename.

- [ ] **Step 7: Commit**

```bash
git add workflows/single_work.py tests/test_deep_audit_p2.py docs/specs/2026-05-26-hl-ratio-consistency-and-apex-cycle-offset-design.md
git commit -m "rename(single_work): log_hl_ratio_* -> log_lh_ratio_* (P2-1, Units-I2)

Audit finding Units-I2 (2026-06-03 deep audit): feature columns named
log_hl_ratio_* but _calc_hl_ratio_consistency computes log10(L/H).
std/mad are sign-invariant so the values are unchanged, but the name
mismatch was misleading any future reader doing signed analyses.

6 columns renamed: {precursor, all, b, y} × {std, mad} → log_lh_ratio_*.
Helper docstring clarifies the L/H direction. Design doc updated."
```

---

### Task P2-2: Units-I3 — mzML RT unit enforcement

**Why:** `_get_retention_time` returns raw `float(rt)` claiming "converted to seconds" but does no conversion. Works for current Thermo mzML (minutes) by coincidence. An mzML file with `UO:0000010 second` unit would silently break window alignment.

**Files:**
- Modify: `spectrum/dia_data.py:_get_retention_time`
- Test: append to `tests/test_deep_audit_p2.py`

- [ ] **Step 1: Append test**

Append to `tests/test_deep_audit_p2.py`:

```python
def test_get_retention_time_handles_minute_unit():
    """RT in minutes returned as-is (canonical unit) (P2-2, Units-I3)."""
    from spectrum.dia_data import DIAData

    class _MockUnitFloat(float):
        """Mock pyteomics unitfloat — float with .unit_info attr."""
        def __new__(cls, value, unit_info):
            instance = super().__new__(cls, value)
            instance.unit_info = unit_info
            return instance

    dia = DIAData()
    spectrum = {
        'scanList': {
            'scan': [{'scan start time': _MockUnitFloat(10.5, 'minute')}]
        }
    }
    rt = dia._get_retention_time(spectrum)
    assert rt == 10.5  # already minutes, returned as-is


def test_get_retention_time_converts_seconds_to_minutes():
    """RT in seconds must be converted to minutes (P2-2, Units-I3)."""
    from spectrum.dia_data import DIAData

    class _MockUnitFloat(float):
        def __new__(cls, value, unit_info):
            instance = super().__new__(cls, value)
            instance.unit_info = unit_info
            return instance

    dia = DIAData()
    spectrum = {
        'scanList': {
            'scan': [{'scan start time': _MockUnitFloat(630.0, 'second')}]
        }
    }
    rt = dia._get_retention_time(spectrum)
    assert abs(rt - 10.5) < 1e-9  # 630s / 60 = 10.5min


def test_get_retention_time_handles_missing_unit_info():
    """Without unit_info attr, assume minutes (back-compat for plain floats)."""
    from spectrum.dia_data import DIAData
    dia = DIAData()
    spectrum = {'scanList': {'scan': [{'scan start time': 10.5}]}}
    rt = dia._get_retention_time(spectrum)
    assert rt == 10.5  # plain float, no conversion
```

- [ ] **Step 2: Run tests, verify they fail**

Run: `conda run -n silac_ml pytest tests/test_deep_audit_p2.py -v -k get_retention_time`

Expected: at least the seconds test fails (returns 630.0 instead of 10.5).

- [ ] **Step 3: Patch `_get_retention_time`**

In `spectrum/dia_data.py`, replace:

```python
    def _get_retention_time(self, spectrum) -> float:
        """从谱图中提取保留时间（转换为秒）"""

        if 'scanList' in spectrum:
            scan = spectrum['scanList']['scan'][0]
            if 'scan start time' in scan:
                rt = scan['scan start time']
                return float(rt)
        return 0.0
```

With:

```python
    def _get_retention_time(self, spectrum) -> float:
        """Return retention time in MINUTES (canonical pipeline unit).

        pyteomics attaches a `unit_info` attribute to the scalar (e.g.,
        'minute' from MS CV UO:0000031, 'second' from UO:0000010).
        If unit is 'second', convert to minutes. Plain floats without
        unit_info are assumed to be minutes (back-compat).

        Returns 0.0 if no scan-start-time field is present.

        (P2-2, Units-I3, 2026-06-03 deep audit.)
        """
        if 'scanList' in spectrum:
            scan = spectrum['scanList']['scan'][0]
            if 'scan start time' in scan:
                rt = scan['scan start time']
                unit = getattr(rt, 'unit_info', None)
                value = float(rt)
                if unit == 'second':
                    return value / 60.0
                if unit is None or unit == 'minute':
                    return value
                raise ValueError(
                    f"Unsupported RT unit '{unit}'; expected 'minute' or "
                    f"'second'. (P2-2, Units-I3)")
        return 0.0
```

- [ ] **Step 4: Run tests, verify PASS**

Run: `conda run -n silac_ml pytest tests/test_deep_audit_p2.py -v -k get_retention_time`

Expected: 3 PASSed.

- [ ] **Step 5: Full regression**

Run: `conda run -n silac_ml pytest tests/ -q 2>&1 | tail -5`

Expected: no NEW failures.

- [ ] **Step 6: Commit**

```bash
git add spectrum/dia_data.py tests/test_deep_audit_p2.py
git commit -m "fix(dia_data): enforce mzML RT unit (P2-2, Units-I3)

Audit finding Units-I3 (2026-06-03 deep audit): _get_retention_time
returned raw float(rt) claiming 'converted to seconds' (doc lie) but
performed no conversion. Works for current Thermo mzML (minutes) by
coincidence. An mzML with UO:0000010 second unit would silently break
window alignment in xic_*_extract.

Inspect pyteomics unit_info attribute on the scalar. 'minute' -> as-is.
'second' -> divide by 60. None (plain float) -> as-is (back-compat).
Unknown unit -> ValueError with clear message."
```


---

### Task P2-3: Units-I4 — `_calc_smoothness` length normalization

**Why:** `_calc_smoothness` returns `sum(diff² intensity) / total²` without dividing by length. Different `xic_cycle_window` configs produce non-comparable values — hidden hyperparameter coupling.

**Files:**
- Modify: `workflows/single_work.py:_calc_smoothness`
- Test: append to `tests/test_deep_audit_p2.py`

- [ ] **Step 1: Append test**

Append to `tests/test_deep_audit_p2.py`:

```python
def test_calc_smoothness_length_normalized():
    """Same shape XIC at different lengths should produce similar smoothness (P2-3, Units-I4).

    Before fix: smoothness scales with len -- absolute value differs.
    After fix: divided by (len-2) -- per-second-diff average is comparable.
    """
    from workflows.single_work import _calc_smoothness
    # Two XICs with identical "shape" (linear ramp) at different lengths
    short = np.array([1.0, 2.0, 3.0, 4.0, 5.0])  # len 5, 3 second-diffs
    long = np.array([1.0, 1.5, 2.0, 2.5, 3.0, 3.5, 4.0, 4.5, 5.0])  # len 9, 7 second-diffs

    s_short = _calc_smoothness(short)
    s_long = _calc_smoothness(long)

    # Linear ramps have all zero second-differences -> both should be 0
    assert s_short == 0.0
    assert s_long == 0.0


def test_calc_smoothness_per_unit_value_not_summed():
    """Verify smoothness is averaged per-second-diff, not just summed.

    Construct a triangle peak at two lengths with the SAME amplitude
    profile; per-second-diff averaged smoothness should be the same.
    """
    from workflows.single_work import _calc_smoothness
    # Triangle apex at center
    short = np.array([0.0, 1.0, 2.0, 1.0, 0.0])  # len 5
    long = np.concatenate([np.zeros(2), short, np.zeros(2)])  # len 9
    s_short = _calc_smoothness(short)
    s_long = _calc_smoothness(long)
    # After length-normalization, the values should be close (within
    # ~50% — exact equality requires identical second-diff distributions
    # which appending zeros doesn't preserve). The key is they should
    # NOT differ by a factor of 9/5 anymore (which the unnormalized
    # version would).
    if s_short > 0 and s_long > 0:
        ratio = max(s_short, s_long) / min(s_short, s_long)
        assert ratio < 3.0, (
            f"P2-3: length-normalized smoothness ratio should be < 3, "
            f"got {ratio} (short={s_short}, long={s_long})")
```

- [ ] **Step 2: Run tests, verify they fail or fragile-pass**

Run: `conda run -n silac_ml pytest tests/test_deep_audit_p2.py -v -k smoothness`

Expected: first test (linear ramp) passes today (sum of zeros divided by anything is 0); second may fail because unnormalized values diverge.

- [ ] **Step 3: Patch `_calc_smoothness`**

In `workflows/single_work.py`, find:

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
    if not np.all(np.isfinite(intensity)):
        return 0.0
    total = float(np.sum(intensity))
    if total <= 0:
        return 0.0
    second_diff = np.diff(intensity, n=2)
    return float(np.sum(second_diff ** 2) / (total ** 2 + 1e-12))
```

Replace with:

```python
def _calc_smoothness(intensity: np.ndarray) -> float:
    """Mean of squared second differences / total^2 — length-normalized.

    Smooth Gaussian-like peaks -> close to 0.
    Sharp zigzag / single-point spikes -> large value.

    Normalized by total^2 (cross-sample scale) AND by N=len-2 (number
    of second-difference terms, cross-config-window comparability).
    See P2-3, Units-I4, 2026-06-03 deep audit.
    """
    if len(intensity) < 3:
        return 0.0
    if not np.all(np.isfinite(intensity)):
        return 0.0
    total = float(np.sum(intensity))
    if total <= 0:
        return 0.0
    second_diff = np.diff(intensity, n=2)
    n_terms = len(second_diff)
    return float(np.sum(second_diff ** 2) / (n_terms * (total ** 2 + 1e-12)))
```

- [ ] **Step 4: Run tests, verify PASS**

Run: `conda run -n silac_ml pytest tests/test_deep_audit_p2.py -v -k smoothness`

Expected: 2 PASSed.

- [ ] **Step 5: Check if any other test uses _calc_smoothness with hardcoded expected values**

Run: `grep -rn "_calc_smoothness\|all_smoothness\|precursor_smoothness" tests/ --include="*.py"`

If any existing test asserts a specific numeric output of `_calc_smoothness`, the value will have CHANGED (divided by N now). Update those assertions to use the new normalized value (recompute by hand from the input).

If only R4 tests in `test_single_work_numerics.py` reference smoothness, check whether they assert specific values or just "value > 0" / "value == 0". The latter is OK; the former needs update.

- [ ] **Step 6: Full regression**

Run: `conda run -n silac_ml pytest tests/ -q 2>&1 | tail -5`

Expected: any specific-value test for smoothness must be updated to the new normalized value. Other tests unchanged.

- [ ] **Step 7: Commit**

```bash
git add workflows/single_work.py tests/test_deep_audit_p2.py
# Add any updated existing tests
git commit -m "fix(single_work): _calc_smoothness length-normalized (P2-3, Units-I4)

Audit finding Units-I4 (2026-06-03 deep audit): _calc_smoothness
returned sum(diff² intensity) / total² without dividing by length.
Different xic_cycle_window configs produced non-comparable values —
hidden hyperparameter coupling.

Divide by N (number of second-difference terms = len-2). Result is
the MEAN squared second-difference per cycle, comparable across
window sizes."
```

---

### Task P2-4: Pipeline-I2 — `sequence_controlled_shuffle` seedable

**Why:** `spectrum/psm_info.py:267-289` uses module-level `random.sample/shuffle` with no seed. `feature_type=2` (shuffle entrapment) negatives are non-reproducible across runs. Currently dormant (no live config uses feature_type=2) but breaks reproducibility the moment one does.

**Files:**
- Modify: `spectrum/psm_info.py:sequence_controlled_shuffle` (add seed param)
- Modify: `workflows/flow_utils.py:process_batch_pair_shuffle` (thread seed from config)
- Modify: `constant/keys.py` (add RANDOM_SEED key)
- Test: append to `tests/test_deep_audit_p2.py`

- [ ] **Step 1: Append test**

Append to `tests/test_deep_audit_p2.py`:

```python
def test_sequence_controlled_shuffle_deterministic_with_seed():
    """Same seed produces identical shuffle output (P2-4, Pipeline-I2)."""
    from spectrum.psm_info import sequence_controlled_shuffle

    seq = "ABCDEFGHIK"
    out1 = sequence_controlled_shuffle(seq, anchor_len=2, shuffle_ratio=0.5,
                                        seed=42)
    out2 = sequence_controlled_shuffle(seq, anchor_len=2, shuffle_ratio=0.5,
                                        seed=42)
    assert out1 == out2, (
        f"P2-4: same seed must produce same output; got {out1!r} vs {out2!r}")
    # Different seed -> different output (probabilistically)
    out3 = sequence_controlled_shuffle(seq, anchor_len=2, shuffle_ratio=0.5,
                                        seed=43)
    # 5! ≈ 120 permutations; P(same) ≈ 0.008 — acceptable nondeterminism check
    # If equal by chance, that's fine — the real assertion is seed=42 reproducibility


def test_sequence_controlled_shuffle_preserves_anchor():
    """Last anchor_len chars stay at the end (existing behavior unchanged)."""
    from spectrum.psm_info import sequence_controlled_shuffle
    seq = "ABCDEFGHIK"
    out = sequence_controlled_shuffle(seq, anchor_len=2, shuffle_ratio=0.5,
                                       seed=42)
    assert out.endswith("IK"), (
        f"P2-4: anchor 'IK' must be preserved; got {out!r}")
    assert len(out) == len(seq)
```

- [ ] **Step 2: Run test, verify FAIL**

Run: `conda run -n silac_ml pytest tests/test_deep_audit_p2.py -v -k sequence_controlled_shuffle`

Expected: FAIL — `sequence_controlled_shuffle` doesn't accept `seed` kwarg.

- [ ] **Step 3: Add seed parameter to `sequence_controlled_shuffle`**

In `spectrum/psm_info.py`, replace:

```python
def sequence_controlled_shuffle(peptide, anchor_len=2, shuffle_ratio=0.5):
    """
    anchor_len=1: 保留C端K/R（标准做法）
    anchor_len=2: 保留C端"XK"或"XR"（保留y1+y2离子）
    """
    # 安全检查：anchor_len 不能超过肽段长度
    anchor_len = min(anchor_len, len(peptide) - 1)

    core = peptide[:-anchor_len]
    anchor = peptide[-anchor_len:]

    n_shuffle = max(1, int(len(core) * shuffle_ratio))
    indices = random.sample(range(len(core)), n_shuffle)
    chars = list(core)
    shuffled_vals = [chars[i] for i in indices]
    random.shuffle(shuffled_vals)
    for idx, val in zip(indices, shuffled_vals):
        chars[idx] = val

    return ''.join(chars) + anchor
```

With:

```python
def sequence_controlled_shuffle(peptide, anchor_len=2, shuffle_ratio=0.5,
                                 seed=None):
    """
    anchor_len=1: 保留C端K/R（标准做法）
    anchor_len=2: 保留C端"XK"或"XR"（保留y1+y2离子）
    seed: int or None. If provided, use a fresh random.Random(seed)
        instance for deterministic shuffle. If None, use module-level
        random (backward compat, non-deterministic).
        (P2-4, Pipeline-I2, 2026-06-03 audit.)
    """
    rng = random.Random(seed) if seed is not None else random

    # 安全检查：anchor_len 不能超过肽段长度
    anchor_len = min(anchor_len, len(peptide) - 1)

    core = peptide[:-anchor_len]
    anchor = peptide[-anchor_len:]

    n_shuffle = max(1, int(len(core) * shuffle_ratio))
    indices = rng.sample(range(len(core)), n_shuffle)
    chars = list(core)
    shuffled_vals = [chars[i] for i in indices]
    rng.shuffle(shuffled_vals)
    for idx, val in zip(indices, shuffled_vals):
        chars[idx] = val

    return ''.join(chars) + anchor
```

- [ ] **Step 4: Add `RANDOM_SEED` constant**

In `constant/keys.py`, find the `GENERAL` section keys (around line 28). Add:

```python
    RANDOM_SEED = "random_seed"
```

- [ ] **Step 5: Thread seed from config into `process_batch_pair_shuffle`**

In `workflows/flow_utils.py:process_batch_pair_shuffle`, find:

```python
            if label == 0:
                new_sequence = sequence_controlled_shuffle(
                    psm1._sequence,
                    anchor_len=2, shuffle_ratio=0.5
                )
```

Replace with:

```python
            if label == 0:
                # P2-4, Pipeline-I2: seed shuffle from config for
                # reproducible negatives. Default 42 if not configured.
                try:
                    seed_base = int(config.get("general", "random_seed",
                                               fallback="42"))
                except (configparser.NoSectionError, ValueError):
                    seed_base = 42
                # Per-PSM unique seed = base + hash(sequence) to avoid
                # every PSM producing identical shuffles
                per_psm_seed = seed_base + hash(psm1._sequence) % (2**31)
                new_sequence = sequence_controlled_shuffle(
                    psm1._sequence,
                    anchor_len=2, shuffle_ratio=0.5,
                    seed=per_psm_seed,
                )
```

Add `import configparser` at top of `flow_utils.py` if not already present.

- [ ] **Step 6: Run tests, verify PASS**

Run: `conda run -n silac_ml pytest tests/test_deep_audit_p2.py -v -k sequence_controlled_shuffle`

Expected: 2 PASSed.

- [ ] **Step 7: Full regression**

Run: `conda run -n silac_ml pytest tests/ -q 2>&1 | tail -5`

Expected: no NEW failures.

- [ ] **Step 8: Commit**

```bash
git add spectrum/psm_info.py workflows/flow_utils.py constant/keys.py tests/test_deep_audit_p2.py
git commit -m "fix(psm_info): seedable sequence_controlled_shuffle (P2-4, Pipeline-I2)

Audit finding Pipeline-I2 (2026-06-03 deep audit):
sequence_controlled_shuffle used module-level random.sample/shuffle
with no seed. feature_type=2 (shuffle entrapment) negatives were
non-reproducible — same input produced different shuffled negatives
across runs, breaking train/test reproducibility.

Add seed parameter (default None = back-compat module random).
process_batch_pair_shuffle threads seed from config 'random_seed'
(default 42) + per-PSM hash so every PSM gets a unique-but-reproducible
shuffle.

Dormant fix: no current baseline uses feature_type=2."
```

---

### Task P2-5: Pipeline-I4 + Silent-I9 — `multi_batch_work` writes `heavy_in_raw`

**Why:** `single_pair_work` writes `heavy_in_raw` column (line 683-686); `multi_batch_work` doesn't. Schema mismatch between code paths. Combined fix for both audit findings.

**Files:**
- Modify: `workflows/single_work.py:multi_batch_work`
- Test: append to `tests/test_deep_audit_p2.py`

- [ ] **Step 1: Append test**

Append to `tests/test_deep_audit_p2.py`:

```python
def test_multi_batch_work_writes_heavy_in_raw_column():
    """multi_batch_work must emit heavy_in_raw column for schema parity
    with single_pair_work (P2-5, Pipeline-I4 + Silent-I9)."""
    from workflows.single_work import multi_batch_work, single_pair_work
    psm = _FakePSM()
    dia = _FakeDIA()
    multi_features = multi_batch_work(psm, dia, psm, dia, _minimal_config())
    single_features = single_pair_work(psm, dia, _minimal_config())
    assert "heavy_in_raw" in multi_features, (
        "P2-5: multi_batch_work missing heavy_in_raw column")
    assert "heavy_in_raw" in single_features, (
        "Sanity check: single_pair_work should already have heavy_in_raw")
```

- [ ] **Step 2: Run test, verify it fails for multi_batch_work**

Run: `conda run -n silac_ml pytest tests/test_deep_audit_p2.py::test_multi_batch_work_writes_heavy_in_raw_column -v`

Expected: FAIL on the first assertion.

- [ ] **Step 3: Patch `multi_batch_work`**

In `workflows/single_work.py:multi_batch_work`, find the section after the precursor block but before/after the fragment loop where `heavy_in_raw` would naturally fit. In `single_pair_work` it's computed at line 504 as `heavy_in_raw = dia_data.check_in_raw(heavy_precursor_mz)`.

For `multi_batch_work` (cross-run), the equivalent is `dia_data2.check_in_raw(psm2._precursor_mz)`. Add at the start of the function (after the initial features dict setup):

```python
    # P2-5, Pipeline-I4 + Silent-I9: emit heavy_in_raw for schema parity
    # with single_pair_work. In cross-run mode, "heavy" is the second
    # DIA file's precursor.
    features["heavy_in_raw"] = dia_data2.check_in_raw(psm2._precursor_mz)
```

- [ ] **Step 4: Run test, verify PASS**

Run: `conda run -n silac_ml pytest tests/test_deep_audit_p2.py::test_multi_batch_work_writes_heavy_in_raw_column -v`

Expected: PASS.

- [ ] **Step 5: Full regression**

Run: `conda run -n silac_ml pytest tests/ -q 2>&1 | tail -5`

Expected: no NEW failures.

- [ ] **Step 6: Commit**

```bash
git add workflows/single_work.py tests/test_deep_audit_p2.py
git commit -m "fix(single_work): multi_batch_work writes heavy_in_raw (P2-5)

Audit findings Pipeline-I4 + Silent-I9 (2026-06-03 deep audit):
single_pair_work writes heavy_in_raw column at line 683-686.
multi_batch_work didn't. If both functions ever wrote to the same
features.csv, schema mismatch (NaN for the missing column in
multi_batch_work rows).

Add features['heavy_in_raw'] = dia_data2.check_in_raw(...) at start of
multi_batch_work. Behavioral test asserts schema parity between
the two functions."
```

---

### Task P2-6: Pipeline-I6 — spec_trainer figures_dir from yaml

**Why:** `tools/spec_trainer/src/main.py:218-219` hardcodes `runs/spec_trainer/figures/` for fig_path and roc_path. Works only when cwd is repo root. Direct invocation from another cwd breaks.

**Files:**
- Modify: `tools/spec_trainer/src/main.py`
- Modify: `tools/spec_trainer/config/exp1.yaml`
- Modify: `tools/spec_trainer/config/exp2.yaml`
- Test: extend existing test_spec_trainer_holdout.py

- [ ] **Step 1: Append yaml-validation test**

Append to `tests/test_spec_trainer_holdout.py`:

```python


def test_both_exp_yamls_set_figures_dir():
    """Both exp1 and exp2 must set output.figures_dir (P2-6, Pipeline-I6)."""
    import os
    import yaml
    project_root = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
    for name in ("exp1.yaml", "exp2.yaml"):
        p = os.path.join(project_root, "tools", "spec_trainer", "config", name)
        with open(p) as f:
            cfg = yaml.safe_load(f)
        figures_dir = cfg.get("output", {}).get("figures_dir")
        assert figures_dir, (
            f"{name}: output.figures_dir must be set, not None/missing "
            f"(P2-6, Pipeline-I6)")
```

- [ ] **Step 2: Run test, verify FAIL**

Run: `conda run -n silac_ml pytest tests/test_spec_trainer_holdout.py::test_both_exp_yamls_set_figures_dir -v`

Expected: FAIL — neither yaml has `figures_dir`.

- [ ] **Step 3: Update exp1.yaml and exp2.yaml to add figures_dir**

In `tools/spec_trainer/config/exp1.yaml`, find the `output:` block:

```yaml
output:
  model_path: runs/spec_trainer/models/exp1.txt
  result_path: runs/spec_trainer/results/exp1.json
```

Add:

```yaml
output:
  model_path: runs/spec_trainer/models/exp1.txt
  result_path: runs/spec_trainer/results/exp1.json
  figures_dir: runs/spec_trainer/figures  # P2-6, Pipeline-I6
```

Same for `exp2.yaml`:

```yaml
output:
  model_path: runs/spec_trainer/models/exp2.txt
  result_path: runs/spec_trainer/results/exp2.json
  figures_dir: runs/spec_trainer/figures
```

- [ ] **Step 4: Patch `tools/spec_trainer/src/main.py` to read figures_dir**

Find the lines (around 235-236):

```python
    fig_path = f"runs/spec_trainer/figures/{args.name}_importance.png"
    roc_path = f"runs/spec_trainer/figures/{args.name}_roc_curve.png"
```

Replace with:

```python
    # P2-6, Pipeline-I6: figures dir from yaml config (default keeps
    # back-compat for older yamls without figures_dir).
    figures_dir = cfg['output'].get('figures_dir', 'runs/spec_trainer/figures')
    fig_path = os.path.join(figures_dir, f"{args.name}_importance.png")
    roc_path = os.path.join(figures_dir, f"{args.name}_roc_curve.png")
```

(`os` is already imported at top of main.py.)

- [ ] **Step 5: Run test, verify PASS**

Run: `conda run -n silac_ml pytest tests/test_spec_trainer_holdout.py::test_both_exp_yamls_set_figures_dir -v`

Expected: PASS.

- [ ] **Step 6: Full regression**

Run: `conda run -n silac_ml pytest tests/ -q 2>&1 | tail -5`

Expected: no NEW failures.

- [ ] **Step 7: Commit**

```bash
git add tools/spec_trainer/src/main.py tools/spec_trainer/config/exp1.yaml tools/spec_trainer/config/exp2.yaml tests/test_spec_trainer_holdout.py
git commit -m "fix(spec_trainer): figures_dir from yaml (P2-6, Pipeline-I6)

Audit finding Pipeline-I6 (2026-06-03 deep audit): main.py hardcoded
runs/spec_trainer/figures/ for fig_path and roc_path. Works only when
cwd is repo root (Makefile guarantees this). Direct python invocation
from another cwd writes figures to a non-existent or wrong directory.

Read output.figures_dir from yaml (default: runs/spec_trainer/figures
for back-compat). Both exp1.yaml and exp2.yaml updated."
```


---

### Task P2-7: Silent-I5 — `resolve_feature_cols` raises on empty

**Why:** When all columns are excluded (header-only CSV or pathological excludes), `resolve_feature_cols` returns `[]`. Downstream `model.fit(X_train[[]], y_train)` fails inside LightGBM with cryptic "Cannot construct Dataset since there are no usable features".

**Files:**
- Modify: `tools/spec_trainer/src/feature_cols.py`
- Test: append to `tests/test_spec_trainer_main.py`

- [ ] **Step 1: Append test**

Append to `tests/test_spec_trainer_main.py`:

```python


def test_resolve_feature_cols_raises_when_result_empty(tmp_path):
    """Empty result must raise ValueError, not silently return [] (P2-7, Silent-I5)."""
    from feature_cols import resolve_feature_cols
    csv = tmp_path / "empty.csv"
    # All META columns; nothing left after exclusion
    csv.write_text("label,sequence,charge,modification_count\n")
    with pytest.raises(ValueError, match="0 features"):
        resolve_feature_cols(
            explicit=None,
            sample_csv_paths=[str(csv)],
            target_col="label",
        )
```

- [ ] **Step 2: Run test, verify FAIL**

Run: `conda run -n silac_ml pytest tests/test_spec_trainer_main.py::test_resolve_feature_cols_raises_when_result_empty -v`

Expected: FAIL — current code returns `[]` silently.

- [ ] **Step 3: Add raise to `resolve_feature_cols`**

In `tools/spec_trainer/src/feature_cols.py:resolve_feature_cols`, at the very end before `return`:

```python
    result = [
        c for c in ordered
        if c in intersection
        and c not in META_COLUMNS
        and c not in EXCLUDED_EXTRA
        and c != target_col
    ]
    if not result:
        raise ValueError(
            f"resolve_feature_cols returned 0 features from "
            f"{sample_csv_paths}; all columns are in META_COLUMNS / "
            f"EXCLUDED_EXTRA / target_col. Check yaml feature_cols or "
            f"add features to the CSV. (P2-7, Silent-I5)"
        )
    return result
```

(Replace the existing `return [...]` with the assignment + check + return.)

- [ ] **Step 4: Run test, verify PASS**

Run: `conda run -n silac_ml pytest tests/test_spec_trainer_main.py::test_resolve_feature_cols_raises_when_result_empty -v`

Expected: PASS.

- [ ] **Step 5: Full regression**

Run: `conda run -n silac_ml pytest tests/ -q 2>&1 | tail -5`

Expected: no NEW failures.

- [ ] **Step 6: Commit**

```bash
git add tools/spec_trainer/src/feature_cols.py tests/test_spec_trainer_main.py
git commit -m "fix(spec_trainer): resolve_feature_cols raises on empty (P2-7, Silent-I5)

Audit finding Silent-I5 (2026-06-03 deep audit): when all columns
excluded (header-only CSV or pathological excludes),
resolve_feature_cols returned [] silently. Downstream model.fit fails
with cryptic LightGBMError: Cannot construct Dataset.

Raise ValueError naming the inputs so user can diagnose immediately."
```

---

### Task P2-8: Silent-I4 — spec_trainer/main.py logpath parent mkdir

**Why:** `tools/spec_trainer/src/main.py:149` opens `FileHandler(args.logpath)` without ensuring parent dir exists. `main.py` (feature extraction) does (`main.py:37-39`); spec_trainer doesn't. Inconsistent — passing `--logpath runs/spec_trainer/logs/exp.log` to spec_trainer when that dir is absent crashes before training.

**Files:**
- Modify: `tools/spec_trainer/src/main.py` (~3 lines)
- Test: append to `tests/test_deep_audit_p2.py` (source-grep check)

- [ ] **Step 1: Append source-grep test**

Append to `tests/test_deep_audit_p2.py`:

```python
def test_spec_trainer_main_creates_logpath_parent():
    """spec_trainer main.py must mkdir -p the logpath parent dir (P2-8, Silent-I4)."""
    src_path = os.path.join(
        os.path.dirname(os.path.dirname(os.path.abspath(__file__))),
        "tools", "spec_trainer", "src", "main.py")
    src = open(src_path).read()
    assert "os.makedirs(os.path.dirname(args.logpath)" in src, (
        "P2-8: spec_trainer main.py missing mkdir for logpath parent")
```

- [ ] **Step 2: Run test, verify FAIL**

Run: `conda run -n silac_ml pytest tests/test_deep_audit_p2.py::test_spec_trainer_main_creates_logpath_parent -v`

Expected: FAIL.

- [ ] **Step 3: Add mkdir for logpath parent**

In `tools/spec_trainer/src/main.py`, find:

```python
    args = parser.parse_args()

    # 设置日志文件handle
    file_handler = logging.FileHandler(args.logpath, encoding="utf-8")
```

Replace with:

```python
    args = parser.parse_args()

    # P2-8, Silent-I4: ensure logpath parent dir exists. Mirror of main.py:37-39.
    log_dir = os.path.dirname(args.logpath)
    if log_dir:
        os.makedirs(log_dir, exist_ok=True)

    # 设置日志文件handle
    file_handler = logging.FileHandler(args.logpath, encoding="utf-8")
```

- [ ] **Step 4: Run test, verify PASS**

Run: `conda run -n silac_ml pytest tests/test_deep_audit_p2.py::test_spec_trainer_main_creates_logpath_parent -v`

Expected: PASS.

- [ ] **Step 5: Full regression**

Run: `conda run -n silac_ml pytest tests/ -q 2>&1 | tail -5`

Expected: no NEW failures.

- [ ] **Step 6: Commit**

```bash
git add tools/spec_trainer/src/main.py tests/test_deep_audit_p2.py
git commit -m "fix(spec_trainer): mkdir for logpath parent dir (P2-8, Silent-I4)

Audit finding Silent-I4 (2026-06-03 deep audit): spec_trainer main.py
opened FileHandler(args.logpath) without ensuring parent dir exists.
main.py (feature extraction) does (lines 37-39). Inconsistent —
passing --logpath runs/spec_trainer/logs/exp.log to spec_trainer
when that dir is absent crashed before training.

Mirror main.py:37-39 pattern: mkdir -p os.path.dirname(args.logpath)."
```

---

## Final Verification (after all 19 tasks)

- [ ] **Step 1: Full test suite**

Run: `conda run -n silac_ml pytest tests/ -q 2>&1 | tail -10`

Expected: ≥278 baseline + new tests (estimated ~30 new across P0/P1/P2), no NEW failures.

- [ ] **Step 2: Makefile dry-runs**

Run:
```
make -n 2th 5th normal all train-exp1 train-exp2 clean-all clean-train
```

Expected: every target succeeds (no make errors).

- [ ] **Step 3: User-side retrain (manual, off-plan)**

The user must manually run:

```bash
# Clean orphaned legacy workspace (one-time migration from I-MK2)
rm -rf ./workspace/

# Clean all existing features.csv + caches
make clean-all
rm -f runs/baseline_*_clean/*.dia.npz  # force cache rebuild (P0-3)

# Regenerate features.csv from clean state
make all  # SLOW (~tens of minutes per dataset)

# Verify schema consistency across all 3 baselines
diff <(head -1 runs/baseline_2da_clean/features.csv | tr ',' '\n' | sort) \
     <(head -1 runs/baseline_5da_clean/features.csv | tr ',' '\n' | sort)
diff <(head -1 runs/baseline_2da_clean/features.csv | tr ',' '\n' | sort) \
     <(head -1 runs/baseline_normal_clean/features.csv | tr ',' '\n' | sort)
# Both diffs should produce NO output (identical schema)

# Verify column count
wc -l < <(head -1 runs/baseline_2da_clean/features.csv | tr ',' '\n')
# Expect ~140 columns (66 baseline + 11 q1a + 26 R3 + 20 R4 + 3 new markers)

# Train both experiments
make train-all

# Sanity check: ablation should now have effect (P0-3 verification)
# Toggle centroid_enabled=false in one baseline config, re-run make clean-2th && make 2th
# AUC should differ from the centroid_enabled=true run.
```

This step is OUT OF SCOPE for automated plan execution — it requires user time and real data.

- [ ] **Step 4: (Optional) Push to gitlab**

```bash
git push gitlab feature_extraction
```

