# Deep Audit Fixes — Critical + Important Bugs in Feature Extraction Pipeline

**Status:** Approved (2026-06-03)
**Branch:** `feature_extraction`
**Plan file:** `docs/superpowers/plans/2026-06-03-deep-audit-fixes.md` (to be created next)

---

## Background

On 2026-06-03 a 3-axis deep audit (units & semantics / end-to-end pipeline coherence / silent failures & error handling) was conducted on the ms2-met DIA-MS SILAC feature extraction pipeline at commit `db5afc3`. The audit uncovered **4 Critical + 15 Important findings**, of which several are **active bugs** silently corrupting the features.csv files currently on disk.

The user has approved fixing all Critical + Important findings (including dormant ones for `multi_batch_work`) in a single phased plan.

### Key audit evidence

- **Existing features.csv files are stale and schema-inconsistent:**
  - `runs/baseline_2da_clean/features.csv` (2026-05-19, 66 cols) — missing all 11 `q1a_*` columns
  - `runs/baseline_5da_clean/features.csv` (2026-05-20, 77 cols)
  - `runs/baseline_normal_clean/features.csv` (2026-05-20, 77 cols)
  - All three predate R3 (2026-05-26) + R4 (2026-06-02), so they lack ~46 newer columns
- After fixes land, the user will manually run `make clean-all && make all` to regenerate consistent features.csv files

### Critical findings recap

| ID | Severity | Status | Description |
|---|---|---|---|
| Silent-C1 | Critical | ACTIVE | Empty-XIC branch writes 0/0.0 indistinguishable from real signal; no marker column |
| Silent-C2 | Critical | ACTIVE | `calc_xic_score` computes "perfect coelution" (apex_delta=0) on all-zero non-empty XIC; `np.argmax` on zeros returns index 0 |
| Silent-C3 | Critical | ACTIVE | `.dia.npz` cache `_format_version=2` doesn't capture centroid params; changing config has no effect (stale cache used) |
| Units-C1 | Critical | DORMANT | `multi_batch_work:202` passes `light_mass` to heavy MS2 XIC extraction (8-10 Da off, far outside ppm tol); all cross-run heavy fragment XICs return empty |
| Pipeline-C1 | Critical (data) | ACTIVE | features.csv schema drift on disk — fixed by user re-running `make all` after the code fixes |

---

## Architecture

19 surgical fixes grouped in 3 phases, each task independently testable and committable. No cross-task dependencies except where noted. Each phase produces a working, runnable pipeline; user can stop and validate after any phase.

```
Phase 0 — Critical bugs (P0, 4 tasks)
  P0-1  Silent-C1   precursor_xic_empty marker columns
  P0-2  Silent-C2   calc_xic_score all-zero non-empty guard
  P0-3  Silent-C3   cache _format_version=3 with centroid params
  P0-4  Units-C1    multi_batch_work heavy_mass fix

Phase 1 — Active Important fixes (P1, 7 tasks)
  P1-1  Units-I1    matched_intensity_percent denominator hoist
  P1-2  Silent-I1   fragment empty-branch list parity (M6 from T1 review)
  P1-3  Pipeline-I1 label NaN guard (raise if _label_type is None)
  P1-4  Pipeline-I3 exp2.yaml is_unbalance: True
  P1-5  Pipeline-I5 resolve_feature_cols multi-file intersection + warning
  P1-6  Silent-I3   logging.warn → logging.debug + per-batch summary
  P1-7  Silent-I8   centroid_spectrum short-spectrum logger + counter

Phase 2 — Dormant Important fixes (P2, 8 tasks)
  P2-1  Units-I2    log_hl_ratio_* → log_lh_ratio_* rename
  P2-2  Units-I3    mzML RT unit enforcement
  P2-3  Units-I4    _calc_smoothness length normalization
  P2-4  Pipeline-I2 sequence_controlled_shuffle seedable
  P2-5  Pipeline-I4 + Silent-I9  multi_batch_work writes heavy_in_raw
  P2-6  Pipeline-I6 spec_trainer figures_dir from yaml
  P2-7  Silent-I5   resolve_feature_cols raises on empty result
  P2-8  Silent-I4   spec_trainer/main.py mkdir for logpath parent
```

---

## Phase 0 — Critical fixes

### P0-1: Silent-C1 — `precursor_xic_empty` marker columns

**Problem:** When XIC retrieval fails (precursor outside m/z window, no fragment matches, etc.), `single_pair_work` (lines 380-417) and `multi_batch_work` (lines 55-74) write all 20 precursor features as `0/0.0`. LightGBM cannot distinguish "extraction failed" from "computed 0".

**Fix:** Add a single boolean column `precursor_xic_empty: 0/1` written in BOTH code paths. Set to 1 when ANY of these conditions trigger the default-zero features:
1. `len(light_xic) == 0`
2. `len(heavy_xic) == 0`
3. `not np.any(light_xic["intensity"] > 0)` (all-zero light intensity — guarded in P0-2)
4. `not np.any(heavy_xic["intensity"] > 0)` (all-zero heavy intensity — guarded in P0-2)

Set to 0 otherwise. LightGBM will learn to gate on this column.

**Coordination with P0-2:** The condition for marker=1 matches the condition that P0-2 adds to short-circuit `calc_xic_score`. Extract a small helper `_is_empty_xic_pair(light_xic, heavy_xic) -> bool` (in `single_work.py`) used by both call sites and by P0-2's guard, ensuring consistency.

**For fragments:** add `fragment_xic_empty_count: int` — count of fragments where light_ions_xic or heavy_ions_xic was empty (already implicitly handled by `valid_fragment_ions_num` but make it explicit for symmetry with the per-fragment skip semantics in `single_pair_work:525-530`).

**Files modified:**
- `workflows/single_work.py` — new `_is_empty_xic_pair` helper; both empty branches and both computed branches of `single_pair_work` + `multi_batch_work` set the marker
- Test: `tests/test_single_work_numerics.py` — assert both marker columns present and correctly valued in:
  - empty branch (value 1, via fake DIA returning empty XIC)
  - computed branch (value 0, via real-ish XIC with intensity > 0)
  - all-zero non-empty case (value 1, after P0-2 lands)

**Acceptance:**
- Empty-XIC trigger: `features["precursor_xic_empty"] == 1`
- Valid XIC trigger: `features["precursor_xic_empty"] == 0`
- All-zero non-empty trigger (post P0-2): `features["precursor_xic_empty"] == 1`
- Both `single_pair_work` and `multi_batch_work` set the marker

---

### P0-2: Silent-C2 — `calc_xic_score` all-zero non-empty guard

**Problem:** `calc_xic_score` lines 1107-1146 guard against `len(xic)==0` but not against an XIC that is non-empty but has all-zero intensities (common when DIA scans exist in the time window but no fragment peak matches the m/z within ppm tolerance — `match_peak_ppm` returns `(NaN, 0.0)` → accumulated intensity stays 0). `np.argmax` on all-zeros returns index 0 → `apex_delta=0` looks like perfect coelution.

**Fix:** At the top of `calc_xic_score` (after the existing `len==0` check if any, or as first guard), use the shared `_is_empty_xic_pair` helper from P0-1:

```python
if _is_empty_xic_pair(light_xic, heavy_xic):
    return _default_xic_score()
```

`_is_empty_xic_pair` returns True if either XIC is empty OR has all-zero intensity. This routes the all-zero non-empty case to the existing default-zero-features path (which P0-1's caller-side logic will mark with `precursor_xic_empty=1`).

**Files modified:**
- `workflows/single_work.py:calc_xic_score`
- Test: `tests/test_single_work_numerics.py` — pass an XIC with `intensity = [0, 0, 0, 0, 0]` and assert the returned dict matches `_default_xic_score()` exactly (no spurious apex_delta=0 "real" computation)

**Acceptance:**
- All-zero non-empty XIC produces identical output to empty XIC
- Adds NO behavioral change for valid XICs

---

### P0-3: Silent-C3 — Cache `_format_version=3` with centroid params

**Problem:** `spectrum/dia_data.py:163-201` writes `_format_version=2` but doesn't serialize `_centroid_enabled` / `_centroid_rel_threshold`. `workflows/flow_utils.py:29-31` only checks file existence to decide rebuild. User changing config has no effect; the model trains against stale centroided data.

**Fix:**
- Bump `_format_version` to 3
- In `save_to_file`: add `_centroid_enabled` and `_centroid_rel_threshold` as npz keys
- In `_check_format_version` (or rename to `_validate_cache`): compare stored centroid params against current self params; if mismatch, log warning and signal the loader to discard + rebuild
- Old version=2 caches are automatically discarded (version mismatch already triggers rebuild)

**Files modified:**
- `spectrum/dia_data.py` — `save_to_file`, `_check_format_version`, `__init__` to capture current params for the validation
- Test: `tests/test_dia_data_window.py` — round-trip: save with params A, attempt load with params B → assert cache rejected (returns False / triggers rebuild signal)

**Acceptance:**
- Save with `centroid_rel_threshold=0.001`, attempt to load instance configured for `0.01` → rejected with clear warning
- Save and load with same params → accepted (no spurious rebuild)
- Old `_format_version=2` caches still rejected (already true)

---

### P0-4: Units-C1 — `multi_batch_work` heavy_mass fix

**Problem:** `workflows/single_work.py:199-204` (`multi_batch_work` heavy MS2 XIC extraction):
```python
heavy_ions_xic, heavy_all_intensity = dia_data2.xic_ms2_peaks_extract(
    psm2._rt, xic_cycle_window,
    precursor_mz=psm2._precursor_mz,
    ions_mass=light_mass,        # BUG: should be heavy_mass
    mass_tol_ppm=mass_tol_ppm)
```
Should be `ions_mass=heavy_mass`. Compare with `single_pair_work:541-546` which is correct.

**Fix:** One-line change `light_mass` → `heavy_mass`.

**Files modified:**
- `workflows/single_work.py:202`
- Test: `tests/test_single_work_numerics.py` — fake dia_data2 with `xic_ms2_peaks_extract` that records the `ions_mass` arg; trigger `multi_batch_work` with a fragment whose light/heavy mass differ; assert recorded `ions_mass == heavy_mass`

**Acceptance:**
- multi_batch_work passes heavy_mass for heavy XIC extraction
- single_pair_work behavior unchanged (was already correct)

---

## Phase 1 — Active Important fixes

### P1-1: Units-I1 — `matched_intensity_percent` denominator

**Problem:** `single_work.py:225-226, 301` (single_pair_work also at 566-567, 637): the denominator `intensitys_map["all"]` accumulates `light_all_intensity + heavy_all_intensity` inside the per-fragment loop. Since `*_all_intensity` is per-PSM (not per-fragment), it's added N_fragments times. Result: `matched_intensity_percent ∝ 1/N_fragments` — a hidden peptide-length proxy.

**Fix:** Hoist the `intensitys_map["all"]` initialization to use `light_all_intensity + heavy_all_intensity` ONCE per PSM, NOT inside the fragment loop. Per-fragment numerators stay where they are.

**Files modified:**
- `workflows/single_work.py` — both `single_pair_work` and `multi_batch_work` fragment loops
- Test: `tests/test_single_work_numerics.py` — fake PSM with 5 fragments → assert denominator independent of fragment count

---

### P1-2: Silent-I1 — Fragment empty-branch list parity (M6 from T1 review unresolved)

**Problem:** When `len(light_ions_xic)==0 or len(heavy_ions_xic)==0` in the per-fragment loop, only 4 lists get appended (`pearsons_map`, `fragment_intensities`, `fragment_cosines`, `fragment_snrs`). The other ~11 per-fragment lists (`fragment_apex_deltas`, `fragment_*_cycle_offsets[_signed]`, `fragment_hl_ratios`, `fragment_base_to_apex_ratios`, `fragment_apex_monotonicities`, `fragment_n_peaks_list`, `fragment_smoothnesses`, `fragment_mz_errs`) get NO entry. Aggregates (`all_*_mean/p50/std/max`) for those features are computed over a strictly smaller list than `valid_fragment_ions_num` implies.

**Fix:** In the empty-XIC fragment branch, append default zeros to ALL per-fragment lists. Update both `single_pair_work` and `multi_batch_work`.

**Files modified:**
- `workflows/single_work.py` — fragment empty branches in both functions
- Test: `tests/test_single_work_numerics.py` — trigger empty fragment branch and assert all per-fragment lists have the same length

---

### P1-3: Pipeline-I1 — Label NaN guard

**Problem:** `workflows/flow_utils.py:88` maps `_label_type=None → label=None`. CSV writes empty field → `pd.read_csv` infers `float64` with NaN. LightGBM `objective: binary` crashes on NaN labels. Currently dormant because all `extract_*.ini` set `positive_species_marker`, but the `intersection_keys` "no-marker mode" at `tools/extract_common.py:127-135` would produce NaN labels for every row.

**Fix:** In `_make_result_row_single` (and equivalent in pair flow), raise `ValueError` if `_label_type is None`, with a clear message pointing at the extract_common config.

**Files modified:**
- `workflows/flow_utils.py:_make_result_row_single` (and other writers)
- Test: `tests/test_flow_utils.py` (may need creating) — assert raises on None label_type

---

### P1-4: Pipeline-I3 — `exp2.yaml` is_unbalance

**Problem:** `exp1.yaml` sets `is_unbalance: True` (data is ~1% positives); `exp2.yaml` doesn't. exp2 trains on the same imbalanced datasets but produces miscalibrated probabilities.

**Fix:** Add `is_unbalance: True` to `exp2.yaml` model.params block.

**Files modified:**
- `tools/spec_trainer/config/exp2.yaml`
- Test: extend `tests/test_spec_trainer_holdout.py::test_exp_yamls_do_not_have_in_sample_test_files` to also assert `is_unbalance: True` for both yamls (or accept reasoning if explicitly set False with a comment)

---

### P1-5: Pipeline-I5 — `resolve_feature_cols` multi-file intersection

**Problem:** `tools/spec_trainer/src/feature_cols.py:resolve_feature_cols` reads column names from `sample_csv_path` (which main.py passes as `train_files[0]`). Other train_files' columns are silently dropped at `load_data` (`df[feature_cols]` returns only the intersection).

**Fix:** Extend `resolve_feature_cols` signature to accept `sample_csv_paths: list[str]` (plural). Compute intersection of all CSV headers. Log a warning naming the dropped columns when intersection ≠ first file's columns.

Update `main.py` call site to pass all `train_files`.

**Files modified:**
- `tools/spec_trainer/src/feature_cols.py`
- `tools/spec_trainer/src/main.py` call site
- Test: `tests/test_spec_trainer_main.py` — add test for multi-file intersection (drop test of "auto-detect from single CSV" if needed adapt)

---

### P1-6: Silent-I3 — `logging.warn` per-PSM dumps

**Problem:** `spectrum/dia_data.py:524-525, 703-712` uses deprecated `logging.warn` per PSM × fragment, dumping `_cycle_left_precursor` array each time. Megabytes of warnings into `extract.log`; effectively silent.

**Fix:**
- Replace `logging.warn` → `logging.debug`
- Add a counter to DIAData (`self._n_out_of_window_xic`) incremented per occurrence
- In `pair_flow.py` batch summary or at DIAData destructor / pair_flow batch end, log once: `logging.info(f"[summary] {n} XIC requests fell outside m/z window")`

**Files modified:**
- `spectrum/dia_data.py`
- `workflows/pair_flow.py` (batch summary location)

---

### P1-7: Silent-I8 — `centroid_spectrum` short-spectrum logger

**Problem:** `spectrum/spectrum_utils.py:60-82` returns empty result for spectra with <3 peaks or all-zero intensity. No log. Downstream XIC sees zero → fake "no signal".

**Fix:**
- Pass a counter through `_load_from_mzml` and increment when `centroid_spectrum` returns empty unexpectedly
- Log a single summary line per file load: `logging.info(f"[centroid] {n}/{total} spectra returned empty (likely <3 peaks)")`

**Files modified:**
- `spectrum/spectrum_utils.py` (return value or side-effect counter)
- `spectrum/dia_data.py:_load_from_mzml` (collect and log summary)

---

## Phase 2 — Dormant Important fixes

### P2-1: Units-I2 — `log_hl_ratio_*` → `log_lh_ratio_*` rename

**Problem:** `_calc_hl_ratio_consistency` computes `log10(light/heavy)` but the feature columns are named `log_hl_ratio_*` (H/L). Std/mad are sign-invariant so functionally OK, but misleading.

**Fix:** Rename 6 columns: `precursor_log_hl_ratio_std/mad`, `all_log_hl_ratio_std/mad`, `b_log_hl_ratio_std/mad`, `y_log_hl_ratio_std/mad` → replace `hl` with `lh`. Update all writers in `single_work.py`. Update spec doc 2026-05-26-hl-ratio-consistency-and-apex-cycle-offset-design.md if needed.

**Files modified:**
- `workflows/single_work.py` — all 6 column writers
- `docs/specs/2026-05-26-hl-ratio-consistency-and-apex-cycle-offset-design.md` (documentation alignment)

---

### P2-2: Units-I3 — mzML RT unit enforcement

**Problem:** `spectrum/dia_data.py:_get_retention_time` returns raw `float(rt)` claiming "converted to seconds" but does no conversion. Works for current Thermo mzML (minutes) by coincidence; mzML files with `UO:0000010 second` unit would silently break window alignment.

**Fix:** In `_get_retention_time`, inspect `rt.unit_info` (pyteomics attaches it). If `'minute'`, return as-is. If `'second'`, divide by 60 (convert to minutes — the rest of the pipeline assumes minutes). If unknown, raise `ValueError` with clear context.

Update docstring.

**Files modified:**
- `spectrum/dia_data.py:_get_retention_time`
- Test: `tests/test_dia_data_window.py` or new test — fake spectrum with each unit_info; assert correct conversion / raise

---

### P2-3: Units-I4 — `_calc_smoothness` length normalization

**Problem:** `_calc_smoothness` returns sum of |Δ²intensity| but doesn't normalize by length. Different `xic_cycle_window` configs produce non-comparable values.

**Fix:** Divide by `len(intensity) - 2` (number of second-difference terms). Update docstring. Update R4 spec doc.

**Files modified:**
- `workflows/single_work.py:_calc_smoothness`
- Test: `tests/test_single_work_numerics.py` — assert same XIC with window 5 vs window 9 produces same smoothness (within length scaling)

---

### P2-4: Pipeline-I2 — `sequence_controlled_shuffle` seedable

**Problem:** `spectrum/psm_info.py:267-289` uses module-level `random.sample/shuffle` with no seed. For `feature_type=2` (shuffle entrapment) runs, negatives are non-reproducible.

**Fix:** Add `seed` parameter to `sequence_controlled_shuffle`, default None (current behavior). Thread through callers from config `random_seed` (default 42).

**Files modified:**
- `spectrum/psm_info.py:sequence_controlled_shuffle`
- `workflows/pair_flow.py` (and any other caller chain)
- `constant/keys.py` add `RANDOM_SEED` key
- Test: `tests/test_psm_shuffle.py` (may need new) — same input + seed produces deterministic output

---

### P2-5: Pipeline-I4 + Silent-I9 — `multi_batch_work` writes `heavy_in_raw`

**Problem:** `single_pair_work` writes `heavy_in_raw` column (line 683-686) but `multi_batch_work` doesn't. If both code paths' outputs ever land in the same training table, schema mismatch.

**Fix:** In `multi_batch_work`, after computing whether `psm2._precursor_mz` is in `dia_data2`'s raw window, write `features["heavy_in_raw"] = bool/int`.

**Files modified:**
- `workflows/single_work.py:multi_batch_work`
- Test: extend behavioral parity test added in T1 to also check `heavy_in_raw` is present in both

---

### P2-6: Pipeline-I6 — spec_trainer figures_dir from yaml

**Problem:** `tools/spec_trainer/src/main.py:218-219` hardcodes `runs/spec_trainer/figures/` for fig_path and roc_path. Works only when cwd is repo root (Makefile guarantees this), breaks for direct invocation from another cwd.

**Fix:** Read from `cfg['output'].get('figures_dir', 'runs/spec_trainer/figures')`. Construct fig_path and roc_path from that base.

Update `exp1.yaml` and `exp2.yaml` to set `output.figures_dir: runs/spec_trainer/figures` (optional documentation, since default is the same).

**Files modified:**
- `tools/spec_trainer/src/main.py`
- `tools/spec_trainer/config/exp1.yaml` (add `figures_dir`)
- `tools/spec_trainer/config/exp2.yaml` (add `figures_dir`)

---

### P2-7: Silent-I5 — `resolve_feature_cols` raises on empty

**Problem:** When all columns are excluded (header-only CSV or pathological exclude lists), `resolve_feature_cols` returns `[]`. Downstream `model.fit(X_train[[]], y_train)` fails inside LightGBM with `Cannot construct Dataset since there are no usable features`.

**Fix:** Raise `ValueError(f"resolve_feature_cols returned 0 features from {sample_csv_paths}; all columns are in META/EXCLUDED?")` when result is empty.

**Files modified:**
- `tools/spec_trainer/src/feature_cols.py`
- Test: `tests/test_spec_trainer_main.py` — add test for header-only CSV (only META columns) → raises

---

### P2-8: Silent-I4 — spec_trainer/main.py mkdir for logpath parent

**Problem:** `tools/spec_trainer/src/main.py:149` opens `FileHandler(args.logpath)` without ensuring parent dir exists. `main.py` (the feature extraction main) does ensure (`main.py:37-39`); spec_trainer doesn't. Inconsistent.

**Fix:** Mirror the mkdir pattern from main.py:37-39 before constructing FileHandler.

**Files modified:**
- `tools/spec_trainer/src/main.py` (~3 lines)

---

## features.csv Schema Changes Summary

After all 19 fixes land + user runs `make clean-all && make all`:

| Change Type | Columns |
|---|---|
| **New** | `precursor_xic_empty` (P0-1), `fragment_xic_empty_count` (P0-1), `heavy_in_raw` in multi_batch_work output (P2-5) |
| **Renamed** | 6 × `log_hl_ratio_*` → `log_lh_ratio_*` (P2-1) |
| **Value semantics changed** | `matched_intensity_percent` denominator no longer × N_fragments (P1-1); `*_smoothness` divided by length (P2-3) |
| **Pre-existing but missing from on-disk CSVs** | 11 q1a_* (in code since 2026-05-20 but 2da CSV is older); 26 R3 columns (cycle_offset + H/L ratio); 20 R4 columns (peak-likeness) |

Estimated final column count: ~140 (up from current 66-77).

---

## Test Strategy

| Test Type | Phase 0 | Phase 1 | Phase 2 |
|---|---|---|---|
| Behavioral (real function call, fake data) | P0-1, P0-2, P0-3, P0-4 | P1-1, P1-2 | P2-3, P2-4, P2-5 |
| Unit (helper in isolation) | — | P1-5 | P2-2, P2-7 |
| Config validation | — | P1-3, P1-4 | P2-6 |
| Source-grep (low-leverage) | — | P1-6, P1-7 (counter wiring) | P2-1, P2-8 |

Acceptance: all new tests PASS, no regressions in the existing 278-passed baseline.

---

## Verification & Retrain Plan

After all 19 commits land:

1. **Test suite**: `conda run -n silac_ml pytest tests/ -q` → expect ≥278 + new tests, no NEW failures
2. **Dry-run Makefile**: `make -n 2th 5th normal all train-exp1 train-exp2 clean-all clean-train` → all succeed
3. **Smoke test for cache invalidation (P0-3)**: small E2E test if feasible
4. **User manual retrain**:
   - `rm -rf ./workspace/` (orphan from earlier I-MK2 migration)
   - `make clean-all` (delete all 3 features.csv + logs)
   - `make all` (regenerate all 3 features.csv with current code)
   - Verify 3 features.csv have identical column sets (`diff <(head -1 ... )`)
   - `make train-all` → produce model + report under `runs/spec_trainer/`
5. **Post-retrain sanity**: report AUC; ablation by toggling `centroid_enabled` should NOW produce different reported AUCs (validates P0-3 fix)

---

## Out of Scope

Findings explicitly NOT addressed in this plan:

- **Minor findings** (M1-M10 across all 3 audits): logging-level cosmetics, `protonmass` redefinitions, `BaseManager.save` swallowing write failures, etc. Catalogued in audit reports for future cleanup.
- **Major refactors**: e.g., extracting `_ensure_parent_dir(path)` helper across spec_trainer (mentioned in T3 review), restructuring `single_work.py` (1100 lines), introducing Pydantic config schema.
- **GroupShuffleSplit** for exp2 (rubber-duck N3): keep current global stratified split; future enhancement.
- **PROJECT_INFO.md doc drift** (final review note): keep as separate doc-only follow-up.
