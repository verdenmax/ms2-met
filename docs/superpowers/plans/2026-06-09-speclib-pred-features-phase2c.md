# Speclib Predicted-Intensity Features — Phase 2c (J5 Adaptive Coverage) Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Add an **adaptive, physics-based** coverage signal for the `feature_type=0` speclib features: instead of (only) a fixed intensity floor, judge a predicted-strong fragment's heavy signal against its own per-fragment expectation `light_apex × global_L:H_ratio`. Emits two additive columns — `global_lh_ratio` and `pred_coverage_adaptive` — reusing the records already collected; no behavior change to existing columns.

**Architecture:** A third pure function `compute_speclib_adaptive(...)` in `workflows/pred_integrate.py` over the same `speclib_frag_records` + `pred_frags`, merged in `single_pair_work` next to the existing I1 / I2-I3-J2 calls. `alpha` comes from config (`pred_signal_alpha`, default 0.2).

**Tech Stack:** Python 3, numpy; `workflows/pred_integrate.py`, `workflows/pred_store.py` (`frag_key`, `frag_pos_for_ion`), `workflows/single_work.py`, `constant/keys.py`; pytest.

**Spec:** `docs/specs/2026-06-08-speclib-predicted-intensity-features-design.md` v1.2 §4.7 (J5). **The plan corrects §4.7's formula** (see Task 5): the literal `E_i = pred_rel_i × light_apex_i × global_LH` double-counts the per-fragment intensity because observed `light_apex_i` already encodes it; the physically-correct expectation is `E_i = light_apex_i × global_LH`.

**Scope note:**
- **In scope (Phase 2c):** `global_lh_ratio` + `pred_coverage_adaptive` columns (additive, always emitted when speclib enabled), the `pred_signal_alpha` config key, corner tests, L1–L4 docs, and the §4.7 spec correction.
- **Deferred (future phase):** the `multi_batch_work` / `feature_type=1/2` integration (the user runs `feature_type=0`); the `pred_extract_only_topk` speedup (conflicts with existing all-fragment features); a J5 variant of `q1a_helpers.is_signal_present_heavy`.

---

## File Structure

- **Modify `constant/keys.py`** — add `PRED_SIGNAL_ALPHA = "pred_signal_alpha"`.
- **Modify `workflows/pred_integrate.py`** — add `ADAPTIVE_KEYS`, `_nan_adaptive()`, `compute_speclib_adaptive(frag_records, pred_frags, top_k, seq_len, alpha) -> dict`. (Leave `compute_speclib_i1` / `compute_speclib_i2_i3_j2` untouched.)
- **Modify `workflows/single_work.py`** — merge the new helper in the existing `if speclib_enabled:` block; read `alpha` from config.
- **Tests:** `tests/test_pred_adaptive.py` + corner `tests/test_pred_adaptive_corner.py`.
- **Docs:** L1–L4 + spec §4.7 correction.

---

## Task 1: Config key `pred_signal_alpha`

**Files:**
- Modify: `constant/keys.py`
- Test: `tests/test_pred_pipeline_integration.py`

- [ ] **Step 1: Write the failing test** (append)

```python
def test_pred_signal_alpha_key_exists():
    from constant.keys import ConfigKeys
    assert ConfigKeys.PRED_SIGNAL_ALPHA == "pred_signal_alpha"
```

- [ ] **Step 2: Run → FAIL**

Run: `python -m pytest tests/test_pred_pipeline_integration.py::test_pred_signal_alpha_key_exists -q`
Expected: FAIL (`AttributeError`).

- [ ] **Step 3: Add the key** inside `class ConfigKeys` after `PRED_PRESENCE_FLOOR`:

```python
    PRED_SIGNAL_ALPHA = "pred_signal_alpha"
```

- [ ] **Step 4: Run → PASS**

- [ ] **Step 5: Commit**

```bash
git add constant/keys.py tests/test_pred_pipeline_integration.py
git commit -m "feat(pred): config key pred_signal_alpha (Phase 2c)"
```

---

## Task 2: Adaptive-coverage pure helper

**Files:**
- Modify: `workflows/pred_integrate.py`
- Test: `tests/test_pred_adaptive.py`

**Definitions:** `cands` = separable records with a finite prediction; `F` = top-`top_k` of `cands` by `pred`.
- **`global_lh_ratio`** = median over `F` of `heavy_apex / light_apex` (only fragments with `light_apex>0 AND heavy_apex>0`); NaN if <1 such fragment.
- **`pred_coverage_adaptive`** = fraction of `F` fragments with `light_apex>0` whose `heavy_apex >= alpha × light_apex × global_lh_ratio`; NaN if no `F` fragment has `light_apex>0` or `global_lh_ratio` is NaN.

- [ ] **Step 1: Write the failing test**

```python
# tests/test_pred_adaptive.py
import math
import numpy as np
from workflows.pred_integrate import compute_speclib_adaptive, ADAPTIVE_KEYS
from workflows.pred_store import frag_key, frag_pos_for_ion


def _rec(ion_type, ion_num, light_apex, heavy_apex):
    return {"ion_type": ion_type, "ion_num": ion_num,
            "light_apex": light_apex, "heavy_apex": heavy_apex,
            "light_mass": 0.0, "heavy_mass": 0.0}


def test_none_or_empty_returns_nan_schema():
    out = compute_speclib_adaptive([_rec("y", 1, 1.0, 1.0)], None, 6, 8, 0.2)
    assert set(out) == set(ADAPTIVE_KEYS)
    assert math.isnan(out["global_lh_ratio"])
    assert math.isnan(out["pred_coverage_adaptive"])


def test_constant_ratio_all_present():
    seq_len = 12
    recs, pred = [], {}
    for j, li in zip((1, 2, 3), (10.0, 20.0, 30.0)):
        recs.append(_rec("y", j, li, li * 2.0))      # H/L = 2 everywhere
        pred[frag_key("y", frag_pos_for_ion("y", j, seq_len), 1)] = li
    out = compute_speclib_adaptive(recs, pred, 6, seq_len, 0.2)
    assert abs(out["global_lh_ratio"] - 2.0) < 1e-9
    # each heavy = 2*light >= 0.2*light*2 = 0.4*light -> all present
    assert abs(out["pred_coverage_adaptive"] - 1.0) < 1e-9


def test_underexpected_fragment_not_present():
    seq_len = 12
    recs, pred = [], {}
    # two at ratio 2.0, one far below expectation
    for j, li, hi in zip((1, 2, 3), (10.0, 20.0, 30.0), (20.0, 40.0, 1.0)):
        recs.append(_rec("y", j, li, hi))
        pred[frag_key("y", frag_pos_for_ion("y", j, seq_len), 1)] = li
    out = compute_speclib_adaptive(recs, pred, 6, seq_len, 0.2)
    assert abs(out["global_lh_ratio"] - 2.0) < 1e-9   # median of (2,2,0.033)
    # frag3: heavy 1.0 vs 0.2*30*2=12.0 -> absent; 2/3 present
    assert abs(out["pred_coverage_adaptive"] - (2.0 / 3.0)) < 1e-9


def test_no_valid_ratio_returns_nan():
    seq_len = 12
    recs, pred = [], {}
    for j in (1, 2):
        recs.append(_rec("y", j, 0.0, 5.0))          # light_apex 0 everywhere
        pred[frag_key("y", frag_pos_for_ion("y", j, seq_len), 1)] = 1.0
    out = compute_speclib_adaptive(recs, pred, 6, seq_len, 0.2)
    assert math.isnan(out["global_lh_ratio"])
    assert math.isnan(out["pred_coverage_adaptive"])
```

- [ ] **Step 2: Run → FAIL** (`ImportError`)

- [ ] **Step 3: Implement** (append to `workflows/pred_integrate.py`)

```python
ADAPTIVE_KEYS = ("global_lh_ratio", "pred_coverage_adaptive")


def _nan_adaptive() -> dict:
    return {k: float("nan") for k in ADAPTIVE_KEYS}


def compute_speclib_adaptive(frag_records, pred_frags, top_k, seq_len,
                             alpha) -> dict:
    """J5 adaptive coverage (spec v1.2 §4.7, corrected formula):
    a predicted-strong fragment is 'present' iff its heavy apex meets the
    per-fragment expectation `alpha * light_apex * global_lh_ratio`, where
    global_lh_ratio = median(H/L) over F. Returns fixed ADAPTIVE_KEYS; NaN
    where undefined. (Observed light_apex already carries the per-fragment
    intensity, so the spec's extra `pred_rel_i` factor is dropped — see §4.7.)
    """
    if not pred_frags or not frag_records:
        logger.debug("adaptive: no pred_frags or no records -> NaN")
        return _nan_adaptive()

    cands = []
    for r in frag_records:
        fp = frag_pos_for_ion(r["ion_type"], r["ion_num"], seq_len)
        pi = pred_frags.get(frag_key(r["ion_type"], fp, 1))
        if pi is not None and np.isfinite(pi):
            cands.append({**r, "pred": float(pi)})
    if not cands:
        return _nan_adaptive()

    cands.sort(key=lambda r: r["pred"], reverse=True)
    F = cands[:top_k]

    out = _nan_adaptive()
    ratios = [r["heavy_apex"] / r["light_apex"]
              for r in F if r["light_apex"] > 0 and r["heavy_apex"] > 0]
    if not ratios:
        return out
    glh = float(np.median(ratios))
    out["global_lh_ratio"] = glh

    valid = [r for r in F if r["light_apex"] > 0]
    if valid:
        present = [r for r in valid
                   if r["heavy_apex"] >= alpha * r["light_apex"] * glh]
        out["pred_coverage_adaptive"] = len(present) / len(valid)
    return out
```

- [ ] **Step 4: Run → PASS** (4 passed)

- [ ] **Step 5: Commit**

```bash
git add workflows/pred_integrate.py tests/test_pred_adaptive.py
git commit -m "feat(pred): J5 adaptive coverage helper (global_lh_ratio + pred_coverage_adaptive)"
```

---

## Task 3: Integrate adaptive coverage into `single_pair_work`

**Files:**
- Modify: `workflows/single_work.py`
- Test: `tests/test_pred_pipeline_integration.py`

- [ ] **Step 1: Write the failing test** (append)

```python
def test_single_pair_work_emits_adaptive_columns():
    from workflows.pred_store import frag_key as _fk3
    feats = single_pair_work(
        _psm2(), _FakeDia(), _cfg2(),
        pred_frags={_fk3("b", 0, 1): 1.0}, speclib_enabled=True)
    assert "global_lh_ratio" in feats
    assert "pred_coverage_adaptive" in feats
```

- [ ] **Step 2: Run → FAIL**

- [ ] **Step 3: Implement** — extend the existing `if speclib_enabled:` block in `single_pair_work` (it already imports + calls `compute_speclib_i1` and `compute_speclib_i2_i3_j2`). Add the import, read `alpha`, and merge:

```python
    if speclib_enabled:
        from workflows.pred_integrate import (compute_speclib_i1,
                                              compute_speclib_i2_i3_j2,
                                              compute_speclib_adaptive)
        presence_floor = (config.getfloat(ConfigKeys.SPECLIB,
                                          ConfigKeys.PRED_PRESENCE_FLOOR,
                                          fallback=0.0)
                          if config.has_section(ConfigKeys.SPECLIB) else 0.0)
        alpha = (config.getfloat(ConfigKeys.SPECLIB,
                                 ConfigKeys.PRED_SIGNAL_ALPHA, fallback=0.2)
                 if config.has_section(ConfigKeys.SPECLIB) else 0.2)
        features["has_lib_pred"] = 1 if pred_frags else 0
        features["psm_is_split_window"] = 0 if is_same_ms2 else 1
        features["heavy_out_of_range"] = 0 if heavy_in_raw else 1
        features.update(compute_speclib_i1(
            speclib_frag_records, pred_frags, pred_top_k, len(psm._sequence)))
        features.update(compute_speclib_i2_i3_j2(
            speclib_frag_records, pred_frags, pred_top_k, len(psm._sequence),
            presence_floor))
        features.update(compute_speclib_adaptive(
            speclib_frag_records, pred_frags, pred_top_k, len(psm._sequence),
            alpha))
```

- [ ] **Step 4: Run → PASS** + full integration file

Run: `python -m pytest tests/test_pred_pipeline_integration.py -q`
Expected: PASS

- [ ] **Step 5: Commit**

```bash
git add workflows/single_work.py tests/test_pred_pipeline_integration.py
git commit -m "feat(pred): emit J5 adaptive coverage from single_pair_work"
```

---

## Task 4: Corner / core tests

**Files:**
- Create: `tests/test_pred_adaptive_corner.py`

- [ ] **Step 1: Write tests** for: empty/None inputs → NaN schema; `alpha` boundary (`heavy_apex == alpha*light*glh` → present, strict `>=`); `alpha=0` → all light>0 fragments present (coverage=1); a fragment with `heavy_apex=0` but `light>0` → absent (pulls coverage down); `global_lh_ratio` is the **median** (robust to one outlier ratio — verify with an odd-count set); fragments excluded from `F` by `top_k` don't affect `global_lh_ratio`/coverage; a single F fragment with both apexes>0 → `global_lh_ratio` finite, coverage defined; mixed b/y pooled. Assert spec-derived values; leave any genuinely-failing test in place as a SUSPECTED BUG with analysis.

- [ ] **Step 2: Run** → all pass (or reported suspected bug).

- [ ] **Step 3: Full-suite regression**

Run: `python -m pytest tests/ -q`
Expected: pre-existing 4 failures only.

- [ ] **Step 4: Commit**

```bash
git add tests/test_pred_adaptive_corner.py
git commit -m "test(pred): corner/core tests for J5 adaptive coverage"
```

---

## Task 5: L1–L4 docs + spec §4.7 correction

**Files:**
- Modify: `docs/specs/2026-06-08-speclib-predicted-intensity-features-design.md` (§4.7, §5, §6, §10)
- Modify: `docs/code/L1_overview.md`, `docs/code/parts/workflows/{L2_role,L3_details,L4_api}.md`, `docs/speclib/L1_overview.md`

- [ ] **Step 1: Spec §4.7 correction** — change the expected-intensity formula to `E_i = light_apex_i × global_LH_ratio` (drop the redundant `pred_rel_i`; observed `light_apex_i` already carries the per-fragment intensity). Mark J5 as implemented (Phase 2c) for `feature_type=0` as the additive columns `global_lh_ratio` + `pred_coverage_adaptive` (the `q1a_helpers.is_signal_present_heavy` variant remains future). Add the two columns to §5; add `pred_signal_alpha` (default 0.2) to §6; update §10 P3 status.
- [ ] **Step 2: L4 API** — under `workflows/pred_integrate.py`, add `compute_speclib_adaptive(frag_records, pred_frags, top_k, seq_len, alpha) -> dict` + `ADAPTIVE_KEYS` with column meanings.
- [ ] **Step 3: L3 details** — extend the I2/I3/J2 subsection: J5 adaptive coverage (`global_lh_ratio` = median(H/L) over F; `pred_coverage_adaptive` = fraction of F meeting `alpha·light·glh`); note the corrected formula.
- [ ] **Step 4: L2 role + L1 overview + speclib L1** — note `pred_integrate.py` now also emits J5 adaptive coverage; Phase 2c done; remaining = `feature_type=1/2` + perf.
- [ ] **Step 5: Commit**

```bash
git add docs/specs/2026-06-08-speclib-predicted-intensity-features-design.md docs/code/L1_overview.md docs/code/parts/workflows/*.md docs/speclib/L1_overview.md
git commit -m "docs(L1-L4+spec): Phase 2c J5 adaptive coverage + correct §4.7 formula"
```

---

## Self-Review

**Spec coverage:** §4.7 J5 → Task 2/3 (adaptive coverage as additive columns), with the formula corrected in Task 5. The `q1a_helpers.is_signal_present_heavy` variant and `feature_type=1/2`/perf are explicitly deferred.

**Placeholder scan:** No placeholders. `compute_speclib_i1` / `compute_speclib_i2_i3_j2` untouched (their tests stay green); the new function is additive with fixed `ADAPTIVE_KEYS` (schema-stable). Task 4 corner cases are concrete.

**Type consistency:** `frag_record` keys identical to Phase 2a/2b producers. `alpha`/`presence_floor` are floats via `config.getfloat(..., fallback=...)`. New columns (`global_lh_ratio`, `pred_coverage_adaptive`) don't collide with any existing feature.

**Future phase:** `feature_type=1/2` integration (attach pred_frags to both PSMs; add separability determination to `multi_batch_work`, whose heavy is in a different raw); `pred_extract_only_topk` speedup; adaptive `is_signal_present_heavy` in `q1a_helpers`.

---

## Execution Handoff

(Filled by the writing-plans skill at hand-off time.)
