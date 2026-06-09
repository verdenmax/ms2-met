# Speclib Predicted-Intensity Features — Phase 2b (I2 + I3 + J2) Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Add the remaining light↔heavy *relationship* features that reuse the per-fragment records the `feature_type=0` path already collects: **I2** (H/L ratio consistency), **I3** (predicted coverage), and **J2** (unexpected-peak contamination) — as a pure-helper extension plus a one-line call in `single_pair_work`, additive and bypassed when speclib is disabled.

**Architecture:** Phase 2a already collects `speclib_frag_records` (every separable fragment, with `light_apex`/`heavy_apex`) and emits I1. Phase 2b adds a second pure function `compute_speclib_i2_i3_j2(...)` in `workflows/pred_integrate.py` over the **same** records + `pred_frags`, and `single_pair_work` merges it next to the existing I1 call. No new pipeline plumbing, no new XIC extraction.

**Tech Stack:** Python 3, numpy; Phase-1/2a modules `workflows/pred_features.py` (`_weighted_pearson`), `workflows/pred_store.py` (`frag_key`, `frag_pos_for_ion`), `workflows/pred_integrate.py`; `workflows/single_work.py`; pytest.

**Spec:** `docs/specs/2026-06-08-speclib-predicted-intensity-features-design.md` **v1.2** — §4.4 (I2), §4.5 (I3), §4.6 (J2), §4.1.5 (separability — already enforced upstream).

**Scope note:**
- **In scope (Phase 2b):** I2 + I3 + J2 columns from the existing `frag_records` (feature_type=0); TDD pure helper + corner tests; one-line integration; **L1–L4 docs**.
- **Deferred to Phase 2c (separate plan):** J5 (adaptive signal-presence floor — modifies `q1a_helpers`, needs a global L:H estimate); the `multi_batch_work`/`feature_type=1/2` path; the `pred_extract_only_topk` speedup; predicted-weighted recompute of existing aggregates.
- **Definitions of "present":** Phase 2b uses a simple absolute floor `presence_floor` (`heavy_apex > floor`) for I3/J2; the adaptive (J5) version is Phase 2c.

---

## File Structure

- **Modify `workflows/pred_integrate.py`** — add `I2I3J2_KEYS`, `_nan_i2i3j2()`, and `compute_speclib_i2_i3_j2(frag_records, pred_frags, top_k, seq_len, presence_floor) -> dict`. (Leave `compute_speclib_i1` untouched.)
- **Modify `workflows/single_work.py`** — in `single_pair_work`, after the existing I1 merge, call the new helper and `features.update(...)` it; read `presence_floor` from config (fallback 0.0).
- **Modify `constant/keys.py`** — add `PRED_PRESENCE_FLOOR = "pred_presence_floor"`.
- **Tests:** extend `tests/test_pred_integrate.py` (or new `tests/test_pred_integrate_i2i3j2.py`) + a corner file `tests/test_pred_i2i3j2_corner.py`.
- **Docs:** L1–L4 — `docs/code/parts/workflows/{L2_role,L3_details,L4_api}.md`, `docs/code/L1_overview.md` (one line), `docs/speclib/L1_overview.md`.

---

## Task 1: Config key for presence floor

**Files:**
- Modify: `constant/keys.py`
- Test: `tests/test_pred_pipeline_integration.py`

- [ ] **Step 1: Write the failing test** (append)

```python
def test_pred_presence_floor_key_exists():
    from constant.keys import ConfigKeys
    assert ConfigKeys.PRED_PRESENCE_FLOOR == "pred_presence_floor"
```

- [ ] **Step 2: Run → FAIL**

Run: `python -m pytest tests/test_pred_pipeline_integration.py::test_pred_presence_floor_key_exists -q`
Expected: FAIL (`AttributeError`).

- [ ] **Step 3: Add the key** inside `class ConfigKeys` after `PRED_TOP_K`:

```python
    PRED_PRESENCE_FLOOR = "pred_presence_floor"
```

- [ ] **Step 4: Run → PASS**

Run: `python -m pytest tests/test_pred_pipeline_integration.py::test_pred_presence_floor_key_exists -q`
Expected: PASS.

- [ ] **Step 5: Commit**

```bash
git add constant/keys.py tests/test_pred_pipeline_integration.py
git commit -m "feat(pred): config key pred_presence_floor (Phase 2b)"
```

---

## Task 2: I2/I3/J2 pure helper

**Files:**
- Modify: `workflows/pred_integrate.py`
- Test: `tests/test_pred_integrate_i2i3j2.py`

**Definitions (spec §4.4/§4.5/§4.6):** Let `cands` = separable records whose `frag_key` is in `pred_frags` (finite pred), `F` = top-`top_k` of `cands` by `pred`. Let `W` = separable records whose `frag_key` is **not** in `pred_frags` (library predicts them absent/weak). "present" = `heavy_apex > presence_floor`.
- **I2** (`pred_hl_ratio_cv`, `pred_hl_ratio_mad`): over `F` fragments with `light_apex>0 AND heavy_apex>0`, `logr_i = log10(heavy_apex/light_apex)`; `pred_hl_ratio_cv` = std(`logr`) (weighted by `pred`); `pred_hl_ratio_mad` = median(|`logr` − median(`logr`)|). NaN if <2 valid.
- **I3** (`pred_coverage`, `pred_coverage_wpred`): `pred_coverage` = (#`F` present)/|`F`|; `pred_coverage_wpred` = (Σ `pred` over present `F`)/(Σ `pred` over `F`). NaN if |`F`|=0.
- **J2** (`unexpected_heavy_fraction`, `unexpected_heavy_intensity_ratio`): `unexpected_heavy_fraction` = (#`W` present)/|`W`| (NaN if |`W`|=0); `unexpected_heavy_intensity_ratio` = (Σ heavy over present `W`)/(Σ heavy over `F` + 1e-9) (NaN if |`F`|=0).

- [ ] **Step 1: Write the failing test**

```python
# tests/test_pred_integrate_i2i3j2.py
import math
import numpy as np
from workflows.pred_integrate import (compute_speclib_i2_i3_j2, I2I3J2_KEYS)
from workflows.pred_store import frag_key, frag_pos_for_ion


def _rec(ion_type, ion_num, light_apex, heavy_apex):
    return {"ion_type": ion_type, "ion_num": ion_num,
            "light_apex": light_apex, "heavy_apex": heavy_apex,
            "light_mass": 0.0, "heavy_mass": 0.0}


def test_none_or_empty_returns_nan_schema():
    out = compute_speclib_i2_i3_j2([_rec("y", 1, 1.0, 1.0)], None, 6, 8, 0.0)
    assert set(out) == set(I2I3J2_KEYS)
    assert math.isnan(out["pred_coverage"])


def test_i2_constant_ratio_zero_dispersion():
    seq_len = 12
    recs, pred = [], {}
    for j, li in zip((1, 2, 3), (10.0, 20.0, 30.0)):
        recs.append(_rec("y", j, li, li * 2.0))      # H/L = 2 for all
        pred[frag_key("y", frag_pos_for_ion("y", j, seq_len), 1)] = li
    out = compute_speclib_i2_i3_j2(recs, pred, 6, seq_len, 0.0)
    assert out["pred_hl_ratio_cv"] < 1e-9
    assert out["pred_hl_ratio_mad"] < 1e-9


def test_i2_outlier_raises_dispersion():
    seq_len = 12
    recs, pred = [], {}
    for j, li, hi in zip((1, 2, 3), (10.0, 20.0, 30.0), (20.0, 40.0, 6.0)):
        recs.append(_rec("y", j, li, hi))            # last ratio 0.2 vs 2.0
        pred[frag_key("y", frag_pos_for_ion("y", j, seq_len), 1)] = li
    out = compute_speclib_i2_i3_j2(recs, pred, 6, seq_len, 0.0)
    assert out["pred_hl_ratio_cv"] > 0.1


def test_i3_coverage_counts_present_heavy():
    seq_len = 12
    recs, pred = [], {}
    for j, hi in zip((1, 2, 3, 4), (5.0, 0.0, 7.0, 0.0)):   # 2 of 4 present
        recs.append(_rec("y", j, 1.0, hi))
        pred[frag_key("y", frag_pos_for_ion("y", j, seq_len), 1)] = 1.0
    out = compute_speclib_i2_i3_j2(recs, pred, 6, seq_len, 0.0)
    assert abs(out["pred_coverage"] - 0.5) < 1e-9


def test_j2_unexpected_heavy_on_unpredicted_fragment():
    seq_len = 12
    # y1,y2 predicted (in F); y3 NOT predicted but has heavy signal -> unexpected
    recs = [_rec("y", 1, 1.0, 1.0), _rec("y", 2, 1.0, 1.0),
            _rec("y", 3, 1.0, 9.0)]
    pred = {frag_key("y", frag_pos_for_ion("y", 1, seq_len), 1): 1.0,
            frag_key("y", frag_pos_for_ion("y", 2, seq_len), 1): 1.0}
    out = compute_speclib_i2_i3_j2(recs, pred, 6, seq_len, 0.0)
    assert abs(out["unexpected_heavy_fraction"] - 1.0) < 1e-9      # 1/1 of W present
    assert out["unexpected_heavy_intensity_ratio"] > 0.0


def test_j2_nan_when_no_unpredicted_fragments():
    seq_len = 12
    recs, pred = [], {}
    for j in (1, 2):
        recs.append(_rec("y", j, 1.0, 1.0))
        pred[frag_key("y", frag_pos_for_ion("y", j, seq_len), 1)] = 1.0
    out = compute_speclib_i2_i3_j2(recs, pred, 6, seq_len, 0.0)
    assert math.isnan(out["unexpected_heavy_fraction"])
```

- [ ] **Step 2: Run → FAIL**

Run: `python -m pytest tests/test_pred_integrate_i2i3j2.py -q`
Expected: FAIL (`ImportError: cannot import name 'compute_speclib_i2_i3_j2'`).

- [ ] **Step 3: Implement** (append to `workflows/pred_integrate.py`)

```python
I2I3J2_KEYS = (
    "pred_hl_ratio_cv",
    "pred_hl_ratio_mad",
    "pred_coverage",
    "pred_coverage_wpred",
    "unexpected_heavy_fraction",
    "unexpected_heavy_intensity_ratio",
)


def _nan_i2i3j2() -> dict:
    return {k: float("nan") for k in I2I3J2_KEYS}


def compute_speclib_i2_i3_j2(frag_records, pred_frags, top_k, seq_len,
                             presence_floor) -> dict:
    """I2 (H/L ratio consistency), I3 (predicted coverage), J2 (unexpected
    heavy on library-unpredicted fragments). Same per-fragment records as I1.
    See spec v1.2 §4.4/§4.5/§4.6. Returns fixed I2I3J2_KEYS; NaN where undefined.
    """
    if not pred_frags or not frag_records:
        logger.debug("i2/i3/j2: no pred_frags or no records -> NaN")
        return _nan_i2i3j2()

    cands, W = [], []
    for r in frag_records:
        fp = frag_pos_for_ion(r["ion_type"], r["ion_num"], seq_len)
        pi = pred_frags.get(frag_key(r["ion_type"], fp, 1))
        if pi is None or not np.isfinite(pi):
            W.append(r)                       # library did not predict it
        else:
            cands.append({**r, "pred": float(pi)})

    out = _nan_i2i3j2()
    if cands:
        cands.sort(key=lambda r: r["pred"], reverse=True)
        F = cands[:top_k]

        # I2: weighted std + MAD of log10(H/L) over F (both apexes > 0)
        logr, wts = [], []
        for r in F:
            if r["light_apex"] > 0 and r["heavy_apex"] > 0:
                logr.append(np.log10(r["heavy_apex"] / r["light_apex"]))
                wts.append(r["pred"])
        if len(logr) >= 2:
            logr = np.asarray(logr, float)
            w = np.asarray(wts, float)
            sw = float(w.sum())
            if sw > 0:
                mean = float(np.sum(w * logr) / sw)
                var = float(np.sum(w * (logr - mean) ** 2) / sw)
                out["pred_hl_ratio_cv"] = float(np.sqrt(max(var, 0.0)))
            med = float(np.median(logr))
            out["pred_hl_ratio_mad"] = float(np.median(np.abs(logr - med)))

        # I3: coverage of F by present heavy
        present = [r for r in F if r["heavy_apex"] > presence_floor]
        out["pred_coverage"] = len(present) / len(F)
        sum_pred = sum(r["pred"] for r in F)
        if sum_pred > 0:
            out["pred_coverage_wpred"] = (
                sum(r["pred"] for r in present) / sum_pred)

        # J2: unexpected heavy on library-unpredicted separable fragments
        if W:
            w_present = [r for r in W if r["heavy_apex"] > presence_floor]
            out["unexpected_heavy_fraction"] = len(w_present) / len(W)
            heavy_F = sum(r["heavy_apex"] for r in F)
            out["unexpected_heavy_intensity_ratio"] = (
                sum(r["heavy_apex"] for r in w_present) / (heavy_F + 1e-9))
    return out
```

- [ ] **Step 4: Run → PASS**

Run: `python -m pytest tests/test_pred_integrate_i2i3j2.py -q`
Expected: PASS (6 passed)

- [ ] **Step 5: Commit**

```bash
git add workflows/pred_integrate.py tests/test_pred_integrate_i2i3j2.py
git commit -m "feat(pred): I2/I3/J2 pure helper over the same separable records"
```

---

## Task 3: Integrate I2/I3/J2 into `single_pair_work`

**Files:**
- Modify: `workflows/single_work.py`
- Test: `tests/test_pred_pipeline_integration.py`

- [ ] **Step 1: Write the failing test** (append; reuses the `_FakeDia`/`_psm2`/`_cfg2` from Phase 2a in this file)

```python
def test_single_pair_work_emits_i2i3j2_columns():
    from workflows.pred_store import frag_key as _fk2
    feats = single_pair_work(
        _psm2(), _FakeDia(), _cfg2(),
        pred_frags={_fk2("b", 0, 1): 1.0}, speclib_enabled=True)
    for col in ("pred_hl_ratio_cv", "pred_hl_ratio_mad", "pred_coverage",
                "pred_coverage_wpred", "unexpected_heavy_fraction",
                "unexpected_heavy_intensity_ratio"):
        assert col in feats
```

- [ ] **Step 2: Run → FAIL** (columns missing)

Run: `python -m pytest tests/test_pred_pipeline_integration.py::test_single_pair_work_emits_i2i3j2_columns -q`
Expected: FAIL.

- [ ] **Step 3: Implement** — in `single_pair_work`, extend the existing `if speclib_enabled:` block (added in Phase 2a, just before `return features`) to also read the floor and merge the new helper:

```python
    if speclib_enabled:
        from workflows.pred_integrate import (compute_speclib_i1,
                                              compute_speclib_i2_i3_j2)
        presence_floor = (config.getfloat(ConfigKeys.SPECLIB,
                                          ConfigKeys.PRED_PRESENCE_FLOOR,
                                          fallback=0.0)
                          if config.has_section(ConfigKeys.SPECLIB) else 0.0)
        features["has_lib_pred"] = 1 if pred_frags else 0
        features["psm_is_split_window"] = 0 if is_same_ms2 else 1
        features["heavy_out_of_range"] = 0 if heavy_in_raw else 1
        features.update(compute_speclib_i1(
            speclib_frag_records, pred_frags, pred_top_k, len(psm._sequence)))
        features.update(compute_speclib_i2_i3_j2(
            speclib_frag_records, pred_frags, pred_top_k, len(psm._sequence),
            presence_floor))
```

- [ ] **Step 4: Run → PASS** + the full Phase-2a/2b integration file

Run: `python -m pytest tests/test_pred_pipeline_integration.py -q`
Expected: PASS

- [ ] **Step 5: Commit**

```bash
git add workflows/single_work.py tests/test_pred_pipeline_integration.py
git commit -m "feat(pred): emit I2/I3/J2 from single_pair_work (feature_type=0)"
```

---

## Task 4: Corner / core tests

**Files:**
- Create: `tests/test_pred_i2i3j2_corner.py`

- [ ] **Step 1: Write tests** asserting correct behavior for: empty/None inputs (NaN schema); `presence_floor` boundary (`heavy_apex == floor` → not present; `> floor` → present); I2 with exactly 1 valid ratio → NaN; I2 weighting (a high-pred outlier moves CV more than a low-pred one); I3 `pred_coverage_wpred` differs from `pred_coverage` when present fragments carry more/less predicted weight; J2 with all-W-absent → `unexpected_heavy_fraction == 0`; J2 `unexpected_heavy_intensity_ratio` denominator uses Σ heavy over F (verify numerically); `top_k` cutoff excludes a low-pred fragment from F (so it moves to neither F nor W — it stays a low-pred cand, not W; assert it does not inflate J2). Each test asserts the value you derive from the spec; **leave any genuinely failing test in place and report it as a suspected bug** (do not weaken correct assertions).

- [ ] **Step 2: Run**

Run: `python -m pytest tests/test_pred_i2i3j2_corner.py -q`
Expected: all pass, OR a clearly-reported suspected bug.

- [ ] **Step 3: Full-suite regression**

Run: `python -m pytest tests/ -q`
Expected: pre-existing 4 failures only.

- [ ] **Step 4: Commit**

```bash
git add tests/test_pred_i2i3j2_corner.py
git commit -m "test(pred): corner/core tests for I2/I3/J2"
```

---

## Task 5: L1–L4 documentation

**Files:**
- Modify: `docs/code/L1_overview.md`, `docs/code/parts/workflows/{L2_role,L3_details,L4_api}.md`, `docs/speclib/L1_overview.md`

- [ ] **Step 1: L4 API** — under `## workflows/pred_integrate.py`, add `compute_speclib_i2_i3_j2(frag_records, pred_frags, top_k, seq_len, presence_floor) -> dict` + `I2I3J2_KEYS`, with the precise column meanings (I2 `pred_hl_ratio_cv`/`_mad`; I3 `pred_coverage`/`_wpred`; J2 `unexpected_heavy_fraction`/`_intensity_ratio`).
- [ ] **Step 2: L3 details** — extend the "谱库 I1 特征接入 feature_type=0" subsection: I2/I3/J2 reuse the same `speclib_frag_records` (W = 可分但库未预测的碎片；present = `heavy_apex > pred_presence_floor`); list the 6 new columns and their definitions.
- [ ] **Step 3: L2 role** — note `pred_integrate.py` now emits I1 + I2/I3/J2 for `feature_type=0`.
- [ ] **Step 4: L1 overview** — one line under the speclib row that Phase 2b (I2/I3/J2) is接入 feature_type=0; Phase 2c = J5 + feature_type 1/2 + 提速.
- [ ] **Step 5: speclib L1** — update the "接入特征提取" note: Phase 2a/2b 已接入 feature_type=0（I1 + I2/I3/J2）。
- [ ] **Step 6: Commit**

```bash
git add docs/code/L1_overview.md docs/code/parts/workflows/*.md docs/speclib/L1_overview.md
git commit -m "docs(L1-L4): Phase 2b I2/I3/J2 integration"
```

---

## Self-Review

**Spec coverage:** §4.4 I2 → Task 2 (`pred_hl_ratio_cv`/`_mad`); §4.5 I3 → Task 2 (`pred_coverage`/`_wpred`); §4.6 J2 → Task 2 (`unexpected_heavy_*`). All over the **same** `frag_records` Phase 2a already collects (separable-only, split-aware upstream). J5/feature_type 1-2/perf → Phase 2c (explicitly out of scope).

**Placeholder scan:** No placeholders. `compute_speclib_i1` is untouched (its 5 tests stay green); the new function is additive with its own fixed `I2I3J2_KEYS` (schema-stable). Task 4 corner tests are described by behavior + concrete cases.

**Type consistency:** `frag_record` keys (`ion_type/ion_num/light_apex/heavy_apex`) identical to Phase 2a producer. `presence_floor` is a float from `config.getfloat(..., fallback=0.0)`. `W`/`cands`/`F` partition the separable records (predicted vs not; top-K of predicted). New columns are distinctly named (`pred_hl_ratio_*` ≠ existing `*_log_lh_ratio_*`).

**Phase 2c (next plan):** J5 adaptive floor (`q1a_helpers`, global L:H median, `pred_use_adaptive_floor`), I3/J2 re-pointed at the J5 presence test; `multi_batch_work`/`feature_type=1/2` (attach pred_frags to both PSMs); `pred_extract_only_topk` speedup.

---

## Execution Handoff

(Filled by the writing-plans skill at hand-off time.)
