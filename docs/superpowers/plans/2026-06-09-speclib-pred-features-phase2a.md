# Speclib Predicted-Intensity Features — Phase 2a (Pipeline Integration + I1) Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Wire the Phase-1 speclib foundation into the live feature-extraction pipeline so the `feature_type=0` (single-flow) path emits the **I1 intensity-pattern** features (per ion-type) plus coverage/meta columns into `result.csv` — as an additive bypass that reproduces current output when speclib is unconfigured.

**Architecture:** Build a `PredStore` once in `PairFlow.distribute()` (one streaming scan of the library, Phase-1 `build_pred_store`), attach each PSM's predicted fragments to its task dict, and compute I1 from per-fragment records collected inside the existing `single_pair_work` loop via a new pure helper `workflows/pred_integrate.py`. The pipeline already determines split/co-isolation (`check_in_same_ms2`), heavy-in-range (`check_in_raw`), and pre-filters to separable fragments, so no separability logic is re-implemented here.

**Tech Stack:** Python 3, numpy, scipy; Phase-1 modules `workflows/pred_features.py` + `workflows/pred_store.py`; existing `workflows/single_work.py` / `pair_flow.py` / `flow_utils.py`; `constant/keys.py`; pytest (synthetic `lib_files` fixture in `tests/conftest.py`).

**Spec:** `docs/specs/2026-06-08-speclib-predicted-intensity-features-design.md` **v1.2** — implements §4.1 (PredStore in pipeline), §4.2 (F = top-K separable), §4.3 (I1) restricted to I1, §7 (per-ion-type metric), §4.1.5 (separability/centroiding — already enforced by the existing loop).

**Scope note (read first):**
- **In scope (Phase 2a):** config keys; PredStore build + `pred_frags` plumbing; the I1 pure helper; integration into **`single_pair_work` only (`feature_type=0`)**; `has_lib_pred` + meta columns; NaN/absent fallback; regression that `speclib_dir` empty reproduces current `result.csv`.
- **Deferred to Phase 2b (separate plan):** I2 (H/L ratio), I3 (coverage), J2 (unexpected-peak), J5 (adaptive floor), the `multi_batch_work` path (`feature_type=1/2`), predicted-weighted recompute of existing features, and the XIC-extraction speedup.
- **Precondition:** the Phase-1 sanity gate has passed on the target library (validated 2026-06-09, spec §11). Centroiding (`centroid_enabled=true`) is required and already the pipeline default.

---

## File Structure

- **Create `workflows/pred_integrate.py`** — pure per-PSM I1 computation:
  - `I1_KEYS` (fixed output schema), `_nan_i1()`, `compute_speclib_i1(frag_records, pred_frags, top_k, seq_len) -> dict`.
- **Modify `constant/keys.py`** — add `SPECLIB`, `SPECLIB_DIR`, `SPECLIB_FASTA`, `SPECLIB_MOD`, `PRED_TOP_K`.
- **Modify `workflows/pair_flow.py`** — build `PredStore` in `distribute()`; pass it to `_build_raw_tasks`, which attaches `pred_frags` to each `feature_type=0` task dict; add a `_build_pred_store()` helper.
- **Modify `workflows/flow_utils.py`** — `process_batch_single` / `process_psm_single`: read `pred_frags` + `speclib_enabled` from the dict and pass to `single_pair_work`.
- **Modify `workflows/single_work.py`** — `single_pair_work` collects `frag_records` in its existing loop and merges `compute_speclib_i1` + `has_lib_pred` + meta columns (guarded by `speclib_enabled`).
- **Tests:** `tests/test_pred_integrate.py` (pure helper), `tests/test_pred_pipeline_integration.py` (plumbing + regression).

---

## Task 1: Config keys

**Files:**
- Modify: `constant/keys.py`
- Test: `tests/test_pred_pipeline_integration.py`

- [ ] **Step 1: Write the failing test**

```python
# tests/test_pred_pipeline_integration.py
from constant.keys import ConfigKeys


def test_speclib_config_keys_exist():
    assert ConfigKeys.SPECLIB == "speclib"
    assert ConfigKeys.SPECLIB_DIR == "speclib_dir"
    assert ConfigKeys.SPECLIB_FASTA == "speclib_fasta"
    assert ConfigKeys.SPECLIB_MOD == "speclib_mod"
    assert ConfigKeys.PRED_TOP_K == "pred_top_k"
```

- [ ] **Step 2: Run test to verify it fails**

Run: `python -m pytest tests/test_pred_pipeline_integration.py::test_speclib_config_keys_exist -v`
Expected: FAIL with `AttributeError: type object 'ConfigKeys' has no attribute 'SPECLIB'`

- [ ] **Step 3: Add the keys**

```python
# append inside class ConfigKeys in constant/keys.py (after CENTROID_REL_THRESHOLD)

    # speclib predicted-intensity features (Phase 2)
    SPECLIB = "speclib"
    SPECLIB_DIR = "speclib_dir"
    SPECLIB_FASTA = "speclib_fasta"
    SPECLIB_MOD = "speclib_mod"
    PRED_TOP_K = "pred_top_k"
```

- [ ] **Step 4: Run test to verify it passes**

Run: `python -m pytest tests/test_pred_pipeline_integration.py::test_speclib_config_keys_exist -v`
Expected: PASS

- [ ] **Step 5: Commit**

```bash
git add constant/keys.py tests/test_pred_pipeline_integration.py
git commit -m "feat(pred): config keys for speclib feature integration"
```

---

## Task 2: I1 pure helper

**Files:**
- Create: `workflows/pred_integrate.py`
- Test: `tests/test_pred_integrate.py`

- [ ] **Step 1: Write the failing test**

```python
# tests/test_pred_integrate.py
import math
from workflows.pred_integrate import compute_speclib_i1, I1_KEYS
from workflows.pred_store import frag_key, frag_pos_for_ion


def _rec(ion_type, ion_num, light_apex, heavy_apex):
    return {"ion_type": ion_type, "ion_num": ion_num,
            "light_apex": light_apex, "heavy_apex": heavy_apex,
            "light_mass": 0.0, "heavy_mass": 0.0}


def test_none_pred_frags_returns_nan_schema():
    out = compute_speclib_i1([_rec("b", 1, 5.0, 5.0)], None, top_k=6, seq_len=8)
    assert set(out) == set(I1_KEYS)
    assert math.isnan(out["spec_pattern_SA"])
    assert out["n_fragments_in_F"] == 0


def test_perfect_match_per_ion_type_high():
    seq_len = 8
    # 3 b ions + 3 y ions; observed heavy == predicted pattern -> SA ~ 1
    recs, pred = [], {}
    for i, inten in zip((1, 2, 3), (1.0, 0.6, 0.3)):
        recs.append(_rec("b", i, inten, inten))          # b_i -> frag_pos i-1
        pred[frag_key("b", frag_pos_for_ion("b", i, seq_len), 1)] = inten
    for j, inten in zip((1, 2, 3), (0.9, 0.5, 0.2)):
        recs.append(_rec("y", j, inten, inten))          # y_j -> frag_pos L-j-1
        pred[frag_key("y", frag_pos_for_ion("y", j, seq_len), 1)] = inten
    out = compute_speclib_i1(recs, pred, top_k=6, seq_len=seq_len)
    assert abs(out["spec_pattern_SA_b"] - 1.0) < 1e-6
    assert abs(out["spec_pattern_SA_y"] - 1.0) < 1e-6
    assert abs(out["spec_pattern_SA"] - 1.0) < 1e-6
    assert out["n_fragments_in_F"] == 6


def test_topk_limits_fragment_set():
    seq_len = 12
    recs, pred = [], {}
    for i, inten in zip(range(1, 9), (0.9, 0.8, 0.7, 0.6, 0.5, 0.4, 0.3, 0.2)):
        recs.append(_rec("y", i, inten, inten))
        pred[frag_key("y", frag_pos_for_ion("y", i, seq_len), 1)] = inten
    out = compute_speclib_i1(recs, pred, top_k=3, seq_len=seq_len)
    assert out["n_fragments_in_F"] == 3


def test_only_b_gives_nan_for_y():
    seq_len = 8
    recs, pred = [], {}
    for i, inten in zip((1, 2), (1.0, 0.5)):
        recs.append(_rec("b", i, inten, inten))
        pred[frag_key("b", frag_pos_for_ion("b", i, seq_len), 1)] = inten
    out = compute_speclib_i1(recs, pred, top_k=6, seq_len=seq_len)
    assert math.isnan(out["spec_pattern_SA_y"])
    assert abs(out["spec_pattern_SA_b"] - 1.0) < 1e-6
    assert abs(out["spec_pattern_SA"] - out["spec_pattern_SA_b"]) < 1e-9


def test_fragment_without_prediction_is_excluded():
    # a record whose frag_key is not in pred_frags must not enter F
    seq_len = 8
    recs = [_rec("y", 1, 1.0, 1.0), _rec("y", 2, 0.5, 0.5)]
    pred = {frag_key("y", frag_pos_for_ion("y", 1, seq_len), 1): 1.0}  # only y1
    out = compute_speclib_i1(recs, pred, top_k=6, seq_len=seq_len)
    assert out["n_fragments_in_F"] == 1
    assert math.isnan(out["spec_pattern_SA_y"])   # <2 fragments
```

- [ ] **Step 2: Run test to verify it fails**

Run: `python -m pytest tests/test_pred_integrate.py -q`
Expected: FAIL with `ModuleNotFoundError: No module named 'workflows.pred_integrate'`

- [ ] **Step 3: Write the implementation**

```python
# workflows/pred_integrate.py
"""Per-PSM speclib predicted-intensity features — Phase 2a (I1 only).

Pure function over (already-separable) per-fragment records + the PSM's
predicted fragments. The pipeline (single_pair_work) does the split-aware
separability filtering and centroiding upstream (spec v1.2 §4.1.5); this
module only does the math. See spec §4.3 (I1) and §7 (per-ion-type metric).
"""
import logging

import numpy as np

from workflows.pred_features import spectral_angle, _weighted_pearson
from workflows.pred_store import frag_key, frag_pos_for_ion

logger = logging.getLogger(__name__)

# Fixed output schema: every PSM emits identical columns (LightGBM-safe).
I1_KEYS = (
    "spec_pattern_SA_b",
    "spec_pattern_SA_y",
    "spec_pattern_SA",
    "spec_pattern_LH_consistency",
    "n_fragments_in_F",
)


def _nan_i1() -> dict:
    d = {k: float("nan") for k in I1_KEYS}
    d["n_fragments_in_F"] = 0
    return d


def compute_speclib_i1(frag_records, pred_frags, top_k, seq_len) -> dict:
    """I1 intensity-pattern features for one PSM (spec §4.3, per ion-type §7).

    frag_records: list of dicts (already-separable fragments) with keys
        ion_type ('b'/'y'), ion_num (1-based), light_apex, heavy_apex.
    pred_frags:   {frag_key: intensity} for this PSM, or None (no coverage).
    Returns a fixed-key dict (I1_KEYS); NaN where undefined.
    """
    if not pred_frags or not frag_records:
        return _nan_i1()

    # attach predicted intensity; keep only fragments the library predicts
    cands = []
    for r in frag_records:
        fp = frag_pos_for_ion(r["ion_type"], r["ion_num"], seq_len)
        pi = pred_frags.get(frag_key(r["ion_type"], fp, 1))
        if pi is None or not np.isfinite(pi):
            continue
        cands.append({**r, "pred": float(pi)})
    if not cands:
        return _nan_i1()

    cands.sort(key=lambda r: r["pred"], reverse=True)
    F = cands[:top_k]

    def sa_for(ion_type):
        sub = [r for r in F if r["ion_type"] == ion_type]
        if len(sub) < 2:
            return float("nan")
        return spectral_angle([r["pred"] for r in sub],
                              [r["heavy_apex"] for r in sub])

    sa_b = sa_for("b")
    sa_y = sa_for("y")
    both = [s for s in (sa_b, sa_y) if np.isfinite(s)]
    sa_comb = float(np.mean(both)) if both else float("nan")

    lh = _weighted_pearson([r["light_apex"] for r in F],
                           [r["heavy_apex"] for r in F],
                           [r["pred"] for r in F])

    return {
        "spec_pattern_SA_b": sa_b,
        "spec_pattern_SA_y": sa_y,
        "spec_pattern_SA": sa_comb,
        "spec_pattern_LH_consistency": lh,
        "n_fragments_in_F": len(F),
    }
```

- [ ] **Step 4: Run test to verify it passes**

Run: `python -m pytest tests/test_pred_integrate.py -q`
Expected: PASS (5 passed)

- [ ] **Step 5: Commit**

```bash
git add workflows/pred_integrate.py tests/test_pred_integrate.py
git commit -m "feat(pred): I1 intensity-pattern pure helper (per ion-type, NaN-safe schema)"
```

---

## Task 3: PredStore build + `pred_frags` attachment in `pair_flow`

**Files:**
- Modify: `workflows/pair_flow.py:172-198` (`_build_raw_tasks`) + `distribute()`
- Test: `tests/test_pred_pipeline_integration.py`

**Context:** `_build_raw_tasks` (static) builds `feature_type=0` tasks as `(a.to_dict(), shared_path)`. We attach `pred_frags` (a `{frag_key: intensity}` dict, or `None`) to each task dict, looked up from a `PredStore`. `distribute()` builds the `PredStore` once (Phase-1 `build_pred_store`) when `[speclib] speclib_dir` is configured.

- [ ] **Step 1: Write the failing test**

```python
# append to tests/test_pred_pipeline_integration.py
from workflows.pair_flow import PairFlow
from workflows.pred_store import build_pred_store, normalize_key, frag_key
from spectrum.speclib import SpecLib
from spectrum.psm_info import PSMInfo
import numpy as np


def _psm(seq, charge, raw):
    return PSMInfo(sequence=seq, charge=charge, modify=[], rt=np.float32(10.0),
                   precursor_mz=np.float32(500.0), raw_title=raw,
                   protein_names="X", label_type="positive")


def test_build_raw_tasks_attaches_pred_frags(lib_files):
    lib = SpecLib.open_dir(str(lib_files), fasta_path=str(lib_files / "db.fasta"),
                           mod_path=str(lib_files / "modification.ini"))
    seq = "PEPTIDEKACDM"
    want = normalize_key(seq, [], 1)
    store = build_pred_store(lib, {want})
    psm = _psm(seq, 1, "r1")
    groups = {psm.get_key(): [psm]}
    tasks, n_skipped = PairFlow._build_raw_tasks(
        groups, {"r1": "/tmp/shared.npz"}, 0, pred_store=store)
    assert len(tasks) == 1
    psm_dict = tasks[0][0]
    assert "pred_frags" in psm_dict
    assert psm_dict["pred_frags"][frag_key("b", 0, 1)] == np.float32(1.0).item() or \
        abs(psm_dict["pred_frags"][frag_key("b", 0, 1)] - 1.0) < 1e-6


def test_build_raw_tasks_pred_frags_none_when_no_store(lib_files):
    psm = _psm("PEPTIDEKACDM", 1, "r1")
    groups = {psm.get_key(): [psm]}
    tasks, _ = PairFlow._build_raw_tasks(groups, {"r1": "/tmp/s.npz"}, 0,
                                         pred_store=None)
    # speclib disabled -> no pred_frags key (schema unchanged)
    assert "pred_frags" not in tasks[0][0]


def test_build_raw_tasks_pred_frags_none_on_miss(lib_files):
    lib = SpecLib.open_dir(str(lib_files), fasta_path=str(lib_files / "db.fasta"),
                           mod_path=str(lib_files / "modification.ini"))
    store = build_pred_store(lib, set())   # nothing wanted
    psm = _psm("NOTINLIBK", 2, "r1")
    groups = {psm.get_key(): [psm]}
    tasks, _ = PairFlow._build_raw_tasks(groups, {"r1": "/tmp/s.npz"}, 0,
                                         pred_store=store)
    # speclib enabled but this peptide missed -> key present, value None
    assert tasks[0][0]["pred_frags"] is None
```

- [ ] **Step 2: Run test to verify it fails**

Run: `python -m pytest tests/test_pred_pipeline_integration.py -q -k build_raw_tasks`
Expected: FAIL — `_build_raw_tasks` does not accept `pred_store`.

- [ ] **Step 3: Modify `_build_raw_tasks` (add `pred_store` param + attach)**

Replace the `feature_type == 0` branch of `_build_raw_tasks` and its signature:

```python
    @staticmethod
    def _build_raw_tasks(psm_groups, name_to_shared, feature_type,
                         pred_store=None):
        """构建任务列表；PSM 的 raw_title 不在配置的 raw 文件中时跳过并计数。
        当 pred_store 给出（speclib 开启）时，给 feature_type=0 的每个任务
        dict 附上 pred_frags（命中=预测碎片 dict，未命中=None）。
        返回 (tasks, n_skipped)。"""
        from workflows.pred_store import normalize_key
        tasks = []
        n_skipped = 0
        if feature_type == 0:
            for group in psm_groups.values():
                for a in group:
                    if a._raw_title not in name_to_shared:
                        n_skipped += 1
                        continue
                    d = a.to_dict()
                    if pred_store is not None:
                        rec = pred_store.get(
                            normalize_key(a._sequence, a._modify, a._charge))
                        d["pred_frags"] = rec["frags"] if rec is not None else None
                    tasks.append((d, name_to_shared[a._raw_title]))
        else:  # feature_type 1 或 2  (Phase 2b: pred_frags 暂不附)
            for group in psm_groups.values():
                for a, b in combinations(group, 2):
                    if (a._raw_title not in name_to_shared
                            or b._raw_title not in name_to_shared):
                        n_skipped += 1
                        continue
                    shared_a = name_to_shared[a._raw_title]
                    shared_b = name_to_shared[b._raw_title]
                    tasks.append(
                        (a.to_dict(), b.to_dict(), shared_a, shared_b, 1))
                    tasks.append(
                        (a.to_dict(), b.to_dict(), shared_a, shared_b, 0))
        return tasks, n_skipped
```

- [ ] **Step 4: Run test to verify it passes**

Run: `python -m pytest tests/test_pred_pipeline_integration.py -q -k build_raw_tasks`
Expected: PASS (3 passed)

- [ ] **Step 5: Add `_build_pred_store` + wire into `distribute()`**

Add a helper method on `PairFlow` and call it in `distribute()` after `self._light_result` is available, then pass the store into the `_build_raw_tasks(...)` call.

```python
# add method on PairFlow
    def _build_pred_store(self):
        """若 [speclib] speclib_dir 配置了，则一遍流式扫库建 PredStore；
        否则返回 None（speclib 关闭，主流程回退现状）。"""
        if not self._config.has_option(ConfigKeys.SPECLIB, ConfigKeys.SPECLIB_DIR):
            return None
        speclib_dir = self._config[ConfigKeys.SPECLIB][ConfigKeys.SPECLIB_DIR].strip()
        if not speclib_dir:
            return None
        from spectrum.speclib import SpecLib
        from workflows.pred_store import build_pred_store, normalize_key
        fasta = self._config[ConfigKeys.SPECLIB][ConfigKeys.SPECLIB_FASTA]
        mod = self._config[ConfigKeys.SPECLIB][ConfigKeys.SPECLIB_MOD]
        lib = SpecLib.open_dir(speclib_dir, fasta_path=fasta, mod_path=mod)
        wanted = {normalize_key(p._sequence, p._modify, p._charge)
                  for p in self._light_result.psm_info}
        logging.info("speclib: 扫库建 PredStore（wanted=%d）...", len(wanted))
        return build_pred_store(lib, wanted)
```

In `distribute()`, find the existing call that builds tasks via `_build_raw_tasks(...)` (after `psm_groups` is built) and pass the store:

```python
        pred_store = self._build_pred_store()   # None when speclib disabled
        tasks, n_skipped = self._build_raw_tasks(
            psm_groups, name_to_shared, feature_type, pred_store=pred_store)
```

> **Implementer verify step:** confirm the exact location/线 where `_build_raw_tasks` is currently called inside `distribute()` (grep `_build_raw_tasks(` in `pair_flow.py`) and `feature_type` is in scope there; add the two lines above immediately before that call and pass `pred_store=pred_store`.

- [ ] **Step 6: Run the build-raw-tasks tests + import check**

Run: `python -m pytest tests/test_pred_pipeline_integration.py -q -k build_raw_tasks && python -c "import workflows.pair_flow"`
Expected: PASS + import OK

- [ ] **Step 7: Commit**

```bash
git add workflows/pair_flow.py tests/test_pred_pipeline_integration.py
git commit -m "feat(pred): build PredStore in distribute() + attach pred_frags to feature_type=0 tasks"
```

---

## Task 4: Worker plumbing (`flow_utils`)

**Files:**
- Modify: `workflows/flow_utils.py` (`process_psm_single`, `process_batch_single`)
- Test: covered by the end-to-end test in Task 6 (no isolated unit test — these are thin pass-throughs).

- [ ] **Step 1: Modify `process_psm_single`**

```python
def process_psm_single(
        psm1_dict: dict,
        shared1_file: str,
        config: configparser.ConfigParser):
    """ 处理单文件 PSM，提取特征。 """
    dia1 = DIAData.load_from_file(shared1_file, use_mmap=True)
    speclib_enabled = "pred_frags" in psm1_dict
    pred_frags = psm1_dict.get("pred_frags")
    psm1 = PSMInfo.from_dict(psm1_dict)

    tot_features = single_pair_work(
        psm=psm1,
        dia_data=dia1,
        config=config,
        pred_frags=pred_frags,
        speclib_enabled=speclib_enabled,
    )
    return _make_result_row_single(psm1, tot_features)
```

- [ ] **Step 2: Modify `process_batch_single`**

In the per-PSM loop (`for (psm_dict,) in batch_psm_dicts:`), extract the two values before `PSMInfo.from_dict` and pass them to `single_pair_work`:

```python
    for (psm_dict,) in batch_psm_dicts:
        try:
            speclib_enabled = "pred_frags" in psm_dict
            pred_frags = psm_dict.get("pred_frags")
            psm = PSMInfo.from_dict(psm_dict)
            features = single_pair_work(
                psm=psm, dia_data=dia_data, config=config,
                pred_frags=pred_frags, speclib_enabled=speclib_enabled)
            results.append(_make_result_row_single(psm, features))
        except Exception:
            n_errors += 1
            logging.exception("single PSM failed")
```

> **Implementer verify step:** open `process_batch_single` and match the existing try/except + append shape (it already catches per-PSM exceptions into `n_errors`); only add the two `speclib_*`/`pred_frags` lines and the two new kwargs. `from_dict` ignores the extra `pred_frags` key (verified: `spectrum/psm_info.py:79-94`).

- [ ] **Step 3: Import check**

Run: `python -c "import workflows.flow_utils"`
Expected: import OK

- [ ] **Step 4: Commit**

```bash
git add workflows/flow_utils.py
git commit -m "feat(pred): pass pred_frags/speclib_enabled through single-flow workers"
```

---

## Task 5: Integrate I1 into `single_pair_work`

**Files:**
- Modify: `workflows/single_work.py` (`single_pair_work` signature + fragment loop + post-loop)
- Test: covered by Task 6.

**Context:** `single_pair_work` already (a) computes `heavy_in_raw = dia_data.check_in_raw(heavy_precursor_mz)` and `is_same_ms2 = dia_data.check_in_same_ms2(...)`, (b) skips non-separable fragments, (c) computes per-fragment `ion_score` with `light_max_int`/`heavy_max_int`. We collect a `frag_record` per separable fragment and call the helper after the loop.

- [ ] **Step 1: Change the signature**

```python
def single_pair_work(
    psm: PSMInfo,
    dia_data: DIAData,
    config: ConfigParser,
    pred_frags=None,
    speclib_enabled: bool = False,
):
```

- [ ] **Step 2: Initialize the record list + top_k near the other fragment lists**

Add next to the `fragment_*` list initializations (e.g., right after `ion_data = []`):

```python
    speclib_frag_records = []   # per separable fragment, for I1 (Phase 2a)
    pred_top_k = config.getint(ConfigKeys.SPECLIB, ConfigKeys.PRED_TOP_K,
                               fallback=6) if config.has_section(
                                   ConfigKeys.SPECLIB) else 6
```

- [ ] **Step 3: Append a record per separable fragment (inside the loop, after `q1a_acc.add`)**

Immediately after the `q1a_acc.add(...)` call in the fragment loop (the fragment has already passed the `heavy_in_raw` and shift/same-ms2 guards, so it is separable), add:

```python
        _light_ap = (float(np.max(light_ions_xic["intensity"]))
                     if len(light_ions_xic) else 0.0)
        _heavy_ap = (float(np.max(heavy_ions_xic["intensity"]))
                     if len(heavy_ions_xic) else 0.0)
        speclib_frag_records.append({
            "ion_type": ions_type, "ion_num": ions_num,
            "light_mass": light_mass, "heavy_mass": heavy_mass,
            "light_apex": _light_ap, "heavy_apex": _heavy_ap,
        })
```

(Place it BEFORE the `if len(light_ions_xic) == 0 or len(heavy_ions_xic) == 0: ... continue` block so empty-heavy separable fragments are still recorded with `heavy_apex=0`.)

- [ ] **Step 4: Merge I1 + meta columns after the loop (before `return features`)**

Just before `return features` at the end of `single_pair_work`, add:

```python
    if speclib_enabled:
        from workflows.pred_integrate import compute_speclib_i1
        features["has_lib_pred"] = 1 if pred_frags else 0
        features["psm_is_split_window"] = 0 if is_same_ms2 else 1
        features["heavy_out_of_range"] = 0 if heavy_in_raw else 1
        features.update(compute_speclib_i1(
            speclib_frag_records, pred_frags, pred_top_k, len(psm._sequence)))
```

> **Note:** `single_pair_work` has a **single** return (`return features`, ~line 813); the empty-precursor case is an `if/else` that falls through to it, so placing the merge just before that one return covers every PSM (empty-XIC PSMs get `_nan_i1()` columns + `has_lib_pred`/meta). `is_same_ms2` and `heavy_in_raw` are in scope (defined earlier in the function, ~lines 559/574).

- [ ] **Step 5: Import check**

Run: `python -c "import workflows.single_work"`
Expected: import OK

- [ ] **Step 6: Commit**

```bash
git add workflows/single_work.py
git commit -m "feat(pred): emit I1 + has_lib_pred + window meta from single_pair_work"
```

---

## Task 6: End-to-end integration + regression test

**Files:**
- Test: `tests/test_pred_pipeline_integration.py`

**Approach:** test `single_pair_work` directly with a small **fake DIA** (stubs only the methods it calls, all returning empty/sane values) — robust and npz-free. The fragment XICs are empty, so I1 is NaN; the test asserts column *presence/absence* (the schema contract), not values. (Task 3 already covers the `pred_frags` attachment; Task 2 covers the I1 math.)

- [ ] **Step 1: Write the failing test**

```python
# append to tests/test_pred_pipeline_integration.py
import numpy as np
import configparser
from workflows.single_work import single_pair_work
from workflows.pred_store import frag_key


_EMPTY_XIC = np.zeros(0, dtype=[("rt", "f8"), ("intensity", "f8"),
                                ("ppm_error", "f8")])


class _FakeDia:
    """Minimal stand-in exposing the methods single_pair_work calls."""
    def xic_peaks_extreact(self, rt, win, mz, ppm):
        return _EMPTY_XIC

    def xic_ms2_peaks_extract(self, rt, win, precursor_mz, ions_mass,
                              mass_tol_ppm):
        return _EMPTY_XIC, 0.0

    def get_window_info(self, mz):
        return {"lower": 400.0, "upper": 600.0, "width": 200.0,
                "centering": 0.5}

    def check_in_raw(self, mz):
        return True

    def check_in_same_ms2(self, a, b):
        return False   # split windows


def _cfg2():
    c = configparser.ConfigParser()
    c["general"] = {"mass_tol_ppm": "10", "xic_cycle_window": "3"}
    c["speclib"] = {"pred_top_k": "6"}
    return c


def _psm2():
    return PSMInfo(sequence="PEPTIDEKACDM", charge=1, modify=[],
                   rt=np.float32(10.0), precursor_mz=np.float32(500.0),
                   raw_title="r1", protein_names="X", label_type="positive")


def test_single_pair_work_emits_speclib_columns_when_enabled():
    feats = single_pair_work(
        _psm2(), _FakeDia(), _cfg2(),
        pred_frags={frag_key("b", 0, 1): 1.0}, speclib_enabled=True)
    for col in ("has_lib_pred", "spec_pattern_SA_b", "spec_pattern_SA_y",
                "spec_pattern_SA", "spec_pattern_LH_consistency",
                "n_fragments_in_F", "psm_is_split_window", "heavy_out_of_range"):
        assert col in feats
    assert feats["has_lib_pred"] == 1
    assert feats["psm_is_split_window"] == 1   # check_in_same_ms2 -> False


def test_single_pair_work_unchanged_when_speclib_disabled():
    feats = single_pair_work(_psm2(), _FakeDia(), _cfg2(),
                             pred_frags=None, speclib_enabled=False)
    assert "has_lib_pred" not in feats
    assert "spec_pattern_SA" not in feats
```

- [ ] **Step 2: Run test to verify it fails**

Run: `python -m pytest tests/test_pred_pipeline_integration.py -q -k single_pair_work`
Expected: FAIL before Task 5 is implemented (no `speclib_enabled` kwarg / no columns). After Tasks 1–5 it should pass.

> **Implementer verify step:** if `single_pair_work` calls a DIA method not stubbed above (run the test — an `AttributeError` names it), add the missing stub returning an empty XIC / sane scalar. The set above matches the calls in the empty-precursor fall-through path (`xic_peaks_extreact`, `get_window_info`, `check_in_raw`, `check_in_same_ms2`, `xic_ms2_peaks_extract`).

- [ ] **Step 3: Run test to verify it passes**

Run: `python -m pytest tests/test_pred_pipeline_integration.py -q`
Expected: PASS (all integration tests)

- [ ] **Step 4: Full-suite regression**

Run: `python -m pytest tests/ -q`
Expected: pre-existing 4 failures only (`test_rescore_tool.py` ×3, `test_training_matrix.py` ×1); everything else green. Confirms the `speclib` disabled path is unchanged.

- [ ] **Step 5: Commit**

```bash
git add tests/test_pred_pipeline_integration.py
git commit -m "test(pred): single_pair_work emits I1 when enabled, unchanged when disabled"
```

---

## Task 7: Documentation

**Files:**
- Modify: `docs/code/parts/workflows/{L2_role,L3_details,L4_api}.md`, `docs/speclib/L1_overview.md`

- [ ] **Step 1: L4 API** — add `## workflows/pred_integrate.py` documenting `compute_speclib_i1` + `I1_KEYS`; note `single_pair_work(..., pred_frags, speclib_enabled)` new params; `_build_raw_tasks(..., pred_store)` and `PairFlow._build_pred_store()`.
- [ ] **Step 2: L3 details** — add a subsection: feature_type=0 路径在 `single_pair_work` 内收集可分碎片 `frag_records`，经 `compute_speclib_i1` 产出 `spec_pattern_SA_b/_y/_SA/_LH_consistency` + `has_lib_pred`/`psm_is_split_window`/`heavy_out_of_range`；`speclib_dir` 空时不出列（schema 不变）。
- [ ] **Step 3: L2 role + speclib L1** — add `pred_integrate.py` to the module table; note Phase 2a 已接入 feature_type=0，Phase 2b（I2/I3/J2/J5 + feature_type=1/2）见 spec §4.4–§4.8/§10。
- [ ] **Step 4: Commit**

```bash
git add docs/code/parts/workflows/*.md docs/speclib/L1_overview.md
git commit -m "docs: L1-L4 for Phase 2a speclib I1 integration"
```

---

## Self-Review

**Spec coverage (against design v1.2):**
- §4.1 PredStore in pipeline → Task 3 ✅
- §4.2 F = top-K separable (separability already enforced by `single_pair_work` guards) → Task 5 (records) + Task 2 (top-K) ✅
- §4.3 I1 → Task 2 ✅ (constructed predicted-heavy = `pred` vs observed `heavy_apex`)
- §7 per-ion-type metric (b/y separate, not one mixed vector) → Task 2 (`sa_b`/`sa_y`/combined) ✅
- §4.1.5 separability/centroiding → enforced upstream by the existing loop (`check_in_same_ms2`/`check_in_raw`/shift guard) + pipeline centroiding default; not re-implemented ✅
- §4.4 I2 / §4.5 I3 / §4.6 J2 / §4.7 J5 / `feature_type=1/2` / perf → **Phase 2b** (explicitly out of scope).

**Placeholder scan:** No placeholders remain. Production code is complete; tests use a self-contained fake DIA (Task 6). Three "implementer verify step" notes flag exact confirmations against live code (the `distribute()` call-site at `pair_flow.py:243`, the `process_batch_single` try/except shape, and any un-stubbed DIA method in the Task 6 fake) — these are confirmations, not missing code. Verified live: `distribute()` reads `feature_type` (line 239) and calls `_build_raw_tasks(...)` (line 243); `single_pair_work` has one return (line 813); `from_dict` ignores extra keys (`psm_info.py:79-94`).

**Type consistency:** `pred_frags` is `{frag_key: intensity}|None` consistently across pair_flow (`rec["frags"]`), flow_utils, single_work, and the helper. `frag_record` keys (`ion_type/ion_num/light_apex/heavy_apex`) are identical in Task 5 (producer) and Task 2 (consumer). `frag_pos_for_ion`/`frag_key` reused from Phase 1. I1 output schema fixed via `I1_KEYS`.

**Phase 2b (next plan):** I2 (`hl_ratio_cv_weighted` on F), I3 (`pred_coverage` via J5-or-floor presence on F), J2 (predicted-weak `unexpected_heavy_*`), J5 (adaptive floor in `q1a_helpers`), the `multi_batch_work`/`feature_type=1/2` path (attach pred_frags to both PSMs of a pair), predicted-weighted recompute of existing aggregates, and `pred_extract_only_topk` speedup.

---

## Execution Handoff

(Filled by the writing-plans skill at hand-off time.)
