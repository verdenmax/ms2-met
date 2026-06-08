# Speclib Predicted-Intensity Features — Phase 1 (Foundation + Sanity Gate) Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Build the reusable, fully-unit-tested foundation (similarity metrics, top-K fragment selection, I1 pattern features, peptide→prediction lookup) plus a standalone sanity-gate CLI that decides whether the pFind library's predicted intensities agree with observed light spectra — **before** touching the feature-extraction hot path.

**Architecture:** Two new pure-Python modules — `workflows/pred_features.py` (vector metrics + fragment selection + I1 features) and `workflows/pred_store.py` (key normalization + streaming peptide→prediction lookup) — plus one CLI `tools/speclib_sanity.py` that wires them to real data for a go/no-go check. Nothing here is wired into `single_work.py`/`pair_flow.py`/`result.csv`; that integration (I2/I3/J2/J5 + columns) is Phase 2, gated on this sanity check passing.

**Tech Stack:** Python 3, numpy, scipy (`scipy.stats.spearmanr`), existing `spectrum.speclib.SpecLib`, pytest. Tests reuse the synthetic-library `lib_files` fixture in `tests/conftest.py`.

**Spec:** `docs/specs/2026-06-08-speclib-predicted-intensity-features-design.md` (this plan implements §4.0 sanity gate, §4.1 lookup, §4.2 top-K, §4.3 I1, §7 metrics).

---

## File Structure

- **Create `workflows/pred_features.py`** — pure functions, no I/O:
  - `spectral_angle(a, b) -> float`
  - `spearman_sim(a, b) -> float`
  - `_weighted_pearson(x, y, w) -> float`
  - `select_topk_separable(fragments, k) -> list`
  - `i1_pattern_features(pred, obs_heavy, obs_light) -> dict`
- **Create `workflows/pred_store.py`** — peptide identity + lookup, no feature math:
  - `normalize_mods(mods) -> tuple`
  - `normalize_key(sequence, mods, charge) -> tuple`
  - `frag_key(ion_type, frag_pos, frag_charge) -> tuple`
  - `class PredStore` (`.get(key)`, `.n_hit`, `.n_miss`, `.wanted`)
  - `build_pred_store(lib, wanted_keys, decode_ms2="objects") -> PredStore`
- **Create `tools/speclib_sanity.py`** — go/no-go CLI:
  - `similarity_distribution(pairs, metric) -> dict`
  - `gate_pass(stats, min_sim) -> bool`
  - `main()` (argparse wiring to real library + raw)
- **Create tests**: `tests/test_pred_features.py`, `tests/test_pred_store.py`, `tests/test_speclib_sanity.py`
- **Modify docs**: `docs/code/parts/workflows/{L2_role,L3_details,L4_api}.md`, `docs/speclib/L1_overview.md` (note the new feature layer + Phase 2 pointer).

---

## Task 1: Spectral-angle metric

**Files:**
- Create: `workflows/pred_features.py`
- Test: `tests/test_pred_features.py`

- [ ] **Step 1: Write the failing test**

```python
# tests/test_pred_features.py
import math
import numpy as np
from workflows.pred_features import spectral_angle


def test_spectral_angle_identical_vectors_is_one():
    assert spectral_angle([1.0, 2.0, 3.0], [1.0, 2.0, 3.0]) == \
        np.float64(1.0) or abs(spectral_angle([1.0, 2.0, 3.0],
                                              [1.0, 2.0, 3.0]) - 1.0) < 1e-9


def test_spectral_angle_orthogonal_is_zero():
    assert abs(spectral_angle([1.0, 0.0], [0.0, 1.0]) - 0.0) < 1e-9


def test_spectral_angle_scaled_vectors_is_one():
    # SA is scale-invariant: same shape, different magnitude → 1.0
    assert abs(spectral_angle([1.0, 2.0, 3.0], [10.0, 20.0, 30.0]) - 1.0) < 1e-9


def test_spectral_angle_degenerate_returns_nan():
    assert math.isnan(spectral_angle([1.0], [1.0]))          # length < 2
    assert math.isnan(spectral_angle([0.0, 0.0], [1.0, 2.0]))  # zero norm
    assert math.isnan(spectral_angle([1.0, 2.0], [1.0, 2.0, 3.0]))  # mismatched len
```

- [ ] **Step 2: Run test to verify it fails**

Run: `python -m pytest tests/test_pred_features.py -q`
Expected: FAIL with `ModuleNotFoundError: No module named 'workflows.pred_features'`

- [ ] **Step 3: Write minimal implementation**

```python
# workflows/pred_features.py
"""Pure feature math for speclib predicted-intensity features (Phase 1).

No I/O, no pipeline coupling — only numeric vector operations so each piece
is unit-testable in isolation. See
docs/specs/2026-06-08-speclib-predicted-intensity-features-design.md.
"""
import numpy as np


def spectral_angle(a, b) -> float:
    """Normalized spectral contrast angle similarity in [0, 1].

    1.0 = identical shape (scale-invariant), 0.0 = orthogonal. Returns NaN
    for degenerate input (length < 2, mismatched length, or zero-norm) so
    callers never confuse 'undefined' with 'truly dissimilar'.
    """
    a = np.asarray(a, dtype=float)
    b = np.asarray(b, dtype=float)
    if a.size < 2 or b.size < 2 or a.size != b.size:
        return float("nan")
    a = np.clip(a, 0.0, None)
    b = np.clip(b, 0.0, None)
    na = float(np.linalg.norm(a))
    nb = float(np.linalg.norm(b))
    if na < 1e-12 or nb < 1e-12:
        return float("nan")
    cos = float(np.dot(a, b) / (na * nb))
    cos = min(1.0, max(-1.0, cos))
    return 1.0 - (2.0 / np.pi) * float(np.arccos(cos))
```

- [ ] **Step 4: Run test to verify it passes**

Run: `python -m pytest tests/test_pred_features.py -q`
Expected: PASS (4 passed)

- [ ] **Step 5: Commit**

```bash
git add workflows/pred_features.py tests/test_pred_features.py
git commit -m "feat(pred): spectral-angle similarity metric (NaN-safe)"
```

---

## Task 2: Spearman-rank similarity metric

**Files:**
- Modify: `workflows/pred_features.py`
- Test: `tests/test_pred_features.py`

- [ ] **Step 1: Write the failing test**

```python
# append to tests/test_pred_features.py
from workflows.pred_features import spearman_sim


def test_spearman_sim_monotonic_same_order_is_one():
    assert abs(spearman_sim([1.0, 2.0, 3.0, 4.0], [10.0, 20.0, 25.0, 40.0]) - 1.0) < 1e-9


def test_spearman_sim_reversed_is_minus_one():
    assert abs(spearman_sim([1.0, 2.0, 3.0, 4.0], [40.0, 30.0, 20.0, 10.0]) + 1.0) < 1e-9


def test_spearman_sim_constant_or_short_returns_nan():
    import math
    assert math.isnan(spearman_sim([1.0, 1.0, 1.0], [1.0, 2.0, 3.0]))  # zero variance
    assert math.isnan(spearman_sim([1.0], [2.0]))                       # length < 2
```

- [ ] **Step 2: Run test to verify it fails**

Run: `python -m pytest tests/test_pred_features.py -q`
Expected: FAIL with `ImportError: cannot import name 'spearman_sim'`

- [ ] **Step 3: Write minimal implementation**

```python
# add to workflows/pred_features.py
from scipy.stats import spearmanr


def spearman_sim(a, b) -> float:
    """Spearman rank correlation in [-1, 1]; robust to absolute-intensity
    miscalibration of the prediction model. NaN for degenerate input."""
    a = np.asarray(a, dtype=float)
    b = np.asarray(b, dtype=float)
    if a.size < 2 or b.size < 2 or a.size != b.size:
        return float("nan")
    if float(np.std(a)) < 1e-12 or float(np.std(b)) < 1e-12:
        return float("nan")
    rho, _ = spearmanr(a, b)
    rho = float(rho)
    return rho if np.isfinite(rho) else float("nan")
```

- [ ] **Step 4: Run test to verify it passes**

Run: `python -m pytest tests/test_pred_features.py -q`
Expected: PASS (7 passed)

- [ ] **Step 5: Commit**

```bash
git add workflows/pred_features.py tests/test_pred_features.py
git commit -m "feat(pred): Spearman-rank similarity metric (NaN-safe)"
```

---

## Task 3: Top-K separable-fragment selection

**Files:**
- Modify: `workflows/pred_features.py`
- Test: `tests/test_pred_features.py`

- [ ] **Step 1: Write the failing test**

```python
# append to tests/test_pred_features.py
from workflows.pred_features import select_topk_separable


def _frag(fid, pred, sep):
    return {"id": fid, "pred_intensity": pred, "separable": sep}


def test_select_topk_picks_highest_pred_among_separable():
    frags = [
        _frag("a", 0.9, True),
        _frag("b", 0.8, False),   # high pred but not separable -> excluded
        _frag("c", 0.5, True),
        _frag("d", 0.7, True),
    ]
    chosen = select_topk_separable(frags, k=2)
    assert [f["id"] for f in chosen] == ["a", "d"]


def test_select_topk_returns_all_when_fewer_than_k():
    frags = [_frag("a", 0.9, True), _frag("b", 0.1, False)]
    chosen = select_topk_separable(frags, k=6)
    assert [f["id"] for f in chosen] == ["a"]
```

- [ ] **Step 2: Run test to verify it fails**

Run: `python -m pytest tests/test_pred_features.py -q`
Expected: FAIL with `ImportError: cannot import name 'select_topk_separable'`

- [ ] **Step 3: Write minimal implementation**

```python
# add to workflows/pred_features.py
def select_topk_separable(fragments, k):
    """Return the up-to-k separable fragments with the highest predicted
    intensity. `fragments` is a list of dicts with keys 'pred_intensity'
    (float) and 'separable' (bool). Non-separable fragments give no
    light/heavy contrast, so they never occupy a slot."""
    separable = [f for f in fragments if f.get("separable")]
    separable.sort(key=lambda f: f["pred_intensity"], reverse=True)
    return separable[:k]
```

- [ ] **Step 4: Run test to verify it passes**

Run: `python -m pytest tests/test_pred_features.py -q`
Expected: PASS (9 passed)

- [ ] **Step 5: Commit**

```bash
git add workflows/pred_features.py tests/test_pred_features.py
git commit -m "feat(pred): top-K separable-fragment selection"
```

---

## Task 4: I1 intensity-pattern features

**Files:**
- Modify: `workflows/pred_features.py`
- Test: `tests/test_pred_features.py`

- [ ] **Step 1: Write the failing test**

```python
# append to tests/test_pred_features.py
import math
from workflows.pred_features import i1_pattern_features


def test_i1_perfect_match_high_scores():
    pred = [1.0, 0.8, 0.4, 0.2]
    obs_heavy = [10.0, 8.0, 4.0, 2.0]     # same shape as pred
    obs_light = [10.0, 8.0, 4.0, 2.0]     # light == heavy shape
    f = i1_pattern_features(pred, obs_heavy, obs_light)
    assert abs(f["spec_pattern_SA_heavy"] - 1.0) < 1e-9
    assert abs(f["spec_pattern_spearman_heavy"] - 1.0) < 1e-9
    assert abs(f["spec_pattern_LH_consistency"] - 1.0) < 1e-9


def test_i1_shuffled_pattern_low_sa():
    pred = [1.0, 0.8, 0.4, 0.2]
    obs_heavy = [0.2, 0.4, 0.8, 1.0]      # reversed shape
    obs_light = [0.2, 0.4, 0.8, 1.0]
    f = i1_pattern_features(pred, obs_heavy, obs_light)
    assert f["spec_pattern_SA_heavy"] < 0.7


def test_i1_degenerate_returns_nan():
    f = i1_pattern_features([1.0], [1.0], [1.0])   # length < 2
    assert math.isnan(f["spec_pattern_SA_heavy"])
    assert math.isnan(f["spec_pattern_LH_consistency"])
```

- [ ] **Step 2: Run test to verify it fails**

Run: `python -m pytest tests/test_pred_features.py -q`
Expected: FAIL with `ImportError: cannot import name 'i1_pattern_features'`

- [ ] **Step 3: Write minimal implementation**

```python
# add to workflows/pred_features.py
def _weighted_pearson(x, y, w) -> float:
    """Pearson correlation of x,y weighted by non-negative weights w.
    NaN for degenerate input (length < 2, mismatched length, zero weight
    sum, or zero weighted variance)."""
    x = np.asarray(x, dtype=float)
    y = np.asarray(y, dtype=float)
    w = np.clip(np.asarray(w, dtype=float), 0.0, None)
    if x.size < 2 or not (x.size == y.size == w.size):
        return float("nan")
    sw = float(w.sum())
    if sw < 1e-12:
        return float("nan")
    mx = float(np.sum(w * x) / sw)
    my = float(np.sum(w * y) / sw)
    cov = float(np.sum(w * (x - mx) * (y - my)) / sw)
    vx = float(np.sum(w * (x - mx) ** 2) / sw)
    vy = float(np.sum(w * (y - my) ** 2) / sw)
    if vx < 1e-12 or vy < 1e-12:
        return float("nan")
    return cov / np.sqrt(vx * vy)


def i1_pattern_features(pred, obs_heavy, obs_light) -> dict:
    """I1 intensity-pattern consistency (spec §4.3).

    `pred`, `obs_heavy`, `obs_light` are aligned over the same fragment set F
    (same order). The predicted *heavy* spectrum equals L's predicted
    intensities placed at the heavy fragments (chemical equivalence), so we
    compare `pred` against `obs_heavy` directly.
    """
    return {
        "spec_pattern_SA_heavy": spectral_angle(pred, obs_heavy),
        "spec_pattern_spearman_heavy": spearman_sim(pred, obs_heavy),
        "spec_pattern_LH_consistency": _weighted_pearson(obs_light, obs_heavy, pred),
    }
```

- [ ] **Step 4: Run test to verify it passes**

Run: `python -m pytest tests/test_pred_features.py -q`
Expected: PASS (12 passed)

- [ ] **Step 5: Commit**

```bash
git add workflows/pred_features.py tests/test_pred_features.py
git commit -m "feat(pred): I1 intensity-pattern features (SA/Spearman/weighted LH)"
```

---

## Task 5: Peptide key normalization + fragment key

**Files:**
- Create: `workflows/pred_store.py`
- Test: `tests/test_pred_store.py`

- [ ] **Step 1: Write the failing test**

```python
# tests/test_pred_store.py
from workflows.pred_store import normalize_mods, normalize_key, frag_key


class _ModSite:
    """Mimics spectrum.speclib.pepdata.ModSite (pos, mod_id)."""
    def __init__(self, pos, mod_id):
        self.pos = pos
        self.mod_id = mod_id


def test_normalize_mods_handles_tuples_and_modsite_equally():
    from_tuples = normalize_mods([(9, 1), (3, 2)])
    from_objs = normalize_mods([_ModSite(9, 1), _ModSite(3, 2)])
    assert from_tuples == from_objs == ((3, 2), (9, 1))   # sorted by position


def test_normalize_key_is_hashable_and_charge_int():
    key = normalize_key("PEPTIDEK", [(9, 1)], "2")
    assert key == ("PEPTIDEK", ((9, 1),), 2)
    assert hash(key)   # usable as dict key


def test_frag_key_normalizes_types():
    assert frag_key("b", 0, 1) == ("b", 0, 1)
    assert frag_key("y", np.int8(2), np.int8(3)) == ("y", 2, 3)
```

Add `import numpy as np` at the top of the test file.

- [ ] **Step 2: Run test to verify it fails**

Run: `python -m pytest tests/test_pred_store.py -q`
Expected: FAIL with `ModuleNotFoundError: No module named 'workflows.pred_store'`

- [ ] **Step 3: Write minimal implementation**

```python
# workflows/pred_store.py
"""Peptide identity normalization + streaming peptide->prediction lookup.

Builds an in-memory {normalize_key -> predictions} store by scanning the
spectral library once and keeping only the wanted (identified) peptides.
See docs/specs/2026-06-08-speclib-predicted-intensity-features-design.md §4.1.
"""


def _as_pairs(mods):
    """Yield (pos:int, mod_id:int) from either (pos, mod_id) tuples or
    ModSite objects (with .pos / .mod_id)."""
    for m in mods:
        if hasattr(m, "pos") and hasattr(m, "mod_id"):
            yield int(m.pos), int(m.mod_id)
        else:
            pos, mid = m
            yield int(pos), int(mid)


def normalize_mods(mods) -> tuple:
    """Canonical, hashable, position-sorted modification tuple."""
    return tuple(sorted(_as_pairs(mods)))


def normalize_key(sequence, mods, charge) -> tuple:
    """Canonical peptide-variant key: (sequence, sorted-mods, int charge)."""
    return (sequence, normalize_mods(mods), int(charge))


def frag_key(ion_type, frag_pos, frag_charge) -> tuple:
    """Canonical fragment key shared by predicted and observed sides."""
    return (str(ion_type), int(frag_pos), int(frag_charge))
```

- [ ] **Step 4: Run test to verify it passes**

Run: `python -m pytest tests/test_pred_store.py -q`
Expected: PASS (3 passed)

- [ ] **Step 5: Commit**

```bash
git add workflows/pred_store.py tests/test_pred_store.py
git commit -m "feat(pred): peptide key + fragment key normalization"
```

---

## Task 6: Streaming PredStore builder

**Files:**
- Modify: `workflows/pred_store.py`
- Test: `tests/test_pred_store.py`

- [ ] **Step 1: Write the failing test**

```python
# append to tests/test_pred_store.py
from spectrum.speclib import SpecLib
from workflows.pred_store import build_pred_store, normalize_key, frag_key


def _open_lib(lib_files):
    return SpecLib.open_dir(
        str(lib_files),
        fasta_path=str(lib_files / "db.fasta"),
        mod_path=str(lib_files / "modification.ini"))


def test_build_pred_store_hits_unmodified_variant(lib_files):
    lib = _open_lib(lib_files)
    # variant 1: PEPTIDEKACDM, no mods, charge 1 -> MS2 record [(pos=0,it=0,1.0)]
    # it=0 -> ion_type 'b', frag_charge 0//2+1 = 1
    want = normalize_key("PEPTIDEKACDM", [], 1)
    store = build_pred_store(lib, {want})
    rec = store.get(want)
    assert rec is not None
    assert rec["frags"][frag_key("b", 0, 1)] == 1.0
    assert store.n_hit == 1 and store.n_miss == 0


def test_build_pred_store_hits_modified_variant_charge2(lib_files):
    lib = _open_lib(lib_files)
    # variant 2: mods [(9,1)], charge 2 -> MS2 record [(pos=2,it=3,0.3)]
    # it=3 -> ion_type 'y', frag_charge 3//2+1 = 2
    want = normalize_key("PEPTIDEKACDM", [(9, 1)], 2)
    store = build_pred_store(lib, {want})
    rec = store.get(want)
    assert rec is not None
    assert rec["frags"][frag_key("y", 2, 2)] == 0.3


def test_build_pred_store_counts_miss(lib_files):
    lib = _open_lib(lib_files)
    present = normalize_key("PEPTIDEKACDM", [], 1)
    absent = normalize_key("NOTINLIBK", [], 2)
    store = build_pred_store(lib, {present, absent})
    assert store.get(absent) is None
    assert store.n_hit == 1 and store.n_miss == 1
```

- [ ] **Step 2: Run test to verify it fails**

Run: `python -m pytest tests/test_pred_store.py -q`
Expected: FAIL with `ImportError: cannot import name 'build_pred_store'`

- [ ] **Step 3: Write minimal implementation**

```python
# add to workflows/pred_store.py
class PredStore:
    """In-memory {normalize_key -> {'frags': {frag_key: intensity},
    'pred_rt': float}} for the identified peptides only."""

    def __init__(self):
        self._d = {}
        self.wanted = set()
        self.n_hit = 0
        self.n_miss = 0

    def get(self, key):
        return self._d.get(key)


def _frag_map(frag_ions) -> dict:
    """Build {frag_key: intensity} from a list of FragIon (objects mode)."""
    out = {}
    for fi in frag_ions:
        out[frag_key(fi.ion_type, fi.frag_pos, fi.frag_charge)] = float(fi.intensity)
    return out


def build_pred_store(lib, wanted_keys, decode_ms2: str = "objects") -> PredStore:
    """Scan `lib` once; keep predictions only for peptides in `wanted_keys`.

    `wanted_keys` is a set of normalize_key tuples. Memory is O(hits).
    """
    store = PredStore()
    store.wanted = set(wanted_keys)

    # index wanted charges by (sequence, normalized-mods)
    want_by_id = {}
    for (seq, norm_mods, chg) in store.wanted:
        want_by_id.setdefault((seq, norm_mods), set()).add(chg)

    for pep in lib.iter_peptides(decode_ms2=decode_ms2):
        pid = (pep.sequence, normalize_mods(pep.mods))
        charges = want_by_id.get(pid)
        if not charges:
            continue
        for chg in charges:
            frags = pep.pred_ms2.get(chg)
            if frags is None:
                continue
            store._d[(pep.sequence, normalize_mods(pep.mods), chg)] = {
                "frags": _frag_map(frags),
                "pred_rt": float(pep.pred_rt) if pep.pred_rt is not None else float("nan"),
            }

    store.n_hit = len(store._d)
    store.n_miss = len(store.wanted) - store.n_hit
    return store
```

- [ ] **Step 4: Run test to verify it passes**

Run: `python -m pytest tests/test_pred_store.py -q`
Expected: PASS (6 passed)

- [ ] **Step 5: Commit**

```bash
git add workflows/pred_store.py tests/test_pred_store.py
git commit -m "feat(pred): streaming PredStore builder (hits-only, O(hits) mem)"
```

---

## Task 7: Sanity-gate pure core (distribution + decision)

**Files:**
- Create: `tools/speclib_sanity.py`
- Test: `tests/test_speclib_sanity.py`

- [ ] **Step 1: Write the failing test**

```python
# tests/test_speclib_sanity.py
import math
from workflows.pred_features import spectral_angle
from tools.speclib_sanity import similarity_distribution, gate_pass


def test_distribution_of_identical_pairs_is_one():
    pairs = [([1.0, 2.0, 3.0], [1.0, 2.0, 3.0]),
             ([0.5, 0.2, 0.9], [5.0, 2.0, 9.0])]
    stats = similarity_distribution(pairs, metric=spectral_angle)
    assert stats["n"] == 2
    assert abs(stats["median"] - 1.0) < 1e-9


def test_distribution_skips_nan_pairs():
    # second pair is degenerate (length 1) -> NaN -> excluded
    pairs = [([1.0, 2.0], [1.0, 2.0]), ([1.0], [1.0])]
    stats = similarity_distribution(pairs, metric=spectral_angle)
    assert stats["n"] == 1


def test_gate_pass_threshold():
    good = {"n": 100, "median": 0.82, "p25": 0.7, "p75": 0.9}
    bad = {"n": 100, "median": 0.4, "p25": 0.2, "p75": 0.6}
    empty = {"n": 0, "median": float("nan"), "p25": float("nan"), "p75": float("nan")}
    assert gate_pass(good, min_sim=0.7) is True
    assert gate_pass(bad, min_sim=0.7) is False
    assert gate_pass(empty, min_sim=0.7) is False
```

- [ ] **Step 2: Run test to verify it fails**

Run: `python -m pytest tests/test_speclib_sanity.py -q`
Expected: FAIL with `ModuleNotFoundError: No module named 'tools.speclib_sanity'`

- [ ] **Step 3: Write minimal implementation**

```python
# tools/speclib_sanity.py
"""Sanity gate (spec §4.0): does the library's predicted fragment intensity
agree with the observed *light* spectrum on confident PSMs? Run this BEFORE
building any predicted-intensity features — if it fails, fix alignment /
units / mod mapping first.

The pure core (similarity_distribution, gate_pass) is unit-tested; main()
wires it to a real library + raw for the manual go/no-go run.
"""
import numpy as np

from workflows.pred_features import spectral_angle, spearman_sim


def similarity_distribution(pairs, metric=spectral_angle) -> dict:
    """pairs: iterable of (pred_vec, obs_vec) aligned over a fragment set.
    Returns {n, median, p25, p75} over finite similarities."""
    sims = []
    for pred_vec, obs_vec in pairs:
        s = metric(pred_vec, obs_vec)
        if np.isfinite(s):
            sims.append(float(s))
    if not sims:
        return {"n": 0, "median": float("nan"),
                "p25": float("nan"), "p75": float("nan")}
    arr = np.asarray(sims, dtype=float)
    return {
        "n": int(arr.size),
        "median": float(np.median(arr)),
        "p25": float(np.percentile(arr, 25)),
        "p75": float(np.percentile(arr, 75)),
    }


def gate_pass(stats, min_sim) -> bool:
    """Gate passes iff we have data and the median similarity clears min_sim."""
    return bool(stats["n"] > 0 and np.isfinite(stats["median"])
                and stats["median"] > min_sim)


_METRICS = {"spectral_angle": spectral_angle, "spearman": spearman_sim}
```

- [ ] **Step 4: Run test to verify it passes**

Run: `python -m pytest tests/test_speclib_sanity.py -q`
Expected: PASS (3 passed)

- [ ] **Step 5: Commit**

```bash
git add tools/speclib_sanity.py tests/test_speclib_sanity.py
git commit -m "feat(sanity): predicted-vs-observed similarity distribution + gate"
```

---

## Task 8: Sanity-gate CLI wiring

**Files:**
- Modify: `tools/speclib_sanity.py`
- Test: `tests/test_speclib_sanity.py`

- [ ] **Step 1: Write the failing test**

```python
# append to tests/test_speclib_sanity.py
import subprocess
import sys


def test_cli_help_exits_zero():
    r = subprocess.run(
        [sys.executable, "-m", "tools.speclib_sanity", "--help"],
        capture_output=True, text=True)
    assert r.returncode == 0
    assert "--library-dir" in r.stdout
    assert "--min-sim" in r.stdout


def test_build_observed_pred_pairs_aligns_on_common_fragments():
    # pred has fragments b1, y2; observed has y2, y3 -> only y2 aligns
    from workflows.pred_store import frag_key
    from tools.speclib_sanity import build_pairs_from_maps
    pred_map = {frag_key("b", 0, 1): 1.0, frag_key("y", 1, 1): 0.4}
    obs_map = {frag_key("y", 1, 1): 50.0, frag_key("y", 2, 1): 10.0}
    pred_vec, obs_vec = build_pairs_from_maps(pred_map, obs_map)
    assert pred_vec == [0.4]
    assert obs_vec == [50.0]
```

- [ ] **Step 2: Run test to verify it fails**

Run: `python -m pytest tests/test_speclib_sanity.py -q`
Expected: FAIL with `ImportError: cannot import name 'build_pairs_from_maps'` (and the `--help` test fails because there is no `main`/`__main__`).

- [ ] **Step 3: Write minimal implementation**

```python
# add to tools/speclib_sanity.py
import argparse
import configparser
import logging
import os
import tempfile

from spectrum.speclib import SpecLib
from spectrum.psm_info import HeavyType
from manager.light_result_manager import LightResultManager
from manager import data_manager
from workflows.pred_store import (
    build_pred_store, normalize_key, frag_key, frag_pos_for_ion)


def build_pairs_from_maps(pred_map: dict, obs_map: dict):
    """Align a predicted {frag_key: intensity} map with an observed
    {frag_key: intensity} map on their common fragments; return
    (pred_vec, obs_vec) as parallel lists in a stable fragment order."""
    common = sorted(set(pred_map) & set(obs_map))
    pred_vec = [pred_map[k] for k in common]
    obs_vec = [obs_map[k] for k in common]
    return pred_vec, obs_vec


def _observed_light_map(psm, dia_data, xic_cycle_window, mass_tol_ppm) -> dict:
    """{frag_key: apex_intensity} for the PSM's light b/y fragments.

    Uses frag_charge=1 (singly-charged) to match the dominant predicted
    fragments; Phase 2 may extend to multi-charge (spec J6).
    """
    out = {}
    seq_len = len(psm._sequence)
    b_ions, y_ions = psm.get_fragment_ions(HeavyType.SILAC)
    for ion_type, ion_num, light_mass, _heavy_mass in (b_ions + y_ions):
        xic, _all = dia_data.xic_ms2_peaks_extract(
            psm._rt, xic_cycle_window,
            precursor_mz=psm._precursor_mz,
            ions_mass=light_mass, mass_tol_ppm=mass_tol_ppm)
        if xic is None or len(xic) == 0:
            continue
        import numpy as np
        apex = float(np.nanmax(xic["intensity"])) if np.any(
            np.isfinite(xic["intensity"])) else 0.0
        if apex > 0:
            # b/y ordinal -> 0-indexed cleavage site; y is REVERSED
            frag_pos = frag_pos_for_ion(ion_type, ion_num, seq_len)
            out[frag_key(ion_type, frag_pos, 1)] = apex
    return out


def main():
    parser = argparse.ArgumentParser(
        description="Speclib sanity gate: predicted vs observed light similarity")
    parser.add_argument("--library-dir", required=True)
    parser.add_argument("--fasta", required=True)
    parser.add_argument("--mod", required=True)
    parser.add_argument("--psm-file", required=True,
                        help="confident light PSMs (same loader as main pipeline)")
    parser.add_argument("--search-engine-type", type=int, default=3)
    parser.add_argument("--raw", required=True, help="DIA mzML for observed light")
    parser.add_argument("--metric", choices=list(_METRICS), default="spectral_angle")
    parser.add_argument("--min-sim", type=float, default=0.7)
    parser.add_argument("--mass-tol-ppm", type=float, default=10.0)
    parser.add_argument("--xic-cycle-window", type=int, default=6)
    parser.add_argument("--limit", type=int, default=2000)
    args = parser.parse_args()

    logging.basicConfig(level=logging.INFO)

    # Build a minimal config the existing managers understand, so observed
    # intensities are extracted exactly like the main pipeline (centroiding,
    # tolerances). Keys verified against main.py / pair_flow.py usage.
    cfg = configparser.ConfigParser()
    cfg["input"] = {
        "search_engine_type": str(args.search_engine_type),
        "light_result_file": args.psm_file,
        "pfind_qvalue_threshold": "0.01",
    }
    cfg["general"] = {
        "mass_tol_ppm": str(args.mass_tol_ppm),
        "xic_cycle_window": str(args.xic_cycle_window),
        "centroid_enabled": "true",
        "centroid_rel_threshold": "0.001",
    }

    lib = SpecLib.open_dir(args.library_dir, fasta_path=args.fasta,
                           mod_path=args.mod)

    lrm = LightResultManager(config=cfg)
    light_result = lrm.get_light_result_object(args.psm_file)
    psms = list(light_result.psm_info)[:args.limit]

    wanted = {normalize_key(p._sequence, p._modify, p._charge) for p in psms}
    store = build_pred_store(lib, wanted)
    logging.info("speclib coverage: hit=%d miss=%d", store.n_hit, store.n_miss)

    tmp_pickle = os.path.join(tempfile.mkdtemp(), "raw_manager.pkl")
    dm = data_manager.DataManager(cfg, path=tmp_pickle)
    dia_data = dm.get_dia_data_object(args.raw)
    pairs = []
    for p in psms:
        rec = store.get(normalize_key(p._sequence, p._modify, p._charge))
        if rec is None:
            continue
        obs_map = _observed_light_map(p, dia_data, args.xic_cycle_window,
                                      args.mass_tol_ppm)
        pred_vec, obs_vec = build_pairs_from_maps(rec["frags"], obs_map)
        if len(pred_vec) >= 2:
            pairs.append((pred_vec, obs_vec))

    stats = similarity_distribution(pairs, metric=_METRICS[args.metric])
    passed = gate_pass(stats, args.min_sim)
    logging.info("sanity stats: %s", stats)
    logging.info("GATE %s (min_sim=%.2f, metric=%s)",
                 "PASS" if passed else "FAIL", args.min_sim, args.metric)
    raise SystemExit(0 if passed else 2)


if __name__ == "__main__":
    main()
```

> **Note for the implementer:** The loaders mirror `main.py`/`pair_flow.py`:
> `LightResultManager(config=cfg).get_light_result_object(path)` returns a
> `LightResult` whose PSMs are iterated via `.psm_info`; `data_manager.DataManager(cfg, path=...).get_dia_data_object(raw)` returns the `DIAData`.
> These names are verified against `manager/light_result_manager.py`,
> `manager/data_manager.py`, and `workflows/pair_flow.py:81,85,232`. The unit
> tests cover only the pure helpers (`build_pairs_from_maps`, distribution,
> gate); the data-wired `main()` is exercised by the manual go/no-go run below.

- [ ] **Step 2b: Verify the module imports cleanly**

Run: `python -c "import tools.speclib_sanity"`
Expected: no error (confirms the manager/speclib imports resolve in this environment).

- [ ] **Step 3: Run tests to verify they pass**

Run: `python -m pytest tests/test_speclib_sanity.py -q`
Expected: PASS (5 passed)

- [ ] **Step 4: Manual go/no-go run (documented, not a unit test)**

Run (on the real library + raw):
```bash
python -m tools.speclib_sanity \
  --library-dir <谱库目录> --fasta merge_human_ecoli_yeast.fasta --mod modification.ini \
  --psm-file <confident PSMs> --search-engine-type 3 \
  --raw <DIA.mzML> --metric spectral_angle --min-sim 0.7
```
Expected: prints coverage + `sanity stats` + `GATE PASS/FAIL`. If FAIL, debug b/y↔frag_pos alignment, charge, units, and mod-id mapping before Phase 2.

- [ ] **Step 5: Commit**

```bash
git add tools/speclib_sanity.py tests/test_speclib_sanity.py
git commit -m "feat(sanity): CLI wiring for predicted-vs-observed-light gate"
```

---

## Task 9: Documentation

**Files:**
- Modify: `docs/code/parts/workflows/L4_api.md`, `docs/code/parts/workflows/L3_details.md`, `docs/code/parts/workflows/L2_role.md`
- Modify: `docs/speclib/L1_overview.md`

- [ ] **Step 1: Add L4 API entries**

Append to `docs/code/parts/workflows/L4_api.md` a `## workflows/pred_features.py` section
documenting `spectral_angle`, `spearman_sim`, `_weighted_pearson`,
`select_topk_separable`, `i1_pattern_features`, and a `## workflows/pred_store.py`
section documenting `normalize_mods`, `normalize_key`, `frag_key`, `PredStore`,
`build_pred_store` — each with its one-line signature + behavior (mirror the
exact signatures implemented in Tasks 1–6).

- [ ] **Step 2: Add L3 detail + L2 role notes**

In `L3_details.md` add a subsection "谱库预测强度特征（Phase 1 基础）" describing the
metric definitions (SA/Spearman/weighted-Pearson), top-K-separable selection, the
constructed predicted-heavy vector for I1, and the streaming hits-only PredStore.
In `L2_role.md` add `pred_features.py` and `pred_store.py` rows to the module table,
and note `tools/speclib_sanity.py` as the go/no-go gate.

- [ ] **Step 3: Cross-link from speclib L1**

In `docs/speclib/L1_overview.md`, under the "未接入 pipeline" note, add a line:
"Phase 1 特征基础（`workflows/pred_features.py`/`pred_store.py`）+ sanity gate
（`tools/speclib_sanity.py`）已就绪；接入主流程见
`docs/specs/2026-06-08-speclib-predicted-intensity-features-design.md` 的 Phase 2。"

- [ ] **Step 4: Commit**

```bash
git add docs/code/parts/workflows/*.md docs/speclib/L1_overview.md
git commit -m "docs: L1-L4 entries for Phase 1 pred-intensity feature foundation"
```

---

## Self-Review

**Spec coverage (against `docs/specs/2026-06-08-...-design.md`):**
- §4.0 sanity gate → Tasks 7–8 ✅
- §4.1 lookup/PredStore + key normalization + b/y↔frag_pos alignment → Tasks 5–6 (alignment encoded as `ion_num-1` in `_observed_light_map`, validated by the gate) ✅
- §4.2 top-K selection → Task 3 ✅ (predicted-intensity *weighting* in aggregation is exercised via `_weighted_pearson`; the per-fragment pipeline weighting lands in Phase 2 integration)
- §4.3 I1 → Task 4 ✅
- §7 metrics → Tasks 1–2 ✅
- §4.4 I2, §4.5 I3, §4.6 J2, §4.7 J5, §4.8 perf, result.csv columns, config keys → **Phase 2** (explicitly out of scope for this plan; see note below).

**Placeholder scan:** No TBD/TODO; every code step shows full code; the one cross-module risk (loader method names) has an explicit verify step (Task 8 Step 2b). ✅

**Type consistency:** `frag_key` shape `(ion_type, frag_pos, frag_charge)` is identical in `pred_store.py`, `_frag_map`, `_observed_light_map`, and tests. `normalize_key` shape `(sequence, sorted-mods, int charge)` consistent across Tasks 5–8. `PredStore.get` returns `{"frags":..., "pred_rt":...}` consistently. ✅

**Phase 2 (separate plan, after the gate passes):** wire `PredStore` into `pair_flow.py` startup + `single_work.py` fragment loop; emit columns I1/I2/I3/J2 + `has_lib_pred`; add J5 adaptive floor behind a config flag in `q1a_helpers.py`; add the `pred_extract_only_topk` speedup; add `[speclib]` config section. Out of scope here.

---

## Execution Handoff

(Filled by the writing-plans skill at hand-off time.)

---

## Execution Notes (deviations found during implementation)

- **Task 1 `spectral_angle`:** the perfect-match case landed at `0.99999999`
  (arccos amplifies sub-1e-12 rounding near cos=1), failing a `<1e-9`
  assertion. Fix: after the `min/max` clamp, snap `cos` to exactly ±1.0 when
  within `1e-12`. No change to orthogonal/degenerate semantics.
- **Task 6 `tests/test_pred_store.py`:** predicted intensities round-trip
  through float32 in the binary (`0.3 → 0.30000001…`), so exact `== 0.3`
  fails. Fix: compare intensities with `pytest.approx(...)` (add `import pytest`).
- **Task 8 `_observed_light_map` (Critical, found in final review):** y-ion →
  `frag_pos` must be REVERSED. The library stores b and y at the same
  0-indexed cleavage site, but `get_fragment_ions` numbers y from the
  C-terminus, so observed `y_m` maps to `seq_len - m - 1`, not `m - 1`. Fixed
  by extracting a pure `frag_pos_for_ion(ion_type, ion_num, seq_len)` helper in
  `pred_store.py` (b: `ion_num-1`; y: `seq_len-ion_num-1`), unit-tested
  (incl. complementary b_i / y_{L-i} share-a-site), and used by both the gate
  and Phase-2 integration.
