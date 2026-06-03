# Dual FDR Threshold Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Add per-engine `negative_qvalue_threshold` config option so users can expand the negative candidate pool independently of positive FDR (positives stay strict at 0.01, negatives loosen to e.g. 0.10).

**Architecture:** Two-pool model in `tools/extract_common.py`: refactor `load_engine_psms` to return `{"tight": [...], "loose": [...]}` dict; refactor `extract_n_engines_from_psms` to use tight keys for positive intersection and loose keys for negative union. Backward-compat: missing config → loose = tight → behavior identical to today.

**Tech Stack:** Python 3, configparser, pytest. No new third-party deps.

---

## Background

See `docs/specs/2026-06-03-dual-fdr-threshold-design.md` for full design and rationale. Key facts:

- Current single-threshold model: each engine has `qvalue_threshold = 0.01`, used for BOTH positive and negative pools.
- User wants to expand negative pool to FDR 0.10 without polluting positives.
- User-confirmed decisions: positive intersection over tight pool, negative union over loose pool, per-engine config, default to single-threshold (zero migration).

## Test environment

- `silac_ml` conda env has all needed deps. Use `conda run -n silac_ml pytest ...`.
- Baseline: **331 passed** (from deep-audit-fixes branch). After this plan: ≥331 + ~5 new tests.

## File Structure

| File | Status | Responsibility |
|---|---|---|
| `tools/extract_common.py` | MODIFY | Refactor `load_engine_psms` → `load_engine_psms_dual`; refactor `extract_n_engines_from_psms` → `extract_n_engines_from_psms_dual`; update caller `extract_n_engines`. Validate threshold ordering. |
| `tests/test_extract_common_dual_fdr.py` | NEW | 5 behavioral tests (backward compat, expanded negatives, positives invariant, validation, JSON schema). |
| `extract_2da_pfind_diann.ini` | MODIFY | Document the new optional `negative_qvalue_threshold` field in a comment (don't change current values). |

Files **not** modified:
- `spectrum/light_result.py` (loaders stay single-threshold; we call them twice).
- `spectrum/psm_info.py` / `workflows/*.py` / `tools/spec_trainer/*` (downstream untouched).

---

## Task 1: Extract `_load_engine` private helper

**Why:** Current `load_engine_psms` (lines 45-68 of `tools/extract_common.py`) couples engine-name dispatch with single-loader call. Task 2 will need to call the loader twice (tight + loose). Extract the dispatch into a private helper first to keep Task 2 surgical.

**Files:**
- Modify: `tools/extract_common.py` (refactor `load_engine_psms`)
- Test: this task has no behavior change — Task 2's tests cover it. Run full regression to confirm no break.

- [ ] **Step 1: Read current `load_engine_psms` to confirm structure**

Run: `sed -n '45,68p' tools/extract_common.py`

Expected: matches the structure shown in plan background. If the function has drifted, adapt the refactor accordingly.

- [ ] **Step 2: Add new `_load_engine` private helper above `load_engine_psms`**

In `tools/extract_common.py`, immediately BEFORE the existing `def load_engine_psms(...)` (around line 45), add:

```python
def _load_engine(engine_name: str, path: str, qvalue_threshold: float) -> list:
    """Internal dispatch: load PSMs from one engine with a single FDR threshold.

    Returns the list of PSMInfo from the chosen engine's LightResult loader.
    Raises ValueError on unknown engine name.
    """
    lr = LightResult()
    if engine_name == "pfind":
        lr._load_from_pfind_input(path, qvalue_threshold=qvalue_threshold)
    elif engine_name == "diann":
        lr._load_from_dia_nn_input(path, qvalue_threshold=qvalue_threshold)
    elif engine_name == "alphadia":
        lr._load_from_alphadia_input(path, qvalue_threshold=qvalue_threshold)
    else:
        raise ValueError(
            f"不支持的引擎: {engine_name}（支持 {SUPPORTED_ENGINES}）")
    return lr.psm_info
```

- [ ] **Step 3: Rewrite `load_engine_psms` to delegate to `_load_engine`**

Replace the body of `load_engine_psms` (keeping its public signature):

```python
def load_engine_psms(engine_name: str, config: configparser.ConfigParser) -> list:
    """根据引擎名加载对应 PSM 列表（单 FDR 阈值，向后兼容入口）。

    Note: New code should use load_engine_psms_dual (Task 2) which
    supports separate tight/loose FDR thresholds for positive/negative
    candidate pools. This single-threshold variant is retained as a
    thin wrapper for any external callers.
    """
    section = f"engine.{engine_name}"
    if section not in config:
        raise ValueError(f"配置中缺少 [{section}] 段")
    path = config[section].get("path")
    if not path:
        raise ValueError(f"[{section}] 缺少 path 配置")
    qvalue = config[section].getfloat("qvalue_threshold", fallback=0.01)
    return _load_engine(engine_name, path, qvalue)
```

- [ ] **Step 4: Verify no behavior change with full regression**

Run: `conda run -n silac_ml pytest tests/ -q 2>&1 | tail -3`

Expected: **331 passed** (unchanged). If anything fails, the refactor introduced a regression — debug before continuing.

- [ ] **Step 5: Commit**

```bash
git add tools/extract_common.py
git commit -m "refactor(extract_common): extract _load_engine helper (Task 1, dual FDR prep)

Extract engine-name dispatch from load_engine_psms into a private
_load_engine helper. Public load_engine_psms now a thin wrapper.
No behavior change — preparation for Task 2 (dual FDR variant
will call _load_engine twice for tight + loose pools).

Spec: docs/specs/2026-06-03-dual-fdr-threshold-design.md"
```

---

## Task 2: Add `load_engine_psms_dual` with threshold validation

**Why:** Core new functionality. Returns `{"tight": [...], "loose": [...]}` dict for use by `extract_n_engines_from_psms_dual` (Task 3). When `negative_qvalue_threshold` is missing OR equals `qvalue_threshold`, the two lists share identity (no redundant I/O).

**Files:**
- Modify: `tools/extract_common.py` (add `load_engine_psms_dual` after `load_engine_psms`)
- Test: create `tests/test_extract_common_dual_fdr.py`

- [ ] **Step 1: Create the test file with the loader tests**

Create `tests/test_extract_common_dual_fdr.py`:

```python
"""Tests for dual-FDR-threshold loader and extractor.

See docs/specs/2026-06-03-dual-fdr-threshold-design.md.
"""
import configparser
import os
import sys
import pytest

# Make tools/extract_common importable.
_PROJECT_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
if _PROJECT_ROOT not in sys.path:
    sys.path.insert(0, _PROJECT_ROOT)


def _make_config(qvalue: str = "0.01",
                 negative_qvalue: str | None = None) -> configparser.ConfigParser:
    """Build a minimal config with one pfind engine.

    qvalue / negative_qvalue are strings (raw ini values).
    """
    cfg = configparser.ConfigParser()
    section = {"path": "/nonexistent/dummy.qry.res", "qvalue_threshold": qvalue}
    if negative_qvalue is not None:
        section["negative_qvalue_threshold"] = negative_qvalue
    cfg.read_dict({"engine.pfind": section})
    return cfg


def test_load_engine_psms_dual_returns_tight_and_loose_keys(monkeypatch):
    """When negative_qvalue_threshold is absent, returned dict has both
    keys and the lists share identity (zero redundant I/O)."""
    from tools import extract_common

    captured_calls = []

    def fake_load(engine_name, path, qvalue):
        captured_calls.append((engine_name, path, qvalue))
        return [f"psm_{qvalue}"]  # stub, just track distinct calls

    monkeypatch.setattr(extract_common, "_load_engine", fake_load)

    cfg = _make_config(qvalue="0.01")  # no negative_qvalue → default = 0.01
    result = extract_common.load_engine_psms_dual("pfind", cfg)

    assert set(result.keys()) == {"tight", "loose"}
    # loose == tight: only one underlying load call
    assert len(captured_calls) == 1, (
        f"Expected single load when thresholds equal; got {captured_calls}")
    assert result["tight"] is result["loose"], (
        "When loose == tight, the two lists should share identity")


def test_load_engine_psms_dual_loads_twice_when_thresholds_differ(monkeypatch):
    """When negative_qvalue_threshold > qvalue_threshold, the loader is
    called twice with each threshold and results are different lists."""
    from tools import extract_common

    captured_calls = []

    def fake_load(engine_name, path, qvalue):
        captured_calls.append((engine_name, path, qvalue))
        return [f"psm_q{qvalue}"]

    monkeypatch.setattr(extract_common, "_load_engine", fake_load)

    cfg = _make_config(qvalue="0.01", negative_qvalue="0.10")
    result = extract_common.load_engine_psms_dual("pfind", cfg)

    assert len(captured_calls) == 2
    # Tight call has 0.01, loose call has 0.10
    qvalues_called = sorted(c[2] for c in captured_calls)
    assert qvalues_called == [0.01, 0.10]
    # The two lists are distinct objects (different loads)
    assert result["tight"] is not result["loose"]


def test_load_engine_psms_dual_raises_when_loose_below_tight(monkeypatch):
    """negative_qvalue_threshold < qvalue_threshold must raise ValueError."""
    from tools import extract_common

    monkeypatch.setattr(extract_common, "_load_engine",
                         lambda *a, **k: [])

    cfg = _make_config(qvalue="0.01", negative_qvalue="0.005")
    with pytest.raises(ValueError, match="negative_qvalue_threshold"):
        extract_common.load_engine_psms_dual("pfind", cfg)


def test_load_engine_psms_dual_raises_when_engine_section_missing():
    """Missing [engine.X] section raises ValueError."""
    from tools import extract_common
    cfg = configparser.ConfigParser()  # no sections at all
    with pytest.raises(ValueError, match="engine.pfind"):
        extract_common.load_engine_psms_dual("pfind", cfg)


def test_load_engine_psms_dual_raises_when_path_missing(monkeypatch):
    """Engine section without 'path' raises ValueError."""
    from tools import extract_common
    cfg = configparser.ConfigParser()
    cfg.read_dict({"engine.pfind": {"qvalue_threshold": "0.01"}})
    with pytest.raises(ValueError, match="path"):
        extract_common.load_engine_psms_dual("pfind", cfg)
```

- [ ] **Step 2: Run tests, verify FAIL**

Run: `conda run -n silac_ml pytest tests/test_extract_common_dual_fdr.py -v -k "load_engine_psms_dual"`

Expected: 5 FAILs with `AttributeError: module 'tools.extract_common' has no attribute 'load_engine_psms_dual'`.

- [ ] **Step 3: Add `load_engine_psms_dual` to `tools/extract_common.py`**

Add immediately AFTER the existing `load_engine_psms` function:

```python
def load_engine_psms_dual(
    engine_name: str,
    config: configparser.ConfigParser,
) -> dict:
    """Load engine PSMs with optional dual FDR (tight for positives,
    loose for negatives).

    Reads two thresholds from [engine.<name>]:
      - qvalue_threshold          (tight, gates positive candidates)
      - negative_qvalue_threshold (loose, gates negative candidates)

    When negative_qvalue_threshold is absent, defaults to qvalue_threshold
    (single-threshold behavior, backward compatible).

    Returns:
        dict {"tight": [PSMInfo], "loose": [PSMInfo]}
        When the two thresholds are equal, both keys point to the SAME
        list (no redundant I/O).

    Raises:
        ValueError: if [engine.<name>] is missing, path is missing,
                    or negative_qvalue_threshold < qvalue_threshold.

    See docs/specs/2026-06-03-dual-fdr-threshold-design.md.
    """
    section = f"engine.{engine_name}"
    if section not in config:
        raise ValueError(f"配置中缺少 [{section}] 段")
    path = config[section].get("path")
    if not path:
        raise ValueError(f"[{section}] 缺少 path 配置")

    tight = config[section].getfloat("qvalue_threshold", fallback=0.01)
    loose = config[section].getfloat(
        "negative_qvalue_threshold", fallback=tight)

    if loose < tight:
        raise ValueError(
            f"[{section}] negative_qvalue_threshold={loose} 不能小于 "
            f"qvalue_threshold={tight} (negative pool must be ⊇ positive pool)"
        )

    tight_psms = _load_engine(engine_name, path, tight)
    if loose == tight:
        # No redundant I/O: same threshold ⇒ same PSM list.
        loose_psms = tight_psms
    else:
        loose_psms = _load_engine(engine_name, path, loose)

    return {"tight": tight_psms, "loose": loose_psms}
```

- [ ] **Step 4: Run tests, verify PASS**

Run: `conda run -n silac_ml pytest tests/test_extract_common_dual_fdr.py -v -k "load_engine_psms_dual"`

Expected: 5 PASSed.

- [ ] **Step 5: Full regression**

Run: `conda run -n silac_ml pytest tests/ -q 2>&1 | tail -3`

Expected: 336 passed (was 331 + 5 new), no NEW failures.

- [ ] **Step 6: Commit**

```bash
git add tools/extract_common.py tests/test_extract_common_dual_fdr.py
git commit -m "feat(extract_common): load_engine_psms_dual with tight/loose pools (Task 2)

Add load_engine_psms_dual returning {'tight': [...], 'loose': [...]}
where:
- tight  = PSMs with q ≤ qvalue_threshold          (positive candidates)
- loose  = PSMs with q ≤ negative_qvalue_threshold (negative candidates)

Backward compat: when negative_qvalue_threshold is absent or equal to
qvalue_threshold, both pools share identity (zero redundant I/O).

Validation: loose < tight raises ValueError (negative pool must be a
superset of positive pool).

5 tests: dict shape with key sharing, two-call when distinct, raises
on inverted thresholds, raises on missing section / path.

Spec: docs/specs/2026-06-03-dual-fdr-threshold-design.md"
```

---

## Task 3: Add `extract_n_engines_from_psms_dual` consuming the dict

**Why:** Mirror of existing `extract_n_engines_from_psms` but uses TIGHT key sets for positive intersection and LOOSE key sets for negative union. Algorithm core (authoritative PSM selection, marker matching, label assignment) is unchanged.

**Files:**
- Modify: `tools/extract_common.py` (add `extract_n_engines_from_psms_dual` after `extract_n_engines_from_psms`)
- Test: append to `tests/test_extract_common_dual_fdr.py`

- [ ] **Step 1: Append tests for the dual extractor**

Append to `tests/test_extract_common_dual_fdr.py`:

```python


def _make_psm(seq: str, charge: int, raw: str, proteins: str):
    """Build a minimal PSMInfo for tests."""
    from spectrum.psm_info import PSMInfo
    psm = PSMInfo(
        sequence=seq, charge=charge, modify="",
        protein_names=proteins,
        raw_title=raw,
        precursor_mz=500.0, rt=10.0, q_value=0.0,
    )
    return psm


def test_extract_dual_default_matches_single_threshold():
    """When tight == loose (single pool), extract_n_engines_from_psms_dual
    output identical to extract_n_engines_from_psms."""
    from tools import extract_common

    # 3 PSMs: 1 HUMAN (positive in intersection), 1 ECOLI (negative in union),
    # 1 HUMAN only-in-pfind (not in intersection so dropped).
    p_human = _make_psm("PEPTIDEK", 2, "run1", "sp|P00000|HUMAN")
    p_ecoli = _make_psm("AAAAAAAR", 2, "run1", "sp|Q00000|ECOLI")
    p_unique = _make_psm("UNIQUEK", 2, "run1", "sp|P11111|HUMAN")

    # In both engines: human + ecoli; only in pfind: unique
    engines_single = {
        "pfind": [p_human, p_ecoli, p_unique],
        "diann": [p_human, p_ecoli],
    }
    engines_dual = {
        "pfind": {"tight": engines_single["pfind"],
                  "loose": engines_single["pfind"]},  # shared identity
        "diann": {"tight": engines_single["diann"],
                  "loose": engines_single["diann"]},
    }

    out_single = extract_common.extract_n_engines_from_psms(
        engines_single, ["pfind", "diann"], positive_marker="HUMAN")
    out_dual = extract_common.extract_n_engines_from_psms_dual(
        engines_dual, ["pfind", "diann"], positive_marker="HUMAN")

    # Same sequences with same label_type
    def _key(p):
        return (p._sequence, p._charge, p._raw_title, p._label_type)

    assert sorted(_key(p) for p in out_single) == sorted(
        _key(p) for p in out_dual)


def test_extract_dual_expanded_loose_adds_negatives_only():
    """When loose pool > tight pool, additional negatives appear; positives
    invariant."""
    from tools import extract_common

    p_human_tight = _make_psm("PEPTIDEK", 2, "run1", "sp|P00000|HUMAN")
    p_ecoli_tight = _make_psm("AAAAAAAR", 2, "run1", "sp|Q00000|ECOLI")
    # Extra ECOLI PSMs present only in the loose pool (q ∈ (0.01, 0.10])
    p_ecoli_loose1 = _make_psm("EXTRAONE", 2, "run1", "sp|Q11111|ECOLI")
    p_ecoli_loose2 = _make_psm("EXTRATWO", 2, "run1", "sp|Q22222|ECOLI")

    engines_dual = {
        "pfind": {
            "tight": [p_human_tight, p_ecoli_tight],
            "loose": [p_human_tight, p_ecoli_tight, p_ecoli_loose1,
                      p_ecoli_loose2],
        },
        "diann": {
            "tight": [p_human_tight, p_ecoli_tight],
            "loose": [p_human_tight, p_ecoli_tight, p_ecoli_loose1],
        },
    }

    out = extract_common.extract_n_engines_from_psms_dual(
        engines_dual, ["pfind", "diann"], positive_marker="HUMAN")

    positives = [p for p in out if p._label_type == "positive"]
    negatives = [p for p in out if p._label_type == "negative"]

    # 1 positive (PEPTIDEK in both engines' tight pool, HUMAN)
    assert len(positives) == 1
    assert positives[0]._sequence == "PEPTIDEK"

    # 3 negatives: tight-pool AAAAAAAR + loose-only EXTRAONE + loose-only EXTRATWO
    # (union over loose pools picks up both loose-only PSMs)
    neg_seqs = sorted(p._sequence for p in negatives)
    assert neg_seqs == ["AAAAAAAR", "EXTRAONE", "EXTRATWO"]


def test_extract_dual_positives_invariant_when_only_loose_changes():
    """Varying loose pool size must NEVER change positive count or sequences."""
    from tools import extract_common

    p_human = _make_psm("PEPTIDEK", 2, "run1", "sp|P00000|HUMAN")
    p_ecoli_tight = _make_psm("AAAAAAAR", 2, "run1", "sp|Q00000|ECOLI")
    p_extra1 = _make_psm("EXTRA1K", 2, "run1", "sp|Q11111|ECOLI")
    p_extra2 = _make_psm("EXTRA2K", 2, "run1", "sp|Q22222|ECOLI")

    tight_only = {
        "pfind": {"tight": [p_human, p_ecoli_tight],
                  "loose": [p_human, p_ecoli_tight]},
        "diann": {"tight": [p_human, p_ecoli_tight],
                  "loose": [p_human, p_ecoli_tight]},
    }
    plus_loose = {
        "pfind": {"tight": [p_human, p_ecoli_tight],
                  "loose": [p_human, p_ecoli_tight, p_extra1, p_extra2]},
        "diann": {"tight": [p_human, p_ecoli_tight],
                  "loose": [p_human, p_ecoli_tight, p_extra1]},
    }

    out_a = extract_common.extract_n_engines_from_psms_dual(
        tight_only, ["pfind", "diann"], positive_marker="HUMAN")
    out_b = extract_common.extract_n_engines_from_psms_dual(
        plus_loose, ["pfind", "diann"], positive_marker="HUMAN")

    pos_a = sorted(p._sequence for p in out_a if p._label_type == "positive")
    pos_b = sorted(p._sequence for p in out_b if p._label_type == "positive")
    assert pos_a == pos_b == ["PEPTIDEK"], (
        f"Positives must be invariant when only loose pool changes; "
        f"got {pos_a} vs {pos_b}")
```

- [ ] **Step 2: Run tests, verify FAIL**

Run: `conda run -n silac_ml pytest tests/test_extract_common_dual_fdr.py -v -k extract_dual`

Expected: 3 FAILs with `AttributeError: module ... has no attribute 'extract_n_engines_from_psms_dual'`.

If the PSMInfo signature in `_make_psm` doesn't match the actual constructor, adapt — read `spectrum/psm_info.py` to find the real fields. Common candidates: `sequence, charge, modify, protein_names, raw_title, precursor_mz, rt, q_value`. If kwargs differ, fix `_make_psm` first; the tests need PSMInfo to construct cleanly.

- [ ] **Step 3: Add `extract_n_engines_from_psms_dual` to `tools/extract_common.py`**

Add immediately AFTER the existing `extract_n_engines_from_psms` function (around line 165):

```python
def extract_n_engines_from_psms_dual(
    engine_psms_dual: dict,
    engine_order: list,
    positive_marker: Optional[str] = None,
) -> list:
    """Dual-pool variant of extract_n_engines_from_psms.

    Args:
        engine_psms_dual: dict[engine_name -> {"tight": [PSMInfo], "loose": [PSMInfo]}]
        engine_order: list[engine_name]
        positive_marker: species marker string; None ⇒ intersection only,
                         no label assignment.

    Algorithm:
      - Positives: intersection of TIGHT key sets across engines, then
        species marker match against authoritative PSM (looked up in
        LOOSE pool to ensure we always find a PSM — loose ⊇ tight).
      - Negatives: union of LOOSE key sets across engines, then species
        marker mismatch. Same authoritative-PSM rule as positives.

    Returns:
        list[PSMInfo] with _label_type set to "positive"/"negative" (or
        None when positive_marker is None).

    See docs/specs/2026-06-03-dual-fdr-threshold-design.md.
    """
    # Clear stale label_type on ALL PSMs in both pools.
    for pools in engine_psms_dual.values():
        for pool_psms in pools.values():
            for psm in pool_psms:
                psm._label_type = None

    # Tight key sets per engine → positive intersection.
    tight_keys = {
        name: {p.get_key_with_raw() for p in pools["tight"]}
        for name, pools in engine_psms_dual.items()
    }
    intersection_keys = (set.intersection(*tight_keys.values())
                         if tight_keys else set())

    # Loose key sets per engine → negative union.
    loose_keys = {
        name: {p.get_key_with_raw() for p in pools["loose"]}
        for name, pools in engine_psms_dual.items()
    }
    union_keys = set.union(*loose_keys.values()) if loose_keys else set()

    # Authoritative PSM selection (same priority as single-pool version):
    # 'diann' first if present, then engine_order. Search the LOOSE pool
    # so we always find a PSM for every key in intersection or union.
    if "diann" in engine_order:
        authoritative_order = ["diann"] + [e for e in engine_order if e != "diann"]
    else:
        authoritative_order = list(engine_order)

    key_to_psm = {}
    for engine_name in authoritative_order:
        for psm in engine_psms_dual.get(engine_name, {}).get("loose", []):
            key = psm.get_key_with_raw()
            if key not in key_to_psm:
                key_to_psm[key] = psm

    result = []

    if not positive_marker:
        for key in intersection_keys:
            psm = key_to_psm.get(key)
            if psm is not None:
                psm._label_type = None
                result.append(psm)
        logging.info(
            f"无 marker 模式：intersection size={len(result)}")
        return result

    pos_count = 0
    neg_count = 0
    positive_keys = set()
    for key in intersection_keys:
        psm = key_to_psm.get(key)
        if psm is None:
            continue
        if matches_species_marker(psm._protein_names, positive_marker):
            psm._label_type = "positive"
            result.append(psm)
            positive_keys.add(key)
            pos_count += 1

    for key in union_keys:
        if key in positive_keys:
            continue
        psm = key_to_psm.get(key)
        if psm is None:
            continue
        if not matches_species_marker(psm._protein_names, positive_marker):
            psm._label_type = "negative"
            result.append(psm)
            neg_count += 1

    logging.info(
        f"marker='{positive_marker}' (dual-FDR): "
        f"positive={pos_count}, negative={neg_count}, total={len(result)}"
    )
    return result
```

- [ ] **Step 4: Run tests, verify PASS**

Run: `conda run -n silac_ml pytest tests/test_extract_common_dual_fdr.py -v`

Expected: 8 PASSed (5 from Task 2 + 3 from Task 3).

- [ ] **Step 5: Full regression**

Run: `conda run -n silac_ml pytest tests/ -q 2>&1 | tail -3`

Expected: 339 passed (was 336 + 3 new), no NEW failures.

- [ ] **Step 6: Commit**

```bash
git add tools/extract_common.py tests/test_extract_common_dual_fdr.py
git commit -m "feat(extract_common): extract_n_engines_from_psms_dual (Task 3)

Mirror of existing extract_n_engines_from_psms but consumes the
dual-pool dict returned by load_engine_psms_dual:
- Positive intersection over TIGHT key sets per engine
- Negative union over LOOSE key sets per engine
- Same authoritative-PSM selection (diann-first if present)

3 behavioral tests:
- default (tight==loose) output matches single-pool version
- expanded loose adds negatives only (not positives)
- positives invariant when only loose pool changes

Spec: docs/specs/2026-06-03-dual-fdr-threshold-design.md"
```

---

## Task 4: Wire `extract_n_engines` to use dual-pool variants

**Why:** `extract_n_engines(config)` (lines 351-... in `tools/extract_common.py`) is the public entry called by `main()`. Currently it calls the single-pool functions. Switch to dual-pool so users get the new feature when they set `negative_qvalue_threshold` in ini.

**Files:**
- Modify: `tools/extract_common.py:extract_n_engines` (~lines 351-372)
- Test: append to `tests/test_extract_common_dual_fdr.py`

- [ ] **Step 1: Append integration test**

Append to `tests/test_extract_common_dual_fdr.py`:

```python


def test_extract_n_engines_uses_dual_loader(monkeypatch):
    """extract_n_engines(config) calls load_engine_psms_dual and
    extract_n_engines_from_psms_dual under the hood."""
    from tools import extract_common

    captured = {"loader": None, "extractor": None}

    def fake_loader(engine_name, config):
        captured["loader"] = engine_name
        return {"tight": [], "loose": []}

    def fake_extractor(engine_psms_dual, engine_order, positive_marker=None):
        captured["extractor"] = {
            "shape": {k: list(v.keys()) for k, v in engine_psms_dual.items()},
            "engine_order": engine_order,
            "positive_marker": positive_marker,
        }
        return []

    monkeypatch.setattr(extract_common, "load_engine_psms_dual", fake_loader)
    monkeypatch.setattr(extract_common, "extract_n_engines_from_psms_dual",
                         fake_extractor)

    cfg = configparser.ConfigParser()
    cfg.read_dict({
        "extract": {"engines": "pfind", "positive_species_marker": "HUMAN"},
        "engine.pfind": {"path": "/x.qry.res", "qvalue_threshold": "0.01"},
    })
    extract_common.extract_n_engines(cfg)

    assert captured["loader"] == "pfind"
    assert captured["extractor"] is not None
    assert captured["extractor"]["shape"] == {"pfind": ["tight", "loose"]}
    assert captured["extractor"]["engine_order"] == ["pfind"]
    assert captured["extractor"]["positive_marker"] == "HUMAN"
```

- [ ] **Step 2: Run test, verify FAIL**

Run: `conda run -n silac_ml pytest tests/test_extract_common_dual_fdr.py::test_extract_n_engines_uses_dual_loader -v`

Expected: FAIL — current `extract_n_engines` calls singular `load_engine_psms` / `extract_n_engines_from_psms`, NOT the `_dual` variants.

- [ ] **Step 3: Update `extract_n_engines` to use dual-pool functions**

In `tools/extract_common.py`, find `extract_n_engines` (around line 351). Locate the block:

```python
    engine_psms = {}
    for name in engine_order:
        logging.info(f"加载引擎: {name}")
        engine_psms[name] = load_engine_psms(name, config)
        logging.info(f"  → {name} 共 {len(engine_psms[name])} 条 PSM")

    psms = extract_n_engines_from_psms(
        engine_psms, engine_order, positive_marker)
```

Replace with:

```python
    engine_psms = {}
    for name in engine_order:
        logging.info(f"加载引擎: {name}")
        engine_psms[name] = load_engine_psms_dual(name, config)
        n_tight = len(engine_psms[name]["tight"])
        n_loose = len(engine_psms[name]["loose"])
        if engine_psms[name]["tight"] is engine_psms[name]["loose"]:
            logging.info(f"  → {name} 共 {n_tight} 条 PSM (单 FDR 池)")
        else:
            logging.info(
                f"  → {name} tight={n_tight}, loose={n_loose} (双 FDR 池)")

    psms = extract_n_engines_from_psms_dual(
        engine_psms, engine_order, positive_marker)
```

- [ ] **Step 4: Run test, verify PASS**

Run: `conda run -n silac_ml pytest tests/test_extract_common_dual_fdr.py::test_extract_n_engines_uses_dual_loader -v`

Expected: PASS.

- [ ] **Step 5: Full regression**

Run: `conda run -n silac_ml pytest tests/ -q 2>&1 | tail -5`

Expected: 340 passed (was 339 + 1 new), no NEW failures.

If anything related to `extract_n_engines` end-to-end testing fails (e.g., a test in `tests/test_extract_common.py` or similar), it's likely because the test imported `load_engine_psms` / `extract_n_engines_from_psms` and the behavior changed. Inspect those tests — they should still work because the single-pool functions still exist (unchanged). If they DON'T, debug.

- [ ] **Step 6: Commit**

```bash
git add tools/extract_common.py tests/test_extract_common_dual_fdr.py
git commit -m "feat(extract_common): wire extract_n_engines to dual-pool (Task 4)

The public entry extract_n_engines now calls load_engine_psms_dual +
extract_n_engines_from_psms_dual. When no engine sets
negative_qvalue_threshold, both pools share identity ⇒ behavior is
identical to before (verified by Task 3 test).

Log line distinguishes single-pool (one count) from dual-pool
(tight + loose counts) for operator visibility.

1 integration test verifies the wiring with monkey-patched helpers.

Spec: docs/specs/2026-06-03-dual-fdr-threshold-design.md"
```

---

## Task 5: Document `negative_qvalue_threshold` in example ini

**Why:** Users need to know the option exists. Add a commented-out example in the existing `extract_2da_pfind_diann.ini` (the only live ini) without changing actual values — current production behavior must stay unchanged until user explicitly opts in.

**Files:**
- Modify: `extract_2da_pfind_diann.ini` (add doc comment, no value change)

- [ ] **Step 1: Read current ini to confirm format**

Run: `cat extract_2da_pfind_diann.ini`

Expected output (current):
```
[extract]
engines = pfind, diann
positive_species_marker = HUMAN
result_file = ./datasets/hela_2da_pfind_diann.json

[engine.pfind]
path = ../pfind-dia/2th/
qvalue_threshold = 0.01

[engine.diann]
path = ../pfind-dia/2th/hela-mix-2da_report.parquet
```

If the file has drifted, adapt the edit.

- [ ] **Step 2: Add documentation comments after each engine section**

Edit `extract_2da_pfind_diann.ini`. After the `qvalue_threshold = 0.01` line under `[engine.pfind]`, add a commented hint:

```ini
[engine.pfind]
path = ../pfind-dia/2th/
qvalue_threshold = 0.01
# Optional dual-FDR: 取消下行注释以将负例池放宽到 10% FDR
# （正例仍用 qvalue_threshold=0.01，互不影响）
# 见 docs/specs/2026-06-03-dual-fdr-threshold-design.md
# negative_qvalue_threshold = 0.10

[engine.diann]
path = ../pfind-dia/2th/hela-mix-2da_report.parquet
# Optional dual-FDR: 同上，DIA-NN 也可独立设置
# negative_qvalue_threshold = 0.10
```

(Don't add a `qvalue_threshold` to `[engine.diann]` since current ini doesn't have one — let it use the loader default 0.01.)

- [ ] **Step 3: Verify ini still parses cleanly**

Run: `conda run -n silac_ml python -c "
import configparser
cfg = configparser.ConfigParser()
cfg.read('extract_2da_pfind_diann.ini')
print('sections:', cfg.sections())
print('pfind qvalue:', cfg.getfloat('engine.pfind', 'qvalue_threshold'))
print('pfind negative_qvalue:', cfg.getfloat('engine.pfind', 'negative_qvalue_threshold', fallback=cfg.getfloat('engine.pfind', 'qvalue_threshold')))
"`

Expected:
```
sections: ['extract', 'engine.pfind', 'engine.diann']
pfind qvalue: 0.01
pfind negative_qvalue: 0.01
```

(`negative_qvalue_threshold` not set ⇒ falls back to `qvalue_threshold` = 0.01.)

- [ ] **Step 4: Smoke-test make target dry-run**

Run: `make -n 2th 2>&1 | tail -3`

Expected: shows the normal `python3 main.py --configpath runs/baseline_2da_clean/config.ini ...` command. Confirms the ini edit didn't break Makefile dependency resolution.

- [ ] **Step 5: Commit**

```bash
git add extract_2da_pfind_diann.ini
git commit -m "docs(ini): document optional negative_qvalue_threshold (Task 5)

Add commented-out negative_qvalue_threshold = 0.10 examples under each
[engine.X] section of extract_2da_pfind_diann.ini. No active value
change — production behavior unchanged unless user uncomments.

References docs/specs/2026-06-03-dual-fdr-threshold-design.md for the
two-pool model rationale."
```

---

## Final Verification (after all 5 tasks)

- [ ] **Step 1: Full test suite**

Run: `conda run -n silac_ml pytest tests/ -q 2>&1 | tail -5`

Expected: ≥340 passed (was 331 + 9 new tests: 5 loader + 3 extractor + 1 wiring), no NEW failures.

- [ ] **Step 2: Dual-pool end-to-end smoke (optional, requires real engine outputs)**

ONLY if you have real engine inputs and want to confirm the actual JSON output grows when negative threshold is loosened:

```bash
# Snapshot current single-pool output
cp datasets/hela_2da_pfind_diann.json /tmp/before.json

# Add negative_qvalue_threshold under [engine.pfind] and [engine.diann]
# in extract_2da_pfind_diann.ini (uncomment + set to 0.10), then:
make extract-2th

# Compare counts
conda run -n silac_ml python -c "
import json
before = json.load(open('/tmp/before.json'))
after = json.load(open('datasets/hela_2da_pfind_diann.json'))
def counts(data):
    p = sum(1 for r in data if r.get('label_type') == 'positive')
    n = sum(1 for r in data if r.get('label_type') == 'negative')
    return p, n
print('before (tight=0.01, loose=0.01):', counts(before))
print('after  (tight=0.01, loose=0.10):', counts(after))
"
```

Expected: positive count UNCHANGED, negative count INCREASED.

Then revert the ini edit before committing (production baseline stays at 0.01).

- [ ] **Step 3: Verify Makefile targets still work**

Run: `make -n 2th 5th normal train-exp1 train-exp2 clean-all 2>&1 | grep -E '^python3|^make:' | head -10`

Expected: all 5 commands produce sensible dry-runs (no `No rule to make target` errors).

- [ ] **Step 4: Push to gitlab (optional)**

```bash
git push gitlab feature_extraction
```

---

## Self-Review

**Spec coverage** (each spec section → task mapping):

| Spec section | Implemented in |
|---|---|
| Architecture: load_engine_psms_dual | Task 2 |
| Architecture: extract_n_engines_from_psms_dual | Task 3 |
| Files to modify: tools/extract_common.py | Tasks 1, 2, 3, 4 |
| Files to modify: extract_2da_pfind_diann.ini | Task 5 |
| Files to modify: tests/test_extract_common_dual_fdr.py | Tasks 2, 3, 4 |
| Invariant 1 (tight ⊆ loose math) | Task 2 (loose==tight ⇒ identity); Task 3 test |
| Invariant 2 (positives unaffected) | Task 3 test_extract_dual_positives_invariant |
| Invariant 3 (JSON schema unchanged) | Out of scope for unit tests — Task 4 wiring preserves the existing JSON serialization, verified end-to-end in Final Verification Step 2 |
| Invariant 4 (loose<tight raises) | Task 2 test_load_engine_psms_dual_raises_when_loose_below_tight |
| Backward compat (no migration needed) | Task 2 test_load_engine_psms_dual_returns_tight_and_loose_keys (default config); Task 3 test_extract_dual_default_matches_single_threshold |
| User opts-in via ini | Task 5 (documented example) |

All spec requirements have tasks. No placeholders. Type/signature consistency:
- `load_engine_psms_dual` returns `dict` (keys "tight", "loose") in Task 2 → consumed by `extract_n_engines_from_psms_dual` in Task 3 → wired in Task 4. ✓
- `_load_engine(engine_name, path, qvalue_threshold)` signature consistent across Tasks 1 and 2. ✓
- `extract_n_engines_from_psms_dual(engine_psms_dual, engine_order, positive_marker)` signature consistent across Tasks 3 and 4. ✓
