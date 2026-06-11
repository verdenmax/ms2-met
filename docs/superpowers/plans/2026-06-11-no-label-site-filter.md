# No-Label-Site Filter Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Drop peptides with no SILAC label site (no K/R) at JSON-generation time in `extract_common`, for both targets and traps, via a labeling-scheme-aware rule (SILAC→require K/R; C13/N15→never filter).

**Architecture:** A shared pure helper `has_label_site(sequence, heavy_type)` in `spectrum/psm_info.py` (next to `HeavyType`). `tools/extract_common.py` reads a new `[extract] labeling` config (default `silac`), maps it to `HeavyType`, and filters `psms` before writing JSON. `tools/trap_domain_filter.py` reuses the shared helper (no behavior change).

**Tech Stack:** Python 3.14, pytest, configparser. Design: `docs/specs/2026-06-11-no-label-site-filter-design.md`.

---

### Task 1: Shared `has_label_site` helper in psm_info

**Files:**
- Modify: `spectrum/psm_info.py` (add function after `get_heavy_increase_mass`, ~line 263)
- Test: `tests/test_psm_info_label_site.py` (create)

- [ ] **Step 1: Write the failing test**

Create `tests/test_psm_info_label_site.py`:
```python
from spectrum.psm_info import has_label_site, HeavyType


def test_silac_requires_kr():
    assert has_label_site("PEPTIDEK", HeavyType.SILAC) is True   # ends in K
    assert has_label_site("PEPTIDER", HeavyType.SILAC) is True   # ends in R
    assert has_label_site("SAMPLERK", HeavyType.SILAC) is True   # internal R + K
    assert has_label_site("ACDEFGHILMNPQSTVWY", HeavyType.SILAC) is False  # no K/R
    assert has_label_site("LQEFLQHVS", HeavyType.SILAC) is False  # real pilot trap


def test_silac_is_default_heavy_type():
    assert has_label_site("PEPTIDEK") is True
    assert has_label_site("ACDEF") is False


def test_cheavy_nheavy_always_have_label_site():
    # whole-atom metabolic labeling: every peptide has C and N -> always labeled
    for ht in (HeavyType.CHEAVY, HeavyType.NHEAVY):
        assert has_label_site("ACDEF", ht) is True        # no K/R but has C/N
        assert has_label_site("PEPTIDEK", ht) is True


def test_empty_sequence_has_no_label_site():
    assert has_label_site("", HeavyType.SILAC) is False
    assert has_label_site("", HeavyType.CHEAVY) is False


def test_lowercase_is_normalized():
    assert has_label_site("peptidek", HeavyType.SILAC) is True
    assert has_label_site("acdef", HeavyType.SILAC) is False
```

- [ ] **Step 2: Run test to verify it fails**

Run: `cd /home/verden/pfind/2025-fall/code/ms2-met && python -m pytest tests/test_psm_info_label_site.py -q`
Expected: FAIL with `ImportError: cannot import name 'has_label_site'`.

- [ ] **Step 3: Write minimal implementation**

In `spectrum/psm_info.py`, add immediately after the `get_heavy_increase_mass` function (after line ~263):
```python
def has_label_site(sequence: str,
                   heavy_type: HeavyType = HeavyType.SILAC) -> bool:
    """Whether the peptide carries a metabolic-label site under heavy_type
    (i.e. whether light/heavy SILAC-style validation is even defined for it).

    SILAC labels only K/R, so a peptide with no K/R has no heavy partner
    (heavy == light) and is unvalidatable. CHEAVY (¹³C) / NHEAVY (¹⁵N) are
    whole-atom metabolic labeling — every peptide contains carbon and
    nitrogen (the N-Cα-C backbone), so every peptide is labeled. Empty
    sequence has no label site.
    """
    seq = str(sequence).upper()
    if not seq:
        return False
    if heavy_type == HeavyType.SILAC:
        return any(aa in "KR" for aa in seq)
    # CHEAVY / NHEAVY: every peptide has C and N -> always labeled.
    return True
```

- [ ] **Step 4: Run test to verify it passes**

Run: `cd /home/verden/pfind/2025-fall/code/ms2-met && python -m pytest tests/test_psm_info_label_site.py -q`
Expected: PASS (5 passed).

- [ ] **Step 5: Commit**

```bash
cd /home/verden/pfind/2025-fall/code/ms2-met
git add spectrum/psm_info.py tests/test_psm_info_label_site.py
git commit -m "feat(psm_info): scheme-aware has_label_site helper (SILAC=K/R, C13/N15=always)"
```

---

### Task 2: trap_domain_filter reuses the shared helper

**Files:**
- Modify: `tools/trap_domain_filter.py:35` (imports) and `:43-48` (remove local `has_label_site`)
- Test: `tests/test_trap_domain_filter.py` (existing — must still pass unchanged)

- [ ] **Step 1: Verify existing tests currently pass (baseline)**

Run: `cd /home/verden/pfind/2025-fall/code/ms2-met && python -m pytest tests/test_trap_domain_filter.py -q`
Expected: PASS (12 passed).

- [ ] **Step 2: Replace the local helper with the shared import**

In `tools/trap_domain_filter.py`, change the import line (currently line 35):
```python
from spectrum.entrapment_classifier import classify_peptide, load_target_fasta
from spectrum.psm_info import has_label_site  # noqa: F401  (re-exported for tests)
```

Then DELETE the local definition (currently lines 43–48):
```python
def has_label_site(sequence: str) -> bool:
    """True iff the peptide carries a SILAC label site (any K or R).

    A peptide with no K/R has no heavy partner (heavy == light), so the
    light/heavy validation is undefined for it (spec §12 class 4)."""
    return any(aa in "KR" for aa in str(sequence).upper())
```

The imported `has_label_site(sequence)` defaults to `HeavyType.SILAC`, so the 1-arg calls in `annotate_traps` (`has_label_site(row["sequence"])`) and in `test_trap_domain_filter.py` behave identically.

- [ ] **Step 3: Run tests to verify they still pass**

Run: `cd /home/verden/pfind/2025-fall/code/ms2-met && python -m pytest tests/test_trap_domain_filter.py -q`
Expected: PASS (12 passed) — `test_has_label_site_detects_kr` still green via the imported helper.

- [ ] **Step 4: Commit**

```bash
cd /home/verden/pfind/2025-fall/code/ms2-met
git add tools/trap_domain_filter.py
git commit -m "refactor(trap): reuse shared has_label_site from psm_info (no behavior change)"
```

---

### Task 3: extract_common labeling config + filter_by_label_site

**Files:**
- Modify: `tools/extract_common.py` — imports (line ~30), add `_LABELING_ALIASES`/`_parse_labeling`/`filter_by_label_site` (near `filter_by_entrapment`, ~line 446), wire into `extract_n_engines` (after line 539)
- Test: `tests/test_extract_label_site.py` (create)

- [ ] **Step 1: Write the failing test**

Create `tests/test_extract_label_site.py`:
```python
import configparser

import numpy as np
import pytest

from spectrum.psm_info import PSMInfo, HeavyType
from tools.extract_common import _parse_labeling, filter_by_label_site


def _psm(seq, label):
    return PSMInfo(sequence=seq, charge=2, modify=[], rt=np.float32(10.0),
                   precursor_mz=np.float32(500.0), raw_title="r1",
                   protein_names="X", label_type=label)


def _cfg(labeling=None):
    c = configparser.ConfigParser()
    c["extract"] = {} if labeling is None else {"labeling": labeling}
    return c


def test_parse_labeling_default_silac():
    assert _parse_labeling(_cfg()) == HeavyType.SILAC


def test_parse_labeling_aliases_case_insensitive():
    assert _parse_labeling(_cfg("SILAC")) == HeavyType.SILAC
    assert _parse_labeling(_cfg("c13")) == HeavyType.CHEAVY
    assert _parse_labeling(_cfg("13C")) == HeavyType.CHEAVY
    assert _parse_labeling(_cfg("cheavy")) == HeavyType.CHEAVY
    assert _parse_labeling(_cfg("n15")) == HeavyType.NHEAVY
    assert _parse_labeling(_cfg("15N")) == HeavyType.NHEAVY
    assert _parse_labeling(_cfg("nheavy")) == HeavyType.NHEAVY


def test_parse_labeling_missing_section_defaults_silac():
    assert _parse_labeling(configparser.ConfigParser()) == HeavyType.SILAC


def test_parse_labeling_invalid_raises():
    with pytest.raises(ValueError, match="labeling"):
        _parse_labeling(_cfg("itraq"))


def test_filter_silac_drops_no_kr_both_classes():
    psms = [_psm("PEPTIDEK", "positive"),   # has K -> keep
            _psm("ACDEF", "positive"),        # no K/R, target -> DROP
            _psm("SAMPLER", "negative"),      # has R -> keep
            _psm("ACDEF", "negative")]        # no K/R, trap -> DROP
    kept = filter_by_label_site(psms, HeavyType.SILAC)
    seqs = [(p._sequence, p._label_type) for p in kept]
    assert seqs == [("PEPTIDEK", "positive"), ("SAMPLER", "negative")]


def test_filter_cheavy_keeps_everything():
    psms = [_psm("PEPTIDEK", "positive"), _psm("ACDEF", "negative")]
    kept = filter_by_label_site(psms, HeavyType.CHEAVY)
    assert len(kept) == 2   # whole-atom labeling -> no-op
```

- [ ] **Step 2: Run test to verify it fails**

Run: `cd /home/verden/pfind/2025-fall/code/ms2-met && python -m pytest tests/test_extract_label_site.py -q`
Expected: FAIL with `ImportError: cannot import name '_parse_labeling'`.

- [ ] **Step 3: Add imports**

In `tools/extract_common.py`, change the psm_info import (currently `from spectrum.psm_info import PSMInfo`, ~line 30):
```python
from spectrum.psm_info import PSMInfo, HeavyType, has_label_site
```

- [ ] **Step 4: Add the parser and filter**

In `tools/extract_common.py`, add immediately before `def extract_n_engines(` (currently ~line 512, after `filter_by_entrapment` ends at line 509):
```python
_LABELING_ALIASES = {
    "silac": HeavyType.SILAC,
    "c13": HeavyType.CHEAVY, "13c": HeavyType.CHEAVY, "cheavy": HeavyType.CHEAVY,
    "n15": HeavyType.NHEAVY, "15n": HeavyType.NHEAVY, "nheavy": HeavyType.NHEAVY,
}


def _parse_labeling(config: configparser.ConfigParser) -> HeavyType:
    """Read [extract] labeling (default 'silac'); map to HeavyType.

    Accepts case-insensitive aliases: silac; c13/13c/cheavy; n15/15n/nheavy.
    Raises ValueError on an unknown value.
    """
    raw = "silac"
    if config.has_section("extract"):
        raw = config["extract"].get("labeling", "silac")
    key = str(raw).strip().lower()
    if key not in _LABELING_ALIASES:
        raise ValueError(
            f"非法 [extract] labeling={raw!r}（合法: {sorted(_LABELING_ALIASES)}）")
    return _LABELING_ALIASES[key]


def filter_by_label_site(psms: list, heavy_type: HeavyType) -> list:
    """Drop PSMs (both target and trap) with no metabolic-label site under
    heavy_type — they cannot be light/heavy validated (spec §12 class 4).

    Under SILAC this drops no-K/R peptides; under CHEAVY/NHEAVY every peptide
    is labeled so nothing is dropped.
    """
    kept = []
    dropped_pos = 0
    dropped_neg = 0
    for psm in psms:
        if has_label_site(psm._sequence, heavy_type):
            kept.append(psm)
            continue
        if psm._label_type == "negative":
            dropped_neg += 1
        else:
            dropped_pos += 1
    logging.info(
        f"label-site 过滤({heavy_type.name}): 剔除 positive={dropped_pos}, "
        f"negative={dropped_neg}, 输出={len(kept)}")
    return kept
```

- [ ] **Step 5: Wire into extract_n_engines**

In `tools/extract_common.py`, in `extract_n_engines`, immediately after the line `psms = extract_n_engines_from_psms_dual(engine_psms, engine_order, positive_marker)` (currently lines 538–539), add:
```python
    # Domain-of-applicability filter: drop peptides with no metabolic-label
    # site (spec §12 class 4). Runs unconditionally, both classes.
    psms = filter_by_label_site(psms, _parse_labeling(config))
```

- [ ] **Step 6: Run test to verify it passes**

Run: `cd /home/verden/pfind/2025-fall/code/ms2-met && python -m pytest tests/test_extract_label_site.py -q`
Expected: PASS (7 passed).

- [ ] **Step 7: Run the extract_common regression tests**

Run: `cd /home/verden/pfind/2025-fall/code/ms2-met && python -m pytest tests/ -q -k "extract or entrapment"`
Expected: PASS (no regressions; previously 74 passed — now more with the new file).

- [ ] **Step 8: Commit**

```bash
cd /home/verden/pfind/2025-fall/code/ms2-met
git add tools/extract_common.py tests/test_extract_label_site.py
git commit -m "feat(extract): scheme-aware no-label-site filter at JSON generation"
```

---

### Task 4: Documentation

**Files:**
- Modify: `docs/specs/2026-06-08-speclib-predicted-intensity-features-design.md` (§12.4)

- [ ] **Step 1: Update spec §12.4**

In `docs/specs/2026-06-08-speclib-predicted-intensity-features-design.md`, replace the §12.4 paragraph that currently says class 4 is "本期只对 trap 侧过滤" with:
```markdown
### 12.4 当前实现范围

**类1（L0/L1）+ 类3（重标出窗）+ 类4（无标记位点）已实现。** 其中**类4（无 K/R / 无标记位点）已落地 `extract_common`（JSON 生成阶段，正负例都剔），且 scheme-aware**：由 `[extract] labeling`（缺省 `silac`）驱动——SILAC 剔无 K/R；CHEAVY(¹³C)/NHEAVY(¹⁵N) 全原子标记下每条肽都被标记 → 不剔（no-op）。判据集中在 `spectrum/psm_info.has_label_site(seq, heavy_type)`，`extract_common` 与 `tools/trap_domain_filter` 共用。类1 仍由 `extract_common` 的 `filter_by_entrapment`（`drop_levels=L0,L1`）在 JSON 阶段完成；**类2（污染物名单）留待后续**。剔除优先级（trap_domain_filter 评估侧）：类1 > 类4 > 类3。

> 设计文档：`docs/specs/2026-06-11-no-label-site-filter-design.md`。
```

- [ ] **Step 2: Commit**

```bash
cd /home/verden/pfind/2025-fall/code/ms2-met
git add docs/specs/2026-06-08-speclib-predicted-intensity-features-design.md
git commit -m "docs(spec): §12.4 — no-label-site (class 4) now in extract_common, scheme-aware"
```

---

## Final verification

- [ ] **Run the full suite**

Run: `cd /home/verden/pfind/2025-fall/code/ms2-met && python -m pytest tests/ -q`
Expected: all pass except the 4 known pre-existing unrelated failures (`test_rescore_tool.py` ×3, `test_training_matrix.py` ×1).

- [ ] **Push**

```bash
cd /home/verden/pfind/2025-fall/code/ms2-met
git push origin feature_extraction
git push gitlab feature_extraction
```
