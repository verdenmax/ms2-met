# Review Fixes (R3+R4 + Makefile + spec_trainer) Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Resolve 1 Critical + 7 Important findings from the 2026-06-03 three-prong code review (R3+R4 features, Makefile/runs structure, spec_trainer integration) without changing user-visible feature semantics.

**Architecture:** Eight independent surgical fixes grouped in three phases. Phase 1 patches Python code (feature wiring + spec_trainer testability). Phase 2 hardens the Makefile (json-path extraction, conditional ini deps, work_directory isolation, conservative clean). Phase 3 adds `test_size` config for held-out evaluation in spec_trainer.

**Tech Stack:** Python 3, pandas, scipy, scikit-learn, pytest, GNU Make, configparser, PyYAML.

---

## Background

Three code-review subagents audited the feature_extraction branch on 2026-06-03 and reported:

- **review-features-r3-r4** (⚠ NEEDS_ATTENTION): 1 Important — `multi_batch_work` is missing 4 R4 precursor peak-likeness columns that `single_pair_work` already emits, so the two code paths emit different schemas.
- **review-makefile-runs** (❌ BLOCKED): 1 Critical — `extract_5da_pfind_diann.ini` and `extract_normal_pfind_diann.ini` do not exist in the repo, so `make 5th / normal / all / train-exp2` cannot build from a clean tree. Plus 3 Important findings around silent JSON-path extraction, `-j` workspace races, and `clean-*` blast radius.
- **review-spec-trainer-integration** (⚠ NEEDS_ATTENTION): 3 Important — three unit tests for `_resolve_feature_cols` are all `pytest.skip`'d because importing `main.py` transitively requires lightgbm (not installed locally); `test_files == train_files` produces in-sample AUC; `model.save()` has no `mkdir -p` so direct `python main.py` invocation crashes.

User has decided:
- **C1** (missing ini files): adopt strategy "A" — make Makefile tolerate missing `extract_*.ini` by skipping the extract dependency when the ini is absent, falling back to the already-present `features.csv`. Do NOT fabricate stub ini files.
- **I-ST2** (in-sample test): adopt strategy "A" — add `test_size` config field; when present and `test_files` is empty/missing, split a held-out set from `train_files` using `sklearn.model_selection.train_test_split(stratify=y)`.

## File Structure

### Files to modify
- `workflows/single_work.py` — add 4 R4 precursor field assignments to `multi_batch_work` (both empty-XIC branch and computed branch).
- `tests/test_single_work_numerics.py` — add a regression test that asserts schema parity between `multi_batch_work` and `single_pair_work`.
- `tools/spec_trainer/src/feature_cols.py` (**new**) — extract `_resolve_feature_cols`, `META_COLUMNS`, `EXCLUDED_EXTRA` so tests load this lightweight module instead of full `main.py`.
- `tools/spec_trainer/src/main.py` — import from new `feature_cols.py`, add `mkdir -p` before `model.save`, add `test_size` handling in `main()`.
- `tests/test_spec_trainer_main.py` — switch to importing `feature_cols` directly (no lightgbm dep), assert tests are actually executed (not skipped) on this machine.
- `tools/spec_trainer/config/exp1.yaml` and `exp2.yaml` — remove `test_files`, add `test_size: 0.2` under `data:`.
- `Makefile` — JSON-path extraction with quote/comment/whitespace tolerance, conditional extract dep when ini missing, conservative `clean-*` recipes, `.NOTPARALLEL` safety guard.
- `main.py` — read `work_directory` from `[general]` config, fallback to `./workspace`.
- `runs/baseline_{2da,5da,normal}_clean/config.ini` — set distinct `work_directory` paths to avoid parallel-make collisions.

### Files to create
- `tools/spec_trainer/src/feature_cols.py` — helper module.
- Test additions inside existing test files (no new test files needed besides modifications).

---

## Task 1: I-F1 — Mirror 4 R4 precursor peak-likeness fields into `multi_batch_work`

**Why:** `single_pair_work` (workflows/single_work.py:404-435) writes 4 R4 precursor fields. `multi_batch_work` (same file:55-94) only writes R3 fields. Downstream consumers expect identical schemas. The bug is silent — pandas concat will fill missing columns with NaN and LightGBM will silently treat them as a feature with all-NaN values for `multi_batch_work` rows.

**Files:**
- Modify: `workflows/single_work.py` (lines 55-94 — `multi_batch_work` precursor block, both branches)
- Test: `tests/test_single_work_numerics.py` (append a new schema-parity test)

- [ ] **Step 1: Write failing behavioral schema-parity test**

Append to `tests/test_single_work_numerics.py`:

```python
def test_multi_batch_work_emits_R4_precursor_keys_in_empty_xic_branch():
    """multi_batch_work must emit the 4 R4 precursor keys (I-F1 regression).

    Trigger the empty-XIC branch (no fragment data available) by giving
    a PSM whose precursor mz / rt resolves to an empty xic_peaks_extreact
    result. We assert the returned features dict contains the 4 R4 keys
    that single_pair_work already emits.

    This is a behavioral test (not source-string match) so a typo in the
    key name would be caught.
    """
    import configparser
    import numpy as np
    from workflows.single_work import multi_batch_work
    from common.psm_info import PSMInfo
    from spectrum.dia_data import DIAData

    # Minimal config — only values needed for multi_batch_work code path.
    cfg = configparser.ConfigParser()
    cfg.read_dict({
        "general": {
            "mass_tol_ppm": "20",
            "xic_cycle_window": "5",
        },
    })

    # Build a PSM whose precursor mz is far outside any realistic window
    # so dia_data.xic_peaks_extreact returns an empty array, forcing
    # multi_batch_work into the empty-XIC branch (lines 55-70 of single_work.py).
    psm = PSMInfo(
        sequence="AAAA", charge=2,
        precursor_mz=1e9,  # impossibly large -> no MS1 match
        rt=10.0,
        raw_title="fake",
        label=1, label_type="target",
    )

    # An empty DIAData backed by zero spectra — xic_peaks_extreact must
    # return an empty structured array on any query.
    dia_data = DIAData()  # Default constructor; should support empty state.

    features = multi_batch_work(psm, dia_data, psm, dia_data, cfg)

    # The 4 R4 precursor keys that were missing per finding I-F1:
    R4_PRECURSOR_KEYS = {
        "precursor_base_to_apex_ratio",
        "precursor_apex_monotonicity",
        "precursor_n_peaks",
        "precursor_smoothness",
    }
    missing = R4_PRECURSOR_KEYS - set(features.keys())
    assert not missing, (
        f"multi_batch_work missing R4 precursor keys {missing} (I-F1). "
        f"Schema drift with single_pair_work — concat will produce NaN columns.")
```

**Note for implementer:** If the `DIAData()` default constructor or `PSMInfo(...)` signature differs from above, adapt minimally — the test only needs an empty-XIC trigger. As a fallback, use `monkeypatch.setattr` on `dia_data.xic_peaks_extreact` to return an empty numpy structured array directly.

- [ ] **Step 2: Run test to verify it fails**

Run: `conda run -n jianyan pytest tests/test_single_work_numerics.py::test_multi_batch_work_emits_R4_precursor_keys_in_empty_xic_branch -v`
Expected: FAIL with assertion `multi_batch_work missing R4 precursor keys {...} (I-F1)`.

If the test errors at construction time (e.g. PSMInfo / DIAData signature mismatch), debug by reading `common/psm_info.py` and `spectrum/dia_data.py` to find the actual ctor; the test must FAIL on the keys-missing assertion, not on import / construction.

- [ ] **Step 3: Patch `multi_batch_work` empty-XIC branch (lines 55-70)**

In `workflows/single_work.py`, after line 70 (`features["precursor_heavy_apex_cycle_offset_signed"] = 0`), insert:

```python
        features["precursor_base_to_apex_ratio"] = 0.0
        features["precursor_apex_monotonicity"] = 0.0
        features["precursor_n_peaks"] = 0
        features["precursor_smoothness"] = 0.0
```

- [ ] **Step 4: Patch `multi_batch_work` computed branch (after line 94)**

After line 94 (`features["precursor_heavy_apex_cycle_offset_signed"] = (precursor_score["heavy_apex_cycle_offset_signed"])`), insert:

```python
        features["precursor_base_to_apex_ratio"] = (
            precursor_score["base_to_apex_ratio"])
        features["precursor_apex_monotonicity"] = (
            precursor_score["apex_monotonicity"])
        features["precursor_n_peaks"] = precursor_score["n_peaks"]
        features["precursor_smoothness"] = precursor_score["smoothness"]
```

- [ ] **Step 5: Run test to verify it passes + run full numerics suite**

Run:
```
conda run -n jianyan pytest tests/test_single_work_numerics.py -v 2>&1 | tail -20
```
Expected: the new behavioral parity test PASSes and all previously-passing tests still pass.

- [ ] **Step 6: Commit**

```bash
git add workflows/single_work.py tests/test_single_work_numerics.py
git commit -m "fix(single_work): mirror 4 R4 precursor fields into multi_batch_work

Review finding I-F1 (2026-06-03 audit): single_pair_work emits
precursor_base_to_apex_ratio / apex_monotonicity / n_peaks / smoothness
but multi_batch_work was missing them, causing schema drift between the
two code paths. Add to both empty-XIC branch (zeros) and computed branch
(read from precursor_score). Plus schema-parity regression test."
```

---

## Task 2: I-ST1 — Extract `feature_cols.py` so tests run without lightgbm

**Note on ordering:** This task must execute BEFORE Task 3 (I-ST3 mkdir). Task 2 rewrites `tests/test_spec_trainer_main.py` in full; Task 3 then APPENDS the mkdir-guard test to the rewritten file. Reverse order would silently drop Task 3's test (rubber-duck finding B1).

**Why:** All 3 unit tests for `_resolve_feature_cols` are `pytest.skip`'d because `_load_main_module()` execs full `main.py`, which imports `models.model_manager → lgb_model → lightgbm` (not installed locally). The harness exists but proves nothing. Fix by extracting the pure-function helper into a tiny module with zero ML deps.

**Files:**
- Create: `tools/spec_trainer/src/feature_cols.py`
- Modify: `tools/spec_trainer/src/main.py` (remove the inlined definition, import instead)
- Modify: `tests/test_spec_trainer_main.py` (import feature_cols directly, drop skip helper, assert tests truly run)

- [ ] **Step 1: Create `tools/spec_trainer/src/feature_cols.py`**

```python
"""Feature column resolution for spec_trainer.

Extracted from main.py so unit tests can exercise the logic without
importing lightgbm/sklearn (see review finding I-ST1, 2026-06-03 audit).
"""
import pandas as pd


# META columns that are not features themselves (PSM identification + label).
# 与 tools/eval_baseline.py:37-41 保持一致。
META_COLUMNS = {
    "sequence", "charge", "raw_title1", "raw_title2",
    "protein_names", "label", "label_type",
    "precursor_mz", "sequence_len",
}

# 额外排除的特征列：modification_count 在训练时倾向于过拟合非物理信号
# （负样本 entrapment 大多带修饰），见 PLAN.md 三-2 分析。
EXCLUDED_EXTRA = {"modification_count"}


def resolve_feature_cols(explicit, sample_csv_path, target_col):
    """Resolve final feature column list.

    If explicit is a non-empty list, return it unchanged (yaml took
    care of selection). Otherwise auto-detect from the CSV column
    header, excluding META_COLUMNS + EXCLUDED_EXTRA + target_col.

    The CSV column order determines the feature order (pandas read_csv
    is deterministic for a given file). Cross-runs with the same
    features.csv produce the same feature_cols list.
    """
    if explicit:
        return list(explicit)
    sample_df = pd.read_csv(sample_csv_path, nrows=0)
    all_cols = list(sample_df.columns)
    return [
        c for c in all_cols
        if c not in META_COLUMNS
        and c not in EXCLUDED_EXTRA
        and c != target_col
    ]
```

- [ ] **Step 2: Replace in `tools/spec_trainer/src/main.py`**

Find the block (lines 28-61 of current main.py):

```python
# META columns that are not features themselves (PSM identification + label).
# 与 tools/eval_baseline.py:37-41 保持一致。
META_COLUMNS = {
    "sequence", "charge", "raw_title1", "raw_title2",
    "protein_names", "label", "label_type",
    "precursor_mz", "sequence_len",
}

# 额外排除的特征列：modification_count 在训练时倾向于过拟合非物理信号
# （负样本 entrapment 大多带修饰），见 PLAN.md 三-2 分析。
EXCLUDED_EXTRA = {"modification_count"}


def _resolve_feature_cols(explicit, sample_csv_path, target_col):
    """Resolve final feature column list.

    If explicit is a non-empty list, return it unchanged (yaml took
    care of selection). Otherwise auto-detect from the CSV column
    header, excluding META_COLUMNS + EXCLUDED_EXTRA + target_col.

    The CSV column order determines the feature order (pandas read_csv
    is deterministic for a given file). Cross-runs with the same
    features.csv produce the same feature_cols list.
    """
    if explicit:
        return list(explicit)
    sample_df = pd.read_csv(sample_csv_path, nrows=0)
    all_cols = list(sample_df.columns)
    return [
        c for c in all_cols
        if c not in META_COLUMNS
        and c not in EXCLUDED_EXTRA
        and c != target_col
    ]
```

Replace with:

```python
# Feature column resolution moved to feature_cols.py for testability
# without lightgbm dependency (review finding I-ST1, 2026-06-03 audit).
from feature_cols import (
    META_COLUMNS,
    EXCLUDED_EXTRA,
    resolve_feature_cols as _resolve_feature_cols,
)
```

(Keep the name `_resolve_feature_cols` for the main.py call site so we don't have to touch `main()`.)

- [ ] **Step 3: Rewrite `tests/test_spec_trainer_main.py` to import feature_cols directly**

Replace the entire file with:

```python
"""Test spec_trainer feature column resolution (lightgbm-free).

After review finding I-ST1 (2026-06-03 audit), the helper is in
tools/spec_trainer/src/feature_cols.py and these tests import it
directly. Previously they tried to import all of main.py and skipped
on missing lightgbm.
"""
import os
import sys


_PROJECT_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
_SPEC_TRAINER_SRC = os.path.join(_PROJECT_ROOT, "tools", "spec_trainer", "src")

if _SPEC_TRAINER_SRC not in sys.path:
    sys.path.insert(0, _SPEC_TRAINER_SRC)

from feature_cols import resolve_feature_cols  # noqa: E402


def test_resolve_feature_cols_explicit_list_passthrough():
    """When yaml provides explicit feature_cols list, return it unchanged."""
    result = resolve_feature_cols(
        explicit=["a", "b", "c"],
        sample_csv_path="/nonexistent.csv",
        target_col="label",
    )
    assert result == ["a", "b", "c"]


def test_resolve_feature_cols_empty_triggers_auto_detect(tmp_path):
    """Empty feature_cols triggers auto-detection from CSV header."""
    csv = tmp_path / "fake.csv"
    csv.write_text(
        "sequence,charge,protein_names,label,precursor_mz,sequence_len,"
        "raw_title1,raw_title2,label_type,modification_count,"
        "precursor_pearson,b_mean,y_p50\n"
    )
    result = resolve_feature_cols(
        explicit=[],
        sample_csv_path=str(csv),
        target_col="label",
    )
    assert result == ["precursor_pearson", "b_mean", "y_p50"]
    assert "label" not in result
    assert "modification_count" not in result
    assert "precursor_mz" not in result
    assert "sequence_len" not in result
    assert "raw_title1" not in result
    assert "protein_names" not in result


def test_resolve_feature_cols_none_triggers_auto_detect(tmp_path):
    """None feature_cols (yaml missing key) also triggers auto-detection."""
    csv = tmp_path / "fake.csv"
    csv.write_text(
        "label,precursor_pearson,b_mean,modification_count\n"
    )
    result = resolve_feature_cols(
        explicit=None,
        sample_csv_path=str(csv),
        target_col="label",
    )
    assert result == ["precursor_pearson", "b_mean"]
```

(Note: this rewrite drops the old `_load_main_module()` skip helper entirely. The mkdir test (I-ST3) will be added as a separate Task 3 after this task completes — by APPENDING to the rewritten file. This split avoids the issue where Task 2 would overwrite a Task-3-added test if done in the wrong order.)

- [ ] **Step 4: Run all 3 tests, verify ALL pass (not skip)**

Run: `conda run -n jianyan pytest tests/test_spec_trainer_main.py -v`

Expected output:
```
test_resolve_feature_cols_explicit_list_passthrough PASSED
test_resolve_feature_cols_empty_triggers_auto_detect PASSED
test_resolve_feature_cols_none_triggers_auto_detect PASSED
3 passed
```

If any test SKIPs, fix the import chain — the whole point of this task is no more skips.

- [ ] **Step 5: Smoke-test main.py still imports cleanly**

Run: `conda run -n jianyan python -c "import sys; sys.path.insert(0, 'tools/spec_trainer/src'); import feature_cols; print(feature_cols.META_COLUMNS)"` — must succeed.

(If you also want to verify main.py: `conda run -n jianyan python -c "import sys; sys.path.insert(0, 'tools/spec_trainer/src'); import main"` — this may error on lightgbm missing, which is fine; we only care that the new `from feature_cols import ...` line is syntactically sound.)

- [ ] **Step 6: Commit**

```bash
git add tools/spec_trainer/src/feature_cols.py tools/spec_trainer/src/main.py tests/test_spec_trainer_main.py
git commit -m "refactor(spec_trainer): extract feature_cols.py for lightgbm-free tests

Review finding I-ST1 (2026-06-03 audit): 3 unit tests for
_resolve_feature_cols were all pytest.skip'd because loading main.py
transitively requires lightgbm. Move META_COLUMNS / EXCLUDED_EXTRA /
resolve_feature_cols into feature_cols.py (no ML deps), import from
main.py. Rewrite tests to import the helper directly.

Verified: tests now run + pass on a machine without lightgbm."
```

---

## Task 3: I-ST3 — `model.save()` mkdir guard in spec_trainer main.py

**Why:** Direct invocation `python tools/spec_trainer/src/main.py --config ...` crashes with FileNotFoundError because `model.save(model_path)` writes to `runs/spec_trainer/models/expN.txt` and that directory only exists if the Makefile `train-*` recipe pre-created it. The yaml-driven path contract (commit ff2033a) is undermined.

**Ordering:** Must run AFTER Task 2 (Task 2 rewrites the test file; this task APPENDS to it).

**Files:**
- Modify: `tools/spec_trainer/src/main.py:226-227` (add mkdir before model.save)
- Modify: `tests/test_spec_trainer_main.py` (append source-inspection test)

- [ ] **Step 1: Append failing source-inspection test**

Append to the END of `tests/test_spec_trainer_main.py` (which was rewritten in Task 2):

```python


def test_main_creates_model_output_directory():
    """main.py must mkdir -p the model output directory before save().

    Regression for review finding I-ST3 (2026-06-03): model.save() had no
    mkdir, so direct python invocation crashed when runs/spec_trainer/models/
    didn't exist (Makefile pre-created it, masking the bug).
    """
    src_path = os.path.join(_SPEC_TRAINER_SRC, "main.py")
    src = open(src_path).read()
    assert "os.makedirs(os.path.dirname(model_path)" in src, (
        "main.py is missing mkdir guard before model.save (I-ST3)")
```

- [ ] **Step 2: Run test to verify it fails**

Run: `conda run -n jianyan pytest tests/test_spec_trainer_main.py::test_main_creates_model_output_directory -v`
Expected: FAIL — source does not yet contain the mkdir call.

- [ ] **Step 3: Add mkdir before `model.save(model_path)`**

In `tools/spec_trainer/src/main.py`, find:

```python
    # Save model
    model_path = cfg['output']['model_path']
    model.save(model_path)
```

Replace with:

```python
    # Save model (ensure parent dir exists; direct python invocation must
    # not rely on Makefile pre-creating runs/spec_trainer/models/).
    model_path = cfg['output']['model_path']
    os.makedirs(os.path.dirname(model_path) or ".", exist_ok=True)
    model.save(model_path)
```

- [ ] **Step 4: Run test to verify it passes (now 4 tests total in this file)**

Run: `conda run -n jianyan pytest tests/test_spec_trainer_main.py -v`

Expected output:
```
test_resolve_feature_cols_explicit_list_passthrough PASSED
test_resolve_feature_cols_empty_triggers_auto_detect PASSED
test_resolve_feature_cols_none_triggers_auto_detect PASSED
test_main_creates_model_output_directory PASSED
4 passed
```

- [ ] **Step 5: Commit**

```bash
git add tools/spec_trainer/src/main.py tests/test_spec_trainer_main.py
git commit -m "fix(spec_trainer): mkdir -p before model.save(model_path)

Review finding I-ST3 (2026-06-03 audit): model.save() crashed on direct
python invocation when runs/spec_trainer/models/ didn't exist. Makefile
was pre-creating it, masking the bug. Add explicit makedirs to honor the
yaml-driven path contract (commit ff2033a)."
```

---

## Task 4: I-MK1 — Robust JSON path extraction from extract_*.ini

**Why:** Current Makefile lines 37-39 use `grep | head -1 | cut -d= -f2- | tr -d ' '`. Problems:
- `tr -d ' '` strips spaces from paths (a path with a space becomes garbage).
- Does not strip quotes.
- Does not strip inline `#comment`.
- If `result_file=` is absent, `$(JSON_*)` is empty string and the target chain becomes silently no-op.

**Files:**
- Modify: `Makefile` (lines 34-39)

- [ ] **Step 1: Define a robust extraction shell function in Makefile**

Replace the existing block (lines 34-39):

```make
# 从 extract_*.ini 中动态抽取 result_file 路径。
# grep 找 ^result_file= 行 -> 取 = 右侧 -> 去前后空格
# 如果 ini 不存在，结果为空，target 会在依赖检查时报缺失文件错。
JSON_2TH    := $(strip $(shell test -f $(INI_2TH)    && grep -E '^[[:space:]]*result_file[[:space:]]*=' $(INI_2TH)    | head -1 | cut -d= -f2- | tr -d ' '))
JSON_5TH    := $(strip $(shell test -f $(INI_5TH)    && grep -E '^[[:space:]]*result_file[[:space:]]*=' $(INI_5TH)    | head -1 | cut -d= -f2- | tr -d ' '))
JSON_NORMAL := $(strip $(shell test -f $(INI_NORMAL) && grep -E '^[[:space:]]*result_file[[:space:]]*=' $(INI_NORMAL) | head -1 | cut -d= -f2- | tr -d ' '))
```

With:

```make
# 从 extract_*.ini 中动态抽取 result_file 路径。
# 行格式：result_file = ./path/to/output.json    # optional comment
# 处理：
#   1) grep 抓首行匹配 ^result_file=
#   2) sed 砍掉 = 之前 + 行尾 #comment + 首尾空白 + 包围引号
#   3) 若 ini 不存在 -> 结果为空字符串（target 会用 ifeq 显式检查）
# 注意：保留路径中部的空格（用 sed 替代旧的 tr -d ' '）。
define EXTRACT_RESULT_FILE
$(strip $(shell test -f $(1) && \
    grep -E '^[[:space:]]*result_file[[:space:]]*=' $(1) | head -1 | \
    sed -E -e 's/^[^=]*=[[:space:]]*//' -e 's/[[:space:]]*#.*$$//' -e 's/^["'"'"']//' -e 's/["'"'"']$$//' -e 's/[[:space:]]+$$//'))
endef

JSON_2TH    := $(call EXTRACT_RESULT_FILE,$(INI_2TH))
JSON_5TH    := $(call EXTRACT_RESULT_FILE,$(INI_5TH))
JSON_NORMAL := $(call EXTRACT_RESULT_FILE,$(INI_NORMAL))
```

- [ ] **Step 2: Smoke-test extraction against the existing 2da ini**

Run: `make -n help 2>&1 | grep "2th"`
Expected: shows `2th    -> ./datasets/hela_2da_pfind_diann.json` (the existing path is reproduced correctly).

- [ ] **Step 3: Smoke-test against a fake ini with whitespace / comment / quotes**

```bash
cat > /tmp/fake_ini.ini <<'EOF'
[extract]
result_file = "./datasets/with space.json"  # this is a comment
EOF
make -n -f Makefile help INI_2TH=/tmp/fake_ini.ini 2>&1 | grep "2th"
```
Expected: shows `2th    -> ./datasets/with space.json` (quote and comment stripped; internal space preserved).

If the output is wrong, iterate on the `sed` chain until it is correct. Then `rm /tmp/fake_ini.ini`.

- [ ] **Step 4: Commit**

```bash
git add Makefile
git commit -m "fix(make): robust extract_*.ini result_file parsing (I-MK1)

Review finding I-MK1 (2026-06-03 audit): old extraction used 'tr -d \" \"'
which strips spaces from paths, and ignored #comments + quotes. Replace
with sed chain that:
- strips leading 'result_file = '
- strips inline '#comment'
- strips surrounding single/double quotes
- preserves spaces inside the path

Tested with both real extract_2da_pfind_diann.ini and a synthetic ini
containing space/quote/comment.

Known limitations (per rubber-duck N5, 2026-06-03):
- result_file paths must NOT contain '#' (will be truncated as comment)
- quotes must be balanced (unbalanced ones are silently stripped)
Both documented in the Makefile comment block."
```

---

## Task 5: C1 — Conditional extract dependency when ini is absent

**Why:** Currently `5th: $(INI_5TH) $(JSON_5TH) ...` declares `$(INI_5TH)` as a hard dependency. When the file does not exist, make immediately errors with "No rule to make target 'extract_5da_pfind_diann.ini'". User decision: when ini is absent, skip the extract step entirely and just verify the `features.csv` is present (or report a clean error pointing at the missing data).

**Files:**
- Modify: `Makefile` (5th and normal target chains)

- [ ] **Step 1: Wrap each ini-dependent stanza with `ifneq ($(wildcard ...),)`**

In `Makefile`, find the `5th` block (lines 101-112):

```make
# ---------- 5th ----------

$(JSON_5TH): $(INI_5TH)
	$(call BANNER,extract 5th)
	$(PY) tools/extract_common.py --configpath $(INI_5TH)

extract-5th: $(JSON_5TH)

5th: $(INI_5TH) $(JSON_5TH) $(DIR_5TH)/config.ini
	$(call BANNER,5th)
	$(PY) main.py --configpath $(DIR_5TH)/config.ini --logpath $(DIR_5TH)/extract.log
	@echo "[done] features written under $(DIR_5TH)/"
```

Replace with:

```make
# ---------- 5th ----------
#
# If extract_5da_pfind_diann.ini exists, declare full pipeline dependency
# (JSON regenerated when ini changes). Otherwise the JSON / features.csv
# is treated as externally-provided; we still run main.py if the user
# explicitly invokes `make 5th`, but only after verifying the necessary
# inputs already exist on disk.
ifneq ($(wildcard $(INI_5TH)),)

$(JSON_5TH): $(INI_5TH)
	$(call BANNER,extract 5th)
	$(PY) tools/extract_common.py --configpath $(INI_5TH)

extract-5th: $(JSON_5TH)

5th: $(INI_5TH) $(JSON_5TH) $(DIR_5TH)/config.ini
	$(call BANNER,5th)
	$(PY) main.py --configpath $(DIR_5TH)/config.ini --logpath $(DIR_5TH)/extract.log
	@echo "[done] features written under $(DIR_5TH)/"

else  # $(INI_5TH) absent — features.csv must be externally provided

extract-5th:
	@echo "[error] $(INI_5TH) not found — cannot extract; provide ini or use a pre-built JSON" >&2
	@false

5th: $(DIR_5TH)/config.ini
	$(call BANNER,5th)
	@if [ ! -f "$(DIR_5TH)/features.csv" ]; then \
		echo "[note] $(INI_5TH) absent — $(DIR_5TH)/features.csv must be present" >&2; \
		echo "       (extract step skipped; main.py will fail if light_result_file in config.ini is invalid)" >&2; \
	fi
	$(PY) main.py --configpath $(DIR_5TH)/config.ini --logpath $(DIR_5TH)/extract.log
	@echo "[done] features written under $(DIR_5TH)/"

endif
```

- [ ] **Step 2: Apply same `ifneq` pattern to `normal` block (lines 114-125)**

Mirror the same structure for `$(INI_NORMAL)` / `$(JSON_NORMAL)` / `$(DIR_NORMAL)` / `normal:` / `extract-normal:`.

- [ ] **Step 3: (No change to 2th — extract_2da_pfind_diann.ini exists.)**

But also wrap the `2th` block in the same `ifneq` for consistency (so behavior is uniform if anyone ever moves the 2da ini).

- [ ] **Step 4: Verify `make -n 2th` still works (ini present case)**

Run: `make -n 2th 2>&1 | head -10`
Expected: no errors, shows the expected `python3 main.py --configpath runs/baseline_2da_clean/config.ini ...` command.

- [ ] **Step 5: Verify `make -n 5th` succeeds (ini absent case)**

Run: `make -n 5th 2>&1 | head -10`
Expected: no error from make ("No rule to make target ..."); shows only the `main.py` invocation (no extract_common step).

- [ ] **Step 6: Verify `make extract-5th` reports clear error**

Run: `make extract-5th 2>&1 | head -5`
Expected: `[error] extract_5da_pfind_diann.ini not found ...` and exit code != 0.

- [ ] **Step 7: Update help text in Makefile**

Find the `help:` target. After the `make extract-normal` line, add a note:

```make
	@echo ""
	@echo "  注：extract-* 仅在对应 extract_*.ini 存在时可用。"
	@echo "      5th / normal 的 ini 默认未提供，features.csv 须外部生成。"
```

- [ ] **Step 8: Commit**

```bash
git add Makefile
git commit -m "fix(make): tolerate missing extract_5da/normal ini (C1)

Review finding C1 (2026-06-03 audit): the 5th / normal targets declared
extract_*.ini as hard deps, but only extract_2da_pfind_diann.ini exists
in-repo (5da / normal features.csv are externally generated). 'make 5th'
on a clean tree errored with 'No rule to make target ...'.

Wrap each dataset's extract chain in 'ifneq (\$(wildcard \$(INI)),)':
- ini present: full pipeline (extract + main.py)
- ini absent : only main.py, with a clean error if features.csv and
               light_result_file are also missing.

extract-5th / extract-normal report explicit errors when invoked
without the corresponding ini."
```

---

## Task 6: I-MK2 — Per-dataset work_directory (main.py reads config)

**Why:** All three `runs/baseline_*/config.ini` set `work_directory = ./workspace`, but `main.py:57` hardcodes `work_path="./workspace"` and ignores the config value entirely. Worse, `make -j all` would parallelize 2th/5th/normal, all writing to the same `./workspace/`. Two-part fix: (a) make `main.py` actually honor the config; (b) give each baseline a distinct workspace; (c) add `.NOTPARALLEL` as a belt-and-suspenders safety guard.

**Files:**
- Modify: `main.py` (read work_directory from config)
- Modify: `runs/baseline_{2da,5da,normal}_clean/config.ini` (set distinct work_directory)
- Modify: `Makefile` (add `.NOTPARALLEL:`)
- Test: `tests/test_main_work_directory.py` (new — verifies main.py uses the config value)

- [ ] **Step 1: Write failing test for `main.py` honoring `work_directory`**

Create `tests/test_main_work_directory.py`:

```python
"""Verify main.py reads work_directory from config (not hardcoded).

Regression for review finding I-MK2 (2026-06-03 audit): main.py
hardcoded work_path='./workspace', causing parallel make targets to
collide on the same workspace directory.
"""
import os


def test_main_uses_config_work_directory():
    """main.py source must read work_directory from the [general] config."""
    project_root = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
    src = open(os.path.join(project_root, "main.py")).read()
    # Look for the config-read pattern. Accept either configparser API or
    # the constant K.GENERAL.WORK_DIRECTORY (see constant/keys.py:28).
    assert (
        'work_directory' in src.lower() or 'WORK_DIRECTORY' in src
    ), "main.py does not appear to read work_directory from config (I-MK2)"
    # Specifically reject the old hardcoded literal as the sole source.
    # (The literal may still appear as a default fallback, so we only
    # check that the work_path argument is no longer a string literal.)
    assert 'work_path="./workspace"' not in src, (
        "main.py still uses hardcoded work_path='./workspace' literal "
        "as the sole value (I-MK2). Read from config with this as fallback.")
```

- [ ] **Step 2: Run test to verify it fails**

Run: `conda run -n jianyan pytest tests/test_main_work_directory.py -v`
Expected: FAIL on the second assertion (`work_path="./workspace"` is currently in source).

- [ ] **Step 3: Patch `main.py` to read work_directory from config**

In `main.py`, find:

```python
    # 进入 workflow , 系统的处理
    workflow = PairFlow(workname="main", config=config,
                        work_path="./workspace")
```

Replace with:

```python
    # 进入 workflow , 系统的处理
    # work_directory 从 [general] 读取，缺省 ./workspace。允许每个 baseline
    # config.ini 设置独立路径，避免并行 make 时多个 pipeline 写同一目录
    # (review finding I-MK2, 2026-06-03 audit)。
    # 显式抓 NoSectionError 因为 configparser 的 fallback 只覆盖缺失 option，
    # 不覆盖缺失 section（rubber-duck B2，2026-06-03）。
    try:
        work_path = config.get("general", "work_directory",
                                fallback="./workspace")
    except configparser.NoSectionError:
        work_path = "./workspace"
    workflow = PairFlow(workname="main", config=config,
                        work_path=work_path)
```

- [ ] **Step 4: Run test to verify it passes**

Run: `conda run -n jianyan pytest tests/test_main_work_directory.py -v`
Expected: PASS.

- [ ] **Step 5: Update each baseline config.ini with distinct work_directory**

Edit each of:
- `runs/baseline_2da_clean/config.ini`
- `runs/baseline_5da_clean/config.ini`
- `runs/baseline_normal_clean/config.ini`

Change `work_directory = ./workspace` to (respectively):
- `work_directory = ./runs/baseline_2da_clean/workspace`
- `work_directory = ./runs/baseline_5da_clean/workspace`
- `work_directory = ./runs/baseline_normal_clean/workspace`

- [ ] **Step 6: (skipped — `.NOTPARALLEL` dropped per rubber-duck N1)**

The original plan added `.NOTPARALLEL:` as belt-and-suspenders. Per rubber-duck review N1 (2026-06-03): globally disabling `make -j` is a user-visible regression, and the only documented race (shared `./workspace/`) is already eliminated by per-dataset workspaces in Step 5. Any further race in `tools/extract_common.py` would be speculative.

Decision: do NOT add `.NOTPARALLEL`. Users who run `make -j all` get correct results because each dataset writes to its own `runs/baseline_*_clean/workspace/`. If a real shared-cache race emerges later, scope the serialization to just the `all` target via order-only prereqs.

(This step is a no-op; included to keep step numbering consistent with prior plan revisions.)

- [ ] **Step 7: Verify `make -n all` still produces sequential plan**

Run: `make -n all 2>&1 | grep -E '^python3|^make\['`
Expected: shows three `python3 main.py ...` commands. Without `.NOTPARALLEL`, `make -j all` may interleave their stdout, but each writes to its own workspace (no data corruption).

- [ ] **Step 8: Commit**

```bash
git add main.py runs/baseline_2da_clean/config.ini runs/baseline_5da_clean/config.ini runs/baseline_normal_clean/config.ini tests/test_main_work_directory.py
git commit -m "fix(main): per-dataset work_directory from config (I-MK2)

Review finding I-MK2 (2026-06-03 audit):
- main.py hardcoded work_path='./workspace', ignoring the [general]
  work_directory config field entirely.
- All 3 baseline config.ini had identical work_directory = ./workspace.
- 'make -j all' would race the 3 pipelines on the same workspace.

Fix:
- main.py now reads config.get('general','work_directory',fallback=...)
  wrapped in try/except NoSectionError (rubber-duck B2: configparser
  fallback does NOT cover missing section, only missing option).
- 3 baseline configs use runs/baseline_<dataset>_clean/workspace each.

Migration note: the old shared ./workspace/ in repo root is now
orphaned. Delete it after upgrading: rm -rf ./workspace/

Note: .NOTPARALLEL was considered (rubber-duck N1) but rejected —
the race is fully resolved by per-dataset workspaces, and globally
disabling -j is too costly."
```

---

## Task 7: I-MK3 — Conservative `clean-*` recipes

**Why:** `clean-2th` currently uses `find $(DIR_2TH) -mindepth 1 -depth ! -path '$(DIR_2TH)/config.ini' -delete`, which wipes the entire baseline directory including `eval/` (hand-curated metrics), `workspace/` (cached intermediate data), and any future artifact. Help text claims it only deletes "features.csv / log". Tighten to match the documented intent.

**Files:**
- Modify: `Makefile` (clean-2th / clean-5th / clean-normal recipes)

- [ ] **Step 1: Tighten `clean-2th` to delete only features.csv + logs**

In `Makefile`, find:

```make
clean-2th:
	@if [ -d $(DIR_2TH) ]; then \
		find $(DIR_2TH) -mindepth 1 -depth ! -path '$(DIR_2TH)/config.ini' -delete; \
		echo "[cleaned] $(DIR_2TH)/ (kept config.ini)"; \
	else \
		echo "[skip] $(DIR_2TH)/ does not exist"; \
	fi
```

Replace with:

```make
clean-2th:
	@if [ -d $(DIR_2TH) ]; then \
		rm -f $(DIR_2TH)/features.csv $(DIR_2TH)/features.csv.PARTIAL_INCOMPLETE $(DIR_2TH)/*.log; \
		echo "[cleaned] $(DIR_2TH)/features.csv + *.log (kept config.ini, eval/, workspace/)"; \
	else \
		echo "[skip] $(DIR_2TH)/ does not exist"; \
	fi
```

- [ ] **Step 2: Same for `clean-5th` and `clean-normal`** (mirror the pattern).

- [ ] **Step 3: Update help text to be accurate**

Find the line in `help:` that currently says `make clean-2th       删除 2da features.csv / log，强制下次重跑` (and the two analogous lines for `clean-5th` / `clean-normal`). Confirm the wording still matches the new conservative recipe (it should — current wording says "features.csv / log" which is now accurate).

If the wording needs adjustment, fix it to say e.g. `删除 2da features.csv / *.log（保留 config.ini / eval/ / workspace/）`.

- [ ] **Step 4: Smoke-test in a fake directory**

```bash
mkdir -p /tmp/fake_baseline/eval /tmp/fake_baseline/workspace
touch /tmp/fake_baseline/features.csv /tmp/fake_baseline/extract.log /tmp/fake_baseline/config.ini /tmp/fake_baseline/eval/metrics.json /tmp/fake_baseline/workspace/cache.npz
make -f Makefile clean-2th DIR_2TH=/tmp/fake_baseline
ls -la /tmp/fake_baseline /tmp/fake_baseline/eval /tmp/fake_baseline/workspace
rm -rf /tmp/fake_baseline
```
Expected: `features.csv` and `extract.log` gone; `config.ini`, `eval/metrics.json`, `workspace/cache.npz` all retained.

- [ ] **Step 5: Commit**

```bash
git add Makefile
git commit -m "fix(make): tighten clean-* to only delete features.csv + logs

Review finding I-MK3 (2026-06-03 audit): clean-{2th,5th,normal} used
find ... -delete to wipe everything except root config.ini. This also
nuked eval/ (hand-curated metrics) and workspace/ (cached intermediate
data), but help text claimed only 'features.csv / log' would go.

Tighten recipes to rm -f \$(DIR)/features.csv \$(DIR)/*.log only.
Verified with synthetic /tmp/fake_baseline: eval/ + workspace/
retained as intended."
```

---

## Task 8: I-ST2 — Add `test_size` config for held-out spec_trainer evaluation

**Why:** Both `exp1.yaml` and `exp2.yaml` set `test_files == train_files`. `evaluate_and_report` therefore measures on data the model just trained on — reported AUC is in-sample, not generalization. User decision: add a `test_size` config field; when present and `test_files` is empty/missing, use `sklearn.model_selection.train_test_split(stratify=y)` to carve out a held-out set.

**Per rubber-duck N4:** extract a pure `resolve_holdout` helper so we can write a real behavioral test asserting train/test disjointness — not just source-grep for `train_test_split`.

**Files:**
- Create: `tools/spec_trainer/src/holdout.py` (pure helper, no model deps)
- Modify: `tools/spec_trainer/src/main.py` (import + use helper in `main()`)
- Modify: `tools/spec_trainer/config/exp1.yaml` (remove test_files, add test_size: 0.2)
- Modify: `tools/spec_trainer/config/exp2.yaml` (same)
- Test: `tests/test_spec_trainer_holdout.py` (new — behavioral test on the helper)

- [ ] **Step 1: Write failing behavioral test for `resolve_holdout`**

Create `tests/test_spec_trainer_holdout.py`:

```python
"""Behavioral tests for spec_trainer holdout split resolution.

Regression for review finding I-ST2 (2026-06-03 audit) + rubber-duck N4:
test the actual branching logic with synthetic data, not just source-grep.
"""
import os
import sys
import numpy as np
import pandas as pd
import pytest

_PROJECT_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
_SRC = os.path.join(_PROJECT_ROOT, "tools", "spec_trainer", "src")
if _SRC not in sys.path:
    sys.path.insert(0, _SRC)

from holdout import resolve_holdout  # noqa: E402


def _synthetic_frame(n=100, seed=0):
    rng = np.random.default_rng(seed)
    X = pd.DataFrame({
        "f1": rng.normal(size=n),
        "f2": rng.normal(size=n),
    })
    y = pd.Series(rng.integers(0, 2, size=n), name="label")
    return X, y


def test_resolve_holdout_uses_distinct_test_files(tmp_path):
    """When test_files distinct from train_files, load them as held-out."""
    X_train, y_train = _synthetic_frame(50, seed=1)
    test_csv = tmp_path / "test.csv"
    test_df = _synthetic_frame(30, seed=2)
    pd.concat([test_df[0], test_df[1]], axis=1).to_csv(test_csv, index=False)

    def loader(files, feature_cols, target_col):
        df = pd.concat([pd.read_csv(f) for f in files], ignore_index=True)
        return df[feature_cols], df[target_col]

    Xt, Xe, yt, ye = resolve_holdout(
        X_train, y_train,
        train_files=["train_a.csv"],
        test_files=[str(test_csv)],
        test_size=0.0,
        feature_cols=["f1", "f2"],
        target_col="label",
        loader=loader,
    )
    assert len(Xe) == 30  # held-out came from test_csv
    assert len(Xt) == 50  # train untouched


def test_resolve_holdout_splits_from_train_when_test_files_empty():
    """When test_files empty + test_size>0, stratified-split from train."""
    X_train, y_train = _synthetic_frame(100, seed=3)
    Xt, Xe, yt, ye = resolve_holdout(
        X_train, y_train,
        train_files=["train.csv"],
        test_files=[],
        test_size=0.25,
        feature_cols=["f1", "f2"],
        target_col="label",
        loader=lambda *a, **k: (_ for _ in ()).throw(
            AssertionError("loader should NOT be called when splitting")),
    )
    # Disjointness: indices in Xt and Xe should not overlap
    assert set(Xt.index).isdisjoint(set(Xe.index)), (
        "train and held-out must be disjoint (I-ST2 + rubber-duck N4)")
    assert len(Xe) == 25  # test_size=0.25 of 100
    assert len(Xt) == 75


def test_resolve_holdout_splits_when_test_files_equals_train():
    """If user accidentally sets test_files==train_files, prefer test_size split."""
    X_train, y_train = _synthetic_frame(100, seed=4)
    train_files = ["a.csv", "b.csv"]
    Xt, Xe, yt, ye = resolve_holdout(
        X_train, y_train,
        train_files=train_files,
        test_files=list(train_files),  # same set
        test_size=0.2,
        feature_cols=["f1", "f2"],
        target_col="label",
        loader=lambda *a, **k: (_ for _ in ()).throw(
            AssertionError("loader should NOT be called when test_files==train_files")),
    )
    assert set(Xt.index).isdisjoint(set(Xe.index))
    assert len(Xe) == 20


def test_resolve_holdout_raises_when_neither_option_provided():
    """No test_files (distinct) and no test_size -> ValueError, no silent in-sample."""
    X_train, y_train = _synthetic_frame(50, seed=5)
    with pytest.raises(ValueError, match="in-sample"):
        resolve_holdout(
            X_train, y_train,
            train_files=["a.csv"],
            test_files=[],
            test_size=0.0,
            feature_cols=["f1", "f2"],
            target_col="label",
            loader=lambda *a, **k: None,
        )


def test_resolve_holdout_stratify_preserves_label_balance():
    """Stratified split must preserve class proportions in both halves."""
    n = 1000
    rng = np.random.default_rng(42)
    X = pd.DataFrame({"f1": rng.normal(size=n)})
    # Heavy class imbalance: 80% class 0, 20% class 1
    y = pd.Series((rng.uniform(size=n) < 0.2).astype(int), name="label")
    train_ratio = y.mean()

    Xt, Xe, yt, ye = resolve_holdout(
        X, y,
        train_files=["x.csv"],
        test_files=[],
        test_size=0.3,
        feature_cols=["f1"],
        target_col="label",
        loader=lambda *a, **k: None,
    )
    # Both halves should have ~20% class 1
    assert abs(yt.mean() - train_ratio) < 0.02
    assert abs(ye.mean() - train_ratio) < 0.02
```

- [ ] **Step 2: Run tests, verify they fail**

Run: `conda run -n jianyan pytest tests/test_spec_trainer_holdout.py -v`
Expected: 5 FAILs (module `holdout` doesn't exist yet).

- [ ] **Step 3: Create `tools/spec_trainer/src/holdout.py`**

```python
"""Held-out set resolution for spec_trainer.

Extracted from main.py so the branching logic can be unit-tested without
importing lightgbm/sklearn-model-manager (see I-ST2 + rubber-duck N4,
2026-06-03 audit).
"""
from sklearn.model_selection import train_test_split


def resolve_holdout(
    X_train, y_train,
    train_files, test_files, test_size,
    feature_cols, target_col, loader,
):
    """Resolve (X_train, X_test, y_train, y_test) per the I-ST2 contract.

    Priority:
    1. test_files set AND distinct from train_files -> load it via `loader`.
       loader signature: loader(files, feature_cols, target_col) -> (X, y)
    2. else if test_size > 0 -> sklearn train_test_split with stratify=y,
       random_state=42.
    3. else -> raise ValueError (refuse silent in-sample evaluation).

    Returns: (X_train_out, X_test_out, y_train_out, y_test_out)
    """
    if test_files and set(test_files) != set(train_files):
        X_test, y_test = loader(test_files, feature_cols, target_col)
        return X_train, X_test, y_train, y_test
    if test_size and test_size > 0:
        X_tr, X_te, y_tr, y_te = train_test_split(
            X_train, y_train,
            test_size=test_size,
            random_state=42,
            stratify=y_train,
        )
        return X_tr, X_te, y_tr, y_te
    raise ValueError(
        "Neither distinct test_files nor data.test_size>0 provided — "
        "would evaluate in-sample (see I-ST2 audit 2026-06-03)")
```

- [ ] **Step 4: Run holdout tests, verify all 5 pass**

Run: `conda run -n jianyan pytest tests/test_spec_trainer_holdout.py -v`
Expected: 5 PASSed.

- [ ] **Step 5: Wire helper into `main.py`**

In `tools/spec_trainer/src/main.py`, find the block (lines 199-208 of current main.py):

```python
    X_train, y_train = load_data(
        cfg['data']['train_files'],
        feature_cols,
        target_col,
    )
    X_test, y_test = load_data(
        cfg['data']['test_files'],
        feature_cols,
        target_col,
    )
```

Replace with:

```python
    X_train, y_train = load_data(
        cfg['data']['train_files'],
        feature_cols,
        target_col,
    )

    # Held-out set resolution (review finding I-ST2, 2026-06-03 audit).
    # Helper extracted to holdout.py for testability (rubber-duck N4).
    from holdout import resolve_holdout
    X_train, X_test, y_train, y_test = resolve_holdout(
        X_train, y_train,
        train_files=cfg['data']['train_files'],
        test_files=cfg['data'].get('test_files') or [],
        test_size=cfg['data'].get('test_size', 0.0),
        feature_cols=feature_cols,
        target_col=target_col,
        loader=load_data,
    )
```

- [ ] **Step 6: Update `tools/spec_trainer/config/exp1.yaml`**

Find:

```yaml
data:
  train_files:
    - runs/baseline_2da_clean/features.csv
  test_files:
    - runs/baseline_2da_clean/features.csv
  feature_cols: []
  target_col: label  # 假设你的标签列叫 'label'
```

Replace with:

```yaml
data:
  train_files:
    - runs/baseline_2da_clean/features.csv
  # test_files 留空 -> 自动从 train_files 用 sklearn train_test_split
  # 切出 held-out（stratify by label, random_state=42）。
  # 见 I-ST2 (2026-06-03 audit)。
  test_files: []
  test_size: 0.2
  feature_cols: []
  target_col: label  # 假设你的标签列叫 'label'
```

- [ ] **Step 7: Update `tools/spec_trainer/config/exp2.yaml` similarly**

Find:

```yaml
data:
  train_files:
    - runs/baseline_2da_clean/features.csv
    - runs/baseline_5da_clean/features.csv
  test_files:
    - runs/baseline_2da_clean/features.csv
    - runs/baseline_5da_clean/features.csv
  feature_cols: []
  target_col: label  # 假设你的标签列叫 'label'
```

Replace with:

```yaml
data:
  train_files:
    - runs/baseline_2da_clean/features.csv
    - runs/baseline_5da_clean/features.csv
  # test_files 留空 -> 自动从合并后的 train_files 切 held-out
  # （stratify by label, random_state=42）。见 I-ST2 (2026-06-03 audit)。
  # 注意：split 在 concat 后做全局 stratified split，不保留 dataset 结构。
  # 评估的是"未见 PSM 的泛化"，不是"未见 dataset 的泛化"。
  # 若需后者，未来加 group_col: dataset 支持 GroupShuffleSplit
  # （rubber-duck N3, 2026-06-03）。
  test_files: []
  test_size: 0.2
  feature_cols: []
  target_col: label  # 假设你的标签列叫 'label'
```

- [ ] **Step 8: Add yaml-validation test**

Append to `tests/test_spec_trainer_holdout.py`:

```python


def test_exp_yamls_do_not_have_in_sample_test_files():
    """exp1/exp2 yaml must not set test_files == train_files (I-ST2)."""
    import yaml
    for name in ("exp1.yaml", "exp2.yaml"):
        p = os.path.join(_PROJECT_ROOT, "tools", "spec_trainer", "config", name)
        with open(p) as f:
            cfg = yaml.safe_load(f)
        train_files = cfg["data"].get("train_files", [])
        test_files = cfg["data"].get("test_files", [])
        if test_files and set(test_files) == set(train_files):
            raise AssertionError(
                f"{name}: test_files == train_files — in-sample AUC! (I-ST2)")
        if not test_files:
            assert "test_size" in cfg["data"], (
                f"{name}: when test_files is empty, must set data.test_size (I-ST2)")
            assert 0.0 < cfg["data"]["test_size"] < 1.0, (
                f"{name}: test_size must be in (0, 1)")
```

- [ ] **Step 9: Run all tests + full suite regression**

Run:
```
conda run -n jianyan pytest tests/test_spec_trainer_holdout.py -v
conda run -n jianyan pytest tests/ -q 2>&1 | tail -5
```
Expected: 6 PASSed in the new file; full suite no NEW failures.

- [ ] **Step 10: Commit**

```bash
git add tools/spec_trainer/src/holdout.py tools/spec_trainer/src/main.py tools/spec_trainer/config/exp1.yaml tools/spec_trainer/config/exp2.yaml tests/test_spec_trainer_holdout.py
git commit -m "feat(spec_trainer): test_size config for held-out evaluation (I-ST2)

Review finding I-ST2 (2026-06-03 audit): exp1.yaml + exp2.yaml had
test_files == train_files, so reported AUC was in-sample (training
accuracy), not generalization.

Per user decision (option A): add data.test_size config; when test_files
is empty/missing OR equals train_files, split from train_files with
train_test_split(stratify=y, random_state=42, test_size=...).

Resolution logic extracted to holdout.py + 6 behavioral unit tests
(rubber-duck N4 — assert train/test disjoint, stratify preserves class
balance, ValueError on silent in-sample).

Behavior:
- test_files set AND distinct from train_files -> use as held-out
- test_files empty/missing OR == train_files   -> stratified split
- neither                                       -> ValueError

Both exp1.yaml and exp2.yaml updated to test_files: [] + test_size: 0.2.

Known limitation (rubber-duck N3): exp2 concats 2da + 5da and does a
single global stratified split — tests 'unseen PSM' generalization,
not 'unseen dataset/window' generalization. Documented in exp2.yaml."
```

---

## Final Verification (after all 8 tasks)

- [ ] Run full test suite: `conda run -n jianyan pytest tests/ -q 2>&1 | tail -5`
- [ ] Run smoke `make -n 2th` / `make -n 5th` / `make -n normal` / `make -n all` — all four must succeed without "No rule to make target".
- [ ] Run smoke `make -n train-exp1` / `make -n train-exp2` — both must succeed.
- [ ] Verify commit count: 8 commits (1 per task) added to feature_extraction branch.
- [ ] (Optional) push to gitlab: `git push gitlab feature_extraction`
