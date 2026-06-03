# Systematic Training Matrix Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Build a 18-experiment training matrix (3 FDR levels × 2 schemes × 3 datasets) using existing spec_trainer infrastructure, with 4 Makefile group targets to run them in batches.

**Architecture:** Generate 18 yaml configs via deterministic bash loops (Scheme 1 in-sample + Scheme 2 cross-dataset), add 4 Makefile group targets (`train-clean-all`, `train-neg05-all`, `train-neg10-all`, `train-all`) that iterate over their yamls. Rename existing `train-all` → `train-legacy-all` to free the more natural name.

**Tech Stack:** PyYAML, configparser, bash heredoc, GNU Make, pytest.

---

## Background

See `docs/specs/2026-06-03-systematic-training-matrix-design.md` for full design. Key facts:

- spec_trainer (`tools/spec_trainer/src/main.py`) handles both schemes natively via `resolve_holdout` (P2-4): empty `test_files` → split via `test_size`; non-empty distinct `test_files` → load as held-out.
- `resolve_feature_cols` (P1-5) handles multi-file intersection for cross-dataset schema alignment.
- `is_unbalance: True` (P1-4) for SILAC ~1% positive class.
- User confirmed: keep `exp1.yaml` / `exp2.yaml` as legacy, naming `in_<ds>_<fdr>` / `cross_test_<held>_<fdr>`, 4 group Makefile targets only.

## Test environment

- `silac_ml` conda env (yaml, configparser, pytest, lightgbm). Baseline: **374 passed** (from neg-FDR variants HEAD `b60c102`).

## File Structure

| Status | Path | Responsibility |
|---|---|---|
| NEW (committed) | `tools/spec_trainer/config/in_<ds>_<fdr>.yaml` × 9 | Scheme 1 in-sample 80/20 |
| NEW (committed) | `tools/spec_trainer/config/cross_test_<held>_<fdr>.yaml` × 9 | Scheme 2 cross-dataset |
| MODIFY | `Makefile` | +3 yaml-list variables, +4 group targets, .PHONY update, rename existing `train-all` → `train-legacy-all`, help update |
| NEW | `tests/test_training_matrix.py` | Parse + schema validation tests for the 18 yamls + Makefile target presence |

Two tasks: T1 generates yamls (mechanical bash), T2 wires Makefile + tests.

---

## Task 1: Generate 18 yaml config files

**Why:** Each (scheme × dataset × FDR) experiment needs its own yaml. Generate via deterministic bash so all 18 follow identical structure (only varying `train_files` / `test_files` / `output.model_path` / `output.result_path`).

**Files:**
- Create: `tools/spec_trainer/config/in_{2da,5da,normal}_{clean,neg05,neg10}.yaml` × 9
- Create: `tools/spec_trainer/config/cross_test_{2da,5da,normal}_{clean,neg05,neg10}.yaml` × 9

- [ ] **Step 1: Generate the 9 in-sample yamls**

Run from the repo root:

```bash
cd /home/verden/pfind/2025-fall/code/ms2-met
for fdr in clean neg05 neg10; do
  for ds in 2da 5da normal; do
    cat > tools/spec_trainer/config/in_${ds}_${fdr}.yaml <<EOF
data:
  train_files:
    - runs/baseline_${ds}_${fdr}/features.csv
  test_files: []
  test_size: 0.2
  feature_cols: []
  target_col: label

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
    is_unbalance: True
    verbose: -1

training:
  num_boost_round: 1000
  early_stopping_rounds: 50
  valid_size: 0.2

output:
  model_path: runs/spec_trainer/models/in_${ds}_${fdr}.txt
  result_path: runs/spec_trainer/results/in_${ds}_${fdr}.json
  figures_dir: runs/spec_trainer/figures
EOF
  done
done
ls tools/spec_trainer/config/in_*.yaml
```

Expected: lists 9 files `in_{2da,5da,normal}_{clean,neg05,neg10}.yaml`.

- [ ] **Step 2: Generate the 9 cross-dataset yamls**

For each FDR level and each held-out dataset, the other 2 datasets at the SAME FDR become training data. Run:

```bash
for fdr in clean neg05 neg10; do
  for held_out in 2da 5da normal; do
    case $held_out in
      2da)    others="5da normal" ;;
      5da)    others="2da normal" ;;
      normal) others="2da 5da" ;;
    esac
    # Build train_files block (2 entries)
    train_block=""
    for o in $others; do
      train_block="${train_block}    - runs/baseline_${o}_${fdr}/features.csv"$'\n'
    done
    # Strip trailing newline for clean output
    train_block="${train_block%$'\n'}"

    cat > tools/spec_trainer/config/cross_test_${held_out}_${fdr}.yaml <<EOF
data:
  train_files:
${train_block}
  test_files:
    - runs/baseline_${held_out}_${fdr}/features.csv
  test_size: 0.0
  feature_cols: []
  target_col: label

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
    is_unbalance: True
    verbose: -1

training:
  num_boost_round: 1000
  early_stopping_rounds: 50
  valid_size: 0.2

output:
  model_path: runs/spec_trainer/models/cross_test_${held_out}_${fdr}.txt
  result_path: runs/spec_trainer/results/cross_test_${held_out}_${fdr}.json
  figures_dir: runs/spec_trainer/figures
EOF
  done
done
ls tools/spec_trainer/config/cross_test_*.yaml
```

Expected: lists 9 files `cross_test_{2da,5da,normal}_{clean,neg05,neg10}.yaml`.

- [ ] **Step 3: Spot-check 4 representative yamls (parse + content)**

```bash
conda run -n silac_ml python3 -c "
import yaml, os
samples = [
    'in_2da_clean.yaml',
    'in_normal_neg10.yaml',
    'cross_test_2da_clean.yaml',
    'cross_test_normal_neg10.yaml',
]
for name in samples:
    p = f'tools/spec_trainer/config/{name}'
    with open(p) as f:
        cfg = yaml.safe_load(f)
    print(f'--- {name} ---')
    print(f'  train_files: {cfg[\"data\"][\"train_files\"]}')
    print(f'  test_files:  {cfg[\"data\"][\"test_files\"]}')
    print(f'  test_size:   {cfg[\"data\"][\"test_size\"]}')
    print(f'  model_path:  {cfg[\"output\"][\"model_path\"]}')
    print()
"
```

Expected output for `in_2da_clean.yaml`:
- train_files = `['runs/baseline_2da_clean/features.csv']`
- test_files = `[]`
- test_size = `0.2`
- model_path = `runs/spec_trainer/models/in_2da_clean.txt`

For `cross_test_normal_neg10.yaml`:
- train_files = `['runs/baseline_2da_neg10/features.csv', 'runs/baseline_5da_neg10/features.csv']`
- test_files = `['runs/baseline_normal_neg10/features.csv']`
- test_size = `0.0`
- model_path = `runs/spec_trainer/models/cross_test_normal_neg10.txt`

If anything diverges, debug the bash loop.

- [ ] **Step 4: Verify total count**

```bash
ls tools/spec_trainer/config/{in_,cross_test_}*.yaml | wc -l
```

Expected: `18`.

- [ ] **Step 5: Commit**

```bash
git add tools/spec_trainer/config/in_*.yaml tools/spec_trainer/config/cross_test_*.yaml
git commit -m "config(spec_trainer): 18 training matrix yamls (Task 1)

Generated via deterministic bash loop for:
- Scheme 1 (in-sample 80/20): in_<dataset>_<fdr>.yaml × 9
  Each trains on one features.csv with test_size=0.2 split.
- Scheme 2 (cross-dataset): cross_test_<held>_<fdr>.yaml × 9
  Each trains on the OTHER two datasets, tests on the held-out.

3 datasets × 3 FDR levels × 2 schemes = 18 experiments.

All 18 use identical LightGBM params (boost_type=gbdt, num_leaves=31,
learning_rate=0.05, is_unbalance=True for SILAC 1% positive class).

Output: runs/spec_trainer/{models,results,figures}/<yaml-basename>.{txt,json,png}

Spec: docs/specs/2026-06-03-systematic-training-matrix-design.md"
```

---

## Task 2: Add 4 Makefile group targets + validation tests

**Why:** User needs `make train-clean-all` etc. to batch-run experiments. Existing `train-all` (= exp1 + exp2) is renamed to `train-legacy-all` to free the comprehensive name.

**Files:**
- Modify: `Makefile` (around lines 549-574 — existing `train-*` block)
- Create: `tests/test_training_matrix.py`

- [ ] **Step 1: Read current Makefile train-* block**

Run: `sed -n '549,575p' Makefile`

Confirm:
- Line ~549: `.PHONY: train-exp1 train-exp2 train-all clean-train`
- Line ~561-565: `train-exp1:` target body
- Line ~568-572: `train-exp2:` target body
- Line ~574: `train-all: train-exp1 train-exp2`

If structure has drifted, adapt accordingly.

- [ ] **Step 2: Replace the train-* block to add matrix groups + rename train-all**

In `Makefile`, find the existing block (the part starting `.PHONY: train-exp1 train-exp2 train-all clean-train`). REPLACE the existing `.PHONY` line and the existing `train-all` line, leaving `train-exp1`, `train-exp2`, `clean-train` intact.

Find:
```make
.PHONY: train-exp1 train-exp2 train-all clean-train
```

Replace with:
```make
.PHONY: train-exp1 train-exp2 clean-train
.PHONY: train-legacy-all train-clean-all train-neg05-all train-neg10-all train-all
```

Find:
```make
train-all: train-exp1 train-exp2
```

Replace with:
```make
# Legacy train-all (now exposed as train-legacy-all to free 'train-all'
# for the 18-experiment matrix; see docs/specs/2026-06-03-systematic-
# training-matrix-design.md).
train-legacy-all: train-exp1 train-exp2

# Systematic training matrix (18 experiments)
# 3 FDR conditions × 2 schemes × 3 datasets.
# Each yaml is invoked via 'python tools/spec_trainer/src/main.py
# --config <yaml> --name <basename>'.

SPEC_CFG := tools/spec_trainer/config

CLEAN_YAMLS := $(SPEC_CFG)/in_2da_clean.yaml \
               $(SPEC_CFG)/in_5da_clean.yaml \
               $(SPEC_CFG)/in_normal_clean.yaml \
               $(SPEC_CFG)/cross_test_2da_clean.yaml \
               $(SPEC_CFG)/cross_test_5da_clean.yaml \
               $(SPEC_CFG)/cross_test_normal_clean.yaml

NEG05_YAMLS := $(SPEC_CFG)/in_2da_neg05.yaml \
               $(SPEC_CFG)/in_5da_neg05.yaml \
               $(SPEC_CFG)/in_normal_neg05.yaml \
               $(SPEC_CFG)/cross_test_2da_neg05.yaml \
               $(SPEC_CFG)/cross_test_5da_neg05.yaml \
               $(SPEC_CFG)/cross_test_normal_neg05.yaml

NEG10_YAMLS := $(SPEC_CFG)/in_2da_neg10.yaml \
               $(SPEC_CFG)/in_5da_neg10.yaml \
               $(SPEC_CFG)/in_normal_neg10.yaml \
               $(SPEC_CFG)/cross_test_2da_neg10.yaml \
               $(SPEC_CFG)/cross_test_5da_neg10.yaml \
               $(SPEC_CFG)/cross_test_normal_neg10.yaml

train-clean-all:
	@mkdir -p runs/spec_trainer/models runs/spec_trainer/results runs/spec_trainer/figures
	@for yaml in $(CLEAN_YAMLS); do \
		name=$$(basename $$yaml .yaml); \
		echo "==================== train $$name ===================="; \
		$(PY) tools/spec_trainer/src/main.py --config $$yaml --name $$name || exit 1; \
	done
	@echo "[done] train-clean-all finished (6 experiments)"

train-neg05-all:
	@mkdir -p runs/spec_trainer/models runs/spec_trainer/results runs/spec_trainer/figures
	@for yaml in $(NEG05_YAMLS); do \
		name=$$(basename $$yaml .yaml); \
		echo "==================== train $$name ===================="; \
		$(PY) tools/spec_trainer/src/main.py --config $$yaml --name $$name || exit 1; \
	done
	@echo "[done] train-neg05-all finished (6 experiments)"

train-neg10-all:
	@mkdir -p runs/spec_trainer/models runs/spec_trainer/results runs/spec_trainer/figures
	@for yaml in $(NEG10_YAMLS); do \
		name=$$(basename $$yaml .yaml); \
		echo "==================== train $$name ===================="; \
		$(PY) tools/spec_trainer/src/main.py --config $$yaml --name $$name || exit 1; \
	done
	@echo "[done] train-neg10-all finished (6 experiments)"

train-all: train-clean-all train-neg05-all train-neg10-all
```

(Recipe lines MUST be TAB-indented. Verify after the edit.)

- [ ] **Step 3: Update `help:` block to document new + renamed targets**

Find in the help block (around lines 132-135):
```make
	@echo "  make train-exp1      训练 exp1（依赖 runs/baseline_2da_clean/features.csv）"
	@echo "  make train-exp2      训练 exp2（combined: 依赖 2da + 5da features.csv）"
	@echo "  make train-all       顺序跑 train-exp1 + train-exp2"
	@echo "  make clean-train     清理 runs/spec_trainer/ 训练产出"
```

Replace with:
```make
	@echo "  make train-exp1         旧实验：训练 exp1（2da 单独）"
	@echo "  make train-exp2         旧实验：训练 exp2（2da + 5da combined）"
	@echo "  make train-legacy-all   旧组合：train-exp1 + train-exp2"
	@echo ""
	@echo "  Systematic training matrix（18 实验：3 FDR × 2 schemes × 3 datasets）："
	@echo "  make train-clean-all    6 个 clean（1% FDR）实验"
	@echo "  make train-neg05-all    6 个 neg05 实验"
	@echo "  make train-neg10-all    6 个 neg10 实验"
	@echo "  make train-all          所有 18 个实验（顺序：clean → neg05 → neg10）"
	@echo ""
	@echo "  make clean-train        清理 runs/spec_trainer/ 训练产出"
```

- [ ] **Step 4: Dry-run all new targets to verify syntax**

Run:
```bash
make -n train-legacy-all 2>&1 | head -3
make -n train-clean-all 2>&1 | head -3
make -n train-neg05-all 2>&1 | head -3
make -n train-neg10-all 2>&1 | head -3
make -n train-all 2>&1 | head -5
```

Expected: no `Makefile:NN: *** ...` errors. Each target produces shell `for` loop + `python3 tools/spec_trainer/src/main.py ...` commands. `train-all` chains 3 sub-targets.

If make-level error, debug the Makefile stanza.

- [ ] **Step 5: Verify help text**

Run: `make help 2>&1 | grep -A12 "Systematic training matrix"`

Expected: shows the 4 new lines (clean-all, neg05-all, neg10-all, train-all).

- [ ] **Step 6: Create the test file**

Create `tests/test_training_matrix.py`:

```python
"""Tests for the systematic training matrix (18 yamls + 4 Makefile group targets).

See docs/specs/2026-06-03-systematic-training-matrix-design.md.
"""
import os
import subprocess

import pytest
import yaml

_PROJECT_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
_CFG_DIR = os.path.join(
    _PROJECT_ROOT, "tools", "spec_trainer", "config")


_DATASETS = ["2da", "5da", "normal"]
_FDRS = ["clean", "neg05", "neg10"]


def _all_in_sample_yamls():
    return [(ds, fdr) for ds in _DATASETS for fdr in _FDRS]


def _all_cross_yamls():
    return [(held, fdr) for held in _DATASETS for fdr in _FDRS]


@pytest.mark.parametrize("ds,fdr", _all_in_sample_yamls())
def test_in_sample_yaml_exists_and_parses(ds, fdr):
    """Each in_<ds>_<fdr>.yaml exists and parses cleanly."""
    p = os.path.join(_CFG_DIR, f"in_{ds}_{fdr}.yaml")
    assert os.path.exists(p), f"missing yaml: {p}"
    with open(p) as f:
        cfg = yaml.safe_load(f)
    assert "data" in cfg
    assert "model" in cfg
    assert "output" in cfg


@pytest.mark.parametrize("ds,fdr", _all_in_sample_yamls())
def test_in_sample_yaml_schema(ds, fdr):
    """in_<ds>_<fdr>.yaml has correct Scheme 1 schema."""
    p = os.path.join(_CFG_DIR, f"in_{ds}_{fdr}.yaml")
    with open(p) as f:
        cfg = yaml.safe_load(f)
    data = cfg["data"]

    # Scheme 1 contract: exactly one train file, no explicit test_files,
    # test_size=0.2 triggers resolve_holdout split.
    assert data["train_files"] == [f"runs/baseline_{ds}_{fdr}/features.csv"], (
        f"in_{ds}_{fdr}: wrong train_files = {data['train_files']!r}")
    assert data["test_files"] == [], (
        f"in_{ds}_{fdr}: test_files must be [] for in-sample; "
        f"got {data['test_files']!r}")
    assert data["test_size"] == 0.2, (
        f"in_{ds}_{fdr}: test_size must be 0.2 for 80/20 split; "
        f"got {data['test_size']!r}")
    assert data["target_col"] == "label"

    # Output paths must include the yaml basename for traceability.
    assert cfg["output"]["model_path"] == (
        f"runs/spec_trainer/models/in_{ds}_{fdr}.txt"), (
        f"in_{ds}_{fdr}: wrong model_path")
    assert cfg["output"]["result_path"] == (
        f"runs/spec_trainer/results/in_{ds}_{fdr}.json"), (
        f"in_{ds}_{fdr}: wrong result_path")


@pytest.mark.parametrize("held,fdr", _all_cross_yamls())
def test_cross_yaml_exists_and_parses(held, fdr):
    """Each cross_test_<held>_<fdr>.yaml exists and parses cleanly."""
    p = os.path.join(_CFG_DIR, f"cross_test_{held}_{fdr}.yaml")
    assert os.path.exists(p), f"missing yaml: {p}"
    with open(p) as f:
        cfg = yaml.safe_load(f)
    assert "data" in cfg
    assert "model" in cfg
    assert "output" in cfg


@pytest.mark.parametrize("held,fdr", _all_cross_yamls())
def test_cross_yaml_schema(held, fdr):
    """cross_test_<held>_<fdr>.yaml has correct Scheme 2 schema.

    Critical: held_out NOT in train_files (data leakage guard)."""
    p = os.path.join(_CFG_DIR, f"cross_test_{held}_{fdr}.yaml")
    with open(p) as f:
        cfg = yaml.safe_load(f)
    data = cfg["data"]

    # The other two datasets at the SAME FDR become train_files.
    expected_train = sorted(
        f"runs/baseline_{d}_{fdr}/features.csv"
        for d in _DATASETS if d != held)
    actual_train = sorted(data["train_files"])
    assert actual_train == expected_train, (
        f"cross_test_{held}_{fdr}: wrong train_files\n"
        f"  expected: {expected_train}\n"
        f"  got:      {actual_train}")

    assert data["test_files"] == [f"runs/baseline_{held}_{fdr}/features.csv"], (
        f"cross_test_{held}_{fdr}: test_files must be the held-out only; "
        f"got {data['test_files']!r}")

    # Data leakage guard
    held_path = f"runs/baseline_{held}_{fdr}/features.csv"
    assert held_path not in data["train_files"], (
        f"cross_test_{held}_{fdr}: DATA LEAKAGE — held-out {held_path!r} "
        f"in train_files {data['train_files']!r}")

    # Output paths
    assert cfg["output"]["model_path"] == (
        f"runs/spec_trainer/models/cross_test_{held}_{fdr}.txt"), (
        f"cross_test_{held}_{fdr}: wrong model_path")
    assert cfg["output"]["result_path"] == (
        f"runs/spec_trainer/results/cross_test_{held}_{fdr}.json"), (
        f"cross_test_{held}_{fdr}: wrong result_path")


def test_all_yamls_share_identical_model_hyperparams():
    """All 18 yamls use the same LightGBM params (so AUC differences
    reflect data/schema effects, not hyperparameter tuning)."""
    yamls = [
        os.path.join(_CFG_DIR, f"in_{ds}_{fdr}.yaml")
        for ds in _DATASETS for fdr in _FDRS
    ] + [
        os.path.join(_CFG_DIR, f"cross_test_{held}_{fdr}.yaml")
        for held in _DATASETS for fdr in _FDRS
    ]
    canonical = None
    canonical_name = None
    for p in yamls:
        with open(p) as f:
            cfg = yaml.safe_load(f)
        params = cfg["model"]["params"]
        if canonical is None:
            canonical = params
            canonical_name = os.path.basename(p)
            continue
        assert params == canonical, (
            f"{os.path.basename(p)}: model.params differ from "
            f"{canonical_name}\n"
            f"  expected: {canonical}\n"
            f"  got:      {params}")


_EXPECTED_TRAIN_TARGETS = [
    "train-clean-all",
    "train-neg05-all",
    "train-neg10-all",
    "train-all",
    "train-legacy-all",
]


@pytest.mark.parametrize("target", _EXPECTED_TRAIN_TARGETS)
def test_makefile_train_target_exists(target):
    """Each new training matrix Makefile target invokable via 'make -n'."""
    result = subprocess.run(
        ["make", "-n", target],
        cwd=_PROJECT_ROOT,
        capture_output=True, text=True)
    combined = result.stdout + result.stderr
    assert "No rule to make target" not in combined, (
        f"Makefile target {target!r} not found:\n{combined}")


def test_makefile_train_all_includes_three_fdr_groups():
    """make -n train-all should reference all 3 FDR groups
    (echo 'train-clean-all', 'train-neg05-all', 'train-neg10-all'
    each appear in the dry-run output)."""
    result = subprocess.run(
        ["make", "-n", "train-all"],
        cwd=_PROJECT_ROOT,
        capture_output=True, text=True)
    out = result.stdout + result.stderr
    # train-all is a phony target depending on the 3 groups; dry-run
    # expands each group's body. Check that yamls from each FDR appear.
    for marker in ("in_2da_clean.yaml", "in_2da_neg05.yaml",
                   "in_2da_neg10.yaml"):
        assert marker in out, (
            f"train-all dry-run should expand all 3 FDR groups; "
            f"missing reference to {marker!r}")


def test_makefile_phony_includes_train_matrix_targets():
    """All 5 train-matrix targets must be in .PHONY."""
    makefile_path = os.path.join(_PROJECT_ROOT, "Makefile")
    with open(makefile_path) as f:
        content = f.read()
    phony_lines = [line for line in content.splitlines()
                   if line.startswith(".PHONY:")]
    phony_targets = set()
    for line in phony_lines:
        phony_targets.update(line.replace(".PHONY:", "").split())
    for t in _EXPECTED_TRAIN_TARGETS:
        assert t in phony_targets, (
            f"Target {t!r} missing from .PHONY")
```

- [ ] **Step 7: Run tests, verify they PASS**

Run: `conda run -n silac_ml pytest tests/test_training_matrix.py -v`

Expected: all PASS (T1 + T2 already in place).
- 9 × `test_in_sample_yaml_exists_and_parses`
- 9 × `test_in_sample_yaml_schema`
- 9 × `test_cross_yaml_exists_and_parses`
- 9 × `test_cross_yaml_schema`
- 1 × `test_all_yamls_share_identical_model_hyperparams`
- 5 × `test_makefile_train_target_exists`
- 1 × `test_makefile_train_all_includes_three_fdr_groups`
- 1 × `test_makefile_phony_includes_train_matrix_targets`

Total: **44 new tests**.

- [ ] **Step 8: Full regression**

Run: `conda run -n silac_ml pytest tests/ -q 2>&1 | tail -3`

Expected: 374 baseline + 44 new = **418 passed**, no NEW failures.

If anything breaks in existing tests (especially in `tests/test_spec_trainer_*` or other `test_extract_common_*`), debug.

- [ ] **Step 9: Commit**

```bash
git add Makefile tests/test_training_matrix.py
git commit -m "build(make)+test: 4 training-matrix group targets + 44 validation tests (Task 2)

Add 4 Makefile group targets:
- train-clean-all    — 6 clean (1% FDR) experiments
- train-neg05-all    — 6 neg05 experiments
- train-neg10-all    — 6 neg10 experiments
- train-all          — sequential: clean → neg05 → neg10 (18 total)

Rename existing 'train-all' (= exp1+exp2) → 'train-legacy-all' to free
the more natural 'train-all' name for the comprehensive matrix.

3 new variables (CLEAN_YAMLS, NEG05_YAMLS, NEG10_YAMLS) list the
6 yamls per FDR. Each target uses a shell for-loop invoking
'python tools/spec_trainer/src/main.py --config <yaml> --name <name>'.
Sequential execution; --exit 1 on first failure.

44 parametrized tests:
- 9+9 schema tests for in-sample yamls (path correctness, test_size=0.2)
- 9+9 schema tests for cross yamls (held-out NOT in train_files —
  data leakage guard, expected train_files = other 2 datasets)
- 1 hyperparameter-consistency check (all 18 use identical LightGBM
  params so AUC differences reflect data, not tuning)
- 5 target-existence + 1 train-all-aggregation + 1 .PHONY check

Help text updated to document new + legacy naming.

Spec: docs/specs/2026-06-03-systematic-training-matrix-design.md"
```

---

## Final Verification (after both tasks)

- [ ] **Step 1: Full test suite**

Run: `conda run -n silac_ml pytest tests/ -q 2>&1 | tail -3`

Expected: **418 passed** (was 374 + 44 new), no NEW failures.

- [ ] **Step 2: Makefile dry-run smoke**

Run:
```bash
make -n train-clean-all 2>&1 | grep "python3 tools/spec_trainer" | wc -l
make -n train-neg05-all 2>&1 | grep "python3 tools/spec_trainer" | wc -l
make -n train-neg10-all 2>&1 | grep "python3 tools/spec_trainer" | wc -l
make -n train-all       2>&1 | grep "python3 tools/spec_trainer" | wc -l
```

Expected: each FDR group dry-run prints 6 python invocations (1 per yaml in the for-loop's `$$yaml` expansion). `train-all` prints 18.

(Actually: the for-loop is a single recipe; `make -n` shows the for-loop literal, not each python call individually. Adjust expectation if needed — the key test is `grep -c "for yaml in"` or similar marker.)

- [ ] **Step 3: User-side smoke (off-plan)**

The user runs (after they've generated all 9 features.csv):

```bash
make train-clean-all   # ~minutes for 6 LightGBM trainings
ls runs/spec_trainer/results/ | grep -E "in_|cross_test_" | sort
# Expected: 6 JSON files (in_2da_clean, in_5da_clean, in_normal_clean,
# cross_test_2da_clean, cross_test_5da_clean, cross_test_normal_clean)

make train-all         # ~tens of minutes for all 18
ls runs/spec_trainer/results/ | wc -l
# Expected: 18 (plus exp1.json / exp2.json if previously trained)
```

This step is OUT OF SCOPE for plan automation (requires 9 features.csv on disk).

- [ ] **Step 4: Push to gitlab (optional)**

```bash
git push gitlab feature_extraction
```

---

## Self-Review

**Spec coverage:**

| Spec section | Plan task |
|---|---|
| 18 yaml files (9 in-sample + 9 cross) | T1 |
| Scheme 1 schema (train_files, test_files=[], test_size=0.2) | T1 step 1 generates; T2 step 6 tests |
| Scheme 2 schema (train=2 others, test=held_out, test_size=0.0) | T1 step 2 generates; T2 step 6 tests |
| 4 Makefile group targets | T2 step 2 |
| Rename train-all → train-legacy-all | T2 step 2 |
| Help text update | T2 step 3 |
| Standard LightGBM params (is_unbalance=True etc.) | T1 step 1/2 generate; T2 step 6 hyperparameter-consistency test |
| Output naming = yaml basename | T1 generates; T2 step 6 tests model_path / result_path |
| Data leakage guard (held NOT in train) | T2 step 6 `test_cross_yaml_schema` explicitly asserts |
| 44 tests | T2 step 6 |

**Placeholder scan:** none. Every bash heredoc, sed substitution, and code block is concrete.

**Type / name consistency:**
- `in_<ds>_<fdr>.yaml` / `cross_test_<held>_<fdr>.yaml` consistent across T1 generation + T2 tests + Makefile yaml lists.
- `_DATASETS = ["2da", "5da", "normal"]` and `_FDRS = ["clean", "neg05", "neg10"]` match the variant names from prior plans.
- Makefile variables `CLEAN_YAMLS` / `NEG05_YAMLS` / `NEG10_YAMLS` consistent with target naming.
- `train-clean-all` / `train-neg05-all` / `train-neg10-all` / `train-all` / `train-legacy-all` consistent across help, .PHONY, target definitions, tests.

No gaps. No placeholders. No type/name drift.
