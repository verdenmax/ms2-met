# Systematic Training Matrix (18 experiments)

**Status:** Approved (2026-06-03)
**Branch:** `feature_extraction`
**Plan file:** `docs/superpowers/plans/2026-06-03-systematic-training-matrix.md` (to be created next)

---

## Background

After dual-FDR + neg-FDR variants, the pipeline produces 9 `features.csv` files (3 datasets × 3 FDR levels). The user wants a systematic training matrix to compare model performance across:

- **3 FDR conditions**: clean (1%) / neg05 (negative-pool FDR 5%) / neg10 (10%)
- **2 training schemes** per FDR:
  - **Scheme 1 (in-sample 80/20)**: each dataset trained alone with held-out 20% test
  - **Scheme 2 (cross-dataset leave-one-out)**: 2 datasets trained, 1 held out as test

= **18 total experiments**.

This builds on:
- `tools/spec_trainer/` infrastructure (already in place)
- `resolve_holdout` helper supporting both explicit `test_files` and split-via-`test_size` (added in P2-4 of deep-audit fixes)
- 9 `runs/baseline_*_{clean,neg05,neg10}/features.csv` produced by the user's `make all-clean / all-neg05 / all-neg10` runs (out-of-plan, user-side).

## User-confirmed design decisions

1. **Keep `exp1.yaml` / `exp2.yaml`** as legacy experiments; not in `train-all` group.
2. **Naming convention**:
   - Scheme 1: `in_<dataset>_<fdr>.yaml`
   - Scheme 2: `cross_test_<held_out_dataset>_<fdr>.yaml`
3. **Makefile target granularity**: 4 group targets only (no per-yaml individual targets).
   - `train-clean-all`, `train-neg05-all`, `train-neg10-all`, `train-all`
4. **Output**: standard spec_trainer locations (`runs/spec_trainer/{models,results,figures}/`), filename = `<yaml-basename>`.

## Architecture

### Experiment matrix

| FDR | Scheme 1 in-sample (3) | Scheme 2 cross-dataset (3) |
|---|---|---|
| **clean** | in_{2da,5da,normal}_clean | cross_test_{2da,5da,normal}_clean |
| **neg05** | in_{2da,5da,normal}_neg05 | cross_test_{2da,5da,normal}_neg05 |
| **neg10** | in_{2da,5da,normal}_neg10 | cross_test_{2da,5da,normal}_neg10 |

**Cross naming rule**: `cross_test_<X>` = train on the OTHER two datasets, test on `<X>`.
- `cross_test_2da_clean` → train: `5da_clean`+`normal_clean`, test: `2da_clean`
- `cross_test_5da_clean` → train: `2da_clean`+`normal_clean`, test: `5da_clean`
- `cross_test_normal_clean` → train: `2da_clean`+`5da_clean`, test: `normal_clean`

### yaml template (Scheme 1, in-sample)

```yaml
data:
  train_files:
    - runs/baseline_<dataset>_<fdr>/features.csv
  test_files: []                # empty → triggers resolve_holdout split
  test_size: 0.2                # 80% train / 20% held-out
  feature_cols: []              # auto-detect via resolve_feature_cols
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
    is_unbalance: True          # SILAC ~1% positives (P1-4)
    verbose: -1

training:
  num_boost_round: 1000
  early_stopping_rounds: 50
  valid_size: 0.2               # internal valid split from train (for ES)

output:
  model_path: runs/spec_trainer/models/in_<dataset>_<fdr>.txt
  result_path: runs/spec_trainer/results/in_<dataset>_<fdr>.json
  figures_dir: runs/spec_trainer/figures
```

### yaml template (Scheme 2, cross-dataset)

```yaml
data:
  train_files:
    - runs/baseline_<other1>_<fdr>/features.csv
    - runs/baseline_<other2>_<fdr>/features.csv
  test_files:
    - runs/baseline_<held_out>_<fdr>/features.csv
  test_size: 0.0                # ignored; explicit test_files takes priority
  feature_cols: []
  target_col: label

model: (same as Scheme 1)
training: (same as Scheme 1)
output:
  model_path: runs/spec_trainer/models/cross_test_<held_out>_<fdr>.txt
  result_path: runs/spec_trainer/results/cross_test_<held_out>_<fdr>.json
  figures_dir: runs/spec_trainer/figures
```

**Note on `resolve_holdout` invariant** (from P2-4): with explicit `test_files` non-empty and distinct from `train_files`, `resolve_holdout` uses them directly. `test_size` is ignored in that branch. We set `test_size: 0` for clarity but it's a no-op.

### Files to create

18 new yaml files in `tools/spec_trainer/config/`:

```
in_2da_clean.yaml      in_2da_neg05.yaml      in_2da_neg10.yaml
in_5da_clean.yaml      in_5da_neg05.yaml      in_5da_neg10.yaml
in_normal_clean.yaml   in_normal_neg05.yaml   in_normal_neg10.yaml
cross_test_2da_clean.yaml    cross_test_2da_neg05.yaml    cross_test_2da_neg10.yaml
cross_test_5da_clean.yaml    cross_test_5da_neg05.yaml    cross_test_5da_neg10.yaml
cross_test_normal_clean.yaml cross_test_normal_neg05.yaml cross_test_normal_neg10.yaml
```

### Files to modify

- `Makefile` — add 4 group targets (`train-clean-all`, `train-neg05-all`, `train-neg10-all`, `train-all`) + 3 internal variables holding the 6-yaml list per FDR + .PHONY + help update.

### Files NOT modified

- `tools/spec_trainer/src/main.py` — no change. Already handles `resolve_holdout` for both schemes (P2-4).
- `tools/spec_trainer/src/feature_cols.py` — no change. Already supports multi-file intersection (P1-5) which both schemes need.
- `tools/spec_trainer/config/exp1.yaml` / `exp2.yaml` — kept as-is (per user decision A).
- Existing `train-exp1` / `train-exp2` / `train-all` / `clean-train` targets — kept; `train-all` will be REDEFINED (see migration note below).

### Migration: existing `train-all` target

Current `Makefile` defines `train-all: train-exp1 train-exp2`. This conflicts with the new `train-all` (which would mean "18 new experiments"). Decision:

- **Rename existing** `train-all` → `train-legacy-all` (depends on `train-exp1 train-exp2`)
- **New** `train-all: train-clean-all train-neg05-all train-neg10-all`

This is a small backward-incompatible change but the name `train-all` is more naturally claimed by the comprehensive 18-experiment matrix than by 2 legacy experiments. Both are documented in `help`.

## Implementation strategy

### Generator script (out of scope, manual sed loop)

The 18 yamls follow a strict template. To avoid copy-paste errors, generate them via a one-time bash loop (executed in T1, then yaml files committed):

```bash
for fdr in clean neg05 neg10; do
    for ds in 2da 5da normal; do
        # Scheme 1: in-sample
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

    # Scheme 2: 3 cross experiments per FDR
    for held_out in 2da 5da normal; do
        # Compute the other 2 datasets
        case $held_out in
          2da) others="5da normal" ;;
          5da) others="2da normal" ;;
          normal) others="2da 5da" ;;
        esac
        train_lines=$(for o in $others; do
            echo "    - runs/baseline_${o}_${fdr}/features.csv"
        done)
        cat > tools/spec_trainer/config/cross_test_${held_out}_${fdr}.yaml <<EOF
data:
  train_files:
${train_lines}
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
```

### Makefile group targets

```make
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

.PHONY: train-clean-all train-neg05-all train-neg10-all train-all train-legacy-all

train-clean-all:
	@mkdir -p runs/spec_trainer/models runs/spec_trainer/results runs/spec_trainer/figures
	@for yaml in $(CLEAN_YAMLS); do \
	    name=$$(basename $$yaml .yaml); \
	    echo "==================== train $$name ===================="; \
	    $(PY) tools/spec_trainer/src/main.py --config $$yaml --name $$name || exit 1; \
	done

train-neg05-all:
	@mkdir -p runs/spec_trainer/models runs/spec_trainer/results runs/spec_trainer/figures
	@for yaml in $(NEG05_YAMLS); do \
	    name=$$(basename $$yaml .yaml); \
	    echo "==================== train $$name ===================="; \
	    $(PY) tools/spec_trainer/src/main.py --config $$yaml --name $$name || exit 1; \
	done

train-neg10-all:
	@mkdir -p runs/spec_trainer/models runs/spec_trainer/results runs/spec_trainer/figures
	@for yaml in $(NEG10_YAMLS); do \
	    name=$$(basename $$yaml .yaml); \
	    echo "==================== train $$name ===================="; \
	    $(PY) tools/spec_trainer/src/main.py --config $$yaml --name $$name || exit 1; \
	done

train-all: train-clean-all train-neg05-all train-neg10-all

# Renamed from old 'train-all' (which was just exp1 + exp2)
train-legacy-all: train-exp1 train-exp2
```

### Output naming (per experiment)

Each `python main.py --config <yaml> --name <name>` produces:

- `runs/spec_trainer/models/<name>.txt`
- `runs/spec_trainer/results/<name>.json` (accuracy, AUC, classification_report, confusion_matrix)
- `runs/spec_trainer/figures/<name>_importance.png` (top 20 features by gain)
- `runs/spec_trainer/figures/<name>_roc_curve.png` (with Youden best-threshold marker)

Total: 18 experiments × 4 outputs = **72 result files**. All in 3 directories so easy to glob.

## Test strategy

3 test groups:

1. **`test_all_18_yamls_exist_and_parse`**: each of the 18 yaml files exists and parses with `yaml.safe_load`.

2. **`test_in_sample_yamls_have_correct_schema`**: for each of 9 `in_<ds>_<fdr>.yaml`:
   - `train_files` has exactly 1 entry pointing to `runs/baseline_<ds>_<fdr>/features.csv`
   - `test_files == []`
   - `test_size == 0.2`
   - `output.model_path` ends with `in_<ds>_<fdr>.txt`

3. **`test_cross_dataset_yamls_have_correct_schema`**: for each of 9 `cross_test_<held>_<fdr>.yaml`:
   - `train_files` has exactly 2 entries (the OTHER datasets at same FDR)
   - `test_files` has exactly 1 entry (the held-out dataset)
   - `held_out NOT in train_files` (data leakage guard)
   - `output.model_path` ends with `cross_test_<held>_<fdr>.txt`

4. **`test_makefile_train_group_targets_exist`**: `make -n train-clean-all`, `train-neg05-all`, `train-neg10-all`, `train-all`, `train-legacy-all` all return without "No rule to make target".

5. **`test_makefile_train_all_aggregates_three_fdr_groups`**: `make -n train-all` invokes all 3 FDR group targets.

Tests are pure-yaml-parse + `make -n` calls, no actual training invocation.

## Migration & verification

After plan execution, user-side steps:

1. (One-time, slow) Generate the 9 features.csv:
   ```bash
   make all-clean      # ~tens of minutes per dataset, 3 datasets
   make all-neg05
   make all-neg10
   ```

2. Train the matrix:
   ```bash
   make train-clean-all   # ~minutes (LightGBM is fast)
   make train-neg05-all
   make train-neg10-all
   # Or:
   make train-all         # all 18 in sequence
   ```

3. Inspect results:
   ```bash
   ls runs/spec_trainer/results/   # 18 JSON files (plus exp1/exp2 if previously trained)
   ls runs/spec_trainer/figures/   # 36 PNG files
   ```

4. (Future scope) Aggregate into a markdown table comparing AUC across the matrix — out of plan, user can write a one-off script when needed.

## Out of scope

- **Actually running the 18 experiments** — requires user's manual `make` invocation after they've generated all 9 features.csv.
- **Results aggregation script** — defer until user knows what comparison they want.
- **Hyperparameter tuning** — all 18 use identical LightGBM params for fair comparison.
- **Group-by-protein cross-validation** — current Scheme 2 splits by dataset (run-level), not by protein. If protein-level leakage matters, follow-up feature.
- **Renaming exp1/exp2** — user chose A (keep as-is).
- **Per-yaml individual `make train-in-2da-clean` targets** — user chose B (group targets only).
