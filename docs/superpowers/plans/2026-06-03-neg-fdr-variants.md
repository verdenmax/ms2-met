# Neg-FDR Variants Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Add 6 new neg-FDR variant baseline configs (`runs/baseline_{2da,5da,normal}_{neg05,neg10}/config.ini`) + 21 new Makefile targets to support `make 2th-neg05` / `make all-neg10` etc., leveraging the dual-FDR feature.

**Architecture:** Mechanical replication of the existing `_clean` pattern across 2 new FDR levels (5%, 10%) × 3 datasets. Each pipeline branch has its own independent runs/ dir + config + Makefile target trio (build / extract / clean), mirroring the existing 3 stanzas. Group targets (`all-neg05`, `all-neg10`, `all-clean`) wire them together. Extract ini files are created by the user out-of-band (gitignored, machine-specific paths).

**Tech Stack:** GNU Make, INI configparser, pytest.

---

## Background

See `docs/specs/2026-06-03-neg-fdr-variants-design.md` for full design. Key facts:

- Dual-FDR feature (`load_engine_psms_dual` + `negative_qvalue_threshold`) already in place — this plan only consumes it.
- User has 3 live extract ini files in repo root (`extract_{2da,5da,normal}_pfind_diann.ini`, all gitignored).
- User confirmed: 2 new FDR levels (5%, 10%), naming `_neg05`/`_neg10`, independent runs dirs, hyphenated Makefile targets.
- Existing `.gitignore` whitelist `!runs/baseline_*/config.ini` automatically covers new `baseline_*_neg{05,10}/` dirs.

## Test environments

- `silac_ml` conda env (has pandas, configparser, pytest). No new deps.
- Baseline: **340 passed** (from dual-FDR branch HEAD `f5c449b`).

## File Structure

| Status | Path | Responsibility |
|---|---|---|
| NEW (gitignored) | `extract_{2da,5da,normal}_neg{05,10}.ini` × 6 | User creates from existing `_pfind_diann.ini` template + adds `negative_qvalue_threshold` |
| NEW (TRACKED) | `runs/baseline_{2da,5da,normal}_{neg05,neg10}/config.ini` × 6 | Per-baseline main.py config (variants of `runs/baseline_*_clean/config.ini`) |
| MODIFY | `Makefile` | +21 targets (6 build + 6 extract + 6 clean + 3 group) + variable declarations + help text |
| NEW | `tests/test_neg_fdr_variants.py` | 3 test groups for ini correctness + baseline config consistency + Makefile target presence |

Tasks decompose along the file types: T1 templates + 6 baseline configs, T2 Makefile, T3 tests. Each task ≤30 minutes of mechanical editing.

---

## Task 1: Create 6 baseline configs in `runs/baseline_*_neg{05,10}/`

**Why:** main.py reads `runs/baseline_<id>/config.ini` to know which JSON to load and where to write features.csv. Each (dataset, FDR-level) combination needs its own config.

**Files:**
- Create: `runs/baseline_2da_neg05/config.ini`
- Create: `runs/baseline_2da_neg10/config.ini`
- Create: `runs/baseline_5da_neg05/config.ini`
- Create: `runs/baseline_5da_neg10/config.ini`
- Create: `runs/baseline_normal_neg05/config.ini`
- Create: `runs/baseline_normal_neg10/config.ini`

Each config = source `runs/baseline_<dataset>_clean/config.ini` with EXACTLY 3 field substitutions:

| Field | Old value (clean) | New value (neg05) |
|---|---|---|
| `light_result_file` | `./datasets/hela_<dataset>_pfind_diann.json` | `./datasets/hela_<dataset>_neg05.json` |
| `work_directory` | `./runs/baseline_<dataset>_clean/workspace` | `./runs/baseline_<dataset>_neg05/workspace` |
| `result_file` | `./runs/baseline_<dataset>_clean/features.csv` | `./runs/baseline_<dataset>_neg05/features.csv` |

(For `neg10`: same with `neg10` in place of `neg05`.)

All other fields (raw_path_*, raw_num, mass_tol_ppm, xic_cycle_window, feature_type, centroid_enabled, centroid_rel_threshold, pfind_qvalue_threshold, search_engine_type) **identical** to source.

- [ ] **Step 1: Confirm `runs/baseline_2da_clean/config.ini` is the canonical template**

Run: `cat runs/baseline_2da_clean/config.ini`

Verify it has `[input]`, `[general]` sections and the 3 fields-to-substitute exist with exactly the expected values. If the actual paths differ from `light_result_file = ./datasets/hela_2da_pfind_diann.json`, adapt substitution.

- [ ] **Step 2: Create the 6 directories**

```bash
mkdir -p runs/baseline_2da_neg05 \
         runs/baseline_2da_neg10 \
         runs/baseline_5da_neg05 \
         runs/baseline_5da_neg10 \
         runs/baseline_normal_neg05 \
         runs/baseline_normal_neg10
```

- [ ] **Step 3: Generate the 6 config.ini files via sed substitution**

For each (dataset, fdr) pair, run:

```bash
# 2da × neg05
sed -e 's|hela_2da_pfind_diann\.json|hela_2da_neg05.json|g' \
    -e 's|runs/baseline_2da_clean/workspace|runs/baseline_2da_neg05/workspace|g' \
    -e 's|runs/baseline_2da_clean/features\.csv|runs/baseline_2da_neg05/features.csv|g' \
    runs/baseline_2da_clean/config.ini \
    > runs/baseline_2da_neg05/config.ini

# 2da × neg10
sed -e 's|hela_2da_pfind_diann\.json|hela_2da_neg10.json|g' \
    -e 's|runs/baseline_2da_clean/workspace|runs/baseline_2da_neg10/workspace|g' \
    -e 's|runs/baseline_2da_clean/features\.csv|runs/baseline_2da_neg10/features.csv|g' \
    runs/baseline_2da_clean/config.ini \
    > runs/baseline_2da_neg10/config.ini

# 5da × neg05
sed -e 's|hela_5da_pfind_diann\.json|hela_5da_neg05.json|g' \
    -e 's|runs/baseline_5da_clean/workspace|runs/baseline_5da_neg05/workspace|g' \
    -e 's|runs/baseline_5da_clean/features\.csv|runs/baseline_5da_neg05/features.csv|g' \
    runs/baseline_5da_clean/config.ini \
    > runs/baseline_5da_neg05/config.ini

# 5da × neg10
sed -e 's|hela_5da_pfind_diann\.json|hela_5da_neg10.json|g' \
    -e 's|runs/baseline_5da_clean/workspace|runs/baseline_5da_neg10/workspace|g' \
    -e 's|runs/baseline_5da_clean/features\.csv|runs/baseline_5da_neg10/features.csv|g' \
    runs/baseline_5da_clean/config.ini \
    > runs/baseline_5da_neg10/config.ini

# normal × neg05
sed -e 's|hela_normal_pfind_diann\.json|hela_normal_neg05.json|g' \
    -e 's|runs/baseline_normal_clean/workspace|runs/baseline_normal_neg05/workspace|g' \
    -e 's|runs/baseline_normal_clean/features\.csv|runs/baseline_normal_neg05/features.csv|g' \
    runs/baseline_normal_clean/config.ini \
    > runs/baseline_normal_neg05/config.ini

# normal × neg10
sed -e 's|hela_normal_pfind_diann\.json|hela_normal_neg10.json|g' \
    -e 's|runs/baseline_normal_clean/workspace|runs/baseline_normal_neg10/workspace|g' \
    -e 's|runs/baseline_normal_clean/features\.csv|runs/baseline_normal_neg10/features.csv|g' \
    runs/baseline_normal_clean/config.ini \
    > runs/baseline_normal_neg10/config.ini
```

- [ ] **Step 4: Spot-check a couple of generated configs**

```bash
diff runs/baseline_2da_clean/config.ini runs/baseline_2da_neg05/config.ini
```

Expected diff: exactly 3 lines changed (light_result_file, work_directory, result_file), all with `neg05` in place of `pfind_diann`/`clean`.

```bash
grep -E "light_result_file|work_directory|^result_file" runs/baseline_*_neg{05,10}/config.ini
```

Expected: every line ends with `_neg05.json|workspace|features.csv` or `_neg10.json|workspace|features.csv` (no leftover `_pfind_diann` or `_clean`).

If any leftover, sed failed — investigate and retry.

- [ ] **Step 5: Verify gitignore whitelist picks them up**

```bash
git check-ignore -v runs/baseline_2da_neg05/config.ini
```

Expected output: nothing (or `::` indicating not ignored — the whitelist `!runs/baseline_*/config.ini` should win). If it IS ignored, .gitignore needs adjustment — check `grep "baseline" .gitignore`.

```bash
git status --short runs/baseline_*_neg*
```

Expected: 6 new files listed as untracked, no other noise.

- [ ] **Step 6: Stage and commit**

```bash
git add runs/baseline_2da_neg05/config.ini \
        runs/baseline_2da_neg10/config.ini \
        runs/baseline_5da_neg05/config.ini \
        runs/baseline_5da_neg10/config.ini \
        runs/baseline_normal_neg05/config.ini \
        runs/baseline_normal_neg10/config.ini
git commit -m "config: 6 neg-FDR baseline configs (Task 1, neg-FDR variants)

For each dataset (2da, 5da, normal) and FDR level (neg05, neg10):
- runs/baseline_<dataset>_neg<NN>/config.ini

Generated by sed substitution from runs/baseline_<dataset>_clean/
config.ini, changing only 3 fields:
- light_result_file -> ./datasets/hela_<dataset>_neg<NN>.json
- work_directory    -> ./runs/baseline_<dataset>_neg<NN>/workspace
- result_file       -> ./runs/baseline_<dataset>_neg<NN>/features.csv

All other settings (raw_path_*, mass_tol_ppm, centroid_enabled, etc.)
identical to the _clean variant.

Spec: docs/specs/2026-06-03-neg-fdr-variants-design.md"
```

---

## Task 2: Add 21 Makefile targets

**Why:** Users invoke pipelines via `make 2th-neg05` etc. Add Makefile variables, conditional blocks (mirror existing `2th`/`5th`/`normal` pattern), clean targets, and group aggregators.

**Files:**
- Modify: `Makefile`

- [ ] **Step 1: Inspect current Makefile variable declaration block**

Run: `sed -n '20,55p' Makefile`

Locate where `INI_2TH`, `INI_5TH`, `INI_NORMAL`, `DIR_*`, `JSON_*` are declared (around lines 25-50). New variable declarations will be inserted IMMEDIATELY AFTER the existing `JSON_NORMAL := ...` line.

- [ ] **Step 2: Add 18 new variable declarations after the existing JSON_* block**

In `Makefile`, find the block:

```make
INI_2TH    ?= extract_2da_pfind_diann.ini
INI_5TH    ?= extract_5da_pfind_diann.ini
INI_NORMAL ?= extract_normal_pfind_diann.ini

# 三个 baseline 目录（含 config.ini + 输出位置）
DIR_2TH    ?= runs/baseline_2da_clean
DIR_5TH    ?= runs/baseline_5da_clean
DIR_NORMAL ?= runs/baseline_normal_clean
```

Replace with (extend to also declare neg05 / neg10 variants):

```make
INI_2TH    ?= extract_2da_pfind_diann.ini
INI_5TH    ?= extract_5da_pfind_diann.ini
INI_NORMAL ?= extract_normal_pfind_diann.ini

# Neg-FDR 变体 ini（dual-FDR：仅放宽负例池）
# 见 docs/specs/2026-06-03-neg-fdr-variants-design.md
INI_2TH_NEG05    ?= extract_2da_neg05.ini
INI_2TH_NEG10    ?= extract_2da_neg10.ini
INI_5TH_NEG05    ?= extract_5da_neg05.ini
INI_5TH_NEG10    ?= extract_5da_neg10.ini
INI_NORMAL_NEG05 ?= extract_normal_neg05.ini
INI_NORMAL_NEG10 ?= extract_normal_neg10.ini

# 三个 baseline 目录（含 config.ini + 输出位置）
DIR_2TH    ?= runs/baseline_2da_clean
DIR_5TH    ?= runs/baseline_5da_clean
DIR_NORMAL ?= runs/baseline_normal_clean

# Neg-FDR 变体 baseline 目录
DIR_2TH_NEG05    ?= runs/baseline_2da_neg05
DIR_2TH_NEG10    ?= runs/baseline_2da_neg10
DIR_5TH_NEG05    ?= runs/baseline_5da_neg05
DIR_5TH_NEG10    ?= runs/baseline_5da_neg10
DIR_NORMAL_NEG05 ?= runs/baseline_normal_neg05
DIR_NORMAL_NEG10 ?= runs/baseline_normal_neg10
```

Find the block:

```make
JSON_2TH    := $(call EXTRACT_RESULT_FILE,$(INI_2TH))
JSON_5TH    := $(call EXTRACT_RESULT_FILE,$(INI_5TH))
JSON_NORMAL := $(call EXTRACT_RESULT_FILE,$(INI_NORMAL))
```

Replace with:

```make
JSON_2TH    := $(call EXTRACT_RESULT_FILE,$(INI_2TH))
JSON_5TH    := $(call EXTRACT_RESULT_FILE,$(INI_5TH))
JSON_NORMAL := $(call EXTRACT_RESULT_FILE,$(INI_NORMAL))

# Neg-FDR 变体 JSON 路径（从对应 ini 抽取）
JSON_2TH_NEG05    := $(call EXTRACT_RESULT_FILE,$(INI_2TH_NEG05))
JSON_2TH_NEG10    := $(call EXTRACT_RESULT_FILE,$(INI_2TH_NEG10))
JSON_5TH_NEG05    := $(call EXTRACT_RESULT_FILE,$(INI_5TH_NEG05))
JSON_5TH_NEG10    := $(call EXTRACT_RESULT_FILE,$(INI_5TH_NEG10))
JSON_NORMAL_NEG05 := $(call EXTRACT_RESULT_FILE,$(INI_NORMAL_NEG05))
JSON_NORMAL_NEG10 := $(call EXTRACT_RESULT_FILE,$(INI_NORMAL_NEG10))
```

- [ ] **Step 3: Add new targets to `.PHONY` declarations**

Find:

```make
.PHONY: help all run clean
.PHONY: 2th 5th normal
.PHONY: extract-2th extract-5th extract-normal
.PHONY: clean-2th clean-5th clean-normal clean-all
```

Add three new lines after:

```make
.PHONY: help all run clean
.PHONY: 2th 5th normal
.PHONY: extract-2th extract-5th extract-normal
.PHONY: clean-2th clean-5th clean-normal clean-all
.PHONY: 2th-neg05 2th-neg10 5th-neg05 5th-neg10 normal-neg05 normal-neg10
.PHONY: extract-2th-neg05 extract-2th-neg10 extract-5th-neg05 extract-5th-neg10 extract-normal-neg05 extract-normal-neg10
.PHONY: clean-2th-neg05 clean-2th-neg10 clean-5th-neg05 clean-5th-neg10 clean-normal-neg05 clean-normal-neg10
.PHONY: all-clean all-neg05 all-neg10
```

- [ ] **Step 4: Update `help` text to list the new targets**

Find the existing `help:` block (around lines 68-95). At the end of the dataset/extract sections (before `make clean-2th` block), add:

```make
	@echo ""
	@echo "  Neg-FDR 变体（dual-FDR，仅放宽负例 FDR；正例保持 1%）："
	@echo "  make 2th-neg05       2Da × negative FDR 5%"
	@echo "  make 2th-neg10       2Da × negative FDR 10%"
	@echo "  make 5th-neg05       5Da × negative FDR 5%"
	@echo "  make 5th-neg10       5Da × negative FDR 10%"
	@echo "  make normal-neg05    Normal × negative FDR 5%"
	@echo "  make normal-neg10    Normal × negative FDR 10%"
	@echo "  make all-clean       别名：make all（FDR 1%）"
	@echo "  make all-neg05       三个数据集 × negative FDR 5%"
	@echo "  make all-neg10       三个数据集 × negative FDR 10%"
	@echo ""
	@echo "  make extract-2th-neg05  仅生成 2da neg05 JSON（其他类同）"
	@echo "  make clean-2th-neg05    删除 2da neg05 features.csv（其他类同）"
```

Insert this BEFORE the existing `@echo "  make clean-2th       删除 2da features.csv / log，强制下次重跑"` line so the help reads logically.

Also find the existing JSON path summary at the bottom of help (the `@echo "当前抽取的 JSON 路径："` block) and add the 6 new variant paths:

```make
	@echo "当前抽取的 JSON 路径："
	@echo "  2th         -> $(JSON_2TH)"
	@echo "  5th         -> $(JSON_5TH)"
	@echo "  normal      -> $(JSON_NORMAL)"
	@echo "  2th-neg05   -> $(JSON_2TH_NEG05)"
	@echo "  2th-neg10   -> $(JSON_2TH_NEG10)"
	@echo "  5th-neg05   -> $(JSON_5TH_NEG05)"
	@echo "  5th-neg10   -> $(JSON_5TH_NEG10)"
	@echo "  normal-neg05-> $(JSON_NORMAL_NEG05)"
	@echo "  normal-neg10-> $(JSON_NORMAL_NEG10)"
```

- [ ] **Step 5: Add 6 new build/extract stanzas (one per (dataset, fdr) pair)**

After the existing `normal:` block (which ends around line 211 with the `endif`), and BEFORE the line `all: 2th 5th normal` (around line 215), insert 6 new stanzas using EXACTLY the same `ifneq/else/endif` pattern as the existing `2th` block.

Each stanza follows this template (parameterized by `<NAME>`, `<DESC>`, `<INI_VAR>`, `<DIR_VAR>`, `<JSON_VAR>`):

```make
# ---------- <NAME> ----------
ifneq ($(wildcard $(<INI_VAR>)),)

$(<JSON_VAR>): $(<INI_VAR>)
	$(call BANNER,extract <NAME>)
	$(PY) tools/extract_common.py --configpath $(<INI_VAR>)

extract-<NAME>: $(<JSON_VAR>)

<NAME>: $(<INI_VAR>) $(<JSON_VAR>) $(<DIR_VAR>)/config.ini
	$(call BANNER,<NAME>)
	$(PY) main.py --configpath $(<DIR_VAR>)/config.ini --logpath $(<DIR_VAR>)/extract.log
	@echo "[done] features written under $(<DIR_VAR>)/"

else  # $(<INI_VAR>) absent — features.csv must be externally provided

extract-<NAME>:
	@echo "[error] $(<INI_VAR>) not found — cannot extract; provide ini or use a pre-built JSON" >&2
	@false

<NAME>: $(<DIR_VAR>)/config.ini
	$(call BANNER,<NAME>)
	@if [ ! -f "$(<DIR_VAR>)/features.csv" ]; then \
		echo "[note] $(<INI_VAR>) absent — $(<DIR_VAR>)/features.csv must be present" >&2; \
		echo "       (extract step skipped; main.py will fail if light_result_file in config.ini is invalid)" >&2; \
	fi
	$(PY) main.py --configpath $(<DIR_VAR>)/config.ini --logpath $(<DIR_VAR>)/extract.log
	@echo "[done] features written under $(<DIR_VAR>)/"

endif

```

Instantiate for all 6 pairs:

1. NAME=`2th-neg05`, INI=`INI_2TH_NEG05`, DIR=`DIR_2TH_NEG05`, JSON=`JSON_2TH_NEG05`
2. NAME=`2th-neg10`, INI=`INI_2TH_NEG10`, DIR=`DIR_2TH_NEG10`, JSON=`JSON_2TH_NEG10`
3. NAME=`5th-neg05`, INI=`INI_5TH_NEG05`, DIR=`DIR_5TH_NEG05`, JSON=`JSON_5TH_NEG05`
4. NAME=`5th-neg10`, INI=`INI_5TH_NEG10`, DIR=`DIR_5TH_NEG10`, JSON=`JSON_5TH_NEG10`
5. NAME=`normal-neg05`, INI=`INI_NORMAL_NEG05`, DIR=`DIR_NORMAL_NEG05`, JSON=`JSON_NORMAL_NEG05`
6. NAME=`normal-neg10`, INI=`INI_NORMAL_NEG10`, DIR=`DIR_NORMAL_NEG10`, JSON=`JSON_NORMAL_NEG10`

Concretely, the first instantiation looks like:

```make
# ---------- 2th-neg05 ----------
ifneq ($(wildcard $(INI_2TH_NEG05)),)

$(JSON_2TH_NEG05): $(INI_2TH_NEG05)
	$(call BANNER,extract 2th-neg05)
	$(PY) tools/extract_common.py --configpath $(INI_2TH_NEG05)

extract-2th-neg05: $(JSON_2TH_NEG05)

2th-neg05: $(INI_2TH_NEG05) $(JSON_2TH_NEG05) $(DIR_2TH_NEG05)/config.ini
	$(call BANNER,2th-neg05)
	$(PY) main.py --configpath $(DIR_2TH_NEG05)/config.ini --logpath $(DIR_2TH_NEG05)/extract.log
	@echo "[done] features written under $(DIR_2TH_NEG05)/"

else  # $(INI_2TH_NEG05) absent — features.csv must be externally provided

extract-2th-neg05:
	@echo "[error] $(INI_2TH_NEG05) not found — cannot extract; provide ini or use a pre-built JSON" >&2
	@false

2th-neg05: $(DIR_2TH_NEG05)/config.ini
	$(call BANNER,2th-neg05)
	@if [ ! -f "$(DIR_2TH_NEG05)/features.csv" ]; then \
		echo "[note] $(INI_2TH_NEG05) absent — $(DIR_2TH_NEG05)/features.csv must be present" >&2; \
		echo "       (extract step skipped; main.py will fail if light_result_file in config.ini is invalid)" >&2; \
	fi
	$(PY) main.py --configpath $(DIR_2TH_NEG05)/config.ini --logpath $(DIR_2TH_NEG05)/extract.log
	@echo "[done] features written under $(DIR_2TH_NEG05)/"

endif

```

Repeat for the other 5 by mechanical substitution.

- [ ] **Step 6: Add 6 new clean-* recipes and 3 new group targets**

After the existing `clean-all: clean-2th clean-5th clean-normal` line (around line 248), insert:

```make
# Neg-FDR variant clean targets (same conservative pattern as clean-2th/5th/normal)
clean-2th-neg05:
	@if [ -d $(DIR_2TH_NEG05) ]; then \
		rm -f $(DIR_2TH_NEG05)/features.csv $(DIR_2TH_NEG05)/features.csv.PARTIAL_INCOMPLETE $(DIR_2TH_NEG05)/*.log; \
		echo "[cleaned] $(DIR_2TH_NEG05)/features.csv + *.log (kept config.ini, eval/, workspace/)"; \
	else \
		echo "[skip] $(DIR_2TH_NEG05)/ does not exist"; \
	fi

clean-2th-neg10:
	@if [ -d $(DIR_2TH_NEG10) ]; then \
		rm -f $(DIR_2TH_NEG10)/features.csv $(DIR_2TH_NEG10)/features.csv.PARTIAL_INCOMPLETE $(DIR_2TH_NEG10)/*.log; \
		echo "[cleaned] $(DIR_2TH_NEG10)/features.csv + *.log (kept config.ini, eval/, workspace/)"; \
	else \
		echo "[skip] $(DIR_2TH_NEG10)/ does not exist"; \
	fi

clean-5th-neg05:
	@if [ -d $(DIR_5TH_NEG05) ]; then \
		rm -f $(DIR_5TH_NEG05)/features.csv $(DIR_5TH_NEG05)/features.csv.PARTIAL_INCOMPLETE $(DIR_5TH_NEG05)/*.log; \
		echo "[cleaned] $(DIR_5TH_NEG05)/features.csv + *.log (kept config.ini, eval/, workspace/)"; \
	else \
		echo "[skip] $(DIR_5TH_NEG05)/ does not exist"; \
	fi

clean-5th-neg10:
	@if [ -d $(DIR_5TH_NEG10) ]; then \
		rm -f $(DIR_5TH_NEG10)/features.csv $(DIR_5TH_NEG10)/features.csv.PARTIAL_INCOMPLETE $(DIR_5TH_NEG10)/*.log; \
		echo "[cleaned] $(DIR_5TH_NEG10)/features.csv + *.log (kept config.ini, eval/, workspace/)"; \
	else \
		echo "[skip] $(DIR_5TH_NEG10)/ does not exist"; \
	fi

clean-normal-neg05:
	@if [ -d $(DIR_NORMAL_NEG05) ]; then \
		rm -f $(DIR_NORMAL_NEG05)/features.csv $(DIR_NORMAL_NEG05)/features.csv.PARTIAL_INCOMPLETE $(DIR_NORMAL_NEG05)/*.log; \
		echo "[cleaned] $(DIR_NORMAL_NEG05)/features.csv + *.log (kept config.ini, eval/, workspace/)"; \
	else \
		echo "[skip] $(DIR_NORMAL_NEG05)/ does not exist"; \
	fi

clean-normal-neg10:
	@if [ -d $(DIR_NORMAL_NEG10) ]; then \
		rm -f $(DIR_NORMAL_NEG10)/features.csv $(DIR_NORMAL_NEG10)/features.csv.PARTIAL_INCOMPLETE $(DIR_NORMAL_NEG10)/*.log; \
		echo "[cleaned] $(DIR_NORMAL_NEG10)/features.csv + *.log (kept config.ini, eval/, workspace/)"; \
	else \
		echo "[skip] $(DIR_NORMAL_NEG10)/ does not exist"; \
	fi

# Group targets (existing 'all' kept; aliased as 'all-clean' for naming clarity)
all-clean: all
all-neg05: 2th-neg05 5th-neg05 normal-neg05
all-neg10: 2th-neg10 5th-neg10 normal-neg10
```

- [ ] **Step 7: Dry-run all new targets to verify syntax**

Run each new target with `-n` (dry-run, no execution):

```bash
make -n 2th-neg05 2>&1 | head -3
make -n 2th-neg10 2>&1 | head -3
make -n 5th-neg05 2>&1 | head -3
make -n 5th-neg10 2>&1 | head -3
make -n normal-neg05 2>&1 | head -3
make -n normal-neg10 2>&1 | head -3
make -n extract-2th-neg05 2>&1 | head -3
make -n extract-2th-neg10 2>&1 | head -3
make -n extract-5th-neg05 2>&1 | head -3
make -n extract-5th-neg10 2>&1 | head -3
make -n extract-normal-neg05 2>&1 | head -3
make -n extract-normal-neg10 2>&1 | head -3
make -n clean-2th-neg05 2>&1 | head -3
make -n clean-2th-neg10 2>&1 | head -3
make -n clean-5th-neg05 2>&1 | head -3
make -n clean-5th-neg10 2>&1 | head -3
make -n clean-normal-neg05 2>&1 | head -3
make -n clean-normal-neg10 2>&1 | head -3
make -n all-clean 2>&1 | head -5
make -n all-neg05 2>&1 | head -10
make -n all-neg10 2>&1 | head -10
```

Expected: no `Makefile:NN: *** ... .  Stop.` errors. For each target, the dry-run prints the python3 command (or `false` error for extract-* when ini is absent).

If ANY target fails with a make error (not python error), debug the new stanza syntax before committing.

- [ ] **Step 8: Verify help text updated**

Run: `make help 2>&1 | head -40`

Expected: shows the new neg-FDR section + JSON paths summary lists 9 entries (3 clean + 6 neg).

- [ ] **Step 9: Commit**

```bash
git add Makefile
git commit -m "build(make): 21 neg-FDR variant targets (Task 2)

Add 6 build + 6 extract + 6 clean + 3 group targets for the neg-FDR
variants (2da, 5da, normal × neg05, neg10).

Each (dataset, fdr) stanza follows the existing ifneq/else/endif
pattern from 2th/5th/normal (ini present → full pipeline; ini absent
→ main.py only with [note] warning). Clean recipes mirror clean-2th
(features.csv + *.log only; preserve config.ini, eval/, workspace/).

Group targets:
- all-clean = alias of existing 'all' (for naming clarity)
- all-neg05 = 2th-neg05 + 5th-neg05 + normal-neg05
- all-neg10 = 2th-neg10 + 5th-neg10 + normal-neg10

18 new variables (INI_*, DIR_*, JSON_*) for the 6 (dataset, fdr) pairs.

Help text and JSON-path summary updated to list all 9 variants.

Spec: docs/specs/2026-06-03-neg-fdr-variants-design.md"
```

---

## Task 3: Validation tests

**Why:** Lock the structure into pytest so future Makefile / config edits don't silently break the variant infrastructure. Tests are pure-Python: parse the configs + grep Makefile, no make/extract invocation.

**Files:**
- Create: `tests/test_neg_fdr_variants.py`

- [ ] **Step 1: Create test file**

Create `tests/test_neg_fdr_variants.py`:

```python
"""Tests for neg-FDR variant infrastructure.

Verifies the 6 baseline configs (runs/baseline_*_neg{05,10}/config.ini)
have correct field substitutions, and the Makefile lists all 21 expected
targets.

See docs/specs/2026-06-03-neg-fdr-variants-design.md.
"""
import configparser
import os
import subprocess
import sys

import pytest

_PROJECT_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))


# Parametrize over all 6 variant configs.
_VARIANTS = [
    ("2da", "neg05"),
    ("2da", "neg10"),
    ("5da", "neg05"),
    ("5da", "neg10"),
    ("normal", "neg05"),
    ("normal", "neg10"),
]


@pytest.mark.parametrize("dataset,fdr", _VARIANTS)
def test_neg_fdr_baseline_config_has_correct_paths(dataset, fdr):
    """Each runs/baseline_<dataset>_<fdr>/config.ini must have light_result_file,
    work_directory, and result_file pointing to its own variant paths."""
    cfg_path = os.path.join(
        _PROJECT_ROOT,
        "runs", f"baseline_{dataset}_{fdr}", "config.ini")
    assert os.path.exists(cfg_path), (
        f"missing variant config: {cfg_path}")

    cfg = configparser.ConfigParser()
    cfg.read(cfg_path)

    # [input] section
    light = cfg.get("input", "light_result_file")
    assert light.endswith(f"hela_{dataset}_{fdr}.json"), (
        f"{cfg_path}: light_result_file={light!r} should end with "
        f"hela_{dataset}_{fdr}.json")

    # [general] section
    work = cfg.get("general", "work_directory")
    assert f"baseline_{dataset}_{fdr}" in work, (
        f"{cfg_path}: work_directory={work!r} should include "
        f"baseline_{dataset}_{fdr}")
    assert work.endswith("workspace"), (
        f"{cfg_path}: work_directory should end with /workspace; got {work}")

    result = cfg.get("general", "result_file")
    assert f"baseline_{dataset}_{fdr}" in result, (
        f"{cfg_path}: result_file={result!r} should include "
        f"baseline_{dataset}_{fdr}")
    assert result.endswith("features.csv"), (
        f"{cfg_path}: result_file should end with /features.csv; got {result}")


@pytest.mark.parametrize("dataset,fdr", _VARIANTS)
def test_neg_fdr_baseline_config_inherits_settings_from_clean(dataset, fdr):
    """Each variant config must share these fields with the _clean
    variant: feature_type, mass_tol_ppm, xic_cycle_window,
    centroid_enabled, centroid_rel_threshold, search_engine_type,
    raw_num. These are dataset properties, not FDR properties."""
    clean_path = os.path.join(
        _PROJECT_ROOT, "runs", f"baseline_{dataset}_clean", "config.ini")
    variant_path = os.path.join(
        _PROJECT_ROOT, "runs", f"baseline_{dataset}_{fdr}", "config.ini")

    if not os.path.exists(clean_path):
        pytest.skip(f"clean baseline missing: {clean_path}")

    clean = configparser.ConfigParser()
    clean.read(clean_path)
    variant = configparser.ConfigParser()
    variant.read(variant_path)

    # Dataset-level fields that MUST match between _clean and _neg*.
    shared_fields = [
        ("input", "raw_num"),
        ("input", "search_engine_type"),
        ("general", "feature_type"),
        ("general", "mass_tol_ppm"),
        ("general", "xic_cycle_window"),
        ("general", "centroid_enabled"),
        ("general", "centroid_rel_threshold"),
    ]
    for section, option in shared_fields:
        if not clean.has_option(section, option):
            continue  # tolerate optional fields
        clean_val = clean.get(section, option)
        variant_val = variant.get(section, option)
        assert clean_val == variant_val, (
            f"{variant_path}: {section}.{option}={variant_val!r} differs "
            f"from clean {clean_val!r} (should be identical — dataset, "
            f"not FDR, property)")


# Parametrize Makefile target presence checks.
_EXPECTED_TARGETS = [
    # Build (6)
    "2th-neg05", "2th-neg10",
    "5th-neg05", "5th-neg10",
    "normal-neg05", "normal-neg10",
    # Extract (6)
    "extract-2th-neg05", "extract-2th-neg10",
    "extract-5th-neg05", "extract-5th-neg10",
    "extract-normal-neg05", "extract-normal-neg10",
    # Clean (6)
    "clean-2th-neg05", "clean-2th-neg10",
    "clean-5th-neg05", "clean-5th-neg10",
    "clean-normal-neg05", "clean-normal-neg10",
    # Group (3)
    "all-clean", "all-neg05", "all-neg10",
]


@pytest.mark.parametrize("target", _EXPECTED_TARGETS)
def test_makefile_target_exists(target):
    """Each new neg-FDR target must be invokable via `make -n <target>`
    without a make error (python command itself may fail later, that's OK)."""
    result = subprocess.run(
        ["make", "-n", target],
        cwd=_PROJECT_ROOT,
        capture_output=True, text=True)
    # Make returns non-zero if target doesn't exist OR if a sub-shell
    # command failed during dry-run resolution. We only care that the
    # output does NOT contain "No rule to make target" — that's the
    # signature of a missing target.
    combined = result.stdout + result.stderr
    assert "No rule to make target" not in combined, (
        f"Makefile target {target!r} not found:\n{combined}")


def test_makefile_phony_includes_new_targets():
    """All new neg-FDR targets must be listed in .PHONY (avoids
    accidental file/directory shadowing)."""
    makefile_path = os.path.join(_PROJECT_ROOT, "Makefile")
    with open(makefile_path) as f:
        content = f.read()
    for target in _EXPECTED_TARGETS:
        # .PHONY: X Y Z ... — look for the target name in any .PHONY line
        phony_lines = [line for line in content.splitlines()
                       if line.startswith(".PHONY:")]
        phony_targets = set()
        for line in phony_lines:
            phony_targets.update(line.replace(".PHONY:", "").split())
        assert target in phony_targets, (
            f"Target {target!r} missing from .PHONY declarations. "
            f"Add it to one of:\n" + "\n".join(phony_lines))
```

- [ ] **Step 2: Run tests, verify they PASS (configs and Makefile are already in place after T1+T2)**

Run: `conda run -n silac_ml pytest tests/test_neg_fdr_variants.py -v`

Expected: all parametrized tests PASS. If any FAIL:
- "missing variant config" → T1 step 3 missed a file; re-run sed for that pair
- "should end with hela_..._neg05.json" → sed substitution wrong; fix that config
- "No rule to make target" → T2 Makefile stanza missing/wrong; re-check
- "missing from .PHONY" → T2 step 3 missed adding to .PHONY; re-add

- [ ] **Step 3: Full regression**

Run: `conda run -n silac_ml pytest tests/ -q 2>&1 | tail -3`

Expected: 340 baseline + new test count (parametrized: 6 + 6 + 21 + 1 = 34 new tests) = **374 passed**, no NEW failures.

- [ ] **Step 4: Commit**

```bash
git add tests/test_neg_fdr_variants.py
git commit -m "test(neg-fdr): variant config + Makefile target validation (Task 3)

3 test groups parametrized over (dataset, fdr) ∈ 6 variants and 21
Makefile targets:

1. test_neg_fdr_baseline_config_has_correct_paths — light_result_file,
   work_directory, result_file all point to the variant's own paths.

2. test_neg_fdr_baseline_config_inherits_settings_from_clean — shared
   fields (raw_num, feature_type, mass_tol_ppm, xic_cycle_window,
   centroid_*, search_engine_type) IDENTICAL to _clean variant.

3. test_makefile_target_exists — each of the 21 new targets is
   invokable via 'make -n <target>' without 'No rule to make target'.

4. test_makefile_phony_includes_new_targets — all 21 in .PHONY.

Locks the variant infrastructure into pytest so future edits can't
silently break the contract.

Spec: docs/specs/2026-06-03-neg-fdr-variants-design.md"
```

---

## Final Verification (after all 3 tasks)

- [ ] **Step 1: Full test suite**

Run: `conda run -n silac_ml pytest tests/ -q 2>&1 | tail -5`

Expected: 374 passed (340 baseline + 34 new), no NEW failures.

- [ ] **Step 2: Help text smoke**

Run: `make help`

Expected: shows the new neg-FDR section + JSON paths summary with all 9 entries.

- [ ] **Step 3: User-side activation reminder**

The new variant ini files are NOT auto-created (gitignored, machine-specific paths). To activate a neg-FDR variant:

1. Copy `extract_<dataset>_pfind_diann.ini` to `extract_<dataset>_neg05.ini`
2. Change `result_file = ./datasets/hela_<dataset>_pfind_diann.json` → `..._neg05.json`
3. Under each `[engine.X]` block, add: `negative_qvalue_threshold = 0.05`
4. Repeat for `neg10` with `0.10`
5. Then `make extract-2th-neg05`, `make 2th-neg05`, etc.

Same for 5da, normal datasets.

Optional helper (NOT in plan scope): a one-liner like:

```bash
for d in 2da 5da normal; do
  for n in 05 10; do
    sed -e "s|hela_${d}_pfind_diann\.json|hela_${d}_neg${n}.json|g" \
        -e "/^qvalue_threshold/a negative_qvalue_threshold = 0.${n}" \
        extract_${d}_pfind_diann.ini > extract_${d}_neg${n}.ini
  done
done
```

- [ ] **Step 4: Push to gitlab (optional)**

```bash
git push gitlab feature_extraction
```

---

## Self-Review

**Spec coverage:**

| Spec section | Implemented in |
|---|---|
| Extract ini files (gitignored, user-created) | Out of plan scope (user-side, Final Verification Step 3) |
| Baseline runs configs (6 new tracked) | Task 1 |
| Makefile variables + 21 targets + help | Task 2 |
| Tests | Task 3 |
| Independent directory layout | Task 1 (mkdir) + Task 2 (DIR_*_NEG variables) |
| dataset-fdr target naming | Task 2 step 5 (each stanza uses `2th-neg05` etc.) |
| Group targets (all-clean, all-neg05, all-neg10) | Task 2 step 6 last 3 lines |
| Independent workspace/eval/log per variant | Task 1 (per-baseline work_directory) |
| Entrapment preserved | Not in plan — user keeps `[entrapment]` block when creating ini |

**Placeholder scan:** none — every code/sed block is concrete.

**Type/name consistency:**
- `INI_<DATASET>_<NEG##>` / `DIR_<DATASET>_<NEG##>` / `JSON_<DATASET>_<NEG##>` consistent across Task 2 steps 2/5/6.
- Target names `<dataset>-neg<NN>` consistent across stanzas, .PHONY, help text, and test list.
- File names `extract_<dataset>_neg<NN>.ini` and `runs/baseline_<dataset>_neg<NN>/config.ini` consistent across tasks 1, 2, 3.

No gaps.
