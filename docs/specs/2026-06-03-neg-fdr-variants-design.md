# Neg-FDR 5% / 10% Variant Configs and Makefile Targets

**Status:** Approved (2026-06-03)
**Branch:** `feature_extraction`
**Plan file:** `docs/superpowers/plans/2026-06-03-neg-fdr-variants.md` (to be created next)

---

## Background

The dual-FDR feature (spec `2026-06-03-dual-fdr-threshold-design.md`) added per-engine `negative_qvalue_threshold`. Production usage requires:
- 3 new live extract ini files per FDR level (2da, 5da, normal) × 2 levels (5%, 10%) = 6 ini files
- 3 new baseline-config dirs per FDR level × 2 levels = 6 `runs/baseline_*_neg{05,10}/config.ini`
- Matching Makefile targets so users can `make 2th-neg05` etc.

Current state: `extract_{2da,5da,normal}_pfind_diann.ini` and `runs/baseline_{2da,5da,normal}_clean/` exist for 1% FDR. The 5da and normal extract ini files were added by the user to `$HOME` and copied into the repo root (they are `.gitignore`'d but present on disk).

## User-confirmed design decisions

1. **Semantic**: dual-FDR mode — `qvalue_threshold = 0.01` (tight, positives) + `negative_qvalue_threshold = 0.05` or `0.10` (loose, negatives). Positives stay 1% strict; negatives expand.
2. **Naming**: `extract_2da_neg05.ini` / `extract_2da_neg10.ini` (short form, no `_pfind_diann` suffix). Matching `runs/baseline_2da_neg05/`, target `make 2th-neg05`.
3. **Entrapment**: keep `[entrapment]` filtering (drop L0/L1) in ALL variants — physical separability is orthogonal to FDR.
4. **Directory layout**: independent `runs/baseline_*_neg{05,10}/` directories (mirror of `_clean`).
5. **Makefile**: `dataset-fdr` target naming (`make 2th-neg05`), plus 3 group targets (`make all-clean`, `make all-neg05`, `make all-neg10`).

## Architecture

```
extract_{2da,5da,normal}_neg{05,10}.ini  (6 files, gitignored — machine-specific paths)
        ↓ make extract-{2th,5th,normal}-neg{05,10}
datasets/hela_{2da,5da,normal}_neg{05,10}.json
        ↓ make {2th,5th,normal}-neg{05,10}
runs/baseline_{2da,5da,normal}_neg{05,10}/features.csv
        ↓ (future: training; out of scope for this plan)
```

Each pipeline branch is fully independent (separate config.ini, separate workspace cache, separate eval dir).

## File structure

### Extract ini files (6 new, NOT tracked — gitignored)

Each based on `extract_<dataset>_pfind_diann.ini` template (which already exists on disk in `$HOME` form for all 3 datasets). Only changes:
- `result_file` → `./datasets/hela_<dataset>_neg<NN>.json`
- Both `[engine.X]` blocks add: `negative_qvalue_threshold = 0.<NN>`
- `[entrapment]` block unchanged

Example (`extract_2da_neg05.ini`):
```ini
[extract]
engines = pfind, diann
positive_species_marker = HUMAN
result_file = ./datasets/hela_2da_neg05.json

[engine.pfind]
path = ../pfind-dia/2th/
qvalue_threshold = 0.01
negative_qvalue_threshold = 0.05

[engine.diann]
path = ../diann/hela-01-22-2da-mix.parquet
qvalue_threshold = 0.01
negative_qvalue_threshold = 0.05

[entrapment]
target_fasta = /home/wskong/dia-nn/DIANN_workflow/data/fasta/uniprotkb_Human_AND_reviewed_true_AND_m_2025_12_30.fasta
drop_levels = L0, L1
```

### Baseline runs configs (6 new, TRACKED via gitignore whitelist)

Each based on `runs/baseline_<dataset>_clean/config.ini`. Only changes:
- `[input]` → `light_result_file = ./datasets/hela_<dataset>_neg<NN>.json`
- `[general]` → `work_directory = ./runs/baseline_<dataset>_neg<NN>/workspace`
- `[general]` → `result_file = ./runs/baseline_<dataset>_neg<NN>/features.csv`

All other fields (raw_paths, mass_tol_ppm, centroid_*, xic_cycle_window, feature_type, search_engine_type) identical to the `_clean` variant.

The existing `.gitignore` whitelist `!runs/baseline_*/config.ini` automatically covers the new dirs — no `.gitignore` change needed.

### Makefile changes

#### New variables (12 per-target + 12 paths)

```make
INI_2TH_NEG05    ?= extract_2da_neg05.ini
INI_2TH_NEG10    ?= extract_2da_neg10.ini
INI_5TH_NEG05    ?= extract_5da_neg05.ini
INI_5TH_NEG10    ?= extract_5da_neg10.ini
INI_NORMAL_NEG05 ?= extract_normal_neg05.ini
INI_NORMAL_NEG10 ?= extract_normal_neg10.ini

DIR_2TH_NEG05    ?= runs/baseline_2da_neg05
DIR_2TH_NEG10    ?= runs/baseline_2da_neg10
DIR_5TH_NEG05    ?= runs/baseline_5da_neg05
DIR_5TH_NEG10    ?= runs/baseline_5da_neg10
DIR_NORMAL_NEG05 ?= runs/baseline_normal_neg05
DIR_NORMAL_NEG10 ?= runs/baseline_normal_neg10

JSON_2TH_NEG05    := $(call EXTRACT_RESULT_FILE,$(INI_2TH_NEG05))
JSON_2TH_NEG10    := $(call EXTRACT_RESULT_FILE,$(INI_2TH_NEG10))
JSON_5TH_NEG05    := $(call EXTRACT_RESULT_FILE,$(INI_5TH_NEG05))
JSON_5TH_NEG10    := $(call EXTRACT_RESULT_FILE,$(INI_5TH_NEG10))
JSON_NORMAL_NEG05 := $(call EXTRACT_RESULT_FILE,$(INI_NORMAL_NEG05))
JSON_NORMAL_NEG10 := $(call EXTRACT_RESULT_FILE,$(INI_NORMAL_NEG10))
```

#### New targets (21)

```make
# Build (6): generate features.csv
2th-neg05  2th-neg10
5th-neg05  5th-neg10
normal-neg05  normal-neg10

# Extract (6): generate JSON only
extract-2th-neg05  extract-2th-neg10
extract-5th-neg05  extract-5th-neg10
extract-normal-neg05  extract-normal-neg10

# Clean (6): rm features.csv + *.log
clean-2th-neg05  clean-2th-neg10
clean-5th-neg05  clean-5th-neg10
clean-normal-neg05  clean-normal-neg10

# Group (3)
all-clean  (alias of existing 'all' for naming clarity; depends on 2th 5th normal)
all-neg05  (depends on 2th-neg05 5th-neg05 normal-neg05)
all-neg10  (depends on 2th-neg10 5th-neg10 normal-neg10)
```

Each follows the same `ifneq ($(wildcard ...),)` pattern as existing 2th/5th/normal targets, so missing ini gracefully falls back to "main.py only" mode.

Updated `help` text lists all variants.

## Implementation strategy

Use a Make function (`define` macro) to template the per-(dataset, fdr) rules. Concretely, instead of writing 21 hand-rolled targets, define:

```make
define DEFINE_PIPELINE
ifneq ($$(wildcard $$($(1)_INI)),)
$$($(1)_JSON): $$($(1)_INI)
	$$(call BANNER,extract $(2))
	$$(PY) tools/extract_common.py --configpath $$($(1)_INI)
extract-$(2): $$($(1)_JSON)
$(2): $$($(1)_INI) $$($(1)_JSON) $$($(1)_DIR)/config.ini
	$$(call BANNER,$(2))
	$$(PY) main.py --configpath $$($(1)_DIR)/config.ini --logpath $$($(1)_DIR)/extract.log
	@echo "[done] features written under $$($(1)_DIR)/"
else
extract-$(2):
	@echo "[error] $$($(1)_INI) not found" >&2
	@false
$(2): $$($(1)_DIR)/config.ini
	$$(call BANNER,$(2))
	@if [ ! -f "$$($(1)_DIR)/features.csv" ]; then \
		echo "[note] $$($(1)_INI) absent; main.py will fail if light_result_file invalid" >&2; \
	fi
	$$(PY) main.py --configpath $$($(1)_DIR)/config.ini --logpath $$($(1)_DIR)/extract.log
	@echo "[done] features written under $$($(1)_DIR)/"
endif
clean-$(2):
	@if [ -d $$($(1)_DIR) ]; then \
		rm -f $$($(1)_DIR)/features.csv $$($(1)_DIR)/features.csv.PARTIAL_INCOMPLETE $$($(1)_DIR)/*.log; \
		echo "[cleaned] $$($(1)_DIR)/features.csv + *.log (kept config.ini, eval/, workspace/)"; \
	else \
		echo "[skip] $$($(1)_DIR)/ does not exist"; \
	fi
endef
```

Wait — that's nice in theory but `$(eval $(call DEFINE_PIPELINE, ...))` with conditional `ifneq` inside `define` is brittle (per the earlier T5 review on the original 2th block; the prior dev kept them as 3 hand-rolled stanzas for grep-ability). Given there are now 9 (3 existing + 6 new) stanzas and the test surface is mechanical, I'll accept the duplication and add 6 new hand-rolled blocks (one per FDR variant).

**Trade-off**: 6 × ~25 lines = ~150 lines of duplication. Acceptable because:
- Each stanza independently grep-able (e.g., `grep "^5th-neg10:" Makefile`)
- Future addition of e.g. `neg15` would copy-paste-edit-1-stanza, no macro debugging
- Matches the existing 3-stanza pattern; reviewer in T5 of dual-FDR explicitly accepted similar duplication

## Test strategy

Tests for ini correctness + Makefile target existence:

1. **`test_neg_fdr_ini_files_have_correct_thresholds`**: for each of the 6 new ini files, parse with configparser, assert `[engine.pfind]` and `[engine.diann]` have `negative_qvalue_threshold` set to 0.05 or 0.10 matching filename, and `qvalue_threshold` is still 0.01. Skip ini files that don't exist on disk (since they're gitignored, may not be present in CI).

2. **`test_neg_fdr_baseline_configs_have_correct_paths`**: for each of the 6 new `runs/baseline_*_neg{05,10}/config.ini`, parse with configparser, assert:
   - `light_result_file` ends with `_neg<NN>.json`
   - `work_directory` matches the dir's neg<NN>
   - `result_file` matches the dir's neg<NN>
   - Other settings (mass_tol_ppm, centroid_enabled, etc.) identical to `_clean` variant

3. **`test_makefile_lists_all_neg_targets`**: parse Makefile via `make -np 2>/dev/null` or grep, assert all 21 new targets exist.

These tests run in `silac_ml` env. Use `pytest` import-test style (no actual `make` invocation, no actual ini file reads beyond config parsing).

## Migration notes

- User-side: the 6 new extract ini files are NOT in git (gitignored); user maintains them locally (they were placed in `$HOME` and need to be copied to repo root with the correct names).
- The repo provides:
  - Template guidance in `extract_common_config.ini.example`
  - 6 tracked baseline configs in `runs/baseline_*_neg{05,10}/`
  - Makefile targets ready to use
- To activate: user creates `extract_{2da,5da,normal}_neg{05,10}.ini` in repo root by copying from existing `extract_<dataset>_pfind_diann.ini` and adding `negative_qvalue_threshold = 0.05` (or `0.10`) under each `[engine.X]` block + changing the `result_file` to `hela_<dataset>_neg<NN>.json`.

A helper script `tools/generate_neg_fdr_inis.sh` (or Python) could automate this but is OUT OF SCOPE; user can do it once with simple text editing.

## Out of scope

- Actually invoking `make extract-2th-neg05` (requires multi-GB engine outputs + ~tens of minutes per run).
- spec_trainer yaml variants for training on the neg05 / neg10 features.csv (defer until user has run extraction and decides training strategy).
- `tools/generate_neg_fdr_inis.sh` helper script (manual creation is one-time).
- Documenting in `extract_common_config.ini.example` (already done in T5 of the dual-FDR plan).
- Modifying anything in `workflows/`, `spectrum/`, `tools/extract_common.py` (all already-built dual-FDR infrastructure).
