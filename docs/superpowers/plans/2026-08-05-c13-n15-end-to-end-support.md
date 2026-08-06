# SILAC / Uniform C13 / Uniform N15 End-to-End Support Plan

> Implementation must proceed task by task with the checkboxes below. This
> document is a repair plan only; creating it does not authorize code changes.

**Goal:** Make the unmodified-peptide pipeline compute, extract, audit, and
train on the correct light/heavy evidence for SILAC, uniform 13C, and uniform
15N labeling, while preserving SILAC defaults and rejecting unsupported
states instead of silently producing incorrect features.

**Architecture:** Introduce one deep labeling module with a small interface
(`HeavyType`, parsing, canonical name, label-site test, and sequence mass
shift). All configuration adapters resolve to that interface. The selected
`HeavyType` then flows through the existing workflow seam into precursor,
fragment, domain-filter, and training-set logic. Existing SILAC-named fields
remain temporary compatibility aliases, not independent sources of truth.

**Tech stack:** Python 3, configparser, NumPy, pandas, pyteomics, pytest.

---

## 1. Scope and locked decisions

### Supported in this repair

- SILAC K/R labeling, including the current modification behavior.
- Uniform 13C and uniform 15N labeling for **unmodified peptides**.
- `feature_type=0` single-raw extraction, the active production path.
- `feature_type=1/2` multi-raw paths at least to behavioral-test parity, so a
  dormant hard-coded SILAC branch is not left behind.
- Synthetic query generation and training-set assembly for all three labels.
- Label-aware entrapment/domain filtering.
- Backward-compatible default: absent configuration means SILAC.

### Explicitly not supported in this repair

- Uniform 13C/15N mass shifts for atoms contributed by PTMs. Those rows must
  fail early with a clear error until modification elemental compositions are
  implemented.
- A scientifically calibrated uniform-label isotope-envelope model without
  labeling purity/enrichment parameters. The existing natural-abundance
  `isotope_correlation` must be marked invalid for C13/N15, not reused.
- Training one universal model across all labeling chemistries. Each labeling
  mode may be trained/evaluated independently first.
- Claiming empirical C13/N15 performance without real labeled raw data.

### Required configuration

The feature-extraction pipeline uses:

```ini
[general]
labeling = silac  # aliases: c13/13c/cheavy, n15/15n/nheavy
```

`tools/extract_common.py` keeps its existing `[extract] labeling` adapter.
The training-set builder keeps `[queries] labeling` and `[assembly] labeling`,
but assembly must verify both phases describe the same canonical labeling.

---

## 2. File map

### New files

- `spectrum/labeling.py` — single labeling interface and implementation.
- `tests/test_labeling_end_to_end.py` — parameterized workflow behavior.

### Modified files

| File | Responsibility |
|---|---|
| `spectrum/psm_info.py` | consume/re-export the labeling interface; keep `PSMInfo` mass methods |
| `constant/keys.py` | define the `[general] labeling` key |
| `workflows/single_work.py` | use the selected label for precursor/fragments/shift features |
| `workflows/flow_utils.py` | preserve labeling metadata and batch behavior |
| `workflows/pair_flow.py` | validate labeling once before task dispatch |
| `tools/extract_common.py` | use the shared parser instead of a second alias table |
| `tools/trap_domain_filter.py` | make label-site filtering label-aware |
| `tools/training_set_builder.py` | enforce cross-phase labeling and generic shift contracts |
| `tools/spec_trainer/src/feature_cols.py` | exclude labeling/provenance metadata |
| `tools/eval_baseline.py` | exclude labeling/provenance metadata |
| `tools/eval_feature_ablation.py` | exclude labeling/provenance metadata |
| `config.ini` | document the new option and SILAC default |
| `training_set_builder.ini.example` | document all aliases and phase consistency |
| `README.md` and hard-negative spec | state supported/unsupported modes accurately |

Existing numerical, label-site, builder, trap-filter, workflow, and feature
contract tests will be extended instead of replaced where they already test
observable behavior through the same interface.

---

## 3. Task L1 — Create the canonical labeling module

**Why:** Alias parsing and labeling rules currently live in multiple files.
The module should concentrate the domain knowledge so a fourth labeling type
would not require repeated switches across callers.

**Interface:**

```python
class HeavyType(Enum):
    SILAC = 1
    CHEAVY = 2
    NHEAVY = 3

def parse_heavy_type(value: str | HeavyType) -> HeavyType: ...
def canonical_labeling_name(heavy_type: HeavyType) -> str: ...
def get_heavy_increase_mass(sequence: str, heavy_type: HeavyType) -> float: ...
def has_label_site(sequence: str, heavy_type: HeavyType) -> bool: ...
def supports_modified_peptide(heavy_type: HeavyType) -> bool: ...
```

The canonical names are exactly `silac`, `c13`, and `n15`.

- [ ] Add parameterized RED tests for every accepted alias, mixed case,
  whitespace, enum passthrough, and invalid values.
- [ ] Move `HeavyType`, the C13/N15 constants, mass-shift calculation, and
  label-site logic into `spectrum/labeling.py`.
- [ ] Re-export the existing public names from `spectrum/psm_info.py` so old
  imports remain valid.
- [ ] Replace `_LABELING_ALIASES` in `extract_common.py` and
  `_parse_heavy_type` in `training_set_builder.py` with the shared parser.
- [ ] Verify no caller maintains a second alias table.

**Tests:**

```bash
pytest -q \
  tests/test_extract_label_site.py \
  tests/test_psm_info_label_site.py \
  tests/test_single_work_numerics.py
```

**Proposed commit:** `refactor: centralize metabolic labeling rules`

---

## 4. Task L2 — Propagate labeling through feature extraction

**Why:** `single_pair_work` and `multi_batch_work` currently call
`get_heavy_info(HeavyType.SILAC)` and calculate `get_SILAC_increase_mass`
regardless of configuration.

- [ ] Add `ConfigKeys.LABELING = "labeling"`.
- [ ] Add one workflow resolver that reads `[general] labeling`, defaults to
  SILAC, and delegates parsing to `spectrum.labeling`.
- [ ] Validate the value once in `PairFlow` before starting worker processes,
  so an invalid value fails before expensive raw loading.
- [ ] In both `single_pair_work` and `multi_batch_work`, use the resolved
  `HeavyType` for:
  - heavy precursor m/z;
  - every b/y heavy fragment mass;
  - heavy-window lookup and Q1a separation;
  - the total label mass-shift feature.
- [ ] Add canonical output metadata `labeling`.
- [ ] Add canonical feature `total_label_shift`.
- [ ] Temporarily emit `total_silac_shift = total_label_shift` as a deprecated
  compatibility alias for existing CSVs/models. It must never be calculated
  separately.
- [ ] Add `labeling` to all model/evaluation metadata exclusion lists.
- [ ] Keep `kr_count` for schema compatibility, but document that it is not a
  label-site count for uniform C13/N15.

### Behavioral tests

Use a recording fake DIA adapter and real `PSMInfo`, parameterized over all
three `HeavyType` values. Assert through the public workflow functions that:

- the requested heavy precursor m/z equals `PSMInfo.get_heavy_info(type)`;
- representative b and y heavy fragment XIC requests use the correct masses;
- `total_label_shift` equals the selected labeling calculation;
- the legacy shift alias equals the canonical value;
- SILAC, C13, and N15 produce distinct expected requests for the same peptide;
- absent `[general] labeling` produces the exact existing SILAC behavior;
- invalid labeling fails before batch dispatch;
- single-raw and multi-raw paths behave consistently.

**Tests:**

```bash
pytest -q \
  tests/test_labeling_end_to_end.py \
  tests/test_single_work_numerics.py \
  tests/test_deep_audit_p0.py \
  tests/test_deep_audit_p1.py
```

**Proposed commit:** `fix: propagate labeling through feature extraction`

---

## 5. Task L3 — Make unsupported physics explicit

### C13/N15 modifications

- [ ] Preserve the existing `PSMInfo._assert_heavy_supported` rejection for
  modified C13/N15 peptides.
- [ ] In query generation, reject `exclude_modified=false` when labeling is
  C13 or N15 before reading/generating candidates.
- [ ] In assembly, reject C13/N15 input feature rows with a nonzero
  `modification_count` or nonempty modification metadata. Do not silently
  mix physically unsupported rows into positives or Gold negatives.
- [ ] Keep SILAC modification behavior unchanged.

### Isotope-envelope feature

The existing theory assumes natural abundance and takes no labeling or
enrichment parameter. Until a separate calibrated model exists:

- [ ] Keep the existing isotope calculation for SILAC.
- [ ] For C13/N15, emit `isotope_correlation = NaN` and
  `isotope_model_valid = 0`; emit `isotope_model_valid = 1` for SILAC.
- [ ] Ensure training handles the NaN consistently and cannot confuse a
  placeholder zero with a measured zero.
- [ ] Document that C13/N15 isotope evidence is disabled, not validated.

### Tests

- [ ] Parameterize modified/unmodified PSM tests across all label types.
- [ ] Assert both builder phases fail clearly for unsupported modified
  C13/N15 input.
- [ ] Assert isotope validity and NaN behavior through both workflow paths.

**Proposed commit:** `fix: reject unsupported uniform-label physics`

---

## 6. Task L4 — Harden the synthetic training-set contract

**Why:** Query generation writes a labeling value to the manifest, but
assembly currently ignores it and silently drops unavailable distribution
columns.

- [ ] Require `labeling` in the query manifest.
- [ ] Canonicalize all manifest values and require exactly one labeling type.
- [ ] Fail assembly when manifest labeling differs from `[assembly] labeling`.
- [ ] Merge `labeling` into output rows as audit metadata, never as a model
  feature.
- [ ] Change the default distribution field from `total_silac_shift` to
  `total_label_shift`.
- [ ] Require every explicitly configured distribution field in both positive
  and Silver tables; do not silently remove missing fields.
- [ ] Allow legacy `total_silac_shift` fallback only for SILAC, with a warning.
  C13/N15 must require the canonical field because old extraction used the
  wrong physics.
- [ ] Require at least one acquisition-range column:
  `heavy_in_raw` or `heavy_out_of_range`. If neither exists, fail rather than
  treating the mandatory Silver range gate as optional.
- [ ] Record the canonical labeling and selected shift column in audit JSON.

### Generator invariants

- [ ] Keep controlled shuffle composition-preserving for all labels.
- [ ] Keep Markov acceptance based on precursor-m/z and label-shift bins.
- [ ] Add tests proving Markov C13/N15 candidates need not have the parent's
  exact K/R count or exact C/N atom count.
- [ ] Confirm target, L/I, contaminant, duplicate, and parent-similarity
  exclusions remain unchanged.

### Builder test matrix

Parameterize generate and assemble tests over `silac`, `c13`, and `n15`, and
add explicit tests for:

- cross-phase labeling mismatch;
- mixed-label manifest rejection;
- missing canonical shift column;
- missing acquisition-range evidence;
- unsupported modifications;
- no-K/R uniform-label peptide retention;
- deterministic generation for the same seed.

**Tests:**

```bash
pytest -q tests/test_training_set_builder.py tests/test_feature_cols_contract.py
```

**Proposed commit:** `fix: enforce labeling contracts in hard-negative builder`

---

## 7. Task L5 — Make domain filtering label-aware

- [ ] Add a `heavy_type` parameter to `annotate_traps`, defaulting to SILAC
  for Python-call compatibility.
- [ ] Add `--labeling` to the trap-domain CLI and parse it through the shared
  labeling module.
- [ ] Replace the misleading internal `has_kr` concept with
  `has_label_site_for_mode` while preserving output reason
  `no_label_site`.
- [ ] Assert a no-K/R entrapment is dropped for SILAC but retained for C13 and
  N15 when it is otherwise L4 and in range.
- [ ] Check diagnostic tools that call `get_fragment_ions(HeavyType.SILAC)`;
  either add `--labeling` or explicitly document them as SILAC-only. Do not
  leave a tool appearing generic while silently using SILAC.

**Tests:**

```bash
pytest -q tests/test_trap_domain_filter.py tests/test_extract_label_site.py
```

**Proposed commit:** `fix: make entrapment filtering label-aware`

---

## 8. Task L6 — Documentation, migration, and verification

### Documentation

- [ ] Add `[general] labeling = silac` to `config.ini` with accepted aliases.
- [ ] Update README wording from SILAC-only to metabolic labeling where the
  behavior is generic; retain SILAC terminology where chemistry is specific.
- [ ] Update the hard-negative spec with the unmodified C13/N15 constraint,
  generic shift field, manifest consistency rule, and isotope limitation.
- [ ] Update `training_set_builder.ini.example` so query and assembly labels
  are visibly paired and the generic shift column is the default.

### Compatibility checks

- [ ] A config without `labeling` remains byte-for-byte equivalent in chosen
  masses/features to SILAC mode, except for deliberately added metadata and
  canonical shift columns.
- [ ] Existing imports of `HeavyType`, `get_heavy_increase_mass`, and
  `has_label_site` from `spectrum.psm_info` continue to work.
- [ ] Existing SILAC CSVs can still be assembled through the documented
  legacy shift fallback.
- [ ] Existing trained models can still locate their saved feature names.

### Automated verification

```bash
python -m compileall -q spectrum workflows tools
pytest -q \
  tests/test_labeling_end_to_end.py \
  tests/test_training_set_builder.py \
  tests/test_single_work_numerics.py \
  tests/test_extract_label_site.py \
  tests/test_psm_info_label_site.py \
  tests/test_heavy_mod_guard.py \
  tests/test_trap_domain_filter.py \
  tests/test_feature_cols_contract.py
pytest -q
git diff --check
```

The three existing rescore tests that invoke a system Python without
`lightgbm` remain an environment issue unless that interpreter is fixed; they
must not be misreported as labeling regressions.

### Real-data smoke verification

Run one small immutable raw/search-result slice per labeling mode with the
same executable path:

```bash
python main.py --configpath <silac.ini> --logpath <silac.log>
python main.py --configpath <c13.ini>   --logpath <c13.log>
python main.py --configpath <n15.ini>   --logpath <n15.log>
```

For each run verify:

- log records the canonical labeling before raw loading;
- sampled precursor and fragment m/z values independently match theoretical
  values;
- heavy requests remain inside expected acquisition windows;
- output contains one canonical labeling and correct total-label shifts;
- no modified C13/N15 PSM is silently processed;
- row losses and per-PSM errors are reported, with no unexplained spike.

If C13/N15 raw data are unavailable, implementation can be declared
**computationally covered for unmodified peptides**, but not empirically
validated.

**Proposed commit:** `docs: document metabolic labeling support matrix`

---

## 9. Completion gates

The repair is complete only when all statements below are true:

- [ ] The only production occurrences of `HeavyType.SILAC` are intentional
  defaults or explicitly SILAC-only behavior; none select chemistry inside a
  generic workflow.
- [ ] The only production calls to `get_SILAC_increase_mass` are inside the
  SILAC implementation or explicit compatibility tests.
- [ ] One config value determines precursor, fragment, Q1a, domain-filter,
  builder, and audit behavior.
- [ ] C13/N15 no-K/R peptides are retained.
- [ ] C13/N15 modified peptides fail before feature generation or assembly.
- [ ] Query/assembly labeling mismatch cannot pass silently.
- [ ] Missing Silver acquisition-range evidence cannot pass silently.
- [ ] Parameterized tests cover all three labels through the workflow
  interface, not only isolated mass helpers.
- [ ] Full regression tests pass apart from separately documented environment
  failures.
- [ ] Real-data evidence is clearly separated from computational support.

## 10. Recommended execution order

Implement and review one commit at a time:

1. L1 canonical labeling module.
2. L2 workflow propagation.
3. L3 unsupported-physics guards.
4. L4 builder contract.
5. L5 domain filters and diagnostic tools.
6. L6 documentation and full verification.

Do not regenerate large production feature tables until L1–L5 and the
targeted test matrix are green. After regeneration, never mix pre-fix C13/N15
features with post-fix features in one training or evaluation table.
