# Dual FDR Threshold for Negative Sample Expansion

**Status:** Approved (2026-06-03)
**Branch:** `feature_extraction`
**Plan file:** `docs/superpowers/plans/2026-06-03-dual-fdr-threshold.md` (to be created next)

---

## Background

Current `tools/extract_common.py` uses a single `qvalue_threshold` per engine (default 0.01) to filter PSMs before constructing positive / negative datasets. The same threshold gates both:

- Positive candidates (must pass intersection across all engines + match species marker)
- Negative candidates (key union across engines + species marker mismatch)

Result: when the user wants more negative samples to balance training, they must either lower the FDR threshold (polluting positives) or generate negatives by other means (shuffle entrapment, synthetic random pairs). Both add noise to the training signal.

## Goal

Add a per-engine `negative_qvalue_threshold` config option that **independently** controls the FDR cutoff for the NEGATIVE candidate pool. Positives stay strict; negatives can expand to a looser threshold (e.g., 0.10) for more training data.

## User-confirmed design decisions

1. **Strategy**: positive candidates use tight FDR; negative candidates use loose FDR (per-step filter). Positives stay pure.
2. **N-engine consistency**: positives still require intersection across engines under TIGHT threshold; negatives still use the existing union-of-keys semantic, but with LOOSE threshold per engine.
3. **Config location**: per-engine independent (each `[engine.X]` section can set `negative_qvalue_threshold`).
4. **Backward compatibility**: missing `negative_qvalue_threshold` falls back to `qvalue_threshold` (zero-migration). New behavior only activates when user explicitly sets a larger value.

## Architecture

### Data flow (new "dual pool" model)

```
extract_*.ini per [engine.X]:
    qvalue_threshold = 0.01           (tight, gates positive candidates)
    negative_qvalue_threshold = 0.10  (loose, gates negative candidates)
                                      (defaults to qvalue_threshold if absent)
                ↓
load_engine_psms_dual(engine_name, config) →
    {
      "tight": [PSM where q ≤ 0.01],   # positive candidate pool
      "loose": [PSM where q ≤ 0.10],   # negative candidate pool (⊇ tight)
    }
                ↓
extract_n_engines_from_psms_dual(engine_psms_dual, engine_order, positive_marker) →
    - intersection_keys = ∩(tight key_set per engine)   # positives
    - union_keys        = ∪(loose key_set per engine)   # negatives
    - same authoritative-PSM selection logic for marker check
                ↓
JSON output (schema unchanged, just more rows)
```

### Key invariants

1. **Positive candidate set is a subset of negative candidate set per-engine** — `q ≤ tight ⇒ q ≤ loose` (since loose ≥ tight). Mathematically guaranteed.
2. **Positives unaffected by loose threshold** — positive intersection is computed over TIGHT key sets only; PSMs in `(tight, loose]` range can become negatives but never positives.
3. **JSON schema unchanged** — only PSM count grows. Downstream `main.py → PairFlow → features.csv` requires no changes.
4. **Defensive guard**: `negative_qvalue_threshold < qvalue_threshold` raises `ValueError` (a tighter "negative" threshold would shrink the negative pool below positives — nonsensical).

## Files to modify

| File | Change |
|---|---|
| `tools/extract_common.py` | Refactor `load_engine_psms` → `load_engine_psms_dual` (returns dict). Refactor `extract_n_engines_from_psms` → `extract_n_engines_from_psms_dual` (consumes dict). Validate threshold ordering. |
| `spectrum/light_result.py` | **No change** — loaders stay single-threshold. We just call each loader twice (once tight, once loose). |
| `extract_2da_pfind_diann.ini` | Add example of `negative_qvalue_threshold = 0.10` in comments. Do NOT set non-default values — keep current production behavior unchanged until user explicitly opts in. |
| `tests/test_extract_common_dual_fdr.py` (new) | 5 tests covering backward compat, loose-pool expansion, positives unaffected, threshold-order validation, and JSON schema unchanged. |

### Files NOT modified

- `spectrum/psm_info.py` — PSM data class untouched.
- `workflows/*.py` — pair flow / single work untouched; consumes the JSON only.
- `tools/spec_trainer/*` — training pipeline untouched; CSV schema unchanged.

## Implementation sketch

### `load_engine_psms_dual` (replaces `load_engine_psms`)

```python
def load_engine_psms_dual(
    engine_name: str,
    config: configparser.ConfigParser,
) -> dict:
    """Load engine PSMs with optional dual FDR (tight for positives,
    loose for negatives).

    Returns:
        dict {"tight": [PSMInfo], "loose": [PSMInfo]}
        Both keys always present. When negative_qvalue_threshold ==
        qvalue_threshold (default), the two lists may share identity
        (same loader call) — but we always return both keys for
        downstream consistency.
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
            f"qvalue_threshold={tight} (negative pool must be ⊇ positive pool)")

    # Load tight pool (positive candidates).
    tight_psms = _load_engine(engine_name, path, tight)

    # Load loose pool (negative candidates). If loose == tight, reuse
    # tight_psms (avoid redundant I/O).
    if loose == tight:
        loose_psms = tight_psms
    else:
        loose_psms = _load_engine(engine_name, path, loose)

    return {"tight": tight_psms, "loose": loose_psms}


def _load_engine(engine_name, path, qvalue_threshold):
    """Internal helper: delegate to existing LightResult loaders."""
    lr = LightResult()
    if engine_name == "pfind":
        lr._load_from_pfind_input(path, qvalue_threshold=qvalue_threshold)
    elif engine_name == "diann":
        lr._load_from_dia_nn_input(path, qvalue_threshold=qvalue_threshold)
    elif engine_name == "alphadia":
        lr._load_from_alphadia_input(path, qvalue_threshold=qvalue_threshold)
    else:
        raise ValueError(f"不支持的引擎: {engine_name}")
    return lr.psm_info
```

### `extract_n_engines_from_psms_dual` (replaces single-pool version)

```python
def extract_n_engines_from_psms_dual(
    engine_psms_dual: dict,
    engine_order: list,
    positive_marker: Optional[str] = None,
) -> list:
    """Same algorithm as extract_n_engines_from_psms but:
    - intersection (positives) uses 'tight' key sets per engine
    - union (negatives) uses 'loose' key sets per engine

    Args:
        engine_psms_dual: dict[engine_name -> {"tight": [...], "loose": [...]}]
    """
    # Clear stale label_type on ALL PSMs (both pools).
    for pools in engine_psms_dual.values():
        for pool_psms in pools.values():
            for psm in pool_psms:
                psm._label_type = None

    # Tight key sets per engine -> positive intersection.
    tight_keys = {
        name: {p.get_key_with_raw() for p in pools["tight"]}
        for name, pools in engine_psms_dual.items()
    }
    intersection_keys = (set.intersection(*tight_keys.values())
                         if tight_keys else set())

    # Loose key sets per engine -> negative union.
    loose_keys = {
        name: {p.get_key_with_raw() for p in pools["loose"]}
        for name, pools in engine_psms_dual.items()
    }
    union_keys = set.union(*loose_keys.values()) if loose_keys else set()

    # Authoritative PSM selection: same priority as before, but search
    # in 'loose' pool (which is a superset of 'tight').
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

    # Build result list using the same positive/negative logic as before.
    # (Body identical to extract_n_engines_from_psms from line 125 onward.)
    ...
```

The body from line 125 (`result = []`) onward is identical to current `extract_n_engines_from_psms`.

### CLI wiring

`tools/extract_common.py:main()` switches from `load_engine_psms` to `load_engine_psms_dual` + `extract_n_engines_from_psms_dual`. JSON output format unchanged.

## Testing strategy

### Backward-compat tests

1. **`test_dual_fdr_default_matches_single_threshold`**: with no `negative_qvalue_threshold` set, the output of `extract_n_engines_from_psms_dual` is identical (same PSM count, same labels) to the legacy `extract_n_engines_from_psms`. Regression guard.

### Functional tests

2. **`test_dual_fdr_expanded_negative_pool_grows_negatives`**: synthetic PSM data with q-values in `[0.005, 0.05, 0.08, 0.15]`. With tight=0.01 negatives = those with q≤0.01 and not species_marker; with loose=0.10 negatives include q∈(0.01, 0.10]. Assert negative count strictly larger when loose > tight.

3. **`test_dual_fdr_positives_unaffected_by_loose_threshold`**: same synthetic data; vary `negative_qvalue_threshold` ∈ {0.01, 0.05, 0.10}; positive count must be invariant.

### Validation tests

4. **`test_dual_fdr_raises_when_loose_below_tight`**: `negative_qvalue_threshold=0.005` and `qvalue_threshold=0.01` → ValueError on `load_engine_psms_dual`.

### Schema preservation

5. **`test_dual_fdr_json_output_schema_unchanged`**: dump output to JSON with and without `negative_qvalue_threshold`; both files have identical fields per row (only row counts differ).

## Migration & deployment

- **Zero migration cost**: 3 production `extract_*.ini` files don't change. They continue using `qvalue_threshold=0.01` for both positive and negative pools.
- **Opt-in**: user adds `negative_qvalue_threshold = 0.10` to any engine section to expand negatives. Re-run `make 2th` (or equivalent) regenerates `datasets/hela_2da_pfind_diann.json` with more negative rows. Then `make all` regenerates `features.csv`.
- **Reproducibility**: explicitly setting both thresholds in ini ⇒ deterministic output. No env vars, no CLI flags.

## Out of scope

- Per-engine "negative_qvalue_threshold" overrides via CLI args (YAGNI; ini suffices).
- Per-PSM q-value weighting in training (would require `spec_trainer` changes; separate feature).
- Loose threshold for POSITIVE pool (rejected — would pollute training labels).
- Auto-tuning loose threshold to hit a target neg:pos ratio (YAGNI).
