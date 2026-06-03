# Rescore Tool Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Build `tools/spec_trainer/rescore.py` — re-evaluate the 18 trained LightGBM models at multiple thresholds without retraining; output long-form CSV + rich console table.

**Architecture:** Single-file CLI tool. Three internal layers: (1) pure metric computation `compute_metrics(y_true, y_proba, threshold)` — no LightGBM dependency; (2) data-source inference + model loading; (3) CLI wrapper + rich rendering. TDD-driven, 5 tests in `tests/test_rescore_tool.py`.

**Tech Stack:** Python 3.11+, lightgbm (load only), pandas, sklearn (train_test_split + roc_auc_score), rich (table rendering), argparse.

---

## File Structure

| Path | Responsibility |
|---|---|
| `tools/spec_trainer/rescore.py` | CLI + data inference + LightGBM scoring + metric loop + CSV + rich output |
| `tests/test_rescore_tool.py` | 5 tests: pure-metric monotonicity, in-sample sanity-check, cross-test data source, model filter, threshold CLI validation |

`rescore.py` is split into 3 pure functions + a `main()`:
- `compute_metrics(y_true, y_proba, threshold) -> dict` — pure, no LightGBM, easy to test
- `infer_data_source(model_basename) -> tuple[str, str]` — name → (csv_path, "in_sample" | "cross_test")
- `score_model(model_path, csv_path, mode, target_col) -> tuple[ndarray, ndarray]` — returns (y_true, y_proba) using shared 20%-split logic for in_sample, full file for cross_test
- `main()` — argparse, model discovery, loop, CSV write, rich table

---

## Task 1: Build the full rescore tool + 5 tests in one TDD pass

**Why one task:** Total ~200 lines (rescore.py) + ~250 lines (tests). Tightly coupled — splitting would force premature interface decisions.

**Files:**
- Create: `tools/spec_trainer/rescore.py`
- Create: `tests/test_rescore_tool.py`

---

- [ ] **Step 1: Create the empty test file with shared imports**

```bash
mkdir -p tests
```

Create `tests/test_rescore_tool.py`:

```python
"""Tests for tools/spec_trainer/rescore.py.

See docs/specs/2026-06-03-rescore-tool-design.md.

Two test classes:
- Pure-logic tests (no LightGBM, no data files) — always run
- Integration tests (need runs/spec_trainer/models/ + runs/baseline_*/features.csv)
  — auto-skip if artifacts missing.
"""
import json
import os
import subprocess
import sys

import numpy as np
import pytest

_PROJECT_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
_TOOL = os.path.join(_PROJECT_ROOT, "tools", "spec_trainer", "rescore.py")

if _PROJECT_ROOT not in sys.path:
    sys.path.insert(0, _PROJECT_ROOT)

# Import functions directly for unit tests.
# Path manipulation: add the tool's parent so 'rescore' resolves.
sys.path.insert(0, os.path.join(_PROJECT_ROOT, "tools", "spec_trainer"))
import rescore  # noqa: E402
```

- [ ] **Step 2: Write Test #3 (monotonicity, pure logic) — FAILING first**

Append to `tests/test_rescore_tool.py`:

```python
def test_rescore_threshold_monotonicity():
    """As threshold increases, neg_recall must monotonically increase
    (or stay the same), and pos_recall must monotonically decrease (or
    stay the same). Pure-logic test, no LightGBM."""
    # Construct fake probabilities and labels.
    # 100 samples, half positive, half negative.
    # Probabilities mixed so threshold sweep produces clear changes.
    rng = np.random.default_rng(42)
    n = 200
    y_true = np.array([1] * 100 + [0] * 100)
    # Positives have higher mean proba, negatives lower (but with overlap).
    proba = np.concatenate([
        rng.beta(8, 2, size=100),   # positives ~ 0.8
        rng.beta(2, 8, size=100),   # negatives ~ 0.2
    ])

    thresholds = [0.1, 0.3, 0.5, 0.7, 0.9]
    prev_neg_recall = -np.inf
    prev_pos_recall = np.inf
    for t in thresholds:
        m = rescore.compute_metrics(y_true, proba, t)
        assert m["neg_recall"] >= prev_neg_recall, (
            f"neg_recall not monotonic increasing: "
            f"prev={prev_neg_recall}, current={m['neg_recall']} at t={t}")
        assert m["pos_recall"] <= prev_pos_recall, (
            f"pos_recall not monotonic decreasing: "
            f"prev={prev_pos_recall}, current={m['pos_recall']} at t={t}")
        prev_neg_recall = m["neg_recall"]
        prev_pos_recall = m["pos_recall"]
```

- [ ] **Step 3: Run test, verify FAIL with import / attribute error**

Run:
```bash
conda run -n silac_ml pytest tests/test_rescore_tool.py::test_rescore_threshold_monotonicity -v
```

Expected: FAIL with `ModuleNotFoundError: No module named 'rescore'` or similar.

- [ ] **Step 4: Create rescore.py skeleton with compute_metrics implemented**

Create `tools/spec_trainer/rescore.py`:

```python
"""Re-score trained LightGBM models at multiple thresholds.

See docs/specs/2026-06-03-rescore-tool-design.md.
"""
from __future__ import annotations

import argparse
import csv
import glob
import logging
import os
import sys
from pathlib import Path

import numpy as np
import pandas as pd
from sklearn.metrics import confusion_matrix, roc_auc_score

# Lazy import lightgbm so that pure-logic tests don't need it.

_PROJECT_ROOT = Path(__file__).resolve().parents[2]
_SPEC_TRAINER_SRC = _PROJECT_ROOT / "tools" / "spec_trainer" / "src"
if str(_SPEC_TRAINER_SRC) not in sys.path:
    sys.path.insert(0, str(_SPEC_TRAINER_SRC))

logger = logging.getLogger(__name__)


def compute_metrics(y_true: np.ndarray, y_proba: np.ndarray,
                    threshold: float) -> dict:
    """Compute per-threshold classification metrics.

    Returns dict with keys: n_pos, n_neg, tn, fp, fn, tp,
    pos_recall, neg_recall, pos_precision, neg_precision, f1_neg, auc.

    `auc` is independent of threshold but included on each row for
    convenience (so a CSV row is fully self-contained).
    """
    y_true = np.asarray(y_true).astype(int)
    y_proba = np.asarray(y_proba).astype(float)
    y_pred = (y_proba > threshold).astype(int)

    cm = confusion_matrix(y_true, y_pred, labels=[0, 1])
    tn, fp = int(cm[0, 0]), int(cm[0, 1])
    fn, tp = int(cm[1, 0]), int(cm[1, 1])

    n_pos = int((y_true == 1).sum())
    n_neg = int((y_true == 0).sum())

    pos_recall = tp / n_pos if n_pos else 0.0
    neg_recall = tn / n_neg if n_neg else 0.0
    pos_prec = tp / (tp + fp) if (tp + fp) else 0.0
    neg_prec = tn / (tn + fn) if (tn + fn) else 0.0
    f1_neg = (
        2 * neg_prec * neg_recall / (neg_prec + neg_recall)
        if (neg_prec + neg_recall) else 0.0
    )
    auc = float(roc_auc_score(y_true, y_proba)) if n_pos and n_neg else float("nan")

    return {
        "n_pos": n_pos,
        "n_neg": n_neg,
        "tn": tn, "fp": fp, "fn": fn, "tp": tp,
        "pos_recall": pos_recall,
        "neg_recall": neg_recall,
        "pos_precision": pos_prec,
        "neg_precision": neg_prec,
        "f1_neg": f1_neg,
        "auc": auc,
    }
```

- [ ] **Step 5: Run test #3 — verify PASS**

Run:
```bash
conda run -n silac_ml pytest tests/test_rescore_tool.py::test_rescore_threshold_monotonicity -v
```

Expected: PASS.

- [ ] **Step 6: Write Test #5 (CLI argparse validation)**

Append to `tests/test_rescore_tool.py`:

```python
def test_rescore_invalid_threshold_rejected():
    """--thresholds 1.5 should be rejected by argparse (threshold must
    be in (0, 1) exclusive)."""
    result = subprocess.run(
        [sys.executable, _TOOL, "--thresholds", "1.5"],
        capture_output=True, text=True)
    assert result.returncode != 0, "argparse must reject threshold > 1"
    combined = result.stdout + result.stderr
    assert "0" in combined and "1" in combined, (
        f"Error message should mention valid range; got:\n{combined}")


def test_rescore_threshold_zero_rejected():
    """--thresholds 0 should be rejected (must be > 0)."""
    result = subprocess.run(
        [sys.executable, _TOOL, "--thresholds", "0"],
        capture_output=True, text=True)
    assert result.returncode != 0


def test_rescore_threshold_one_rejected():
    """--thresholds 1.0 should be rejected (must be < 1)."""
    result = subprocess.run(
        [sys.executable, _TOOL, "--thresholds", "1.0"],
        capture_output=True, text=True)
    assert result.returncode != 0
```

- [ ] **Step 7: Run tests — verify FAIL (rescore.py has no main yet)**

Run:
```bash
conda run -n silac_ml pytest tests/test_rescore_tool.py -v -k "rejected"
```

Expected: FAIL (script runs without crashing because no argparse; or AttributeError, depending on state).

- [ ] **Step 8: Add CLI argparse + threshold validator**

Append to `tools/spec_trainer/rescore.py`:

```python
def _threshold_arg(s: str) -> float:
    """Argparse validator: 0 < t < 1."""
    try:
        t = float(s)
    except ValueError:
        raise argparse.ArgumentTypeError(
            f"threshold must be a float, got {s!r}")
    if not (0.0 < t < 1.0):
        raise argparse.ArgumentTypeError(
            f"threshold must be in (0, 1) exclusive, got {t}")
    return t


def _build_parser() -> argparse.ArgumentParser:
    p = argparse.ArgumentParser(
        description="Re-evaluate trained LightGBM models at multiple thresholds.")
    p.add_argument(
        "--thresholds", type=_threshold_arg, nargs="+", required=True,
        help="Decision thresholds to evaluate (e.g. 0.5 0.7 0.9 0.99). "
             "Each value must be in (0, 1) exclusive.")
    p.add_argument(
        "--models", nargs="+", default=None,
        help="Model basenames to include (e.g. in_2da_clean). "
             "Default: all .txt files in runs/spec_trainer/models/.")
    p.add_argument(
        "--output", type=str, default="runs/spec_trainer/rescore_summary.csv",
        help="Output CSV path.")
    p.add_argument(
        "--models-dir", type=str, default="runs/spec_trainer/models",
        help="Directory containing .txt model files.")
    p.add_argument(
        "--features-root", type=str, default="runs/baseline_{ds}_{fdr}",
        help="features.csv parent dir template; {ds} ∈ {2da,5da,normal}, "
             "{fdr} ∈ {clean,neg05,neg10}.")
    return p


def main(argv=None) -> int:
    parser = _build_parser()
    args = parser.parse_args(argv)
    # Stub — to be expanded.
    return 0


if __name__ == "__main__":
    sys.exit(main())
```

- [ ] **Step 9: Run CLI validation tests — verify PASS**

Run:
```bash
conda run -n silac_ml pytest tests/test_rescore_tool.py -v -k "rejected"
```

Expected: 3 PASS.

- [ ] **Step 10: Add `infer_data_source` + `discover_models` + `score_model` + `main` body**

Append to `tools/spec_trainer/rescore.py`:

```python
def infer_data_source(model_basename: str,
                       features_root_template: str) -> tuple[Path, str]:
    """Map a model basename to (csv_path, mode).

    - 'in_<ds>_<fdr>' → (runs/baseline_<ds>_<fdr>/features.csv, 'in_sample')
    - 'cross_test_<held>_<fdr>' → (runs/baseline_<held>_<fdr>/features.csv,
                                    'cross_test')

    Raises ValueError on unrecognized format.
    """
    if model_basename.startswith("in_"):
        parts = model_basename.split("_")
        if len(parts) != 3:
            raise ValueError(
                f"in_* model name must have 3 underscore parts, "
                f"got {model_basename!r}")
        _, ds, fdr = parts
        csv = features_root_template.format(ds=ds, fdr=fdr) + "/features.csv"
        return Path(csv), "in_sample"

    if model_basename.startswith("cross_test_"):
        parts = model_basename.split("_")
        if len(parts) != 4:
            raise ValueError(
                f"cross_test_* model name must have 4 underscore parts, "
                f"got {model_basename!r}")
        _, _, held, fdr = parts
        csv = features_root_template.format(ds=held, fdr=fdr) + "/features.csv"
        return Path(csv), "cross_test"

    raise ValueError(
        f"Model name must start with 'in_' or 'cross_test_'; "
        f"got {model_basename!r}")


def discover_models(models_dir: Path,
                     filter_names: list[str] | None) -> list[Path]:
    """List .txt files under models_dir. If filter_names given, restrict
    to those (matched by basename without .txt)."""
    all_paths = sorted(models_dir.glob("*.txt"))
    if filter_names is None:
        return all_paths
    wanted = set(filter_names)
    found = [p for p in all_paths if p.stem in wanted]
    missing = wanted - {p.stem for p in found}
    if missing:
        logger.warning(
            "Requested models not found in %s: %s", models_dir, sorted(missing))
    return found


def score_model(model_path: Path, csv_path: Path, mode: str,
                target_col: str = "label") -> tuple[np.ndarray, np.ndarray]:
    """Load model, load CSV, return (y_true, y_proba).

    mode='in_sample' uses the SAME 20% stratified split as training
    (random_state=42, stratify=y); 'cross_test' uses the full CSV.

    Feature columns auto-resolved via feature_cols.resolve_feature_cols
    so the EXCLUDED_EXTRA list (window_width etc.) stays consistent.
    """
    # Lazy import lightgbm and feature_cols so pure tests don't need them.
    import lightgbm as lgb
    from feature_cols import resolve_feature_cols
    from sklearn.model_selection import train_test_split

    booster = lgb.Booster(model_file=str(model_path))
    df = pd.read_csv(csv_path)
    feature_cols = resolve_feature_cols(
        explicit=None,
        sample_csv_paths=[str(csv_path)],
        target_col=target_col,
    )

    if mode == "in_sample":
        # Replicate the EXACT split used by spec_trainer's resolve_holdout.
        # spec_trainer uses test_size=0.2, random_state=42, stratify=y.
        y_full = df[target_col].astype(int)
        X_full = df[feature_cols]
        _, X_te, _, y_te = train_test_split(
            X_full, y_full, test_size=0.2, random_state=42,
            stratify=y_full)
    elif mode == "cross_test":
        X_te = df[feature_cols]
        y_te = df[target_col].astype(int)
    else:
        raise ValueError(f"Unknown mode: {mode!r}")

    y_proba = booster.predict(X_te.values)
    return y_te.values, y_proba


def _print_console_table(rows: list[dict]) -> None:
    """Render rows as a rich.Table, grouped by experiment."""
    try:
        from rich.console import Console
        from rich.table import Table
    except ImportError:
        # rich is optional; fall back to plain print.
        for r in rows:
            print(r)
        return

    table = Table(title="Rescore Summary", show_lines=False)
    cols = [
        "experiment", "thresh", "n_pos", "n_neg",
        "TN", "FP", "FN", "TP",
        "PosRec", "NegRec", "PosPrec", "NegPrec", "F1_neg", "AUC",
    ]
    for c in cols:
        table.add_column(c, justify="right" if c not in ("experiment",) else "left")

    last_exp = None
    for r in rows:
        exp = r["experiment"]
        # Insert separator between experiments for readability.
        if last_exp is not None and exp != last_exp:
            table.add_section()
        last_exp = exp
        table.add_row(
            exp,
            f"{r['threshold']:.3f}",
            str(r["n_pos"]),
            str(r["n_neg"]),
            str(r["tn"]), str(r["fp"]), str(r["fn"]), str(r["tp"]),
            f"{r['pos_recall']:.4f}",
            f"{r['neg_recall']:.4f}",
            f"{r['pos_precision']:.4f}",
            f"{r['neg_precision']:.4f}",
            f"{r['f1_neg']:.4f}",
            f"{r['auc']:.4f}" if not np.isnan(r["auc"]) else "n/a",
        )
    Console().print(table)


def _write_csv(rows: list[dict], output_path: Path) -> None:
    output_path.parent.mkdir(parents=True, exist_ok=True)
    fieldnames = [
        "experiment", "threshold",
        "n_pos", "n_neg", "tn", "fp", "fn", "tp",
        "pos_recall", "neg_recall", "pos_precision", "neg_precision",
        "f1_neg", "auc",
    ]
    with open(output_path, "w", newline="") as f:
        w = csv.DictWriter(f, fieldnames=fieldnames)
        w.writeheader()
        for r in rows:
            w.writerow({k: r[k] for k in fieldnames})


def main(argv=None) -> int:
    parser = _build_parser()
    args = parser.parse_args(argv)

    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s | %(levelname)s | %(message)s")

    models_dir = Path(args.models_dir)
    if not models_dir.exists():
        logger.error("Models dir not found: %s", models_dir)
        return 1

    model_paths = discover_models(models_dir, args.models)
    if not model_paths:
        logger.error("No models matched %r in %s", args.models, models_dir)
        return 1

    logger.info("Found %d models", len(model_paths))

    rows = []
    for mp in model_paths:
        try:
            csv_path, mode = infer_data_source(mp.stem, args.features_root)
        except ValueError as e:
            logger.warning("Skipping %s: %s", mp.name, e)
            continue
        if not csv_path.exists():
            logger.warning("Skipping %s: features.csv not found at %s",
                           mp.name, csv_path)
            continue
        logger.info("Scoring %s (mode=%s, data=%s)", mp.stem, mode, csv_path)
        try:
            y_true, y_proba = score_model(mp, csv_path, mode)
        except Exception as e:
            logger.warning("Skipping %s: scoring failed: %s", mp.name, e)
            continue

        for t in args.thresholds:
            m = compute_metrics(y_true, y_proba, t)
            rows.append({"experiment": mp.stem, "threshold": t, **m})

    if not rows:
        logger.error("No rows produced; aborting.")
        return 1

    output_path = Path(args.output)
    _write_csv(rows, output_path)
    logger.info("Wrote %d rows to %s", len(rows), output_path)
    _print_console_table(rows)
    return 0
```

- [ ] **Step 11: Write Test #4 (--models filter)**

Append to `tests/test_rescore_tool.py`:

```python
def _have_artifact(rel_path: str) -> bool:
    """Check whether a project-relative path exists."""
    return os.path.exists(os.path.join(_PROJECT_ROOT, rel_path))


@pytest.mark.skipif(
    not _have_artifact("runs/spec_trainer/models/in_2da_clean.txt"),
    reason="model not trained yet")
@pytest.mark.skipif(
    not _have_artifact("runs/baseline_2da_clean/features.csv"),
    reason="features.csv not generated yet")
def test_rescore_models_filter(tmp_path):
    """--models in_2da_clean should produce 1 experiment × N thresholds rows."""
    output = tmp_path / "out.csv"
    result = subprocess.run(
        [sys.executable, _TOOL,
         "--thresholds", "0.5", "0.9",
         "--models", "in_2da_clean",
         "--output", str(output)],
        capture_output=True, text=True, cwd=_PROJECT_ROOT)
    assert result.returncode == 0, (
        f"rescore failed:\nSTDOUT:\n{result.stdout}\nSTDERR:\n{result.stderr}")
    import csv as _csv
    with open(output) as f:
        rows = list(_csv.DictReader(f))
    assert len(rows) == 2, f"expected 2 rows (1 model × 2 thresholds), got {len(rows)}"
    exps = {r["experiment"] for r in rows}
    assert exps == {"in_2da_clean"}, f"expected only in_2da_clean; got {exps}"
```

- [ ] **Step 12: Run filter test — verify PASS**

Run:
```bash
conda run -n silac_ml pytest tests/test_rescore_tool.py::test_rescore_models_filter -v
```

Expected: PASS (assuming `runs/spec_trainer/models/in_2da_clean.txt` exists from prior smoke test).

If model file missing, test auto-skips — that's fine. **Manually run the tool to verify**:
```bash
conda run -n silac_ml python3 tools/spec_trainer/rescore.py \
    --thresholds 0.5 0.9 \
    --models in_2da_clean \
    --output /tmp/rescore_test.csv
cat /tmp/rescore_test.csv
```

Expected: header + 2 rows.

- [ ] **Step 13: Write Test #1 (sanity check against existing in_2da_clean.json)**

Append to `tests/test_rescore_tool.py`:

```python
@pytest.mark.skipif(
    not _have_artifact("runs/spec_trainer/models/in_2da_clean.txt"),
    reason="model not trained yet")
@pytest.mark.skipif(
    not _have_artifact("runs/spec_trainer/results/in_2da_clean.json"),
    reason="result JSON not present")
@pytest.mark.skipif(
    not _have_artifact("runs/baseline_2da_clean/features.csv"),
    reason="features.csv not generated yet")
def test_rescore_in_sample_split_matches_training(tmp_path):
    """rescore.py at threshold 0.5 must reproduce the confusion matrix
    that spec_trainer recorded in results/in_2da_clean.json.

    Both use random_state=42 + test_size=0.2 + stratify=y → must match
    exactly.
    """
    output = tmp_path / "out.csv"
    result = subprocess.run(
        [sys.executable, _TOOL,
         "--thresholds", "0.5",
         "--models", "in_2da_clean",
         "--output", str(output)],
        capture_output=True, text=True, cwd=_PROJECT_ROOT)
    assert result.returncode == 0, f"rescore failed: {result.stderr}"

    import csv as _csv
    with open(output) as f:
        rows = list(_csv.DictReader(f))
    assert len(rows) == 1
    r = rows[0]

    with open(os.path.join(
            _PROJECT_ROOT, "runs/spec_trainer/results/in_2da_clean.json")) as f:
        ref = json.load(f)
    ref_cm = ref["confusion_matrix"]  # [[TN, FP], [FN, TP]]
    ref_tn, ref_fp = ref_cm[0]
    ref_fn, ref_tp = ref_cm[1]

    assert int(r["tn"]) == ref_tn, (
        f"TN mismatch: rescore={r['tn']}, json={ref_tn}")
    assert int(r["fp"]) == ref_fp
    assert int(r["fn"]) == ref_fn
    assert int(r["tp"]) == ref_tp
    # AUC is threshold-free; should match exactly.
    assert abs(float(r["auc"]) - float(ref["auc"])) < 1e-6, (
        f"AUC mismatch: rescore={r['auc']}, json={ref['auc']}")
```

- [ ] **Step 14: Run sanity check test — verify PASS**

Run:
```bash
conda run -n silac_ml pytest tests/test_rescore_tool.py::test_rescore_in_sample_split_matches_training -v
```

Expected: PASS (the trained model + result JSON are both in place).

If FAIL on TN/FP/FN/TP mismatch, the data-source / split logic doesn't match training — debug `infer_data_source` and `score_model`'s `train_test_split` call (must be `random_state=42, stratify=y, test_size=0.2`).

- [ ] **Step 15: Write Test #2 (cross_test uses full file)**

Append to `tests/test_rescore_tool.py`:

```python
@pytest.mark.skipif(
    not _have_artifact("runs/spec_trainer/models/cross_test_2da_clean.txt"),
    reason="cross_test model not trained yet")
@pytest.mark.skipif(
    not _have_artifact("runs/baseline_2da_clean/features.csv"),
    reason="features.csv not generated yet")
def test_rescore_cross_test_uses_full_held_file(tmp_path):
    """cross_test_<X>_<fdr> uses ENTIRE features.csv as test set
    (not a 20% split). Row count of test set must equal full CSV rows.
    """
    output = tmp_path / "out.csv"
    result = subprocess.run(
        [sys.executable, _TOOL,
         "--thresholds", "0.5",
         "--models", "cross_test_2da_clean",
         "--output", str(output)],
        capture_output=True, text=True, cwd=_PROJECT_ROOT)
    assert result.returncode == 0, f"rescore failed: {result.stderr}"

    import csv as _csv
    with open(output) as f:
        rows = list(_csv.DictReader(f))
    r = rows[0]
    test_total = int(r["n_pos"]) + int(r["n_neg"])

    full_csv_rows = sum(
        1 for _ in open(os.path.join(
            _PROJECT_ROOT, "runs/baseline_2da_clean/features.csv"))) - 1  # minus header

    assert test_total == full_csv_rows, (
        f"cross_test should use full file: test_total={test_total}, "
        f"full_csv_rows={full_csv_rows}")
```

- [ ] **Step 16: Run cross_test test — verify PASS**

Run:
```bash
conda run -n silac_ml pytest tests/test_rescore_tool.py::test_rescore_cross_test_uses_full_held_file -v
```

Expected: PASS.

- [ ] **Step 17: Run full test file + full regression**

```bash
conda run -n silac_ml pytest tests/test_rescore_tool.py -v
conda run -n silac_ml pytest 2>&1 | tail -3
```

Expected:
- `test_rescore_tool.py`: 7 PASS (1 monotonicity + 3 CLI validation + 1 filter + 1 sanity + 1 cross_test)
- Full regression: was 422 → now 429 (+7 new), no regressions

- [ ] **Step 18: End-to-end smoke test on all 18 models, 4 thresholds**

```bash
conda run -n silac_ml python3 tools/spec_trainer/rescore.py \
    --thresholds 0.5 0.7 0.9 0.99 \
    --output runs/spec_trainer/rescore_summary.csv
head -5 runs/spec_trainer/rescore_summary.csv
wc -l runs/spec_trainer/rescore_summary.csv
```

Expected:
- 18 models × 4 thresholds = 72 data rows + 1 header = **73 lines**
- Header: `experiment,threshold,n_pos,n_neg,tn,fp,fn,tp,pos_recall,neg_recall,pos_precision,neg_precision,f1_neg,auc`
- Console prints a `rich` table grouped by experiment

If models are missing, the tool should log warnings and skip, not crash.

- [ ] **Step 19: Commit**

```bash
git add tools/spec_trainer/rescore.py tests/test_rescore_tool.py
git commit -m "feat(spec_trainer): rescore tool — multi-threshold post-hoc evaluation

Build tools/spec_trainer/rescore.py: load each trained LightGBM model,
re-score against either the matching 20%-split (in_*) or the full
held-out features.csv (cross_test_*), then compute per-threshold
metrics for any number of user-supplied thresholds.

CLI:
  python tools/spec_trainer/rescore.py \\
    --thresholds 0.5 0.7 0.9 0.99 \\
    [--models in_2da_clean cross_test_normal_neg10 ...] \\
    [--output runs/spec_trainer/rescore_summary.csv]

Output: long-form CSV (experiment, threshold, n_pos, n_neg, tn, fp,
fn, tp, pos_recall, neg_recall, pos_precision, neg_precision, f1_neg,
auc) + rich.Table console rendering grouped by experiment.

Key design:
- Pure compute_metrics() is LightGBM-free → testable without GPU/
  heavy import.
- infer_data_source() uses model basename prefix (in_ or cross_test_)
  to map to the right features.csv and split mode.
- score_model() reuses feature_cols.resolve_feature_cols so the
  EXCLUDED_EXTRA list (window_width etc.) stays consistent with
  training.
- random_state=42 + stratify=y in the in_sample 20% split exactly
  reproduces spec_trainer's resolve_holdout split.

7 tests:
- 1 monotonicity (pure logic, no LightGBM)
- 3 CLI threshold validation (rejects 0, 1, 1.5)
- 1 models filter
- 1 sanity check: rescore at threshold 0.5 reproduces existing
  in_2da_clean.json confusion matrix exactly
- 1 cross_test data source check: row count matches full features.csv

Tests: 429 passed (was 422 + 7 new).

Spec: docs/specs/2026-06-03-rescore-tool-design.md"
```

---

## Self-Review

**Spec coverage:**
- ✅ CLI form `--thresholds T1 T2 ... --models name1 name2 ... --output PATH` (Step 8)
- ✅ Data source inference: `in_<ds>_<fdr>` → 20% split with random_state=42; `cross_test_<held>_<fdr>` → full file (Step 10 `infer_data_source` + `score_model`)
- ✅ feature_cols.resolve_feature_cols reused (Step 10 `score_model`)
- ✅ CSV schema matches spec (Step 10 `_write_csv` fieldnames)
- ✅ Rich console (Step 10 `_print_console_table` with fallback)
- ✅ Sanity check test (Step 13)
- ✅ Threshold validator 0 < t < 1 (Step 8 `_threshold_arg`)
- ✅ Missing model file / features.csv → skip + warn (Step 10 `main` loop)
- ✅ All 5 spec tests covered: monotonicity (#3), sanity (#1), cross_test (#2), models filter (#4), invalid threshold (#5 → expanded to 3 tests covering edge cases 0, 1, 1.5)

**Placeholder scan:** None. All code blocks are complete.

**Type consistency:**
- `compute_metrics` returns dict with the same keys used by `_write_csv` fieldnames
- `infer_data_source` returns `(Path, str)`; both consumers handle accordingly
- `score_model` returns `(np.ndarray, np.ndarray)`; both feed into `compute_metrics`
- CLI `--features-root` template uses `{ds}` and `{fdr}` consistently across `infer_data_source`

No gaps. Plan ready.
