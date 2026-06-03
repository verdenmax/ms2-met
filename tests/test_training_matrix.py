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

    assert data["train_files"] == [f"runs/baseline_{ds}_{fdr}/features.csv"], (
        f"in_{ds}_{fdr}: wrong train_files = {data['train_files']!r}")
    assert data["test_files"] == [], (
        f"in_{ds}_{fdr}: test_files must be [] for in-sample; "
        f"got {data['test_files']!r}")
    assert data["test_size"] == 0.2, (
        f"in_{ds}_{fdr}: test_size must be 0.2 for 80/20 split; "
        f"got {data['test_size']!r}")
    assert data["target_col"] == "label"

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

    held_path = f"runs/baseline_{held}_{fdr}/features.csv"
    assert held_path not in data["train_files"], (
        f"cross_test_{held}_{fdr}: DATA LEAKAGE — held-out {held_path!r} "
        f"in train_files {data['train_files']!r}")

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
    """make -n train-all should reference all 3 FDR groups."""
    result = subprocess.run(
        ["make", "-n", "train-all"],
        cwd=_PROJECT_ROOT,
        capture_output=True, text=True)
    out = result.stdout + result.stderr
    for marker in ("in_2da_clean.yaml", "in_2da_neg05.yaml",
                   "in_2da_neg10.yaml"):
        assert marker in out, (
            f"train-all dry-run should expand all 3 FDR groups; "
            f"missing reference to {marker!r}")


def test_makefile_train_all_uses_recursive_make_for_sequential_order():
    """train-all must invoke sub-targets via $(MAKE) (recursive) rather
    than phony prereqs, so that 'make -j N train-all' still runs the
    3 FDR groups strictly sequentially (clean → neg05 → neg10), per
    the documented order.

    Regression for code-review finding (2026-06-03): phony-prereq form
    `train-all: train-clean-all train-neg05-all train-neg10-all` would
    let make parallelize the 3 groups under -j, interleaving banners
    and breaking the documented sequential order.
    """
    makefile_path = os.path.join(_PROJECT_ROOT, "Makefile")
    with open(makefile_path) as f:
        content = f.read()

    # Find the train-all target block: everything from 'train-all:' to
    # the next blank line / next target. Recursive $(MAKE) calls must
    # be present for each of the 3 FDR groups.
    in_block = False
    block_lines = []
    for line in content.splitlines():
        if line.startswith("train-all:"):
            in_block = True
            block_lines.append(line)
            continue
        if in_block:
            if line.startswith("\t") or line == "":
                block_lines.append(line)
                if line == "":
                    break
            else:
                break
    block = "\n".join(block_lines)

    for sub in ("train-clean-all", "train-neg05-all", "train-neg10-all"):
        assert f"$(MAKE) {sub}" in block, (
            f"train-all should invoke '{sub}' via recursive $(MAKE) "
            f"to enforce sequential execution; got block:\n{block}")


def test_makefile_neg_features_have_autobuild_rules():
    """All 9 features.csv paths used by train-*-all must have a pattern
    rule so 'make' can auto-trigger extraction when the file is missing.

    Regression for code-review finding (2026-06-03): originally only the
    3 *_clean features.csv had autobuild rules; the 6 neg{05,10}
    variants would fail opaquely if the user ran train-neg05-all on a
    fresh checkout.
    """
    makefile_path = os.path.join(_PROJECT_ROOT, "Makefile")
    with open(makefile_path) as f:
        content = f.read()

    for ds in _DATASETS:
        for fdr in _FDRS:
            target = f"runs/baseline_{ds}_{fdr}/features.csv:"
            assert target in content, (
                f"Missing autobuild rule for {target!r}; train-*-all "
                f"cannot guarantee features.csv presence")


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
