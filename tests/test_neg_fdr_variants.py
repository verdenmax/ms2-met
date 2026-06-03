"""Tests for neg-FDR variant infrastructure.

Verifies the 6 baseline configs (runs/baseline_*_neg{05,10}/config.ini)
have correct field substitutions, and the Makefile lists all 21 expected
targets.

See docs/specs/2026-06-03-neg-fdr-variants-design.md.
"""
import configparser
import os
import subprocess

import pytest

_PROJECT_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))


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

    light = cfg.get("input", "light_result_file")
    assert light.endswith(f"hela_{dataset}_{fdr}.json"), (
        f"{cfg_path}: light_result_file={light!r} should end with "
        f"hela_{dataset}_{fdr}.json")

    work = cfg.get("general", "work_directory")
    # Shared workspace at project root (./workspace), side-by-side with
    # runs/. All 9 baselines share the same .dia.npz cache (safe because
    # all baselines use identical centroid params; P0-3 version+params
    # validation handles invalidation if anyone changes centroid config).
    assert work == "./workspace", (
        f"{cfg_path}: work_directory={work!r} should be './workspace' "
        f"(shared across all 9 baselines for cache reuse)")

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
            continue
        clean_val = clean.get(section, option)
        variant_val = variant.get(section, option)
        assert clean_val == variant_val, (
            f"{variant_path}: {section}.{option}={variant_val!r} differs "
            f"from clean {clean_val!r} (should be identical — dataset, "
            f"not FDR, property)")


_EXPECTED_TARGETS = [
    "2th-neg05", "2th-neg10",
    "5th-neg05", "5th-neg10",
    "normal-neg05", "normal-neg10",
    "extract-2th-neg05", "extract-2th-neg10",
    "extract-5th-neg05", "extract-5th-neg10",
    "extract-normal-neg05", "extract-normal-neg10",
    "clean-2th-neg05", "clean-2th-neg10",
    "clean-5th-neg05", "clean-5th-neg10",
    "clean-normal-neg05", "clean-normal-neg10",
    "all-clean", "all-neg05", "all-neg10",
]


@pytest.mark.parametrize("target", _EXPECTED_TARGETS)
def test_makefile_target_exists(target):
    """Each new neg-FDR target must be invokable via `make -n <target>`
    without 'No rule to make target' error."""
    result = subprocess.run(
        ["make", "-n", target],
        cwd=_PROJECT_ROOT,
        capture_output=True, text=True)
    combined = result.stdout + result.stderr
    assert "No rule to make target" not in combined, (
        f"Makefile target {target!r} not found:\n{combined}")


def test_makefile_phony_includes_new_targets():
    """All new neg-FDR targets must be listed in .PHONY (avoids
    accidental file/directory shadowing)."""
    makefile_path = os.path.join(_PROJECT_ROOT, "Makefile")
    with open(makefile_path) as f:
        content = f.read()
    phony_lines = [line for line in content.splitlines()
                   if line.startswith(".PHONY:")]
    phony_targets = set()
    for line in phony_lines:
        phony_targets.update(line.replace(".PHONY:", "").split())
    for target in _EXPECTED_TARGETS:
        assert target in phony_targets, (
            f"Target {target!r} missing from .PHONY declarations. "
            f"Add it to one of:\n" + "\n".join(phony_lines))
