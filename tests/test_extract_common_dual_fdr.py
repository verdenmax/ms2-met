"""Tests for dual-FDR-threshold loader and extractor.

See docs/specs/2026-06-03-dual-fdr-threshold-design.md.
"""
import configparser
import os
import sys
import pytest

# Make tools/extract_common importable.
_PROJECT_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
if _PROJECT_ROOT not in sys.path:
    sys.path.insert(0, _PROJECT_ROOT)


def _make_config(qvalue: str = "0.01",
                 negative_qvalue: str | None = None) -> configparser.ConfigParser:
    """Build a minimal config with one pfind engine.

    qvalue / negative_qvalue are strings (raw ini values).
    """
    cfg = configparser.ConfigParser()
    section = {"path": "/nonexistent/dummy.qry.res", "qvalue_threshold": qvalue}
    if negative_qvalue is not None:
        section["negative_qvalue_threshold"] = negative_qvalue
    cfg.read_dict({"engine.pfind": section})
    return cfg


def test_load_engine_psms_dual_returns_tight_and_loose_keys(monkeypatch):
    """When negative_qvalue_threshold is absent, returned dict has both
    keys and the lists share identity (zero redundant I/O)."""
    from tools import extract_common

    captured_calls = []

    def fake_load(engine_name, path, qvalue):
        captured_calls.append((engine_name, path, qvalue))
        return [f"psm_{qvalue}"]

    monkeypatch.setattr(extract_common, "_load_engine", fake_load)

    cfg = _make_config(qvalue="0.01")
    result = extract_common.load_engine_psms_dual("pfind", cfg)

    assert set(result.keys()) == {"tight", "loose"}
    assert len(captured_calls) == 1, (
        f"Expected single load when thresholds equal; got {captured_calls}")
    assert result["tight"] is result["loose"], (
        "When loose == tight, the two lists should share identity")


def test_load_engine_psms_dual_loads_twice_when_thresholds_differ(monkeypatch):
    """When negative_qvalue_threshold > qvalue_threshold, the loader is
    called twice with each threshold and results are different lists."""
    from tools import extract_common

    captured_calls = []

    def fake_load(engine_name, path, qvalue):
        captured_calls.append((engine_name, path, qvalue))
        return [f"psm_q{qvalue}"]

    monkeypatch.setattr(extract_common, "_load_engine", fake_load)

    cfg = _make_config(qvalue="0.01", negative_qvalue="0.10")
    result = extract_common.load_engine_psms_dual("pfind", cfg)

    assert len(captured_calls) == 2
    qvalues_called = sorted(c[2] for c in captured_calls)
    assert qvalues_called == [0.01, 0.10]
    assert result["tight"] is not result["loose"]


def test_load_engine_psms_dual_raises_when_loose_below_tight(monkeypatch):
    """negative_qvalue_threshold < qvalue_threshold must raise ValueError."""
    from tools import extract_common

    monkeypatch.setattr(extract_common, "_load_engine",
                         lambda *a, **k: [])

    cfg = _make_config(qvalue="0.01", negative_qvalue="0.005")
    with pytest.raises(ValueError, match="negative_qvalue_threshold"):
        extract_common.load_engine_psms_dual("pfind", cfg)


def test_load_engine_psms_dual_raises_when_engine_section_missing():
    """Missing [engine.X] section raises ValueError."""
    from tools import extract_common
    cfg = configparser.ConfigParser()
    with pytest.raises(ValueError, match="engine.pfind"):
        extract_common.load_engine_psms_dual("pfind", cfg)


def test_load_engine_psms_dual_raises_when_path_missing(monkeypatch):
    """Engine section without 'path' raises ValueError."""
    from tools import extract_common
    cfg = configparser.ConfigParser()
    cfg.read_dict({"engine.pfind": {"qvalue_threshold": "0.01"}})
    with pytest.raises(ValueError, match="path"):
        extract_common.load_engine_psms_dual("pfind", cfg)
