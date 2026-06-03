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


def _make_psm(seq: str, charge: int, raw: str, proteins: str):
    """Build a minimal PSMInfo for tests.

    NOTE: modify=[] (empty LIST, not string) — get_key_with_raw iterates
    modify with tuple(tuple(pair) for pair in self._modify), which would
    break on string chars.
    """
    from spectrum.psm_info import PSMInfo
    return PSMInfo(
        sequence=seq, charge=charge, modify=[],
        rt=10.0, precursor_mz=500.0,
        raw_title=raw, protein_names=proteins,
        q_value=0.0,
    )


def test_extract_dual_default_matches_single_threshold():
    """When tight == loose (single pool), extract_n_engines_from_psms_dual
    output identical to extract_n_engines_from_psms."""
    from tools import extract_common

    p_human = _make_psm("PEPTIDEK", 2, "run1", "sp|P00000|HUMAN")
    p_ecoli = _make_psm("AAAAAAAR", 2, "run1", "sp|Q00000|ECOLI")
    p_unique = _make_psm("UNIQUEK", 2, "run1", "sp|P11111|HUMAN")

    engines_single = {
        "pfind": [p_human, p_ecoli, p_unique],
        "diann": [p_human, p_ecoli],
    }
    engines_dual = {
        "pfind": {"tight": engines_single["pfind"],
                  "loose": engines_single["pfind"]},
        "diann": {"tight": engines_single["diann"],
                  "loose": engines_single["diann"]},
    }

    out_single = extract_common.extract_n_engines_from_psms(
        engines_single, ["pfind", "diann"], positive_marker="HUMAN")
    out_dual = extract_common.extract_n_engines_from_psms_dual(
        engines_dual, ["pfind", "diann"], positive_marker="HUMAN")

    def _key(p):
        return (p._sequence, p._charge, p._raw_title, p._label_type)

    assert sorted(_key(p) for p in out_single) == sorted(
        _key(p) for p in out_dual)


def test_extract_dual_expanded_loose_adds_negatives_only():
    """When loose pool > tight pool, additional negatives appear; positives
    invariant."""
    from tools import extract_common

    p_human_tight = _make_psm("PEPTIDEK", 2, "run1", "sp|P00000|HUMAN")
    p_ecoli_tight = _make_psm("AAAAAAAR", 2, "run1", "sp|Q00000|ECOLI")
    p_ecoli_loose1 = _make_psm("EXTRAONE", 2, "run1", "sp|Q11111|ECOLI")
    p_ecoli_loose2 = _make_psm("EXTRATWO", 2, "run1", "sp|Q22222|ECOLI")

    engines_dual = {
        "pfind": {
            "tight": [p_human_tight, p_ecoli_tight],
            "loose": [p_human_tight, p_ecoli_tight, p_ecoli_loose1,
                      p_ecoli_loose2],
        },
        "diann": {
            "tight": [p_human_tight, p_ecoli_tight],
            "loose": [p_human_tight, p_ecoli_tight, p_ecoli_loose1],
        },
    }

    out = extract_common.extract_n_engines_from_psms_dual(
        engines_dual, ["pfind", "diann"], positive_marker="HUMAN")

    positives = [p for p in out if p._label_type == "positive"]
    negatives = [p for p in out if p._label_type == "negative"]

    assert len(positives) == 1
    assert positives[0]._sequence == "PEPTIDEK"

    neg_seqs = sorted(p._sequence for p in negatives)
    assert neg_seqs == ["AAAAAAAR", "EXTRAONE", "EXTRATWO"]


def test_extract_dual_positives_invariant_when_only_loose_changes():
    """Varying loose pool size must NEVER change positive count or sequences."""
    from tools import extract_common

    p_human = _make_psm("PEPTIDEK", 2, "run1", "sp|P00000|HUMAN")
    p_ecoli_tight = _make_psm("AAAAAAAR", 2, "run1", "sp|Q00000|ECOLI")
    p_extra1 = _make_psm("EXTRA1K", 2, "run1", "sp|Q11111|ECOLI")
    p_extra2 = _make_psm("EXTRA2K", 2, "run1", "sp|Q22222|ECOLI")

    tight_only = {
        "pfind": {"tight": [p_human, p_ecoli_tight],
                  "loose": [p_human, p_ecoli_tight]},
        "diann": {"tight": [p_human, p_ecoli_tight],
                  "loose": [p_human, p_ecoli_tight]},
    }
    plus_loose = {
        "pfind": {"tight": [p_human, p_ecoli_tight],
                  "loose": [p_human, p_ecoli_tight, p_extra1, p_extra2]},
        "diann": {"tight": [p_human, p_ecoli_tight],
                  "loose": [p_human, p_ecoli_tight, p_extra1]},
    }

    out_a = extract_common.extract_n_engines_from_psms_dual(
        tight_only, ["pfind", "diann"], positive_marker="HUMAN")
    out_b = extract_common.extract_n_engines_from_psms_dual(
        plus_loose, ["pfind", "diann"], positive_marker="HUMAN")

    pos_a = sorted(p._sequence for p in out_a if p._label_type == "positive")
    pos_b = sorted(p._sequence for p in out_b if p._label_type == "positive")
    assert pos_a == pos_b == ["PEPTIDEK"], (
        f"Positives must be invariant when only loose pool changes; "
        f"got {pos_a} vs {pos_b}")


def test_extract_n_engines_uses_dual_loader(monkeypatch):
    """extract_n_engines(config) calls load_engine_psms_dual and
    extract_n_engines_from_psms_dual under the hood."""
    from tools import extract_common

    captured = {"loader": None, "extractor": None}

    def fake_loader(engine_name, config):
        captured["loader"] = engine_name
        return {"tight": [], "loose": []}

    def fake_extractor(engine_psms_dual, engine_order, positive_marker=None):
        captured["extractor"] = {
            "shape": {k: list(v.keys()) for k, v in engine_psms_dual.items()},
            "engine_order": engine_order,
            "positive_marker": positive_marker,
        }
        return []

    monkeypatch.setattr(extract_common, "load_engine_psms_dual", fake_loader)
    monkeypatch.setattr(extract_common, "extract_n_engines_from_psms_dual",
                         fake_extractor)

    cfg = configparser.ConfigParser()
    cfg.read_dict({
        "extract": {"engines": "pfind", "positive_species_marker": "HUMAN"},
        "engine.pfind": {"path": "/x.qry.res", "qvalue_threshold": "0.01"},
    })
    extract_common.extract_n_engines(cfg)

    assert captured["loader"] == "pfind"
    assert captured["extractor"] is not None
    assert captured["extractor"]["shape"] == {"pfind": ["tight", "loose"]}
    assert captured["extractor"]["engine_order"] == ["pfind"]
    assert captured["extractor"]["positive_marker"] == "HUMAN"
