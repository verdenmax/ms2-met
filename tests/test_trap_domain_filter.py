"""Tests for the trap domain-of-applicability filter (spec §12).

The policy `beyond_tool_limit` is pure and locked here. The expected values
are derived directly from spec §12.1 (class 1 = L0/L1 homolog, class 3 =
heavy out of window; class 1 takes precedence).
"""
import pandas as pd

from tools.trap_domain_filter import (
    beyond_tool_limit, annotate_traps, has_label_site)


def test_l0_homolog_is_dropped():
    assert beyond_tool_limit("L0", 0) == (True, "homolog_L0")


def test_l1_homolog_is_dropped():
    assert beyond_tool_limit("L1", 0) == (True, "homolog_L1")


def test_heavy_out_of_window_is_dropped():
    assert beyond_tool_limit("L4", 1) == (True, "heavy_out_of_window")


def test_genuine_trap_in_window_is_kept():
    assert beyond_tool_limit("L4", 0) == (False, None)


def test_no_label_site_is_dropped():
    # peptide with no K/R -> heavy == light -> SILAC undefined (class 4)
    assert beyond_tool_limit("L4", 0, has_kr=False) == (True, "no_label_site")


def test_has_kr_defaults_true_keeps_genuine_trap():
    # default has_kr=True preserves the original 2-arg behavior
    assert beyond_tool_limit("L4", 0) == (False, None)


def test_homolog_takes_precedence_over_no_label_site():
    assert beyond_tool_limit("L0", 0, has_kr=False) == (True, "homolog_L0")


def test_no_label_site_takes_precedence_over_out_of_window():
    assert beyond_tool_limit("L4", 1, has_kr=False) == (True, "no_label_site")


def test_has_label_site_detects_kr():
    assert has_label_site("PEPTIDEK") is True       # ends in K
    assert has_label_site("PEPTIDER") is True        # ends in R
    assert has_label_site("ACDEFGHILMNPQSTVWY") is False   # no K/R
    assert has_label_site("LQEFLQHVS") is False       # real pilot example


def test_homolog_takes_precedence_over_out_of_window():
    # both apply -> class 1 (more fundamental indistinguishability) wins
    assert beyond_tool_limit("L0", 1) == (True, "homolog_L0")


def test_heavy_out_of_range_accepts_bool_and_str():
    assert beyond_tool_limit("L4", True)[0] is True
    assert beyond_tool_limit("L4", "1")[0] is True
    assert beyond_tool_limit("L4", "0")[0] is False


class _FakeTarget:
    """Minimal stand-in: classify by membership in a small human 'proteome'."""
    raw_text = "PEPTIDEKHUMANSEQ"
    li_normalized_text = raw_text.replace("I", "L")
    n_proteins = 1


def test_annotate_traps_keeps_positives_and_flags_traps(monkeypatch):
    import tools.trap_domain_filter as mod
    # PEPTIDEK is a substring of the fake target -> L0; YEASTONLYK -> L4

    def fake_classify(seq, target):
        return "L0" if seq in target.raw_text else "L4"

    monkeypatch.setattr(mod, "classify_peptide", fake_classify)
    df = pd.DataFrame([
        {"sequence": "PEPTIDEK", "label_type": "positive", "heavy_out_of_range": 0},
        {"sequence": "PEPTIDEK", "label_type": "negative", "heavy_out_of_range": 0},
        {"sequence": "YEASTONLYK", "label_type": "negative", "heavy_out_of_range": 0},
        {"sequence": "YEASTONLYK", "label_type": "negative", "heavy_out_of_range": 1},
    ])
    out = mod.annotate_traps(df, _FakeTarget())
    # positive untouched
    assert out.loc[0, "entrap_level"] == "target"
    assert not out.loc[0, "domain_drop"]
    # negative homolog dropped
    assert out.loc[1, "domain_drop"] and out.loc[1, "domain_reason"] == "homolog_L0"
    # genuine in-window trap kept
    assert not out.loc[2, "domain_drop"]
    # genuine but out-of-window dropped
    assert out.loc[3, "domain_drop"] and out.loc[3, "domain_reason"] == "heavy_out_of_window"
