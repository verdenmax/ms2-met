"""Corner-case + core-path hardening tests for J5 adaptive coverage.

Module under test: workflows.pred_integrate.compute_speclib_adaptive
Spec: docs/specs/2026-06-08-speclib-predicted-intensity-features-design.md
      v1.2 4.7 (corrected formula).

Semantics asserted here (derived from the spec):
  * cands = separable records with a FINITE prediction.
  * F     = top-`top_k` of cands ordered by pred descending.
  * global_lh_ratio = median over F of heavy_apex/light_apex for fragments
    with BOTH apexes > 0 (NaN if none).
  * pred_coverage_adaptive = fraction of F fragments with light_apex > 0
    whose heavy_apex >= alpha * light_apex * global_lh_ratio
    (NaN if no F fragment has light_apex > 0, or glh is NaN).

These tests intentionally probe boundaries so real bugs surface; they do not
modify the module or the existing happy-path tests.
"""
import math

import pytest

from workflows.pred_integrate import compute_speclib_adaptive, ADAPTIVE_KEYS
from workflows.pred_store import frag_key, frag_pos_for_ion

SEQ_LEN = 12


def _rec(ion_type, ion_num, light_apex, heavy_apex):
    return {"ion_type": ion_type, "ion_num": ion_num,
            "light_apex": light_apex, "heavy_apex": heavy_apex,
            "light_mass": 0.0, "heavy_mass": 0.0}


def _build(rows, seq_len=SEQ_LEN):
    """rows: iterable of (ion_type, ion_num, light, heavy, pred).

    pred may be None (key absent from pred dict) or a float (incl. NaN).
    Returns (frag_records, pred_frags).
    """
    recs, pred = [], {}
    for ion_type, ion_num, light, heavy, p in rows:
        recs.append(_rec(ion_type, ion_num, light, heavy))
        if p is not None:
            key = frag_key(ion_type, frag_pos_for_ion(ion_type, ion_num, seq_len), 1)
            pred[key] = p
    return recs, pred


# --------------------------------------------------------------------------- #
# Empty / missing-input -> all-NaN schema
# --------------------------------------------------------------------------- #
def test_empty_records_returns_nan_schema():
    out = compute_speclib_adaptive([], {("y", 10, 1): 1.0}, 6, SEQ_LEN, 0.2)
    assert set(out) == set(ADAPTIVE_KEYS)
    assert math.isnan(out["global_lh_ratio"])
    assert math.isnan(out["pred_coverage_adaptive"])


def test_pred_frags_none_returns_nan_schema():
    recs, _ = _build([("y", 1, 10.0, 20.0, 10.0)])
    out = compute_speclib_adaptive(recs, None, 6, SEQ_LEN, 0.2)
    assert set(out) == set(ADAPTIVE_KEYS)
    assert math.isnan(out["global_lh_ratio"])
    assert math.isnan(out["pred_coverage_adaptive"])


def test_pred_frags_empty_dict_returns_nan_schema():
    recs, _ = _build([("y", 1, 10.0, 20.0, 10.0)])
    out = compute_speclib_adaptive(recs, {}, 6, SEQ_LEN, 0.2)
    assert math.isnan(out["global_lh_ratio"])
    assert math.isnan(out["pred_coverage_adaptive"])


def test_no_record_matches_a_prediction_returns_nan():
    # records exist, pred dict non-empty, but keys don't line up -> no cands.
    recs, _ = _build([("y", 1, 10.0, 20.0, None),
                      ("y", 2, 10.0, 20.0, None)])
    pred = {("b", 99, 1): 5.0}
    out = compute_speclib_adaptive(recs, pred, 6, SEQ_LEN, 0.2)
    assert math.isnan(out["global_lh_ratio"])
    assert math.isnan(out["pred_coverage_adaptive"])


# --------------------------------------------------------------------------- #
# alpha boundary: strict >= is inclusive
# --------------------------------------------------------------------------- #
def test_alpha_boundary_exact_equality_is_present():
    # Two anchors fix glh = median([2.0, 2.0, 0.4]) = 2.0.
    # Boundary fragment: heavy = alpha*light*glh = 0.2*10*2.0 = 4.0 exactly.
    rows = [
        ("y", 1, 10.0, 20.0, 30.0),   # ratio 2.0, present (20 >= 4)
        ("y", 2, 10.0, 20.0, 20.0),   # ratio 2.0, present (20 >= 4)
        ("y", 3, 10.0, 4.0, 10.0),    # ratio 0.4; heavy == threshold 4.0
    ]
    recs, pred = _build(rows)
    out = compute_speclib_adaptive(recs, pred, 6, SEQ_LEN, 0.2)
    assert out["global_lh_ratio"] == pytest.approx(2.0)
    # All three valid (light>0); boundary one is present via inclusive >=.
    assert out["pred_coverage_adaptive"] == pytest.approx(1.0)


def test_alpha_boundary_just_below_is_absent():
    rows = [
        ("y", 1, 10.0, 20.0, 30.0),
        ("y", 2, 10.0, 20.0, 20.0),
        ("y", 3, 10.0, 3.9999999, 10.0),  # just below threshold 4.0 -> absent
    ]
    recs, pred = _build(rows)
    out = compute_speclib_adaptive(recs, pred, 6, SEQ_LEN, 0.2)
    assert out["global_lh_ratio"] == pytest.approx(2.0)
    assert out["pred_coverage_adaptive"] == pytest.approx(2.0 / 3.0)


# --------------------------------------------------------------------------- #
# alpha = 0 -> threshold is 0, every light>0 fragment counts (heavy>=0 always)
# --------------------------------------------------------------------------- #
def test_alpha_zero_all_light_positive_present():
    rows = [
        ("y", 1, 10.0, 20.0, 30.0),   # ratio 2.0 -> fixes glh
        ("y", 2, 10.0, 0.0, 20.0),    # heavy 0 but light>0; threshold 0 -> present
        ("y", 3, 5.0, 1.0, 10.0),     # present
    ]
    recs, pred = _build(rows)
    out = compute_speclib_adaptive(recs, pred, 6, SEQ_LEN, 0.0)
    # ratios = [20/10=2.0, 1/5=0.2] (heavy=0 frag excluded) -> median 1.1.
    assert out["global_lh_ratio"] == pytest.approx(1.1)
    # alpha=0 -> threshold 0 -> every light>0 fragment present (heavy>=0 always).
    assert out["pred_coverage_adaptive"] == pytest.approx(1.0)


# --------------------------------------------------------------------------- #
# heavy=0, light>0 with alpha>0: absent AND excluded from ratio set
# --------------------------------------------------------------------------- #
def test_heavy_zero_light_positive_pulls_coverage_down():
    rows = [
        ("y", 1, 10.0, 20.0, 30.0),   # ratio 2.0
        ("y", 2, 10.0, 20.0, 20.0),   # ratio 2.0
        ("y", 3, 10.0, 0.0, 10.0),    # heavy 0: excluded from ratios, absent
    ]
    recs, pred = _build(rows)
    out = compute_speclib_adaptive(recs, pred, 6, SEQ_LEN, 0.2)
    # ratios = [2.0, 2.0] (heavy=0 one excluded) -> median 2.0.
    assert out["global_lh_ratio"] == pytest.approx(2.0)
    # valid = all 3 (light>0); present = 2 -> coverage 2/3.
    assert out["pred_coverage_adaptive"] == pytest.approx(2.0 / 3.0)


# --------------------------------------------------------------------------- #
# global_lh_ratio is the MEDIAN, robust to one extreme ratio
# --------------------------------------------------------------------------- #
def test_global_lh_ratio_is_median_not_mean():
    # ratios = [2.0, 2.0, 100.0] -> median 2.0 (mean would be 34.67).
    rows = [
        ("y", 1, 10.0, 20.0, 30.0),    # ratio 2.0
        ("y", 2, 10.0, 20.0, 20.0),    # ratio 2.0
        ("y", 3, 1.0, 100.0, 10.0),    # ratio 100.0 (extreme)
    ]
    recs, pred = _build(rows)
    out = compute_speclib_adaptive(recs, pred, 6, SEQ_LEN, 0.2)
    assert out["global_lh_ratio"] == pytest.approx(2.0)


def test_global_lh_ratio_even_count_is_mean_of_middle_two():
    # ratios = [1.0, 2.0, 3.0, 4.0] -> median = (2.0+3.0)/2 = 2.5.
    rows = [
        ("y", 1, 10.0, 10.0, 40.0),   # 1.0
        ("y", 2, 10.0, 20.0, 30.0),   # 2.0
        ("y", 3, 10.0, 30.0, 20.0),   # 3.0
        ("y", 4, 10.0, 40.0, 10.0),   # 4.0
    ]
    recs, pred = _build(rows)
    out = compute_speclib_adaptive(recs, pred, 6, SEQ_LEN, 0.2)
    assert out["global_lh_ratio"] == pytest.approx(2.5)


# --------------------------------------------------------------------------- #
# top_k truncation: dropped cands must NOT influence glh or coverage
# --------------------------------------------------------------------------- #
def test_top_k_drops_do_not_affect_glh_or_coverage():
    # top_k = 2. Highest-pred two (preds 100, 90) form F with ratio 2.0 each.
    # The two dropped (preds 10, 5) carry extreme ratios and an under-expected
    # heavy; if (incorrectly) included they'd shift median and coverage.
    rows = [
        ("y", 1, 10.0, 20.0, 100.0),  # F: ratio 2.0, present
        ("y", 2, 10.0, 20.0, 90.0),   # F: ratio 2.0, present
        ("y", 3, 1.0, 100.0, 10.0),   # dropped: ratio 100.0
        ("y", 4, 10.0, 0.0, 5.0),     # dropped: under-expected
    ]
    recs, pred = _build(rows)
    out = compute_speclib_adaptive(recs, pred, 2, SEQ_LEN, 0.2)
    assert out["global_lh_ratio"] == pytest.approx(2.0)
    assert out["pred_coverage_adaptive"] == pytest.approx(1.0)


# --------------------------------------------------------------------------- #
# single F fragment with both apexes > 0
# --------------------------------------------------------------------------- #
def test_single_fragment_glh_and_coverage_defined():
    rows = [("b", 1, 10.0, 30.0, 50.0)]  # ratio 3.0
    recs, pred = _build(rows)
    out = compute_speclib_adaptive(recs, pred, 6, SEQ_LEN, 0.2)
    assert out["global_lh_ratio"] == pytest.approx(3.0)
    # threshold = 0.2 * 10 * 3.0 = 6.0; heavy 30 >= 6 -> present.
    assert out["pred_coverage_adaptive"] == pytest.approx(1.0)


def test_single_fragment_below_threshold_coverage_zero():
    # Only fragment defines glh from itself; can it ever fail its own check?
    # threshold = alpha*light*glh = alpha*heavy. With alpha=1.1 > 1,
    # heavy < 1.1*heavy -> absent -> coverage 0 (but glh still that ratio).
    rows = [("b", 1, 10.0, 30.0, 50.0)]  # ratio 3.0
    recs, pred = _build(rows)
    out = compute_speclib_adaptive(recs, pred, 6, SEQ_LEN, 1.1)
    assert out["global_lh_ratio"] == pytest.approx(3.0)
    assert out["pred_coverage_adaptive"] == pytest.approx(0.0)


# --------------------------------------------------------------------------- #
# all F fragments have light=0 -> glh NaN AND coverage NaN
# --------------------------------------------------------------------------- #
def test_all_light_zero_glh_and_coverage_nan():
    rows = [
        ("y", 1, 0.0, 5.0, 30.0),
        ("y", 2, 0.0, 7.0, 20.0),
    ]
    recs, pred = _build(rows)
    out = compute_speclib_adaptive(recs, pred, 6, SEQ_LEN, 0.2)
    assert math.isnan(out["global_lh_ratio"])
    assert math.isnan(out["pred_coverage_adaptive"])


# --------------------------------------------------------------------------- #
# mixed b/y pooled into F
# --------------------------------------------------------------------------- #
def test_mixed_b_and_y_pooled():
    rows = [
        ("b", 1, 10.0, 20.0, 40.0),   # ratio 2.0
        ("y", 1, 10.0, 20.0, 30.0),   # ratio 2.0
        ("b", 2, 10.0, 4.0, 20.0),    # ratio 0.4; heavy 4 == threshold -> present
        ("y", 2, 10.0, 3.0, 10.0),    # ratio 0.3; heavy 3 < threshold 4 -> absent
    ]
    recs, pred = _build(rows)
    out = compute_speclib_adaptive(recs, pred, 6, SEQ_LEN, 0.2)
    # ratios = [2.0, 2.0, 0.4, 0.3] -> median = (0.4+2.0)/2 = 1.2.
    assert out["global_lh_ratio"] == pytest.approx(1.2)
    # threshold per frag = 0.2 * 10 * 1.2 = 2.4.
    # present: heavy >= 2.4 -> 20,20,4,3 -> all four present.
    assert out["pred_coverage_adaptive"] == pytest.approx(1.0)


def test_mixed_b_and_y_some_absent():
    rows = [
        ("b", 1, 10.0, 20.0, 40.0),   # ratio 2.0
        ("y", 1, 10.0, 20.0, 30.0),   # ratio 2.0
        ("b", 2, 10.0, 1.0, 20.0),    # heavy 1 < threshold -> absent
        ("y", 2, 10.0, 2.0, 10.0),    # heavy 2 < threshold -> absent
    ]
    recs, pred = _build(rows)
    out = compute_speclib_adaptive(recs, pred, 6, SEQ_LEN, 0.2)
    # ratios = [2.0, 2.0, 0.1, 0.2] -> median = (0.2+2.0)/2 = 1.1.
    assert out["global_lh_ratio"] == pytest.approx(1.1)
    # threshold = 0.2 * 10 * 1.1 = 2.2; present: 20,20 ; absent: 1,2.
    assert out["pred_coverage_adaptive"] == pytest.approx(0.5)


# --------------------------------------------------------------------------- #
# NaN predicted intensity -> excluded from cands (so never in F)
# --------------------------------------------------------------------------- #
def test_nan_predicted_intensity_excluded_from_cands():
    # The NaN-pred fragment carries an extreme ratio (100). If it were (wrongly)
    # kept as a candidate it would enter F (top_k large) and shift the median.
    rows = [
        ("y", 1, 10.0, 20.0, 30.0),         # ratio 2.0
        ("y", 2, 10.0, 20.0, 20.0),         # ratio 2.0
        ("y", 3, 1.0, 100.0, float("nan")), # NaN pred -> excluded
    ]
    recs, pred = _build(rows)
    out = compute_speclib_adaptive(recs, pred, 6, SEQ_LEN, 0.2)
    # ratios from cands only = [2.0, 2.0] -> median 2.0.
    assert out["global_lh_ratio"] == pytest.approx(2.0)
    assert out["pred_coverage_adaptive"] == pytest.approx(1.0)


def test_inf_predicted_intensity_excluded_from_cands():
    rows = [
        ("y", 1, 10.0, 20.0, 30.0),
        ("y", 2, 10.0, 20.0, 20.0),
        ("y", 3, 1.0, 100.0, float("inf")),  # non-finite -> excluded
    ]
    recs, pred = _build(rows)
    out = compute_speclib_adaptive(recs, pred, 6, SEQ_LEN, 0.2)
    assert out["global_lh_ratio"] == pytest.approx(2.0)
    assert out["pred_coverage_adaptive"] == pytest.approx(1.0)


# --------------------------------------------------------------------------- #
# large alpha -> most fragments absent
# --------------------------------------------------------------------------- #
def test_large_alpha_drives_coverage_to_zero():
    # Uniform ratio glh = 2.0; threshold = 5.0*light*2.0 = 10*light = 5*heavy.
    # heavy < 5*heavy -> every fragment absent -> coverage 0.
    rows = [
        ("y", 1, 10.0, 20.0, 30.0),
        ("y", 2, 10.0, 20.0, 20.0),
        ("y", 3, 10.0, 20.0, 10.0),
    ]
    recs, pred = _build(rows)
    out = compute_speclib_adaptive(recs, pred, 6, SEQ_LEN, 5.0)
    assert out["global_lh_ratio"] == pytest.approx(2.0)
    assert out["pred_coverage_adaptive"] == pytest.approx(0.0)


def test_large_alpha_only_very_strong_heavy_present():
    # One fragment with hugely over-expected heavy survives alpha=5.0.
    rows = [
        ("y", 1, 10.0, 20.0, 30.0),    # ratio 2.0
        ("y", 2, 10.0, 20.0, 20.0),    # ratio 2.0
        ("y", 3, 1.0, 200.0, 10.0),    # ratio 200; heavy 200 >= 5*1*2=10 -> present
    ]
    recs, pred = _build(rows)
    out = compute_speclib_adaptive(recs, pred, 6, SEQ_LEN, 5.0)
    # ratios = [2.0, 2.0, 200.0] -> median 2.0.
    assert out["global_lh_ratio"] == pytest.approx(2.0)
    # thresholds: frag1/2 = 5*10*2 = 100 (heavy 20 < 100 absent);
    # frag3 = 5*1*2 = 10 (heavy 200 >= 10 present). -> 1/3.
    assert out["pred_coverage_adaptive"] == pytest.approx(1.0 / 3.0)
