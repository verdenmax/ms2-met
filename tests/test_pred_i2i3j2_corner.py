"""Corner-case + core-path hardening for compute_speclib_i2_i3_j2.

These tests derive the EXPECTED value independently from the spec
(docs/specs/2026-06-08-speclib-predicted-intensity-features-design.md v1.2
4.4 I2 / 4.5 I3 / 4.6 J2) and assert it. Happy-path coverage lives in
tests/test_pred_integrate_i2i3j2.py and is intentionally not touched here.
"""
import math

import numpy as np
import pytest

from workflows.pred_integrate import compute_speclib_i2_i3_j2, I2I3J2_KEYS
from workflows.pred_store import frag_key, frag_pos_for_ion

SEQ_LEN = 12


def _rec(ion_type, ion_num, light_apex, heavy_apex):
    return {"ion_type": ion_type, "ion_num": ion_num,
            "light_apex": light_apex, "heavy_apex": heavy_apex,
            "light_mass": 0.0, "heavy_mass": 0.0}


def _key(ion_type, ion_num, seq_len=SEQ_LEN):
    return frag_key(ion_type, frag_pos_for_ion(ion_type, ion_num, seq_len), 1)


# --------------------------------------------------------------------------
# Empty / None / {} -> all-NaN fixed schema
# --------------------------------------------------------------------------
def test_empty_records_returns_nan_schema():
    out = compute_speclib_i2_i3_j2([], {_key("y", 1): 1.0}, 6, SEQ_LEN, 0.0)
    assert set(out) == set(I2I3J2_KEYS)
    assert all(math.isnan(out[k]) for k in I2I3J2_KEYS)


def test_pred_none_returns_nan_schema():
    out = compute_speclib_i2_i3_j2([_rec("y", 1, 1.0, 1.0)], None, 6,
                                   SEQ_LEN, 0.0)
    assert set(out) == set(I2I3J2_KEYS)
    assert all(math.isnan(out[k]) for k in I2I3J2_KEYS)


def test_pred_empty_dict_returns_nan_schema():
    out = compute_speclib_i2_i3_j2([_rec("y", 1, 1.0, 1.0)], {}, 6,
                                   SEQ_LEN, 0.0)
    assert set(out) == set(I2I3J2_KEYS)
    assert all(math.isnan(out[k]) for k in I2I3J2_KEYS)


# --------------------------------------------------------------------------
# presence_floor boundary: strict '>' (heavy == floor is NOT present)
# --------------------------------------------------------------------------
def test_presence_floor_is_strict_for_coverage():
    floor = 100.0
    # heavy: 100 (== floor -> absent), 101 (> floor -> present), 200 (present)
    recs, pred = [], {}
    for j, hv in zip((1, 2, 3), (100.0, 101.0, 200.0)):
        recs.append(_rec("y", j, 1.0, hv))
        pred[_key("y", j)] = 1.0
    out = compute_speclib_i2_i3_j2(recs, pred, 6, SEQ_LEN, floor)
    # 2 of 3 present
    assert out["pred_coverage"] == pytest.approx(2.0 / 3.0)


def test_presence_floor_is_strict_for_unexpected_fraction():
    floor = 100.0
    # one predicted present F fragment (keeps cands non-empty)
    recs = [_rec("y", 1, 1.0, 200.0)]
    pred = {_key("y", 1): 5.0}
    # two W (unpredicted) fragments: heavy == floor (absent), heavy > floor
    recs.append(_rec("y", 2, 1.0, 100.0))   # == floor -> absent
    recs.append(_rec("y", 3, 1.0, 101.0))   # > floor  -> present
    out = compute_speclib_i2_i3_j2(recs, pred, 6, SEQ_LEN, floor)
    assert out["unexpected_heavy_fraction"] == pytest.approx(0.5)


# --------------------------------------------------------------------------
# I2 needs >= 2 valid ratios; a single valid ratio -> cv & mad NaN
# --------------------------------------------------------------------------
def test_i2_single_valid_ratio_is_nan():
    recs = [_rec("y", 1, 10.0, 20.0),    # valid (both apex > 0)
            _rec("y", 2, 0.0, 5.0)]      # invalid (light == 0)
    pred = {_key("y", 1): 1.0, _key("y", 2): 1.0}
    out = compute_speclib_i2_i3_j2(recs, pred, 6, SEQ_LEN, 0.0)
    assert math.isnan(out["pred_hl_ratio_cv"])
    assert math.isnan(out["pred_hl_ratio_mad"])


# --------------------------------------------------------------------------
# I2 fragment with a zero apex is excluded from the ratio set
# --------------------------------------------------------------------------
def test_i2_zero_apex_fragment_excluded_from_ratios():
    # Two clean equal ratios + one zero-light fragment (would div-by-zero if
    # wrongly included). Result must be a finite ~0 dispersion.
    recs = [_rec("y", 1, 10.0, 20.0),
            _rec("y", 2, 30.0, 60.0),
            _rec("y", 3, 0.0, 1000.0)]   # excluded from ratio set
    pred = {_key("y", 1): 1.0, _key("y", 2): 1.0, _key("y", 3): 1.0}
    out = compute_speclib_i2_i3_j2(recs, pred, 6, SEQ_LEN, 0.0)
    assert math.isfinite(out["pred_hl_ratio_cv"])
    assert out["pred_hl_ratio_cv"] == pytest.approx(0.0, abs=1e-9)
    assert out["pred_hl_ratio_mad"] == pytest.approx(0.0, abs=1e-9)


# --------------------------------------------------------------------------
# I2 predicted-intensity weighting: a high-pred outlier shifts cv MORE than
# the same outlier at low pred.
# --------------------------------------------------------------------------
def _i2_cv_with_outlier_weight(outlier_high):
    # ratios: two at H/L = 1 (log10 = 0), one outlier at H/L = 10 (log10 = 1)
    recs, pred = [], {}
    rows = [("y", 1, 10.0, 10.0),     # normal
            ("y", 2, 10.0, 10.0),     # normal
            ("y", 3, 10.0, 100.0)]    # outlier (log10 ratio = 1)
    if outlier_high:
        weights = (1.0, 1.0, 8.0)
    else:
        weights = (8.0, 8.0, 1.0)
    for (it, num, lo, hv), w in zip(rows, weights):
        recs.append(_rec(it, num, lo, hv))
        pred[_key(it, num)] = w
    out = compute_speclib_i2_i3_j2(recs, pred, 6, SEQ_LEN, 0.0)
    return out["pred_hl_ratio_cv"]


def test_i2_weighting_high_pred_outlier_shifts_cv_more():
    cv_high = _i2_cv_with_outlier_weight(outlier_high=True)
    cv_low = _i2_cv_with_outlier_weight(outlier_high=False)
    assert cv_high == pytest.approx(0.4)
    assert cv_low == pytest.approx(0.23529411764705882)
    assert cv_high > cv_low


# --------------------------------------------------------------------------
# I3 pred_coverage_wpred differs from pred_coverage when present fragments
# carry disproportionate predicted weight.
# --------------------------------------------------------------------------
def test_i3_wpred_differs_from_plain_coverage():
    floor = 0.0
    recs, pred = [], {}
    # two present, high pred; one absent (heavy == 0), low pred
    rows = [("y", 1, 1.0, 10.0, 10.0),   # present, pred 10
            ("y", 2, 1.0, 10.0, 10.0),   # present, pred 10
            ("y", 3, 1.0, 0.0, 1.0)]     # absent (heavy==0), pred 1
    for it, num, lo, hv, w in rows:
        recs.append(_rec(it, num, lo, hv))
        pred[_key(it, num)] = w
    out = compute_speclib_i2_i3_j2(recs, pred, 6, SEQ_LEN, floor)
    assert out["pred_coverage"] == pytest.approx(2.0 / 3.0)
    # weighted: (10+10) / (10+10+1)
    assert out["pred_coverage_wpred"] == pytest.approx(20.0 / 21.0)
    assert out["pred_coverage_wpred"] != pytest.approx(out["pred_coverage"])


# --------------------------------------------------------------------------
# I3 all-absent -> coverage 0 and wpred 0 (not NaN; F non-empty)
# --------------------------------------------------------------------------
def test_i3_all_absent_coverage_zero():
    floor = 100.0
    recs, pred = [], {}
    for j in (1, 2, 3):
        recs.append(_rec("y", j, 1.0, 50.0))   # 50 <= floor -> absent
        pred[_key("y", j)] = float(j)           # nonzero preds -> sum_pred > 0
    out = compute_speclib_i2_i3_j2(recs, pred, 6, SEQ_LEN, floor)
    assert out["pred_coverage"] == pytest.approx(0.0)
    assert out["pred_coverage_wpred"] == pytest.approx(0.0)


# --------------------------------------------------------------------------
# J2 all-W-absent -> fraction 0 and intensity ratio 0
# --------------------------------------------------------------------------
def test_j2_all_w_absent_zero():
    floor = 100.0
    # F: one predicted present fragment
    recs = [_rec("y", 1, 1.0, 200.0)]
    pred = {_key("y", 1): 5.0}
    # W: two unpredicted, both absent (heavy <= floor)
    recs.append(_rec("y", 2, 1.0, 50.0))
    recs.append(_rec("y", 3, 1.0, 100.0))   # == floor -> absent
    out = compute_speclib_i2_i3_j2(recs, pred, 6, SEQ_LEN, floor)
    assert out["unexpected_heavy_fraction"] == pytest.approx(0.0)
    assert out["unexpected_heavy_intensity_ratio"] == pytest.approx(0.0)


# --------------------------------------------------------------------------
# J2 intensity ratio denominator = sum heavy over ALL F (not over W, and
# including absent-but-nonzero F fragments).
# --------------------------------------------------------------------------
def test_j2_intensity_ratio_divides_by_sum_heavy_over_F():
    floor = 100.0
    recs, pred = [], {}
    # F: present heavy 200, absent-but-nonzero heavy 50 -> sum_heavy_F = 250
    recs.append(_rec("y", 1, 1.0, 200.0)); pred[_key("y", 1)] = 10.0
    recs.append(_rec("y", 2, 1.0, 50.0)); pred[_key("y", 2)] = 8.0
    # W: one unpredicted present fragment, heavy 150
    recs.append(_rec("y", 3, 1.0, 150.0))
    out = compute_speclib_i2_i3_j2(recs, pred, 6, SEQ_LEN, floor)
    assert out["unexpected_heavy_fraction"] == pytest.approx(1.0)
    # 150 / 250 (NOT 150/150=1.0 which would be dividing by sum over W)
    assert out["unexpected_heavy_intensity_ratio"] == pytest.approx(0.6)


# --------------------------------------------------------------------------
# top_k cutoff: a predicted fragment beyond top_k is a cand (predicted), so
# it must NOT count as W even though it's dropped from F.
# --------------------------------------------------------------------------
def test_topk_dropped_predicted_fragment_is_not_W():
    floor = 0.0
    top_k = 2
    recs, pred = [], {}
    # three predicted fragments; pred 10, 9, 1 -> F = top 2 (10, 9)
    recs.append(_rec("y", 1, 1.0, 1.0)); pred[_key("y", 1)] = 10.0
    recs.append(_rec("y", 2, 1.0, 1.0)); pred[_key("y", 2)] = 9.0
    # low-pred predicted fragment dropped from F, with a large present heavy.
    recs.append(_rec("y", 3, 1.0, 9999.0)); pred[_key("y", 3)] = 1.0
    out = compute_speclib_i2_i3_j2(recs, pred, top_k, SEQ_LEN, floor)
    # No real W exists -> fraction must be NaN, NOT inflated by the dropped frag
    assert math.isnan(out["unexpected_heavy_fraction"])
    assert math.isnan(out["unexpected_heavy_intensity_ratio"])


# --------------------------------------------------------------------------
# Mixed b/y ions are pooled in F for I2/I3 (PSM-level, not per-ion).
# --------------------------------------------------------------------------
def test_mixed_by_ions_pooled_in_F():
    floor = 0.0
    recs, pred = [], {}
    # 2 b + 2 y, all H/L = 2 (constant ratio), two present + two absent
    rows = [("b", 1, 10.0, 20.0),    # present
            ("b", 2, 10.0, 20.0),    # present
            ("y", 1, 10.0, 0.0),     # absent (heavy 0)
            ("y", 2, 10.0, 0.0)]     # absent (heavy 0)
    for it, num, lo, hv in rows:
        recs.append(_rec(it, num, lo, hv))
        pred[_key(it, num)] = 1.0
    out = compute_speclib_i2_i3_j2(recs, pred, 6, SEQ_LEN, floor)
    # coverage pools b and y: 2 present of 4
    assert out["pred_coverage"] == pytest.approx(0.5)
    # the two present b ions share H/L = 2 -> dispersion ~0, finite
    assert out["pred_hl_ratio_cv"] == pytest.approx(0.0, abs=1e-9)


# --------------------------------------------------------------------------
# Non-finite predicted intensity (NaN) -> fragment treated as unpredicted (W).
# --------------------------------------------------------------------------
def test_nan_predicted_intensity_routes_fragment_to_W():
    floor = 0.0
    recs, pred = [], {}
    # one clean predicted present F fragment
    recs.append(_rec("y", 1, 1.0, 100.0)); pred[_key("y", 1)] = 5.0
    # one fragment whose prediction is NaN -> should be W (unpredicted)
    recs.append(_rec("y", 2, 1.0, 50.0)); pred[_key("y", 2)] = float("nan")
    out = compute_speclib_i2_i3_j2(recs, pred, 6, SEQ_LEN, floor)
    # the NaN-pred fragment is the only W member and it is present
    assert out["unexpected_heavy_fraction"] == pytest.approx(1.0)
    # ratio = 50 / 100 (heavy_F = 100 from the single F fragment)
    assert out["unexpected_heavy_intensity_ratio"] == pytest.approx(0.5)


# --------------------------------------------------------------------------
# n_both_present / pred_both_present_fraction: top-K fragments where BOTH
# light AND heavy carry signal (strict '>' floor on each channel).
# --------------------------------------------------------------------------
def test_both_present_floor_strict_on_each_channel():
    floor = 100.0
    # j=1: light == floor (absent), heavy present -> not both
    # j=2: light present, heavy == floor (absent) -> not both
    # j=3: both > floor -> both present
    recs, pred = [], {}
    for j, lv, hv in zip((1, 2, 3), (100.0, 200.0, 200.0),
                         (200.0, 100.0, 300.0)):
        recs.append(_rec("y", j, lv, hv))
        pred[_key("y", j)] = 1.0
    out = compute_speclib_i2_i3_j2(recs, pred, 6, SEQ_LEN, floor)
    assert out["n_both_present"] == 1
    assert out["pred_both_present_fraction"] == pytest.approx(1.0 / 3.0)


def test_both_present_respects_top_k_cutoff():
    floor = 100.0
    # 3 predicted, all both-present, but top_k=2 keeps only the 2 strongest;
    # the excluded weak fragment must NOT be counted.
    recs, pred = [], {}
    for j, pv in zip((1, 2, 3), (9.0, 8.0, 1.0)):
        recs.append(_rec("y", j, 500.0, 500.0))
        pred[_key("y", j)] = pv
    out = compute_speclib_i2_i3_j2(recs, pred, 2, SEQ_LEN, floor)
    assert out["n_both_present"] == 2
    assert out["pred_both_present_fraction"] == pytest.approx(1.0)


def test_both_present_nan_when_no_library_coverage():
    # pred is empty -> no F -> distinguishes "unknown" (NaN) from "0 present".
    out = compute_speclib_i2_i3_j2([_rec("y", 1, 500.0, 500.0)], {}, 6,
                                   SEQ_LEN, 100.0)
    assert math.isnan(out["pred_both_present_fraction"])
    assert math.isnan(out["n_both_present"])
