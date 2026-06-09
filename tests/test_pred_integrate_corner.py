"""Corner/core-path hardening for compute_speclib_i1 (Phase 2a, I1).

Each test asserts the CORRECT behavior derived from
workflows/pred_integrate.py + workflows/pred_features.py and the design
spec v1.2 (§4.3 I1, §7 per-ion-type metric). A failing test here is a
SUSPECTED BUG unless the asserted behavior is confirmed acceptable.
"""
import math

import numpy as np
import pytest

from workflows.pred_integrate import compute_speclib_i1, I1_KEYS
from workflows.pred_features import spectral_angle
from workflows.pred_store import frag_key, frag_pos_for_ion


def _rec(ion_type, ion_num, light_apex, heavy_apex):
    return {"ion_type": ion_type, "ion_num": ion_num,
            "light_apex": light_apex, "heavy_apex": heavy_apex,
            "light_mass": 0.0, "heavy_mass": 0.0}


def _pkey(ion_type, ion_num, seq_len):
    return frag_key(ion_type, frag_pos_for_ion(ion_type, ion_num, seq_len), 1)


def _assert_nan_schema(out):
    assert set(out) == set(I1_KEYS)
    assert math.isnan(out["spec_pattern_SA_b"])
    assert math.isnan(out["spec_pattern_SA_y"])
    assert math.isnan(out["spec_pattern_SA"])
    assert math.isnan(out["spec_pattern_LH_consistency"])
    assert out["n_fragments_in_F"] == 0


# --- empty / degenerate inputs ------------------------------------------------

def test_empty_frag_records_returns_nan_schema():
    out = compute_speclib_i1([], {_pkey("b", 1, 8): 1.0}, top_k=6, seq_len=8)
    _assert_nan_schema(out)


def test_empty_pred_frags_dict_returns_nan_schema():
    out = compute_speclib_i1([_rec("b", 1, 5.0, 5.0)], {}, top_k=6, seq_len=8)
    _assert_nan_schema(out)


def test_topk_zero_gives_empty_F_and_nan():
    seq_len = 8
    recs, pred = [], {}
    for i, inten in zip((1, 2), (1.0, 0.5)):
        recs.append(_rec("y", i, inten, inten))
        pred[_pkey("y", i, seq_len)] = inten
    out = compute_speclib_i1(recs, pred, top_k=0, seq_len=seq_len)
    assert out["n_fragments_in_F"] == 0
    assert math.isnan(out["spec_pattern_SA_b"])
    assert math.isnan(out["spec_pattern_SA_y"])
    assert math.isnan(out["spec_pattern_SA"])
    assert math.isnan(out["spec_pattern_LH_consistency"])


# --- heavy-absent / zero-norm semantics --------------------------------------

def test_pred_strong_absent_in_heavy_pulls_sa_down_finite():
    """A predicted-strong fragment missing in heavy (heavy_apex=0) still
    enters F and adds a zero to obs, so SA_y is finite and < 1."""
    seq_len = 8
    recs = [_rec("y", 1, 1.0, 1.0), _rec("y", 2, 0.5, 0.0)]
    pred = {_pkey("y", 1, seq_len): 1.0, _pkey("y", 2, seq_len): 0.5}
    out = compute_speclib_i1(recs, pred, top_k=6, seq_len=seq_len)
    assert out["n_fragments_in_F"] == 2
    assert np.isfinite(out["spec_pattern_SA_y"])
    assert out["spec_pattern_SA_y"] < 1.0
    assert out["spec_pattern_SA_y"] == pytest.approx(
        spectral_angle([1.0, 0.5], [1.0, 0.0]))


def test_all_heavy_zero_in_subset_gives_nan_for_that_type():
    """A subset whose entire obs (heavy_apex) is zero has zero norm -> NaN,
    while the other ion type with non-zero obs stays finite."""
    seq_len = 8
    recs = [
        _rec("y", 1, 1.0, 0.0), _rec("y", 2, 0.5, 0.0),  # all heavy zero
        _rec("b", 1, 1.0, 1.0), _rec("b", 2, 0.5, 0.5),  # finite
    ]
    pred = {
        _pkey("y", 1, seq_len): 1.0, _pkey("y", 2, seq_len): 0.5,
        _pkey("b", 1, seq_len): 1.0, _pkey("b", 2, seq_len): 0.5,
    }
    out = compute_speclib_i1(recs, pred, top_k=6, seq_len=seq_len)
    assert math.isnan(out["spec_pattern_SA_y"])
    assert out["spec_pattern_SA_b"] == pytest.approx(1.0)
    # combined = mean over only the finite per-ion SA -> SA_b
    assert out["spec_pattern_SA"] == pytest.approx(out["spec_pattern_SA_b"])


# --- top-K cutoffs / ion-type combination ------------------------------------

def test_topk_cuts_off_one_ion_type_entirely():
    """top_k keeps only the two high-pred b ions; both y ions drop out, so
    SA_y is NaN and combined SA equals SA_b."""
    seq_len = 8
    recs, pred = [], {}
    for i, inten in zip((1, 2), (0.95, 0.90)):           # strong b ions
        recs.append(_rec("b", i, inten, inten))
        pred[_pkey("b", i, seq_len)] = inten
    for j, inten in zip((1, 2), (0.20, 0.10)):           # weak y ions
        recs.append(_rec("y", j, inten, inten))
        pred[_pkey("y", j, seq_len)] = inten
    out = compute_speclib_i1(recs, pred, top_k=2, seq_len=seq_len)
    assert out["n_fragments_in_F"] == 2
    assert math.isnan(out["spec_pattern_SA_y"])
    assert np.isfinite(out["spec_pattern_SA_b"])
    assert out["spec_pattern_SA"] == pytest.approx(out["spec_pattern_SA_b"])


def test_combined_sa_is_mean_of_finite_per_ion_sas():
    """spec_pattern_SA is the arithmetic mean of the finite per-ion SAs.
    Construct a case where SA_b != SA_y and verify numerically."""
    seq_len = 10
    recs = [
        _rec("b", 1, 1.0, 1.0), _rec("b", 2, 0.5, 0.5),  # perfect -> SA_b=1
        _rec("y", 1, 1.0, 1.0), _rec("y", 2, 0.5, 0.0),  # mismatch -> SA_y<1
    ]
    pred = {
        _pkey("b", 1, seq_len): 1.0, _pkey("b", 2, seq_len): 0.5,
        _pkey("y", 1, seq_len): 1.0, _pkey("y", 2, seq_len): 0.5,
    }
    out = compute_speclib_i1(recs, pred, top_k=6, seq_len=seq_len)
    sa_b, sa_y = out["spec_pattern_SA_b"], out["spec_pattern_SA_y"]
    assert np.isfinite(sa_b) and np.isfinite(sa_y)
    assert sa_b != pytest.approx(sa_y)
    assert out["spec_pattern_SA"] == pytest.approx((sa_b + sa_y) / 2.0)


def test_ties_in_pred_are_deterministic_and_no_crash():
    seq_len = 12
    recs, pred = [], {}
    for i in range(1, 6):                                 # all equal pred
        recs.append(_rec("y", i, 1.0, 1.0))
        pred[_pkey("y", i, seq_len)] = 0.5
    out1 = compute_speclib_i1(recs, pred, top_k=3, seq_len=seq_len)
    out2 = compute_speclib_i1(recs, pred, top_k=3, seq_len=seq_len)
    assert out1["n_fragments_in_F"] == 3
    # deterministic: same keys, NaN-aware value equality
    assert set(out1) == set(out2)
    for k in out1:
        v1, v2 = out1[k], out2[k]
        if isinstance(v1, float) and math.isnan(v1):
            assert isinstance(v2, float) and math.isnan(v2)
        else:
            assert v1 == v2
    # identical equal vectors -> perfect SA, no crash on ties
    assert out1["spec_pattern_SA_y"] == pytest.approx(1.0)


# --- prediction lookup / exclusion semantics ---------------------------------

def test_non_finite_pred_value_excludes_fragment():
    seq_len = 8
    recs = [_rec("y", 1, 1.0, 1.0), _rec("y", 2, 0.5, 0.5), _rec("y", 3, 0.3, 0.3)]
    pred = {
        _pkey("y", 1, seq_len): 1.0,
        _pkey("y", 2, seq_len): float("nan"),   # excluded
        _pkey("y", 3, seq_len): 0.3,
    }
    out = compute_speclib_i1(recs, pred, top_k=6, seq_len=seq_len)
    assert out["n_fragments_in_F"] == 2  # the NaN-pred fragment dropped


def test_frag_record_without_matching_key_excluded():
    seq_len = 8
    recs = [_rec("y", 1, 1.0, 1.0), _rec("y", 2, 0.5, 0.5)]
    pred = {_pkey("y", 1, seq_len): 1.0}     # only y1 present
    out = compute_speclib_i1(recs, pred, top_k=6, seq_len=seq_len)
    assert out["n_fragments_in_F"] == 1
    assert math.isnan(out["spec_pattern_SA_y"])  # only 1 -> NaN


def test_y_reversal_required_for_match():
    """A y record only matches a prediction keyed at seq_len-ion_num-1.
    A b-style key (ion_num-1) for the same y record must NOT match."""
    seq_len = 8
    rec = _rec("y", 1, 1.0, 1.0)
    correct_key = frag_key("y", seq_len - 1 - 1, 1)       # = ("y", 6, 1)
    wrong_key = frag_key("y", 1 - 1, 1)                   # = ("y", 0, 1)
    assert correct_key != wrong_key
    # wrong key -> no match -> NaN schema
    out_wrong = compute_speclib_i1([rec], {wrong_key: 1.0},
                                   top_k=6, seq_len=seq_len)
    assert out_wrong["n_fragments_in_F"] == 0
    # correct key -> matches
    out_ok = compute_speclib_i1([rec], {correct_key: 1.0},
                                top_k=6, seq_len=seq_len)
    assert out_ok["n_fragments_in_F"] == 1
