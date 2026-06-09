import math
import numpy as np
from workflows.pred_integrate import (compute_speclib_i2_i3_j2, I2I3J2_KEYS)
from workflows.pred_store import frag_key, frag_pos_for_ion


def _rec(ion_type, ion_num, light_apex, heavy_apex):
    return {"ion_type": ion_type, "ion_num": ion_num,
            "light_apex": light_apex, "heavy_apex": heavy_apex,
            "light_mass": 0.0, "heavy_mass": 0.0}


def test_none_or_empty_returns_nan_schema():
    out = compute_speclib_i2_i3_j2([_rec("y", 1, 1.0, 1.0)], None, 6, 8, 0.0)
    assert set(out) == set(I2I3J2_KEYS)
    assert math.isnan(out["pred_coverage"])


def test_i2_constant_ratio_zero_dispersion():
    seq_len = 12
    recs, pred = [], {}
    for j, li in zip((1, 2, 3), (10.0, 20.0, 30.0)):
        recs.append(_rec("y", j, li, li * 2.0))
        pred[frag_key("y", frag_pos_for_ion("y", j, seq_len), 1)] = li
    out = compute_speclib_i2_i3_j2(recs, pred, 6, seq_len, 0.0)
    assert out["pred_hl_ratio_cv"] < 1e-9
    assert out["pred_hl_ratio_mad"] < 1e-9


def test_i2_outlier_raises_dispersion():
    seq_len = 12
    recs, pred = [], {}
    for j, li, hi in zip((1, 2, 3), (10.0, 20.0, 30.0), (20.0, 40.0, 6.0)):
        recs.append(_rec("y", j, li, hi))
        pred[frag_key("y", frag_pos_for_ion("y", j, seq_len), 1)] = li
    out = compute_speclib_i2_i3_j2(recs, pred, 6, seq_len, 0.0)
    assert out["pred_hl_ratio_cv"] > 0.1


def test_i3_coverage_counts_present_heavy():
    seq_len = 12
    recs, pred = [], {}
    for j, hi in zip((1, 2, 3, 4), (5.0, 0.0, 7.0, 0.0)):
        recs.append(_rec("y", j, 1.0, hi))
        pred[frag_key("y", frag_pos_for_ion("y", j, seq_len), 1)] = 1.0
    out = compute_speclib_i2_i3_j2(recs, pred, 6, seq_len, 0.0)
    assert abs(out["pred_coverage"] - 0.5) < 1e-9


def test_j2_unexpected_heavy_on_unpredicted_fragment():
    seq_len = 12
    recs = [_rec("y", 1, 1.0, 1.0), _rec("y", 2, 1.0, 1.0),
            _rec("y", 3, 1.0, 9.0)]
    pred = {frag_key("y", frag_pos_for_ion("y", 1, seq_len), 1): 1.0,
            frag_key("y", frag_pos_for_ion("y", 2, seq_len), 1): 1.0}
    out = compute_speclib_i2_i3_j2(recs, pred, 6, seq_len, 0.0)
    assert abs(out["unexpected_heavy_fraction"] - 1.0) < 1e-9
    assert out["unexpected_heavy_intensity_ratio"] > 0.0


def test_j2_nan_when_no_unpredicted_fragments():
    seq_len = 12
    recs, pred = [], {}
    for j in (1, 2):
        recs.append(_rec("y", j, 1.0, 1.0))
        pred[frag_key("y", frag_pos_for_ion("y", j, seq_len), 1)] = 1.0
    out = compute_speclib_i2_i3_j2(recs, pred, 6, seq_len, 0.0)
    assert math.isnan(out["unexpected_heavy_fraction"])
