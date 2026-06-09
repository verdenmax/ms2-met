import math
import numpy as np
from workflows.pred_integrate import compute_speclib_adaptive, ADAPTIVE_KEYS
from workflows.pred_store import frag_key, frag_pos_for_ion


def _rec(ion_type, ion_num, light_apex, heavy_apex):
    return {"ion_type": ion_type, "ion_num": ion_num,
            "light_apex": light_apex, "heavy_apex": heavy_apex,
            "light_mass": 0.0, "heavy_mass": 0.0}


def test_none_or_empty_returns_nan_schema():
    out = compute_speclib_adaptive([_rec("y", 1, 1.0, 1.0)], None, 6, 8, 0.2)
    assert set(out) == set(ADAPTIVE_KEYS)
    assert math.isnan(out["global_lh_ratio"])
    assert math.isnan(out["pred_coverage_adaptive"])


def test_constant_ratio_all_present():
    seq_len = 12
    recs, pred = [], {}
    for j, li in zip((1, 2, 3), (10.0, 20.0, 30.0)):
        recs.append(_rec("y", j, li, li * 2.0))
        pred[frag_key("y", frag_pos_for_ion("y", j, seq_len), 1)] = li
    out = compute_speclib_adaptive(recs, pred, 6, seq_len, 0.2)
    assert abs(out["global_lh_ratio"] - 2.0) < 1e-9
    assert abs(out["pred_coverage_adaptive"] - 1.0) < 1e-9


def test_underexpected_fragment_not_present():
    seq_len = 12
    recs, pred = [], {}
    for j, li, hi in zip((1, 2, 3), (10.0, 20.0, 30.0), (20.0, 40.0, 1.0)):
        recs.append(_rec("y", j, li, hi))
        pred[frag_key("y", frag_pos_for_ion("y", j, seq_len), 1)] = li
    out = compute_speclib_adaptive(recs, pred, 6, seq_len, 0.2)
    assert abs(out["global_lh_ratio"] - 2.0) < 1e-9
    assert abs(out["pred_coverage_adaptive"] - (2.0 / 3.0)) < 1e-9


def test_no_valid_ratio_returns_nan():
    seq_len = 12
    recs, pred = [], {}
    for j in (1, 2):
        recs.append(_rec("y", j, 0.0, 5.0))
        pred[frag_key("y", frag_pos_for_ion("y", j, seq_len), 1)] = 1.0
    out = compute_speclib_adaptive(recs, pred, 6, seq_len, 0.2)
    assert math.isnan(out["global_lh_ratio"])
    assert math.isnan(out["pred_coverage_adaptive"])
