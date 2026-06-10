"""TDD for compute_speclib_coelut: fragment-level light<->heavy co-elution
features (spec §13). heavy_coelut = heavy intensity at the LIGHT fragment's
apex cycle (+-1); these features suppress off-peak interference that inflates
the max-in-window pred_coverage / spec_pattern_SA.
"""
import math

import numpy as np
import pytest

from workflows.pred_integrate import compute_speclib_coelut, COELUT_KEYS
from workflows.pred_store import frag_key, frag_pos_for_ion


def _rec(ion_type, ion_num, light_apex, heavy_apex, heavy_coelut):
    return {"ion_type": ion_type, "ion_num": ion_num,
            "light_apex": light_apex, "heavy_apex": heavy_apex,
            "heavy_coelut": heavy_coelut, "light_mass": 0.0, "heavy_mass": 0.0}


def _key(it, num, seq_len):
    return frag_key(it, frag_pos_for_ion(it, num, seq_len), 1)


def test_none_or_empty_returns_nan_schema():
    out = compute_speclib_coelut([_rec("y", 1, 1., 1., 1.)], None, 6, 8, 0.0)
    assert set(out) == set(COELUT_KEYS)
    assert all(math.isnan(out[k]) for k in COELUT_KEYS)


def test_coverage_coelut_counts_only_coeluting():
    seq_len = 12
    recs, pred = [], {}
    # all 4 have heavy_apex > floor (max-in-window present), but only j=1,3
    # have heavy AT the light apex cycle (heavy_coelut > floor)
    for j, ha, hc in ((1, 5000, 5000), (2, 5000, 0), (3, 5000, 4000), (4, 5000, 0)):
        recs.append(_rec("y", j, 1000., ha, hc))
        pred[_key("y", j, seq_len)] = 1.0
    out = compute_speclib_coelut(recs, pred, 6, seq_len, 100.0)
    assert out["pred_coverage_coelut"] == pytest.approx(0.5)        # 2/4 co-elute
    assert out["frag_offtime_fraction"] == pytest.approx(0.5)        # 2/4 off-time


def test_offtime_fraction_nan_when_no_heavy_signal():
    seq_len = 12
    recs, pred = [], {}
    for j in (1, 2, 3):
        recs.append(_rec("y", j, 1000., 0.0, 0.0))   # no heavy at all
        pred[_key("y", j, seq_len)] = 1.0
    out = compute_speclib_coelut(recs, pred, 6, seq_len, 100.0)
    assert out["pred_coverage_coelut"] == pytest.approx(0.0)
    assert math.isnan(out["frag_offtime_fraction"])   # no heavy -> undefined


def test_pure_interference_drops_coverage_to_zero():
    # heavy present everywhere (max-in-window) but always off-peak -> the
    # signature of the NPESDS... interference trap.
    seq_len = 12
    recs, pred = [], {}
    for j in (1, 2, 3, 4):
        recs.append(_rec("y", j, 1000., 8000., 0.0))   # heavy off-peak
        pred[_key("y", j, seq_len)] = 1.0
    out = compute_speclib_coelut(recs, pred, 6, seq_len, 100.0)
    assert out["pred_coverage_coelut"] == pytest.approx(0.0)
    assert out["frag_offtime_fraction"] == pytest.approx(1.0)


def test_sa_coelut_uses_coeluting_heavy_not_max():
    # heavy_apex would give a high SA, but heavy_coelut is anti-correlated
    # with the prediction -> SA_coelut must be low (proves it uses coelut).
    seq_len = 12
    recs, pred = [], {}
    for j, pv, ha, hc in ((1, 1.0, 100, 10), (2, 0.6, 60, 60), (3, 0.3, 30, 100)):
        recs.append(_rec("y", j, 1000., ha, hc))
        pred[_key("y", j, seq_len)] = pv
    out = compute_speclib_coelut(recs, pred, 6, seq_len, 0.0)
    assert np.isfinite(out["spec_pattern_SA_coelut"])
    assert out["spec_pattern_SA_coelut"] < 0.7    # anti-correlated coelut -> low


def test_records_without_heavy_coelut_key_default_zero():
    # backward-compat: records lacking 'heavy_coelut' are treated as 0 (absent)
    seq_len = 12
    rec = {"ion_type": "y", "ion_num": 1, "light_apex": 1000.,
           "heavy_apex": 5000., "light_mass": 0.0, "heavy_mass": 0.0}
    rec2 = {"ion_type": "y", "ion_num": 2, "light_apex": 1000.,
            "heavy_apex": 5000., "light_mass": 0.0, "heavy_mass": 0.0}
    pred = {_key("y", 1, seq_len): 1.0, _key("y", 2, seq_len): 1.0}
    out = compute_speclib_coelut([rec, rec2], pred, 6, seq_len, 100.0)
    assert out["pred_coverage_coelut"] == pytest.approx(0.0)   # no coelut -> 0
