import math
import numpy as np
import pytest
from workflows.pred_integrate import compute_speclib_i1, I1_KEYS
from workflows.pred_store import frag_key, frag_pos_for_ion


def _rec(ion_type, ion_num, light_apex, heavy_apex):
    return {"ion_type": ion_type, "ion_num": ion_num,
            "light_apex": light_apex, "heavy_apex": heavy_apex,
            "light_mass": 0.0, "heavy_mass": 0.0}


def test_none_pred_frags_returns_nan_schema():
    out = compute_speclib_i1([_rec("b", 1, 5.0, 5.0)], None, top_k=6, seq_len=8)
    assert set(out) == set(I1_KEYS)
    assert math.isnan(out["spec_pattern_SA"])
    assert out["n_fragments_in_F"] == 0


def test_perfect_match_per_ion_type_high():
    seq_len = 8
    recs, pred = [], {}
    for i, inten in zip((1, 2, 3), (1.0, 0.6, 0.3)):
        recs.append(_rec("b", i, inten, inten))
        pred[frag_key("b", frag_pos_for_ion("b", i, seq_len), 1)] = inten
    for j, inten in zip((1, 2, 3), (0.9, 0.5, 0.2)):
        recs.append(_rec("y", j, inten, inten))
        pred[frag_key("y", frag_pos_for_ion("y", j, seq_len), 1)] = inten
    out = compute_speclib_i1(recs, pred, top_k=6, seq_len=seq_len)
    assert abs(out["spec_pattern_SA_b"] - 1.0) < 1e-6
    assert abs(out["spec_pattern_SA_y"] - 1.0) < 1e-6
    assert abs(out["spec_pattern_SA"] - 1.0) < 1e-6
    assert out["n_fragments_in_F"] == 6


def test_topk_limits_fragment_set():
    seq_len = 12
    recs, pred = [], {}
    for i, inten in zip(range(1, 9), (0.9, 0.8, 0.7, 0.6, 0.5, 0.4, 0.3, 0.2)):
        recs.append(_rec("y", i, inten, inten))
        pred[frag_key("y", frag_pos_for_ion("y", i, seq_len), 1)] = inten
    out = compute_speclib_i1(recs, pred, top_k=3, seq_len=seq_len)
    assert out["n_fragments_in_F"] == 3


def test_only_b_gives_nan_for_y():
    seq_len = 8
    recs, pred = [], {}
    for i, inten in zip((1, 2), (1.0, 0.5)):
        recs.append(_rec("b", i, inten, inten))
        pred[frag_key("b", frag_pos_for_ion("b", i, seq_len), 1)] = inten
    out = compute_speclib_i1(recs, pred, top_k=6, seq_len=seq_len)
    assert math.isnan(out["spec_pattern_SA_y"])
    assert abs(out["spec_pattern_SA_b"] - 1.0) < 1e-6
    assert abs(out["spec_pattern_SA"] - out["spec_pattern_SA_b"]) < 1e-9


def test_fragment_without_prediction_is_excluded():
    seq_len = 8
    recs = [_rec("y", 1, 1.0, 1.0), _rec("y", 2, 0.5, 0.5)]
    pred = {frag_key("y", frag_pos_for_ion("y", 1, seq_len), 1): 1.0}
    out = compute_speclib_i1(recs, pred, top_k=6, seq_len=seq_len)
    assert out["n_fragments_in_F"] == 1
    assert math.isnan(out["spec_pattern_SA_y"])


def test_spearman_rank_order_is_scale_invariant():
    # Same RANK order, wildly different magnitudes between predicted and
    # observed-heavy -> Spearman = 1.0 per ion type, while the magnitude-
    # sensitive spectral angle is < 1. This is the whole point of adding a
    # rank metric: it is immune to the b2-dominance / b:y scale inflation.
    seq_len = 10
    recs, pred = [], {}
    for i, pv, hv in zip((1, 2, 3), (1.0, 0.6, 0.3), (1000.0, 20.0, 5.0)):
        recs.append(_rec("b", i, pv, hv))
        pred[frag_key("b", frag_pos_for_ion("b", i, seq_len), 1)] = pv
    for j, pv, hv in zip((1, 2, 3), (0.9, 0.5, 0.2), (800.0, 400.0, 100.0)):
        recs.append(_rec("y", j, pv, hv))
        pred[frag_key("y", frag_pos_for_ion("y", j, seq_len), 1)] = pv
    out = compute_speclib_i1(recs, pred, top_k=6, seq_len=seq_len)
    assert out["spec_pattern_spearman_b"] == pytest.approx(1.0)
    assert out["spec_pattern_spearman_y"] == pytest.approx(1.0)
    assert out["spec_pattern_spearman"] == pytest.approx(1.0)
    # spectral angle is magnitude-sensitive, so SA_b is pulled below 1 here
    assert out["spec_pattern_SA_b"] < 1.0


def test_spearman_needs_three_points_per_ion_type():
    # n=2 Spearman is degenerate (always +-1), so a 2-fragment ion type yields
    # NaN spearman while SA (defined for >=2) still computes.
    seq_len = 8
    recs, pred = [], {}
    for i, pv, hv in zip((1, 2), (1.0, 0.5), (100.0, 50.0)):
        recs.append(_rec("b", i, pv, hv))
        pred[frag_key("b", frag_pos_for_ion("b", i, seq_len), 1)] = pv
    out = compute_speclib_i1(recs, pred, top_k=6, seq_len=seq_len)
    assert math.isnan(out["spec_pattern_spearman_b"])
    assert math.isnan(out["spec_pattern_spearman_y"])
    assert math.isnan(out["spec_pattern_spearman"])
    assert np.isfinite(out["spec_pattern_SA_b"])
