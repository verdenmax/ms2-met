import math
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
