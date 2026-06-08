import math
import numpy as np

from workflows.pred_features import (
    spectral_angle, spearman_sim, select_topk_separable, i1_pattern_features)


def test_spectral_angle_identical_vectors_is_one():
    assert abs(spectral_angle([1.0, 2.0, 3.0], [1.0, 2.0, 3.0]) - 1.0) < 1e-9


def test_spectral_angle_orthogonal_is_zero():
    assert abs(spectral_angle([1.0, 0.0], [0.0, 1.0]) - 0.0) < 1e-9


def test_spectral_angle_scaled_vectors_is_one():
    assert abs(spectral_angle([1.0, 2.0, 3.0], [10.0, 20.0, 30.0]) - 1.0) < 1e-9


def test_spectral_angle_degenerate_returns_nan():
    assert math.isnan(spectral_angle([1.0], [1.0]))
    assert math.isnan(spectral_angle([0.0, 0.0], [1.0, 2.0]))
    assert math.isnan(spectral_angle([1.0, 2.0], [1.0, 2.0, 3.0]))


def test_spearman_sim_monotonic_same_order_is_one():
    assert abs(spearman_sim([1.0, 2.0, 3.0, 4.0], [10.0, 20.0, 25.0, 40.0]) - 1.0) < 1e-9


def test_spearman_sim_reversed_is_minus_one():
    assert abs(spearman_sim([1.0, 2.0, 3.0, 4.0], [40.0, 30.0, 20.0, 10.0]) + 1.0) < 1e-9


def test_spearman_sim_constant_or_short_returns_nan():
    assert math.isnan(spearman_sim([1.0, 1.0, 1.0], [1.0, 2.0, 3.0]))
    assert math.isnan(spearman_sim([1.0], [2.0]))


def _frag(fid, pred, sep):
    return {"id": fid, "pred_intensity": pred, "separable": sep}


def test_select_topk_picks_highest_pred_among_separable():
    frags = [_frag("a", 0.9, True), _frag("b", 0.8, False),
             _frag("c", 0.5, True), _frag("d", 0.7, True)]
    chosen = select_topk_separable(frags, k=2)
    assert [f["id"] for f in chosen] == ["a", "d"]


def test_select_topk_returns_all_when_fewer_than_k():
    frags = [_frag("a", 0.9, True), _frag("b", 0.1, False)]
    chosen = select_topk_separable(frags, k=6)
    assert [f["id"] for f in chosen] == ["a"]


def test_i1_perfect_match_high_scores():
    pred = [1.0, 0.8, 0.4, 0.2]
    obs_heavy = [10.0, 8.0, 4.0, 2.0]
    obs_light = [10.0, 8.0, 4.0, 2.0]
    f = i1_pattern_features(pred, obs_heavy, obs_light)
    assert abs(f["spec_pattern_SA_heavy"] - 1.0) < 1e-9
    assert abs(f["spec_pattern_spearman_heavy"] - 1.0) < 1e-9
    assert abs(f["spec_pattern_LH_consistency"] - 1.0) < 1e-9


def test_i1_shuffled_pattern_low_sa():
    pred = [1.0, 0.8, 0.4, 0.2]
    obs_heavy = [0.2, 0.4, 0.8, 1.0]
    obs_light = [0.2, 0.4, 0.8, 1.0]
    f = i1_pattern_features(pred, obs_heavy, obs_light)
    assert f["spec_pattern_SA_heavy"] < 0.7


def test_i1_degenerate_returns_nan():
    f = i1_pattern_features([1.0], [1.0], [1.0])
    assert math.isnan(f["spec_pattern_SA_heavy"])
    assert math.isnan(f["spec_pattern_LH_consistency"])
