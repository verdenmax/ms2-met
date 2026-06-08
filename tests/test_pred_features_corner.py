"""Corner-case / core-path hardening for workflows.pred_features.

Asserts behavior derived from reading the code and the design spec
(docs/specs/2026-06-08-speclib-predicted-intensity-features-design.md §4.2/§4.3/§7).
These complement (do not replace) tests/test_pred_features.py.
"""
import math

import numpy as np
import pytest

from workflows.pred_features import (
    spectral_angle, spearman_sim, _weighted_pearson,
    select_topk_separable, i1_pattern_features)


# --------------------------------------------------------------------------
# spectral_angle
# --------------------------------------------------------------------------
def test_spectral_angle_exact_length_two_identical_is_one():
    assert spectral_angle([1.0, 2.0], [1.0, 2.0]) == pytest.approx(1.0)


def test_spectral_angle_negative_entries_clipped_to_zero():
    # Negatives must be clipped to 0 before the dot product, so these two
    # vectors collapse to identical non-negative shapes -> SA == 1.0.
    assert spectral_angle([-1.0, 2.0], [-5.0, 2.0]) == pytest.approx(1.0)
    # A clipped vector equals the explicitly-clipped version.
    assert spectral_angle([-3.0, 4.0], [1.0, 1.0]) == pytest.approx(
        spectral_angle([0.0, 4.0], [1.0, 1.0]))


def test_spectral_angle_one_hot_same_is_one():
    assert spectral_angle([1.0, 0.0, 0.0], [1.0, 0.0, 0.0]) == pytest.approx(1.0)


def test_spectral_angle_one_hot_different_is_zero():
    assert spectral_angle([1.0, 0.0, 0.0], [0.0, 1.0, 0.0]) == pytest.approx(0.0)


def test_spectral_angle_partial_overlap_is_one_third():
    # dot=1, ||a||=||b||=sqrt(2) -> cos=0.5 -> arccos=pi/3
    # SA = 1 - (2/pi)*(pi/3) = 1 - 2/3 = 1/3
    assert spectral_angle([1.0, 1.0, 0.0], [0.0, 1.0, 1.0]) == pytest.approx(1.0 / 3.0)


def test_spectral_angle_scale_invariant_large_magnitude():
    assert spectral_angle([1e12, 2e12, 3e12], [1.0, 2.0, 3.0]) == pytest.approx(1.0)


def test_spectral_angle_nan_input_is_sane():
    # Spec §7: similarities live in [0,1] for SA, with NaN reserved for
    # degenerate input. A vector carrying np.nan should therefore yield
    # NaN (or at worst a value inside [0,1]) -- never a spurious finite
    # number outside the valid range.
    r = spectral_angle([np.nan, 1.0], [1.0, 2.0])
    assert math.isnan(r) or (0.0 <= r <= 1.0), f"SA(nan) returned {r!r}"


def test_spectral_angle_inf_input_is_sane():
    r = spectral_angle([np.inf, 1.0], [1.0, 2.0])
    assert math.isnan(r) or (0.0 <= r <= 1.0), f"SA(inf) returned {r!r}"


# --------------------------------------------------------------------------
# spearman_sim
# --------------------------------------------------------------------------
def test_spearman_sim_tied_ranks_known_value():
    # [1,1,2] vs [1,2,2]: a single concordant tie-block pair -> rho = 0.5
    assert spearman_sim([1.0, 1.0, 2.0], [1.0, 2.0, 2.0]) == pytest.approx(0.5)


def test_spearman_sim_perfectly_anti_monotonic_is_minus_one():
    assert spearman_sim([1.0, 2.0, 3.0, 4.0], [4.0, 3.0, 2.0, 1.0]) == pytest.approx(-1.0)


def test_spearman_sim_length_two_anti_is_minus_one():
    assert spearman_sim([1.0, 2.0], [2.0, 1.0]) == pytest.approx(-1.0)


def test_spearman_sim_near_constant_with_tiny_noise_is_finite():
    # Variance just above the 1e-12 std floor -> a finite, in-range value.
    r = spearman_sim([1.0, 1.0, 1.0, 1.0000001], [1.0, 2.0, 3.0, 4.0])
    assert np.isfinite(r) and -1.0 <= r <= 1.0


# --------------------------------------------------------------------------
# _weighted_pearson
# --------------------------------------------------------------------------
def test_weighted_pearson_negative_weights_clipped():
    # First weight clipped 0 -> only the last two points contribute,
    # which are perfectly correlated -> 1.0.
    assert _weighted_pearson([1.0, 2.0, 3.0], [1.0, 2.0, 3.0], [-1.0, 1.0, 1.0]) == pytest.approx(1.0)


def test_weighted_pearson_all_zero_weights_is_nan():
    assert math.isnan(_weighted_pearson([1.0, 2.0, 3.0], [1.0, 2.0, 3.0], [0.0, 0.0, 0.0]))


def test_weighted_pearson_uniform_equals_ordinary_pearson():
    x = [1.0, 2.0, 3.0, 4.0]
    y = [2.0, 1.0, 4.0, 3.0]
    expected = float(np.corrcoef(x, y)[0, 1])
    assert _weighted_pearson(x, y, [1.0, 1.0, 1.0, 1.0]) == pytest.approx(expected)


def test_weighted_pearson_anti_correlated_is_minus_one():
    assert _weighted_pearson([1.0, 2.0, 3.0], [3.0, 2.0, 1.0], [1.0, 1.0, 1.0]) == pytest.approx(-1.0)


def test_weighted_pearson_degenerate_sizes_nan():
    assert math.isnan(_weighted_pearson([1.0], [1.0], [1.0]))
    assert math.isnan(_weighted_pearson([1.0, 2.0], [1.0, 2.0], [1.0]))


# --------------------------------------------------------------------------
# select_topk_separable
# --------------------------------------------------------------------------
def _frag(fid, pred, sep):
    return {"id": fid, "pred_intensity": pred, "separable": sep}


def test_select_topk_empty_list():
    assert select_topk_separable([], k=3) == []


def test_select_topk_all_non_separable_is_empty():
    frags = [_frag("a", 0.9, False), _frag("b", 0.8, False)]
    assert select_topk_separable(frags, k=3) == []


def test_select_topk_k_zero_is_empty():
    frags = [_frag("a", 0.9, True), _frag("b", 0.8, True)]
    assert select_topk_separable(frags, k=0) == []


def test_select_topk_k_larger_than_available_returns_all_separable():
    frags = [_frag("a", 0.9, True), _frag("b", 0.8, False), _frag("c", 0.5, True)]
    chosen = select_topk_separable(frags, k=10)
    assert [f["id"] for f in chosen] == ["a", "c"]


def test_select_topk_ties_are_deterministic_and_stable():
    # All separable with equal pred_intensity: Python's sort is stable, so
    # original input order is preserved among ties (deterministic).
    frags = [_frag("a", 0.5, True), _frag("b", 0.5, True),
             _frag("c", 0.5, True), _frag("d", 0.5, True)]
    chosen = select_topk_separable(frags, k=2)
    assert [f["id"] for f in chosen] == ["a", "b"]
    # Deterministic across repeated calls.
    assert [f["id"] for f in select_topk_separable(frags, k=2)] == ["a", "b"]


# --------------------------------------------------------------------------
# i1_pattern_features
# --------------------------------------------------------------------------
def test_i1_minimal_length_two_perfect():
    f = i1_pattern_features([1.0, 2.0], [1.0, 2.0], [1.0, 2.0])
    assert f["spec_pattern_SA_heavy"] == pytest.approx(1.0)
    assert f["spec_pattern_spearman_heavy"] == pytest.approx(1.0)
    assert f["spec_pattern_LH_consistency"] == pytest.approx(1.0)


def test_i1_mismatched_lengths_all_nan():
    f = i1_pattern_features([1.0, 2.0, 3.0], [1.0, 2.0], [1.0, 2.0])
    assert math.isnan(f["spec_pattern_SA_heavy"])
    assert math.isnan(f["spec_pattern_spearman_heavy"])
    assert math.isnan(f["spec_pattern_LH_consistency"])


def test_i1_constant_obs_light_only_lh_consistency_nan():
    # obs_light has zero (weighted) variance -> LH consistency NaN, but the
    # SA/Spearman of pred vs obs_heavy remain well defined.
    pred = [1.0, 2.0, 3.0, 4.0]
    obs_heavy = [1.0, 2.0, 3.0, 4.0]
    obs_light = [5.0, 5.0, 5.0, 5.0]
    f = i1_pattern_features(pred, obs_heavy, obs_light)
    assert f["spec_pattern_SA_heavy"] == pytest.approx(1.0)
    assert f["spec_pattern_spearman_heavy"] == pytest.approx(1.0)
    assert math.isnan(f["spec_pattern_LH_consistency"])


def test_select_topk_excludes_nan_pred_intensity():
    # A separable fragment with a non-finite predicted intensity must not be
    # selected over valid ones (NaN would otherwise sort to the front and be
    # chosen). Regression for review note (2026-06-08).
    frags = [
        {"id": "bad", "pred_intensity": float("nan"), "separable": True},
        {"id": "good", "pred_intensity": 0.5, "separable": True},
    ]
    chosen = select_topk_separable(frags, k=1)
    assert [f["id"] for f in chosen] == ["good"]
