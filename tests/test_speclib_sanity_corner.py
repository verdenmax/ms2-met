"""Corner-case / core-path hardening for tools.speclib_sanity.

Asserts behavior derived from reading the code and the design spec
(docs/specs/2026-06-08-speclib-predicted-intensity-features-design.md §4.0).
"""
import math

from workflows.pred_features import spectral_angle, spearman_sim
from workflows.pred_store import frag_key
from tools.speclib_sanity import (
    similarity_distribution, gate_pass, build_pairs_from_maps)


# --------------------------------------------------------------------------
# similarity_distribution
# --------------------------------------------------------------------------
def test_similarity_distribution_empty_pairs_nan_stats():
    stats = similarity_distribution([], metric=spectral_angle)
    assert stats["n"] == 0
    assert math.isnan(stats["median"])
    assert math.isnan(stats["p25"])
    assert math.isnan(stats["p75"])


def test_similarity_distribution_all_nan_pairs_dropped():
    # Every pair is degenerate (length < 2) -> SA NaN -> nothing counted.
    pairs = [([1.0], [1.0]), ([2.0], [3.0])]
    stats = similarity_distribution(pairs, metric=spectral_angle)
    assert stats["n"] == 0
    assert math.isnan(stats["median"])


def test_similarity_distribution_single_valid_pair():
    stats = similarity_distribution([([1.0, 2.0, 3.0], [1.0, 2.0, 3.0])],
                                    metric=spectral_angle)
    assert stats["n"] == 1
    assert abs(stats["median"] - 1.0) < 1e-9
    assert abs(stats["p25"] - 1.0) < 1e-9
    assert abs(stats["p75"] - 1.0) < 1e-9


def test_similarity_distribution_mix_valid_and_nan_counts_only_valid():
    pairs = [([1.0, 2.0, 3.0], [1.0, 2.0, 3.0]),  # valid -> 1.0
             ([1.0], [1.0])]                        # degenerate -> NaN
    stats = similarity_distribution(pairs, metric=spectral_angle)
    assert stats["n"] == 1
    assert abs(stats["median"] - 1.0) < 1e-9


def test_similarity_distribution_spearman_metric_path():
    pairs = [([1.0, 2.0, 3.0, 4.0], [10.0, 20.0, 30.0, 40.0]),   # rho=1
             ([1.0, 2.0, 3.0, 4.0], [40.0, 30.0, 20.0, 10.0])]   # rho=-1
    stats = similarity_distribution(pairs, metric=spearman_sim)
    assert stats["n"] == 2
    assert abs(stats["median"] - 0.0) < 1e-9


# --------------------------------------------------------------------------
# gate_pass
# --------------------------------------------------------------------------
def test_gate_pass_median_exactly_equal_is_false_strict():
    stats = {"n": 100, "median": 0.7, "p25": 0.6, "p75": 0.8}
    assert gate_pass(stats, min_sim=0.7) is False


def test_gate_pass_median_just_above_is_true():
    stats = {"n": 100, "median": 0.7000001, "p25": 0.6, "p75": 0.8}
    assert gate_pass(stats, min_sim=0.7) is True


def test_gate_pass_zero_n_is_false():
    stats = {"n": 0, "median": 0.99, "p25": 0.99, "p75": 0.99}
    assert gate_pass(stats, min_sim=0.7) is False


def test_gate_pass_nan_median_is_false():
    stats = {"n": 5, "median": float("nan"), "p25": float("nan"), "p75": float("nan")}
    assert gate_pass(stats, min_sim=0.7) is False


# --------------------------------------------------------------------------
# build_pairs_from_maps
# --------------------------------------------------------------------------
def test_build_pairs_no_common_keys_returns_empty():
    pred_map = {frag_key("b", 0, 1): 1.0}
    obs_map = {frag_key("y", 5, 1): 9.0}
    pred_vec, obs_vec = build_pairs_from_maps(pred_map, obs_map)
    assert pred_vec == []
    assert obs_vec == []


def test_build_pairs_all_keys_common_sorted_order():
    pred_map = {frag_key("y", 1, 1): 0.4, frag_key("b", 0, 1): 1.0}
    obs_map = {frag_key("y", 1, 1): 40.0, frag_key("b", 0, 1): 100.0}
    pred_vec, obs_vec = build_pairs_from_maps(pred_map, obs_map)
    # sorted() of frag_key tuples: ('b',0,1) < ('y',1,1)
    assert pred_vec == [1.0, 0.4]
    assert obs_vec == [100.0, 40.0]


def test_build_pairs_deterministic_across_calls():
    pred_map = {frag_key("y", 2, 1): 0.2, frag_key("b", 0, 1): 1.0,
                frag_key("y", 1, 1): 0.4}
    obs_map = {frag_key("y", 2, 1): 20.0, frag_key("b", 0, 1): 100.0,
               frag_key("y", 1, 1): 40.0}
    first = build_pairs_from_maps(pred_map, obs_map)
    second = build_pairs_from_maps(pred_map, obs_map)
    assert first == second
    # Parallel alignment is preserved by the shared sorted key order.
    assert first[0] == [1.0, 0.4, 0.2]
    assert first[1] == [100.0, 40.0, 20.0]


def test_observed_light_map_reverses_y_frag_pos():
    """y_1 must key at the C-terminal cleavage site (frag_pos L-2), not 0.
    Regression for the y-ion alignment bug (final review 2026-06-08)."""
    import numpy as np
    import pytest
    from spectrum.psm_info import PSMInfo, HeavyType
    from workflows.pred_store import frag_key
    from tools.speclib_sanity import _observed_light_map

    psm = PSMInfo(sequence="PEPK", charge=2, modify=[],
                  rt=np.float32(50.0), precursor_mz=np.float32(100.0),
                  raw_title="r", protein_names="X")
    b_ions, y_ions = psm.get_fragment_ions(HeavyType.SILAC)
    y1_mass = next(lm for _it, num, lm, _hm in y_ions if num == 1)
    b1_mass = next(lm for _it, num, lm, _hm in b_ions if num == 1)

    class _MassDia:
        def xic_ms2_peaks_extract(self, rt, win, precursor_mz, ions_mass,
                                  mass_tol_ppm):
            xic = np.zeros(2, dtype=[("rt", "f8"), ("intensity", "f8")])
            xic["intensity"] = np.array([ions_mass, ions_mass * 0.5])
            return xic, 0.0

    m = _observed_light_map(psm, _MassDia(), 6, 10.0)
    L = 4
    # b_1 -> frag_pos 0
    assert m[frag_key("b", 0, 1)] == pytest.approx(b1_mass)
    # y_1 -> frag_pos L-2 = 2 (reversed); buggy code would have put y_3 here
    assert m[frag_key("y", L - 2, 1)] == pytest.approx(y1_mass)
