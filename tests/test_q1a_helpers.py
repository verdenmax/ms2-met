"""Tests for workflows/q1a_helpers.py — Q1a fragment-pairing math."""
import numpy as np
import pytest


XIC_DTYPE = [
    ("rt", "f8"), ("intensity", "f8"),
    ("ppm_error", "f8"), ("mz", "f8"),
]


def _xic(rts, intensities):
    """Build a minimal XIC numpy record array for tests."""
    n = len(rts)
    arr = np.zeros(n, dtype=XIC_DTYPE)
    arr["rt"] = rts
    arr["intensity"] = intensities
    return arr


# ----------------------------------------------------------------------
# is_signal_present_light
# ----------------------------------------------------------------------

def test_light_present_when_max_intensity_above_floor():
    from workflows.q1a_helpers import is_signal_present_light
    xic = _xic([10, 11, 12, 13, 14], [50, 200, 500, 200, 50])
    assert is_signal_present_light(xic, intensity_floor=100) is True


def test_light_absent_when_max_intensity_below_floor():
    from workflows.q1a_helpers import is_signal_present_light
    xic = _xic([10, 11, 12], [10, 50, 80])
    assert is_signal_present_light(xic, intensity_floor=100) is False


def test_light_absent_when_xic_empty():
    from workflows.q1a_helpers import is_signal_present_light
    xic = _xic([], [])
    assert is_signal_present_light(xic, intensity_floor=100) is False


def test_light_absent_when_all_nan_intensity():
    from workflows.q1a_helpers import is_signal_present_light
    xic = _xic([10, 11, 12], [np.nan, np.nan, np.nan])
    assert is_signal_present_light(xic, intensity_floor=100) is False


# ----------------------------------------------------------------------
# is_signal_present_heavy (three-condition AND)
# ----------------------------------------------------------------------

def test_heavy_present_perfect_pair():
    """Heavy XIC matches light shape + intensity + apex → present."""
    from workflows.q1a_helpers import is_signal_present_heavy
    light = _xic([10, 11, 12, 13, 14], [50, 200, 500, 200, 50])
    heavy = _xic([10, 11, 12, 13, 14], [40, 160, 400, 160, 40])  # same shape
    assert is_signal_present_heavy(
        light, heavy, intensity_floor=100,
        apex_delta_fraction=0.3, pearson_min=0.5) is True


def test_heavy_absent_when_intensity_below_floor():
    from workflows.q1a_helpers import is_signal_present_heavy
    light = _xic([10, 11, 12, 13, 14], [50, 200, 500, 200, 50])
    heavy = _xic([10, 11, 12, 13, 14], [5, 20, 50, 20, 5])  # all < 100
    assert is_signal_present_heavy(
        light, heavy, intensity_floor=100) is False


def test_heavy_absent_when_apex_delta_too_large():
    """Heavy apex far from light apex → not coeluting → absent."""
    from workflows.q1a_helpers import is_signal_present_heavy
    light = _xic([10, 11, 12, 13, 14], [50, 200, 500, 200, 50])  # apex at 12
    heavy = _xic([10, 11, 12, 13, 14], [500, 200, 50, 20, 10])    # apex at 10
    # peak width = 4; apex delta = 2; 2 > 0.3*4 = 1.2 → absent
    assert is_signal_present_heavy(
        light, heavy, intensity_floor=100,
        apex_delta_fraction=0.3) is False


def test_heavy_absent_when_pearson_below_threshold():
    """Same intensity + coeluting but anti-correlated shape → absent."""
    from workflows.q1a_helpers import is_signal_present_heavy
    light = _xic([10, 11, 12, 13, 14], [50, 200, 500, 200, 50])
    # Inverted: same total intensity but anti-correlated
    heavy = _xic([10, 11, 12, 13, 14], [500, 200, 500, 200, 500])
    assert is_signal_present_heavy(
        light, heavy, intensity_floor=100,
        apex_delta_fraction=1.0,  # disable apex check to isolate pearson
        pearson_min=0.9) is False


def test_heavy_absent_when_empty_xic():
    from workflows.q1a_helpers import is_signal_present_heavy
    light = _xic([10, 11, 12], [50, 200, 500])
    heavy = _xic([], [])
    assert is_signal_present_heavy(light, heavy, intensity_floor=100) is False


# ----------------------------------------------------------------------
# Window separability
# ----------------------------------------------------------------------

def test_is_split_window_same_bounds_returns_false():
    from workflows.q1a_helpers import is_split_window
    w_L = {"width": 2.0, "centering": 0.5, "lower": 500.0, "upper": 502.0}
    w_H = {"width": 2.0, "centering": 0.7, "lower": 500.0, "upper": 502.0}
    assert is_split_window(w_L, w_H) is False


def test_is_split_window_different_bounds_returns_true():
    from workflows.q1a_helpers import is_split_window
    w_L = {"width": 2.0, "centering": 0.5, "lower": 500.0, "upper": 502.0}
    w_H = {"width": 2.0, "centering": 0.5, "lower": 504.0, "upper": 506.0}
    assert is_split_window(w_L, w_H) is True


def test_is_split_window_nan_treated_as_split():
    """If either window lookup fails (NaN bounds), be conservative
    and treat as split (more inclusion in q1a)."""
    from workflows.q1a_helpers import is_split_window
    w_L = {"width": 2.0, "centering": 0.5, "lower": 500.0, "upper": 502.0}
    w_H = {"width": 0.0, "centering": 0.5,
           "lower": float("nan"), "upper": float("nan")}
    assert is_split_window(w_L, w_H) is True


# ----------------------------------------------------------------------
# Fragment separability
# ----------------------------------------------------------------------

def test_shifted_fragment_always_separable():
    """A fragment with K or R is shifted by SILAC; always separable
    by m/z regardless of window configuration."""
    from workflows.q1a_helpers import is_separable_fragment
    assert is_separable_fragment(
        light_mass=300.0, heavy_mass=310.0, split_window=True) is True
    assert is_separable_fragment(
        light_mass=300.0, heavy_mass=310.0, split_window=False) is True


def test_unshifted_fragment_separable_only_in_split_window():
    """A fragment with no K/R has equal light/heavy mass. It can only
    be separated by the DIA window (its precursor isolation differs)."""
    from workflows.q1a_helpers import is_separable_fragment
    assert is_separable_fragment(
        light_mass=300.0, heavy_mass=300.0, split_window=True) is True
    assert is_separable_fragment(
        light_mass=300.0, heavy_mass=300.0, split_window=False) is False
