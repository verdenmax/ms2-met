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
