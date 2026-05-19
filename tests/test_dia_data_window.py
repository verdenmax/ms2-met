"""Tests for DIAData.get_window_info — extended (lower, upper) return."""
import numpy as np
import pytest

from spectrum.dia_data import DIAData


def _make_minimal_dia(windows):
    """Build a minimal DIAData with N MS2 windows.

    windows: list of (lower_mz, upper_mz) — defines one DIA cycle.
    """
    d = DIAData.__new__(DIAData)
    n = len(windows)
    d._precursor_lower_mz = np.array(
        [np.nan] + [lo for lo, _ in windows], dtype=np.float64)
    d._precursor_upper_mz = np.array(
        [np.nan] + [hi for _, hi in windows], dtype=np.float64)
    # ms2_indexs points into the global arrays (skip the MS1 at idx 0)
    d.ms2_indexs = np.arange(1, n + 1)
    d._cycle_left_precursor = np.array([lo for lo, _ in windows])
    return d


def test_get_window_info_returns_lower_upper():
    """get_window_info must return lower and upper bounds for caller
    to detect co-isolation by exact-pair match."""
    dia = _make_minimal_dia([(500.0, 502.0), (502.0, 504.0)])
    info = dia.get_window_info(501.0)
    assert info["lower"] == 500.0
    assert info["upper"] == 502.0
    assert info["width"] == 2.0


def test_get_window_info_no_match_returns_zero_width_nans():
    """When no window contains the precursor, width=0 and bounds=NaN."""
    dia = _make_minimal_dia([(500.0, 502.0)])
    info = dia.get_window_info(900.0)
    assert info["width"] == 0.0
    assert np.isnan(info["lower"])
    assert np.isnan(info["upper"])


def test_get_window_info_boundary_inclusive_with_tolerance():
    """A precursor at the upper boundary should still match (existing
    code uses 0.1 Da tolerance)."""
    dia = _make_minimal_dia([(500.0, 502.0)])
    info = dia.get_window_info(502.0)
    assert info["lower"] == 500.0
    assert info["upper"] == 502.0
