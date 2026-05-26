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


def test_ms2_cycle_idx_maps_to_owning_ms1_position():
    """MS2 cycle_idx = position of owning MS1 in ms1_indexs."""
    d = DIAData.__new__(DIAData)
    # Suppose global spectrum indexing:
    #   idx 0,1: MS1 of cycle 0, cycle 1
    #   idx 2,3: MS2 of cycle 0 (precursor_scan_id = ms1 scan_id at idx 0)
    #   idx 4:   MS2 of cycle 1 (precursor_scan_id = ms1 scan_id at idx 1)
    d.ms1_indexs = np.array([0, 1], dtype=np.int32)
    d.ms2_indexs = np.array([2, 3, 4], dtype=np.int32)
    # precursor_scan_ids is keyed by global spectrum index;
    # MS1 entries get -1 (no precursor); MS2 entries get owning MS1 scan_id.
    d.precursor_scan_ids = np.array([-1, -1, 100, 100, 101], dtype=np.int32)
    # _scan_id_to_index maps scan_id -> global index
    d._scan_id_to_index = np.zeros(200, dtype=np.int32)
    d._scan_id_to_index[100] = 0
    d._scan_id_to_index[101] = 1

    # MS2 at ms2_indexs[0]==2 belongs to MS1 at global idx 0 == ms1_indexs[0]
    assert d._ms2_cycle_idx(2) == 0
    # MS2 at ms2_indexs[1]==3 belongs to MS1 at global idx 0 (cycle 0)
    assert d._ms2_cycle_idx(3) == 0
    # MS2 at ms2_indexs[2]==4 belongs to MS1 at global idx 1 (cycle 1)
    assert d._ms2_cycle_idx(4) == 1


def test_ms2_cycle_idx_returns_minus_one_when_owning_ms1_missing():
    """If the owning MS1 isn't in ms1_indexs (shouldn't happen but be safe),
    return -1 rather than a wrong cycle number."""
    d = DIAData.__new__(DIAData)
    d.ms1_indexs = np.array([0, 5], dtype=np.int32)
    d.precursor_scan_ids = np.array([-1, 7, -1, 7, -1, -1], dtype=np.int32)
    d._scan_id_to_index = np.zeros(20, dtype=np.int32)
    d._scan_id_to_index[7] = 3  # but 3 isn't in ms1_indexs

    # MS2 at global idx 1: owning MS1 = scan_id 7 -> global idx 3, not in ms1_indexs
    assert d._ms2_cycle_idx(1) == -1
