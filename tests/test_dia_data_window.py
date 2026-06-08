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


def test_ms1_xic_returns_cycle_idx_field():
    """xic_peaks_extreact dtype includes cycle_idx = ms1_indexs position."""
    d = DIAData.__new__(DIAData)
    # 5 MS1 spectra at global indices 0..4, equally spaced RT
    d.ms1_indexs = np.array([0, 1, 2, 3, 4], dtype=np.int32)
    d.ms1_indexs_rt = np.array([10.0, 20.0, 30.0, 40.0, 50.0], dtype=np.float32)
    d.rt_values = np.array([10.0, 20.0, 30.0, 40.0, 50.0], dtype=np.float32)
    # Empty peak lists so match_peak_ppm returns (nan, 0) safely
    d._peak_start_idx_list = np.zeros(5, dtype=np.int64)
    d._peak_stop_idx_list = np.zeros(5, dtype=np.int64)
    d._mz_values = np.array([], dtype=np.float32)
    d._intensity_values = np.array([], dtype=np.float32)

    xic = d.xic_peaks_extreact(
        rt=np.float32(30.0), xic_cycle_window=2,
        precursor_mz=np.float32(500.0), mass_tol_ppm=np.float32(10.0))

    assert "cycle_idx" in xic.dtype.names
    # Center = ms1_indexs[2] (RT=30); window=2 -> indices 0..4 of ms1_indexs
    assert list(xic["cycle_idx"]) == [0, 1, 2, 3, 4]


def test_ms2_xic_returns_cycle_idx_field():
    """xic_ms2_peaks_extract dtype includes cycle_idx that maps
    each entry to its owning MS1's position in ms1_indexs."""
    d = DIAData.__new__(DIAData)
    # Layout (global indices):
    #   0: MS1 (cycle 0), 1: MS2 of cycle 0
    #   2: MS1 (cycle 1), 3: MS2 of cycle 1
    #   4: MS1 (cycle 2), 5: MS2 of cycle 2
    d.ms1_indexs = np.array([0, 2, 4], dtype=np.int32)
    d.ms1_indexs_rt = np.array([10.0, 30.0, 50.0], dtype=np.float32)
    d.ms2_indexs = np.array([1, 3, 5], dtype=np.int32)
    d.ms2_indexs_rt = np.array([15.0, 35.0, 55.0], dtype=np.float32)
    d.rt_values = np.array(
        [10.0, 15.0, 30.0, 35.0, 50.0, 55.0], dtype=np.float32)
    d.precursor_scan_ids = np.array(
        [-1, 100, -1, 101, -1, 102], dtype=np.int32)
    d._scan_id_to_index = np.zeros(200, dtype=np.int32)
    d._scan_id_to_index[100] = 0
    d._scan_id_to_index[101] = 2
    d._scan_id_to_index[102] = 4
    # All MS2 windows contain precursor_mz=500.0
    d._precursor_lower_mz = np.array(
        [np.nan, 499.0, np.nan, 499.0, np.nan, 499.0], dtype=np.float64)
    d._precursor_upper_mz = np.array(
        [np.nan, 501.0, np.nan, 501.0, np.nan, 501.0], dtype=np.float64)
    # Empty peak lists so match_peak_ppm returns harmlessly
    d._peak_start_idx_list = np.zeros(6, dtype=np.int64)
    d._peak_stop_idx_list = np.zeros(6, dtype=np.int64)
    d._mz_values = np.array([], dtype=np.float32)
    d._intensity_values = np.array([], dtype=np.float32)
    d._cycle_left_precursor = np.array([499.0], dtype=np.float32)

    xic, _ = d.xic_ms2_peaks_extract(
        rt=np.float32(35.0), xic_cycle_window=1,
        precursor_mz=np.float32(500.0), ions_mass=np.float32(200.0),
        mass_tol_ppm=np.float32(10.0))

    assert "cycle_idx" in xic.dtype.names
    # Center is MS2 at rt=35 (global idx 3, cycle 1).
    # Window=1 -> 1 left + center + 1 right = 3 entries; cycles [0,1,2].
    assert list(xic["cycle_idx"]) == [0, 1, 2]


def test_ms2_xic_empty_path_still_has_cycle_idx_in_dtype():
    """Early-return path (no matching window) must still emit cycle_idx."""
    d = DIAData.__new__(DIAData)
    d.ms2_indexs = np.array([], dtype=np.int32)
    d.ms2_indexs_rt = np.array([], dtype=np.float32)

    xic, _ = d.xic_ms2_peaks_extract(
        rt=np.float32(10.0), xic_cycle_window=3,
        precursor_mz=np.float32(500.0), ions_mass=np.float32(200.0),
        mass_tol_ppm=np.float32(10.0))
    assert "cycle_idx" in xic.dtype.names
    assert len(xic) == 0


def test_ms2_xic_finds_window_beyond_5_scans():
    """多窗口 DIA：含 precursor_mz 的隔离窗口每 N 个 MS2 谱图才出现一次。
    当它距 pos 超过 5 时，旧的 ±5 中心搜索会返回空 XIC（漏掉信号）。
    回归：center 搜索须向两侧无界扩展找最近匹配窗口。"""
    N = 10
    d = DIAData.__new__(DIAData)
    d.ms2_indexs = np.arange(1, 21, dtype=np.int32)      # 20 个 MS2 (2 cycle×10 窗口)
    d.ms2_indexs_rt = np.arange(20, dtype=np.float32)    # rt 0..19
    # global idx g(1..20) 的窗口 = (g-1)%N；窗口 9 = [518,520]
    d._precursor_lower_mz = np.array(
        [np.nan] + [500.0 + 2 * ((g - 1) % N) for g in range(1, 21)],
        dtype=np.float64)
    d._precursor_upper_mz = np.array(
        [np.nan] + [502.0 + 2 * ((g - 1) % N) for g in range(1, 21)],
        dtype=np.float64)
    d.rt_values = np.arange(21, dtype=np.float32)
    d._peak_start_idx_list = np.zeros(21, dtype=np.int64)
    d._peak_stop_idx_list = np.zeros(21, dtype=np.int64)
    d._mz_values = np.array([], dtype=np.float32)
    d._intensity_values = np.array([], dtype=np.float32)
    d.ms1_indexs = np.array([0], dtype=np.int32)
    d.precursor_scan_ids = np.array([-1] + [100] * 20, dtype=np.int32)
    d._scan_id_to_index = np.zeros(200, dtype=np.int32)
    d._scan_id_to_index[100] = 0
    d._n_out_of_window_xic = 0

    # precursor_mz=519 在窗口 9（global idx 10 与 20）；rt=2 → pos≈2，最近匹配在 j=9（7 远）
    xic, _ = d.xic_ms2_peaks_extract(
        rt=np.float32(2.0), xic_cycle_window=2,
        precursor_mz=np.float32(519.0), ions_mass=np.float32(200.0),
        mass_tol_ppm=np.float32(10.0))
    assert len(xic) > 0, "含 precursor 的窗口距 pos>5，center 搜索未找到 → 空 XIC"
