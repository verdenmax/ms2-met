"""Tests for DIAData.get_window_info — extended (lower, upper) return."""
import numpy as np
import pytest

from spectrum.dia_data import DIAData
from spectrum.spectrum_utils import match_peak_panel_ppm


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


def test_get_window_info_boundary_tie_uses_canonical_lower_window():
    """An exact shared boundary uses the deterministic lower-bound tie-break."""
    dia = _make_minimal_dia([(500.0, 502.0), (502.0, 504.0)])

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
    assert np.all(np.isnan(xic["ppm_error"]))


def test_ms2_xic_selects_one_centered_window_per_overlapping_cycle():
    """Boundary-overlapping windows must not duplicate global cycle_idx."""
    d = DIAData.__new__(DIAData)
    # Three cycles, each with two overlapping windows containing m/z 500.6.
    # The [500, 502] window is more centered than [499, 501].
    d.ms1_indexs = np.array([0, 3, 6], dtype=np.int32)
    d.ms1_indexs_rt = np.array([10.0, 20.0, 30.0], dtype=np.float32)
    d.ms2_indexs = np.array([1, 2, 4, 5, 7, 8], dtype=np.int32)
    d.ms2_indexs_rt = np.array(
        [10.1, 10.2, 20.1, 20.2, 30.1, 30.2], dtype=np.float32)
    d.rt_values = np.array(
        [10.0, 10.1, 10.2, 20.0, 20.1, 20.2,
         30.0, 30.1, 30.2], dtype=np.float32)
    d.precursor_scan_ids = np.array(
        [-1, 100, 100, -1, 101, 101, -1, 102, 102], dtype=np.int32)
    d._scan_id_to_index = np.zeros(200, dtype=np.int32)
    d._scan_id_to_index[100] = 0
    d._scan_id_to_index[101] = 3
    d._scan_id_to_index[102] = 6
    d._precursor_lower_mz = np.array(
        [np.nan, 499.0, 500.0, np.nan, 499.0, 500.0,
         np.nan, 499.0, 500.0])
    d._precursor_upper_mz = np.array(
        [np.nan, 501.0, 502.0, np.nan, 501.0, 502.0,
         np.nan, 501.0, 502.0])

    selected = d._select_ms2_xic_indices(
        np.float32(20.0), 1, np.float32(500.6))

    assert selected == [2, 5, 8]
    assert [d._ms2_cycle_idx(index) for index in selected] == [0, 1, 2]

    # Window metadata and same-window routing must use the same selected
    # center scan rather than the first matching/left-boundary window.
    info = d.get_window_info(500.6, rt=20.0)
    assert selected[1] == 5
    assert info["lower"] == 500.0
    assert info["upper"] == 502.0
    assert info["centering"] == pytest.approx(0.3)
    assert d.check_in_same_ms2(500.6, 500.8, rt=20.0) is True
    # Both masses occur in both overlapping windows, but their respective
    # most-centered scans differ.
    assert d.check_in_same_ms2(500.4, 500.6, rt=20.0) is False

    # The historical table-feature API and the Phase 2 panel API must route
    # through the exact same scan selection policy.
    proton = 1.00727646677
    ion_mass = 200.0
    fragment_mz = np.array([
        ion_mass + proton,
        (ion_mass + 2 * proton) / 2,
    ])
    d.get_spectrum_by_index = lambda index: (
        fragment_mz, np.array([float(index), float(index)]))
    panel, _ = d.xic_ms2_fragment_panel_extract(
        np.float32(20.0), 1, np.float32(500.6), [ion_mass],
        np.float32(10.0))
    legacy, _ = d.xic_ms2_peaks_extract(
        np.float32(20.0), 1, np.float32(500.6), np.float32(ion_mass),
        np.float32(10.0))

    assert panel[0][1]["cycle_idx"].tolist() == [0, 1, 2]
    assert legacy["cycle_idx"].tolist() == [0, 1, 2]


def test_window_resolution_follows_rt_for_staggered_cycles():
    """Metadata/routing must follow the cycle actually selected at PSM RT."""
    d = DIAData.__new__(DIAData)
    # Cycle 0 and cycle 1 use shifted/staggered isolation boundaries.
    d.ms1_indexs = np.array([0, 3], dtype=np.int32)
    d.ms1_indexs_rt = np.array([10.0, 20.0], dtype=np.float32)
    d.ms2_indexs = np.array([1, 2, 4, 5], dtype=np.int32)
    d.ms2_indexs_rt = np.array(
        [10.1, 10.2, 20.1, 20.2], dtype=np.float32)
    d.rt_values = np.array(
        [10.0, 10.1, 10.2, 20.0, 20.1, 20.2], dtype=np.float32)
    d.precursor_scan_ids = np.array(
        [-1, 100, 100, -1, 101, 101], dtype=np.int32)
    d._scan_id_to_index = np.zeros(200, dtype=np.int32)
    d._scan_id_to_index[100] = 0
    d._scan_id_to_index[101] = 3
    d._precursor_lower_mz = np.array(
        [np.nan, 499.0, 501.0, np.nan, 499.5, 501.5])
    d._precursor_upper_mz = np.array(
        [np.nan, 501.0, 503.0, np.nan, 501.5, 503.5])

    early = d.get_window_info(501.4, rt=10.0)
    late = d.get_window_info(501.4, rt=20.0)

    assert (early["lower"], early["upper"]) == (501.0, 503.0)
    assert (late["lower"], late["upper"]) == (499.5, 501.5)
    assert d.get_window_info(501.4, rt=20.0)["lower"] == 499.5
    assert d._select_ms2_xic_indices(10.0, 0, 501.4) == [2]
    assert d._select_ms2_xic_indices(20.0, 0, 501.4) == [4]


def test_window_resolution_checks_both_sides_of_rt_insertion_point():
    """The closest matching cycle may be immediately before insertion."""
    d = DIAData.__new__(DIAData)
    d.ms1_indexs = np.array([0, 2], dtype=np.int32)
    d.ms2_indexs = np.array([1, 3], dtype=np.int32)
    d.ms2_indexs_rt = np.array([9.0, 11.0], dtype=np.float32)
    d.precursor_scan_ids = np.array([-1, 100, -1, 101], dtype=np.int32)
    d._scan_id_to_index = np.zeros(200, dtype=np.int32)
    d._scan_id_to_index[100] = 0
    d._scan_id_to_index[101] = 2
    d._precursor_lower_mz = np.array([np.nan, 499.0, np.nan, 499.0])
    d._precursor_upper_mz = np.array([np.nan, 501.0, np.nan, 501.0])

    assert d._select_ms2_xic_indices(9.2, 0, 500.0) == [1]


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


def test_ms2_xic_pools_charge_states_and_weights_ppm_by_intensity():
    d = DIAData.__new__(DIAData)
    d.ms2_indexs = np.array([0], dtype=np.int32)
    d.ms2_indexs_rt = np.array([10.0], dtype=np.float32)
    d.rt_values = np.array([10.0], dtype=np.float32)
    d._precursor_lower_mz = np.array([499.0])
    d._precursor_upper_mz = np.array([501.0])
    proton = 1.00727646677
    ion_mass = 200.0
    charge1 = ion_mass + proton
    charge2 = (ion_mass + 2 * proton) / 2
    mz = np.array([charge1 * (1 + 10e-6), charge2 * (1 - 10e-6)])
    intensity = np.array([10.0, 30.0])
    d.get_spectrum_by_index = lambda _: (mz, intensity)
    d._ms2_cycle_idx = lambda _: 7

    resolved, _ = d.xic_ms2_charge_resolved_extract(
        rt=np.float32(10.0), xic_cycle_window=0,
        precursor_mz=np.float32(500.0), ions_mass=np.float32(ion_mass),
        mass_tol_ppm=np.float32(20.0))
    assert resolved[1]["intensity"][0] == pytest.approx(10.0)
    assert resolved[2]["intensity"][0] == pytest.approx(30.0)
    assert resolved[1]["ppm_error"][0] == pytest.approx(10.0, abs=0.05)
    assert resolved[2]["ppm_error"][0] == pytest.approx(-10.0, abs=0.05)

    xic, _ = d.xic_ms2_peaks_extract(
        rt=np.float32(10.0), xic_cycle_window=0,
        precursor_mz=np.float32(500.0), ions_mass=np.float32(ion_mass),
        mass_tol_ppm=np.float32(20.0))

    assert xic["intensity"][0] == pytest.approx(40.0)
    assert xic["ppm_error"][0] == pytest.approx(-5.0, abs=0.05)


def test_charge_resolved_xic_rejects_invalid_charge_contract():
    d = DIAData.__new__(DIAData)
    d.ms2_indexs = np.array([], dtype=np.int32)
    d.ms2_indexs_rt = np.array([], dtype=np.float32)
    with pytest.raises(ValueError, match="positive integers"):
        d.xic_ms2_charge_resolved_extract(
            rt=np.float32(10), xic_cycle_window=1,
            precursor_mz=np.float32(500), ions_mass=np.float32(200),
            mass_tol_ppm=np.float32(10), fragment_charges=(0, 1))


def test_fragment_panel_loads_each_selected_ms2_scan_once():
    d = DIAData.__new__(DIAData)
    d.ms2_indexs = np.array([0, 1, 2], dtype=np.int32)
    d.ms2_indexs_rt = np.array([9.0, 10.0, 11.0], dtype=np.float32)
    d.rt_values = d.ms2_indexs_rt.copy()
    d._precursor_lower_mz = np.array([499.0] * 3)
    d._precursor_upper_mz = np.array([501.0] * 3)
    d._ms2_cycle_idx = lambda index: index
    calls = []

    def spectrum(index):
        calls.append(index)
        return np.array([101.0, 201.0]), np.array([3.0, 5.0])

    d.get_spectrum_by_index = spectrum
    panel, _ = d.xic_ms2_fragment_panel_extract(
        rt=np.float32(10.0), xic_cycle_window=1,
        precursor_mz=np.float32(500.0),
        ions_masses=np.array([100.0, 200.0]),
        mass_tol_ppm=np.float32(200.0), fragment_charges=(1,))

    assert calls == [0, 1, 2]
    assert len(panel) == 2
    assert all(len(item[1]) == 3 for item in panel)


def test_ms1_panels_load_each_scan_once():
    d = DIAData.__new__(DIAData)
    d.ms1_indexs = np.array([0, 1, 2], dtype=np.int32)
    d.ms1_indexs_rt = np.array([9.0, 10.0, 11.0], dtype=np.float32)
    d.rt_values = d.ms1_indexs_rt.copy()
    calls = []

    def spectrum(index):
        calls.append(index)
        return np.array([500.0, 600.0]), np.array([3.0, 5.0])

    d.get_spectrum_by_index = spectrum
    panels = d.xic_peaks_panels_extract(
        np.float32(10.0), 1, [[500.0], [600.0]], np.float32(10.0))

    assert calls == [0, 1, 2]
    assert len(panels) == 2
    assert panels[0]["intensity"].tolist() == pytest.approx([3.0] * 3)
    assert panels[1]["intensity"].tolist() == pytest.approx([5.0] * 3)


def test_isotope_target_panel_counts_overlapping_centroid_once():
    targets = np.array([500.0000, 500.0040])
    mz = np.array([500.0020])
    intensity = np.array([123.0])
    ppm, observed = match_peak_panel_ppm(mz, intensity, targets, 10.0)
    assert observed == pytest.approx(123.0)
    assert abs(float(ppm)) < 10.0


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
