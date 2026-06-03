"""Tests for DIAData npz format-version handling and refactored
_load_from_mzml peak storage."""
import os

import numpy as np
import pytest

from spectrum.dia_data import DIAData


# ---- npz format version round-trip ----

def _make_minimal_dia_for_save():
    """Build a tiny DIAData that's valid to save_to_file."""
    d = DIAData.__new__(DIAData)
    d.has_mobility = False
    d.has_ms1 = True
    d._max_mz_value = 1000.0
    d._min_mz_value = 400.0
    d._zeroth_frame = 0
    d._scan_max_index = 1
    d.frame_max_index = 2

    d.ms1_indexs = np.array([0], dtype=np.int32)
    d.ms1_indexs_rt = np.array([1.0], dtype=np.float32)
    d.ms2_indexs = np.array([1, 2], dtype=np.int32)
    d.ms2_indexs_rt = np.array([1.1, 1.2], dtype=np.float32)
    d.precursor_scan_ids = np.array([-1, 100, 100], dtype=np.int64)
    d._mz_values = np.array([100.0, 200.0, 300.0], dtype=np.float32)
    d.rt_values = np.array([1.0, 1.1, 1.2], dtype=np.float32)
    d._intensity_values = np.array([10.0, 20.0, 30.0], dtype=np.float32)
    d.mobility_values = np.array([1e-6, 0.0], dtype=np.float32)
    d._cycle_left_precursor = np.array([400.0, 500.0], dtype=np.float32)
    d._scan_id_to_index = np.array([0, 1, 2], dtype=np.int64)
    d._peak_start_idx_list = np.array([0, 1, 2], dtype=np.int64)
    d._peak_stop_idx_list = np.array([1, 2, 3], dtype=np.int64)
    d._precursor_lower_mz = np.array([np.nan, 400.0, 500.0], dtype=np.float32)
    d._precursor_upper_mz = np.array([np.nan, 500.0, 600.0], dtype=np.float32)
    # Optional fields - None
    d._quad_max_mz_value = None
    d._quad_min_mz_value = None
    # Centroid params (required by save_to_file since P0-3 / format v3).
    d._centroid_enabled = True
    d._centroid_rel_threshold = 1e-3
    return d


def _save_legacy_npz(path, dia, version=None):
    """Save a DIAData to npz mimicking the legacy format-version handling.

    If `version` is None: writes the npz WITHOUT a `_format_version` key
    (legacy pre-T5 format). If `version` is an int: writes the npz with
    `_format_version = np.int32(version)`. All other field names mirror
    `DIAData.save_to_file` so that tests around format-version rejection
    only need to vary the version sentinel.
    """
    payload = {
        'has_mobility': dia.has_mobility,
        'has_ms1': dia.has_ms1,
        '_max_mz_value': dia._max_mz_value,
        '_min_mz_value': dia._min_mz_value,
        '_zeroth_frame': dia._zeroth_frame,
        '_scan_max_index': dia._scan_max_index,
        'frame_max_index': dia.frame_max_index,
        'ms1_indexs': dia.ms1_indexs,
        'ms1_indexs_rt': dia.ms1_indexs_rt,
        'ms2_indexs': dia.ms2_indexs,
        'ms2_indexs_rt': dia.ms2_indexs_rt,
        'precursor_scan_ids': dia.precursor_scan_ids,
        '_mz_values': dia._mz_values,
        'rt_values': dia.rt_values,
        '_intensity_values': dia._intensity_values,
        'mobility_values': dia.mobility_values,
        '_cycle_left_precursor': dia._cycle_left_precursor,
        '_scan_id_to_index': dia._scan_id_to_index,
        '_peak_start_idx_list': dia._peak_start_idx_list,
        '_peak_stop_idx_list': dia._peak_stop_idx_list,
        '_precursor_lower_mz': dia._precursor_lower_mz,
        '_precursor_upper_mz': dia._precursor_upper_mz,
    }
    if version is not None:
        payload['_format_version'] = np.int32(version)
    np.savez_compressed(str(path), **payload)


def test_save_writes_format_version_2(tmp_path):
    """save_to_file persists _format_version=3 (bumped from 2 in P0-3)."""
    d = _make_minimal_dia_for_save()
    out = tmp_path / "x.dia.npz"
    d.save_to_file(str(out))

    with np.load(str(out)) as data:
        assert '_format_version' in data, \
            "expected '_format_version' key in saved npz"
        assert int(data['_format_version']) == 3


def test_load_roundtrip_succeeds(tmp_path):
    """save_to_file → load_from_file recovers the same arrays."""
    d = _make_minimal_dia_for_save()
    out = tmp_path / "y.dia.npz"
    d.save_to_file(str(out))

    d2 = DIAData.load_from_file(str(out), use_mmap=False)
    np.testing.assert_array_equal(d2._mz_values, d._mz_values)
    np.testing.assert_array_equal(d2._intensity_values, d._intensity_values)
    np.testing.assert_array_equal(d2.ms1_indexs, d.ms1_indexs)
    assert d2._max_mz_value == d._max_mz_value


def test_load_rejects_missing_format_version(tmp_path):
    """Old npz without _format_version must be rejected with a clear message
    naming the path so the user knows what to delete."""
    d = _make_minimal_dia_for_save()
    out = tmp_path / "old.dia.npz"
    _save_legacy_npz(out, d, version=None)

    with pytest.raises(ValueError, match=r"没有 _format_version"):
        DIAData.load_from_file(str(out), use_mmap=False)


def test_load_rejects_wrong_format_version(tmp_path):
    """npz with _format_version != 3 must be rejected."""
    d = _make_minimal_dia_for_save()
    out = tmp_path / "wrong.dia.npz"
    _save_legacy_npz(out, d, version=99)

    with pytest.raises(ValueError, match=r"_format_version=99"):
        DIAData.load_from_file(str(out), use_mmap=False)


# ---- _load_from_mzml refactor: chunk + concat peak storage ----

class _FakeMzmlReader:
    """Mimic the context manager that pyteomics.mzml.read returns."""
    def __init__(self, spectra):
        self._spectra = spectra
    def __enter__(self):
        return iter(self._spectra)
    def __exit__(self, exc_type, exc_val, exc_tb):
        return False


def _make_spectrum(scan_num, ms_level, rt, mz_arr, int_arr,
                   precursor_scan_num=None, precursor_mz=None,
                   iso_lower_off=0.0, iso_upper_off=0.0):
    """Build a dict matching pyteomics.mzml output shape."""
    spectrum = {
        'id': f'controllerType=0 controllerNumber=1 scan={scan_num}',
        'spectrum title': f'spec{scan_num} dummy',
        'ms level': ms_level,
        'm/z array': np.asarray(mz_arr, dtype=np.float32),
        'intensity array': np.asarray(int_arr, dtype=np.float32),
        'scanList': {
            'scan': [{'scan start time': float(rt)}],
        },
    }
    if ms_level > 1 and precursor_scan_num is not None:
        spectrum['precursorList'] = {
            'precursor': [{
                'spectrumRef': (
                    f'controllerType=0 controllerNumber=1 '
                    f'scan={precursor_scan_num}'),
                'selectedIonList': {
                    'selectedIon': [{
                        'selected ion m/z': precursor_mz,
                        'charge state': 2,
                    }],
                },
                'isolationWindow': {
                    'isolation window lower offset': iso_lower_off,
                    'isolation window upper offset': iso_upper_off,
                },
            }],
        }
    return spectrum


def test_load_from_mzml_chunk_concat_preserves_arrays(monkeypatch):
    """With centroid disabled, refactored _load_from_mzml must produce
    arrays identical to feeding raw peaks in directly."""
    spectra = [
        _make_spectrum(
            scan_num=1, ms_level=1, rt=1.0,
            mz_arr=[400.0, 401.0, 402.0],
            int_arr=[10.0, 20.0, 30.0],
        ),
        _make_spectrum(
            scan_num=2, ms_level=2, rt=1.05,
            mz_arr=[200.0, 201.0],
            int_arr=[5.0, 15.0],
            precursor_scan_num=1, precursor_mz=500.0,
            iso_lower_off=1.0, iso_upper_off=1.0,
        ),
        _make_spectrum(
            scan_num=3, ms_level=2, rt=1.10,
            mz_arr=[300.0, 301.0, 302.0, 303.0],
            int_arr=[1.0, 2.0, 3.0, 4.0],
            precursor_scan_num=1, precursor_mz=600.0,
            iso_lower_off=2.0, iso_upper_off=2.0,
        ),
    ]

    from spectrum import dia_data as dd
    monkeypatch.setattr(dd.mzml, 'read',
                        lambda p: _FakeMzmlReader(spectra))

    d = DIAData()
    d._centroid_enabled = False
    d._load_from_mzml('fake.mzML')

    assert d._mz_values.shape == (9,), \
        f"expected 9 concatenated peaks, got {d._mz_values.shape}"
    assert d._intensity_values.shape == (9,)

    assert list(d._peak_start_idx_list) == [0, 3, 5]
    assert list(d._peak_stop_idx_list) == [3, 5, 9]

    np.testing.assert_array_equal(
        d._mz_values[0:3], np.array([400.0, 401.0, 402.0], dtype=np.float32))
    np.testing.assert_array_equal(
        d._intensity_values[0:3], np.array([10.0, 20.0, 30.0], dtype=np.float32))
    np.testing.assert_array_equal(
        d._mz_values[5:9],
        np.array([300.0, 301.0, 302.0, 303.0], dtype=np.float32))

    np.testing.assert_array_equal(d.ms1_indexs, np.array([0], dtype=np.int32))
    np.testing.assert_array_equal(d.ms2_indexs, np.array([1, 2], dtype=np.int32))


# ---- centroid integration into _load_from_mzml ----

def _profile_gaussian(centers, heights, sigma=0.005, n_per_peak=11,
                      span_sigmas=3.0):
    """Mirror of helper in test_centroid_spectrum: synthesize a profile
    spectrum from isolated Gaussian peaks."""
    mz_chunks = []
    int_chunks = []
    for c, h in zip(centers, heights):
        rel = np.linspace(-span_sigmas, span_sigmas, n_per_peak)
        mz_chunks.append(c + rel * sigma)
        int_chunks.append(h * np.exp(-0.5 * rel ** 2))
    mz = np.concatenate(mz_chunks).astype(np.float32)
    intensity = np.concatenate(int_chunks).astype(np.float32)
    order = np.argsort(mz)
    return mz[order], intensity[order]


def test_load_from_mzml_with_centroid_enabled_compresses_peaks(monkeypatch):
    """With _centroid_enabled=True, profile spectra are compressed to
    one peak per Gaussian; _mz_values length drops from 22 (2 spectra
    × 11 samples) to N peaks."""
    mz1, int1 = _profile_gaussian([400.0], [1000.0])  # 11 points → 1 peak
    mz2, int2 = _profile_gaussian([500.0, 600.0], [800.0, 1200.0])
    # 22 points → 2 peaks

    spectra = [
        _make_spectrum(
            scan_num=1, ms_level=1, rt=1.0,
            mz_arr=mz1, int_arr=int1,
        ),
        _make_spectrum(
            scan_num=2, ms_level=2, rt=1.05,
            mz_arr=mz2, int_arr=int2,
            precursor_scan_num=1, precursor_mz=500.0,
            iso_lower_off=1.0, iso_upper_off=1.0,
        ),
    ]

    from spectrum import dia_data as dd
    monkeypatch.setattr(dd.mzml, 'read',
                        lambda p: _FakeMzmlReader(spectra))

    d = DIAData()
    d._centroid_enabled = True
    d._centroid_rel_threshold = 1e-3
    d._load_from_mzml('fake.mzML')

    # 1 peak from spectrum 0, 2 peaks from spectrum 1
    assert d._mz_values.shape == (3,), \
        f"centroid expected 3 total peaks, got {d._mz_values.shape}"
    assert list(d._peak_start_idx_list) == [0, 1]
    assert list(d._peak_stop_idx_list) == [1, 3]
    # Recovered m/z within 0.001 Da of true centers.
    assert abs(d._mz_values[0] - 400.0) < 0.001
    assert abs(d._mz_values[1] - 500.0) < 0.001
    assert abs(d._mz_values[2] - 600.0) < 0.001


def test_load_from_mzml_skips_centroid_for_already_centroid(monkeypatch):
    """A spectrum carrying 'centroid spectrum' cv term must be
    passed through verbatim, even with _centroid_enabled=True."""
    spectrum = _make_spectrum(
        scan_num=1, ms_level=1, rt=1.0,
        mz_arr=[400.0, 500.0, 600.0],
        int_arr=[10.0, 20.0, 30.0],
    )
    spectrum['centroid spectrum'] = ''  # mark as already centroid

    from spectrum import dia_data as dd
    monkeypatch.setattr(dd.mzml, 'read',
                        lambda p: _FakeMzmlReader([spectrum]))

    d = DIAData()
    d._centroid_enabled = True
    d._load_from_mzml('fake.mzML')

    # Verbatim pass-through
    assert d._mz_values.shape == (3,)
    np.testing.assert_array_equal(
        d._mz_values, np.array([400.0, 500.0, 600.0], dtype=np.float32))
    np.testing.assert_array_equal(
        d._intensity_values, np.array([10.0, 20.0, 30.0], dtype=np.float32))


# ---- DataManager wires config ----

def test_data_manager_passes_centroid_config_to_dia_data(monkeypatch, tmp_path):
    """DataManager.get_dia_data_object reads CENTROID_ENABLED and
    CENTROID_REL_THRESHOLD from [general] and sets them on DIAData
    before _load_from_mzml runs."""
    import configparser
    from manager.data_manager import DataManager

    cfg = configparser.ConfigParser()
    cfg['general'] = {
        'centroid_enabled': 'false',
        'centroid_rel_threshold': '0.005',
    }

    captured = {}

    def fake_load(self, mzml_file_path):
        # Capture the centroid fields at load-time.
        captured['enabled'] = self._centroid_enabled
        captured['threshold'] = self._centroid_rel_threshold
        # Skip actual loading work.
        return None

    monkeypatch.setattr(
        'spectrum.dia_data.DIAData._load_from_mzml', fake_load)

    dm = DataManager(config=cfg, path=str(tmp_path / 'mgr.pkl'))
    dm.get_dia_data_object('does_not_exist.mzML')

    assert captured['enabled'] is False
    assert captured['threshold'] == pytest.approx(0.005)


def test_data_manager_defaults_when_keys_missing(monkeypatch, tmp_path):
    """Missing keys must fall back to DIAData defaults (True / 1e-3)."""
    import configparser
    from manager.data_manager import DataManager

    cfg = configparser.ConfigParser()
    cfg['general'] = {}

    captured = {}

    def fake_load(self, mzml_file_path):
        captured['enabled'] = self._centroid_enabled
        captured['threshold'] = self._centroid_rel_threshold
        return None

    monkeypatch.setattr(
        'spectrum.dia_data.DIAData._load_from_mzml', fake_load)

    dm = DataManager(config=cfg, path=str(tmp_path / 'mgr.pkl'))
    dm.get_dia_data_object('does_not_exist.mzML')

    assert captured['enabled'] is True
    assert captured['threshold'] == pytest.approx(1e-3)


# ============================================================================
# Functional coverage extension (2026-06-01).
#
# Additional integration tests beyond T6/T7/T8: DataManager config-section
# edge cases (absent section, None config), _load_from_mzml structural
# edge cases (multi-cycle, empty peak arrays), and centroid-disabled paths
# with already-centroid input. Reuses existing helpers; does not redefine.
# ============================================================================


def test_data_manager_no_general_section_at_all(monkeypatch, tmp_path):
    """When config has no [general] section, has_section() returns False
    and DIAData retains its constructor defaults."""
    import configparser
    from manager.data_manager import DataManager

    cfg = configparser.ConfigParser()
    # No sections at all

    captured = {}

    def fake_load(self, mzml_file_path):
        captured['enabled'] = self._centroid_enabled
        captured['threshold'] = self._centroid_rel_threshold
        return None

    monkeypatch.setattr(
        'spectrum.dia_data.DIAData._load_from_mzml', fake_load)

    dm = DataManager(config=cfg, path=str(tmp_path / 'mgr.pkl'))
    dm.get_dia_data_object('does_not_exist.mzML')

    assert captured['enabled'] is True
    assert captured['threshold'] == pytest.approx(1e-3)


def test_data_manager_with_none_config(monkeypatch, tmp_path):
    """DataManager(config=None) must not crash; DIAData keeps defaults."""
    from manager.data_manager import DataManager

    captured = {}

    def fake_load(self, mzml_file_path):
        captured['enabled'] = self._centroid_enabled
        captured['threshold'] = self._centroid_rel_threshold
        return None

    monkeypatch.setattr(
        'spectrum.dia_data.DIAData._load_from_mzml', fake_load)

    dm = DataManager(config=None, path=str(tmp_path / 'mgr.pkl'))
    dm.get_dia_data_object('does_not_exist.mzML')

    assert captured['enabled'] is True
    assert captured['threshold'] == pytest.approx(1e-3)


def test_load_from_mzml_already_centroid_with_centroid_disabled(monkeypatch):
    """Both 'centroid spectrum' cv term AND _centroid_enabled=False —
    redundant skips. Input must pass through verbatim regardless."""
    spectrum = _make_spectrum(
        scan_num=1, ms_level=1, rt=1.0,
        mz_arr=[100.0, 200.0, 300.0],
        int_arr=[5.0, 10.0, 15.0],
    )
    spectrum['centroid spectrum'] = ''

    from spectrum import dia_data as dd
    monkeypatch.setattr(dd.mzml, 'read',
                        lambda p: _FakeMzmlReader([spectrum]))

    d = DIAData()
    d._centroid_enabled = False
    d._load_from_mzml('fake.mzML')

    assert d._mz_values.shape == (3,)
    np.testing.assert_array_equal(
        d._mz_values, np.array([100.0, 200.0, 300.0], dtype=np.float32))


def test_load_from_mzml_multi_cycle_dia_structure(monkeypatch):
    """Two DIA cycles (each: 1 MS1 + 2 MS2). Verify ms1_indexs and
    ms2_indexs partition correctly; peak indices stay consistent."""
    spectra = [
        # Cycle 1
        _make_spectrum(scan_num=1, ms_level=1, rt=1.0,
                       mz_arr=[400.0, 401.0], int_arr=[10.0, 20.0]),
        _make_spectrum(scan_num=2, ms_level=2, rt=1.01,
                       mz_arr=[200.0, 201.0], int_arr=[5.0, 15.0],
                       precursor_scan_num=1, precursor_mz=500.0,
                       iso_lower_off=1.0, iso_upper_off=1.0),
        _make_spectrum(scan_num=3, ms_level=2, rt=1.02,
                       mz_arr=[300.0, 301.0, 302.0],
                       int_arr=[1.0, 2.0, 3.0],
                       precursor_scan_num=1, precursor_mz=600.0,
                       iso_lower_off=2.0, iso_upper_off=2.0),
        # Cycle 2
        _make_spectrum(scan_num=4, ms_level=1, rt=2.0,
                       mz_arr=[400.0, 401.0], int_arr=[12.0, 22.0]),
        _make_spectrum(scan_num=5, ms_level=2, rt=2.01,
                       mz_arr=[200.0], int_arr=[7.0],
                       precursor_scan_num=4, precursor_mz=500.0,
                       iso_lower_off=1.0, iso_upper_off=1.0),
        _make_spectrum(scan_num=6, ms_level=2, rt=2.02,
                       mz_arr=[300.0, 301.0], int_arr=[4.0, 5.0],
                       precursor_scan_num=4, precursor_mz=600.0,
                       iso_lower_off=2.0, iso_upper_off=2.0),
    ]

    from spectrum import dia_data as dd
    monkeypatch.setattr(dd.mzml, 'read',
                        lambda p: _FakeMzmlReader(spectra))

    d = DIAData()
    d._centroid_enabled = False  # exact peak counts
    d._load_from_mzml('fake.mzML')

    # MS1/MS2 partitioning
    np.testing.assert_array_equal(d.ms1_indexs,
                                  np.array([0, 3], dtype=np.int32))
    np.testing.assert_array_equal(d.ms2_indexs,
                                  np.array([1, 2, 4, 5], dtype=np.int32))

    # Total peaks: 2+2+3+2+1+2 = 12
    assert d._mz_values.shape == (12,)
    assert int(d._peak_stop_idx_list[-1]) == 12

    # rt_values are populated for all 6 spectra
    np.testing.assert_array_almost_equal(
        d.rt_values,
        np.array([1.0, 1.01, 1.02, 2.0, 2.01, 2.02], dtype=np.float32),
        decimal=5)


def test_load_from_mzml_handles_spectrum_with_zero_peaks(monkeypatch):
    """A spectrum with empty m/z array must not break index bookkeeping.
    _peak_start_idx_list[i] == _peak_stop_idx_list[i] for that spectrum."""
    spectra = [
        _make_spectrum(scan_num=1, ms_level=1, rt=1.0,
                       mz_arr=[400.0, 401.0], int_arr=[10.0, 20.0]),
        _make_spectrum(scan_num=2, ms_level=2, rt=1.01,
                       mz_arr=[], int_arr=[],  # EMPTY
                       precursor_scan_num=1, precursor_mz=500.0,
                       iso_lower_off=1.0, iso_upper_off=1.0),
        _make_spectrum(scan_num=3, ms_level=2, rt=1.02,
                       mz_arr=[300.0], int_arr=[5.0],
                       precursor_scan_num=1, precursor_mz=600.0,
                       iso_lower_off=2.0, iso_upper_off=2.0),
    ]

    from spectrum import dia_data as dd
    monkeypatch.setattr(dd.mzml, 'read',
                        lambda p: _FakeMzmlReader(spectra))

    d = DIAData()
    d._centroid_enabled = False
    d._load_from_mzml('fake.mzML')

    # Total peaks = 2 + 0 + 1 = 3
    assert d._mz_values.shape == (3,)
    # Bookkeeping: spectrum 1 (empty) has start == stop
    assert int(d._peak_start_idx_list[1]) == 2
    assert int(d._peak_stop_idx_list[1]) == 2
    # Spectrum 2 picks up at peak idx 2
    assert int(d._peak_start_idx_list[2]) == 2
    assert int(d._peak_stop_idx_list[2]) == 3


def test_load_from_mzml_centroid_below_threshold_yields_empty_chunk(monkeypatch):
    """When centroid filters all peaks (e.g., rel_threshold > max), the
    spectrum contributes 0 peaks. _peak_start_idx_list[i] ==
    _peak_stop_idx_list[i] and concat across the loop stays consistent."""
    # Spectrum 0: a strong gaussian peak — centroid keeps 1 peak
    mz0, int0 = _profile_gaussian([400.0], [1000.0])
    # Spectrum 1: a flat-low input so the centroid detector finds no local
    # max and produces 0 peaks.
    mz1 = np.array([500.0, 500.001, 500.002], dtype=np.float32)
    int1 = np.array([0.1, 0.1, 0.1], dtype=np.float32)

    spectra = [
        _make_spectrum(scan_num=1, ms_level=1, rt=1.0,
                       mz_arr=mz0, int_arr=int0),
        _make_spectrum(scan_num=2, ms_level=2, rt=1.01,
                       mz_arr=mz1, int_arr=int1,
                       precursor_scan_num=1, precursor_mz=500.0,
                       iso_lower_off=1.0, iso_upper_off=1.0),
    ]

    from spectrum import dia_data as dd
    monkeypatch.setattr(dd.mzml, 'read',
                        lambda p: _FakeMzmlReader(spectra))

    d = DIAData()
    d._centroid_enabled = True
    d._centroid_rel_threshold = 1e-3
    d._load_from_mzml('fake.mzML')

    # Spectrum 0 → 1 centroid; spectrum 1 → 0 centroids
    assert d._mz_values.shape == (1,)
    # Bookkeeping: spectrum 0 occupies [0:1]; spectrum 1 occupies [1:1]
    assert int(d._peak_start_idx_list[0]) == 0
    assert int(d._peak_stop_idx_list[0]) == 1
    assert int(d._peak_start_idx_list[1]) == 1
    assert int(d._peak_stop_idx_list[1]) == 1


# ============================================================================
# scan_id sizing fix (followup-scan-id-sizing, 2026-06-02).
#
# Pre-existing bug: _preallocate_arrays used `_scan_id_to_index =
# np.zeros(total_spectra + 10)`. This assumed scan_ids ∈ [0, total_spectra+10),
# which is false for real DIA mzML where:
#   - pParse / ProteoWizard may filter out scans, leaving remaining ones with
#     their ORIGINAL (larger) scan_nums.
#   - Thermo raw has interleaved lock-mass / cal scans that get dropped,
#     leaving scan_nums up to ~2-5x total_spectra in the mzML.
# Symptom: `_scan_id_to_index[scan_id] = spectrum_idx` IndexError when
# scan_id ≥ total_spectra + 10.
#
# Fix: size by `max(scan_id) + 1`, computed in the first pass of
# _load_from_mzml. Drop the magic +10 band-aid.
# ============================================================================


def test_load_from_mzml_handles_sparse_scan_ids(monkeypatch):
    """Reproducer for the pre-existing _scan_id_to_index sizing bug.

    3 spectra with scan_nums 100, 200, 300 — much larger than
    total_spectra (3). The old code allocated `_scan_id_to_index =
    np.zeros(3 + 10) = size 13`, then `_scan_id_to_index[100] = 0`
    raised IndexError. After the fix, the array is sized by
    max(scan_id) + 1 = 301, and the writes succeed.
    """
    spectra = [
        _make_spectrum(scan_num=100, ms_level=1, rt=1.0,
                       mz_arr=[400.0, 401.0], int_arr=[10.0, 20.0]),
        _make_spectrum(scan_num=200, ms_level=2, rt=1.05,
                       mz_arr=[200.0], int_arr=[5.0],
                       precursor_scan_num=100, precursor_mz=500.0,
                       iso_lower_off=1.0, iso_upper_off=1.0),
        _make_spectrum(scan_num=300, ms_level=2, rt=1.10,
                       mz_arr=[300.0, 301.0], int_arr=[1.0, 2.0],
                       precursor_scan_num=100, precursor_mz=600.0,
                       iso_lower_off=2.0, iso_upper_off=2.0),
    ]

    from spectrum import dia_data as dd
    monkeypatch.setattr(dd.mzml, 'read',
                        lambda p: _FakeMzmlReader(spectra))

    d = DIAData()
    d._centroid_enabled = False
    d._load_from_mzml('fake.mzML')  # must not raise IndexError

    # Sanity: array sized to accommodate max scan_id
    assert len(d._scan_id_to_index) >= 301, (
        f"_scan_id_to_index should be sized by max(scan_id)+1, got "
        f"{len(d._scan_id_to_index)}")

    # Lookup: scan_id -> spectrum_idx mapping is correct
    assert int(d._scan_id_to_index[100]) == 0
    assert int(d._scan_id_to_index[200]) == 1
    assert int(d._scan_id_to_index[300]) == 2


def test_get_spectrum_works_for_sparse_scan_ids(monkeypatch):
    """Round-trip: scan_id -> spectrum_idx -> peaks via public API.

    Verifies that `get_spectrum(scan_id)` returns the correct peaks
    for sparse scan_ids (not just dense 0..N-1).
    """
    spectra = [
        _make_spectrum(scan_num=100, ms_level=1, rt=1.0,
                       mz_arr=[400.0, 401.0], int_arr=[10.0, 20.0]),
        _make_spectrum(scan_num=500, ms_level=2, rt=1.05,
                       mz_arr=[200.0, 201.0, 202.0],
                       int_arr=[5.0, 15.0, 25.0],
                       precursor_scan_num=100, precursor_mz=500.0,
                       iso_lower_off=1.0, iso_upper_off=1.0),
    ]

    from spectrum import dia_data as dd
    monkeypatch.setattr(dd.mzml, 'read',
                        lambda p: _FakeMzmlReader(spectra))

    d = DIAData()
    d._centroid_enabled = False
    d._load_from_mzml('fake.mzML')

    # scan_id 100 -> spec 0 -> [400.0, 401.0]
    mz, intensity = d.get_spectrum(100)
    np.testing.assert_array_equal(
        mz, np.array([400.0, 401.0], dtype=np.float32))
    np.testing.assert_array_equal(
        intensity, np.array([10.0, 20.0], dtype=np.float32))

    # scan_id 500 -> spec 1 -> [200.0, 201.0, 202.0]
    mz, intensity = d.get_spectrum(500)
    np.testing.assert_array_equal(
        mz, np.array([200.0, 201.0, 202.0], dtype=np.float32))


def test_sparse_scan_ids_npz_save_load_roundtrip(tmp_path, monkeypatch):
    """The corrected _scan_id_to_index size must survive npz save/load."""
    spectra = [
        _make_spectrum(scan_num=42, ms_level=1, rt=1.0,
                       mz_arr=[400.0], int_arr=[10.0]),
        _make_spectrum(scan_num=999, ms_level=2, rt=1.05,
                       mz_arr=[200.0], int_arr=[5.0],
                       precursor_scan_num=42, precursor_mz=500.0,
                       iso_lower_off=1.0, iso_upper_off=1.0),
    ]

    from spectrum import dia_data as dd
    monkeypatch.setattr(dd.mzml, 'read',
                        lambda p: _FakeMzmlReader(spectra))

    d = DIAData()
    d._centroid_enabled = False
    d._load_from_mzml('fake.mzML')

    npz_path = tmp_path / "sparse.dia.npz"
    d.save_to_file(str(npz_path))

    d2 = DIAData.load_from_file(str(npz_path), use_mmap=False)

    # Lookup array survives round-trip with same size
    assert len(d2._scan_id_to_index) == len(d._scan_id_to_index)
    assert int(d2._scan_id_to_index[42]) == 0
    assert int(d2._scan_id_to_index[999]) == 1
