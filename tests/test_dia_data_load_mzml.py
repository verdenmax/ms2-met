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
    """save_to_file persists _format_version=2."""
    d = _make_minimal_dia_for_save()
    out = tmp_path / "x.dia.npz"
    d.save_to_file(str(out))

    with np.load(str(out)) as data:
        assert '_format_version' in data, \
            "expected '_format_version' key in saved npz"
        assert int(data['_format_version']) == 2


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
    """npz with _format_version != 2 must be rejected."""
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
