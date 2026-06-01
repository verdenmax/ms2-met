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
    # Manually build a "legacy" npz with no version key.
    payload = {
        'has_mobility': d.has_mobility,
        'has_ms1': d.has_ms1,
        '_max_mz_value': d._max_mz_value,
        '_min_mz_value': d._min_mz_value,
        '_zeroth_frame': d._zeroth_frame,
        '_scan_max_index': d._scan_max_index,
        'frame_max_index': d.frame_max_index,
        'ms1_indexs': d.ms1_indexs,
        'ms1_indexs_rt': d.ms1_indexs_rt,
        'ms2_indexs': d.ms2_indexs,
        'ms2_indexs_rt': d.ms2_indexs_rt,
        'precursor_scan_ids': d.precursor_scan_ids,
        '_mz_values': d._mz_values,
        'rt_values': d.rt_values,
        '_intensity_values': d._intensity_values,
        'mobility_values': d.mobility_values,
        '_cycle_left_precursor': d._cycle_left_precursor,
        '_scan_id_to_index': d._scan_id_to_index,
        '_peak_start_idx_list': d._peak_start_idx_list,
        '_peak_stop_idx_list': d._peak_stop_idx_list,
        '_precursor_lower_mz': d._precursor_lower_mz,
        '_precursor_upper_mz': d._precursor_upper_mz,
    }
    np.savez_compressed(str(out), **payload)

    with pytest.raises(ValueError, match=r"_format_version"):
        DIAData.load_from_file(str(out), use_mmap=False)


def test_load_rejects_wrong_format_version(tmp_path):
    """npz with _format_version != 2 must be rejected."""
    d = _make_minimal_dia_for_save()
    out = tmp_path / "wrong.dia.npz"
    payload = {
        '_format_version': np.int32(99),
        'has_mobility': d.has_mobility,
        'has_ms1': d.has_ms1,
        '_max_mz_value': d._max_mz_value,
        '_min_mz_value': d._min_mz_value,
        '_zeroth_frame': d._zeroth_frame,
        '_scan_max_index': d._scan_max_index,
        'frame_max_index': d.frame_max_index,
        'ms1_indexs': d.ms1_indexs,
        'ms1_indexs_rt': d.ms1_indexs_rt,
        'ms2_indexs': d.ms2_indexs,
        'ms2_indexs_rt': d.ms2_indexs_rt,
        'precursor_scan_ids': d.precursor_scan_ids,
        '_mz_values': d._mz_values,
        'rt_values': d.rt_values,
        '_intensity_values': d._intensity_values,
        'mobility_values': d.mobility_values,
        '_cycle_left_precursor': d._cycle_left_precursor,
        '_scan_id_to_index': d._scan_id_to_index,
        '_peak_start_idx_list': d._peak_start_idx_list,
        '_peak_stop_idx_list': d._peak_stop_idx_list,
        '_precursor_lower_mz': d._precursor_lower_mz,
        '_precursor_upper_mz': d._precursor_upper_mz,
    }
    np.savez_compressed(str(out), **payload)

    with pytest.raises(ValueError, match=r"_format_version"):
        DIAData.load_from_file(str(out), use_mmap=False)
