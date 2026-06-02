"""End-to-end functional tests for the mzML on-load centroiding feature.

These tests exercise the full chain: config.ini → DataManager →
DIAData._load_from_mzml → centroid_spectrum → save_to_file (npz) →
load_from_file. Unlike the per-task unit tests in
`test_centroid_spectrum.py` and `test_dia_data_load_mzml.py`, the
tests here intentionally do NOT mock any link — they verify that the
real pieces compose correctly on a real mzML.

Tests are skipped if the real test mzML fixture is not present.
"""
import configparser
import os

import numpy as np
import pytest

from manager.data_manager import DataManager
from spectrum.dia_data import DIAData


_REAL_MZML_PATH = os.path.abspath(os.path.join(
    os.path.dirname(__file__), "..", "..",
    "20190830_HF_ZHW_hela_SILAC_DIA_350_1000_Rep1.mzML"))

_HAS_REAL_MZML = os.path.exists(_REAL_MZML_PATH)
_skip_no_mzml = pytest.mark.skipif(
    not _HAS_REAL_MZML,
    reason=f"real mzML fixture not present at {_REAL_MZML_PATH}",
)


def _make_config(**general_overrides):
    """Build a minimal ConfigParser with overridable [general] section."""
    cfg = configparser.ConfigParser()
    cfg['general'] = {str(k): str(v) for k, v in general_overrides.items()}
    return cfg


# ---- Inlined helper (copy of _make_minimal_dia_for_save from
#      tests/test_dia_data_load_mzml.py; kept inline so this file is
#      self-contained per the plan). ----
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
    d._quad_max_mz_value = None
    d._quad_min_mz_value = None
    return d


def _sum_first_n_raw_samples(mzml_path, n):
    """Read first n spectra via pyteomics; return total profile sample count."""
    from pyteomics import mzml
    total = 0
    with mzml.read(mzml_path) as reader:
        for i, spec in enumerate(reader):
            if i >= n:
                break
            total += len(spec['m/z array'])
    return total


# ---------------------------------------------------------------------------
# Test 1: explicit centroid path on real mzML
# ---------------------------------------------------------------------------
@_skip_no_mzml
def test_e2e_config_drives_centroid_enabled_real_mzml():
    """config.ini[general] centroid_enabled=true must propagate to DIAData
    and the real mzML load must produce strictly fewer peaks than raw
    profile samples (centroiding actually fired)."""
    cfg = _make_config(centroid_enabled=True,
                       centroid_rel_threshold=0.001)
    dm = DataManager(config=cfg)
    dia = dm.get_dia_data_object(_REAL_MZML_PATH)

    # Config flowed through.
    assert dia._centroid_enabled is True
    assert dia._centroid_rel_threshold == pytest.approx(0.001)

    # Centroiding actually ran: compare first 10 spectra (using the
    # _peak_stop_idx_list cumulative count) against raw profile sample sums.
    n = min(10, len(dia._peak_stop_idx_list))
    raw_samples = _sum_first_n_raw_samples(_REAL_MZML_PATH, n)
    centroided_peaks = int(dia._peak_stop_idx_list[n - 1])
    assert raw_samples > 0, "real mzML produced 0 raw samples in first 10 spectra"
    assert centroided_peaks < 0.30 * raw_samples, (
        f"centroid_enabled produced {centroided_peaks} peaks for the first "
        f"{n} spectra vs {raw_samples} raw samples — expected <30% "
        f"(centroiding may not have fired)")

    # ms1/ms2 indices populated and disjoint.
    assert len(dia.ms1_indexs) > 0
    assert len(dia.ms2_indexs) > 0
    assert set(dia.ms1_indexs.tolist()).isdisjoint(
        set(dia.ms2_indexs.tolist())), \
        "ms1_indexs and ms2_indexs must be disjoint"

    # Numeric sanity.
    assert np.all(np.isfinite(dia._mz_values))
    assert np.all(np.isfinite(dia._intensity_values))

    # End-to-end index bookkeeping.
    assert int(dia._peak_stop_idx_list[-1]) == len(dia._mz_values)


# ---------------------------------------------------------------------------
# Test 2: disable path preserves profile peaks.
# DECISION: loading the full 165 MB profile mzML with centroid OFF would
# concatenate ~all raw samples and double-allocate during np.concatenate
# (see _load_from_mzml memory-tradeoff comment), easily multi-GB. To keep
# the test fast and bounded we use a synthetic 3-spectrum fake reader —
# the e2e theme remains: real DataManager → real DIAData path with
# _centroid_enabled=False, just with synthesized peaks.
# ---------------------------------------------------------------------------
def test_e2e_config_disable_path_preserves_profile_peaks(monkeypatch):
    """With centroid_enabled=false, the cumulative peak count through the
    Nth spectrum equals the SUM of raw `m/z array` lengths for those
    spectra (no centroiding mutated the peaks).

    Uses a synthetic 3-spectrum fake reader because loading the full
    real profile mzML with centroid disabled is memory-heavy (peaks
    are not compressed; concat doubles peak memory during build).
    """
    # Local copies of fake-reader helpers (kept self-contained).
    class _FakeMzmlReader:
        def __init__(self, spectra):
            self._spectra = spectra

        def __enter__(self):
            return iter(self._spectra)

        def __exit__(self, *a):
            return False

    def _make_spectrum(scan_num, ms_level, rt, mz_arr, int_arr,
                      precursor_scan_num=None, precursor_mz=None):
        s = {
            'id': f'controllerType=0 controllerNumber=1 scan={scan_num}',
            'spectrum title': f'spec{scan_num} dummy',
            'ms level': ms_level,
            'm/z array': np.asarray(mz_arr, dtype=np.float32),
            'intensity array': np.asarray(int_arr, dtype=np.float32),
            'scanList': {'scan': [{'scan start time': float(rt)}]},
        }
        if ms_level > 1:
            s['precursorList'] = {
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
                        'isolation window lower offset': 1.0,
                        'isolation window upper offset': 1.0,
                    },
                }],
            }
        return s

    spectra = [
        _make_spectrum(1, 1, 1.0,
                       [400.0, 400.5, 401.0, 401.5, 402.0],
                       [1.0, 2.0, 3.0, 2.0, 1.0]),
        _make_spectrum(2, 2, 1.05,
                       [200.0, 201.0, 202.0],
                       [5.0, 10.0, 5.0],
                       precursor_scan_num=1, precursor_mz=500.0),
        _make_spectrum(3, 2, 1.10,
                       [300.0, 300.5, 301.0, 301.5],
                       [1.0, 4.0, 4.0, 1.0],
                       precursor_scan_num=1, precursor_mz=600.0),
    ]
    expected_first_n = 5  # spectrum 0 has 5 samples; we'll check stop[0]
    # ...but we also confirm cumulative count through ALL 3 equals 12.
    expected_total = 5 + 3 + 4

    from spectrum import dia_data as dd
    monkeypatch.setattr(dd.mzml, 'read',
                        lambda p: _FakeMzmlReader(spectra))

    cfg = _make_config(centroid_enabled=False)
    dm = DataManager(config=cfg)
    dia = dm.get_dia_data_object('fake.mzML')

    assert dia._centroid_enabled is False
    # Cumulative through spectrum 0 == raw len of spectrum 0
    assert int(dia._peak_stop_idx_list[0]) == expected_first_n
    # Full cumulative count matches sum of raw m/z lengths
    assert int(dia._peak_stop_idx_list[-1]) == expected_total
    assert len(dia._mz_values) == expected_total


# ---------------------------------------------------------------------------
# Test 3: real save/load roundtrip after real centroid load
# ---------------------------------------------------------------------------
@_skip_no_mzml
def test_e2e_save_load_roundtrip_after_real_centroid_load(tmp_path):
    """Real mzML → centroid load → save_to_file → np.load shows
    _format_version=2 → DIAData.load_from_file returns byte-identical
    arrays and scalars."""
    cfg = _make_config(centroid_enabled=True)  # default threshold = 1e-3
    dm = DataManager(config=cfg)
    dia = dm.get_dia_data_object(_REAL_MZML_PATH)

    out = tmp_path / 'roundtrip.dia.npz'
    dia.save_to_file(str(out))

    # Format version persisted.
    with np.load(str(out)) as data:
        assert '_format_version' in data
        assert int(data['_format_version']) == 2

    reloaded = DIAData.load_from_file(str(out), use_mmap=False)

    # Array bit-for-bit equality.
    np.testing.assert_array_equal(reloaded._mz_values, dia._mz_values)
    np.testing.assert_array_equal(
        reloaded._intensity_values, dia._intensity_values)
    np.testing.assert_array_equal(reloaded.ms1_indexs, dia.ms1_indexs)
    np.testing.assert_array_equal(reloaded.ms2_indexs, dia.ms2_indexs)
    np.testing.assert_array_equal(
        reloaded._peak_start_idx_list, dia._peak_start_idx_list)
    np.testing.assert_array_equal(
        reloaded._peak_stop_idx_list, dia._peak_stop_idx_list)

    # Scalar fields.
    def _eq_or_both_nan(a, b):
        if a is None and b is None:
            return True
        try:
            if np.isnan(a) and np.isnan(b):
                return True
        except (TypeError, ValueError):
            pass
        return a == b

    assert _eq_or_both_nan(reloaded._max_mz_value, dia._max_mz_value)
    assert _eq_or_both_nan(reloaded._min_mz_value, dia._min_mz_value)
    assert reloaded._zeroth_frame == dia._zeroth_frame
    assert reloaded._scan_max_index == dia._scan_max_index
    assert reloaded.frame_max_index == dia.frame_max_index
    assert reloaded.has_mobility == dia.has_mobility
    assert reloaded.has_ms1 == dia.has_ms1


# ---------------------------------------------------------------------------
# Test 4: old (pre-T5) cache rejected by full pipeline (no mzML needed)
# ---------------------------------------------------------------------------
def test_e2e_old_cache_rejected_via_full_pipeline(tmp_path):
    """An npz cache lacking `_format_version` (legacy pre-T5 profile-peaks
    format) must be rejected by DIAData.load_from_file with a ValueError
    naming the file path so the user knows what to delete."""
    d = _make_minimal_dia_for_save()
    out = tmp_path / 'legacy.dia.npz'

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
        # NOTE: intentionally NO `_format_version` key.
    }
    np.savez_compressed(str(out), **payload)

    with pytest.raises(ValueError) as exc_info:
        DIAData.load_from_file(str(out), use_mmap=False)

    # Error must name the file path so the user can find/delete it.
    assert str(out) in str(exc_info.value), (
        f"ValueError should include the npz path; got: {exc_info.value!r}")
