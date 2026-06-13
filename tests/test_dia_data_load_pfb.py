"""Tests for DIAData._load_from_pfb."""
import os

import numpy as np
import pytest

from spectrum.dia_data import DIAData
from tests.pfb_test_helpers import write_pfb

_MS1 = {"scan": 1, "ms_level": 1, "rt": 60.0, "instrument_type": "FTMS",
        "mz": [350.0, 351.0], "intensity": [10.0, 20.0]}
_MS2A = {"scan": 2, "ms_level": 2, "rt": 66.0, "instrument_type": "FTMS",
         "charge": 2, "mh_plus": 1000.5, "ion_injection_time": 63.0,
         "activation_center": 501.0, "activation_type": "HCD",
         "precursor_scan": 1, "activation_window": 2.0, "nce": 27.0,
         "monoisotopic_mz": 501.0, "mz": [100.0, 101.0, 102.0],
         "intensity": [5.0, 6.0, 7.0]}
_MS2B = {"scan": 3, "ms_level": 2, "rt": 72.0, "instrument_type": "FTMS",
         "charge": 3, "mh_plus": 1500.0, "ion_injection_time": 50.0,
         "activation_center": 503.0, "activation_type": "HCD",
         "precursor_scan": 1, "activation_window": 2.0, "nce": 27.0,
         "monoisotopic_mz": 503.0, "mz": [200.0], "intensity": [9.0]}


def test_load_from_pfb_builds_equivalent_arrays(tmp_path):
    p = tmp_path / "x.pfb"
    write_pfb(str(p), [_MS1, _MS2A, _MS2B])
    d = DIAData()
    d._load_from_pfb(str(p))

    np.testing.assert_allclose(
        d._mz_values, [350.0, 351.0, 100.0, 101.0, 102.0, 200.0])
    np.testing.assert_allclose(
        d._intensity_values, [10.0, 20.0, 5.0, 6.0, 7.0, 9.0])
    assert d._mz_values.dtype == np.float32

    np.testing.assert_array_equal(d._peak_start_idx_list, [0, 2, 5])
    np.testing.assert_array_equal(d._peak_stop_idx_list, [2, 5, 6])

    np.testing.assert_array_equal(d.precursor_scan_ids, [-1, 1, 1])
    np.testing.assert_array_equal(d.ms1_indexs, [0])
    np.testing.assert_array_equal(d.ms2_indexs, [1, 2])

    # PFB RT 是秒；DIAData 规范单位是分钟 → 60/66/72s 应转成 1.0/1.1/1.2min
    np.testing.assert_allclose(d.rt_values, [1.0, 1.1, 1.2], rtol=1e-6)

    assert np.isnan(d._precursor_lower_mz[0])
    np.testing.assert_allclose(d._precursor_lower_mz[1:], [500.0, 502.0])
    np.testing.assert_allclose(d._precursor_upper_mz[1:], [502.0, 504.0])
    assert float(d._min_mz_value) == pytest.approx(500.0)
    assert float(d._max_mz_value) == pytest.approx(504.0)

    assert d._scan_id_to_index[1] == 0
    assert d._scan_id_to_index[2] == 1
    assert d._scan_id_to_index[3] == 2
    assert d.has_ms1 is True


def test_load_from_pfb_empty_file(tmp_path):
    p = tmp_path / "empty.pfb"
    write_pfb(str(p), [])
    d = DIAData()
    d._load_from_pfb(str(p))
    assert len(d._mz_values) == 0
    assert len(d.ms1_indexs) == 0
    assert len(d.ms2_indexs) == 0


def test_load_from_pfb_converts_rt_seconds_to_minutes(tmp_path):
    """PFB RT is in seconds; DIAData canonical unit is minutes (matches the
    mzML path via _get_retention_time). The loader must divide by 60 so RT-
    based XIC searchsorted lookups stay consistent across raw formats."""
    ms1 = {"scan": 1, "ms_level": 1, "rt": 7200.0, "instrument_type": "FTMS",
           "mz": [400.0], "intensity": [1.0]}
    ms2 = {"scan": 2, "ms_level": 2, "rt": 90.0, "instrument_type": "FTMS",
           "charge": 2, "mh_plus": 800.0, "ion_injection_time": 50.0,
           "activation_center": 500.0, "activation_type": "HCD",
           "precursor_scan": 1, "activation_window": 4.0, "nce": 27.0,
           "monoisotopic_mz": 500.0, "mz": [123.0], "intensity": [2.0]}
    p = tmp_path / "rt.pfb"
    write_pfb(str(p), [ms1, ms2])
    d = DIAData()
    d._load_from_pfb(str(p))
    # 7200s → 120.0 min, 90s → 1.5 min
    np.testing.assert_allclose(d.rt_values, [120.0, 1.5], rtol=1e-6)
    np.testing.assert_allclose(d.ms1_indexs_rt, [120.0], rtol=1e-6)
    np.testing.assert_allclose(d.ms2_indexs_rt, [1.5], rtol=1e-6)


def test_get_dia_data_object_dispatches_by_extension(monkeypatch):
    from manager.data_manager import DataManager

    called = {}

    def fake_pfb(self, path):
        called["pfb"] = path

    def fake_mzml(self, path):
        called["mzml"] = path

    monkeypatch.setattr(DIAData, "_load_from_pfb", fake_pfb)
    monkeypatch.setattr(DIAData, "_load_from_mzml", fake_mzml)

    dm = DataManager(config=None, path=None)
    dm.get_dia_data_object("/tmp/sample.pfb")
    assert called == {"pfb": "/tmp/sample.pfb"}

    called.clear()
    dm.get_dia_data_object("/tmp/sample.mzML")
    assert called == {"mzml": "/tmp/sample.mzML"}


_REAL_PFB = os.path.expanduser(
    "~/share/2026_04_27_kongweisa_diann_ZHOUHUdataset/2th/"
    "20190830_HF_ZHW_hela_SILAC_DDIA_500_550_2Da_Rep1.pfb")


@pytest.mark.skipif(not os.path.exists(_REAL_PFB),
                    reason="real .pfb sample not available")
def test_real_pfb_header_and_first_spectra():
    from spectrum import pfb_reader
    with open(_REAL_PFB, "rb") as fh:
        addr_list_addr, scan_num = pfb_reader.read_header(fh)
        assert scan_num == 80096
        specs = []
        for s in pfb_reader.iter_spectra(fh, scan_num):
            specs.append(s)
            if len(specs) >= 2:
                break
        footer = pfb_reader.read_footer(fh, addr_list_addr, scan_num)
    s1, s2 = specs
    assert s1.ms_level == 1
    assert s1.rt == pytest.approx(0.1972939, rel=1e-4)
    assert s2.ms_level == 2
    assert s2.activation_center == pytest.approx(501.0)
    assert s2.activation_window == pytest.approx(2.0)
    assert footer[0] == pfb_reader.HEADER_SIZE


def test_get_dia_data_object_dispatch_uppercase_and_none(monkeypatch):
    from manager.data_manager import DataManager

    called = {}
    monkeypatch.setattr(DIAData, "_load_from_pfb",
                        lambda self, path: called.__setitem__("pfb", path))
    monkeypatch.setattr(DIAData, "_load_from_mzml",
                        lambda self, path: called.__setitem__("mzml", path))
    dm = DataManager(config=None, path=None)

    # uppercase extension routes to PFB (case-insensitive)
    dm.get_dia_data_object("/tmp/SAMPLE.PFB")
    assert called == {"pfb": "/tmp/SAMPLE.PFB"}

    # None path is safe and falls through to mzML (no crash)
    called.clear()
    dm.get_dia_data_object(None)
    assert called == {"mzml": None}


def test_pfb_and_mzml_load_equivalent_arrays(monkeypatch, tmp_path):
    """Strongest drop-in guarantee: the SAME logical spectra loaded as mzML
    vs PFB must yield array-identical DIAData. mzML RT is minutes (plain float
    treated as minutes); PFB RT is seconds (=minutes*60) and /60 on load, so
    both rt_values land on the same minutes."""
    from tests.test_dia_data_load_mzml import _FakeMzmlReader, _make_spectrum
    from spectrum import dia_data as dd

    # --- mzML side (rt in minutes; window = selected_ion_mz +/- offset) ---
    mzml_spectra = [
        _make_spectrum(scan_num=1, ms_level=1, rt=1.0,
                       mz_arr=[400.0, 401.0, 402.0], int_arr=[10.0, 20.0, 30.0]),
        _make_spectrum(scan_num=2, ms_level=2, rt=1.05,
                       mz_arr=[200.0, 201.0], int_arr=[5.0, 15.0],
                       precursor_scan_num=1, precursor_mz=500.0,
                       iso_lower_off=1.0, iso_upper_off=1.0),
        _make_spectrum(scan_num=3, ms_level=2, rt=1.10,
                       mz_arr=[300.0, 301.0, 302.0, 303.0],
                       int_arr=[1.0, 2.0, 3.0, 4.0],
                       precursor_scan_num=1, precursor_mz=600.0,
                       iso_lower_off=2.0, iso_upper_off=2.0),
    ]
    monkeypatch.setattr(dd.mzml, 'read', lambda p: _FakeMzmlReader(mzml_spectra))
    d_mzml = DIAData()
    d_mzml._centroid_enabled = False
    d_mzml._load_from_mzml('fake.mzML')

    # --- PFB side (rt in seconds = minutes*60; window = center +/- window/2) ---
    pfb_spectra = [
        {"scan": 1, "ms_level": 1, "rt": 60.0, "instrument_type": "FTMS",
         "mz": [400.0, 401.0, 402.0], "intensity": [10.0, 20.0, 30.0]},
        {"scan": 2, "ms_level": 2, "rt": 63.0, "instrument_type": "FTMS",
         "charge": 2, "mh_plus": 999.0, "ion_injection_time": 50.0,
         "activation_center": 500.0, "activation_type": "HCD",
         "precursor_scan": 1, "activation_window": 2.0, "nce": 27.0,
         "monoisotopic_mz": 500.0, "mz": [200.0, 201.0], "intensity": [5.0, 15.0]},
        {"scan": 3, "ms_level": 2, "rt": 66.0, "instrument_type": "FTMS",
         "charge": 2, "mh_plus": 1199.0, "ion_injection_time": 50.0,
         "activation_center": 600.0, "activation_type": "HCD",
         "precursor_scan": 1, "activation_window": 4.0, "nce": 27.0,
         "monoisotopic_mz": 600.0, "mz": [300.0, 301.0, 302.0, 303.0],
         "intensity": [1.0, 2.0, 3.0, 4.0]},
    ]
    pfb_path = str(tmp_path / "eq.pfb")
    write_pfb(pfb_path, pfb_spectra)
    d_pfb = DIAData()
    d_pfb._load_from_pfb(pfb_path)

    # --- assert full array parity ---
    np.testing.assert_allclose(d_pfb._mz_values, d_mzml._mz_values)
    np.testing.assert_allclose(d_pfb._intensity_values, d_mzml._intensity_values)
    np.testing.assert_array_equal(d_pfb._peak_start_idx_list, d_mzml._peak_start_idx_list)
    np.testing.assert_array_equal(d_pfb._peak_stop_idx_list, d_mzml._peak_stop_idx_list)
    np.testing.assert_array_equal(d_pfb.precursor_scan_ids, d_mzml.precursor_scan_ids)
    np.testing.assert_array_equal(d_pfb.ms1_indexs, d_mzml.ms1_indexs)
    np.testing.assert_array_equal(d_pfb.ms2_indexs, d_mzml.ms2_indexs)
    np.testing.assert_allclose(d_pfb.rt_values, d_mzml.rt_values)
    np.testing.assert_allclose(d_pfb.ms1_indexs_rt, d_mzml.ms1_indexs_rt)
    np.testing.assert_allclose(d_pfb.ms2_indexs_rt, d_mzml.ms2_indexs_rt)
    np.testing.assert_allclose(d_pfb._precursor_lower_mz, d_mzml._precursor_lower_mz, equal_nan=True)
    np.testing.assert_allclose(d_pfb._precursor_upper_mz, d_mzml._precursor_upper_mz, equal_nan=True)
    assert float(d_pfb._min_mz_value) == pytest.approx(float(d_mzml._min_mz_value))
    assert float(d_pfb._max_mz_value) == pytest.approx(float(d_mzml._max_mz_value))
    np.testing.assert_array_equal(d_pfb._scan_id_to_index, d_mzml._scan_id_to_index)
    assert d_pfb.has_ms1 == d_mzml.has_ms1
