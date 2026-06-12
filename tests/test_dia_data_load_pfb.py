"""Tests for DIAData._load_from_pfb."""
import os

import numpy as np
import pytest

from spectrum.dia_data import DIAData
from tests.pfb_test_helpers import write_pfb

_MS1 = {"scan": 1, "ms_level": 1, "rt": 1.0, "instrument_type": "FTMS",
        "mz": [350.0, 351.0], "intensity": [10.0, 20.0]}
_MS2A = {"scan": 2, "ms_level": 2, "rt": 1.1, "instrument_type": "FTMS",
         "charge": 2, "mh_plus": 1000.5, "ion_injection_time": 63.0,
         "activation_center": 501.0, "activation_type": "HCD",
         "precursor_scan": 1, "activation_window": 2.0, "nce": 27.0,
         "monoisotopic_mz": 501.0, "mz": [100.0, 101.0, 102.0],
         "intensity": [5.0, 6.0, 7.0]}
_MS2B = {"scan": 3, "ms_level": 2, "rt": 1.2, "instrument_type": "FTMS",
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
