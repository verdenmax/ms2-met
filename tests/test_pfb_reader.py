"""Tests for spectrum.pfb_reader."""
import struct

import numpy as np
import pytest

from spectrum import pfb_reader
from tests.pfb_test_helpers import write_pfb

_MS1 = {"scan": 1, "ms_level": 1, "rt": 1.5, "instrument_type": "FTMS",
        "mz": [350.0, 351.0], "intensity": [10.0, 20.0]}
_MS2 = {"scan": 2, "ms_level": 2, "rt": 2.0, "instrument_type": "FTMS",
        "charge": 2, "mh_plus": 1000.5, "ion_injection_time": 63.0,
        "activation_center": 501.0, "activation_type": "HCD",
        "precursor_scan": 1, "activation_window": 2.0, "nce": 27.0,
        "monoisotopic_mz": 501.0, "mz": [100.0, 101.0, 102.0],
        "intensity": [5.0, 6.0, 7.0]}


def test_read_header_returns_addr_and_scan_num(tmp_path):
    p = tmp_path / "x.pfb"
    addr_list = write_pfb(str(p), [_MS1, _MS2])
    with open(p, "rb") as fh:
        addr_list_addr, scan_num = pfb_reader.read_header(fh)
        assert scan_num == 2
        assert addr_list_addr > addr_list[-1]  # footer starts after last spectrum
        # header is exactly 24 bytes -> first spectrum at offset 24
        assert pfb_reader.HEADER_SIZE == 24
        assert fh.tell() == 24
        assert addr_list[0] == 24


def test_parse_property_str_ms1():
    out = pfb_reader.parse_property_str("1\t1\t0.197\tFTMS")
    assert out == {"scan": 1, "ms_level": 1, "rt": 0.197,
                   "instrument_type": "FTMS"}


def test_parse_property_str_ms2():
    s = "2\t2\t0.4538569\tFTMS\t2\t1000.993\t63\t501\tHCD\t1\t2\t27.00\t501"
    out = pfb_reader.parse_property_str(s)
    assert out["scan"] == 2 and out["ms_level"] == 2
    assert out["instrument_type"] == "FTMS"
    assert out["charge"] == 2
    assert out["activation_center"] == 501.0
    assert out["precursor_scan"] == 1
    assert out["activation_window"] == 2.0
    assert out["nce"] == 27.0
    assert out["monoisotopic_mz"] == 501.0


def test_parse_property_str_ms2_wrong_field_count_raises():
    # MS2 with only 11 fields (missing pXtract-3 fields) -> clear error
    s = "2\t2\t0.45\tFTMS\t2\t1000.9\t63\t501\tHCD\t1\t2"
    with pytest.raises(ValueError, match="MS2"):
        pfb_reader.parse_property_str(s)


def test_iter_spectra_yields_ms1_and_ms2(tmp_path):
    p = tmp_path / "x.pfb"
    write_pfb(str(p), [_MS1, _MS2])
    with open(p, "rb") as fh:
        _addr, scan_num = pfb_reader.read_header(fh)
        specs = list(pfb_reader.iter_spectra(fh, scan_num))
    assert len(specs) == 2
    s1, s2 = specs
    assert s1.ms_level == 1 and s1.scan == 1 and s1.rt == 1.5
    np.testing.assert_allclose(s1.mz, [350.0, 351.0])
    np.testing.assert_allclose(s1.intensity, [10.0, 20.0])
    assert s1.charge is None
    assert s2.ms_level == 2 and s2.precursor_scan == 1
    assert s2.activation_center == 501.0 and s2.activation_window == 2.0
    np.testing.assert_allclose(s2.mz, [100.0, 101.0, 102.0])
    np.testing.assert_allclose(s2.intensity, [5.0, 6.0, 7.0])
    assert s2.mz.dtype == np.float64
    assert s2.intensity.dtype == np.float64


def test_iter_spectra_empty_file(tmp_path):
    p = tmp_path / "empty.pfb"
    write_pfb(str(p), [])
    with open(p, "rb") as fh:
        _addr, scan_num = pfb_reader.read_header(fh)
        assert scan_num == 0
        assert list(pfb_reader.iter_spectra(fh, scan_num)) == []


def test_iter_spectra_truncated_raises(tmp_path):
    p = tmp_path / "trunc.pfb"
    write_pfb(str(p), [_MS1, _MS2])
    # Truncate the file mid-body
    full = p.read_bytes()
    p.write_bytes(full[:30])
    with open(p, "rb") as fh:
        _addr, scan_num = pfb_reader.read_header(fh)
        with pytest.raises(ValueError, match="truncated"):
            list(pfb_reader.iter_spectra(fh, scan_num))


def test_iter_spectra_empty_peak_spectrum_stays_aligned(tmp_path):
    """A spectrum with zero peaks parses as empty float64 arrays and does not
    desync the reader for the following spectrum."""
    empty_ms2 = {"scan": 5, "ms_level": 2, "rt": 3.0, "instrument_type": "FTMS",
                 "charge": 2, "mh_plus": 800.0, "ion_injection_time": 50.0,
                 "activation_center": 600.0, "activation_type": "HCD",
                 "precursor_scan": 1, "activation_window": 2.0, "nce": 27.0,
                 "monoisotopic_mz": 600.0, "mz": [], "intensity": []}
    p = tmp_path / "empty_peaks.pfb"
    write_pfb(str(p), [empty_ms2, _MS1])
    with open(p, "rb") as fh:
        _addr, scan_num = pfb_reader.read_header(fh)
        specs = list(pfb_reader.iter_spectra(fh, scan_num))
    assert len(specs) == 2
    s0, s1 = specs
    assert s0.scan == 5 and len(s0.mz) == 0 and len(s0.intensity) == 0
    assert s0.mz.dtype == np.float64 and s0.intensity.dtype == np.float64
    # reader stayed byte-aligned: next spectrum parsed correctly
    assert s1.scan == 1 and s1.ms_level == 1
    np.testing.assert_allclose(s1.mz, [350.0, 351.0])


def test_iter_spectra_negative_length_raises(tmp_path):
    """A corrupt negative property_str_len must raise (not silently read rest)."""
    p = tmp_path / "neg.pfb"
    write_pfb(str(p), [_MS1, _MS2])
    raw = bytearray(p.read_bytes())
    # property_str_len of the first spectrum is the int32 at offset 24 (HEADER_SIZE)
    struct.pack_into("<i", raw, 24, -1)
    p.write_bytes(raw)
    with open(p, "rb") as fh:
        _addr, scan_num = pfb_reader.read_header(fh)
        with pytest.raises(ValueError):
            list(pfb_reader.iter_spectra(fh, scan_num))
