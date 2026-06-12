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
