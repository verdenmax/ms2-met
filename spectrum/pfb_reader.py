"""PFB (pFind/pXtract binary spectrum) format reader.

Pure parsing: reads the binary structure into typed per-spectrum records.
No numpy-array-building / DIAData knowledge lives here.

Format (little-endian), verified against real samples:
  Header (24 bytes): 3xint32 (reserved) + int64 addr_list_addr + int32 scan_num
  Loop body x scan_num:
    int32 property_str_len
    char[property_str_len]  property_str (UTF-8, '\t'-separated, may end \x00)
    int32 peak_num
    float64[peak_num]  mz
    float64[peak_num]  intensity
  Footer: int64[scan_num]  addr_list (per-spectrum file offsets)
"""
from __future__ import annotations

import struct
from dataclasses import dataclass
from typing import BinaryIO, Iterator

import numpy as np

_HEADER_STRUCT = struct.Struct("<iiiqi")
HEADER_SIZE = _HEADER_STRUCT.size  # 24

_MS1_FIELD_COUNT = 4
_MS2_FIELD_COUNT = 13


@dataclass
class PFBSpectrum:
    scan: int
    ms_level: int
    rt: float
    instrument_type: str
    mz: np.ndarray
    intensity: np.ndarray
    charge: int | None = None
    mh_plus: float | None = None
    ion_injection_time: float | None = None
    activation_center: float | None = None
    activation_type: str | None = None
    precursor_scan: int | None = None
    activation_window: float | None = None
    nce: float | None = None
    monoisotopic_mz: float | None = None


def read_header(fh: BinaryIO) -> tuple[int, int]:
    """Read the 24-byte header. Returns (addr_list_addr, scan_num).

    Leaves the file positioned at the first spectrum (offset 24).
    """
    raw = fh.read(HEADER_SIZE)
    if len(raw) < HEADER_SIZE:
        raise ValueError(
            f"PFB header truncated: expected {HEADER_SIZE} bytes, "
            f"got {len(raw)}")
    _e1, _e2, _e3, addr_list_addr, scan_num = _HEADER_STRUCT.unpack(raw)
    return addr_list_addr, scan_num
