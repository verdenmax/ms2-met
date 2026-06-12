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


def parse_property_str(s: str) -> dict:
    """Parse a tab-separated property string into typed fields.

    Layout decided by token[1] (MsType): MS1 -> 4 tokens, MS2 -> 13 tokens.
    """
    toks = s.split("\t")
    if len(toks) < _MS1_FIELD_COUNT:
        raise ValueError(f"PFB property_str has too few fields: {toks!r}")
    ms_level = int(toks[1])
    base = {
        "scan": int(toks[0]),
        "ms_level": ms_level,
        "rt": float(toks[2]),
        "instrument_type": toks[3],
    }
    if ms_level == 1:
        if len(toks) != _MS1_FIELD_COUNT:
            raise ValueError(
                f"MS1 property_str expects {_MS1_FIELD_COUNT} fields, "
                f"got {len(toks)}: {toks!r}")
        return base
    if ms_level == 2:
        if len(toks) != _MS2_FIELD_COUNT:
            raise ValueError(
                f"MS2 property_str expects {_MS2_FIELD_COUNT} fields, "
                f"got {len(toks)}: {toks!r}")
        base.update({
            "charge": int(toks[4]),
            "mh_plus": float(toks[5]),
            "ion_injection_time": float(toks[6]),
            "activation_center": float(toks[7]),
            "activation_type": toks[8],
            "precursor_scan": int(toks[9]),
            "activation_window": float(toks[10]),
            "nce": float(toks[11]),
            "monoisotopic_mz": float(toks[12]),
        })
        return base
    raise ValueError(f"Unknown MsType={ms_level} in property_str: {toks!r}")
