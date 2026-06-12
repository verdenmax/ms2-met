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


def _read_exact(fh: BinaryIO, n: int, spec_idx: int, what: str) -> bytes:
    if n < 0:
        raise ValueError(
            f"PFB corrupt: negative byte count {n} for spectrum {spec_idx} "
            f"{what}")
    raw = fh.read(n)
    if len(raw) < n:
        raise ValueError(
            f"PFB truncated reading spectrum {spec_idx} {what}: "
            f"want {n} bytes, got {len(raw)} at offset {fh.tell()}")
    return raw


def iter_spectra(fh: BinaryIO, scan_num: int) -> Iterator[PFBSpectrum]:
    """Sequentially read `scan_num` spectra from the loop body.

    `fh` must be positioned at the first spectrum (call read_header first).
    """
    for i in range(scan_num):
        (slen,) = struct.unpack("<i", _read_exact(fh, 4, i, "property_str_len"))
        prop = _read_exact(fh, slen, i, "property_str").decode(
            "utf-8").rstrip("\x00")
        fields = parse_property_str(prop)
        (pnum,) = struct.unpack("<i", _read_exact(fh, 4, i, "peak_num"))
        if pnum > 0:
            mz = np.frombuffer(
                _read_exact(fh, pnum * 8, i, "mz"), dtype="<f8").astype(
                np.float64)
            intensity = np.frombuffer(
                _read_exact(fh, pnum * 8, i, "intensity"), dtype="<f8").astype(
                np.float64)
        else:
            mz = np.empty(0, dtype=np.float64)
            intensity = np.empty(0, dtype=np.float64)
        yield PFBSpectrum(mz=mz, intensity=intensity, **fields)


def iter_scan_ids(fh: BinaryIO, scan_num: int) -> Iterator[int]:
    """Pass-1: yield each spectrum's scan number, seeking past peak arrays.

    `fh` must be positioned at the first spectrum (call read_header first).
    Does NOT decode peak arrays (cheap two-pass like the mzML loader).
    """
    for i in range(scan_num):
        (slen,) = struct.unpack("<i", _read_exact(fh, 4, i, "property_str_len"))
        prop = _read_exact(fh, slen, i, "property_str").decode(
            "utf-8").rstrip("\x00")
        scan = int(prop.split("\t", 1)[0])
        (pnum,) = struct.unpack("<i", _read_exact(fh, 4, i, "peak_num"))
        fh.seek(pnum * 16, 1)  # skip mz(8) + intensity(8) per peak
        yield scan


def read_footer(fh: BinaryIO, addr_list_addr: int, scan_num: int) -> list[int]:
    """Read the footer addr_list (per-spectrum file offsets). For validation."""
    fh.seek(addr_list_addr)
    raw = fh.read(scan_num * 8)
    if len(raw) < scan_num * 8:
        raise ValueError(
            f"PFB footer truncated: want {scan_num * 8} bytes, got {len(raw)}")
    return list(struct.unpack(f"<{scan_num}q", raw))
