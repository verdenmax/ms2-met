"""Helpers for building synthetic .pfb files in tests."""
import struct
import numpy as np

_HEADER_SIZE = 24


def make_property_str(spec: dict) -> str:
    """Build a tab-separated property_str from a spec dict (MS1 or MS2)."""
    ms_level = spec["ms_level"]
    parts = [str(spec["scan"]), str(ms_level), str(spec["rt"]),
             spec["instrument_type"]]
    if ms_level == 2:
        parts += [
            str(spec["charge"]),
            str(spec["mh_plus"]),
            str(spec["ion_injection_time"]),
            str(spec["activation_center"]),
            spec["activation_type"],
            str(spec["precursor_scan"]),
            str(spec["activation_window"]),
            str(spec["nce"]),
            str(spec["monoisotopic_mz"]),
        ]
    return "\t".join(parts)


def write_pfb(path, spectra, empties=(0, 0, 0)):
    """Write a synthetic .pfb file. Returns the footer addr_list (offsets).

    Each spec dict: scan, ms_level, rt, instrument_type, mz(list), intensity(list).
    MS2 adds: charge, mh_plus, ion_injection_time, activation_center,
    activation_type, precursor_scan, activation_window, nce, monoisotopic_mz.
    """
    addr_list = []
    body = bytearray()
    for spec in spectra:
        addr_list.append(_HEADER_SIZE + len(body))
        pstr = make_property_str(spec).encode("utf-8")
        body += struct.pack("<i", len(pstr))
        body += pstr
        mz = np.asarray(spec["mz"], dtype="<f8")
        inten = np.asarray(spec["intensity"], dtype="<f8")
        assert len(mz) == len(inten)
        body += struct.pack("<i", len(mz))
        body += mz.tobytes()
        body += inten.tobytes()
    addr_list_addr = _HEADER_SIZE + len(body)
    with open(path, "wb") as f:
        f.write(struct.pack("<iiiqi", empties[0], empties[1], empties[2],
                            addr_list_addr, len(spectra)))
        f.write(body)
        if spectra:
            f.write(struct.pack(f"<{len(spectra)}q", *addr_list))
    return addr_list
