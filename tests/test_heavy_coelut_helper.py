"""Tests for heavy_coelut_at_light_apex (single_work.py) — the cycle-alignment
core of the light<->heavy co-elution feature (spec §13).

Constructs XIC structured arrays directly to exercise the cross-window cycle
alignment, the ±1 tolerance, and the -1 sentinel guard (audit finding C2).
"""
import numpy as np

from workflows.single_work import heavy_coelut_at_light_apex

_DT = [("rt", "f8"), ("ppm_error", "f8"), ("intensity", "f8"),
       ("cycle_idx", "i4")]


def _xic(points):
    """points: list of (cycle_idx, intensity)."""
    a = np.zeros(len(points), dtype=_DT)
    for i, (cyc, inten) in enumerate(points):
        a["cycle_idx"][i] = cyc
        a["intensity"][i] = inten
        a["rt"][i] = float(cyc)
    return a


def test_coeluting_heavy_at_light_apex_is_captured():
    # light apex at cycle 5; heavy real peak at cycle 5 -> captured
    light = _xic([(3, 100), (4, 500), (5, 2000), (6, 400)])
    heavy = _xic([(4, 50), (5, 1500), (6, 80)])
    assert heavy_coelut_at_light_apex(light, heavy) == 1500.0


def test_offpeak_heavy_is_excluded():
    # light apex at cycle 5; heavy ONLY has a big peak at cycle 40 (off-peak)
    light = _xic([(4, 500), (5, 2000), (6, 400)])
    heavy = _xic([(39, 100), (40, 9000), (41, 120)])
    assert heavy_coelut_at_light_apex(light, heavy) == 0.0


def test_plus_minus_one_cycle_tolerance():
    # heavy peak one cycle off (cycle 6 vs light apex 5) -> within ±1, counted
    light = _xic([(5, 2000)])
    heavy = _xic([(6, 800), (40, 9000)])
    assert heavy_coelut_at_light_apex(light, heavy) == 800.0


def test_minus1_sentinel_heavy_not_matched_near_cycle0():
    # C2 regression: light apex at cycle 0; a heavy interference row carries
    # cycle_idx=-1. |−1−0|<=1 would WRONGLY match without the >=0 guard.
    light = _xic([(0, 2000)])
    heavy = _xic([(-1, 9000), (40, 50)])
    assert heavy_coelut_at_light_apex(light, heavy) == 0.0


def test_light_apex_on_minus1_sentinel_returns_zero():
    # C2 regression: if the light apex row itself is the -1 sentinel, bail.
    light = _xic([(-1, 5000)])     # apex row has invalid cycle
    heavy = _xic([(-1, 7000), (0, 6000)])
    assert heavy_coelut_at_light_apex(light, heavy) == 0.0


def test_empty_or_zero_light_returns_zero():
    empty = np.zeros(0, dtype=_DT)
    heavy = _xic([(5, 1000)])
    assert heavy_coelut_at_light_apex(empty, heavy) == 0.0
    assert heavy_coelut_at_light_apex(_xic([(5, 1000)]), empty) == 0.0
    # light present but all-zero intensity -> no apex -> 0
    assert heavy_coelut_at_light_apex(_xic([(5, 0), (6, 0)]), heavy) == 0.0


def test_no_heavy_within_tolerance_returns_zero():
    light = _xic([(5, 2000)])
    heavy = _xic([(8, 1000), (9, 1000)])   # all > 1 cycle away
    assert heavy_coelut_at_light_apex(light, heavy) == 0.0
