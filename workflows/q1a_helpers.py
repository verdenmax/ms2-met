"""Q1a fragment-pairing helpers.

Implements §4.2 of docs/specs/2026-05-13-silac-validation-framework.md.

Q1a measures, over the *separable* theoretical b/y fragments of a PSM,
how many fragments have BOTH a credible light signal AND a credible
heavy signal at the predicted m/z and rt. Each TP is one piece of
independent physical evidence that the light search-engine call is
correct; each FN is the inverse.

Public surface:
    - is_signal_present_light(xic, intensity_floor) -> bool
    - is_signal_present_heavy(light_xic, heavy_xic, ...) -> bool
    - is_split_window(light_window_info, heavy_window_info) -> bool
    - is_separable_fragment(light_mass, heavy_mass, split_window) -> bool
    - Q1aAccumulator: stateful per-PSM accumulator producing 11 features
"""
from __future__ import annotations

import numpy as np
from scipy.stats import pearsonr


# Three-condition signal-present thresholds (spec §4.2 implementation
# decisions, locked 2026-05-19; defaults may be tuned later via config).
DEFAULT_INTENSITY_FLOOR = 100.0
DEFAULT_APEX_DELTA_FRACTION = 0.3   # heavy apex within 0.3 * light_peak_width
DEFAULT_PEARSON_MIN = 0.5

# Below this absolute m/z difference (Da), light and heavy fragment
# masses are considered equal (i.e. the fragment carries no K/R, so
# SILAC does not shift it). Smaller than the smallest SILAC delta
# (R = +10.008 Da, K = +8.014 Da) by many orders.
SHIFT_EPSILON = 0.001


def is_signal_present_light(xic, intensity_floor: float = DEFAULT_INTENSITY_FLOOR) -> bool:
    """Light signal is 'present' iff the XIC has a peak above the floor.

    Only the intensity criterion is applied here; light is the
    reference signal, so we don't compare it against itself for shape
    or coelution.
    """
    if xic is None or len(xic) == 0:
        return False
    max_int = float(np.nanmax(xic["intensity"])) if xic.size > 0 else 0.0
    if not np.isfinite(max_int):
        return False
    return max_int > intensity_floor
