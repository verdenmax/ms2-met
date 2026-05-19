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


def _peak_width(xic) -> float:
    """Span of rt values in the XIC (used as a denominator for
    apex_delta normalization). Returns 0 for empty/single-point XICs."""
    if xic is None or len(xic) < 2:
        return 0.0
    rts = np.asarray(xic["rt"], dtype="f8")
    return float(rts.max() - rts.min())


def is_signal_present_heavy(
    light_xic,
    heavy_xic,
    intensity_floor: float = DEFAULT_INTENSITY_FLOOR,
    apex_delta_fraction: float = DEFAULT_APEX_DELTA_FRACTION,
    pearson_min: float = DEFAULT_PEARSON_MIN,
) -> bool:
    """Heavy 'present' iff three conditions all hold:
      1. heavy max intensity > intensity_floor
      2. |heavy_apex_rt - light_apex_rt| < apex_delta_fraction * light_peak_width
      3. pearsonr(aligned_light, aligned_heavy) > pearson_min
    """
    if heavy_xic is None or len(heavy_xic) == 0:
        return False
    if light_xic is None or len(light_xic) == 0:
        return False

    heavy_max = float(np.nanmax(heavy_xic["intensity"]))
    if not np.isfinite(heavy_max) or heavy_max <= intensity_floor:
        return False

    light_apex_rt = float(light_xic["rt"][np.nanargmax(light_xic["intensity"])])
    heavy_apex_rt = float(heavy_xic["rt"][np.nanargmax(heavy_xic["intensity"])])
    apex_delta = abs(heavy_apex_rt - light_apex_rt)
    light_pw = _peak_width(light_xic)
    if light_pw > 0 and apex_delta >= apex_delta_fraction * light_pw:
        return False

    # Pearson correlation on shared rt grid (defensive sort first, mirrors calc_xic_score)
    light_sorted = light_xic[np.argsort(light_xic["rt"])]
    heavy_sorted = heavy_xic[np.argsort(heavy_xic["rt"])]
    rt_start = max(light_sorted["rt"].min(), heavy_sorted["rt"].min())
    rt_end = min(light_sorted["rt"].max(), heavy_sorted["rt"].max())
    if rt_start >= rt_end:
        return False
    common_rt = np.linspace(rt_start, rt_end, 100)
    l_int = np.interp(common_rt, light_sorted["rt"], light_sorted["intensity"])
    h_int = np.interp(common_rt, heavy_sorted["rt"], heavy_sorted["intensity"])
    if np.std(l_int) < 1e-10 or np.std(h_int) < 1e-10:
        return False
    try:
        corr, _ = pearsonr(l_int, h_int)
    except (ValueError, RuntimeWarning):
        return False
    if not np.isfinite(corr):
        return False
    return bool(corr > pearson_min)
