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
    intensities = np.asarray(xic["intensity"])
    # Detect all-NaN before calling nanmax (which would emit a RuntimeWarning
    # then return NaN). is_signal_present_light returns False for that case.
    if not np.any(np.isfinite(intensities)):
        return False
    max_int = float(np.nanmax(intensities))
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

    if not np.any(np.isfinite(light_xic["intensity"])):
        return False
    if not np.any(np.isfinite(heavy_xic["intensity"])):
        return False

    heavy_max = float(np.nanmax(heavy_xic["intensity"]))
    if not np.isfinite(heavy_max) or heavy_max <= intensity_floor:
        return False

    light_apex_rt = float(light_xic["rt"][np.nanargmax(light_xic["intensity"])])
    heavy_apex_rt = float(heavy_xic["rt"][np.nanargmax(heavy_xic["intensity"])])
    apex_delta = abs(heavy_apex_rt - light_apex_rt)
    light_pw = _peak_width(light_xic)
    if light_pw <= 0:
        return False  # No peak shape; can't validate coelution
    if apex_delta >= apex_delta_fraction * light_pw:
        return False

    # Pearson correlation on shared rt grid (defensive sort first, mirrors calc_xic_score)
    light_sorted = light_xic[np.argsort(light_xic["rt"])]
    heavy_sorted = heavy_xic[np.argsort(heavy_xic["rt"])]
    rt_start = max(light_sorted["rt"].min(), heavy_sorted["rt"].min())
    rt_end = min(light_sorted["rt"].max(), heavy_sorted["rt"].max())
    if rt_start >= rt_end:
        return False
    # Cap interpolation grid to avoid oversampling narrow peaks
    # (which would bias pearson upward).
    n_points = min(100, max(len(light_sorted), len(heavy_sorted), 10))
    common_rt = np.linspace(rt_start, rt_end, n_points)
    l_int = np.interp(common_rt, light_sorted["rt"], light_sorted["intensity"])
    h_int = np.interp(common_rt, heavy_sorted["rt"], heavy_sorted["intensity"])
    if np.std(l_int) < 1e-10 or np.std(h_int) < 1e-10:
        return False
    try:
        corr, _ = pearsonr(l_int, h_int)
    except ValueError:
        return False
    if not np.isfinite(corr):
        return False
    return bool(corr > pearson_min)


def is_split_window(w_light: dict, w_heavy: dict):
    """Return True if windows differ, False if same, None if undecidable.

    None means: window lookup failed for at least one of light/heavy
    (bounds are NaN/None). Caller should treat this as 'cannot judge
    separation' rather than defaulting to either case.
    """
    l_lo, l_hi = w_light.get("lower"), w_light.get("upper")
    h_lo, h_hi = w_heavy.get("lower"), w_heavy.get("upper")

    def _is_nan_like(v):
        if v is None:
            return True
        try:
            return bool(np.isnan(v))
        except (TypeError, ValueError):
            return False

    if any(_is_nan_like(v) for v in (l_lo, l_hi, h_lo, h_hi)):
        return None
    return (l_lo != h_lo) or (l_hi != h_hi)


def is_separable_fragment(
    light_mass: float, heavy_mass: float, split_window,
    shift_epsilon: float = SHIFT_EPSILON,
) -> bool:
    """A fragment is separable iff:
      (a) It carries K or R and is shifted (light_mass != heavy_mass), OR
      (b) The DIA windows are KNOWN to be split.

    Unknown (None) windows are NOT treated as split for unshifted
    fragments — we don't have a physical basis to claim separation.
    """
    is_shifted = (heavy_mass - light_mass) > shift_epsilon
    if is_shifted:
        return True
    return split_window is True


class Q1aAccumulator:
    """Per-PSM accumulator for Q1a features.

    Usage:
        acc = Q1aAccumulator(split_window=is_split_window(w_L, w_H))
        for ion_type, position, light_mass, heavy_mass in fragments:
            light_xic = dia.xic_ms2_peaks_extract(...)
            heavy_xic = dia.xic_ms2_peaks_extract(...)
            acc.add(ion_type, light_mass, heavy_mass, light_xic, heavy_xic)
        features.update(acc.compute_features())

    All 11 output features are produced by compute_features().
    """

    MIN_VALID_TOTAL = 3  # spec §4.2: q1a_valid = (total >= 3)
    VALID_ION_TYPES = ("b", "y")

    def __init__(
        self,
        split_window: bool,
        intensity_floor: float = DEFAULT_INTENSITY_FLOOR,
        apex_delta_fraction: float = DEFAULT_APEX_DELTA_FRACTION,
        pearson_min: float = DEFAULT_PEARSON_MIN,
    ):
        self.split_window = split_window
        self.intensity_floor = intensity_floor
        self.apex_delta_fraction = apex_delta_fraction
        self.pearson_min = pearson_min
        # Bucket counters: keys are (mechanism, ion_type, outcome)
        # mechanism ∈ {"shifted", "unshifted_separable"}
        # ion_type ∈ {"b", "y"}
        # outcome ∈ {"TP", "FN"}
        self._counts: dict[tuple, int] = {}

    def add(self, ion_type: str, light_mass: float, heavy_mass: float,
            light_xic, heavy_xic) -> None:
        """Process one theoretical fragment.

        Side effects: increments internal counters. Has no return value.
        Fragments that are not separable OR have no light signal are
        silently dropped from Q1a statistics.
        """
        if ion_type not in self.VALID_ION_TYPES:
            raise ValueError(
                f"ion_type must be one of {self.VALID_ION_TYPES}, "
                f"got {ion_type!r}")
        if not np.isfinite(light_mass) or not np.isfinite(heavy_mass):
            import warnings
            warnings.warn(
                f"Skipping fragment with non-finite mass: "
                f"light={light_mass}, heavy={heavy_mass}",
                RuntimeWarning, stacklevel=2)
            return
        if heavy_mass < light_mass - SHIFT_EPSILON:
            import warnings
            warnings.warn(
                f"Skipping fragment with negative heavy_mass shift "
                f"(physically impossible for SILAC): "
                f"light={light_mass}, heavy={heavy_mass}",
                RuntimeWarning, stacklevel=2)
            return
        if not is_separable_fragment(light_mass, heavy_mass, self.split_window):
            return
        is_shifted = (heavy_mass - light_mass) > SHIFT_EPSILON
        mechanism = "shifted" if is_shifted else "unshifted_separable"

        if not is_signal_present_light(light_xic, self.intensity_floor):
            return

        heavy_present = is_signal_present_heavy(
            light_xic, heavy_xic,
            intensity_floor=self.intensity_floor,
            apex_delta_fraction=self.apex_delta_fraction,
            pearson_min=self.pearson_min,
        )
        outcome = "TP" if heavy_present else "FN"

        key = (mechanism, ion_type, outcome)
        self._counts[key] = self._counts.get(key, 0) + 1

    def _sum(self, mechanism=None, ion_type=None, outcome=None) -> int:
        """Sum counters filtered by mechanism/ion_type/outcome (None means wildcard)."""
        total = 0
        for (m, i, o), n in self._counts.items():
            if mechanism is not None and m != mechanism:
                continue
            if ion_type is not None and i != ion_type:
                continue
            if outcome is not None and o != outcome:
                continue
            total += n
        return total

    def _recall(self, mechanism=None, ion_type=None) -> float:
        tp = self._sum(mechanism=mechanism, ion_type=ion_type, outcome="TP")
        fn = self._sum(mechanism=mechanism, ion_type=ion_type, outcome="FN")
        total = tp + fn
        # MIN_VALID_TOTAL only applies to overall + per-mechanism recall.
        # Per-ion-type recall (q1a_y_recall / q1a_b_recall) is reported
        # whenever the bucket is non-empty.
        if ion_type is None:
            if total < self.MIN_VALID_TOTAL:
                return float("nan")
        else:
            if total == 0:
                return float("nan")
        return tp / total

    def compute_features(self) -> dict:
        """Return the 11-field Q1a feature dict.

        Conventions:
          - recall is NaN when its bucket has < MIN_VALID_TOTAL (3) entries.
          - q1a_recall_unshifted_separable is additionally NaN under
            co-isolation (where unshifted_separable count is always 0).
          - count features are always integers.
        """
        tp_total = self._sum(outcome="TP")
        fn_total = self._sum(outcome="FN")
        total = tp_total + fn_total
        tp_shifted = self._sum(mechanism="shifted", outcome="TP")
        tp_unsh = self._sum(mechanism="unshifted_separable", outcome="TP")

        # In co-iso, unshifted_separable bucket is by construction empty.
        if not self.split_window:
            recall_unsh = float("nan")
        else:
            recall_unsh = self._recall(mechanism="unshifted_separable")

        return {
            "q1a_recall": self._recall(),
            "q1a_recall_shifted": self._recall(mechanism="shifted"),
            "q1a_recall_unshifted_separable": recall_unsh,
            "q1a_y_recall": self._recall(ion_type="y"),
            "q1a_b_recall": self._recall(ion_type="b"),
            "q1a_TP_count": tp_total,
            "q1a_FN_count": fn_total,
            "q1a_TP_shifted": tp_shifted,
            "q1a_TP_unshifted_separable": tp_unsh,
            "q1a_total_count": total,
            "q1a_valid": 1 if total >= self.MIN_VALID_TOTAL else 0,
        }
