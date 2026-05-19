"""Tests for workflows/q1a_helpers.py — Q1a fragment-pairing math."""
import numpy as np
import pytest


XIC_DTYPE = [
    ("rt", "f8"), ("intensity", "f8"),
    ("ppm_error", "f8"), ("mz", "f8"),
]


def _xic(rts, intensities):
    """Build a minimal XIC numpy record array for tests."""
    n = len(rts)
    arr = np.zeros(n, dtype=XIC_DTYPE)
    arr["rt"] = rts
    arr["intensity"] = intensities
    return arr


# ----------------------------------------------------------------------
# is_signal_present_light
# ----------------------------------------------------------------------

def test_light_present_when_max_intensity_above_floor():
    from workflows.q1a_helpers import is_signal_present_light
    xic = _xic([10, 11, 12, 13, 14], [50, 200, 500, 200, 50])
    assert is_signal_present_light(xic, intensity_floor=100) is True


def test_light_absent_when_max_intensity_below_floor():
    from workflows.q1a_helpers import is_signal_present_light
    xic = _xic([10, 11, 12], [10, 50, 80])
    assert is_signal_present_light(xic, intensity_floor=100) is False


def test_light_absent_when_xic_empty():
    from workflows.q1a_helpers import is_signal_present_light
    xic = _xic([], [])
    assert is_signal_present_light(xic, intensity_floor=100) is False


def test_light_absent_when_all_nan_intensity():
    from workflows.q1a_helpers import is_signal_present_light
    xic = _xic([10, 11, 12], [np.nan, np.nan, np.nan])
    assert is_signal_present_light(xic, intensity_floor=100) is False


# ----------------------------------------------------------------------
# is_signal_present_heavy (three-condition AND)
# ----------------------------------------------------------------------

def test_heavy_present_perfect_pair():
    """Heavy XIC matches light shape + intensity + apex → present."""
    from workflows.q1a_helpers import is_signal_present_heavy
    light = _xic([10, 11, 12, 13, 14], [50, 200, 500, 200, 50])
    heavy = _xic([10, 11, 12, 13, 14], [40, 160, 400, 160, 40])  # same shape
    assert is_signal_present_heavy(
        light, heavy, intensity_floor=100,
        apex_delta_fraction=0.3, pearson_min=0.5) is True


def test_heavy_absent_when_intensity_below_floor():
    from workflows.q1a_helpers import is_signal_present_heavy
    light = _xic([10, 11, 12, 13, 14], [50, 200, 500, 200, 50])
    heavy = _xic([10, 11, 12, 13, 14], [5, 20, 50, 20, 5])  # all < 100
    assert is_signal_present_heavy(
        light, heavy, intensity_floor=100) is False


def test_heavy_absent_when_apex_delta_too_large():
    """Heavy apex far from light apex → not coeluting → absent."""
    from workflows.q1a_helpers import is_signal_present_heavy
    light = _xic([10, 11, 12, 13, 14], [50, 200, 500, 200, 50])  # apex at 12
    heavy = _xic([10, 11, 12, 13, 14], [500, 200, 50, 20, 10])    # apex at 10
    # peak width = 4; apex delta = 2; 2 > 0.3*4 = 1.2 → absent
    assert is_signal_present_heavy(
        light, heavy, intensity_floor=100,
        apex_delta_fraction=0.3) is False


def test_heavy_absent_when_pearson_below_threshold():
    """Same intensity + coeluting but anti-correlated shape → absent."""
    from workflows.q1a_helpers import is_signal_present_heavy
    light = _xic([10, 11, 12, 13, 14], [50, 200, 500, 200, 50])
    # Inverted: same total intensity but anti-correlated
    heavy = _xic([10, 11, 12, 13, 14], [500, 200, 500, 200, 500])
    assert is_signal_present_heavy(
        light, heavy, intensity_floor=100,
        apex_delta_fraction=1.0,  # disable apex check to isolate pearson
        pearson_min=0.9) is False


def test_heavy_absent_when_empty_xic():
    from workflows.q1a_helpers import is_signal_present_heavy
    light = _xic([10, 11, 12], [50, 200, 500])
    heavy = _xic([], [])
    assert is_signal_present_heavy(light, heavy, intensity_floor=100) is False


# ----------------------------------------------------------------------
# Window separability
# ----------------------------------------------------------------------

def test_is_split_window_same_bounds_returns_false():
    from workflows.q1a_helpers import is_split_window
    w_L = {"width": 2.0, "centering": 0.5, "lower": 500.0, "upper": 502.0}
    w_H = {"width": 2.0, "centering": 0.7, "lower": 500.0, "upper": 502.0}
    assert is_split_window(w_L, w_H) is False


def test_is_split_window_different_bounds_returns_true():
    from workflows.q1a_helpers import is_split_window
    w_L = {"width": 2.0, "centering": 0.5, "lower": 500.0, "upper": 502.0}
    w_H = {"width": 2.0, "centering": 0.5, "lower": 504.0, "upper": 506.0}
    assert is_split_window(w_L, w_H) is True


def test_is_split_window_nan_returns_none():
    """NaN bounds = unknown isolation. Helper must signal this clearly,
    NOT default to 'split' (which over-admits unshifted fragments)."""
    from workflows.q1a_helpers import is_split_window
    w_L = {"width": 2.0, "centering": 0.5, "lower": 500.0, "upper": 502.0}
    w_H = {"width": 0.0, "centering": 0.5,
           "lower": float("nan"), "upper": float("nan")}
    assert is_split_window(w_L, w_H) is None


def test_is_split_window_numpy_nan_handled():
    """is_split_window must also recognize numpy NaN, not just Python float NaN."""
    from workflows.q1a_helpers import is_split_window
    w_L = {"width": 2.0, "centering": 0.5,
           "lower": np.float64(500.0), "upper": np.float64(502.0)}
    w_H = {"width": 0.0, "centering": 0.5,
           "lower": np.float64("nan"), "upper": np.float64("nan")}
    assert is_split_window(w_L, w_H) is None


def test_is_separable_fragment_unshifted_with_unknown_window_returns_false():
    """When the window is 'unknown' (None passed in), unshifted fragments
    are NOT separable — we have no physical basis to claim separation."""
    from workflows.q1a_helpers import is_separable_fragment
    assert is_separable_fragment(
        light_mass=300.0, heavy_mass=300.0, split_window=None) is False
    # shifted always separable, regardless
    assert is_separable_fragment(
        light_mass=300.0, heavy_mass=310.0, split_window=None) is True


def test_q1a_accumulator_unknown_window_drops_unshifted():
    """Accumulator with None (unknown) split_window must drop unshifted
    fragments entirely — they don't count to total, TP, or FN."""
    from workflows.q1a_helpers import Q1aAccumulator
    rts = np.linspace(10, 14, 5)
    light_int = [50, 200, 500, 200, 50]
    light = np.zeros(5, dtype=[("rt", "f8"), ("intensity", "f8"),
                               ("ppm_error", "f8"), ("mz", "f8")])
    light["rt"] = rts
    light["intensity"] = light_int
    heavy = light.copy()
    acc = Q1aAccumulator(split_window=None)
    for _ in range(5):
        acc.add(ion_type="b", light_mass=300.0, heavy_mass=300.0,
                light_xic=light, heavy_xic=heavy)
    feats = acc.compute_features()
    assert feats["q1a_total_count"] == 0
    assert feats["q1a_TP_unshifted_separable"] == 0
    for _ in range(5):
        acc.add(ion_type="y", light_mass=300.0, heavy_mass=310.0,
                light_xic=light, heavy_xic=heavy)
    feats = acc.compute_features()
    assert feats["q1a_total_count"] == 5
    assert feats["q1a_TP_shifted"] == 5


# ----------------------------------------------------------------------
# Fragment separability
# ----------------------------------------------------------------------

def test_shifted_fragment_always_separable():
    """A fragment with K or R is shifted by SILAC; always separable
    by m/z regardless of window configuration."""
    from workflows.q1a_helpers import is_separable_fragment
    assert is_separable_fragment(
        light_mass=300.0, heavy_mass=310.0, split_window=True) is True
    assert is_separable_fragment(
        light_mass=300.0, heavy_mass=310.0, split_window=False) is True


def test_unshifted_fragment_separable_only_in_split_window():
    """A fragment with no K/R has equal light/heavy mass. It can only
    be separated by the DIA window (its precursor isolation differs)."""
    from workflows.q1a_helpers import is_separable_fragment
    assert is_separable_fragment(
        light_mass=300.0, heavy_mass=300.0, split_window=True) is True
    assert is_separable_fragment(
        light_mass=300.0, heavy_mass=300.0, split_window=False) is False


# ----------------------------------------------------------------------
# Q1aAccumulator (per-PSM, builds the 11 output features)
# ----------------------------------------------------------------------

def _silac_pair(light_int, heavy_int, n=5):
    """Build a (light_xic, heavy_xic) pair with given peak intensities,
    aligned apex, gaussian-ish shape."""
    rts = np.linspace(10, 14, n)
    factor_l = light_int / 500.0
    factor_h = heavy_int / 500.0
    light = _xic(rts, [50 * factor_l, 200 * factor_l, 500 * factor_l,
                       200 * factor_l, 50 * factor_l])
    heavy = _xic(rts, [50 * factor_h, 200 * factor_h, 500 * factor_h,
                       200 * factor_h, 50 * factor_h])
    return light, heavy


def _empty_xic():
    return _xic([], [])


def test_q1a_accumulator_perfect_silac_recall_1():
    """5 shifted fragments all paired → q1a_recall = 1, valid = 1."""
    from workflows.q1a_helpers import Q1aAccumulator
    acc = Q1aAccumulator(split_window=False)
    for ion_type in ("y", "y", "y", "y", "b"):
        light, heavy = _silac_pair(500, 400)
        acc.add(ion_type=ion_type,
                light_mass=300.0, heavy_mass=310.0,
                light_xic=light, heavy_xic=heavy)
    feats = acc.compute_features()
    assert feats["q1a_TP_count"] == 5
    assert feats["q1a_FN_count"] == 0
    assert feats["q1a_recall"] == 1.0
    assert feats["q1a_valid"] == 1
    assert feats["q1a_total_count"] == 5
    assert feats["q1a_recall_shifted"] == 1.0
    # No unshifted_separable contributions
    assert np.isnan(feats["q1a_recall_unshifted_separable"])


def test_q1a_accumulator_trap_no_heavy_recall_0():
    """5 shifted fragments where heavy XIC is empty → FN, recall=0."""
    from workflows.q1a_helpers import Q1aAccumulator
    acc = Q1aAccumulator(split_window=False)
    for _ in range(5):
        light, _ = _silac_pair(500, 0)
        acc.add(ion_type="y",
                light_mass=300.0, heavy_mass=310.0,
                light_xic=light, heavy_xic=_empty_xic())
    feats = acc.compute_features()
    assert feats["q1a_TP_count"] == 0
    assert feats["q1a_FN_count"] == 5
    assert feats["q1a_recall"] == 0.0
    assert feats["q1a_valid"] == 1


def test_q1a_accumulator_total_lt_3_recall_nan_valid_0():
    """Only 2 separable fragments → q1a_valid=0, q1a_recall=NaN."""
    from workflows.q1a_helpers import Q1aAccumulator
    acc = Q1aAccumulator(split_window=False)
    for _ in range(2):
        light, heavy = _silac_pair(500, 400)
        acc.add(ion_type="y",
                light_mass=300.0, heavy_mass=310.0,
                light_xic=light, heavy_xic=heavy)
    feats = acc.compute_features()
    assert feats["q1a_total_count"] == 2
    assert feats["q1a_valid"] == 0
    assert np.isnan(feats["q1a_recall"])
    assert np.isnan(feats["q1a_recall_shifted"])


def test_q1a_accumulator_unshifted_skipped_under_co_iso():
    """Co-isolation + unshifted (b ion no K/R) fragments → not added."""
    from workflows.q1a_helpers import Q1aAccumulator
    acc = Q1aAccumulator(split_window=False)
    for _ in range(5):
        light, heavy = _silac_pair(500, 400)
        acc.add(ion_type="b",
                light_mass=300.0, heavy_mass=300.0,
                light_xic=light, heavy_xic=heavy)
    feats = acc.compute_features()
    assert feats["q1a_total_count"] == 0
    assert feats["q1a_valid"] == 0
    assert np.isnan(feats["q1a_recall"])


def test_q1a_accumulator_unshifted_separable_under_split_iso():
    """Split window + unshifted fragment → counts under
    q1a_recall_unshifted_separable."""
    from workflows.q1a_helpers import Q1aAccumulator
    acc = Q1aAccumulator(split_window=True)
    for _ in range(5):
        light, heavy = _silac_pair(500, 400)
        acc.add(ion_type="b",
                light_mass=300.0, heavy_mass=300.0,
                light_xic=light, heavy_xic=heavy)
    feats = acc.compute_features()
    assert feats["q1a_total_count"] == 5
    assert feats["q1a_TP_unshifted_separable"] == 5
    assert feats["q1a_recall_unshifted_separable"] == 1.0
    # And the *_shifted slice is empty here → NaN
    assert np.isnan(feats["q1a_recall_shifted"])


def test_q1a_accumulator_light_invalid_excluded():
    """Fragments where light signal is below floor → neither TP nor FN."""
    from workflows.q1a_helpers import Q1aAccumulator
    acc = Q1aAccumulator(split_window=False, intensity_floor=100)
    # Light intensity 50 < floor 100 → fragment excluded
    light = _xic([10, 11, 12], [10, 30, 50])
    heavy = _xic([10, 11, 12], [10, 30, 50])
    for _ in range(5):
        acc.add(ion_type="y",
                light_mass=300.0, heavy_mass=310.0,
                light_xic=light, heavy_xic=heavy)
    feats = acc.compute_features()
    assert feats["q1a_total_count"] == 0
    assert np.isnan(feats["q1a_recall"])


def test_q1a_accumulator_y_b_split():
    """y and b counts are tracked separately."""
    from workflows.q1a_helpers import Q1aAccumulator
    acc = Q1aAccumulator(split_window=False)
    # 3 y TP, 2 b TP
    for _ in range(3):
        light, heavy = _silac_pair(500, 400)
        acc.add(ion_type="y",
                light_mass=300.0, heavy_mass=310.0,
                light_xic=light, heavy_xic=heavy)
    for _ in range(2):
        light, heavy = _silac_pair(500, 400)
        acc.add(ion_type="b",
                light_mass=300.0, heavy_mass=310.0,
                light_xic=light, heavy_xic=heavy)
    feats = acc.compute_features()
    assert feats["q1a_y_recall"] == 1.0
    assert feats["q1a_b_recall"] == 1.0
    assert feats["q1a_TP_count"] == 5
    assert feats["q1a_total_count"] == 5


# ----------------------------------------------------------------------
# Ablation feature grouping guard
# ----------------------------------------------------------------------

def test_q1a_features_are_not_in_sequence_only_group():
    """q1a_* are SILAC-pairing features. They must NEVER end up in
    sequence_only when split_features() runs the ablation grouping."""
    from tools.eval_feature_ablation import split_features

    # Synthesize a feature column list that includes q1a_* alongside
    # both sequence-only and known silac-only features.
    all_features = [
        "modification_count", "kr_count", "sequence_len",  # sequence
        "precursor_pearson", "all_cosine_p50",             # silac existing
        "q1a_recall", "q1a_recall_shifted",
        "q1a_recall_unshifted_separable", "q1a_y_recall",
        "q1a_b_recall", "q1a_TP_count", "q1a_FN_count",
        "q1a_TP_shifted", "q1a_TP_unshifted_separable",
        "q1a_total_count", "q1a_valid",
    ]
    groups = split_features(all_features)
    q1a_features = [f for f in all_features if f.startswith("q1a_")]
    assert len(q1a_features) == 11, "test should reference all 11 q1a features"
    for q1a_feat in q1a_features:
        assert q1a_feat not in groups["sequence_only"], (
            f"{q1a_feat} accidentally classified as sequence_only")
        assert q1a_feat in groups["silac_only"], (
            f"{q1a_feat} missing from silac_only")


# ----------------------------------------------------------------------
# RuntimeWarning regression (all-NaN intensity must not warn)
# ----------------------------------------------------------------------

def test_light_present_no_warning_on_all_nan(recwarn):
    """is_signal_present_light must NOT emit RuntimeWarning when the
    XIC has all-NaN intensity (we early-return False before nanmax)."""
    import warnings
    from workflows.q1a_helpers import is_signal_present_light
    xic = _xic([10, 11, 12], [np.nan, np.nan, np.nan])
    with warnings.catch_warnings():
        warnings.simplefilter("error", RuntimeWarning)
        assert is_signal_present_light(xic) is False


def test_heavy_present_handles_all_nan_light_intensity():
    """is_signal_present_heavy must not raise on all-NaN light."""
    from workflows.q1a_helpers import is_signal_present_heavy
    light = _xic([10, 11, 12], [np.nan, np.nan, np.nan])
    heavy = _xic([10, 11, 12], [100, 500, 100])
    assert is_signal_present_heavy(light, heavy) is False


def test_heavy_present_handles_all_nan_heavy_intensity():
    """And mirror for heavy."""
    from workflows.q1a_helpers import is_signal_present_heavy
    light = _xic([10, 11, 12], [100, 500, 100])
    heavy = _xic([10, 11, 12], [np.nan, np.nan, np.nan])
    assert is_signal_present_heavy(light, heavy) is False


def test_heavy_present_fails_when_light_peak_width_zero():
    """Single-point light XIC has no peak shape → can't judge apex
    coelution → must return False (not silently skip the check)."""
    from workflows.q1a_helpers import is_signal_present_heavy
    light = _xic([10.0], [500])
    heavy = _xic([10.0, 11.0, 12.0], [100, 500, 100])
    assert is_signal_present_heavy(light, heavy) is False


def test_accumulator_rejects_unknown_ion_type():
    """Unknown ion_type must raise ValueError — silently bucketing
    inflates q1a_total_count but not q1a_y_recall/q1a_b_recall."""
    from workflows.q1a_helpers import Q1aAccumulator
    acc = Q1aAccumulator(split_window=True)
    light, heavy = _silac_pair(500, 400)
    with pytest.raises(ValueError, match="ion_type"):
        acc.add(ion_type="Y",
                light_mass=300.0, heavy_mass=310.0,
                light_xic=light, heavy_xic=heavy)
    with pytest.raises(ValueError, match="ion_type"):
        acc.add(ion_type="a",
                light_mass=300.0, heavy_mass=310.0,
                light_xic=light, heavy_xic=heavy)


def test_accumulator_rejects_negative_heavy_mass():
    """Negative heavy delta is physically impossible for SILAC.
    Must drop with warning — not silently re-classify as unshifted."""
    import warnings
    from workflows.q1a_helpers import Q1aAccumulator
    acc = Q1aAccumulator(split_window=True)
    light, heavy = _silac_pair(500, 400)
    with warnings.catch_warnings(record=True) as w:
        warnings.simplefilter("always")
        acc.add(ion_type="y",
                light_mass=310.0, heavy_mass=300.0,
                light_xic=light, heavy_xic=heavy)
        assert acc.compute_features()["q1a_total_count"] == 0
        assert any("heavy_mass" in str(ww.message).lower() or
                   "negative" in str(ww.message).lower() for ww in w)


def test_accumulator_rejects_nan_mass():
    """NaN mass → skip with warning."""
    import warnings
    from workflows.q1a_helpers import Q1aAccumulator
    acc = Q1aAccumulator(split_window=True)
    light, heavy = _silac_pair(500, 400)
    with warnings.catch_warnings(record=True):
        warnings.simplefilter("always")
        acc.add(ion_type="y",
                light_mass=float("nan"), heavy_mass=310.0,
                light_xic=light, heavy_xic=heavy)
        acc.add(ion_type="y",
                light_mass=300.0, heavy_mass=float("nan"),
                light_xic=light, heavy_xic=heavy)
        assert acc.compute_features()["q1a_total_count"] == 0
