"""Unit tests for spectrum.spectrum_utils.centroid_spectrum."""
import numpy as np
import pytest

from spectrum.spectrum_utils import centroid_spectrum
from spectrum.dia_data import _is_already_centroid, DIAData


def _gaussian_profile(centers, heights, sigma=0.005, n_per_peak=11,
                      span_sigmas=3.0, dtype=np.float32):
    """Build a synthetic profile spectrum: isolated Gaussian peaks.

    Returns (mz, intensity) with peaks well-separated so no overlap.
    """
    mz_chunks = []
    int_chunks = []
    for c, h in zip(centers, heights):
        # n_per_peak points across +/- span_sigmas around the center
        rel = np.linspace(-span_sigmas, span_sigmas, n_per_peak)
        mz_chunks.append(c + rel * sigma)
        int_chunks.append(h * np.exp(-0.5 * rel ** 2))
    mz = np.concatenate(mz_chunks).astype(dtype)
    intensity = np.concatenate(int_chunks).astype(dtype)
    # Sort by mz (sanity; should already be sorted)
    order = np.argsort(mz)
    return mz[order], intensity[order]


def test_isolated_gaussian_peaks_recovered():
    """5 well-separated Gaussian peaks → 5 centroids close to true centers."""
    true_centers = [400.0, 500.0, 600.0, 700.0, 800.0]
    heights = [1000.0, 800.0, 1200.0, 600.0, 1500.0]
    mz, intensity = _gaussian_profile(true_centers, heights, sigma=0.005,
                                      n_per_peak=11)

    out_mz, out_int = centroid_spectrum(mz, intensity, rel_threshold=1e-3)

    assert len(out_mz) == 5, f"expected 5 centroids, got {len(out_mz)}"
    assert len(out_int) == 5
    # Each output should match a true center within 0.001 Da.
    for c in true_centers:
        diffs = np.abs(out_mz - c)
        assert diffs.min() < 0.001, (
            f"no centroid within 0.001 of {c}; min diff = {diffs.min()}")
    # Intensities should be near the input peak heights (parabolic-apex is
    # picked as peak-top sample intensity = approx height since center sample
    # sits at the true center).
    for h in heights:
        diffs = np.abs(out_int - h)
        assert diffs.min() < h * 0.05, (
            f"no intensity within 5% of {h}; min diff = {diffs.min()}")


def test_relative_threshold_filters_low_peaks():
    """Peaks below max*rel_threshold must be dropped."""
    true_centers = [500.0, 510.0, 520.0]
    # 520 is at 5e-4 of base peak — below default 1e-3 threshold
    heights = [1000.0, 800.0, 0.5]
    mz, intensity = _gaussian_profile(true_centers, heights, sigma=0.005,
                                      n_per_peak=11)

    out_mz, out_int = centroid_spectrum(mz, intensity, rel_threshold=1e-3)

    assert len(out_mz) == 2, f"expected 2 centroids (3rd below threshold), got {len(out_mz)}"
    assert all(abs(out_mz - 520.0) > 0.5), "520 peak should be filtered"


def test_empty_input_returns_empty():
    out_mz, out_int = centroid_spectrum(
        np.array([], dtype=np.float32),
        np.array([], dtype=np.float32),
    )
    assert out_mz.shape == (0,)
    assert out_int.shape == (0,)


def test_short_input_returns_empty():
    """Length < 3 cannot have an interior local maximum."""
    out_mz, out_int = centroid_spectrum(
        np.array([100.0, 101.0], dtype=np.float32),
        np.array([5.0, 6.0], dtype=np.float32),
    )
    assert out_mz.shape == (0,)
    assert out_int.shape == (0,)


def test_flat_top_picks_leftmost_does_not_raise():
    """Flat-top plateau ``[1, 10, 10, 10, 1]``: the asymmetric detection
    rule (``> left & >= right``) selects only the leftmost shoulder
    (i=1), where ``denom = (y0-y1) + (y2-y1) = -9 + 0 = -9`` is strictly
    negative — so no zero-division occurs and no exception is raised.
    The resulting centroid must still fall inside the flat-top region.

    Note: under the detection rule the fallback ``safe``/``where=`` branch
    is unreachable for any detected peak (denom is strictly negative for
    every selected index), so this test only asserts the observable
    behaviour — no raise, centroid inside the plateau.
    """
    mz = np.array([100.0, 100.01, 100.02, 100.03, 100.04],
                  dtype=np.float32)
    intensity = np.array([1.0, 10.0, 10.0, 10.0, 1.0], dtype=np.float32)
    # Should not raise; should produce at least one centroid.
    out_mz, out_int = centroid_spectrum(mz, intensity, rel_threshold=1e-3)
    assert len(out_mz) >= 1
    # The centroid m/z should fall within the flat-top region.
    assert 100.005 <= out_mz[0] <= 100.035


def test_dtype_preserved_float32():
    mz, intensity = _gaussian_profile([500.0], [1000.0], dtype=np.float32)
    out_mz, out_int = centroid_spectrum(mz, intensity)
    assert out_mz.dtype == np.float32
    assert out_int.dtype == np.float32


def test_strictly_monotonic_mz_in_output():
    """Output m/z must be strictly increasing (no duplicates)."""
    true_centers = [400.0, 500.0, 600.0]
    heights = [1000.0, 1000.0, 1000.0]
    mz, intensity = _gaussian_profile(true_centers, heights)
    out_mz, _ = centroid_spectrum(mz, intensity)
    assert np.all(np.diff(out_mz) > 0)


def test_is_already_centroid_true_when_cv_term_present():
    spectrum = {
        'm/z array': np.array([100.0, 200.0]),
        'intensity array': np.array([1.0, 2.0]),
        'centroid spectrum': '',  # pyteomics stores cv terms as keys
    }
    assert _is_already_centroid(spectrum) is True


def test_is_already_centroid_false_when_profile():
    spectrum = {
        'm/z array': np.array([100.0, 200.0]),
        'intensity array': np.array([1.0, 2.0]),
        'profile spectrum': '',
    }
    assert _is_already_centroid(spectrum) is False


def test_is_already_centroid_false_when_neither_term():
    spectrum = {
        'm/z array': np.array([100.0, 200.0]),
        'intensity array': np.array([1.0, 2.0]),
    }
    assert _is_already_centroid(spectrum) is False


# ============================================================================
# Audit-time additional tests (2026-06-01).
#
# Added after running centroid_spectrum on real DIA Orbitrap profile mzML
# data and inspecting the algorithm against a 14-point issue checklist.
# Covers: input-contract enforcement (shape mismatch), edge cases (length-3
# inputs, monotonic ramps, array-boundary peaks, threshold boundary, adjacent
# peaks), sub-bin m/z accuracy on a synthetic offset Gaussian, NaN/Inf
# resilience, the documented "apex-sample" intensity contract, and a
# real-mzML integration sanity check (skipped if the fixture is missing).
# ============================================================================


def test_shape_mismatch_raises_valueerror():
    """A shape mismatch is a caller bug, not a graceful-degradation case.
    Silently returning empty would mask off-by-one / array-misalign bugs
    in the load pipeline (T6/T7 slice from pyteomics output)."""
    mz = np.array([100.0, 101.0, 102.0], dtype=np.float32)
    intensity = np.array([1.0, 2.0], dtype=np.float32)  # one short
    with pytest.raises(ValueError, match=r"length"):
        centroid_spectrum(mz, intensity)


def test_minimum_length_3_with_peak():
    """Length-3 with a clear local-max at the only interior position
    produces exactly one centroid (symmetric → bin centre)."""
    mz = np.array([100.0, 100.01, 100.02], dtype=np.float32)
    intensity = np.array([1.0, 10.0, 1.0], dtype=np.float32)
    out_mz, out_int = centroid_spectrum(mz, intensity, rel_threshold=1e-3)
    assert len(out_mz) == 1
    assert abs(out_mz[0] - 100.01) < 1e-5
    assert out_int[0] == np.float32(10.0)


def test_minimum_length_3_monotonic_returns_empty():
    """Length-3 monotonic ramp has no interior local-max."""
    mz = np.array([100.0, 100.01, 100.02], dtype=np.float32)
    intensity = np.array([1.0, 2.0, 3.0], dtype=np.float32)
    out_mz, out_int = centroid_spectrum(mz, intensity)
    assert len(out_mz) == 0
    assert len(out_int) == 0


def test_monotonic_increasing_input_returns_empty():
    """Strictly increasing intensity → no interior local-max possible."""
    mz = np.linspace(100.0, 200.0, 100).astype(np.float32)
    intensity = (np.arange(100) + 1.0).astype(np.float32)
    out_mz, out_int = centroid_spectrum(mz, intensity)
    assert len(out_mz) == 0
    assert len(out_int) == 0


def test_peak_at_array_boundary_not_detected():
    """A 'peak' at index 0 or n-1 cannot be detected (the rule requires
    interior neighbours on both sides). Documented limitation — real
    mzML always extends past peak shoulders so this is not a production
    bug, but the contract is pinned here."""
    mz = np.array([100.0, 100.01, 100.02, 100.03], dtype=np.float32)

    # Peak at left boundary (idx 0)
    intensity_left = np.array([10.0, 5.0, 2.0, 1.0], dtype=np.float32)
    out_mz, _ = centroid_spectrum(mz, intensity_left, rel_threshold=1e-3)
    assert len(out_mz) == 0, "leftmost-boundary peak must not be detected"

    # Peak at right boundary (idx n-1)
    intensity_right = np.array([1.0, 2.0, 5.0, 10.0], dtype=np.float32)
    out_mz, _ = centroid_spectrum(mz, intensity_right, rel_threshold=1e-3)
    assert len(out_mz) == 0, "rightmost-boundary peak must not be detected"


def test_adjacent_peaks_two_bins_apart_both_detected():
    """Two peaks separated by one valley sample (idx 1 and 3) must both
    be detected. Common in MS for closely-spaced isotopes."""
    mz = np.array([100.00, 100.01, 100.02, 100.03, 100.04],
                  dtype=np.float32)
    intensity = np.array([1.0, 10.0, 3.0, 8.0, 1.0], dtype=np.float32)
    out_mz, out_int = centroid_spectrum(mz, intensity, rel_threshold=1e-3)
    assert len(out_mz) == 2
    assert abs(out_mz[0] - 100.01) < 0.005
    assert abs(out_mz[1] - 100.03) < 0.005


def test_intensity_at_threshold_exactly_kept():
    """The ``>=`` comparison must keep peaks at intensity == max*rel_threshold."""
    # max=1000, weak peak at exactly 1.0 (== 1e-3 * max).
    mz = np.array([100.0, 100.01, 100.02, 100.03, 100.04, 100.05,
                   100.06], dtype=np.float32)
    intensity = np.array([0.1, 1000.0, 0.1, 0.5, 1.0, 0.5, 0.1],
                         dtype=np.float32)
    out_mz, _ = centroid_spectrum(mz, intensity, rel_threshold=1e-3)
    assert len(out_mz) == 2, "weak peak at exactly threshold must be kept"


def test_intensity_just_below_threshold_dropped():
    """Weak peak with intensity < max * rel_threshold must be filtered."""
    mz = np.array([100.0, 100.01, 100.02, 100.03, 100.04, 100.05,
                   100.06], dtype=np.float32)
    intensity = np.array([0.1, 1000.0, 0.1, 0.5, 0.999, 0.5, 0.1],
                         dtype=np.float32)
    out_mz, _ = centroid_spectrum(mz, intensity, rel_threshold=1e-3)
    assert len(out_mz) == 1, "weak peak below threshold must be dropped"
    assert abs(out_mz[0] - 100.01) < 0.005


def test_subbin_accuracy_against_offset_gaussian():
    """For a Gaussian peak with its true centre at +0.3 bins from the
    nearest sample, parabolic refinement must place the centroid within
    0.1 ppm of the true centre — far tighter than bin-spacing alone
    could achieve. This is the test that actually validates the value
    of the parabolic-refinement step over picking bin centres."""
    # Realistic Orbitrap-like setup near m/z 500:
    bin_step = 0.001  # ~1 mDa between samples
    sigma = 0.0025    # FWHM ≈ 0.006 Da, ~12 ppm
    n = 21
    mz = (500.0 + (np.arange(n) - n // 2) * bin_step).astype(np.float64)
    # True centre offset +0.3 * bin_step from the nearest sample.
    true_center = 500.0 + 0.3 * bin_step
    intensity = (
        1e6 * np.exp(-0.5 * ((mz - true_center) / sigma) ** 2)
    ).astype(np.float64)

    out_mz, _ = centroid_spectrum(mz, intensity, rel_threshold=1e-3)
    assert len(out_mz) == 1, f"expected 1 centroid, got {len(out_mz)}"
    error_ppm = (out_mz[0] - true_center) / true_center * 1e6
    assert abs(error_ppm) < 0.1, (
        f"sub-bin centroid error {error_ppm:+.4f} ppm "
        f"(true={true_center:.6f}, got={out_mz[0]:.6f}, "
        f"bin_step={bin_step})")


def test_intensity_output_equals_apex_sample_not_interpolated_apex():
    """Documented contract: ``out_int`` is the apex SAMPLE height
    (``y1``), not the parabolic-interpolated apex. For an offset
    Gaussian where the true apex sits between samples, the sample
    height underestimates the true peak height. This test pins the
    contract so the next person doesn't 'fix' it."""
    bin_step = 0.001
    sigma = 0.0025
    n = 21
    mz = (500.0 + (np.arange(n) - n // 2) * bin_step).astype(np.float64)
    true_center = 500.0 + 0.3 * bin_step
    true_height = 1e6
    intensity = (
        true_height
        * np.exp(-0.5 * ((mz - true_center) / sigma) ** 2)
    ).astype(np.float64)
    apex_sample_idx = int(np.argmax(intensity))
    expected_out_int = intensity[apex_sample_idx]

    _, out_int = centroid_spectrum(mz, intensity, rel_threshold=1e-3)
    assert out_int[0] == pytest.approx(expected_out_int), (
        "out_int must equal the apex SAMPLE height, not the "
        "interpolated parabolic apex")
    # And it must be slightly less than the true height (due to offset).
    assert out_int[0] < true_height


def test_nan_in_intensity_returns_safely():
    """NaN propagates through ``max()`` and comparisons silently; the
    function must NOT crash. It is permitted to return empty (NaN
    poisons the threshold). Production callers must not pass NaN
    intensities — this test pins the safe-fail contract."""
    mz = np.linspace(100.0, 110.0, 11).astype(np.float32)
    intensity = np.array([1.0, 10.0, 1.0, np.nan, 1.0, 8.0, 1.0,
                          1.0, 5.0, 1.0, 1.0], dtype=np.float32)
    # Must not raise.
    out_mz, out_int = centroid_spectrum(mz, intensity)
    # Output may be empty — we only require it to be a well-formed pair.
    assert len(out_mz) == len(out_int)


def test_inf_in_intensity_does_not_crash():
    """+Inf must not crash. max becomes Inf, threshold becomes Inf;
    behaviour-wise everything is filtered (no real peak >= Inf), but
    the function returns a clean empty pair."""
    mz = np.linspace(100.0, 110.0, 11).astype(np.float32)
    intensity = np.array([1.0, 10.0, 1.0, np.inf, 1.0, 8.0, 1.0,
                          1.0, 5.0, 1.0, 1.0], dtype=np.float32)
    out_mz, out_int = centroid_spectrum(mz, intensity)
    assert len(out_mz) == len(out_int)


# ---- Real-data integration sanity (skipped if mzML missing) ----

import os
_REAL_MZML_PATH = os.path.abspath(os.path.join(
    os.path.dirname(__file__), "..", "..",
    "20190830_HF_ZHW_hela_SILAC_DIA_350_1000_Rep1.mzML"))


@pytest.mark.skipif(
    not os.path.exists(_REAL_MZML_PATH),
    reason=f"real mzML fixture not present at {_REAL_MZML_PATH}",
)
def test_real_mzml_centroid_invariants():
    """Integration check on the first few spectra of a real DIA Orbitrap
    profile mzML:

      - centroid count is 1%-30% of profile sample count (typical FTMS)
      - max(out_int) == max(intensity_array) (apex preserved exactly)
      - out_mz is strictly monotonic
      - no NaN/Inf in output
      - out_mz is bounded by [mz.min(), mz.max()] (no extrapolation)

    This is the single best smoke-test of correctness against
    pyteomics-emitted data; if any of these break, the load pipeline
    will produce wrong features downstream.
    """
    from pyteomics import mzml
    n_checked = 0
    with mzml.read(_REAL_MZML_PATH) as r:
        for spectrum in r:
            mz = spectrum['m/z array']
            it = spectrum['intensity array']
            if len(mz) < 100:
                continue  # skip degenerate spectra
            out_mz, out_int = centroid_spectrum(mz, it,
                                                rel_threshold=1e-3)

            ratio = len(out_mz) / len(mz)
            assert 0.01 <= ratio <= 0.30, (
                f"centroid ratio {ratio:.3f} outside expected 1%-30% "
                f"(spectrum has {len(mz)} samples → {len(out_mz)} "
                f"peaks)")
            assert out_int.max() == it.max(), \
                "max intensity must equal input max (apex preserved)"
            assert np.all(np.diff(out_mz) > 0), \
                "output mz must be strictly increasing"
            assert np.all(np.isfinite(out_mz))
            assert np.all(np.isfinite(out_int))
            assert out_mz.min() >= mz.min()
            assert out_mz.max() <= mz.max()

            n_checked += 1
            if n_checked >= 5:
                break
    assert n_checked > 0, "expected to check at least one spectrum"


def test_dia_data_defaults_have_centroid_fields():
    """DIAData() must expose centroid config fields with documented
    defaults: enabled=True, rel_threshold=1e-3."""
    d = DIAData()
    assert d._centroid_enabled is True
    assert d._centroid_rel_threshold == pytest.approx(1e-3)
