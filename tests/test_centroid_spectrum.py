"""Unit tests for spectrum.spectrum_utils.centroid_spectrum."""
import numpy as np
import pytest

from spectrum.spectrum_utils import centroid_spectrum


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


def test_flat_top_no_zero_division():
    """Three equal samples at the apex → parabola denominator = 0, must fall
    back to bin-center m/z without raising ZeroDivisionError."""
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
