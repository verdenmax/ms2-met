import numpy as np
import pytest


def test_match_peak_ppm_reports_intensity_weighted_centroid():
    from spectrum.spectrum_utils import match_peak_ppm

    target = np.float32(500.0)
    mz = np.array([499.995, 500.005], dtype="f8")  # -10 and +10 ppm
    intensity = np.array([9.0, 1.0], dtype="f8")
    ppm, total = match_peak_ppm(mz, intensity, target, np.float32(20.0))
    assert ppm == pytest.approx(-8.0, abs=1e-4)
    assert total == pytest.approx(10.0)


def test_match_peak_ppm_keeps_no_match_distinct_from_zero_error():
    from spectrum.spectrum_utils import match_peak_ppm

    ppm, total = match_peak_ppm(
        np.array([600.0]), np.array([10.0]),
        np.float32(500.0), np.float32(20.0))
    assert np.isnan(ppm)
    assert total == 0.0
