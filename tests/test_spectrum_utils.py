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


def test_match_peak_targets_matches_scalar_contract_independently():
    from spectrum.spectrum_utils import match_peak_ppm, match_peak_targets_ppm

    mz = np.array([100.0, 200.001, 200.002, 300.0], dtype="f8")
    intensity = np.array([1.0, 4.0, 6.0, 2.0], dtype="f8")
    targets = np.array([200.0, 250.0, 300.0], dtype="f8")

    errors, observed = match_peak_targets_ppm(
        mz, intensity, targets, np.float32(20.0))
    expected = [
        match_peak_ppm(mz, intensity, target, np.float32(20.0))
        for target in targets
    ]

    assert errors == pytest.approx([value[0] for value in expected], nan_ok=True)
    assert observed == pytest.approx([value[1] for value in expected])


def test_match_peak_targets_keeps_overlapping_targets_independent():
    from spectrum.spectrum_utils import match_peak_targets_ppm

    errors, observed = match_peak_targets_ppm(
        np.array([500.002]), np.array([7.0]),
        np.array([500.000, 500.004]), np.float32(10.0))

    assert np.isfinite(errors).all()
    assert observed.tolist() == pytest.approx([7.0, 7.0])
