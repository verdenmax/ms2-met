"""Numerical correctness tests for SILAC / fragment / heavy-mass calculations."""
import numpy as np
import pytest

from spectrum.psm_info import PSMInfo, HeavyType


# Exact UniMod-canonical mass deltas
MASS_DELTA_C13_C12 = 1.003355
MASS_DELTA_N15_N14 = 0.997035
PROTON_MASS = 1.00727646677


def _make_basic_psm(seq="PEPTIDEK", charge=2):
    from pyteomics import mass
    # Seed precursor_mz with the *correct* +charge m/z so heavy-mass math is
    # end-to-end checkable. mass.fast_mass(ion_type='M', charge=z) returns m/z.
    precursor_mz = mass.fast_mass(seq, ion_type='M', charge=charge)
    return PSMInfo(
        sequence=seq, charge=charge, modify=[],
        rt=np.float32(50.0),
        precursor_mz=np.float32(precursor_mz),
        raw_title="r1", protein_names="X_HUMAN",
    )


def test_cheavy_uses_c13_delta_not_proton_mass():
    """CHEAVY heavy precursor mass shift must use 1.003355 (13C-12C),
    not proton mass 1.00727..., per UniMod convention."""
    from pyteomics import mass

    psm = _make_basic_psm(seq="GAVK", charge=2)
    light_neutral = mass.fast_mass("GAVK", ion_type="M", charge=0)
    n_carbons = mass.Composition("GAVK")["C"]

    expected_heavy_neutral = light_neutral + n_carbons * MASS_DELTA_C13_C12
    expected_heavy_mz = (expected_heavy_neutral + 2 * PROTON_MASS) / 2

    heavy_mz = psm.get_C_N_HEAVY_precursor_mz(HeavyType.CHEAVY)
    rel_err = abs(heavy_mz - expected_heavy_mz) / expected_heavy_mz * 1e6
    assert rel_err < 5, (
        f"CHEAVY off by {rel_err:.1f} ppm. "
        f"Got {heavy_mz}, expected {expected_heavy_mz}")


def test_nheavy_uses_n15_delta():
    """NHEAVY heavy precursor mass shift must use 0.997035 (15N-14N),
    per UniMod convention."""
    from pyteomics import mass

    psm = _make_basic_psm(seq="GAVK", charge=2)
    light_neutral = mass.fast_mass("GAVK", ion_type="M", charge=0)
    n_nitrogens = mass.Composition("GAVK")["N"]

    expected_heavy_neutral = light_neutral + n_nitrogens * MASS_DELTA_N15_N14
    expected_heavy_mz = (expected_heavy_neutral + 2 * PROTON_MASS) / 2

    heavy_mz = psm.get_C_N_HEAVY_precursor_mz(HeavyType.NHEAVY)
    rel_err = abs(heavy_mz - expected_heavy_mz) / expected_heavy_mz * 1e6
    assert rel_err < 5, (
        f"NHEAVY off by {rel_err:.1f} ppm. "
        f"Got {heavy_mz}, expected {expected_heavy_mz}")


def test_fragment_ions_store_neutral_mass_not_mz():
    """get_fragment_ions returns neutral masses (charge=0), not +1 m/z.
    This means dia_data.py's formula (ions_mass + z*proton)/z is CORRECT
    (no double-proton bug) — bug #1 was a false positive."""
    from pyteomics import mass

    psm = _make_basic_psm(seq="PEPTIDEK", charge=2)
    b_ions, y_ions = psm.get_fragment_ions(HeavyType.SILAC)

    # b3 = "PEP" prefix (index 0 = b1, 1 = b2, 2 = b3)
    ion_type, position, light_mass, _heavy = b_ions[2]
    assert ion_type == "b"
    assert position == 3

    # fast_mass default (charge=0) returns neutral mass
    expected_neutral = mass.fast_mass("PEP", ion_type="b", charge=0)
    rel_err = abs(light_mass - expected_neutral) / expected_neutral * 1e6
    assert rel_err < 5, (
        f"b3 neutral mass off by {rel_err:.1f} ppm; "
        f"got {light_mass}, expected {expected_neutral}")

    # Verify it is NOT the +1 m/z (which would indicate a double-proton issue)
    wrong_mz = mass.fast_mass("PEP", ion_type="b", charge=1)
    assert abs(light_mass - wrong_mz) > 0.5, (
        "Fragment mass equals +1 m/z — would indicate unexpected charge=1 usage")


def test_psm_info_imports_from_any_cwd(tmp_path, monkeypatch):
    """spectrum.psm_info must import successfully from a non-repo cwd
    (unimod.xml path is resolved relative to __file__, not os.getcwd())."""
    import sys

    monkeypatch.chdir(tmp_path)
    for m in list(sys.modules):
        if m.startswith("spectrum.psm_info"):
            del sys.modules[m]
    import spectrum.psm_info  # noqa: F401  — must not raise FileNotFoundError


def test_calc_xic_score_sorted_unsorted_give_same_pearson():
    """np.interp silently returns wrong values when xp is not sorted.
    calc_xic_score must defend by sorting first."""
    from workflows.single_work import calc_xic_score

    n = 7
    sorted_rt = np.linspace(0.0, 10.0, n).astype("f8")
    intensities_l = np.array([1, 5, 20, 30, 15, 3, 1], dtype="f8")
    intensities_h = np.array([2, 4, 18, 28, 12, 4, 2], dtype="f8")

    dt = [("rt", "f8"), ("ppm_error", "f8"), ("intensity", "f8")]
    light_sorted = np.zeros(n, dtype=dt)
    light_sorted["rt"] = sorted_rt
    light_sorted["intensity"] = intensities_l
    heavy_sorted = light_sorted.copy()
    heavy_sorted["intensity"] = intensities_h

    perm = np.array([3, 1, 0, 5, 2, 6, 4])
    light_unsorted = light_sorted[perm]
    heavy_unsorted = heavy_sorted[perm]

    p_sorted = calc_xic_score(light_sorted, heavy_sorted)
    p_unsorted = calc_xic_score(light_unsorted, heavy_unsorted)

    assert abs(p_sorted["pearson"] - p_unsorted["pearson"]) < 1e-3, (
        f"Pearson changed under permutation: "
        f"sorted={p_sorted['pearson']:.4f} vs "
        f"unsorted={p_unsorted['pearson']:.4f}")


def test_calc_snr_bounded_when_median_zero():
    """Sparse SILAC XIC: 1-2 nonzero scans out of 7 -> median is 0 -> SNR
    must not blow up to 1e10+."""
    from workflows.single_work import _calc_snr

    intensity = np.array([0, 0, 0, 1000, 0, 0, 0], dtype="f4")
    snr = _calc_snr(intensity)
    assert snr < 1e4, f"SNR={snr} blew up (>1e4); need a noise floor"
    assert snr > 5  # still meaningfully > 1


def test_calc_snr_normal_peak():
    """Normal Gaussian-ish peak should produce reasonable SNR."""
    from workflows.single_work import _calc_snr
    intensity = np.array([10, 50, 200, 500, 200, 50, 10], dtype="f4")
    snr = _calc_snr(intensity)
    assert snr > 1
    assert snr < 1e4


def test_calc_snr_empty_or_zero():
    """Edge cases: empty array, all zeros."""
    from workflows.single_work import _calc_snr
    assert _calc_snr(np.array([], dtype="f4")) == 0.0
    assert _calc_snr(np.array([0.0, 0.0, 0.0], dtype="f4")) == 0.0
