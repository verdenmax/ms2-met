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


def test_apex_delta_signed_emitted_alongside_unsigned():
    """calc_xic_score should emit apex_delta_signed (preserving sign)
    in addition to apex_delta (abs)."""
    import numpy as np
    from workflows.single_work import calc_xic_score

    dt = [("rt", "f4"), ("intensity", "f4"), ("ppm_error", "f4"), ("mz", "f4")]
    n = 5
    light = np.zeros(n, dtype=dt)
    light["rt"] = [10.0, 11.0, 12.0, 13.0, 14.0]
    light["intensity"] = [1, 5, 100, 5, 1]  # apex at idx 2 (rt=12)
    heavy = light.copy()
    heavy["intensity"] = [1, 100, 5, 1, 1]  # apex at idx 1 (rt=11)
    result = calc_xic_score(light, heavy)
    # heavy apex (11) is earlier than light apex (12) → signed = light - heavy = +1
    assert "apex_delta_signed" in result
    assert abs(result["apex_delta_signed"] - 1.0) < 1e-3
    # Existing unsigned key still works
    assert "apex_delta" in result
    assert abs(result["apex_delta"] - 1.0) < 1e-3


def test_extract_ion_pearson_features_returns_nan_std_for_single_element():
    """N=1 has no defined std spread; returning 0 misleads the model."""
    import math
    from workflows.single_work import extract_ion_pearson_features
    out = extract_ion_pearson_features([0.85])
    assert math.isnan(out["std"]), (
        "Single-element std must be NaN, not 0 (which conflates "
        "'one ion' with 'many ions of identical value')"
    )


def test_calc_xic_score_handles_pearsonr_constant_input():
    """When one XIC is constant, scipy.stats.pearsonr may emit
    ConstantInputWarning and return NaN. calc_xic_score must
    coerce NaN to 0.0."""
    import numpy as np
    from workflows.single_work import calc_xic_score

    dt = [("rt", "f4"), ("intensity", "f4"), ("ppm_error", "f4"), ("mz", "f4")]
    n = 5
    light = np.zeros(n, dtype=dt)
    light["rt"] = [10.0, 11.0, 12.0, 13.0, 14.0]
    light["intensity"] = [1.0, 1.0, 1.0, 1.0, 1.0]  # constant
    heavy = light.copy()
    heavy["intensity"] = [1, 5, 100, 5, 1]
    result = calc_xic_score(light, heavy)
    # pearson should be 0.0, not NaN
    assert np.isfinite(result["pearson"])


def test_extract_ion_numeric_features_emits_max_field():
    """extract_ion_numeric_features must include `_max` so caller can
    track the worst-offending fragment (e.g. largest apex_cycle_offset)."""
    from workflows.single_work import extract_ion_numeric_features
    out = extract_ion_numeric_features([0.1, 0.5, 0.3, 0.9, 0.2], "demo")
    assert "demo_max" in out
    assert abs(out["demo_max"] - 0.9) < 1e-9
    assert "demo_mean" in out  # existing fields preserved
    assert "demo_p50" in out
    assert "demo_std" in out


def test_extract_ion_numeric_features_max_empty_list_is_zero():
    """Empty list -> demo_max = 0.0 (consistent with other defaults)."""
    from workflows.single_work import extract_ion_numeric_features
    out = extract_ion_numeric_features([], "demo")
    assert out["demo_max"] == 0.0


def _make_xic(cycles, rts, intensities):
    dt = [("rt", "f8"), ("ppm_error", "f8"),
          ("intensity", "f8"), ("cycle_idx", "i4")]
    arr = np.zeros(len(cycles), dtype=dt)
    arr["cycle_idx"] = cycles
    arr["rt"] = rts
    arr["intensity"] = intensities
    return arr


def test_calc_cycle_offset_apex_at_center_returns_zero():
    """When apex aligns with the center RT entry, offset is (0, 0)."""
    from workflows.single_work import _calc_cycle_offset
    xic = _make_xic(
        cycles=[5, 6, 7, 8, 9], rts=[10, 11, 12, 13, 14],
        intensities=[1, 5, 100, 5, 1])
    abs_off, signed = _calc_cycle_offset(xic, center_rt=12.0)
    assert abs_off == 0
    assert signed == 0


def test_calc_cycle_offset_apex_before_center_is_negative():
    """Apex one cycle earlier than center -> signed = -1, abs = 1."""
    from workflows.single_work import _calc_cycle_offset
    xic = _make_xic(
        cycles=[5, 6, 7, 8, 9], rts=[10, 11, 12, 13, 14],
        intensities=[1, 100, 5, 1, 1])  # apex at cycle 6
    abs_off, signed = _calc_cycle_offset(xic, center_rt=12.0)
    assert signed == -1
    assert abs_off == 1


def test_calc_cycle_offset_apex_after_center_is_positive():
    """Apex two cycles after center -> signed = +2, abs = 2."""
    from workflows.single_work import _calc_cycle_offset
    xic = _make_xic(
        cycles=[5, 6, 7, 8, 9], rts=[10, 11, 12, 13, 14],
        intensities=[1, 1, 5, 1, 100])  # apex at cycle 9
    abs_off, signed = _calc_cycle_offset(xic, center_rt=12.0)
    assert signed == 2
    assert abs_off == 2


def test_calc_cycle_offset_empty_xic_returns_zero():
    """Empty XIC -> (0, 0)."""
    from workflows.single_work import _calc_cycle_offset
    xic = _make_xic(cycles=[], rts=[], intensities=[])
    abs_off, signed = _calc_cycle_offset(xic, center_rt=0.0)
    assert (abs_off, signed) == (0, 0)


def test_calc_cycle_offset_skips_invalid_cycle_idx():
    """cycle_idx == -1 entries (defensive) are excluded from center search,
    and an apex with cycle_idx == -1 returns (0, 0)."""
    from workflows.single_work import _calc_cycle_offset
    # Center RT entry has cycle_idx == -1 -> picks next-closest valid entry
    xic = _make_xic(
        cycles=[5, -1, 7, 8, 9], rts=[10, 11, 12, 13, 14],
        intensities=[1, 5, 100, 5, 1])  # apex at idx 2 (cycle 7)
    abs_off, signed = _calc_cycle_offset(xic, center_rt=11.0)
    # All cycle_idx>=0 entries: cycles [5,7,8,9] at rts [10,12,13,14]
    # Closest to 11 is rt=10 (cycle 5) or rt=12 (cycle 7), tie -> argmin picks rt=10
    # apex cycle = 7, center cycle = 5 -> signed = 2
    assert signed == 2
    assert abs_off == 2

    # If apex itself has cycle_idx -1 -> return (0, 0)
    xic2 = _make_xic(
        cycles=[5, 6, -1, 8, 9], rts=[10, 11, 12, 13, 14],
        intensities=[1, 1, 100, 1, 1])  # apex at idx 2 (cycle_idx -1)
    abs_off2, signed2 = _calc_cycle_offset(xic2, center_rt=11.0)
    assert (abs_off2, signed2) == (0, 0)


def test_calc_xic_score_emits_cycle_offset_when_center_rt_provided():
    """calc_xic_score(light, heavy, center_rt=...) returns 4 new fields:
    light_apex_cycle_offset, light_apex_cycle_offset_signed,
    heavy_apex_cycle_offset, heavy_apex_cycle_offset_signed."""
    from workflows.single_work import calc_xic_score
    dt = [("rt", "f8"), ("ppm_error", "f8"),
          ("intensity", "f8"), ("cycle_idx", "i4")]
    n = 5
    light = np.zeros(n, dtype=dt)
    light["rt"] = [10, 11, 12, 13, 14]
    light["cycle_idx"] = [0, 1, 2, 3, 4]
    light["intensity"] = [1, 5, 100, 5, 1]  # apex at cycle 2 (center)
    heavy = light.copy()
    heavy["intensity"] = [1, 100, 5, 1, 1]  # apex at cycle 1 (one early)

    result = calc_xic_score(light, heavy, center_rt=12.0)
    assert result["light_apex_cycle_offset"] == 0
    assert result["light_apex_cycle_offset_signed"] == 0
    assert result["heavy_apex_cycle_offset"] == 1
    assert result["heavy_apex_cycle_offset_signed"] == -1


def test_calc_xic_score_supports_separate_heavy_center_rt():
    """When light and heavy come from different DIAData (multi_batch_work),
    heavy may have a different RT origin."""
    from workflows.single_work import calc_xic_score
    dt = [("rt", "f8"), ("ppm_error", "f8"),
          ("intensity", "f8"), ("cycle_idx", "i4")]
    n = 5
    light = np.zeros(n, dtype=dt)
    light["rt"] = [10, 11, 12, 13, 14]
    light["cycle_idx"] = [0, 1, 2, 3, 4]
    light["intensity"] = [1, 5, 100, 5, 1]

    # Heavy XIC at a different RT range (different raw)
    heavy = np.zeros(n, dtype=dt)
    heavy["rt"] = [20, 21, 22, 23, 24]
    heavy["cycle_idx"] = [10, 11, 12, 13, 14]
    heavy["intensity"] = [1, 5, 100, 5, 1]  # apex at cycle 12

    result = calc_xic_score(
        light, heavy, center_rt=12.0, heavy_center_rt=22.0)
    assert result["light_apex_cycle_offset"] == 0
    assert result["heavy_apex_cycle_offset"] == 0


def test_calc_xic_score_omits_cycle_offset_default_zero():
    """Backwards compat: not passing center_rt returns zero for new fields."""
    from workflows.single_work import calc_xic_score
    dt = [("rt", "f8"), ("ppm_error", "f8"),
          ("intensity", "f8"), ("cycle_idx", "i4")]
    n = 3
    light = np.zeros(n, dtype=dt)
    light["rt"] = [10, 11, 12]
    light["cycle_idx"] = [0, 1, 2]
    light["intensity"] = [1, 100, 1]
    heavy = light.copy()
    result = calc_xic_score(light, heavy)
    assert result["light_apex_cycle_offset"] == 0
    assert result["light_apex_cycle_offset_signed"] == 0
    assert result["heavy_apex_cycle_offset"] == 0
    assert result["heavy_apex_cycle_offset_signed"] == 0


def test_default_xic_score_has_cycle_offset_zero_fields():
    """The early-return default dict must include all 4 cycle offset keys."""
    from workflows.single_work import _default_xic_score
    d = _default_xic_score()
    assert d["light_apex_cycle_offset"] == 0
    assert d["light_apex_cycle_offset_signed"] == 0
    assert d["heavy_apex_cycle_offset"] == 0
    assert d["heavy_apex_cycle_offset_signed"] == 0


def test_calc_hl_ratio_consistency_basic_std_and_mad():
    """Returns (std, mad) of log10(ratios > 0)."""
    from workflows.single_work import _calc_hl_ratio_consistency
    # ratios: 1, 10, 100 -> log10 = 0, 1, 2 -> mean=1, std=sqrt(2/3)
    std_v, mad_v = _calc_hl_ratio_consistency([1.0, 10.0, 100.0])
    assert abs(std_v - np.std([0.0, 1.0, 2.0])) < 1e-9
    # median = 1.0 -> |log10-median| = [1, 0, 1] -> mad = median = 1.0
    assert abs(mad_v - 1.0) < 1e-9


def test_calc_hl_ratio_consistency_drops_non_positive():
    """ratios <= 0 are excluded from log10."""
    from workflows.single_work import _calc_hl_ratio_consistency
    std_v, mad_v = _calc_hl_ratio_consistency([1.0, 0.0, -5.0, 100.0])
    # Only [1, 100] survive -> log10 = [0, 2] -> std=1, median=1, mad=1
    assert abs(std_v - 1.0) < 1e-9
    assert abs(mad_v - 1.0) < 1e-9


def test_calc_hl_ratio_consistency_empty_list_returns_zero():
    """Empty list -> (0.0, 0.0)."""
    from workflows.single_work import _calc_hl_ratio_consistency
    assert _calc_hl_ratio_consistency([]) == (0.0, 0.0)
    assert _calc_hl_ratio_consistency([0.0, -1.0]) == (0.0, 0.0)


def test_calc_hl_ratio_consistency_single_element_std_is_nan():
    """count=1 -> std is NaN (consistent with Bug #21 convention); mad=0."""
    import math
    from workflows.single_work import _calc_hl_ratio_consistency
    std_v, mad_v = _calc_hl_ratio_consistency([5.0])
    assert math.isnan(std_v)
    assert mad_v == 0.0


def test_calc_cycle_offset_all_zero_intensity_returns_zero():
    """When XIC has entries but every intensity is 0 (no peak matched),
    np.argmax would silently pick the left-edge cycle and emit a fake
    non-zero offset. Guard returns (0, 0) instead."""
    from workflows.single_work import _calc_cycle_offset
    # XIC populated but every intensity == 0 (typical for MS2 fragment
    # XICs that find spectra but no peak matches the fragment m/z)
    xic = _make_xic(
        cycles=[10, 11, 12, 13, 14], rts=[20, 21, 22, 23, 24],
        intensities=[0, 0, 0, 0, 0])
    abs_off, signed = _calc_cycle_offset(xic, center_rt=22.0)
    assert abs_off == 0
    assert signed == 0


def test_calc_cycle_offset_zero_intensity_with_nonzero_apex_still_works():
    """Sanity: the guard only triggers when ALL intensities are 0;
    a single non-zero entry still produces a meaningful offset."""
    from workflows.single_work import _calc_cycle_offset
    xic = _make_xic(
        cycles=[10, 11, 12, 13, 14], rts=[20, 21, 22, 23, 24],
        intensities=[0, 0, 0, 0, 100])  # apex at cycle 14
    abs_off, signed = _calc_cycle_offset(xic, center_rt=22.0)
    # Center = cycle 12 (rt=22), apex = cycle 14 -> signed = 2
    assert signed == 2
    assert abs_off == 2


def test_calc_base_to_apex_ratio_real_peak_returns_low_value():
    """Real chromatographic peak: edges decay to near-zero -> ratio close to 0."""
    from workflows.single_work import _calc_base_to_apex_ratio
    intensity = np.array([1, 5, 50, 100, 50, 5, 1], dtype="f8")
    ratio = _calc_base_to_apex_ratio(intensity)
    # base = (1+1)/2 = 1, apex = 100, ratio = 0.01
    assert ratio < 0.05


def test_calc_base_to_apex_ratio_plateau_returns_high():
    """Plateau / continuous background: edges are nearly as high as apex."""
    from workflows.single_work import _calc_base_to_apex_ratio
    intensity = np.array([80, 90, 100, 100, 90, 80, 80], dtype="f8")
    ratio = _calc_base_to_apex_ratio(intensity)
    # base = (80+80)/2 = 80, apex = 100, ratio = 0.8
    assert ratio > 0.7


def test_calc_base_to_apex_ratio_edge_cases():
    """Empty / short / all-zero XIC returns 0.0."""
    from workflows.single_work import _calc_base_to_apex_ratio
    assert _calc_base_to_apex_ratio(np.array([], dtype="f8")) == 0.0
    assert _calc_base_to_apex_ratio(np.array([1, 2], dtype="f8")) == 0.0
    assert _calc_base_to_apex_ratio(np.array([0, 0, 0], dtype="f8")) == 0.0


def test_calc_apex_monotonicity_perfect_peak_returns_one():
    """Strictly monotonic up to apex, then strictly down: monotonicity = 1.0."""
    from workflows.single_work import _calc_apex_monotonicity
    intensity = np.array([1, 5, 50, 100, 50, 5, 1], dtype="f8")
    # All 6 diffs go in the "right" direction -> 0 violations -> 1.0
    assert _calc_apex_monotonicity(intensity) == 1.0


def test_calc_apex_monotonicity_zigzag_returns_low():
    """Zigzag (multiple direction reversals) returns low monotonicity."""
    from workflows.single_work import _calc_apex_monotonicity
    intensity = np.array([10, 50, 20, 100, 30, 60, 5], dtype="f8")
    # apex at idx 3, left=[10,50,20,100] diffs [40,-30,80] -> 1 viol (50->20)
    # right=[100,30,60,5] diffs [-70,30,-55] -> 1 viol (30->60)
    # total_pairs = 6, violations = 2, monotonicity = 1 - 2/6 ~= 0.667
    result = _calc_apex_monotonicity(intensity)
    assert 0.5 < result < 0.75


def test_calc_apex_monotonicity_apex_at_edge():
    """Apex at index 0 or last index: function handles boundary safely."""
    from workflows.single_work import _calc_apex_monotonicity
    # Apex at idx 0: left=[100], right=[100,50,10] diffs [-50,-40] all <=0 -> 0 viol
    result_left = _calc_apex_monotonicity(
        np.array([100, 50, 10], dtype="f8"))
    assert result_left == 1.0
    # Apex at last idx: left=[10,50,100] diffs [40,50] -> 0 viol, right=[100] -> 0 pairs
    result_right = _calc_apex_monotonicity(
        np.array([10, 50, 100], dtype="f8"))
    assert result_right == 1.0
    # Edge: empty / short
    assert _calc_apex_monotonicity(np.array([], dtype="f8")) == 0.0
    assert _calc_apex_monotonicity(np.array([1, 2], dtype="f8")) == 0.0


def test_calc_apex_monotonicity_nan_returns_zero():
    """NaN in XIC must return 0.0 (defensive), NOT silently return 1.0.

    Without a finiteness guard, np.argmax(NaN-containing) returns the
    NaN index and np.diff comparisons against NaN are always False,
    making garbage input look like a perfect peak. This is exactly the
    wrong direction for a peak-likeness metric.
    """
    import math
    from workflows.single_work import _calc_apex_monotonicity
    # NaN in middle
    assert _calc_apex_monotonicity(
        np.array([1, 5, np.nan, 100, 50, 5, 1], dtype="f8")) == 0.0
    # All-NaN
    assert _calc_apex_monotonicity(
        np.array([np.nan, np.nan, np.nan], dtype="f8")) == 0.0
    # NaN at edge
    assert _calc_apex_monotonicity(
        np.array([np.nan, 50, 100, 50, 10], dtype="f8")) == 0.0
    # Inf also non-finite
    assert _calc_apex_monotonicity(
        np.array([1, 5, np.inf, 5, 1], dtype="f8")) == 0.0


def test_calc_n_peaks_single_peak_returns_one():
    """Classic unimodal peak: find_peaks returns 1 local maximum."""
    from workflows.single_work import _calc_n_peaks
    intensity = np.array([1, 5, 50, 100, 50, 5, 1], dtype="f8")
    assert _calc_n_peaks(intensity) == 1


def test_calc_n_peaks_bimodal_returns_two():
    """Two well-separated peaks both with prominence > 0.3 * apex."""
    from workflows.single_work import _calc_n_peaks
    intensity = np.array([1, 100, 1, 1, 80, 1, 1], dtype="f8")
    # Two peaks both >= 80 (>= 0.3 * 100 = 30) and dipping back to 1 between
    assert _calc_n_peaks(intensity) == 2


def test_calc_n_peaks_small_noise_suppressed_by_prominence():
    """Small noise bump << 30% of apex should be filtered out."""
    from workflows.single_work import _calc_n_peaks
    # Main peak at idx 3 with apex=100; small bump at idx 6 with height=20 (20%)
    intensity = np.array([1, 50, 100, 50, 1, 1, 20, 1], dtype="f8")
    # Only the main peak should count
    assert _calc_n_peaks(intensity) == 1


def test_calc_n_peaks_edge_cases():
    """Empty / short / all-zero XIC returns 0."""
    from workflows.single_work import _calc_n_peaks
    assert _calc_n_peaks(np.array([], dtype="f8")) == 0
    assert _calc_n_peaks(np.array([1, 2], dtype="f8")) == 0
    assert _calc_n_peaks(np.array([0, 0, 0, 0, 0], dtype="f8")) == 0


def test_calc_smoothness_smooth_curve_low_value():
    """Smooth Gaussian-like peak has low smoothness value."""
    from workflows.single_work import _calc_smoothness
    intensity = np.array([1, 10, 40, 80, 100, 80, 40, 10, 1], dtype="f8")
    # Compute: sum of (second diff)^2 / total^2 - should be small
    s = _calc_smoothness(intensity)
    assert 0 < s < 0.05


def test_calc_smoothness_zigzag_high_value():
    """Sharp zigzag / single-point spike has high smoothness value."""
    from workflows.single_work import _calc_smoothness
    smooth = _calc_smoothness(
        np.array([1, 10, 40, 80, 100, 80, 40, 10, 1], dtype="f8"))
    zigzag = _calc_smoothness(
        np.array([1, 100, 1, 100, 1, 100, 1], dtype="f8"))
    assert zigzag > 10 * smooth  # zigzag should be at least 10x smoother peak


def test_calc_smoothness_edge_cases():
    """Empty / short / all-zero XIC returns 0.0."""
    from workflows.single_work import _calc_smoothness
    assert _calc_smoothness(np.array([], dtype="f8")) == 0.0
    assert _calc_smoothness(np.array([1, 2], dtype="f8")) == 0.0
    assert _calc_smoothness(np.array([0, 0, 0, 0, 0], dtype="f8")) == 0.0


def test_calc_base_to_apex_ratio_nan_returns_zero():
    """NaN in XIC must return 0.0 (defensive consistent with sibling helpers).

    Without a guard, np.max(NaN-containing) returns NaN, apex <= 0 is False
    (NaN compares False), and nan / nan = nan would pollute downstream.
    """
    from workflows.single_work import _calc_base_to_apex_ratio
    # NaN in middle
    assert _calc_base_to_apex_ratio(
        np.array([1, 5, np.nan, 100, 50, 5, 1], dtype="f8")) == 0.0
    # All-NaN
    assert _calc_base_to_apex_ratio(
        np.array([np.nan, np.nan, np.nan], dtype="f8")) == 0.0
    # Inf
    assert _calc_base_to_apex_ratio(
        np.array([1, 5, np.inf, 5, 1], dtype="f8")) == 0.0


def test_calc_xic_score_emits_peak_likeness_fields():
    """calc_xic_score returns 4 new peak-likeness fields, all computed
    on the heavy XIC (consistent with existing snr/peak_symmetry)."""
    from workflows.single_work import calc_xic_score
    dt = [("rt", "f8"), ("ppm_error", "f8"),
          ("intensity", "f8"), ("cycle_idx", "i4")]
    n = 7
    light = np.zeros(n, dtype=dt)
    light["rt"] = [10, 11, 12, 13, 14, 15, 16]
    light["cycle_idx"] = [0, 1, 2, 3, 4, 5, 6]
    light["intensity"] = [1, 5, 50, 100, 50, 5, 1]
    heavy = light.copy()
    result = calc_xic_score(light, heavy)
    assert "base_to_apex_ratio" in result
    assert result["base_to_apex_ratio"] < 0.05
    assert "apex_monotonicity" in result
    assert result["apex_monotonicity"] == 1.0
    assert "n_peaks" in result
    assert result["n_peaks"] == 1
    assert "smoothness" in result
    assert 0 < result["smoothness"] < 0.5


def test_default_xic_score_has_peak_likeness_zero_fields():
    """The default-zero dict must include all 4 new peak-likeness keys."""
    from workflows.single_work import _default_xic_score
    d = _default_xic_score()
    assert d["base_to_apex_ratio"] == 0.0
    assert d["apex_monotonicity"] == 0.0
    assert d["n_peaks"] == 0
    assert d["smoothness"] == 0.0


def test_multi_batch_work_emits_R4_precursor_keys_in_empty_xic_branch():
    """multi_batch_work must emit the 4 R4 precursor keys (I-F1 regression).

    Trigger the empty-XIC branch using fake DIAData / PSMInfo stubs whose
    xic_peaks_extreact returns an empty structured array. Behavioral test
    (not source-string match), so a typo in the key name would be caught.
    """
    import configparser
    import numpy as np
    from workflows.single_work import multi_batch_work

    XIC_DTYPE = [("rt", "f8"), ("ppm_error", "f8"),
                 ("intensity", "f8"), ("cycle_idx", "i4")]

    class FakeDIA:
        def xic_peaks_extreact(self, *a, **kw):
            return np.array([], dtype=XIC_DTYPE)

        def xic_ms2_peaks_extract(self, *a, **kw):
            return np.array([], dtype=XIC_DTYPE), 0.0

        def get_window_info(self, precursor_mz):
            return {"width": 0.0, "centering": 0.0,
                    "lower": 0.0, "upper": 0.0, "center": 0.0}

        def check_in_raw(self, precursor_mz):
            return False

    class FakePSM:
        _sequence = "AAAA"
        _charge = 2
        _modify = []
        _rt = np.float32(10.0)
        _precursor_mz = np.float32(1e9)

        def get_heavy_info(self, heavy_type):
            return self._precursor_mz, []

    cfg = configparser.ConfigParser()
    cfg.read_dict({
        "general": {
            "mass_tol_ppm": "20",
            "xic_cycle_window": "5",
        },
    })

    features = multi_batch_work(FakePSM(), FakeDIA(), FakePSM(), FakeDIA(), cfg)

    R4_PRECURSOR_KEYS = {
        "precursor_base_to_apex_ratio",
        "precursor_apex_monotonicity",
        "precursor_n_peaks",
        "precursor_smoothness",
    }
    missing = R4_PRECURSOR_KEYS - set(features.keys())
    assert not missing, (
        f"multi_batch_work missing R4 precursor keys {missing} (I-F1). "
        f"Schema drift with single_pair_work — concat will produce NaN columns.")


def test_robust_apex_idx_single_peak():
    from workflows.single_work import _robust_apex_idx
    assert _robust_apex_idx(np.array([1, 3, 9, 3, 1], dtype="f8")) == 2


def test_robust_apex_idx_flat_top_returns_center_not_left_edge():
    """平顶并列时取近峰顶区中位索引，而非 argmax 的最左点。"""
    from workflows.single_work import _robust_apex_idx
    # 三点等高平顶在 idx 2,3,4 -> 期望 apex_idx == 3
    assert _robust_apex_idx(
        np.array([1, 3, 5, 5, 5, 3, 1], dtype="f8")) == 3


def test_robust_apex_idx_empty_and_allzero():
    from workflows.single_work import _robust_apex_idx
    assert _robust_apex_idx(np.array([], dtype="f8")) == 0
    assert _robust_apex_idx(np.array([0, 0, 0], dtype="f8")) == 0


def test_robust_apex_idx_nonfinite_falls_back_to_argmax():
    from workflows.single_work import _robust_apex_idx
    arr = np.array([1, np.nan, 9, 1], dtype="f8")
    # argmax on NaN-containing array is implementation-defined but must not raise
    idx = _robust_apex_idx(arr)
    assert isinstance(idx, int)


def test_shape_irregularity_clean_peak_is_zero():
    from workflows.single_work import _calc_shape_irregularity
    assert _calc_shape_irregularity(
        np.array([1, 3, 9, 3, 1], dtype="f8")) == 0.0


def test_shape_irregularity_monotone_ramp_is_zero():
    """纯单调斜坡本身不算'形状不规则'（由 centering_defect 来抓）。"""
    from workflows.single_work import _calc_shape_irregularity
    assert _calc_shape_irregularity(
        np.array([9, 7, 5, 3, 1], dtype="f8")) == 0.0


def test_shape_irregularity_flat_top_penalized():
    """平顶持平 (diff==0) 计违例 -> 非零。"""
    from workflows.single_work import _calc_shape_irregularity
    r = _calc_shape_irregularity(np.array([1, 3, 5, 5, 5, 3, 1], dtype="f8"))
    assert r > 0.0


def test_shape_irregularity_zigzag_high():
    from workflows.single_work import _calc_shape_irregularity
    r = _calc_shape_irregularity(np.array([1, 5, 2, 6, 1], dtype="f8"))
    assert r >= 0.25


def test_shape_irregularity_short_or_nonfinite_zero():
    from workflows.single_work import _calc_shape_irregularity
    assert _calc_shape_irregularity(np.array([1, 2], dtype="f8")) == 0.0
    assert _calc_shape_irregularity(
        np.array([1, np.nan, 3], dtype="f8")) == 0.0


def test_narrow_defect_single_cycle_spike_is_one():
    from workflows.single_work import _calc_narrow_defect
    assert _calc_narrow_defect(np.array([0, 0, 9, 0, 0], dtype="f8")) == 1.0


def test_narrow_defect_broad_peak_low():
    from workflows.single_work import _calc_narrow_defect
    # support (>=0.5*max) = 4 cycles -> 0.25
    r = _calc_narrow_defect(np.array([2, 5, 9, 9, 8, 5, 1], dtype="f8"))
    assert r <= 0.34


def test_narrow_defect_short_zero_nonfinite():
    from workflows.single_work import _calc_narrow_defect
    assert _calc_narrow_defect(np.array([1, 2], dtype="f8")) == 0.0
    assert _calc_narrow_defect(np.array([0, 0, 0], dtype="f8")) == 0.0
    assert _calc_narrow_defect(np.array([1, np.inf, 1], dtype="f8")) == 0.0


def test_centering_defect_apex_at_center_is_zero():
    from workflows.single_work import _calc_centering_defect
    xic = _make_xic(cycles=[5, 6, 7, 8, 9], rts=[10, 11, 12, 13, 14],
                    intensities=[1, 5, 100, 5, 1])  # apex at center rt=12
    assert _calc_centering_defect(xic, center_rt=12.0) == 0.0


def test_centering_defect_apex_at_edge_near_one():
    from workflows.single_work import _calc_centering_defect
    # apex at last cycle (rt=14); center rt=12 -> offset 2, half=(5-1)/2=2 -> 1.0
    xic = _make_xic(cycles=[5, 6, 7, 8, 9], rts=[10, 11, 12, 13, 14],
                    intensities=[1, 1, 5, 1, 100])
    assert _calc_centering_defect(xic, center_rt=12.0) == 1.0


def test_centering_defect_empty_is_zero():
    from workflows.single_work import _calc_centering_defect
    xic = _make_xic(cycles=[], rts=[], intensities=[])
    assert _calc_centering_defect(xic, center_rt=0.0) == 0.0


def test_extract_ion_numeric_features_stats_subset():
    from workflows.single_work import extract_ion_numeric_features
    out = extract_ion_numeric_features(
        [0.1, 0.5, 0.3, 0.9], "demo", stats=("mean", "max"))
    assert set(out.keys()) == {"demo_mean", "demo_max"}
    assert abs(out["demo_max"] - 0.9) < 1e-9


def test_extract_ion_numeric_features_stats_subset_empty():
    from workflows.single_work import extract_ion_numeric_features
    out = extract_ion_numeric_features([], "demo", stats=("mean", "max"))
    assert out == {"demo_mean": 0.0, "demo_max": 0.0}


def test_extract_ion_numeric_features_default_still_four():
    from workflows.single_work import extract_ion_numeric_features
    out = extract_ion_numeric_features([1.0, 2.0, 3.0], "demo")
    assert set(out.keys()) == {
        "demo_mean", "demo_p50", "demo_std", "demo_max"}


def _peak_xic(intensities, cycles=None, rts=None):
    dt = [("rt", "f8"), ("ppm_error", "f8"),
          ("intensity", "f8"), ("cycle_idx", "i4")]
    n = len(intensities)
    arr = np.zeros(n, dtype=dt)
    arr["intensity"] = intensities
    arr["rt"] = rts if rts is not None else np.arange(n, dtype="f8") + 10.0
    arr["cycle_idx"] = cycles if cycles is not None else np.arange(n)
    return arr


def test_calc_xic_score_emits_new_shape_defect_keys():
    from workflows.single_work import calc_xic_score
    light = _peak_xic([1, 5, 50, 100, 50, 5, 1])
    heavy = light.copy()
    r = calc_xic_score(light, heavy, center_rt=13.0)
    for k in ("light_centering_defect", "heavy_centering_defect",
              "light_shape_irregularity", "heavy_shape_irregularity",
              "light_base_to_apex_ratio", "light_n_peaks",
              "light_smoothness", "light_narrow_defect",
              "heavy_narrow_defect"):
        assert k in r, f"missing {k}"
    # clean centered peak -> defects all low
    assert r["heavy_shape_irregularity"] == 0.0
    assert r["heavy_centering_defect"] == 0.0


def test_calc_xic_score_default_has_new_shape_defect_keys():
    from workflows.single_work import _default_xic_score
    d = _default_xic_score()
    for k in ("light_centering_defect", "heavy_centering_defect",
              "light_shape_irregularity", "heavy_shape_irregularity",
              "light_base_to_apex_ratio", "light_n_peaks",
              "light_smoothness", "light_narrow_defect",
              "heavy_narrow_defect"):
        assert d[k] == 0 or d[k] == 0.0, f"{k} default not zero"


def test_precursor_shape_cols_keys_and_values():
    from workflows.single_work import (
        _precursor_shape_cols, _empty_precursor_shape_cols)
    score = {
        "light_centering_defect": 0.2, "heavy_centering_defect": 0.3,
        "light_shape_irregularity": 0.1, "heavy_shape_irregularity": 0.4,
        "light_base_to_apex_ratio": 0.5, "light_n_peaks": 2,
        "light_smoothness": 0.6, "light_narrow_defect": 0.7,
        "heavy_narrow_defect": 0.8,
    }
    cols = _precursor_shape_cols(score)
    assert cols["precursor_heavy_shape_irregularity"] == 0.4
    assert cols["precursor_light_n_peaks"] == 2
    assert cols["precursor_heavy_narrow_defect"] == 0.8
    empty = _empty_precursor_shape_cols()
    assert set(empty.keys()) == set(cols.keys())
    assert all(v == 0 or v == 0.0 for v in empty.values())
    assert "precursor_apex_monotonicity" not in cols
