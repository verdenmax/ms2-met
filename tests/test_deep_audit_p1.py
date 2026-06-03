"""Phase 1 (Active Important) behavioral tests for deep audit fixes.

See docs/specs/2026-06-03-deep-audit-fixes-design.md.
"""
import configparser
import numpy as np
import pytest

# Reuse fakes from p0 file by importing.
from tests.test_deep_audit_p0 import (
    _empty_xic, _real_xic, _FakePSM, _FakeDIA, _minimal_config,
)


class _MultiFragPSM:
    """PSM stub returning N fragments — for testing denominator independence."""
    def __init__(self, mz=500.0, rt=10.0, n_fragments=3):
        self._precursor_mz = mz
        self._rt = rt
        self._sequence = "AAAAK"
        self._charge = 2
        self._raw_title = "fake.mzML"
        self._protein_names = "HUMAN"
        self._label_type = "positive"
        self._modify = []
        self._n_fragments = n_fragments

    def get_heavy_info(self, heavy_type):
        # Return n_fragments distinct y-ions with light/heavy mass pairs
        # (non-zero SILAC shift so they're not skipped by same-mass guard).
        frags = [("y", i, 100.0 + i, 108.0 + i)
                 for i in range(1, self._n_fragments + 1)]
        return self._precursor_mz + 4.0, frags


def _assert_denominator_hoisted(pct_3, pct_5, label):
    """Verify that matched_intensity_percent uses a per-PSM denominator
    (P1-1, Units-I1).

    `_FakeDIA.xic_ms2_peaks_extract` returns the same XIC for every
    fragment and a per-call `all_intensity` of 100.0. The audit-correct
    behavior is:

      numerator   = N_fragments * sum(light_xic) + N_fragments * sum(heavy_xic)
                  = N_fragments * (sum(light_xic) + sum(heavy_xic))   # scales w/ N
      denominator = light_all_intensity + heavy_all_intensity = 200    # per-PSM constant
      pct(N)      = N * matched_per_frag / 200                         # linear in N

    Buggy behavior accumulates the denominator inside the loop, so both
    numerator AND denominator scale with N and pct(N) becomes constant
    in N (i.e. pct_5 / pct_3 == 1). This test fails on the buggy code
    and passes after the hoist.
    """
    assert pct_3 is not None and pct_5 is not None, (
        f"[{label}] matched_intensity_percent must be present in both; "
        f"got {pct_3} and {pct_5}")
    assert pct_3 > 0 and pct_5 > 0, (
        f"[{label}] non-positive matched_intensity_percent: "
        f"pct_3={pct_3}, pct_5={pct_5}")
    ratio = pct_5 / pct_3
    # After fix: ratio == 5/3 (numerator scales with N, denominator
    # constant). Before fix: ratio == 1 (both scale with N).
    assert abs(ratio - 5.0 / 3.0) < 0.05, (
        f"[{label}] P1-1: matched_intensity_percent denominator was "
        f"NOT hoisted out of the fragment loop. Expected pct_5/pct_3 "
        f"≈ 5/3 ≈ 1.667 (numerator scales with N, denominator is the "
        f"per-PSM constant light_all+heavy_all). Got "
        f"pct_3={pct_3}, pct_5={pct_5}, ratio={ratio}. A ratio near "
        f"1.0 means the denominator is still being multiplied by "
        f"N_fragments, leaking peptide length into the feature.")


def test_matched_intensity_percent_independent_of_fragment_count():
    """single_pair_work: matched_intensity_percent denominator hoist
    (P1-1, Units-I1)."""
    from workflows.single_work import single_pair_work
    cfg = _minimal_config()

    psm_3 = _MultiFragPSM(n_fragments=3)
    psm_5 = _MultiFragPSM(n_fragments=5)
    dia = _FakeDIA(force_empty=False)

    f_3frags = single_pair_work(psm_3, dia, cfg)
    f_5frags = single_pair_work(psm_5, dia, cfg)

    _assert_denominator_hoisted(
        f_3frags.get("matched_intensity_percent"),
        f_5frags.get("matched_intensity_percent"),
        label="single_pair_work")


def test_matched_intensity_percent_independent_of_fragment_count_multi_batch():
    """multi_batch_work: matched_intensity_percent denominator hoist
    (P1-1, Units-I1)."""
    from workflows.single_work import multi_batch_work
    cfg = _minimal_config()
    psm_3 = _MultiFragPSM(n_fragments=3)
    psm_5 = _MultiFragPSM(n_fragments=5)
    dia = _FakeDIA(force_empty=False)

    f_3frags = multi_batch_work(psm_3, dia, psm_3, dia, cfg)
    f_5frags = multi_batch_work(psm_5, dia, psm_5, dia, cfg)

    _assert_denominator_hoisted(
        f_3frags.get("matched_intensity_percent"),
        f_5frags.get("matched_intensity_percent"),
        label="multi_batch_work")


def test_fragment_empty_branch_aggregates_no_nan_when_all_empty():
    """When all fragments hit empty-XIC, aggregates must be 0.0 (not NaN)
    because all per-fragment lists are appended with zeros (P1-2, Silent-I1)."""
    from workflows.single_work import single_pair_work
    psm = _MultiFragPSM(n_fragments=3)
    dia = _FakeDIA(force_empty=True)
    cfg = _minimal_config()
    features = single_pair_work(psm, dia, cfg)

    # All 3 fragments empty -> all aggregates should be 0.0, not NaN.
    # Pick representative aggregates from each per-fragment list family.
    for key in (
        "all_apex_delta_mean",
        "all_apex_delta_signed_mean",
        "all_mz_avg_err_mean",
        "all_light_apex_cycle_offset_mean",
        "all_heavy_apex_cycle_offset_mean",
        "all_base_to_apex_ratio_mean",
        "all_apex_monotonicity_mean",
        "all_n_peaks_mean",
        "all_smoothness_mean",
    ):
        if key not in features:
            continue  # skip if a column was renamed/missing in current code
        v = features[key]
        assert not (isinstance(v, float) and np.isnan(v)), (
            f"P1-2: {key} should be 0.0 (not NaN) when all fragments "
            f"hit empty-XIC branch; got {v}. Likely cause: per-fragment "
            f"list not appended in empty branch.")


def test_fragment_empty_branch_aggregates_no_nan_multi_batch():
    """Same parity test for multi_batch_work (P1-2)."""
    from workflows.single_work import multi_batch_work
    psm = _MultiFragPSM(n_fragments=3)
    dia = _FakeDIA(force_empty=True)
    cfg = _minimal_config()
    features = multi_batch_work(psm, dia, psm, dia, cfg)

    for key in (
        "all_apex_delta_mean",
        "all_base_to_apex_ratio_mean",
        "all_n_peaks_mean",
    ):
        if key not in features:
            continue
        v = features[key]
        assert not (isinstance(v, float) and np.isnan(v)), (
            f"P1-2 multi_batch_work: {key} should be 0.0 not NaN; got {v}")


class _MixedFragDIA(_FakeDIA):
    """DIA stub where the FIRST fragment returns empty XIC (test the
    bug's mixed case), but subsequent fragments return real XIC.

    Used to verify list-length parity: without the P1-2 fix, the empty
    fragment doesn't append to per-fragment lists, so aggregate mean is
    computed over (N-1) instead of N → biased.
    """
    def __init__(self):
        super().__init__(force_empty=False)
        self._call_count = 0

    def xic_ms2_peaks_extract(self, rt, window, precursor_mz, ions_mass,
                              mass_tol_ppm):
        self._call_count += 1
        # First TWO calls = empty (1 fragment = 2 calls: light + heavy)
        # Subsequent calls = real XIC.
        if self._call_count <= 2:
            return _empty_xic(), 0.0
        return super().xic_ms2_peaks_extract(rt, window, precursor_mz,
                                              ions_mass, mass_tol_ppm)


def test_mixed_empty_and_valid_fragments_list_parity():
    """When some fragments hit empty-XIC and others produce real values,
    per-fragment list lengths must match valid_fragment_ions_num (P1-2).

    Strategy: trigger empty-XIC for the FIRST fragment of 3, valid for
    the other 2. valid_fragment_ions_num counts all 3 (the fragments
    that reached the XIC extraction stage). With the P1-2 fix, the
    empty fragment appends 0 to all per-fragment lists, so aggregates
    are computed over 3 fragments. Without the fix, only 2 would be
    aggregated for the missing lists.

    We verify by computing the expected all_apex_delta_mean: 0 (empty
    fragment) + 0 (real, both peaks at rt=10.0 -> apex_delta=0) + 0
    = 0 / 3 = 0. That's the same as without the fix (0+0/2 = 0).
    So we use n_peaks instead: empty fragment's n_peaks=0; valid
    fragment's n_peaks > 0 because the XIC has a real peak.
    Without fix: n_peaks_mean = (n_real + n_real) / 2 = real value.
    With fix: n_peaks_mean = (0 + n_real + n_real) / 3 = real * 2/3.
    Ratio: with fix is smaller than without fix.

    Instead of asserting exact values (brittle), we monkey-patch
    extract_ion_numeric_features to capture the list lengths.
    """
    from workflows.single_work import single_pair_work
    import workflows.single_work as sw_mod
    captured = {}

    orig_extract = sw_mod.extract_ion_numeric_features

    def spy_extract(values, prefix):
        if prefix == "all_apex_delta":
            captured["apex_delta_len"] = len(values)
        if prefix == "all_n_peaks":
            captured["n_peaks_len"] = len(values)
        return orig_extract(values, prefix)

    psm = _MultiFragPSM(n_fragments=3)
    dia = _MixedFragDIA()

    sw_mod.extract_ion_numeric_features = spy_extract
    try:
        features = single_pair_work(psm, dia, _minimal_config())
    finally:
        sw_mod.extract_ion_numeric_features = orig_extract

    n_valid = features["valid_fragment_ions_num"]
    assert n_valid == 3, (
        f"expected valid_fragment_ions_num=3 (1 empty + 2 valid both "
        f"reach XIC stage); got {n_valid}")
    # P1-2 contract: per-fragment list lengths == valid_fragment_ions_num
    assert captured["apex_delta_len"] == n_valid, (
        f"P1-2: fragment_apex_deltas length ({captured['apex_delta_len']}) "
        f"must equal valid_fragment_ions_num ({n_valid}). Empty-XIC "
        f"fragment was not appended.")
    assert captured["n_peaks_len"] == n_valid, (
        f"P1-2: fragment_n_peaks_list length ({captured['n_peaks_len']}) "
        f"must equal valid_fragment_ions_num ({n_valid}).")
