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
