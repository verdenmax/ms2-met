"""Physical counterexamples for sensitivity of the original main group's cuts."""
from dataclasses import replace

import numpy as np
import pytest

from tests.test_fragment_structure import fragment, features, trace, DTYPE
from workflows.fragment_reliability import FEATURE_NAMES, _stable_pair
from workflows.fragment_structure import FEATURE_NAMES as QDS_FEATURES


def reliability(records, **kwargs):
    return features(records, include_reliability=True, **kwargs)


def test_single_common_point_has_high_pearson_but_no_stable_cut():
    out = reliability([fragment('y', 3, values=[0, 0, 800, 0, 0, 0, 0])])
    assert out['ms2_structure_main_cut_fraction'] == pytest.approx(1/7)
    assert out[FEATURE_NAMES[0]] == 0
    assert out[FEATURE_NAMES[1]] == 1
    assert out['fragment_reliability_status'] == 'ok'


def test_broad_pair_survives_and_does_not_change_any_qds_value_or_input():
    record = fragment('y', 3, values=[0, 250, 800, 500, 200, 0, 0])
    before = record.light[1].copy()
    old, out = features([record]), reliability([record])
    for key in QDS_FEATURES:
        assert np.isclose(old[key], out[key], equal_nan=True), key
    np.testing.assert_array_equal(before, record.light[1])
    assert out[FEATURE_NAMES[0]] == pytest.approx(1/7)
    assert out[FEATURE_NAMES[1]] == 5/8


def test_remaining_pearson_is_insufficient_if_apices_separate():
    light = trace([200, 0, 1000, 0, 500, 0, 0])
    heavy = trace([500, 0, 1000, 0, 200, 0, 0], heavy=True)
    # After removing cycle 102, Pearson is still positive but apex gap is 4.
    assert not _stable_pair(light, heavy)
    # Exactly-at-floor residual intensity is not a surviving signal.
    assert not _stable_pair(trace([0, 100, 1000, 100, 0]),
                            trace([0, 100, 1000, 100, 0], heavy=True))


def test_cycle_intersection_is_used_instead_of_array_position(monkeypatch):
    from workflows import fragment_reliability as module
    light = trace([1000, 800, 400, 0])
    heavy = trace([800, 400, 0], heavy=True)
    heavy['cycle_idx'] += 1
    seen = []
    def capture(l, h):
        seen.append((l['cycle_idx'].tolist(), h['cycle_idx'].tolist()))
        return True
    monkeypatch.setattr(module, 'is_signal_present_heavy', capture)
    assert _stable_pair(light, heavy)
    assert seen == [([100, 102, 103], [102, 103])]


def test_equal_joint_maxima_break_ties_at_earliest_cycle(monkeypatch):
    from workflows import fragment_reliability as module
    seen = []
    monkeypatch.setattr(module, 'is_signal_present_heavy',
                        lambda l,h: seen.append(l['cycle_idx'].tolist()) or True)
    assert _stable_pair(trace([0, 800, 800, 200, 0]), trace([0, 800, 800, 200, 0], heavy=True))
    assert seen == [[100, 102, 103, 104]]


def test_exact_peak_aliases_remain_ambiguous_after_perturbation():
    a = fragment('b', 2)
    b = fragment('b', 4)
    out = reliability([a, b])
    assert out['ms2_structure_main_ambiguous_cut_fraction'] == 1
    assert out[FEATURE_NAMES[0]] == 0 and out[FEATURE_NAMES[1]] == 1


def test_independent_support_for_one_cut_counts_once_and_can_rescue_it():
    out = reliability([fragment('b', 3, peak_id=1, values=[0, 0, 800, 0, 0, 0, 0]),
                       fragment('y', 5, peak_id=2)])
    assert out['ms2_structure_main_trace_count'] == 2
    assert out[FEATURE_NAMES[0]] == pytest.approx(1/7)
    assert out[FEATURE_NAMES[1]] == 5/8


def test_main_group_is_not_reselected_and_ordinal_one_never_fills_gaps():
    # Original largest group has two fragile targets, another peak is stable.
    records = [fragment('y', 2, peak_id=1, values=[0, 800, 0, 0, 0, 0, 0]),
               fragment('y', 3, peak_id=2, values=[0, 800, 0, 0, 0, 0, 0]),
               fragment('y', 4, peak_id=3, values=[0, 0, 0, 0, 200, 800, 200])]
    out = reliability(records)
    assert out['ms2_structure_main_trace_count'] == 2
    assert out[FEATURE_NAMES[0]] == 0
    short = reliability([fragment('b', 1), fragment('y', 1, peak_id=2)])
    assert short[FEATURE_NAMES[0]] == 0 and short[FEATURE_NAMES[1]] == 1


def test_missing_acquisition_is_na_but_acquired_empty_is_zero_evidence():
    empty = reliability([fragment(values=np.zeros(7))])
    assert empty['fragment_reliability_valid'] == 1
    assert empty[FEATURE_NAMES[0]] == 0 and empty[FEATURE_NAMES[1]] == 1
    missing = fragment()
    missing = replace(missing, heavy={1: np.empty(0, dtype=DTYPE), 2: missing.heavy[2]})
    out = reliability([missing])
    assert out['fragment_reliability_valid'] == 0
    assert out['fragment_reliability_status'] == 'no_ms2_scans'
    assert all(np.isnan(out[key]) for key in FEATURE_NAMES)


def test_charge_one_cannot_borrow_charge_two_and_coisolated_unshifted_is_na():
    f = fragment()
    f = replace(f, light={1: f.light[2], 2: f.light[1]}, heavy={1: f.heavy[2], 2: f.heavy[1]})
    assert reliability([f], charge=1)[FEATURE_NAMES[0]] == 0
    assert reliability([f], charge=2)[FEATURE_NAMES[0]] > 0
    from workflows.fragment_structure import fragment_structure_features
    out = fragment_structure_features('AAAAK', 2, [fragment('b', 2, shifted=False)],
        split_window=False, center_rt=10.08, include_reliability=True)
    assert out['fragment_reliability_status'] == 'no_separable_targets'
    assert np.isnan(out[FEATURE_NAMES[0]])


def test_registry_is_additive_and_excludes_status_from_formal_arms():
    from tools.spec_trainer.src.feature_groups import (
        experiment_arm_features, MS2_RELIABILITY_FEATURES, METADATA_COLUMNS, ELIGIBILITY_FEATURES)
    old = experiment_arm_features('ms1_ms2_qds')
    new = experiment_arm_features('ms1_ms2_qds_r')
    assert new-old == MS2_RELIABILITY_FEATURES == set(FEATURE_NAMES)
    assert 'fragment_reliability_valid' in ELIGIBILITY_FEATURES
    assert {'fragment_reliability_version', 'fragment_reliability_status'} <= METADATA_COLUMNS
    assert not old.intersection(FEATURE_NAMES)
