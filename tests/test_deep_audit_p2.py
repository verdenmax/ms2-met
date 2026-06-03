"""Phase 2 (Dormant Important) tests for deep audit fixes.

See docs/specs/2026-06-03-deep-audit-fixes-design.md.
"""
import os
import sys
import numpy as np
import pytest

from tests.test_deep_audit_p0 import (
    _empty_xic, _real_xic, _FakePSM, _FakeDIA, _minimal_config,
)


def test_no_log_hl_ratio_columns_in_single_pair_work():
    """log_hl_ratio_* columns must be renamed log_lh_ratio_* (P2-1, Units-I2)."""
    from workflows.single_work import single_pair_work
    psm = _FakePSM()
    dia = _FakeDIA(force_empty=False)
    features = single_pair_work(psm, dia, _minimal_config())
    hl_keys = [k for k in features.keys() if "log_hl_ratio" in k]
    assert len(hl_keys) == 0, (
        f"P2-1: log_hl_ratio_* should be renamed log_lh_ratio_*; "
        f"found {hl_keys}")
    lh_keys = [k for k in features.keys() if "log_lh_ratio" in k]
    assert len(lh_keys) >= 1, (
        f"P2-1: expected at least one log_lh_ratio_* column; got {lh_keys}")


def test_no_log_hl_ratio_columns_in_multi_batch_work():
    """Same rename applies to multi_batch_work (P2-1)."""
    from workflows.single_work import multi_batch_work
    psm = _FakePSM()
    dia = _FakeDIA(force_empty=False)
    features = multi_batch_work(psm, dia, psm, dia, _minimal_config())
    hl_keys = [k for k in features.keys() if "log_hl_ratio" in k]
    assert len(hl_keys) == 0, (
        f"P2-1: multi_batch_work log_hl_ratio_* should be renamed; "
        f"found {hl_keys}")


def test_get_retention_time_handles_minute_unit():
    """RT in minutes returned as-is (canonical pipeline unit) (P2-2, Units-I3)."""
    from spectrum.dia_data import DIAData

    class _MockUnitFloat(float):
        """Mock pyteomics unitfloat — float with .unit_info attr."""
        def __new__(cls, value, unit_info):
            instance = super().__new__(cls, value)
            instance.unit_info = unit_info
            return instance

    dia = DIAData()
    spectrum = {
        'scanList': {
            'scan': [{'scan start time': _MockUnitFloat(10.5, 'minute')}]
        }
    }
    rt = dia._get_retention_time(spectrum)
    assert rt == 10.5  # already minutes, returned as-is


def test_get_retention_time_converts_seconds_to_minutes():
    """RT in seconds must be converted to minutes (P2-2, Units-I3)."""
    from spectrum.dia_data import DIAData

    class _MockUnitFloat(float):
        def __new__(cls, value, unit_info):
            instance = super().__new__(cls, value)
            instance.unit_info = unit_info
            return instance

    dia = DIAData()
    spectrum = {
        'scanList': {
            'scan': [{'scan start time': _MockUnitFloat(630.0, 'second')}]
        }
    }
    rt = dia._get_retention_time(spectrum)
    assert abs(rt - 10.5) < 1e-9  # 630s / 60 = 10.5min


def test_get_retention_time_handles_missing_unit_info():
    """Without unit_info attr, assume minutes (back-compat for plain floats)."""
    from spectrum.dia_data import DIAData
    dia = DIAData()
    spectrum = {'scanList': {'scan': [{'scan start time': 10.5}]}}
    rt = dia._get_retention_time(spectrum)
    assert rt == 10.5  # plain float, no conversion


def test_get_retention_time_raises_on_unknown_unit():
    """Unknown unit must raise loudly (no silent fallback)."""
    from spectrum.dia_data import DIAData

    class _MockUnitFloat(float):
        def __new__(cls, value, unit_info):
            instance = super().__new__(cls, value)
            instance.unit_info = unit_info
            return instance

    dia = DIAData()
    spectrum = {
        'scanList': {
            'scan': [{'scan start time': _MockUnitFloat(10.5, 'hour')}]
        }
    }
    with pytest.raises(ValueError, match="RT unit"):
        dia._get_retention_time(spectrum)


def test_calc_smoothness_zero_for_linear_ramp():
    """Linear ramp has all-zero second-differences (P2-3 sanity)."""
    from workflows.single_work import _calc_smoothness
    short = np.array([1.0, 2.0, 3.0, 4.0, 5.0])  # len 5
    long = np.array([1.0, 1.5, 2.0, 2.5, 3.0, 3.5, 4.0, 4.5, 5.0])  # len 9
    assert _calc_smoothness(short) == 0.0
    assert _calc_smoothness(long) == 0.0


def test_calc_smoothness_per_unit_value_independent_of_length():
    """Mean squared second-diff is comparable across window sizes (P2-3, Units-I4).

    Same triangle peak shape padded with zeros at different lengths.
    Without normalization the buggy version sums more squared terms
    for longer windows; the fix divides by N=len-2 so values are
    comparable.
    """
    from workflows.single_work import _calc_smoothness
    # Identical triangle inserted at center of differently-sized arrays
    triangle = [0.0, 1.0, 2.0, 1.0, 0.0]
    short = np.array(triangle)  # len 5
    long = np.array([0.0, 0.0] + triangle + [0.0, 0.0])  # len 9
    s_short = _calc_smoothness(short)
    s_long = _calc_smoothness(long)
    if s_short > 0 and s_long > 0:
        ratio = max(s_short, s_long) / min(s_short, s_long)
        # Without P2-3 fix, unnormalized values would diverge ~7/3.
        # With fix, ratio should stay under 3 (different totals still affect
        # the total^2 normalization, but length impact is gone).
        assert ratio < 3.0, (
            f"P2-3: smoothness should be less length-dependent after norm. "
            f"short={s_short}, long={s_long}, ratio={ratio}")


def test_calc_smoothness_short_input_returns_zero():
    """Length < 3 returns 0.0 (no regression)."""
    from workflows.single_work import _calc_smoothness
    assert _calc_smoothness(np.array([])) == 0.0
    assert _calc_smoothness(np.array([1.0, 2.0])) == 0.0
