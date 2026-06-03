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
