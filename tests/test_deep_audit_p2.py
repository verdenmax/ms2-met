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
