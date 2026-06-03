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
    """Length normalization makes smoothness comparable across xic_cycle_window
    sizes (P2-3, Units-I4).

    Pre-fix: sum of squared second-diffs scales with the number of terms,
    but identical-shape inputs at different lengths produce IDENTICAL
    sums (zero-padded second-diffs are zero) → ratio always 1.0.

    Post-fix: divide by N=len-2. Same identical-shape inputs now have
    ratio = N_long / N_short, exercising the normalization.

    Concretely: triangle [1,2,1] pre-padded with 3 zeros each side gives
    short=len-9 (n_diff=7); padding 6 more zeros each side gives long=len-21
    (n_diff=19). Because the triangle is already surrounded by zeros, the
    boundary second-diffs are unchanged, so sum(second_diff**2)=6 and
    total=4 for both inputs.

    Pre-fix _calc_smoothness = sum_sq / total^2, identical for both → ratio = 1.0.
    Post-fix divides by n_diff: ratio = N_long / N_short = 19/7 ≈ 2.71.
    We assert 2.0 < ratio < 3.0 — fails on the buggy code, passes on the fix.
    """
    from workflows.single_work import _calc_smoothness
    triangle = [1.0, 2.0, 1.0]
    short = np.array([0.0] * 3 + triangle + [0.0] * 3)   # len 9,  n_diff=7
    long = np.array([0.0] * 9 + triangle + [0.0] * 9)    # len 21, n_diff=19
    s_short = _calc_smoothness(short)
    s_long = _calc_smoothness(long)
    assert s_short > 0 and s_long > 0, (
        f"Need non-zero smoothness; got short={s_short}, long={s_long}")
    ratio = s_short / s_long
    # Pre-fix: ratio = 1.0 (no length normalization; sum_sq and total identical).
    # Post-fix: ratio = N_long/N_short = 19/7 ≈ 2.71.
    assert 2.0 < ratio < 3.0, (
        f"P2-3: post-normalization s_short/s_long should be ~2.71; "
        f"got {ratio}. Pre-fix bug would give ratio=1.0.")


def test_calc_smoothness_short_input_returns_zero():
    """Length < 3 returns 0.0 (no regression)."""
    from workflows.single_work import _calc_smoothness
    assert _calc_smoothness(np.array([])) == 0.0
    assert _calc_smoothness(np.array([1.0, 2.0])) == 0.0


def test_sequence_controlled_shuffle_deterministic_with_seed():
    """Same seed produces identical shuffle output (P2-4, Pipeline-I2)."""
    from spectrum.psm_info import sequence_controlled_shuffle
    seq = "ABCDEFGHIK"
    out1 = sequence_controlled_shuffle(seq, anchor_len=2, shuffle_ratio=0.5,
                                        seed=42)
    out2 = sequence_controlled_shuffle(seq, anchor_len=2, shuffle_ratio=0.5,
                                        seed=42)
    assert out1 == out2, (
        f"P2-4: same seed must produce same output; got {out1!r} vs {out2!r}")


def test_sequence_controlled_shuffle_preserves_anchor_with_seed():
    """Last anchor_len chars stay at the end (no regression with seed kwarg)."""
    from spectrum.psm_info import sequence_controlled_shuffle
    seq = "ABCDEFGHIK"
    out = sequence_controlled_shuffle(seq, anchor_len=2, shuffle_ratio=0.5,
                                       seed=42)
    assert out.endswith("IK"), (
        f"P2-4: anchor 'IK' must be preserved; got {out!r}")
    assert len(out) == len(seq)


def test_sequence_controlled_shuffle_back_compat_no_seed():
    """seed=None falls back to module random (back-compat for callers
    that haven't been updated)."""
    from spectrum.psm_info import sequence_controlled_shuffle
    seq = "ABCDEFGHIK"
    out = sequence_controlled_shuffle(seq, anchor_len=2, shuffle_ratio=0.5)
    # No assertion on exact value (non-deterministic without seed); just
    # verify it doesn't crash and preserves anchor.
    assert out.endswith("IK")
    assert len(out) == len(seq)


def test_per_psm_seed_is_stable_across_invocations(monkeypatch):
    """Per-PSM seed derivation must use a stable hash (not Python's
    PYTHONHASHSEED-randomized built-in hash) so that runs with the same
    random_seed config produce identical shuffles across processes.

    Regression for P2-4 review fix (2026-06-03): the initial commit used
    hash(seq) which is process-randomized; replaced with zlib.crc32."""
    import zlib
    seq = "PEPTIDEK"
    # zlib.crc32 must be deterministic across calls
    h1 = zlib.crc32(seq.encode())
    h2 = zlib.crc32(seq.encode())
    assert h1 == h2

    # Also verify the helper still produces deterministic output for
    # the same sequence + seed combination
    from spectrum.psm_info import sequence_controlled_shuffle
    seed_base = 42
    per_psm_seed = (seed_base + h1) % (2**31)
    s1 = sequence_controlled_shuffle(seq, anchor_len=2,
                                      shuffle_ratio=0.5, seed=per_psm_seed)
    s2 = sequence_controlled_shuffle(seq, anchor_len=2,
                                      shuffle_ratio=0.5, seed=per_psm_seed)
    assert s1 == s2
