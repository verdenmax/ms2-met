"""Corner/core-path hardening for single_pair_work speclib I1 emission.

Uses the direct-call _FakeDia pattern from
tests/test_pred_pipeline_integration.py. Each test asserts the CORRECT
behavior derived from workflows/single_work.py (speclib block, lines
~829-835) + the design spec v1.2 §4.3. A failing test is a SUSPECTED BUG
unless confirmed acceptable.
"""
import configparser
import math

import numpy as np

from workflows.single_work import single_pair_work
from workflows.pred_integrate import I1_KEYS
from workflows.pred_store import frag_key as _fk
from spectrum.psm_info import PSMInfo

_EMPTY_XIC = np.zeros(0, dtype=[("rt", "f8"), ("intensity", "f8"),
                                ("ppm_error", "f8")])

_SPECLIB_COLS = ("has_lib_pred", "psm_is_split_window", "heavy_out_of_range",
                 *I1_KEYS)


class _FakeDia:
    """Base fake: empty XICs, heavy in raw, not split (same MS2 -> False)."""
    in_raw = True
    same_ms2 = False

    def xic_peaks_extreact(self, rt, win, mz, ppm):
        return _EMPTY_XIC

    def xic_ms2_peaks_extract(self, rt, win, precursor_mz, ions_mass,
                              mass_tol_ppm):
        return _EMPTY_XIC, 0.0

    def get_window_info(self, mz, rt=None):
        return {"lower": 400.0, "upper": 600.0, "width": 200.0,
                "centering": 0.5}

    def check_in_raw(self, mz):
        return self.in_raw

    def check_in_same_ms2(self, a, b, rt=None):
        return self.same_ms2


class _HeavyOutOfRangeDia(_FakeDia):
    in_raw = False


class _SameWindowDia(_FakeDia):
    same_ms2 = True


def _xic1(intensity):
    a = np.zeros(1, dtype=[("rt", "f8"), ("ppm_error", "f8"),
                           ("intensity", "f8"), ("cycle_idx", "i4")])
    a["rt"][0] = 10.0
    a["intensity"][0] = intensity
    a["cycle_idx"][0] = 0
    return a


class _SignalSplitDia(_FakeDia):
    """Split-window (same_ms2=False) but returns a REAL non-zero XIC for
    every fragment query, so a separable fragment actually carries signal."""
    same_ms2 = False

    def xic_ms2_peaks_extract(self, rt, win, precursor_mz, ions_mass,
                              mass_tol_ppm):
        return _xic1(1000.0), 1000.0


class _SignalSameWindowDia(_SignalSplitDia):
    """Same as _SignalSplitDia but co-isolated (same_ms2=True): unshifted
    fragments must be excluded from F despite the real heavy signal."""
    same_ms2 = True


def _cfg2():
    c = configparser.ConfigParser()
    c["general"] = {"mass_tol_ppm": "10", "xic_cycle_window": "3"}
    c["speclib"] = {"pred_top_k": "6"}
    return c


def _psm2():
    return PSMInfo(sequence="PEPTIDEKACDM", charge=1, modify=[],
                   rt=np.float32(10.0), precursor_mz=np.float32(500.0),
                   raw_title="r1", protein_names="X", label_type="positive")


def _assert_cols_present(feats):
    for col in _SPECLIB_COLS:
        assert col in feats, f"missing column {col}"


# --- enabled but no coverage --------------------------------------------------

def test_enabled_no_coverage_columns_present_all_nan():
    feats = single_pair_work(_psm2(), _FakeDia(), _cfg2(),
                             pred_frags=None, speclib_enabled=True)
    _assert_cols_present(feats)
    assert feats["has_lib_pred"] == 0
    assert math.isnan(feats["spec_pattern_SA_b"])
    assert math.isnan(feats["spec_pattern_SA_y"])
    assert math.isnan(feats["spec_pattern_SA"])
    assert math.isnan(feats["spec_pattern_LH_consistency"])
    assert feats["n_fragments_in_F"] == 0


# --- heavy out of range -------------------------------------------------------

def test_heavy_out_of_range_flag_and_empty_F():
    """check_in_raw False -> heavy_out_of_range=1 and every fragment is
    skipped (heavy_in_raw gate), so n_fragments_in_F=0."""
    feats = single_pair_work(_psm2(), _HeavyOutOfRangeDia(), _cfg2(),
                             pred_frags={_fk("b", 0, 1): 1.0},
                             speclib_enabled=True)
    _assert_cols_present(feats)
    assert feats["heavy_out_of_range"] == 1
    assert feats["n_fragments_in_F"] == 0
    assert math.isnan(feats["spec_pattern_SA"])
    # coverage existed even though no fragment survived
    assert feats["has_lib_pred"] == 1


# --- split-window flag --------------------------------------------------------

def test_split_window_flag_when_not_same_ms2():
    feats = single_pair_work(_psm2(), _FakeDia(), _cfg2(),
                             pred_frags={_fk("b", 0, 1): 1.0},
                             speclib_enabled=True)
    assert feats["psm_is_split_window"] == 1
    assert feats["heavy_out_of_range"] == 0


def test_not_split_window_when_same_ms2():
    feats = single_pair_work(_psm2(), _SameWindowDia(), _cfg2(),
                             pred_frags={_fk("b", 0, 1): 1.0},
                             speclib_enabled=True)
    assert feats["psm_is_split_window"] == 0


# --- separability contamination guard (positive, non-empty XICs) -------------
# b1 ("P") is UNSHIFTED (heavy_mass == light_mass; verified: SILAC shifts only
# K/R-containing fragments). The pair below isolates ONLY the b1 prediction and
# feeds a real 1000-count heavy signal for every fragment. The same input must
# behave oppositely depending on co-isolation, proving the §4.1.5 guard runs.

def test_same_window_excludes_unshifted_fragment_despite_heavy_signal():
    """Co-isolated (same_ms2=True): b1 is unshifted -> skipped before XIC
    extraction, so its real heavy signal must NOT leak into F/coverage."""
    feats = single_pair_work(_psm2(), _SignalSameWindowDia(), _cfg2(),
                             pred_frags={_fk("b", 0, 1): 1.0},
                             speclib_enabled=True)
    assert feats["psm_is_split_window"] == 0
    assert feats["n_fragments_in_F"] == 0
    assert math.isnan(feats["pred_coverage"])
    assert math.isnan(feats["pred_both_present_fraction"])
    assert math.isnan(feats["n_both_present"])


def test_split_window_includes_unshifted_fragment_with_heavy_signal():
    """Split-window (same_ms2=False): the SAME b1 prediction + heavy signal is
    now separable -> b1 enters F and its signal is counted (contrast partner of
    the same-window test above)."""
    feats = single_pair_work(_psm2(), _SignalSplitDia(), _cfg2(),
                             pred_frags={_fk("b", 0, 1): 1.0},
                             speclib_enabled=True)
    assert feats["psm_is_split_window"] == 1
    assert feats["n_fragments_in_F"] == 1
    assert feats["pred_coverage"] == 1.0
    assert feats["n_both_present"] == 1
    assert feats["pred_both_present_fraction"] == 1.0


# --- schema stability for LightGBM -------------------------------------------

def test_column_set_identical_coverage_vs_no_coverage():
    """Enabled-with-coverage and enabled-no-coverage PSMs must emit an
    identical column set (LightGBM schema stability)."""
    feats_cov = single_pair_work(_psm2(), _FakeDia(), _cfg2(),
                                 pred_frags={_fk("b", 0, 1): 1.0},
                                 speclib_enabled=True)
    feats_nocov = single_pair_work(_psm2(), _FakeDia(), _cfg2(),
                                   pred_frags=None, speclib_enabled=True)
    assert set(feats_cov) == set(feats_nocov)
    assert feats_cov["has_lib_pred"] == 1
    assert feats_nocov["has_lib_pred"] == 0
