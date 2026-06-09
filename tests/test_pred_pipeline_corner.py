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

    def get_window_info(self, mz):
        return {"lower": 400.0, "upper": 600.0, "width": 200.0,
                "centering": 0.5}

    def check_in_raw(self, mz):
        return self.in_raw

    def check_in_same_ms2(self, a, b):
        return self.same_ms2


class _HeavyOutOfRangeDia(_FakeDia):
    in_raw = False


class _SameWindowDia(_FakeDia):
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
