"""End-to-end test: multi_batch_work and single_pair_work emit q1a_* features."""
import configparser

import numpy as np
import pytest


def _minimal_config():
    cfg = configparser.ConfigParser()
    cfg["general"] = {
        "mass_tol_ppm": "10",
        "xic_cycle_window": "3",
    }
    return cfg


XIC_DTYPE = [("rt", "f8"), ("ppm_error", "f8"),
             ("intensity", "f8"), ("mz", "f8")]


class StubDIA:
    """Stub DIAData with no signal — exercises the q1a path returning
    all-NaN/zero features (no fragment passes light_present)."""

    def xic_peaks_extreact(self, *args, **kwargs):
        return np.array([], dtype=XIC_DTYPE)

    def xic_ms2_peaks_extract(self, *args, **kwargs):
        return np.array([], dtype=XIC_DTYPE), 0.0

    def get_window_info(self, mz):
        return {"width": 2.0, "centering": 0.5,
                "lower": 499.0, "upper": 501.0}

    def check_in_same_ms2(self, mz1, mz2):
        return True

    def check_in_raw(self, mz):
        return True


EXPECTED_Q1A_KEYS = {
    "q1a_recall", "q1a_recall_shifted", "q1a_recall_unshifted_separable",
    "q1a_y_recall", "q1a_b_recall",
    "q1a_TP_count", "q1a_FN_count",
    "q1a_TP_shifted", "q1a_TP_unshifted_separable",
    "q1a_total_count", "q1a_valid",
}


def test_multi_batch_work_emits_q1a_features():
    """multi_batch_work must add 11 q1a_* keys to its features dict."""
    from spectrum.psm_info import PSMInfo
    from workflows import single_work

    psm = PSMInfo(
        sequence="PEPTIDEK", charge=2, modify=[],
        rt=np.float32(10.0), precursor_mz=np.float32(500.0),
        raw_title="r1", protein_names="X_HUMAN",
    )
    dia = StubDIA()
    features = single_work.multi_batch_work(
        psm1=psm, dia_data1=dia,
        psm2=psm, dia_data2=dia,
        config=_minimal_config(),
    )

    missing = EXPECTED_Q1A_KEYS - set(features.keys())
    assert not missing, f"missing q1a keys in multi_batch_work: {missing}"
    # With empty XICs, no fragment passed light_present → counts=0, valid=0
    assert features["q1a_total_count"] == 0
    assert features["q1a_valid"] == 0
    assert np.isnan(features["q1a_recall"])


def test_single_pair_work_emits_q1a_features():
    """single_pair_work must also add q1a_* keys."""
    from spectrum.psm_info import PSMInfo
    from workflows import single_work

    psm = PSMInfo(
        sequence="PEPTIDEK", charge=2, modify=[],
        rt=np.float32(10.0), precursor_mz=np.float32(500.0),
        raw_title="r1", protein_names="X_HUMAN",
    )
    dia = StubDIA()
    features = single_work.single_pair_work(
        psm=psm, dia_data=dia, config=_minimal_config(),
    )

    missing = EXPECTED_Q1A_KEYS - set(features.keys())
    assert not missing, f"missing q1a keys in single_pair_work: {missing}"
