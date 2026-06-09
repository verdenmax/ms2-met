from constant.keys import ConfigKeys


def test_speclib_config_keys_exist():
    assert ConfigKeys.SPECLIB == "speclib"
    assert ConfigKeys.SPECLIB_DIR == "speclib_dir"
    assert ConfigKeys.SPECLIB_FASTA == "speclib_fasta"
    assert ConfigKeys.SPECLIB_MOD == "speclib_mod"
    assert ConfigKeys.PRED_TOP_K == "pred_top_k"


from workflows.pair_flow import PairFlow
from workflows.pred_store import build_pred_store, normalize_key, frag_key
from spectrum.speclib import SpecLib
from spectrum.psm_info import PSMInfo
import numpy as np


def _psm(seq, charge, raw):
    return PSMInfo(sequence=seq, charge=charge, modify=[], rt=np.float32(10.0),
                   precursor_mz=np.float32(500.0), raw_title=raw,
                   protein_names="X", label_type="positive")


def test_build_raw_tasks_attaches_pred_frags(lib_files):
    lib = SpecLib.open_dir(str(lib_files), fasta_path=str(lib_files / "db.fasta"),
                           mod_path=str(lib_files / "modification.ini"))
    seq = "PEPTIDEKACDM"
    store = build_pred_store(lib, {normalize_key(seq, [], 1)})
    psm = _psm(seq, 1, "r1")
    groups = {psm.get_key(): [psm]}
    tasks, n_skipped = PairFlow._build_raw_tasks(
        groups, {"r1": "/tmp/shared.npz"}, 0, pred_store=store)
    assert len(tasks) == 1
    psm_dict = tasks[0][0]
    assert "pred_frags" in psm_dict
    assert abs(psm_dict["pred_frags"][frag_key("b", 0, 1)] - 1.0) < 1e-6


def test_build_raw_tasks_pred_frags_none_when_no_store():
    psm = _psm("PEPTIDEKACDM", 1, "r1")
    groups = {psm.get_key(): [psm]}
    tasks, _ = PairFlow._build_raw_tasks(groups, {"r1": "/tmp/s.npz"}, 0, pred_store=None)
    assert "pred_frags" not in tasks[0][0]


def test_build_raw_tasks_pred_frags_none_on_miss(lib_files):
    lib = SpecLib.open_dir(str(lib_files), fasta_path=str(lib_files / "db.fasta"),
                           mod_path=str(lib_files / "modification.ini"))
    store = build_pred_store(lib, set())
    psm = _psm("NOTINLIBK", 2, "r1")
    groups = {psm.get_key(): [psm]}
    tasks, _ = PairFlow._build_raw_tasks(groups, {"r1": "/tmp/s.npz"}, 0, pred_store=store)
    assert tasks[0][0]["pred_frags"] is None


import configparser
from workflows.single_work import single_pair_work
from workflows.pred_store import frag_key as _fk

_EMPTY_XIC = np.zeros(0, dtype=[("rt", "f8"), ("intensity", "f8"), ("ppm_error", "f8")])


class _FakeDia:
    def xic_peaks_extreact(self, rt, win, mz, ppm):
        return _EMPTY_XIC
    def xic_ms2_peaks_extract(self, rt, win, precursor_mz, ions_mass, mass_tol_ppm):
        return _EMPTY_XIC, 0.0
    def get_window_info(self, mz):
        return {"lower": 400.0, "upper": 600.0, "width": 200.0, "centering": 0.5}
    def check_in_raw(self, mz):
        return True
    def check_in_same_ms2(self, a, b):
        return False


def _cfg2():
    c = configparser.ConfigParser()
    c["general"] = {"mass_tol_ppm": "10", "xic_cycle_window": "3"}
    c["speclib"] = {"pred_top_k": "6"}
    return c


def _psm2():
    return PSMInfo(sequence="PEPTIDEKACDM", charge=1, modify=[],
                   rt=np.float32(10.0), precursor_mz=np.float32(500.0),
                   raw_title="r1", protein_names="X", label_type="positive")


def test_single_pair_work_emits_speclib_columns_when_enabled():
    feats = single_pair_work(_psm2(), _FakeDia(), _cfg2(),
                             pred_frags={_fk("b", 0, 1): 1.0}, speclib_enabled=True)
    for col in ("has_lib_pred", "spec_pattern_SA_b", "spec_pattern_SA_y",
                "spec_pattern_SA", "spec_pattern_LH_consistency",
                "n_fragments_in_F", "psm_is_split_window", "heavy_out_of_range"):
        assert col in feats
    assert feats["has_lib_pred"] == 1
    assert feats["psm_is_split_window"] == 1


def test_single_pair_work_unchanged_when_speclib_disabled():
    feats = single_pair_work(_psm2(), _FakeDia(), _cfg2(),
                             pred_frags=None, speclib_enabled=False)
    assert "has_lib_pred" not in feats
    assert "spec_pattern_SA" not in feats


def test_pred_presence_floor_key_exists():
    from constant.keys import ConfigKeys
    assert ConfigKeys.PRED_PRESENCE_FLOOR == "pred_presence_floor"
