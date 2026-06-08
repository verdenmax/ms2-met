import numpy as np
import pytest

from spectrum.speclib import SpecLib
from workflows.pred_store import (
    normalize_mods, normalize_key, frag_key, build_pred_store)


class _ModSite:
    """Mimics spectrum.speclib.pepdata.ModSite (pos, mod_id)."""
    def __init__(self, pos, mod_id):
        self.pos = pos
        self.mod_id = mod_id


def test_normalize_mods_handles_tuples_and_modsite_equally():
    from_tuples = normalize_mods([(9, 1), (3, 2)])
    from_objs = normalize_mods([_ModSite(9, 1), _ModSite(3, 2)])
    assert from_tuples == from_objs == ((3, 2), (9, 1))


def test_normalize_key_is_hashable_and_charge_int():
    key = normalize_key("PEPTIDEK", [(9, 1)], "2")
    assert key == ("PEPTIDEK", ((9, 1),), 2)
    assert hash(key)


def test_frag_key_normalizes_types():
    assert frag_key("b", 0, 1) == ("b", 0, 1)
    assert frag_key("y", np.int8(2), np.int8(3)) == ("y", 2, 3)


def _open_lib(lib_files):
    return SpecLib.open_dir(
        str(lib_files),
        fasta_path=str(lib_files / "db.fasta"),
        mod_path=str(lib_files / "modification.ini"))


def test_build_pred_store_hits_unmodified_variant(lib_files):
    lib = _open_lib(lib_files)
    want = normalize_key("PEPTIDEKACDM", [], 1)
    store = build_pred_store(lib, {want})
    rec = store.get(want)
    assert rec is not None
    assert rec["frags"][frag_key("b", 0, 1)] == pytest.approx(1.0)
    assert store.n_hit == 1 and store.n_miss == 0


def test_build_pred_store_hits_modified_variant_charge2(lib_files):
    lib = _open_lib(lib_files)
    want = normalize_key("PEPTIDEKACDM", [(9, 1)], 2)
    store = build_pred_store(lib, {want})
    rec = store.get(want)
    assert rec is not None
    assert rec["frags"][frag_key("y", 2, 2)] == pytest.approx(0.3)


def test_build_pred_store_counts_miss(lib_files):
    lib = _open_lib(lib_files)
    present = normalize_key("PEPTIDEKACDM", [], 1)
    absent = normalize_key("NOTINLIBK", [], 2)
    store = build_pred_store(lib, {present, absent})
    assert store.get(absent) is None
    assert store.n_hit == 1 and store.n_miss == 1
