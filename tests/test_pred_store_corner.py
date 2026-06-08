"""Corner-case / core-path hardening for workflows.pred_store.

Asserts behavior derived from reading the code and the design spec
(docs/specs/2026-06-08-speclib-predicted-intensity-features-design.md §4.1).
The synthetic `lib_files` fixture (tests/conftest.py) decodes to:
    variant0 mods=()      : chg1 -> ('b',0,1)=1.0   chg2 -> ('y',1,1)=0.5
    variant1 mods=((9,1),) : chg1 -> ('b',0,1)=0.8   chg2 -> ('y',2,2)=0.3
"""
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


# --------------------------------------------------------------------------
# normalize_mods
# --------------------------------------------------------------------------
def test_normalize_mods_empty_is_empty_tuple():
    assert normalize_mods([]) == ()


def test_normalize_mods_sorted_and_unsorted_match():
    assert normalize_mods([(9, 1), (3, 2)]) == normalize_mods([(3, 2), (9, 1)])
    assert normalize_mods([(9, 1), (3, 2)]) == ((3, 2), (9, 1))


def test_normalize_mods_coerces_numpy_ints_to_python_int():
    out = normalize_mods([(np.int64(9), np.int64(1))])
    assert out == ((9, 1),)
    (pos, mid), = out
    assert type(pos) is int and type(mid) is int


def test_normalize_mods_preserves_duplicate_positions():
    # Same position, different mod ids: both entries must survive, sorted.
    assert normalize_mods([(3, 2), (3, 1)]) == ((3, 1), (3, 2))


def test_normalize_mods_modsite_objects_equal_tuples():
    assert normalize_mods([_ModSite(9, 1), _ModSite(3, 2)]) == ((3, 2), (9, 1))


# --------------------------------------------------------------------------
# normalize_key
# --------------------------------------------------------------------------
def test_normalize_key_charge_float_and_str_both_int():
    k_float = normalize_key("PEPTIDEK", [(9, 1)], 2.0)
    k_str = normalize_key("PEPTIDEK", [(9, 1)], "2")
    k_int = normalize_key("PEPTIDEK", [(9, 1)], 2)
    assert k_float == k_str == k_int
    assert k_int[2] == 2 and type(k_int[2]) is int


def test_normalize_key_equal_keys_hash_equal():
    a = normalize_key("PEPTIDEK", [(3, 2), (9, 1)], 2)
    b = normalize_key("PEPTIDEK", [(9, 1), (3, 2)], "2")
    assert a == b
    assert hash(a) == hash(b)
    assert len({a, b}) == 1


# --------------------------------------------------------------------------
# build_pred_store
# --------------------------------------------------------------------------
def _open_lib(lib_files):
    return SpecLib.open_dir(
        str(lib_files),
        fasta_path=str(lib_files / "db.fasta"),
        mod_path=str(lib_files / "modification.ini"))


def test_build_pred_store_empty_wanted_keys(lib_files):
    store = build_pred_store(_open_lib(lib_files), set())
    assert store.n_hit == 0
    assert store.n_miss == 0
    assert store.wanted == set()


def test_build_pred_store_same_pep_both_charges_two_hits(lib_files):
    lib = _open_lib(lib_files)
    chg1 = normalize_key("PEPTIDEKACDM", [], 1)
    chg2 = normalize_key("PEPTIDEKACDM", [], 2)
    store = build_pred_store(lib, {chg1, chg2})
    assert store.n_hit == 2
    assert store.n_miss == 0
    assert store.get(chg1)["frags"][frag_key("b", 0, 1)] == pytest.approx(1.0)
    assert store.get(chg2)["frags"][frag_key("y", 1, 1)] == pytest.approx(0.5)


def test_build_pred_store_same_seq_different_mods_is_miss(lib_files):
    lib = _open_lib(lib_files)
    # Sequence exists, but this modification combination does not.
    bogus = normalize_key("PEPTIDEKACDM", [(5, 2)], 1)
    store = build_pred_store(lib, {bogus})
    assert store.get(bogus) is None
    assert store.n_hit == 0
    assert store.n_miss == 1


def test_build_pred_store_get_unstored_key_is_none(lib_files):
    lib = _open_lib(lib_files)
    want = normalize_key("PEPTIDEKACDM", [], 1)
    store = build_pred_store(lib, {want})
    assert store.get(normalize_key("PEPTIDEKACDM", [], 2)) is None
    assert store.get(("NOPE", (), 3)) is None


def test_build_pred_store_variant1_charge1_decodes_b0(lib_files):
    # The 0.8-intensity record belongs to variant1 at charge 1 and decodes
    # to ('b', 0, 1) -- NOT ('y',1,1)/charge2 (that record is the 0.5 of
    # variant0/charge2). Asserting the actual, correct decoding.
    lib = _open_lib(lib_files)
    want = normalize_key("PEPTIDEKACDM", [(9, 1)], 1)
    store = build_pred_store(lib, {want})
    rec = store.get(want)
    assert rec is not None
    assert rec["frags"][frag_key("b", 0, 1)] == pytest.approx(0.8)


def test_build_pred_store_variant0_charge2_decodes_y1(lib_files):
    lib = _open_lib(lib_files)
    want = normalize_key("PEPTIDEKACDM", [], 2)
    store = build_pred_store(lib, {want})
    rec = store.get(want)
    assert rec is not None
    assert rec["frags"][frag_key("y", 1, 1)] == pytest.approx(0.5)
