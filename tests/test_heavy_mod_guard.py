"""Guard: CHEAVY/NHEAVY full-metabolic heavy mass is unimplemented for
*modified* peptides (modification C/N atoms are not 13C/15N-shifted), so the
path must fail loudly instead of returning a silently-wrong mass.

SILAC is unaffected (it only labels K/R, not PTMs) and unmodified CHEAVY/NHEAVY
stays correct.
"""
import numpy as np
import pytest

from spectrum.psm_info import PSMInfo, HeavyType


def _make_psm(seq, modify):
    from pyteomics import mass
    mz = mass.fast_mass(seq, ion_type="M", charge=2)
    return PSMInfo(
        sequence=seq, charge=2, modify=modify,
        rt=np.float32(50.0), precursor_mz=np.float32(mz),
        raw_title="r1", protein_names="X_HUMAN",
    )


def test_cheavy_modified_precursor_raises():
    psm = _make_psm("PEPTMDEK", modify=[(4, 35)])  # Oxidation on M
    with pytest.raises(NotImplementedError):
        psm.get_C_N_HEAVY_precursor_mz(HeavyType.CHEAVY)


def test_nheavy_modified_precursor_raises():
    psm = _make_psm("PEPTMDEK", modify=[(4, 35)])
    with pytest.raises(NotImplementedError):
        psm.get_C_N_HEAVY_precursor_mz(HeavyType.NHEAVY)


def test_cheavy_modified_fragment_ions_raises():
    psm = _make_psm("PEPTMDEK", modify=[(4, 35)])
    with pytest.raises(NotImplementedError):
        psm.get_fragment_ions(HeavyType.CHEAVY)


def test_cheavy_unmodified_precursor_still_works():
    psm = _make_psm("PEPTIDEK", modify=[])
    val = psm.get_C_N_HEAVY_precursor_mz(HeavyType.CHEAVY)
    assert val > 0


def test_silac_modified_fragment_ions_ok():
    """SILAC labels only K/R, so a PTM does not need heavy-shifting."""
    psm = _make_psm("PEPTMDEK", modify=[(4, 35)])
    b_ions, y_ions = psm.get_fragment_ions(HeavyType.SILAC)
    assert len(b_ions) > 0 and len(y_ions) > 0
