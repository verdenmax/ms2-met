from spectrum.psm_info import has_label_site, HeavyType


def test_silac_requires_kr():
    assert has_label_site("PEPTIDEK", HeavyType.SILAC) is True   # ends in K
    assert has_label_site("PEPTIDER", HeavyType.SILAC) is True   # ends in R
    assert has_label_site("SAMPLERK", HeavyType.SILAC) is True   # internal R + K
    assert has_label_site("ACDEFGHILMNPQSTVWY", HeavyType.SILAC) is False  # no K/R
    assert has_label_site("LQEFLQHVS", HeavyType.SILAC) is False  # real pilot trap


def test_silac_is_default_heavy_type():
    assert has_label_site("PEPTIDEK") is True
    assert has_label_site("ACDEF") is False


def test_cheavy_nheavy_always_have_label_site():
    # whole-atom metabolic labeling: every peptide has C and N -> always labeled
    for ht in (HeavyType.CHEAVY, HeavyType.NHEAVY):
        assert has_label_site("ACDEF", ht) is True        # no K/R but has C/N
        assert has_label_site("PEPTIDEK", ht) is True


def test_empty_sequence_has_no_label_site():
    assert has_label_site("", HeavyType.SILAC) is False
    assert has_label_site("", HeavyType.CHEAVY) is False


def test_lowercase_is_normalized():
    assert has_label_site("peptidek", HeavyType.SILAC) is True
    assert has_label_site("acdef", HeavyType.SILAC) is False


def test_single_residue_silac():
    assert has_label_site("K", HeavyType.SILAC) is True
    assert has_label_site("R", HeavyType.SILAC) is True
    assert has_label_site("A", HeavyType.SILAC) is False
