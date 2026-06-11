import configparser

import numpy as np
import pytest

from spectrum.psm_info import PSMInfo, HeavyType
from tools.extract_common import _parse_labeling, filter_by_label_site


def _psm(seq, label):
    return PSMInfo(sequence=seq, charge=2, modify=[], rt=np.float32(10.0),
                   precursor_mz=np.float32(500.0), raw_title="r1",
                   protein_names="X", label_type=label)


def _cfg(labeling=None):
    c = configparser.ConfigParser()
    c["extract"] = {} if labeling is None else {"labeling": labeling}
    return c


def test_parse_labeling_default_silac():
    assert _parse_labeling(_cfg()) == HeavyType.SILAC


def test_parse_labeling_aliases_case_insensitive():
    assert _parse_labeling(_cfg("SILAC")) == HeavyType.SILAC
    assert _parse_labeling(_cfg("c13")) == HeavyType.CHEAVY
    assert _parse_labeling(_cfg("13C")) == HeavyType.CHEAVY
    assert _parse_labeling(_cfg("cheavy")) == HeavyType.CHEAVY
    assert _parse_labeling(_cfg("n15")) == HeavyType.NHEAVY
    assert _parse_labeling(_cfg("15N")) == HeavyType.NHEAVY
    assert _parse_labeling(_cfg("nheavy")) == HeavyType.NHEAVY


def test_parse_labeling_missing_section_defaults_silac():
    assert _parse_labeling(configparser.ConfigParser()) == HeavyType.SILAC


def test_parse_labeling_ignores_labeling_in_wrong_section():
    c = configparser.ConfigParser()
    c["other"] = {"labeling": "c13"}  # only [extract] labeling is honored
    assert _parse_labeling(c) == HeavyType.SILAC


def test_parse_labeling_invalid_raises():
    with pytest.raises(ValueError, match="labeling"):
        _parse_labeling(_cfg("itraq"))


def test_filter_silac_drops_no_kr_both_classes():
    psms = [_psm("PEPTIDEK", "positive"),   # has K -> keep
            _psm("ACDEF", "positive"),        # no K/R, target -> DROP
            _psm("SAMPLER", "negative"),      # has R -> keep
            _psm("ACDEF", "negative")]        # no K/R, trap -> DROP
    kept = filter_by_label_site(psms, HeavyType.SILAC)
    seqs = [(p._sequence, p._label_type) for p in kept]
    assert seqs == [("PEPTIDEK", "positive"), ("SAMPLER", "negative")]


def test_filter_cheavy_keeps_everything():
    psms = [_psm("PEPTIDEK", "positive"), _psm("ACDEF", "negative")]
    kept = filter_by_label_site(psms, HeavyType.CHEAVY)
    assert len(kept) == 2   # whole-atom labeling -> no-op
