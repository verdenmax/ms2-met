"""Behavioral contract for SILAC/C13/N15 labeling support."""

import configparser

import numpy as np
import pytest


@pytest.mark.parametrize(
    ("value", "expected_name"),
    [
        ("silac", "SILAC"),
        (" SILAC ", "SILAC"),
        ("c13", "CHEAVY"),
        ("13C", "CHEAVY"),
        ("cheavy", "CHEAVY"),
        ("n15", "NHEAVY"),
        ("15N", "NHEAVY"),
        ("nheavy", "NHEAVY"),
    ],
)
def test_parse_heavy_type_aliases(value, expected_name):
    from spectrum.labeling import parse_heavy_type

    assert parse_heavy_type(value).name == expected_name


def test_parse_heavy_type_accepts_enum_and_rejects_unknown():
    from spectrum.labeling import HeavyType, parse_heavy_type

    assert parse_heavy_type(HeavyType.CHEAVY) is HeavyType.CHEAVY
    with pytest.raises(ValueError, match="unknown labeling"):
        parse_heavy_type("dimethyl")


@pytest.mark.parametrize(
    ("heavy_type", "canonical"),
    [
        ("SILAC", "silac"),
        ("CHEAVY", "c13"),
        ("NHEAVY", "n15"),
    ],
)
def test_canonical_labeling_names(heavy_type, canonical):
    from spectrum.labeling import (
        HeavyType,
        canonical_labeling_name,
    )

    assert canonical_labeling_name(HeavyType[heavy_type]) == canonical


def test_labeling_rules_are_available_through_one_interface():
    from spectrum.labeling import (
        HeavyType,
        get_heavy_increase_mass,
        has_label_site,
        supports_modified_peptide,
    )

    sequence = "ACDEFGHIK"
    shifts = {
        heavy_type: get_heavy_increase_mass(sequence, heavy_type)
        for heavy_type in HeavyType
    }
    assert all(value > 0 for value in shifts.values())
    assert len(set(shifts.values())) == 3
    assert has_label_site("ACDEF", HeavyType.SILAC) is False
    assert has_label_site("ACDEF", HeavyType.CHEAVY) is True
    assert has_label_site("ACDEF", HeavyType.NHEAVY) is True
    assert supports_modified_peptide(HeavyType.SILAC) is True
    assert supports_modified_peptide(HeavyType.CHEAVY) is False
    assert supports_modified_peptide(HeavyType.NHEAVY) is False


def test_psm_info_keeps_legacy_labeling_imports():
    from spectrum.labeling import HeavyType as CanonicalHeavyType
    from spectrum.psm_info import (
        HeavyType,
        get_heavy_increase_mass,
        has_label_site,
    )

    assert HeavyType is CanonicalHeavyType
    assert get_heavy_increase_mass("PEPTIDEK", HeavyType.SILAC) > 0
    assert has_label_site("PEPTIDEK", HeavyType.SILAC)


def _config(labeling=None):
    values = {"mass_tol_ppm": "20", "xic_cycle_window": "3"}
    if labeling is not None:
        values["labeling"] = labeling
    cfg = configparser.ConfigParser()
    cfg.read_dict({"general": values})
    return cfg


def _empty_xic():
    return np.array([], dtype=[
        ("rt", "f8"), ("ppm_error", "f8"),
        ("intensity", "f8"), ("cycle_idx", "i4"),
    ])


class _RecordingPSM:
    def __init__(self):
        self._precursor_mz = 500.0
        self._rt = 10.0
        self._sequence = "ACDEFGHIK"
        self._charge = 2
        self._raw_title = "raw"
        self._protein_names = "target"
        self._label_type = "positive"
        self._modify = []
        self.requested_types = []

    def get_heavy_info(self, heavy_type):
        from spectrum.labeling import get_heavy_increase_mass

        self.requested_types.append(heavy_type)
        shift = get_heavy_increase_mass(self._sequence, heavy_type)
        return self._precursor_mz + shift / self._charge, [
            ("b", 1, 100.0, 100.0 + shift / 3),
            ("y", 1, 200.0, 200.0 + shift / 2),
        ]


class _EmptyRecordingDIA:
    def xic_peaks_extreact(self, *args, **kwargs):
        return _empty_xic()

    def xic_ms2_peaks_extract(self, *args, **kwargs):
        return _empty_xic(), 0.0

    def get_window_info(self, mz):
        return {
            "lower": mz - 1.0, "upper": mz + 1.0,
            "width": 2.0, "centering": 0.5,
        }

    def check_in_same_ms2(self, *args):
        return False

    def check_in_raw(self, mz):
        return True


@pytest.mark.parametrize(
    ("configured", "expected"),
    [("silac", "SILAC"), ("c13", "CHEAVY"), ("n15", "NHEAVY")],
)
@pytest.mark.parametrize("workflow_name", ["single_pair_work", "multi_batch_work"])
def test_workflows_use_configured_labeling(configured, expected, workflow_name):
    from spectrum.labeling import (
        HeavyType,
        canonical_labeling_name,
        get_heavy_increase_mass,
    )
    from workflows import single_work

    psm = _RecordingPSM()
    dia = _EmptyRecordingDIA()
    workflow = getattr(single_work, workflow_name)
    if workflow_name == "single_pair_work":
        features = workflow(psm, dia, _config(configured))
    else:
        features = workflow(psm, dia, psm, dia, _config(configured))

    heavy_type = HeavyType[expected]
    assert psm.requested_types == [heavy_type]
    expected_shift = get_heavy_increase_mass(psm._sequence, heavy_type)
    assert features["total_label_shift"] == pytest.approx(expected_shift)
    assert features["total_silac_shift"] == pytest.approx(expected_shift)
    assert features["labeling"] == canonical_labeling_name(heavy_type)
    if heavy_type is HeavyType.SILAC:
        assert features["isotope_model_valid"] == 1
        assert features["isotope_correlation"] == 0.0
    else:
        assert features["isotope_model_valid"] == 0
        assert np.isnan(features["isotope_correlation"])


def test_workflow_labeling_defaults_to_silac_and_rejects_unknown():
    from spectrum.labeling import HeavyType
    from workflows.single_work import resolve_workflow_heavy_type

    assert resolve_workflow_heavy_type(_config()) is HeavyType.SILAC
    with pytest.raises(ValueError, match="unknown labeling"):
        resolve_workflow_heavy_type(_config("itraq"))
