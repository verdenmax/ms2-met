import json
import configparser
from types import SimpleNamespace

import numpy as np
import pytest

from spectrum.psm_info import PSMInfo
from workflows.modified_psm_policy import apply_modified_psm_policy


def _psm(sequence, label_type, modify):
    return PSMInfo(
        sequence=sequence,
        charge=2,
        modify=modify,
        rt=np.float32(10),
        precursor_mz=np.float32(500),
        raw_title="run1",
        protein_names="P1",
        label_type=label_type,
    )


def test_c13_drop_with_audit_filters_before_workers_and_pins_counts(tmp_path):
    psms = [
        _psm("PEPTIDEK", "positive", []),
        _psm("PEPTMDEK", "positive", [(4, 35)]),
        _psm("ACDMK", "negative", [(3, 35), (0, 1)]),
    ]
    result_file = tmp_path / "features.csv"

    kept, audit = apply_modified_psm_policy(
        psms, "c13", "drop_with_audit", result_file=str(result_file))

    assert [psm._sequence for psm in kept] == ["PEPTIDEK"]
    assert audit["n_input_psms"] == 3
    assert audit["n_dropped_psms"] == 2
    assert audit["dropped_counts_by_label_type"] == {
        "negative": 1, "positive": 1,
    }
    assert audit["dropped_psm_counts_by_unimod_id"] == {"1": 1, "35": 2}
    on_disk = json.loads(
        (tmp_path / "features.csv.modified_psm_audit.json").read_text())
    assert len(on_disk["dropped_psms"]) == 2
    assert on_disk["dropped_psms"][0]["modifications"] == [
        {"position": 4, "unimod_id": 35},
    ]


def test_n15_default_rejects_modified_psms_without_partial_extraction(tmp_path):
    with pytest.raises(ValueError, match="drop_with_audit"):
        apply_modified_psm_policy(
            [_psm("PEPTMDEK", "positive", [(4, 35)])],
            "n15", None, result_file=str(tmp_path / "features.csv"))


def test_silac_retains_modified_psms_and_still_writes_audit(tmp_path):
    psm = _psm("PEPTMDEK", "positive", [(4, 35)])
    kept, audit = apply_modified_psm_policy(
        [psm], "silac", "drop_with_audit",
        result_file=str(tmp_path / "features.csv"))
    assert kept == [psm]
    assert audit["action"] == "retained_supported"
    assert audit["n_modified_psms"] == 1
    assert audit["n_dropped_psms"] == 0


def test_invalid_modified_psm_policy_is_rejected(tmp_path):
    with pytest.raises(ValueError, match="非法 modified_psm_policy"):
        apply_modified_psm_policy(
            [_psm("PEPTIDEK", "positive", [])],
            "c13", "guess", result_file=str(tmp_path / "features.csv"))


def test_pair_flow_applies_manifest_and_policy_before_distribute(
        tmp_path, monkeypatch):
    from tools.extract_common import write_psms_to_json
    from workflows.pair_flow import PairFlow

    psms = [
        _psm("PEPTIDEK", "positive", []),
        _psm("PEPTMDEK", "negative", [(4, 35)]),
    ]
    dataset = tmp_path / "psms.json"
    result = tmp_path / "features.csv"
    write_psms_to_json(psms, str(dataset), labeling="c13")
    config = configparser.ConfigParser()
    config.read_dict({
        "input": {
            "search_engine_type": "0",
            "light_result_file": str(dataset),
        },
        "general": {
            "labeling": "c13",
            "modified_psm_policy": "drop_with_audit",
            "result_file": str(result),
        },
    })
    flow = PairFlow("preflight", config, str(tmp_path / "work"))

    def fake_load():
        flow._light_result = SimpleNamespace(
            psm_info=list(psms), peptide_len=len(psms))

    observed = {}

    def fake_distribute():
        observed["sequences"] = [
            psm._sequence for psm in flow._light_result.psm_info]

    monkeypatch.setattr(flow, "load", fake_load)
    monkeypatch.setattr(flow, "distribute", fake_distribute)
    flow.run()

    assert observed["sequences"] == ["PEPTIDEK"]
    assert flow._light_result.peptide_len == 1
    assert (tmp_path / "features.csv.modified_psm_audit.json").exists()
