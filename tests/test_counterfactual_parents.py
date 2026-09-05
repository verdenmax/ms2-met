import json

import numpy as np
import pandas as pd
import pytest

from spectrum.labeling import HeavyType
from spectrum.psm_dataset_manifest import validate_manifest, write_manifest
from spectrum.psm_identity import (
    PEPTIDE_GROUP_ID_SCHEMA,
    peptide_group_id,
)
from spectrum.psm_info import PSMInfo
from tools.counterfactual_parents import (
    PARENT_TRUTH_RULE,
    PREPARATION_SCHEMA,
    ParentPreparationConfig,
    ParentPreparationJob,
    prepare_counterfactual_parents,
    run_job,
)


def _psm(sequence="PEPTIDEK", *, charge=2, raw="raw_train",
         label_type="positive", modify=None):
    return PSMInfo(
        sequence=sequence,
        charge=charge,
        modify=modify or [],
        rt=np.float32(12.5),
        precursor_mz=np.float32(500.0 + charge),
        raw_title=raw,
        protein_names="TARGET",
        label_type=label_type,
    )


def _cfg(**overrides):
    values = dict(
        dataset_split="label_train",
    )
    values.update(overrides)
    return ParentPreparationConfig(**values)


def _splits():
    return pd.DataFrame([
        {"raw_title": "raw_train", "dataset_split": "label_train"},
        {"raw_title": "raw_dev", "dataset_split": "label_dev"},
    ])


def test_prepares_filtered_positives_in_split_and_groups_across_charge_and_li():
    train_i = _psm("PEPTIDEK", charge=2)
    train_l = _psm("PEPTLDEK", charge=3)
    modified = _psm("MELTPEPK", charge=2, modify=[(2, 35)])
    outside = _psm("PEPTIDEK", charge=2, raw="raw_dev")
    negative = _psm("NEGATIVEK", label_type="negative")

    result = prepare_counterfactual_parents(
        [outside, negative, train_l, modified, train_i],
        _splits(), _cfg())

    assert result.audit["schema"] == PREPARATION_SCHEMA
    assert result.audit["peptide_group_id_schema"] == PEPTIDE_GROUP_ID_SCHEMA
    assert result.audit["counts"]["prepared_parents"] == 2
    assert result.audit["counts"]["peptide_groups"] == 1
    assert result.audit["failures"] == {
        "modified": 1,
        "not_positive": 1,
        "outside_dataset_split": 1,
    }
    assert [psm._charge for psm in result.psms] == [2, 3]
    assert {psm._peptide_group_id for psm in result.psms} == {
        peptide_group_id("PEPTIDEK")}
    assert all(psm._heavy_confirmed is True for psm in result.psms)
    assert all(psm._dataset_split == "label_train" for psm in result.psms)
    assert set(result.manifest["heavy_confirmed"]) == {1}
    assert set(result.manifest["parent_truth_rule"]) == {PARENT_TRUTH_RULE}
    assert result.audit["parent_truth"] == {
        "source": "input_psms.label_type",
        "accepted_value": "positive",
        "rule": PARENT_TRUTH_RULE,
        "upstream_filtered_json_is_authoritative": True,
    }


def test_requires_complete_raw_mapping_and_unique_positive_identity():
    parent = _psm()
    with pytest.raises(ValueError, match="missing from raw split"):
        prepare_counterfactual_parents(
            [parent],
            pd.DataFrame([{
                "raw_title": "other_raw", "dataset_split": "label_train"}]),
            _cfg())

    with pytest.raises(ValueError, match="duplicate eligible positive parent"):
        prepare_counterfactual_parents(
            [parent, parent], _splits(), _cfg())


def test_rejects_input_without_an_eligible_positive():
    with pytest.raises(ValueError, match="no eligible positive"):
        prepare_counterfactual_parents(
            [_psm(label_type="negative")], _splits(), _cfg())


def test_cli_adapter_writes_prepared_json_manifest_and_audit(tmp_path):
    parent = _psm()
    input_psms = tmp_path / "all.json"
    input_psms.write_text(
        json.dumps([parent.to_dict()]), encoding="utf-8")
    write_manifest(str(input_psms), [parent], HeavyType.SILAC)
    split_path = tmp_path / "splits.csv"
    _splits().to_csv(split_path, index=False)
    output_psms = tmp_path / "prepared.json"
    output_manifest = tmp_path / "prepared.tsv"
    output_audit = tmp_path / "prepared.audit.json"
    config_path = tmp_path / "prepare.ini"
    config_path.write_text(
        "[counterfactual_parents]\n"
        "dataset_split=label_train\n",
        encoding="utf-8")
    job = ParentPreparationJob(
        input_psms=str(input_psms),
        raw_split_table=str(split_path),
        output_psms=str(output_psms),
        output_manifest=str(output_manifest),
        output_audit=str(output_audit),
        prepare=_cfg(),
    )

    audit = run_job(job, source_config_path=str(config_path))

    payload = json.loads(output_psms.read_text())
    manifest = pd.read_csv(output_manifest, sep="\t")
    on_disk_audit = json.loads(output_audit.read_text())
    assert payload[0]["heavy_confirmed"] is True
    assert payload[0]["dataset_split"] == "label_train"
    assert payload[0]["peptide_group_id"] == peptide_group_id("PEPTIDEK")
    assert manifest.iloc[0]["peptide_group_id"] == peptide_group_id(
        "PEPTIDEK")
    assert manifest.iloc[0]["parent_truth_rule"] == PARENT_TRUTH_RULE
    assert audit["counts"]["prepared_parents"] == 1
    assert on_disk_audit["inputs"]["psms"]["sha256"]
    assert "confirmation_table" not in on_disk_audit["inputs"]
    sidecar = validate_manifest(
        str(output_psms), HeavyType.SILAC, require=True)
    assert sidecar["dataset"]["counts_by_label_type"] == {"positive": 1}
