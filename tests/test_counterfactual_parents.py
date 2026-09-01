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
        confirmation_rule="manual_pair_review_v1",
    )
    values.update(overrides)
    return ParentPreparationConfig(**values)


def _confirmations(rows):
    return pd.DataFrame(rows, columns=[
        "sequence", "charge", "raw_title1", "label_type",
        "heavy_confirmed",
    ])


def _splits():
    return pd.DataFrame([
        {"raw_title": "raw_train", "dataset_split": "label_train"},
        {"raw_title": "raw_dev", "dataset_split": "label_dev"},
    ])


def test_prepares_only_confirmed_in_split_and_groups_across_charge_and_li():
    train_i = _psm("PEPTIDEK", charge=2)
    train_l = _psm("PEPTLDEK", charge=3)
    unconfirmed = _psm("MELTPEPK", charge=2)
    outside = _psm("PEPTIDEK", charge=2, raw="raw_dev")
    negative = _psm("NEGATIVEK", label_type="negative")
    confirmations = _confirmations([
        ["PEPTIDEK", 2, "raw_train", "positive", 1],
        ["PEPTLDEK", 3, "raw_train", "positive", "confirmed"],
        ["MELTPEPK", 2, "raw_train", "positive", 0],
        ["PEPTIDEK", 2, "raw_dev", "positive", True],
    ])

    result = prepare_counterfactual_parents(
        [outside, negative, train_l, unconfirmed, train_i],
        confirmations, _splits(), _cfg())

    assert result.audit["schema"] == PREPARATION_SCHEMA
    assert result.audit["peptide_group_id_schema"] == PEPTIDE_GROUP_ID_SCHEMA
    assert result.audit["counts"]["prepared_parents"] == 2
    assert result.audit["counts"]["peptide_groups"] == 1
    assert result.audit["failures"] == {
        "not_heavy_confirmed": 1,
        "not_positive": 1,
        "outside_dataset_split": 1,
    }
    assert [psm._charge for psm in result.psms] == [2, 3]
    assert {psm._peptide_group_id for psm in result.psms} == {
        peptide_group_id("PEPTIDEK")}
    assert all(psm._heavy_confirmed is True for psm in result.psms)
    assert all(psm._dataset_split == "label_train" for psm in result.psms)
    assert set(result.manifest["heavy_confirmed"]) == {1}
    assert set(result.manifest["confirmation_rule"]) == {
        "manual_pair_review_v1"}


def test_requires_complete_raw_mapping_and_unique_confirmation_identity():
    parent = _psm()
    confirmation = _confirmations([
        ["PEPTIDEK", 2, "raw_train", "positive", 1],
    ])
    with pytest.raises(ValueError, match="missing from raw split"):
        prepare_counterfactual_parents(
            [parent], confirmation,
            pd.DataFrame([{
                "raw_title": "other_raw", "dataset_split": "label_train"}]),
            _cfg())

    duplicate = pd.concat([confirmation, confirmation], ignore_index=True)
    with pytest.raises(ValueError, match="duplicate parent identity"):
        prepare_counterfactual_parents(
            [parent], duplicate, _splits(), _cfg())


def test_does_not_derive_confirmation_or_allow_unversioned_rule():
    parent = _psm()
    confirmation = _confirmations([
        ["PEPTIDEK", 2, "raw_train", "positive", 0],
    ])
    with pytest.raises(ValueError, match="no eligible heavy-confirmed"):
        prepare_counterfactual_parents(
            [parent], confirmation, _splits(), _cfg())
    with pytest.raises(ValueError, match="confirmation_rule"):
        prepare_counterfactual_parents(
            [parent], confirmation, _splits(),
            _cfg(confirmation_rule=""))


def test_cli_adapter_writes_prepared_json_manifest_and_audit(tmp_path):
    parent = _psm()
    input_psms = tmp_path / "all.json"
    input_psms.write_text(
        json.dumps([parent.to_dict()]), encoding="utf-8")
    write_manifest(str(input_psms), [parent], HeavyType.SILAC)
    confirmation_path = tmp_path / "confirm.csv"
    _confirmations([
        ["PEPTIDEK", 2, "raw_train", "positive", 1],
    ]).to_csv(confirmation_path, index=False)
    split_path = tmp_path / "splits.csv"
    _splits().to_csv(split_path, index=False)
    output_psms = tmp_path / "prepared.json"
    output_manifest = tmp_path / "prepared.tsv"
    output_audit = tmp_path / "prepared.audit.json"
    config_path = tmp_path / "prepare.ini"
    config_path.write_text(
        "[counterfactual_parents]\n"
        "dataset_split=label_train\n"
        "confirmation_rule=manual_pair_review_v1\n",
        encoding="utf-8")
    job = ParentPreparationJob(
        input_psms=str(input_psms),
        confirmation_table=str(confirmation_path),
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
    assert audit["counts"]["prepared_parents"] == 1
    assert on_disk_audit["inputs"]["confirmation_table"]["sha256"]
    sidecar = validate_manifest(
        str(output_psms), HeavyType.SILAC, require=True)
    assert sidecar["dataset"]["counts_by_label_type"] == {"positive": 1}
