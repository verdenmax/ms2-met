import json
from pathlib import Path

import pandas as pd
import yaml

from spectrum.psm_identity import peptide_group_id
from tools.counterfactual_group_holdout import (
    SOURCE_COMPOSITION,
    SOURCE_ENTRAPMENT,
    SOURCE_KR,
    SOURCE_LOCAL,
    SOURCE_POSITIVE,
    HoldoutDesign,
    build_group_holdout,
    write_group_holdout_bundle,
)


def _eligible(**values):
    row = {
        "heavy_in_raw": 1,
        "heavy_out_of_range": 0,
        "precursor_xic_empty": 0,
        "q1a_valid": 1,
        "isotope_model_valid": 1,
        "precursor_pearson": 0.8,
        "charge": 2,
        "precursor_mz": 500.0,
        "rt": 20.0,
        "raw_title1": "raw_rep1",
        "label_type": "positive",
    }
    row.update(values)
    return row


def _counterfactual_fixture():
    rows = []
    for number in range(1, 7):
        parent = f"P{number}"
        sequence = f"PEPTLDEK{number}"
        base = {
            "parent_id": parent,
            "group_id": parent,
            "candidate_family_id": parent,
            "peptide_group_id": peptide_group_id(sequence),
        }
        rows.append(_eligible(
            **base, sequence=sequence, label=1,
            negative_source=SOURCE_POSITIVE, query_id=pd.NA,
            raw_title1="raw_rep1", rt=20.0 + number))
        if number == 2:
            rows.append(_eligible(
                **base, sequence=sequence, label=1,
                negative_source=SOURCE_POSITIVE, query_id=pd.NA,
                raw_title1="raw_rep2", rt=20.5 + number))
        for source, marker in (
                (SOURCE_COMPOSITION, "C"),
                (SOURCE_KR, "K"),
                (SOURCE_LOCAL, "L")):
            candidate_sequence = (
                "COLLIDEI" if number == 1 and source == SOURCE_LOCAL
                else f"{marker}ANDIDATE{number}")
            for candidate_number in range(2):
                rows.append(_eligible(
                    **base, sequence=candidate_sequence, label=0,
                    label_type="negative", negative_source=source,
                    query_id=f"Q-{number}-{marker}-{candidate_number}",
                    rt=30.0 + number + candidate_number / 10))
    # This row models the real extraction symptom: its declared parent was
    # removed upstream and therefore cannot safely enter any split.
    rows.append(_eligible(
        sequence="ORPHAN", label=0, label_type="negative",
        negative_source=SOURCE_LOCAL, query_id="Q-ORPHAN",
        parent_id="P-MISSING", group_id="P-MISSING",
        candidate_family_id="P-MISSING",
        peptide_group_id=peptide_group_id("MISSING")))
    return pd.DataFrame(rows)


def _entrapment_fixture():
    return pd.DataFrame([
        _eligible(
            sequence="COLLIDEL", label=0, label_type="negative",
            raw_title1="raw_rep3", rt=40.0),
        _eligible(
            sequence="REALERRORK", label=0, label_type="negative",
            raw_title1="raw_rep1", rt=41.0),
        # Although this bridge row fails the cohort, grouping must happen first
        # and therefore still force its connected P3 family into test.
        _eligible(
            sequence="CANDIDATE3", label=0, label_type="negative",
            raw_title1="raw_rep2", rt=41.5, heavy_in_raw=0),
        _eligible(
            sequence="DISCARDEDCORRECT", label=1, label_type="positive",
            raw_title1="raw_rep2", rt=42.0),
    ])


def test_group_holdout_forces_entrapment_collision_and_drops_orphans():
    bundle = build_group_holdout(
        _counterfactual_fixture(), _entrapment_fixture(),
        HoldoutDesign(holdout_fraction=0.20, seed=19))

    manifest = bundle.split_manifest
    assert "Q-ORPHAN" not in set(manifest["query_id"].dropna())
    assert bundle.audit["inputs"]["n_orphan_synthetic_rows_dropped"] == 1
    assert bundle.audit["inputs"]["n_entrapment_correct_rows_discarded"] == 1

    p1_rows = manifest[manifest["parent_id"].eq("P1")]
    p3_rows = manifest[manifest["parent_id"].eq("P3")]
    entrapment_rows = manifest[manifest["negative_source"].eq(SOURCE_ENTRAPMENT)]
    assert set(p1_rows["experiment_split"]) == {"test"}
    assert set(p3_rows["experiment_split"]) == {"test"}
    assert set(entrapment_rows["experiment_split"]) == {"test"}
    assert bundle.audit["split"]["n_entrapment_connected_groups"] > bundle.audit[
        "split"]["n_evaluable_entrapment_connected_groups"]
    train_groups = set(manifest.loc[
        manifest["experiment_split"].eq("train"), "leakage_group_id"])
    test_groups = set(manifest.loc[
        manifest["experiment_split"].eq("test"), "leakage_group_id"])
    assert train_groups.isdisjoint(test_groups)
    assert bundle.audit["validation"][
        "n_train_primary_test_overlapping_groups"] == 0

    errors = bundle.primary_test[bundle.primary_test["label"].eq(0)]
    assert set(errors["negative_source"]) == {SOURCE_ENTRAPMENT}
    assert set(bundle.synthetic_diagnostics["negative_source"]).issubset({
        SOURCE_COMPOSITION, SOURCE_KR, SOURCE_LOCAL})


def test_variants_share_positives_and_select_one_candidate_per_parent_source():
    bundle = build_group_holdout(
        _counterfactual_fixture(), _entrapment_fixture(),
        HoldoutDesign(holdout_fraction=0.20, seed=23))
    variants = bundle.train_variants
    positive_ids = {
        name: set(frame.loc[
            frame["negative_source"].eq(SOURCE_POSITIVE),
            "experiment_sample_id"])
        for name, frame in variants.items()
    }
    assert len({frozenset(ids) for ids in positive_ids.values()}) == 1

    expected_sources = {
        "m_c": {SOURCE_POSITIVE, SOURCE_COMPOSITION},
        "m_k": {SOURCE_POSITIVE, SOURCE_KR},
        "m_l": {SOURCE_POSITIVE, SOURCE_LOCAL},
        "m_all": {
            SOURCE_POSITIVE, SOURCE_COMPOSITION, SOURCE_KR, SOURCE_LOCAL},
    }
    for name, frame in variants.items():
        assert set(frame["negative_source"]) == expected_sources[name]
        synthetic = frame[frame["negative_source"].ne(SOURCE_POSITIVE)]
        assert not synthetic.duplicated(["parent_id", "negative_source"]).any()


def test_split_and_candidate_selection_are_independent_of_input_row_order():
    counterfactual = _counterfactual_fixture()
    entrapment = _entrapment_fixture()
    design = HoldoutDesign(holdout_fraction=0.20, seed=31)
    first = build_group_holdout(counterfactual, entrapment, design)
    second = build_group_holdout(
        counterfactual.sample(frac=1, random_state=7),
        entrapment.sample(frac=1, random_state=8), design)

    first_split = dict(zip(
        first.split_manifest["experiment_sample_id"],
        first.split_manifest["experiment_split"]))
    second_split = dict(zip(
        second.split_manifest["experiment_sample_id"],
        second.split_manifest["experiment_split"]))
    assert first_split == second_split
    for variant in first.train_variants:
        assert set(first.train_variants[variant]["experiment_sample_id"]) == set(
            second.train_variants[variant]["experiment_sample_id"])


def test_written_bundle_pins_one_test_and_canonical_training_contract(tmp_path):
    bundle = build_group_holdout(
        _counterfactual_fixture(), _entrapment_fixture(),
        HoldoutDesign(holdout_fraction=0.20, seed=37))
    template = tmp_path / "template.yaml"
    template.write_text(yaml.safe_dump({
        "data": {"train_files": [], "test_files": []},
        "model": {"type": "lightgbm", "params": {"seed": 42}},
        "training": {"cv_folds": 5, "cv_seed": 42},
        "operating_point": {
            "target_fprs": [0.01, 0.05, 0.10],
            "primary_target_fpr": 0.05,
        },
        "output": {"model_path": "unused", "result_path": "unused"},
    }), encoding="utf-8")
    root = write_group_holdout_bundle(bundle, tmp_path / "bundle", template)

    test_paths = set()
    for variant in ("m_c", "m_k", "m_l", "m_all"):
        config = yaml.safe_load((root / "configs" / f"{variant}.yaml").read_text())
        test_paths.add(tuple(config["data"]["test_files"]))
        assert config["data"]["group_col"] == "leakage_group_id"
        assert config["data"]["feature_arm"] == "ms1_ms2_no_prediction"
        assert config["data"]["cohort"] == "evidence_observed"
        assert config["operating_point"]["target_fprs"] == [0.01, 0.05, 0.1]
        assert config["operating_point"]["primary_target_fpr"] == 0.05
        assert config["evaluation_semantics"]["positive_class"] == (
            "incorrect_identification")
        assert "per-member outer-OOF" in config[
            "evaluation_semantics"]["external_threshold_contract"]
    assert test_paths == {(str(root / "test_gold_entrapment.csv"),)}

    status = json.loads((root / "bundle_status.json").read_text())
    checksums = json.loads((root / "artifact_checksums.json").read_text())
    assert status["status"] == "complete"
    assert status["metric_semantics"] == "error_identification_positive_v1"
    assert "test_gold_entrapment.csv" in checksums["artifacts"]

    from tools.spec_trainer.src.feature_groups import audit_feature_registry
    header = pd.read_csv(root / "train_m_c.csv", nrows=0).columns
    registry_audit = audit_feature_registry(header)
    assert registry_audit.is_complete
