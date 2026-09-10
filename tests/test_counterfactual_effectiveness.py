import hashlib
import json
from pathlib import Path

import numpy as np
import pandas as pd
import pytest
import yaml

from spectrum.psm_identity import peptide_group_id
from tools.counterfactual_effectiveness import (
    MODEL_SOURCES,
    SOURCE_REAL_CORRECT,
    SOURCE_REAL_ERROR,
    EffectivenessDesign,
    build_effectiveness_bundle,
    summarize_effectiveness,
    verify_effectiveness_bundle,
    write_effectiveness_bundle,
)
from tools.counterfactual_group_holdout import (
    SOURCE_COMPOSITION,
    SOURCE_KR,
    SOURCE_LOCAL,
    SOURCE_POSITIVE,
)
from tools.spec_trainer.src.cv_core import (
    METRIC_SEMANTICS_VERSION,
    evaluate_at_threshold,
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
        "q_value": 0.005,
    }
    row.update(values)
    return row


def _real_fixture():
    rows = []
    for number in range(12):
        sequence = f"REALPEPTIDEK{number}"
        for rep in range(2):
            rows.append(_eligible(
                sequence=sequence, label=1, raw_title1=f"raw_rep{rep}",
                rt=20.0 + number + rep / 10))
    for number in range(9):
        rows.append(_eligible(
            sequence=f"TRAPERRORK{number}", label=0,
            label_type="negative", rt=50.0 + number))
    return pd.DataFrame(rows)


def _counterfactual_fixture():
    rows = []
    for number in range(8):
        parent = f"P{number}"
        sequence = f"REALPEPTIDEK{number}"
        family = {
            "parent_id": parent,
            "group_id": parent,
            "candidate_family_id": parent,
            "peptide_group_id": peptide_group_id(sequence),
        }
        rows.append(_eligible(
            **family, sequence=sequence, label=1,
            negative_source=SOURCE_POSITIVE, query_id=pd.NA,
            rt=20.0 + number))
        for source, marker in (
                (SOURCE_COMPOSITION, "C"),
                (SOURCE_KR, "K"),
                (SOURCE_LOCAL, "L")):
            for candidate in range(2):
                rows.append(_eligible(
                    **family, sequence=f"{marker}CANDIDATE{number}", label=0,
                    label_type="negative", negative_source=source,
                    query_id=f"Q-{marker}-{number}-{candidate}",
                    rt=30.0 + number + candidate / 10))
    rows.append(_eligible(
        sequence="ORPHAN", label=0, label_type="negative",
        negative_source=SOURCE_LOCAL, query_id="Q-ORPHAN",
        parent_id="P-MISSING", group_id="P-MISSING",
        candidate_family_id="P-MISSING",
        peptide_group_id=peptide_group_id("MISSING")))
    return pd.DataFrame(rows)


def _design(bootstrap_reps=30):
    return EffectivenessDesign(
        outer_folds=3, seed=17, bootstrap_reps=bootstrap_reps,
        bootstrap_seed=23, minimum_recall_gain=0.03,
        max_fpr_increase=0.01)


def _template(path: Path):
    path.write_text(yaml.safe_dump({
        "data": {"train_files": [], "test_files": []},
        "model": {"type": "lightgbm", "params": {"seed": 42}},
        "training": {"cv_folds": 3, "cv_seed": 42},
        "operating_point": {"target_fprs": [0.05]},
        "output": {"model_path": "unused", "result_path": "unused"},
    }), encoding="utf-8")


def _write_fake_training_result(root: Path, fold: int, model: str,
                                frame: pd.DataFrame):
    operating_points = {}
    for target in (1, 5, 10):
        vote = frame[f"fpr_{target}_error_vote_fraction"].to_numpy()
        point = evaluate_at_threshold(frame["label"], 1.0 - vote, 0.5)
        operating_points[f"fpr_{target}"] = {
            "external_ensemble": {
                "method": "fold_calibrated_majority_vote",
                "test_metrics": point,
            },
        }
    config_path = root / "configs" / f"fold_{fold}" / f"{model}.yaml"
    result = {
        "metric_semantics": METRIC_SEMANTICS_VERSION,
        "positive_class": "incorrect_identification",
        "mode": "cross_test",
        "operating_points": operating_points,
        "provenance": {
            "config_sha256": hashlib.sha256(config_path.read_bytes()).hexdigest(),
        },
    }
    output = root / "training" / f"fold_{fold}" / model
    (output / "training.cv.json").write_text(
        json.dumps(result), encoding="utf-8")


def test_outer_folds_test_every_real_row_once_without_family_leakage():
    bundle = build_effectiveness_bundle(
        _counterfactual_fixture(), _real_fixture(), _design())
    real = bundle.real_rows
    assert set(real["negative_source"]) == {
        SOURCE_REAL_CORRECT, SOURCE_REAL_ERROR}
    assert not real["experiment_sample_id"].duplicated().any()
    assert set(real["experiment_outer_fold"]) == {0, 1, 2}
    assert bundle.audit["validation"]["every_real_row_is_tested_once"] is True
    assert bundle.audit["inputs"]["counterfactual"][
        "n_orphan_synthetic_rows"] == 1
    orphan = bundle.synthetic_diagnostics.loc[
        bundle.synthetic_diagnostics["experiment_orphan_synthetic"]]
    assert len(orphan) == 1
    assert orphan.iloc[0]["experiment_role"] == "orphan_graph_bridge"

    for fold in range(3):
        train = real[real["experiment_outer_fold"].ne(fold)]
        test = real[real["experiment_outer_fold"].eq(fold)]
        candidates = bundle.selected_synthetic[
            bundle.selected_synthetic["experiment_outer_fold"].ne(fold)]
        test_groups = set(test["leakage_group_id"])
        assert test_groups.isdisjoint(set(train["leakage_group_id"]))
        assert test_groups.isdisjoint(set(candidates["leakage_group_id"]))
        assert set(test["label"]) == {0, 1}


def test_real_baseline_and_augmented_models_share_all_real_training_rows(tmp_path):
    bundle = build_effectiveness_bundle(
        _counterfactual_fixture(), _real_fixture(), _design())
    template = tmp_path / "template.yaml"
    _template(template)
    root = write_effectiveness_bundle(bundle, tmp_path / "bundle", template)

    for fold in range(3):
        configs = {
            model: yaml.safe_load((
                root / "configs" / f"fold_{fold}" / f"{model}.yaml").read_text())
            for model in MODEL_SOURCES
        }
        real_file = str(root / "folds" / f"fold_{fold}" / "train_real_q01.csv")
        test_file = str(root / "folds" / f"fold_{fold}" / "test_real_q01.csv")
        assert all(config["data"]["train_files"][0] == real_file
                   for config in configs.values())
        assert {tuple(config["data"]["test_files"])
                for config in configs.values()} == {(test_file,)}
        assert len(configs["m_real"]["data"]["train_files"]) == 1
        assert len(configs["m_real_all"]["data"]["train_files"]) == 4
        assert all(config["data"]["group_col"] == "leakage_group_id"
                   for config in configs.values())
        assert all(config["data"]["frozen_group_graph"] is True
                   for config in configs.values())
        assert all(config["data"]["predefined_cv_fold_col"] ==
                   "experiment_inner_fold" for config in configs.values())
        assert all(len(config["data"]["predefined_inner_valid_cols"]) == 3
                   for config in configs.values())
        assert all(config["operating_point"]["primary_target_fpr"] == 0.05
                   for config in configs.values())

        real_train = pd.read_csv(real_file)
        for source in ("composition", "kr", "local"):
            synthetic = pd.read_csv(
                root / "folds" / f"fold_{fold}" /
                f"train_synthetic_{source}.csv")
            protocol = synthetic.merge(
                real_train[["leakage_group_id", "experiment_inner_fold"]]
                .drop_duplicates(), on="leakage_group_id",
                suffixes=("_synthetic", "_real"), validate="many_to_one")
            assert (protocol["experiment_inner_fold_synthetic"] ==
                    protocol["experiment_inner_fold_real"]).all()


def test_effectiveness_split_is_stable_under_input_row_order():
    cf = _counterfactual_fixture()
    real = _real_fixture()
    first = build_effectiveness_bundle(cf, real, _design())
    second = build_effectiveness_bundle(
        cf.sample(frac=1, random_state=3),
        real.sample(frac=1, random_state=4), _design())
    first_folds = dict(zip(
        first.real_rows["experiment_sample_id"],
        first.real_rows["experiment_outer_fold"]))
    second_folds = dict(zip(
        second.real_rows["experiment_sample_id"],
        second.real_rows["experiment_outer_fold"]))
    assert first_folds == second_folds
    assert set(first.selected_synthetic["experiment_sample_id"]) == set(
        second.selected_synthetic["experiment_sample_id"])


def test_real_input_must_be_prefiltered_to_q01():
    real = _real_fixture()
    real.loc[0, "q_value"] = 0.0101
    with pytest.raises(ValueError, match="prefiltered"):
        build_effectiveness_bundle(
            _counterfactual_fixture(), real, _design())


def test_real_input_rejects_inconsistent_entrapment_label_type():
    real = _real_fixture()
    real.loc[real["label"].eq(0).idxmax(), "label_type"] = "positive"
    with pytest.raises(ValueError, match="label_type"):
        build_effectiveness_bundle(
            _counterfactual_fixture(), real, _design())


def test_orphan_relationships_still_bridge_real_and_parent_families():
    counterfactual = _counterfactual_fixture()
    orphan_index = counterfactual["query_id"].eq("Q-ORPHAN").idxmax()
    counterfactual.loc[orphan_index, "sequence"] = "REALPEPTIDEK8"
    counterfactual.loc[orphan_index, "group_id"] = "P0"
    bundle = build_effectiveness_bundle(
        counterfactual, _real_fixture(), _design())
    groups = bundle.real_rows.groupby("sequence")["leakage_group_id"].first()
    assert groups["REALPEPTIDEK0"] == groups["REALPEPTIDEK8"]
    assert not bundle.selected_synthetic["query_id"].eq("Q-ORPHAN").any()


def test_existing_real_peptide_family_relationship_is_preserved():
    real = _real_fixture()
    real["peptide_group_id"] = ""
    real.loc[real["sequence"].isin(
        ["REALPEPTIDEK0", "REALPEPTIDEK8"]), "peptide_group_id"] = "shared-real"

    bundle = build_effectiveness_bundle(
        _counterfactual_fixture(), real, _design())

    groups = bundle.real_rows.groupby("sequence")["leakage_group_id"].first()
    assert groups["REALPEPTIDEK0"] == groups["REALPEPTIDEK8"]


def test_all_synthetic_sources_must_have_eligible_candidates():
    counterfactual = _counterfactual_fixture()
    counterfactual = counterfactual.loc[
        counterfactual["negative_source"].ne(SOURCE_LOCAL)]
    with pytest.raises(ValueError, match="no eligible.*local_mass_gap"):
        build_effectiveness_bundle(
            counterfactual, _real_fixture(), _design())


@pytest.mark.parametrize("values", [
    {"minimum_recall_gain": -0.1},
    {"minimum_recall_gain": 1.1},
    {"max_fpr_increase": -0.1},
    {"max_fpr_increase": 1.1},
    {"familywise_alpha": 0.0},
])
def test_design_rejects_invalid_success_criteria(values):
    with pytest.raises(ValueError):
        EffectivenessDesign(**values)


def test_bundle_verification_rejects_a_changed_frozen_file(tmp_path):
    bundle = build_effectiveness_bundle(
        _counterfactual_fixture(), _real_fixture(), _design())
    template = tmp_path / "template.yaml"
    _template(template)
    root = write_effectiveness_bundle(bundle, tmp_path / "bundle", template)
    assert verify_effectiveness_bundle(root)["verified_artifacts"] > 0
    changed = root / "folds" / "fold_0" / "train_real_q01.csv"
    changed.write_text(changed.read_text() + "\n", encoding="utf-8")
    with pytest.raises(ValueError, match="artifact changed"):
        verify_effectiveness_bundle(root)


def test_summary_uses_all_real_outer_predictions_and_paired_groups(tmp_path):
    bundle = build_effectiveness_bundle(
        _counterfactual_fixture(), _real_fixture(), _design(bootstrap_reps=40))
    template = tmp_path / "template.yaml"
    _template(template)
    root = write_effectiveness_bundle(bundle, tmp_path / "bundle", template)

    for fold in range(3):
        test = pd.read_csv(root / "folds" / f"fold_{fold}" / "test_real_q01.csv")
        for model in MODEL_SOURCES:
            is_error = test["label"].eq(0).to_numpy()
            augmented = model != "m_real"
            frame = test[[
                "experiment_sample_id", "experiment_outer_fold",
                "leakage_group_id", "label", "negative_source", "sequence",
                "charge",
            ]].copy()
            frame["ensemble_trust_score"] = np.where(
                is_error, 0.1 if augmented else 0.8, 0.9)
            for target in (1, 5, 10):
                frame[f"fpr_{target}_error_vote_fraction"] = np.where(
                    is_error, 1.0 if augmented else 0.0, 0.0)
            output = root / "training" / f"fold_{fold}" / model
            output.mkdir(parents=True)
            frame.to_csv(output / "training.cv.test_scores.csv", index=False)
            _write_fake_training_result(root, fold, model, frame)

    summary = summarize_effectiveness(root)
    assert summary["n_actual_correct"] == int(bundle.real_rows["label"].eq(1).sum())
    assert summary["n_actual_error"] == int(bundle.real_rows["label"].eq(0).sum())
    assert all(item["effectiveness_supported"]
               for item in summary["comparisons"])
    assert summary["models"]["m_real_all"]["fnr_at_fpr5"] == 0.0
    external = summary["models"]["m_real_all"]["operating_points"][
        "fpr_5"]["external_ensemble"]
    assert external["method"] == "fold_calibrated_majority_vote"
    baseline_point = summary["models"]["m_real"]["operating_points"][
        "fpr_5"]["external_ensemble"]["test_metrics"]
    augmented_point = external["test_metrics"]
    assert (baseline_point["tp"], baseline_point["fp"],
            baseline_point["fn"], baseline_point["tn"]) == (
                0, 0, summary["n_actual_error"], summary["n_actual_correct"])
    assert baseline_point["fpr"] == 0.0
    assert baseline_point["fnr"] == 1.0
    assert (augmented_point["tp"], augmented_point["fp"],
            augmented_point["fn"], augmented_point["tn"]) == (
                summary["n_actual_error"], 0, 0, summary["n_actual_correct"])
    assert augmented_point["fpr"] == 0.0
    assert augmented_point["fnr"] == 0.0
    pooled = pd.read_csv(root / "pooled_real_test_predictions.csv")
    assert set(pooled["metric_semantics"]) == {METRIC_SEMANTICS_VERSION}
    assert set(pooled["positive_class"]) == {"incorrect_identification"}
    assert all(item["multiplicity_adjustment"].startswith("bonferroni")
               for item in summary["comparisons"])
    assert (root / "effectiveness_summary.csv").is_file()
    assert (root / "paired_group_bootstrap.csv").is_file()
    status = json.loads((root / "bundle_status.json").read_text())
    assert status["status"] == "complete"
