import importlib.util
import json
import os
import subprocess
import sys
from pathlib import Path

import numpy as np
import pandas as pd
import pytest
import yaml

sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..",
                                "tools", "spec_trainer", "src"))

from feature_groups import EVIDENCE_CORE_FEATURES

_HAS_LGB = importlib.util.find_spec("lightgbm") is not None
requires_lgb = pytest.mark.skipif(not _HAS_LGB, reason="lightgbm not installed")


def _rows(seed=4, complete_core=False):
    rng = np.random.default_rng(seed)
    rows = []
    strata = [("correct", 1, 70), ("t5", 0, 24),
              ("t5_10", 0, 24), ("t10_20", 0, 24)]
    for tier, label, count in strata:
        for i in range(count):
            row = {
                "sequence": f"{tier}_PEP_{i}",
                "charge": 2,
                "precursor_mz": 400.0 + len(rows) * 0.01,
                "rt": 10.0 + len(rows) * 0.02,
                "raw_title1": "raw_A",
                "label_type": "positive" if label else "negative",
                "label": label,
                "precursor_pearson": rng.normal(label, 0.2),
                "heavy_in_raw": 1,
                "heavy_out_of_range": 0,
                "precursor_xic_empty": 0,
                "q1a_valid": 1,
                "has_lib_pred": 1,
                "isotope_model_valid": 1,
                "_tier": tier,
            }
            if complete_core:
                for j, column in enumerate(sorted(EVIDENCE_CORE_FEATURES)):
                    row[column] = rng.normal(label + j * 0.001, 0.2)
            rows.append(row)
    return pd.DataFrame(rows)


def _write_pools(root, complete_core=False, perturb=False, dataset="2da"):
    source = _rows(complete_core=complete_core)
    included = {
        "neg05": {"correct", "t5"},
        "neg10": {"correct", "t5", "t5_10"},
        "neg20": {"correct", "t5", "t5_10", "t10_20"},
    }
    for pool, tiers in included.items():
        frame = source[source["_tier"].isin(tiers)].drop(columns="_tier")
        frame = frame.copy().reset_index(drop=True)
        if perturb and pool == "neg05":
            frame.loc[0, "precursor_pearson"] += 1
        path = root / f"baseline_{dataset}_{pool}" / "features.csv"
        path.parent.mkdir(parents=True)
        frame.to_csv(path, index=False)


def _write_all_dataset_pools(root, complete_core=False):
    for dataset in ("2da", "5da", "normal"):
        _write_pools(root, complete_core=complete_core, dataset=dataset)


def _cfg(feature_arm="ms1_only", rounds=15):
    return {
        "data": {
            "feature_cols": [], "target_col": "label",
            "feature_arm": feature_arm, "drop_features": [],
            "cohort": "evidence_common", "group_col": "sequence",
            "require_complete_arm": feature_arm == "evidence_core",
        },
        "model": {"type": "lightgbm", "params": {
            "objective": "binary", "metric": ["auc", "binary_logloss"],
            "num_leaves": 7, "learning_rate": 0.1,
            "min_data_in_leaf": 3, "verbose": -1,
            "bagging_fraction": 0.8, "bagging_freq": 1,
            "seed": 42, "feature_fraction_seed": 42,
            "bagging_seed": 42, "data_random_seed": 42,
            "deterministic": True, "force_col_wise": True,
        }},
        "training": {
            "num_boost_round": rounds, "early_stopping_rounds": 5,
            "early_stopping_first_metric_only": True,
            "cv_folds": 2, "cv_seed": 42, "valid_size": 0.25,
            "min_class_groups_per_split": 1,
        },
        "operating_point": {
            "target_fprs": [0.05, 0.10], "primary_target_fpr": 0.10,
        },
        "output": {"model_path": "unused.txt",
                   "result_path": "unused.json"},
    }


def test_prepare_validates_nesting_and_reuses_one_fold_map(tmp_path):
    import fixed_negpool

    _write_pools(tmp_path)
    prepared = fixed_negpool.prepare_fixed_negpool(
        fixed_negpool.feature_paths(tmp_path, "2da"), _cfg(),
        min_test_errors_per_tier=1, split_candidates=32)
    assert prepared.identity_cols == [
        "sequence", "charge", "precursor_mz", "rt", "raw_title1",
        "label_type"]
    assert prepared.validation["nesting"]["nested_error_rows"] is True
    assert prepared.validation["nesting"]["identical_correct_rows"] is True
    assert set(prepared.frame["negative_tier"]) == {
        "correct", "t5", "t5_10", "t10_20"}
    test = prepared.frame[prepared.frame["fixed_split"].eq("test")]
    assert set(test["negative_tier"]) == {
        "correct", "t5", "t5_10", "t10_20"}
    assert prepared.group_fold_map["sequence"].is_unique
    assert set(prepared.frame.loc[
        prepared.frame["fixed_split"].eq("train"), "outer_fold"]) == {0, 1}


def test_prepare_rejects_shared_feature_drift(tmp_path):
    import fixed_negpool

    _write_pools(tmp_path, perturb=True)
    with pytest.raises(ValueError, match="formal feature values differ"):
        fixed_negpool.prepare_fixed_negpool(
            fixed_negpool.feature_paths(tmp_path, "2da"), _cfg(),
            min_test_errors_per_tier=1, split_candidates=8)


def test_validation_only_prepare_does_not_generate_assignments(tmp_path):
    import fixed_negpool

    _write_pools(tmp_path)
    prepared = fixed_negpool.prepare_fixed_negpool(
        fixed_negpool.feature_paths(tmp_path, "2da"), _cfg(),
        min_test_errors_per_tier=1, split_candidates=8,
        generate_assignments=False)
    assert set(prepared.frame["fixed_split"]) == {
        "deferred_to_frozen_manifest"}
    assert set(prepared.frame["outer_fold"]) == {-1}
    assert prepared.validation["fixed_split"]["method"] == \
        "not_generated_consume_frozen_manifest"


@pytest.mark.parametrize("combined", [False, True])
def test_fixed_preparation_preserves_family_after_bridge_is_filtered(
        tmp_path, combined):
    import fixed_negpool

    datasets = ("2da", "5da", "normal") if combined else ("2da",)
    for dataset in datasets:
        _write_pools(tmp_path, dataset=dataset)
        for path in fixed_negpool.feature_paths(tmp_path, dataset).values():
            frame = pd.read_csv(path)
            for column in ("parent_id", "group_id", "query_id"):
                frame[column] = pd.Series(pd.NA, index=frame.index, dtype="string")
            frame["negative_source"] = "gold_real"
            parents = frame.index[frame.label.eq(1)][:2]
            children = frame.index[frame.label.eq(0)][:2]
            for index, (parent, child) in enumerate(zip(parents, children), 1):
                frame.loc[[parent, child], ["parent_id", "group_id"]] = f"P{index}"
                frame.loc[parent, "sequence"] = "PEPTIDEK"
                frame.loc[[parent, child], "charge"] = index + 1
                frame.loc[child, "sequence"] = f"CHILD{index}K"
                frame.loc[child, "query_id"] = f"Q{index}"
                frame.loc[child, "negative_source"] = "silver_synthetic_shuffle"
            # This parent is the only remaining sequence bridge for Q2.
            frame.loc[parents[1], "q1a_valid"] = 0
            frame.to_csv(path, index=False)

    cfg = _cfg()
    if combined:
        prepared = fixed_negpool.prepare_combined_fixed_negpool(
            tmp_path, cfg, generate_assignments=False)
    else:
        prepared = fixed_negpool.prepare_fixed_negpool(
            fixed_negpool.feature_paths(tmp_path, "2da"), cfg,
            generate_assignments=False)
    children = prepared.frame.query_id.isin(["Q1", "Q2"])
    assert children.sum() == 2 * len(datasets)
    assert prepared.frame.loc[children, prepared.split_group_col].nunique() == 1


def test_candidate_relationship_ids_form_connected_leakage_groups():
    import fixed_negpool

    frame = pd.DataFrame({
        "sequence": ["PARENT", "CHILD", "OTHER"],
        "group_id": ["P1", pd.NA, "P2"],
        "parent_id": [pd.NA, "P1", "P2"],
    })
    group_col, audit = fixed_negpool._assign_leakage_groups(
        frame, "sequence")
    assert group_col == "leakage_group_id"
    assert frame.loc[0, group_col] == frame.loc[1, group_col]
    assert frame.loc[0, group_col] != frame.loc[2, group_col]
    assert audit["relationship_ids_applied"] is True
    assert audit["candidate_family_leakage_protected"] is True


def test_partial_relationship_ids_do_not_overclaim_family_protection():
    import fixed_negpool

    frame = pd.DataFrame({
        "sequence": ["KNOWN", "CHILD", "UNKNOWN"],
        "parent_id": ["P1", "P1", pd.NA],
    })
    _, audit = fixed_negpool._assign_leakage_groups(frame, "sequence")
    assert audit["relationship_ids_applied"] is True
    assert audit["candidate_family_leakage_protected"] is False
    assert audit["relationship_id_coverage_fraction"] == pytest.approx(2 / 3)


def test_unique_query_ids_do_not_claim_candidate_family_protection():
    import fixed_negpool

    frame = pd.DataFrame({
        "sequence": ["A", "B", "C"],
        "query_id": ["Q1", "Q2", "Q3"],
    })
    _, audit = fixed_negpool._assign_leakage_groups(frame, "sequence")
    assert audit["relationship_ids_applied"] is True
    assert audit["family_relationship_columns_available"] == []
    assert audit["candidate_family_leakage_protected"] is False
    assert audit["relationship_id_coverage_fraction"] == 0.0


def test_unresolved_query_parent_does_not_claim_family_protection():
    import fixed_negpool

    frame = pd.DataFrame({
        "sequence": ["PARENT", "CHILD"],
        "query_id": [pd.NA, "Q1"],
        "group_id": ["P1", pd.NA],
        "parent_id": [pd.NA, "MISSING_PARENT"],
    })
    _, audit = fixed_negpool._assign_leakage_groups(frame, "sequence")
    assert audit["n_query_rows"] == 1
    assert audit["n_unresolved_query_parent_rows"] == 1
    assert audit["candidate_family_leakage_protected"] is False


def test_sibling_queries_cannot_resolve_each_others_missing_parent():
    import fixed_negpool

    frame = pd.DataFrame({
        "sequence": ["UNRELATED_ROOT", "CHILD_A", "CHILD_B"],
        "query_id": [pd.NA, "Q1", "Q2"],
        "group_id": ["OTHER", pd.NA, pd.NA],
        "parent_id": [pd.NA, "P1", "P1"],
    })
    _, audit = fixed_negpool._assign_leakage_groups(frame, "sequence")
    assert audit["n_query_rows"] == 2
    assert audit["n_unresolved_query_parent_rows"] == 2
    assert audit["candidate_family_leakage_protected"] is False


def test_prepare_combined_uses_global_sequence_split_and_twelve_strata(
        tmp_path):
    import fixed_negpool

    _write_all_dataset_pools(tmp_path)
    prepared = fixed_negpool.prepare_combined_fixed_negpool(
        tmp_path, _cfg(), min_test_errors_per_tier=1,
        split_candidates=32)
    assert set(prepared.frame["dataset"]) == {"2da", "5da", "normal"}
    assert prepared.frame["sample_id"].is_unique
    # Test has teeth: local source IDs deliberately collide across datasets,
    # while namespaced combined IDs must not.
    assert prepared.frame["source_sample_id"].duplicated().any()
    per_sequence = prepared.frame.groupby("sequence")["fixed_split"].nunique()
    assert per_sequence.max() == 1
    assert len(prepared.validation[
        "fixed_split"]["test_fraction_by_stratum"]) == 12
    assert len(prepared.validation[
        "fixed_split"]["minimum_test_rows_by_stratum"]) == 9
    assert len(prepared.validation[
        "fixed_test_dataset_tier_counts"]) == 12
    train = prepared.frame[prepared.frame["fixed_split"].eq("train")]
    assert train.groupby("sequence")["outer_fold"].nunique().max() == 1
    assert set(train["outer_fold"]) == {0, 1}


def test_paired_bootstrap_has_all_predeclared_comparisons():
    import fixed_negpool

    frame = pd.DataFrame({
        "sequence": [f"P{i}" for i in range(12)],
        "label": [1] * 6 + [0] * 6,
    })
    predictions = {}
    for offset, model in enumerate(("M5", "M10", "M20")):
        trust = np.array([0.9] * 6 + [0.4 - 0.1 * offset] * 6)
        predictions[model] = {
            "trust_score": trust,
            "fpr_5_vote_fraction": np.array([0.] * 6 + [1.] * 6),
            "fpr_10_vote_fraction": np.array([0.] * 6 + [1.] * 6),
        }
    result = fixed_negpool._bootstrap_paired(
        frame, predictions, reps=20, seed=1, group_col="sequence")
    assert len(result) == 3 * 4
    assert set(zip(result["model_a"], result["model_b"])) == {
        ("M5", "M10"), ("M10", "M20"), ("M5", "M20")}
    assert set(result["n_bootstrap"]) == {20}
    assert set(result["metric_semantics"]) == {
        "error_identification_positive_v1"}
    assert set(result["positive_class"]) == {"incorrect_identification"}


def test_failed_fixed_bundle_overwrite_preserves_previous_result(
        tmp_path, monkeypatch):
    import fixed_negpool

    output = tmp_path / "existing"
    output.mkdir()
    (output / "summary.json").write_text(
        '{"old": true}', encoding="utf-8")
    (output / "keep.txt").write_text("old bundle", encoding="utf-8")

    def fail_in_staging(*args, **kwargs):
        staging = Path(args[3])
        staging.mkdir(parents=True, exist_ok=True)
        (staging / "partial.txt").write_text("partial", encoding="utf-8")
        raise RuntimeError("synthetic training failure")

    monkeypatch.setattr(
        fixed_negpool, "_run_fixed_negpool_into_root", fail_in_staging)
    with pytest.raises(RuntimeError, match="synthetic training failure"):
        fixed_negpool.run_fixed_negpool(
            tmp_path / "config.yaml", tmp_path / "features", "2da",
            output, overwrite=True)
    assert json.loads((output / "summary.json").read_text()) == {"old": True}
    assert (output / "keep.txt").read_text() == "old bundle"
    assert not (output / "partial.txt").exists()


def test_make_fixed_negpool_2da_uses_runtime_roots(tmp_path):
    project_root = Path(__file__).resolve().parents[1]
    feature_root = tmp_path / "features"
    output_root = tmp_path / "fixed"
    for pool in ("neg05", "neg10", "neg20"):
        path = feature_root / f"baseline_2da_{pool}" / "features.csv"
        path.parent.mkdir(parents=True)
        path.touch()
    result = subprocess.run(
        ["make", "-n", "train-fixed-test-negpool-2da",
         f"FEATURE_ROOT={feature_root}",
         f"FIXED_NEGPOOL_OUTPUT_ROOT={output_root}",
         "FIXED_NEGPOOL_BOOTSTRAPS=17"],
        cwd=project_root, capture_output=True, text=True)
    assert result.returncode == 0, result.stderr
    assert f'--feature-root "{feature_root}"' in result.stdout
    assert f'--output-root "{output_root}/2da"' in result.stdout
    assert f'--config "{output_root}/configs/cv_in_2da_neg20.yaml"' \
        in result.stdout
    assert '--bootstrap-reps "17"' in result.stdout


def test_make_fixed_negpool_combined_uses_all_nine_inputs(tmp_path):
    project_root = Path(__file__).resolve().parents[1]
    feature_root = tmp_path / "features"
    output_root = tmp_path / "fixed"
    for dataset in ("2da", "5da", "normal"):
        for pool in ("neg05", "neg10", "neg20"):
            path = feature_root / f"baseline_{dataset}_{pool}" / "features.csv"
            path.parent.mkdir(parents=True, exist_ok=True)
            path.touch()
    result = subprocess.run(
        ["make", "-n", "train-fixed-test-negpool-combined",
         f"FEATURE_ROOT={feature_root}",
         f"FIXED_NEGPOOL_OUTPUT_ROOT={output_root}",
         "FIXED_NEGPOOL_BOOTSTRAPS=19"],
        cwd=project_root, capture_output=True, text=True)
    assert result.returncode == 0, result.stderr
    assert '--dataset combined' in result.stdout
    assert f'--output-root "{output_root}/combined"' in result.stdout
    assert '--bootstrap-reps "19"' in result.stdout


@requires_lgb
def test_run_fixed_negpool_writes_complete_paired_bundle(tmp_path):
    import fixed_negpool

    feature_root = tmp_path / "features"
    output_root = tmp_path / "out"
    _write_pools(feature_root, complete_core=True)
    config = _cfg(feature_arm="evidence_core", rounds=10)
    config_path = tmp_path / "cfg.yaml"
    config_path.write_text(yaml.safe_dump(config))
    result = fixed_negpool.run_fixed_negpool(
        config_path, feature_root, "2da", output_root,
        min_test_errors_per_tier=1, split_candidates=16,
        bootstrap_reps=5)
    assert result["experiment"] == "fixed_test_nested_negative_pool_v1"
    assert set(result["models"]) == {"M5", "M10", "M20"}
    assert (output_root / "summary.json").exists()
    assert (output_root / "manifests" / "fixed_test_manifest.csv").exists()
    fixed = pd.read_csv(output_root / "fixed_test_summary.csv")
    assert fixed["model"].tolist() == ["M5", "M10", "M20"]
    assert fixed["n_rows"].nunique() == 1
    assert set(fixed["metric_semantics"]) == {
        "error_identification_positive_v1"}
    assert set(fixed["positive_class"]) == {"incorrect_identification"}
    assert len(pd.read_csv(output_root / "tier_summary.csv")) == 9
    assert len(pd.read_csv(output_root / "paired_bootstrap.csv")) == 12
    loaded = json.loads((output_root / "summary.json").read_text())
    assert loaded["positive_class"] == "incorrect_identification"
    with pytest.raises(FileExistsError, match="refusing to overwrite"):
        fixed_negpool.run_fixed_negpool(
            config_path, feature_root, "2da", output_root,
            min_test_errors_per_tier=1, split_candidates=16,
            bootstrap_reps=1)


@requires_lgb
def test_run_combined_fixed_negpool_writes_domain_outputs(tmp_path):
    import fixed_negpool

    feature_root = tmp_path / "features"
    output_root = tmp_path / "combined"
    _write_all_dataset_pools(feature_root, complete_core=True)
    config = _cfg(feature_arm="evidence_core", rounds=10)
    config_path = tmp_path / "cfg.yaml"
    config_path.write_text(yaml.safe_dump(config))
    result = fixed_negpool.run_fixed_negpool(
        config_path, feature_root, "combined", output_root,
        min_test_errors_per_tier=1, split_candidates=16,
        bootstrap_reps=3)
    assert result["experiment"] == \
        "combined_fixed_test_nested_negative_pool_v1"
    assert result["design"]["combined_domain_reporting"] == \
        "pooled + equal-weight macro + per-dataset"
    fixed = pd.read_csv(output_root / "fixed_test_summary.csv")
    assert fixed["model"].tolist() == ["M5", "M10", "M20"]
    domain = pd.read_csv(output_root / "domain_summary.csv")
    assert len(domain) == 12
    assert set(domain["test_dataset"]) == {
        "2da", "5da", "normal", "macro_equal_weight"}
    assert len(pd.read_csv(output_root / "domain_tier_summary.csv")) == 27
    by_domain = pd.read_csv(output_root / "paired_bootstrap_by_domain.csv")
    assert len(by_domain) == 36
    assert set(by_domain["test_dataset"]) == {"2da", "5da", "normal"}
    manifest = pd.read_csv(
        output_root / "manifests" / "fixed_test_manifest.csv")
    assert {"dataset", "source_sample_id", "dataset_tier_stratum"} \
        <= set(manifest)
    used = yaml.safe_load((output_root / "config_used.yaml").read_text())
    assert len(used["data"]["train_files"]) == 3
    assert used["data"]["combined_fixed_test"]["stratification"] == \
        "dataset_x_negative_tier"
