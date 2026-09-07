"""Value-level regressions for the September sample-generation audit."""
import json
import sys
from pathlib import Path

import numpy as np
import pandas as pd
import pytest

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "tools/spec_trainer/src"))

from cohort import apply_training_cohort
from cv_core import make_cv_splits
from cv_train import _inner_split, assemble_oof
from feature_cols import resolve_feature_cols, validate_synthetic_features
from sample_groups import prepare_cv_groups, validate_cv_groups
from spectrum.psm_info import PSMInfo
from tests.test_counterfactual_negatives import _all_source_config, _parent, _target
from tools.counterfactual_negatives import build_counterfactual_negatives
from tools.training_set_builder import (
    QueryBuildConfig, _check_heldout_disjoint, _deduplicate_training, _join_silver_manifest,
    generate_queries,
)
from workflows.flow_utils import _make_result_row_single


REQUIRED_MANIFEST = (
    "query_id", "parent_id", "sequence", "charge",
    "generator", "negative_source", "labeling",
)


def _build(parents):
    return build_counterfactual_negatives(
        parents, _target("|".join(p._sequence for p in parents)),
        _all_source_config(seed=11, composition_shuffle_per_parent=2,
                           kr_position_shuffle_per_parent=2,
                           local_denovo_per_parent=0))


def test_counterfactual_li_dedup_preserves_independent_raw_observations():
    result = _build([_parent("GAILLLLK", raw=raw) for raw in ("raw1", "raw2")])
    for _, part in result.manifest.groupby("raw_title"):
        assert len(part) == 4
        assert part.sequence.str.replace("I", "L").is_unique
    assert set(result.manifest.raw_title) == {"raw1", "raw2"}
    assert result.audit["failures"]["synthetic_kr_position_shuffle:duplicate"] > 0
    for psm in result.psms:
        row = _make_result_row_single(PSMInfo.from_dict(psm.to_dict()), {})
        assert row["negative_source"] == (
            "gold_positive" if row["label"] == 1 else psm._negative_source)
        assert row["peptide_group_id"] == psm._peptide_group_id


def test_external_query_generator_deduplicates_li_equivalents(tmp_path, monkeypatch):
    fasta = tmp_path / "target.fasta"
    fasta.write_text(">p\nGAILLLLK\n")
    positives = tmp_path / "parents.csv"
    pd.DataFrame([{"sequence": "GAILLLLK", "charge": 2, "label": 1,
                   "heavy_confirmed": 1}]).to_csv(positives, index=False)
    proposals = iter(["GLLLLAIK", "GLLLIALK", "AGILLLLK"])
    monkeypatch.setattr("tools.training_set_builder._cleavage_preserving_shuffle",
                        lambda *args, **kwargs: next(proposals))
    manifest = tmp_path / "queries.tsv"
    generate_queries(QueryBuildConfig(
        positives=str(positives), target_fasta=str(fasta),
        output_manifest=str(manifest), output_fasta=str(tmp_path / "queries.fasta"),
        shuffle_per_parent=2, markov_per_parent=0))
    rows = pd.read_csv(manifest, sep="\t")
    assert rows.sequence.tolist() == ["GLLLLAIK", "AGILLLLK"]


def _families():
    rows = []
    for i in range(12):
        for kind, label in (("parent", 1), ("child", 0)):
            rows.append({"sequence": f"{kind}{i}", "label": label,
                         "parent_id": f"P{i}", "group_id": f"P{i}",
                         "peptide_group_id": f"PG{i}",
                         "query_id": f"Q{i}" if label == 0 else None,
                         "negative_source": ("synthetic_composition_shuffle"
                                             if label == 0 else "gold_positive"),
                         "all_p75": float(i)})
    return pd.DataFrame(rows)


def test_cv_enforces_families_in_outer_and_inner_splits():
    frame = _families()
    with pytest.raises(ValueError, match="family"):
        validate_cv_groups(frame, frame.sequence)
    group_col, audit = prepare_cv_groups(frame, "sequence")
    assert audit["candidate_family_leakage_protected"]
    groups = frame[group_col]
    for train, test in make_cv_splits(frame.label, groups, n_folds=3):
        assert set(frame.iloc[train].parent_id).isdisjoint(frame.iloc[test].parent_id)
        fit, valid = _inner_split(frame[["all_p75"]], frame.label, groups,
                                  train, valid_size=0.25, seed=42)
        assert set(frame.iloc[fit].parent_id).isdisjoint(frame.iloc[valid].parent_id)
    # The pre-fit API guard runs even without LightGBM installed.
    with pytest.raises(ValueError, match="family"):
        assemble_oof(frame, frame[["all_p75"]], frame.label, frame.sequence,
                     {}, ["all_p75"], "unused")


def test_cv_connects_charge_li_and_shared_child_sequences():
    frame = pd.DataFrame({
        "sequence": ["PEPTIDEK", "PEPTLDEK", "WRONGAAK", "WRONGAAK"],
        "charge": [2, 3, 2, 2],
        "group_id": ["P1", "P2", "P2", "P3"],
        "peptide_group_id": ["PG1", "PG1", "PG1", "PG3"],
    })
    group, _ = prepare_cv_groups(frame, "peptide_group_id")
    assert frame[group].nunique() == 1


@pytest.mark.parametrize("column", ["parent_id", "peptide_group_id"])
def test_cv_rejects_missing_candidate_family_metadata(column):
    with pytest.raises(ValueError, match=column):
        prepare_cv_groups(_families().drop(columns=column), "sequence")


def _eligibility_rows():
    frame = _families()
    for column, value in (("heavy_in_raw", 1), ("heavy_out_of_range", 0),
                          ("precursor_xic_empty", 0), ("q1a_valid", 1),
                          ("isotope_model_valid", 1)):
        frame[column] = value
    frame["has_lib_pred"] = frame.label
    frame["spec_pattern_SA"] = np.where(frame.label.eq(1), 0.9, np.nan)
    return frame


def test_observed_training_keeps_unpredicted_children_and_audits_sources(tmp_path):
    frame = _eligibility_rows()
    kept, audit = apply_training_cohort(frame, "evidence_observed")
    assert len(kept) == 24
    source = audit["by_source"]["synthetic_composition_shuffle"]
    assert source["prediction_coverage"] == {"n_with_prediction": 0, "n_rows": 12}
    assert source["n_dropped"] == 0
    validate_synthetic_features(kept, ["all_p75"])
    with pytest.raises(ValueError, match="evidence_observed"):
        apply_training_cohort(frame, "evidence_common")
    with pytest.raises(ValueError, match="coverage"):
        validate_synthetic_features(frame, ["spec_pattern_SA"])
    path = tmp_path / "features.csv"
    frame.to_csv(path, index=False)
    with pytest.raises(ValueError, match="has_lib_pred"):
        resolve_feature_cols(None, [str(path)], "label")
    with pytest.raises(ValueError, match="coverage"):
        resolve_feature_cols(["spec_pattern_SA"], [str(path)], "label")
    assert resolve_feature_cols(["all_p75"], [str(path)], "label") == ["all_p75"]
    frame["has_lib_pred"] = 1
    validate_synthetic_features(frame, ["spec_pattern_SA"])
    assert len(apply_training_cohort(frame, "evidence_common")[0]) == 24


@pytest.mark.parametrize("column,value", [("rt", 20.0), ("precursor_mz", 999.0),
                                         ("raw_title1", "different_raw")])
def test_counterfactual_join_rejects_stale_observation(column, value):
    result = _build([_parent()])
    features = pd.DataFrame([_make_result_row_single(p, {}) for p in result.psms])
    features.loc[features.label.eq(0), column] = value
    with pytest.raises(ValueError, match="mismatch"):
        _join_silver_manifest(features, result.manifest, REQUIRED_MANIFEST)


def test_same_query_id_with_updated_parent_rt_rejects_old_features():
    parent = _parent()
    old = _build([parent])
    parent._rt = 20.0
    new = _build([parent])
    assert old.manifest.query_id.tolist() == new.manifest.query_id.tolist()
    features = pd.DataFrame([_make_result_row_single(p, {}) for p in old.psms])
    with pytest.raises(ValueError, match="rt mismatch"):
        _join_silver_manifest(features, new.manifest, REQUIRED_MANIFEST)


def test_counterfactual_join_accepts_float32_rounding_and_raw_aliases():
    result = _build([_parent()])
    manifest = result.manifest.copy()
    manifest["rt"] += 0.0000002
    manifest["raw_title"] = "/raws/raw1.pfb"
    features = pd.DataFrame([_make_result_row_single(p, {}) for p in result.psms])
    joined, keys = _join_silver_manifest(features, manifest, REQUIRED_MANIFEST)
    assert keys == ["query_id"]
    assert len(joined) == len(result.manifest)
    assert not any(column.endswith("_manifest") for column in joined
                   if column in ("rt_manifest", "precursor_mz_manifest"))


@pytest.mark.parametrize("frame", [
    pd.DataFrame({"sequence": ["PEPTIDEK"]}),
    pd.DataFrame({"raw_title1": [None]}),
    pd.DataFrame({"raw_title1": ["  "]}),
])
@pytest.mark.parametrize("side", ["train", "heldout"])
def test_heldout_check_rejects_incomplete_raw_provenance(frame, side):
    good = pd.DataFrame({"raw_title1": ["valid_raw"]})
    with pytest.raises(ValueError, match="raw-title"):
        _check_heldout_disjoint([frame if side == "train" else good],
                                frame if side == "heldout" else good, True)


def test_heldout_normalizes_raw_paths_and_reports_li_peptide_overlap():
    train = pd.DataFrame({"raw_title": ["/data/run.pfb"], "sequence": ["PEPTIDEK"]})
    heldout = pd.DataFrame({"Run": ["run"], "sequence": ["PEPTLDEK"]})
    with pytest.raises(ValueError, match="raw leakage"):
        _check_heldout_disjoint([train], heldout, True)
    heldout["Run"] = "other"
    report = _check_heldout_disjoint([train], heldout, True)
    assert report["sequence_overlap"]["n_overlapping_sequences"] == 1
    assert "domain holdout" in report["sequence_overlap"]["interpretation"]


def test_raw_aliases_coalesce_rows_but_reject_conflicting_values():
    train = pd.DataFrame({"raw_title": ["a", None], "Run": [None, "b"]})
    heldout = pd.DataFrame({"raw_title1": ["c"]})
    assert _check_heldout_disjoint([train], heldout, True)["n_train_raws"] == 2
    train.loc[0, "Run"] = "conflict"
    with pytest.raises(ValueError, match="conflicting"):
        _check_heldout_disjoint([train], heldout, True)


def test_cv_main_persists_connected_family_folds(tmp_path):
    pytest.importorskip("lightgbm")
    import cv_train
    import yaml
    from tests.test_cv_train import _toy_cfg

    frame = _families()
    path = tmp_path / "features.csv"
    frame.to_csv(path, index=False)
    cfg = _toy_cfg(tmp_path)
    cfg["data"].update(train_files=[str(path)], feature_cols=["all_p75"])
    cfg["training"].update(cv_folds=3, num_boost_round=5, early_stopping_rounds=2)
    config = tmp_path / "train.yaml"
    config.write_text(yaml.safe_dump(cfg))
    cv_train.main(["--config", str(config), "--name", "family-regression",
                   "--logpath", str(tmp_path / "train.log")])
    output = json.loads((tmp_path / "r.cv.json").read_text())
    assert output["experiment"]["split_groups"]["candidate_family_leakage_protected"]
    oof = pd.read_csv(output["artifacts"]["oof_predictions_csv"])
    assert oof.groupby("peptide_group_id").oof_fold.nunique().eq(1).all()
    assert "negative_source" in oof


def test_mixed_raw_aliases_do_not_collapse_independent_observations():
    frame = pd.DataFrame({
        "sequence": ["PEPTIDEK"] * 3, "charge": [2] * 3,
        "label": [1] * 3, "negative_source": ["gold_positive"] * 3,
        "raw_title1": ["a", None, None], "Run": [None, "b", "c"],
    })
    result = _deduplicate_training(frame)
    assert len(result) == 3
    assert set(result.raw_title1) == {"a", "b", "c"}


@pytest.mark.parametrize("missing", [None, "", "  "])
def test_partial_missing_child_query_id_is_rejected(missing):
    result = _build([_parent()])
    features = pd.DataFrame([_make_result_row_single(p, {}) for p in result.psms])
    features.loc[1, "query_id"] = missing
    features.loc[1, "rt"] = 999
    with pytest.raises(ValueError, match="missing query_id"):
        _join_silver_manifest(features, result.manifest, REQUIRED_MANIFEST)


def test_query_ids_are_normalized_before_join():
    result = _build([_parent()])
    features = pd.DataFrame([_make_result_row_single(p, {}) for p in result.psms])
    features.loc[features.label.eq(0), "query_id"] += "  "
    joined, _ = _join_silver_manifest(features, result.manifest, REQUIRED_MANIFEST)
    assert len(joined) == len(result.manifest)


def test_group_connection_survives_filtered_bridge_and_regrouping():
    frame = pd.DataFrame({
        "sequence": ["PEPTIDEK", "PEPTIDEK", "CHILDAK", "CHILDBK"],
        "charge": [2, 3, 2, 3],
        "query_id": [None, None, "Q1", "Q2"],
        "parent_id": ["P1", "P2", "P1", "P2"],
        "negative_source": ["gold_positive", "gold_positive",
                            "silver_synthetic_shuffle", "silver_synthetic_shuffle"],
    })
    column, _ = prepare_cv_groups(frame, "sequence")
    retained = frame.drop(index=1).copy()
    # Mirrors fixed-negpool's combined-domain regrouping after cohort filtering.
    regrouped, _ = prepare_cv_groups(retained, "sequence")
    assert retained[regrouped].nunique() == 1
    validate_cv_groups(retained, retained[column])


def test_legacy_training_entry_rejects_synthetic_row_splits(tmp_path):
    pytest.importorskip("lightgbm")
    from tools.spec_trainer.src.main import load_data
    path = tmp_path / "features.csv"
    _families().to_csv(path, index=False)
    with pytest.raises(ValueError, match="cv_train.py"):
        load_data([str(path)], ["all_p75"], "label")
