import copy
import json
from pathlib import Path

import numpy as np
import pandas as pd
import pytest

torch = pytest.importorskip("torch")

from tools.deep_trainer.checkpoint import (
    load_checkpoint, load_predictor, save_checkpoint,
)
from tools.deep_trainer.experiment import _validate_deep_config
from tools.deep_trainer.experiment import run_experiment
from tools.deep_trainer.model import TabularMLP, n_trainable_parameters
from tools.deep_trainer.missingness_sensitivity import _assert_matched
from tools.deep_trainer.preprocessing import FoldPreprocessor
from tools.deep_trainer.spec_adapter import _assert_columns_equal
from tools.deep_trainer.training import fit_mlp, predict_trust


def test_fold_preprocessor_fits_train_only_and_preserves_missingness():
    train = np.array([
        [1.0, np.nan, 5.0],
        [3.0, 2.0, 5.0],
        [5.0, 4.0, 5.0],
    ])
    external = np.array([[1000.0, np.nan, 5.0]])
    fitted = FoldPreprocessor.fit(train, add_missing_indicators=True)

    assert fitted.medians.tolist() == [3.0, 3.0, 5.0]
    assert fitted.n_output_features == 6
    transformed = fitted.transform(external)
    assert transformed.shape == (1, 6)
    assert transformed[0, 4] == 1.0
    # External 1000 must not alter a train-fitted mean of 3.
    assert transformed[0, 0] > 100.0


def test_fold_preprocessor_all_missing_and_constant_columns_are_finite():
    fitted = FoldPreprocessor.fit(np.array([
        [np.nan, 7.0], [np.nan, 7.0],
    ]))
    transformed = fitted.transform(np.array([[np.nan, 7.0]]))
    assert np.isfinite(transformed).all()
    assert transformed.tolist() == [[0.0, 0.0, 1.0, 0.0]]


def test_frozen_identity_allows_lossless_float_text_roundtrip():
    current = pd.DataFrame({
        "sample_id": ["a", "b", "c"],
        "rt": ["123.400000", "10.123456789012345", None],
        "sequence": ["PEPTIDE", "OTHER", "THIRD"],
    })
    frozen = pd.DataFrame({
        "sample_id": ["c", "a", "b"],
        "rt": [np.nan, 123.4, 10.123456789012344],
        "sequence": ["THIRD", "PEPTIDE", "OTHER"],
    })
    _assert_columns_equal(
        current, frozen, ["rt", "sequence"], "membership")


@pytest.mark.parametrize("column,value", [
    ("rt", 123.5),
    ("sequence", "DIFFERENT"),
])
def test_frozen_identity_still_rejects_real_value_changes(column, value):
    current = pd.DataFrame({
        "sample_id": ["a"], "rt": [123.4], "sequence": ["PEPTIDE"],
    })
    frozen = current.copy()
    frozen.loc[0, column] = value
    with pytest.raises(ValueError, match=rf"column '{column}'"):
        _assert_columns_equal(
            current, frozen, ["rt", "sequence"], "membership")


def test_missingness_sensitivity_requires_matched_runtime_and_rows():
    columns = {
        "sample_id": ["b", "a"],
        "dataset": ["2da", "2da"],
        "sequence": ["B", "A"],
        "charge": [2, 2],
        "precursor_mz": [500.0, 400.0],
        "rt": [20.0, 10.0],
        "raw_title1": ["r2", "r1"],
        "label_type": ["heavy", "heavy"],
        "label": [0, 1],
        "negative_tier": ["E20", "correct"],
        "__source_row": [1, 0],
    }
    left = pd.DataFrame(columns)
    right = left.iloc[::-1].reset_index(drop=True)
    provenance = {
        "python": "3.13", "torch": "2.13", "numpy": "2.3",
        "pandas": "2.3", "git_commit": "abc",
    }
    identity = _assert_matched(
        left, right, {"provenance": provenance},
        {"provenance": provenance})
    assert identity["sample_id"].tolist() == ["a", "b"]

    changed = {"provenance": {**provenance, "torch": "2.10"}}
    with pytest.raises(ValueError, match="provenance torch"):
        _assert_matched(
            left, right, {"provenance": provenance}, changed)


def test_tabular_mlp_outputs_one_correct_identification_logit_per_row():
    model = TabularMLP(8, hidden_dims=[6, 3], dropout=0.0)
    logits = model(torch.zeros((5, 8)))
    assert logits.shape == (5,)
    assert model.architecture()["type"] == "tabular_mlp_v1"
    assert n_trainable_parameters(model) > 0


def _toy_training_config():
    return {
        "model": {"hidden_dims": [12, 6], "dropout": 0.0},
        "training": {
            "device": "cpu",
            "deterministic": True,
            "torch_num_threads": 1,
            "epochs": 20,
            "batch_size": 32,
            "learning_rate": 0.01,
            "weight_decay": 0.0,
            "gradient_clip_norm": 5.0,
            "patience": 5,
            "min_delta": 0.0,
            "class_weighting": "none",
        },
    }


def test_fit_mlp_returns_finite_trust_scores_without_label_inversion():
    rng = np.random.default_rng(3)
    train_x = rng.normal(size=(160, 4)).astype("f4")
    train_y = (train_x[:, 0] > 0).astype(int)
    valid_x = rng.normal(size=(80, 4)).astype("f4")
    valid_y = (valid_x[:, 0] > 0).astype(int)

    def accuracy(labels, trust):
        return np.mean((trust >= 0.5) == labels)

    fitted = fit_mlp(
        train_x, train_y, valid_x, valid_y, _toy_training_config(),
        validation_score=accuracy, seed=7)
    trust = predict_trust(fitted.model, valid_x, device="cpu")
    assert np.isfinite(trust).all()
    assert ((trust >= 0.5) == valid_y).mean() > 0.80
    assert fitted.best_epoch >= 1


def test_checkpoint_roundtrip_includes_preprocessor_and_semantics(tmp_path):
    rng = np.random.default_rng(4)
    raw = rng.normal(size=(120, 3))
    labels = (raw[:, 0] > 0).astype(int)
    preprocessor = FoldPreprocessor.fit(raw)
    values = preprocessor.transform(raw)
    config = _toy_training_config()
    fitted = fit_mlp(
        values[:80], labels[:80], values[80:], labels[80:], config,
        validation_score=lambda y, p: np.mean((p >= 0.5) == y), seed=9)
    path = tmp_path / "model.pt"
    save_checkpoint(
        path, fitted, preprocessor, ["a", "b", "c"], {
            "metric_semantics": "error_identification_positive_v1",
            "positive_class": "incorrect_identification",
        })

    model, loaded_preprocessor, payload = load_checkpoint(path)
    before = predict_trust(fitted.model, values, device="cpu")
    after_values = loaded_preprocessor.transform(raw)
    after = predict_trust(model, after_values, device="cpu")
    assert np.allclose(before, after)
    assert payload["feature_names"] == ["a", "b", "c"]
    assert payload["metadata"]["positive_class"] == "incorrect_identification"

    predictor = load_predictor(path)
    shuffled = pd.DataFrame(raw, columns=["a", "b", "c"])[
        ["c", "a", "b"]]
    assert np.allclose(before, predictor.predict_frame(shuffled))
    with pytest.raises(ValueError, match="missing checkpoint features"):
        predictor.predict_frame(shuffled.drop(columns="b"))


def test_deep_config_pins_project_evaluation_semantics():
    import yaml
    from pathlib import Path

    config = yaml.safe_load(Path(
        "tools/deep_trainer/config/tabular_mlp.yaml").read_text())
    assert config["evaluation_semantics"] == {
        "positive_class": "incorrect_identification",
        "stored_label": "1=correct_identification, 0=incorrect_identification",
        "model_score": "trust_score=P(correct_identification)",
        "metric_score": "error_score=1-trust_score",
    }
    assert config["training"]["class_weighting"] == "none"
    assert config["training"]["early_stopping_metric"] == "roc_auc"
    _validate_deep_config(config)


@pytest.mark.parametrize(
    "mutation", ["semantics", "model", "metric", "weighting"])
def test_deep_config_rejects_protocol_drift(mutation):
    import copy
    import yaml
    from pathlib import Path

    config = yaml.safe_load(Path(
        "tools/deep_trainer/config/tabular_mlp.yaml").read_text())
    config = copy.deepcopy(config)
    if mutation == "semantics":
        config["evaluation_semantics"]["positive_class"] = \
            "correct_identification"
    elif mutation == "model":
        config["model"]["type"] = "unknown"
    elif mutation == "metric":
        config["training"]["early_stopping_metric"] = "error_pr_auc"
    else:
        config["training"]["class_weighting"] = "balanced"
    with pytest.raises(ValueError):
        _validate_deep_config(config)


def test_result_payload_semantics_are_json_safe():
    payload = {
        "metric_semantics": "error_identification_positive_v1",
        "positive_class": "incorrect_identification",
        "model_score": "trust_score=P(correct_identification)",
    }
    assert json.loads(json.dumps(payload)) == payload


def _deep_toy_config():
    return {
        "experiment": {"negative_pool_models": ["M20"]},
        "protocol": {"label_source": "synthetic test contract"},
        "preprocessing": {"add_missing_indicators": True},
        "model": {
            "type": "tabular_mlp_v1", "hidden_dims": [8], "dropout": 0.0,
        },
        "training": {
            "seeds": [7], "early_stopping_metric": "roc_auc",
            "device": "cpu", "deterministic": True,
            "torch_num_threads": 1, "epochs": 2, "batch_size": 64,
            "inference_batch_size": 128, "learning_rate": 0.01,
            "weight_decay": 0.0, "gradient_clip_norm": 5.0,
            "patience": 2, "min_delta": 0.0,
            "class_weighting": "none", "min_class_groups_per_split": 1,
        },
        "comparators": {"logistic_regression": {
            "enabled": True, "C": 1.0, "max_iter": 100,
            "solver": "lbfgs",
        }},
        "comparison": {"bootstrap_reps": 5, "bootstrap_seed": 11},
        "evaluation_semantics": {
            "positive_class": "incorrect_identification",
            "stored_label": (
                "1=correct_identification, 0=incorrect_identification"),
            "model_score": "trust_score=P(correct_identification)",
            "metric_score": "error_score=1-trust_score",
        },
    }


def _build_frozen_protocol(tmp_path):
    """Build a tiny completed LightGBM-shaped bundle without fitting LGBM."""
    import sys

    spec_src = Path("tools/spec_trainer/src").resolve()
    if str(spec_src) not in sys.path:
        sys.path.insert(0, str(spec_src))
    import fixed_negpool
    from feature_groups import EVIDENCE_CORE_FEATURES

    rng = np.random.default_rng(27)
    feature_root = tmp_path / "features"
    rows = []
    strata = [
        ("correct", 1, 50), ("t5", 0, 20),
        ("t5_10", 0, 20), ("t10_20", 0, 20),
    ]
    for tier, label, count in strata:
        for index in range(count):
            row = {
                "sequence": f"{tier}_PEPTIDE_{index}",
                "charge": 2,
                "precursor_mz": 400.0 + len(rows) * 0.01,
                "rt": 10.0 + len(rows) * 0.01,
                "raw_title1": "toy_raw",
                "label_type": "positive" if label else "negative",
                "label": label,
                "heavy_in_raw": 1,
                "heavy_out_of_range": 0,
                "precursor_xic_empty": 0,
                "q1a_valid": 1,
                "has_lib_pred": 1,
                "isotope_model_valid": 1,
                "labeling": "silac",
                "isotope_model": "silac_residue_shift_v1",
                "group_id": f"family_{tier}_{index}",
                "parent_id": f"family_{tier}_{index}",
                "_tier": tier,
            }
            for offset, feature in enumerate(sorted(EVIDENCE_CORE_FEATURES)):
                value = rng.normal(label + offset * 0.001, 0.25)
                if index % 13 == 0 and offset % 11 == 0:
                    value = np.nan
                row[feature] = value
            rows.append(row)
    source = pd.DataFrame(rows)
    included = {
        "neg05": {"correct", "t5"},
        "neg10": {"correct", "t5", "t5_10"},
        "neg20": {"correct", "t5", "t5_10", "t10_20"},
    }
    for pool, tiers in included.items():
        frame = source[source["_tier"].isin(tiers)].drop(columns="_tier")
        path = feature_root / f"baseline_2da_{pool}" / "features.csv"
        path.parent.mkdir(parents=True)
        frame.to_csv(path, index=False)

    split_config = {
        "data": {
            "feature_cols": [], "target_col": "label",
            "feature_arm": "evidence_core", "drop_features": [],
            "cohort": "evidence_common", "group_col": "sequence",
            "require_complete_arm": True,
        },
        "model": {"type": "lightgbm", "params": {
            "objective": "binary", "metric": ["auc", "binary_logloss"],
        }},
        "training": {
            "cv_folds": 2, "cv_seed": 42, "valid_size": 0.25,
            "min_class_groups_per_split": 1,
        },
        "operating_point": {
            "target_fprs": [0.05, 0.10], "primary_target_fpr": 0.10,
        },
    }
    split_path = tmp_path / "split.yaml"
    import yaml
    split_path.write_text(yaml.safe_dump(split_config), encoding="utf-8")
    prepared = fixed_negpool.prepare_fixed_negpool(
        fixed_negpool.feature_paths(feature_root, "2da"), split_config,
        min_test_errors_per_tier=1, split_candidates=32)

    protocol_root = tmp_path / "frozen"
    fixed_negpool._write_prepared(prepared, protocol_root)
    test = prepared.frame[prepared.frame["fixed_split"].eq("test")]
    predictions = pd.DataFrame({
        "sample_id": test["sample_id"].to_numpy(),
        "M20_trust_score": np.where(test["label"].eq(1), 0.85, 0.15),
        "M20_error_score": np.where(test["label"].eq(1), 0.15, 0.85),
        "M20_fpr5_error_vote_fraction": np.where(
            test["label"].eq(1), 0.0, 1.0),
        "M20_fpr10_error_vote_fraction": np.where(
            test["label"].eq(1), 0.0, 1.0),
    })
    prediction_path = protocol_root / "predictions" / \
        "fixed_test_predictions.csv"
    prediction_path.parent.mkdir(parents=True)
    predictions.to_csv(prediction_path, index=False)
    summary = {
        "metric_semantics": "error_identification_positive_v1",
        "positive_class": "incorrect_identification",
        "dataset": "2da",
        "design": {
            "cohort": "evidence_common", "feature_arm": "evidence_core",
            "split_group_col": prepared.split_group_col,
        },
        "provenance": {"inputs": [
            fixed_negpool._file_fingerprint(path)
            for path in fixed_negpool.feature_paths(
                feature_root, "2da").values()
        ]},
        "frozen_bundle": {
            "schema": "fixed_negpool_frozen_bundle_v2", "complete": True,
            "feature_cols": list(prepared.feature_cols),
            "feature_cols_sha256": fixed_negpool._feature_schema_sha256(
                prepared.feature_cols),
            "artifact_sha256": fixed_negpool._frozen_bundle_hashes(
                protocol_root),
        },
        "models": {
            "M20": {"feature_cols": list(prepared.feature_cols)},
        },
    }
    (protocol_root / "summary.json").write_text(
        json.dumps(summary), encoding="utf-8")
    deep_config_path = tmp_path / "deep.yaml"
    deep_config_path.write_text(
        yaml.safe_dump(_deep_toy_config()), encoding="utf-8")
    return feature_root, split_path, deep_config_path, protocol_root


def test_run_experiment_consumes_frozen_protocol_end_to_end(tmp_path):
    feature_root, split_path, deep_path, protocol_root = \
        _build_frozen_protocol(tmp_path)
    output = tmp_path / "deep-result"
    summary = run_experiment(
        deep_path, split_path, feature_root, "2da", protocol_root, output)

    assert set(summary["models"]) == {
        "LightGBM_M20", "MLP_M20_seed7", "Logistic_M20_seed7"}
    point = summary["models"]["MLP_M20_seed7"][
        "fixed_test_metrics"]["operating_points"]["fpr_5"]
    assert "external_ensemble" in point
    assert "test_metrics" not in point
    assert point["external_ensemble"]["test_metrics"][
        "positive_class"] == "incorrect_identification"
    assert summary["split_contract"]["membership"] == \
        "loaded_from_frozen_LightGBM_manifest"
    assert summary["generalization_audit"][
        "candidate_family_leakage_protected"] is True
    assert (output / "paired_model_bootstrap.csv").is_file()
    bootstrap = pd.read_csv(output / "paired_model_bootstrap.csv")
    assert set(bootstrap["metric_semantics"]) == {
        "error_identification_positive_v1"}
    assert set(bootstrap["positive_class"]) == {
        "incorrect_identification"}
    assert (output / "missingness_audit.csv").is_file()
    status = json.loads((output / "bundle_status.json").read_text())
    assert status["status"] == "complete"


def test_prepare_protocol_rejects_modified_feature_or_fold_map(tmp_path):
    from tools.deep_trainer.spec_adapter import prepare_protocol

    feature_root, split_path, _, protocol_root = _build_frozen_protocol(
        tmp_path)
    fold_map = protocol_root / "manifests" / "fold_map.csv"
    fold_map.write_text(fold_map.read_text() + "\n", encoding="utf-8")
    with pytest.raises(ValueError, match="were modified"):
        prepare_protocol(split_path, feature_root, "2da", protocol_root)

    # Restore a fresh frozen bundle, then mutate one feature input.
    other = tmp_path / "other"
    feature_root, split_path, _, protocol_root = _build_frozen_protocol(other)
    feature = feature_root / "baseline_2da_neg20" / "features.csv"
    feature.write_text(feature.read_text() + "\n", encoding="utf-8")
    with pytest.raises(ValueError, match="feature content differs"):
        prepare_protocol(split_path, feature_root, "2da", protocol_root)


def test_prepare_protocol_rejects_feature_resolution_drift(tmp_path):
    from tools.deep_trainer.spec_adapter import prepare_protocol
    import yaml

    feature_root, split_path, _, protocol_root = _build_frozen_protocol(
        tmp_path)
    split = yaml.safe_load(split_path.read_text(encoding="utf-8"))
    split["data"]["drop_features"] = ["precursor_pearson"]
    split_path.write_text(yaml.safe_dump(split), encoding="utf-8")
    with pytest.raises(ValueError, match="feature schema differs"):
        prepare_protocol(split_path, feature_root, "2da", protocol_root)


def test_failed_overwrite_preserves_previous_bundle(tmp_path):
    feature_root, split_path, deep_path, protocol_root = \
        _build_frozen_protocol(tmp_path)
    output = tmp_path / "existing"
    output.mkdir()
    (output / "summary.json").write_text('{"old": true}', encoding="utf-8")
    (output / "keep.txt").write_text("previous bundle", encoding="utf-8")
    import yaml
    broken = copy.deepcopy(_deep_toy_config())
    broken["training"]["seeds"] = []
    broken_path = tmp_path / "broken.yaml"
    broken_path.write_text(yaml.safe_dump(broken), encoding="utf-8")

    with pytest.raises(ValueError, match="training.seeds"):
        run_experiment(
            broken_path, split_path, feature_root, "2da", protocol_root,
            output, overwrite=True)
    assert json.loads((output / "summary.json").read_text()) == {"old": True}
    assert (output / "keep.txt").read_text() == "previous bundle"


def test_make_deep_target_passes_frozen_protocol_root(tmp_path):
    import subprocess

    feature_root = tmp_path / "features"
    for dataset in ("2da", "5da", "normal"):
        for pool in ("neg05", "neg10", "neg20"):
            path = feature_root / f"baseline_{dataset}_{pool}" / "features.csv"
            path.parent.mkdir(parents=True, exist_ok=True)
            path.touch()
    protocol = tmp_path / "protocol"
    protocol.mkdir()
    (protocol / "summary.json").write_text("{}", encoding="utf-8")
    output = tmp_path / "deep-output"
    result = subprocess.run([
        "make", "-n", "train-deep-mlp-combined",
        f"FEATURE_ROOT={feature_root}", f"DEEP_PROTOCOL_ROOT={protocol}",
        f"DEEP_OUTPUT_ROOT={output}",
    ], capture_output=True, text=True, cwd=Path(__file__).resolve().parents[1])
    assert result.returncode == 0, result.stderr
    assert f'--protocol-root "{protocol}"' in result.stdout
    assert f'--output-root "{output}/tabular-mlp/combined"' in result.stdout
