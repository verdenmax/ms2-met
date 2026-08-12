import json

import numpy as np
import pytest

torch = pytest.importorskip("torch")

from tools.deep_trainer.checkpoint import load_checkpoint, save_checkpoint
from tools.deep_trainer.experiment import _validate_deep_config
from tools.deep_trainer.model import TabularMLP, n_trainable_parameters
from tools.deep_trainer.preprocessing import FoldPreprocessor
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
