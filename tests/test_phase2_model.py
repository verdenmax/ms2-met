import hashlib
import numpy as np
import pandas as pd
import pytest
import torch
from types import SimpleNamespace

from spectrum.dia_data import XIC_DTYPE
from spectrum.psm_info import PSMInfo
from tools.deep_trainer.phase2.data import XICDataset, collate_xic
from tools.deep_trainer.phase2.extraction import extract_signal_sample
from tools.deep_trainer.phase2.model import XICFusionNetwork
from tools.deep_trainer.phase2 import protocol as protocol_module
from tools.deep_trainer.phase2 import experiment as experiment_module
from tools.deep_trainer.phase2.experiment import _fixed_metrics
from tools.deep_trainer.phase2.protocol import FrozenXICProtocol
from tools.deep_trainer.phase2.checkpoint import (
    load_checkpoint, predict_indices, save_checkpoint,
)
from tools.deep_trainer.phase2.schema import ExtractionSettings
from tools.deep_trainer.phase2.store import (
    open_signal_dataset, write_signal_dataset,
)
from tools.deep_trainer.phase2.training import fit_xic_model



class _FakeDia:
    ms1_indexs = np.arange(10, dtype=np.int32)

    @staticmethod
    def find_near_ms1_idx(_rt):
        return 4

    @staticmethod
    def _xic(scale):
        return np.asarray([
            (9.9, -1.0, scale, 3),
            (10.0, 0.5, 4.0 * scale, 4),
            (10.1, np.nan, 0.0, 5),
        ], dtype=XIC_DTYPE)

    def xic_peaks_extreact(self, _rt, _window, mz, _ppm):
        return self._xic(1.0 + float(mz) / 1000.0)

    def xic_peaks_panel_extract(self, _rt, _window, targets, _ppm):
        return self._xic(1.0 + float(np.mean(targets)) / 1000.0)

    @staticmethod
    def check_in_raw(_mz):
        return True

    @staticmethod
    def check_in_same_ms2(_left, _right):
        return True

    def xic_ms2_charge_resolved_extract(
            self, _rt, _window, precursor_mz, ions_mass, mass_tol_ppm,
            fragment_charges):
        scale = 1.0 + float(precursor_mz) / 1000.0 \
            + float(ions_mass) / 10000.0
        return {
            charge: self._xic(scale * charge)
            for charge in fragment_charges
        }, 1000.0


def _psm(**overrides):
    values = {
        "sequence": "AK", "charge": 2, "modify": [],
        "rt": np.float32(10.0), "precursor_mz": np.float32(500.0),
        "raw_title": "raw-a", "protein_names": "HUMAN_P",
        "q_value": 0.005, "label_type": "positive",
    }
    values.update(overrides)
    return PSMInfo(**values)


def _write_training_signals(tmp_path, n_samples=8, name="signals"):
    settings = ExtractionSettings(
        xic_cycle_window=1, mass_tol_ppm=10.0, fragment_charges=(1, 2))
    samples = []
    for index in range(n_samples):
        sample = extract_signal_sample(
            _psm(precursor_mz=np.float32(500.0 + index)),
            _FakeDia(), settings, {
                "sample_id": f"sample-{index}",
                "label": index % 2,
                "fixed_split": "train",
                "outer_fold": index % 2,
            })
        samples.append(sample)
    output = tmp_path / name
    write_signal_dataset(
        samples, output, settings, build_metadata={"mode": "test"},
        shard_size=4)
    return open_signal_dataset(output)


def _model():
    return XICFusionNetwork(
        trace_hidden_dim=4, embedding_dim=3, fragment_hidden_dim=7,
        attention_dim=5, fusion_hidden_dims=[9], dropout=0.0)


def test_xic_fusion_forward_is_finite_and_returns_trust_logits(tmp_path):
    source = _write_training_signals(tmp_path)
    dataset = XICDataset(source, [0, 1])
    batch = collate_xic([dataset[0], dataset[1]])
    model = _model()

    logits, attention = model(batch, return_attention=True)

    assert logits.shape == (2,)
    assert attention.shape == batch["fragment_mask"].shape
    assert torch.isfinite(logits).all()
    assert torch.isfinite(attention).all()
    assert torch.allclose(attention.sum(dim=1), torch.ones(2))
    assert model.architecture()["type"] == "xic_fusion_attention_v1"


def test_xic_attention_handles_samples_with_no_eligible_fragments(tmp_path):
    source = _write_training_signals(tmp_path)
    dataset = XICDataset(source, [0])
    record = dataset[0]
    record["fragment_mask"][:] = False
    batch = collate_xic([record])

    logits, attention = _model()(batch, return_attention=True)

    assert torch.isfinite(logits).all()
    assert attention.sum().item() == 0.0


def test_xic_model_rejects_embedding_indices_outside_contract(tmp_path):
    source = _write_training_signals(tmp_path)
    record = XICDataset(source, [0])[0]
    batch = collate_xic([record])
    batch["fragment_charge"][0, 0] = 99

    with np.testing.assert_raises_regex(ValueError, "charge"):
        _model()(batch)


def test_xic_training_and_dataset_bound_checkpoint_roundtrip(tmp_path):
    source = _write_training_signals(tmp_path)
    config = {
        "model": {
            "type": "xic_fusion_attention_v1",
            "trace_hidden_dim": 4,
            "embedding_dim": 3,
            "fragment_hidden_dim": 7,
            "attention_dim": 5,
            "fusion_hidden_dims": [9],
            "dropout": 0.0,
            "include_predicted_intensity": False,
        },
        "training": {
            "device": "cpu", "deterministic": True,
            "torch_num_threads": 1, "epochs": 2, "batch_size": 2,
            "inference_batch_size": 2, "learning_rate": 0.001,
            "weight_decay": 0.0, "gradient_clip_norm": 5.0,
            "patience": 2, "min_delta": 0.0,
            "class_weighting": "none", "num_workers": 0,
        },
    }
    fitted = fit_xic_model(
        source, [0, 1, 2, 3], [4, 5, 6, 7], config,
        validation_score=lambda labels, trust: float(
            np.mean(trust[np.asarray(labels) == 1])
            - np.mean(trust[np.asarray(labels) == 0])),
        seed=19)
    checkpoint = tmp_path / "model.pt"
    identity = {
        "signal_checksums_sha256": source.complete["checksums_sha256"],
    }
    save_checkpoint(
        checkpoint, fitted, dataset_identity=identity,
        metadata={"metric_semantics": "error_identification_positive_v1"})

    model, payload = load_checkpoint(
        checkpoint, source=source, device="cpu")
    scores = predict_indices(
        checkpoint, source, [0, 3], device="cpu", batch_size=2)

    assert model.architecture() == fitted.model.architecture()
    assert payload["metadata"]["metric_semantics"] \
        == "error_identification_positive_v1"
    assert scores.shape == (2,)
    assert np.isfinite(scores).all()

    other = _write_training_signals(
        tmp_path, n_samples=6, name="other-signals")
    with pytest.raises(ValueError, match="different Phase 2 XIC dataset"):
        load_checkpoint(checkpoint, source=other)


def test_xic_protocol_requires_exact_full_frozen_assignments(
        tmp_path, monkeypatch):
    settings = ExtractionSettings(
        xic_cycle_window=1, mass_tol_ppm=10.0, fragment_charges=(1, 2))
    rows = []
    samples = []
    for index in range(6):
        split = "train" if index < 4 else "test"
        fold = index % 2 if split == "train" else -1
        row = {
            "sample_id": f"protocol-{index}",
            "dataset": "2da", "label": index % 2,
            "negative_tier": "correct" if index % 2 else "t5",
            "fixed_split": split, "outer_fold": fold,
            "leakage_group_id": f"group-{index}",
            "inner_valid_for_fold_0": split == "train" and index == 1,
            "inner_valid_for_fold_1": split == "train" and index == 2,
            "sequence": "AK", "charge": 2,
            "precursor_mz": 500.0 + index, "rt": 10.0 + index,
        }
        rows.append(row)
        samples.append(extract_signal_sample(
            _psm(rt=np.float32(row["rt"]),
                 precursor_mz=np.float32(row["precursor_mz"])),
            _FakeDia(), settings, row.copy()))
    protocol_root = tmp_path / "protocol"
    protocol_root.mkdir()
    summary_path = protocol_root / "summary.json"
    summary_path.write_text('{"schema":"frozen"}\n', encoding="utf-8")
    summary_sha = hashlib.sha256(summary_path.read_bytes()).hexdigest()
    manifest_hashes = {
        "preflight": "a", "membership": "b", "fixed_manifest": "c",
        "fold_map": "d", "predictions": "e",
    }
    output = tmp_path / "full-signals"
    write_signal_dataset(
        samples, output, settings, build_metadata={
            "mode": "full_frozen_protocol",
            "sample_ids_exactly_equal_frozen_protocol": True,
            "frozen_sample_ids_exact_match": True,
            "prediction_included": False,
            "frozen_protocol_contract": {
                "contract": "lightgbm_fixed_negpool_manifest_v1",
                "manifest_sha256": manifest_hashes,
            },
            "frozen_protocol_summary": {"sha256": summary_sha},
        })
    prepared = SimpleNamespace(
        frame=pd.DataFrame(rows), sample_id_col="sample_id",
        target_col="label", dataset_col="dataset",
        tier_col="negative_tier", split_col="fixed_split",
        outer_fold_col="outer_fold", group_col="leakage_group_id",
        base_group_col="sequence",
        inner_valid_cols=[
            "inner_valid_for_fold_0", "inner_valid_for_fold_1"],
        identity_cols=["sequence", "charge", "precursor_mz", "rt"],
        validation={"frozen_protocol": {
            "contract": "lightgbm_fixed_negpool_manifest_v1",
            "manifest_sha256": manifest_hashes,
        }},
        protocol_root=str(protocol_root),
        model_tiers={"M20": ("t5",)}, target_fprs=[0.05, 0.10],
    )
    monkeypatch.setattr(
        protocol_module, "prepare_protocol", lambda *_args: prepared)

    protocol = protocol_module.prepare_xic_protocol(
        str(output), "split.yaml", "features", str(protocol_root))

    assert protocol.validation["sample_ids_exact_match"] is True
    assert protocol.frame["source_index"].tolist() == list(range(6))
    assert len(protocol.training_frame("M20")) == 4
    assert len(protocol.test_frame()) == 2


def test_phase2_fixed_metrics_pin_error_positive_fp_fn_semantics():
    # Stored 1 means actually correct (statistical negative); stored 0 means
    # actually incorrect (statistical positive).
    labels = np.asarray([1, 1, 0, 0])
    values = {
        "trust_score": np.asarray([0.9, 0.8, 0.2, 0.7]),
        "fpr_5_vote_fraction": np.asarray([0.0, 1.0, 1.0, 0.0]),
        "fpr_10_vote_fraction": np.asarray([0.0, 0.0, 1.0, 1.0]),
    }

    metrics = _fixed_metrics(labels, values)
    at_fpr5 = metrics["operating_points"]["fpr_5"][
        "external_ensemble"]["test_metrics"]
    at_fpr10 = metrics["operating_points"]["fpr_10"][
        "external_ensemble"]["test_metrics"]

    assert at_fpr5["fp"] == 1
    assert at_fpr5["fn"] == 1
    assert at_fpr5["fpr"] == 0.5
    assert metrics["fnr_at_fpr5"] == 0.5
    assert at_fpr10["fp"] == 0
    assert at_fpr10["fn"] == 0
    assert metrics["error_recall_at_fpr10"] == 1.0


def test_phase2_experiment_smoke_writes_canonical_frozen_test_bundle(
        tmp_path, monkeypatch):
    source = _write_training_signals(tmp_path, n_samples=18)
    rows = []
    for index in range(18):
        is_train = index < 12
        fold = (index // 2) % 2 if is_train else -1
        rows.append({
            "sample_id": f"sample-{index}",
            "dataset": (
                "2da" if index < 14 else "5da" if index < 16 else "normal"),
            "label": index % 2,
            "negative_tier": "correct" if index % 2 else "t5",
            "fixed_split": "train" if is_train else "test",
            "outer_fold": fold,
            "leakage_group_id": f"group-{index}",
            "sequence": f"PEPTIDE{index}",
            "inner_valid_for_fold_0": is_train and index in {2, 3},
            "inner_valid_for_fold_1": is_train and index in {0, 1},
            "source_index": index,
        })
    protocol_root = tmp_path / "frozen"
    protocol_root.mkdir()
    (protocol_root / "summary.json").write_text("{}\n", encoding="utf-8")
    prepared = SimpleNamespace(
        sample_id_col="sample_id", dataset_col="dataset",
        target_col="label", tier_col="negative_tier",
        split_col="fixed_split", outer_fold_col="outer_fold",
        group_col="leakage_group_id", base_group_col="sequence",
        inner_valid_cols=[
            "inner_valid_for_fold_0", "inner_valid_for_fold_1"],
        model_tiers={"M20": ("t5",)},
        target_fprs=[0.01, 0.05, 0.10],
        protocol_root=str(protocol_root),
    )
    protocol = FrozenXICProtocol(
        source=source, prepared=prepared, frame=pd.DataFrame(rows),
        validation={"build_contract": {
            "signal_checksums_sha256": source.complete[
                "checksums_sha256"],
            "prediction_included": False,
        }})
    monkeypatch.setattr(
        experiment_module, "prepare_xic_protocol",
        lambda *_args, **_kwargs: protocol)

    def baseline(_root, test, _sample_id, _models):
        labels = test["label"].to_numpy(dtype=int)
        return {"LightGBM_M20": {
            "trust_score": np.where(labels == 1, 0.8, 0.2),
            "fpr_5_vote_fraction": 1.0 - labels,
            "fpr_10_vote_fraction": 1.0 - labels,
        }}

    monkeypatch.setattr(
        experiment_module, "load_frozen_lightgbm_predictions", baseline)
    config = {
        "schema": "phase2_xic_training_config_v1",
        "experiment": {"negative_pool_models": ["M20"]},
        "model": {
            "type": "xic_fusion_attention_v1",
            "trace_hidden_dim": 4, "embedding_dim": 3,
            "fragment_hidden_dim": 7, "attention_dim": 5,
            "fusion_hidden_dims": [9], "dropout": 0.0,
            "include_predicted_intensity": False,
        },
        "training": {
            "seeds": [42], "early_stopping_metric": "roc_auc",
            "device": "cpu", "deterministic": True,
            "torch_num_threads": 1, "num_workers": 0,
            "epochs": 1, "batch_size": 2, "inference_batch_size": 2,
            "learning_rate": 0.001, "weight_decay": 0.0,
            "gradient_clip_norm": 5.0, "patience": 1,
            "min_delta": 0.0, "class_weighting": "none",
            "min_class_groups_per_split": 1,
        },
        "comparison": {"bootstrap_reps": 5, "bootstrap_seed": 7},
        "evaluation_semantics": {
            "positive_class": "incorrect_identification",
            "stored_label": (
                "1=correct_identification, 0=incorrect_identification"),
            "model_score": "trust_score=P(correct_identification)",
            "metric_score": "error_score=1-trust_score",
        },
    }
    config_path = tmp_path / "train.yaml"
    import yaml
    config_path.write_text(yaml.safe_dump(config), encoding="utf-8")
    split_config = tmp_path / "split.yaml"
    split_config.write_text("data: {}\n", encoding="utf-8")
    staging = tmp_path / "staging"
    staging.mkdir()
    final = tmp_path / "final"

    summary = experiment_module._run_staging(
        config_path, split_config, "features", protocol_root,
        source.root, staging, final, prepare_only=False)

    table = pd.read_csv(staging / "fixed_test_summary.csv")
    assert summary["metric_semantics"] \
        == "error_identification_positive_v1"
    assert summary["positive_class"] == "incorrect_identification"
    assert set(table["model"]) == {
        "LightGBM_M20", "XIC_M20_seed42", "XIC_M20_ensemble"}
    assert {"roc_auc", "error_pr_auc", "fnr_at_fpr5",
            "error_recall_at_fpr10"} <= set(table.columns)
    assert (staging / "models" / "xic_m20_seed42.fold0.pt").is_file()
    assert (staging / "models" / "xic_m20_seed42.fold1.pt").is_file()
