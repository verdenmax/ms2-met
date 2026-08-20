from copy import deepcopy
import hashlib
import logging
import numpy as np
import os
import pandas as pd
from pathlib import Path
import pytest
import shutil
import torch
from types import SimpleNamespace
import yaml

from spectrum.dia_data import XIC_DTYPE
from spectrum.psm_info import PSMInfo
from tools.deep_trainer.phase2.data import (
    XICDataset, _normalize_record, collate_xic,
)
from tools.deep_trainer.phase2.extraction import extract_signal_sample
from tools.deep_trainer.phase2.model import (
    ResidualTraceEncoder, XICFusionNetwork, XICPairInteractionNetwork,
    model_from_architecture, model_implementation_sha256,
    n_trainable_parameters,
)
from tools.deep_trainer.phase2 import model as xic_model_module
from tools.deep_trainer.phase2 import protocol as protocol_module
from tools.deep_trainer.phase2 import experiment as experiment_module
from tools.deep_trainer.phase2.experiment import _fixed_metrics
from tools.deep_trainer.phase2.protocol import FrozenXICProtocol
from tools.deep_trainer.phase2.checkpoint import (
    load_checkpoint, predict_indices, save_checkpoint,
)
from tools.deep_trainer.phase2 import checkpoint as checkpoint_module
from tools.deep_trainer.phase2.schema import ExtractionSettings
from tools.deep_trainer.phase2.store import (
    open_signal_dataset, write_signal_dataset,
)
from tools.deep_trainer.phase2.training import (
    attention_head_diagnostics, fit_xic_model,
)
from tools.deep_trainer.training import configure_torch



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
    def check_in_same_ms2(_left, _right, rt=None):
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


def _strong_model():
    return XICPairInteractionNetwork(
        trace_hidden_dim=4, trace_blocks=2, precursor_hidden_dim=8,
        embedding_dim=3, fragment_hidden_dim=7, set_blocks=2,
        attention_dim=5, attention_heads=2, fusion_hidden_dims=[9],
        dropout=0.0)


def test_xic_adapter_normalization_has_exact_five_field_semantics():
    precursor_intensity = np.zeros((4, 3), dtype="f4")
    precursor_ppm = np.zeros((4, 3), dtype="f4")
    precursor_rt = np.zeros((4, 3), dtype="f4")
    precursor_scan = np.zeros((4, 3), dtype=bool)
    precursor_peak = np.zeros((4, 3), dtype=bool)
    precursor_intensity[0] = [0.0, 999.0, 9999.0]
    precursor_ppm[0] = [0.0, -3.0, 5.0]
    precursor_rt[0] = [0.0, 0.4, 1.2]
    precursor_scan[0] = [False, True, True]
    precursor_peak[0] = [False, True, True]
    # A real scan with no matched peak must differ from padding.
    precursor_rt[1, 1] = 0.4
    precursor_scan[1, 1] = True

    fragment_intensity = np.zeros((2, 2, 3), dtype="f4")
    fragment_rt = np.zeros((2, 2, 3), dtype="f4")
    fragment_scan = np.zeros((2, 2, 3), dtype=bool)
    fragment_peak = np.zeros((2, 2, 3), dtype=bool)
    fragment_ppm = np.zeros((2, 2, 3), dtype="f4")
    fragment_intensity[0, 0, 1] = 500.0
    fragment_rt[0, 0, 1] = -0.6
    fragment_scan[0, 0, 1] = True
    fragment_peak[0, 0, 1] = True
    # This ineligible fragment must not alter either sample-level scale.
    fragment_intensity[1, 0, 1] = 1e6
    fragment_rt[1, 0, 1] = 100.0
    fragment_scan[1, 0, 1] = True
    fragment_peak[1, 0, 1] = True
    record = {
        "precursor_intensity": precursor_intensity,
        "precursor_ppm_error": precursor_ppm,
        "precursor_rt_delta": precursor_rt,
        "precursor_scan_mask": precursor_scan,
        "precursor_peak_mask": precursor_peak,
        "fragment_intensity": fragment_intensity,
        "fragment_ppm_error": fragment_ppm,
        "fragment_rt_delta": fragment_rt,
        "fragment_scan_mask": fragment_scan,
        "fragment_peak_mask": fragment_peak,
        "fragment_attempted": np.array([True, False]),
        "fragment_separable": np.array([True, False]),
        "fragment_ion_type": np.array([0, 1]),
        "fragment_charge": np.array([1, 2]),
        "metadata": {"sample_id": "exact-normalization", "label": 1},
    }

    normalized = _normalize_record(
        record, mass_tol_ppm=10.0,
        include_predicted_intensity=False)

    expected_intensity = np.log1p(999.0) / np.log1p(9999.0)
    np.testing.assert_allclose(
        normalized["precursor"][0],
        [0.0, expected_intensity, 1.0], rtol=1e-6)
    np.testing.assert_allclose(
        normalized["precursor"][1], [0.0, -0.3, 0.5])
    np.testing.assert_allclose(
        normalized["precursor"][2], [0.0, 1.0 / 3.0, 1.0],
        rtol=1e-6)
    assert normalized["precursor"][3].tolist() == [0.0, 1.0, 1.0]
    assert normalized["precursor"][4].tolist() == [0.0, 1.0, 1.0]
    assert normalized["precursor"][5:10, 1].tolist() == [
        0.0, 0.0, pytest.approx(1.0 / 3.0), 1.0, 0.0,
    ]
    np.testing.assert_allclose(normalized["signal_scale"], [
        np.log1p(9999.0), np.log1p(1.2),
    ], rtol=1e-6)


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
    assert attention[:, :2].sum().item() == 0.0
    assert model.architecture()["type"] == "xic_fusion_attention_v2"


def test_xic_attention_handles_samples_with_no_eligible_fragments(tmp_path):
    source = _write_training_signals(tmp_path)
    dataset = XICDataset(source, [0])
    record = dataset[0]
    record["fragment_mask"][:] = False
    batch = collate_xic([record])

    logits, attention = _model()(batch, return_attention=True)

    assert torch.isfinite(logits).all()
    assert attention.sum().item() == 0.0


def test_xic_v3_pair_interaction_forward_preserves_model_interface(tmp_path):
    source = _write_training_signals(tmp_path)
    dataset = XICDataset(source, [0, 1])
    batch = collate_xic([dataset[0], dataset[1]])
    model = _strong_model()

    logits, attention = model(batch, return_attention=True)

    assert logits.shape == (2,)
    assert attention.shape == batch["fragment_mask"].shape
    assert torch.isfinite(logits).all()
    assert torch.isfinite(attention).all()
    assert torch.allclose(attention.sum(dim=1), torch.ones(2))
    assert attention[:, :2].sum().item() == 0.0
    architecture = model.architecture()
    assert architecture["type"] == "xic_pair_interaction_v3"
    restored = model_from_architecture(architecture)
    assert restored.architecture() == architecture
    assert n_trainable_parameters(model) > n_trainable_parameters(_model())
    _, attention_heads = model(batch, return_attention_heads=True)
    assert attention_heads.shape == (
        len(batch["label"]), model.attention_heads,
        batch["fragment"].shape[1],
    )
    torch.testing.assert_close(
        attention_heads.sum(dim=-1),
        torch.ones_like(attention_heads.sum(dim=-1)))


def test_residual_trace_encoder_preserves_raw_intensity_summaries():
    encoder = ResidualTraceEncoder(
        hidden_dim=4, n_blocks=1, dropout=0.0).eval()
    values = torch.zeros(2, 5, 4)
    values[0, 0] = torch.tensor([0.0, 0.2, 0.8, 1.0])
    scan_mask = torch.tensor([
        [False, True, True, False],
        [False, False, False, False],
    ])

    encoded = encoder(values, scan_mask)

    torch.testing.assert_close(
        encoded[:, -2:], torch.tensor([[0.5, 0.8], [0.0, 0.0]]))


@pytest.mark.skipif(not torch.cuda.is_available(), reason="CUDA unavailable")
def test_xic_v3_cuda_deterministic_forward_backward_smoke(tmp_path,
                                                          monkeypatch):
    monkeypatch.delenv("CUBLAS_WORKSPACE_CONFIG", raising=False)
    configure_torch(31, deterministic=True)
    source = _write_training_signals(tmp_path)
    batch = collate_xic([XICDataset(source, [0])[0]])
    cuda_batch = {
        key: value.cuda() if torch.is_tensor(value) else value
        for key, value in batch.items()
    }
    model = _strong_model().cuda().train()

    loss = torch.nn.functional.binary_cross_entropy_with_logits(
        model(cuda_batch), cuda_batch["label"])
    loss.backward()

    assert os.environ["CUBLAS_WORKSPACE_CONFIG"] == ":4096:8"
    assert torch.isfinite(loss)
    assert all(
        parameter.grad is None or torch.isfinite(parameter.grad).all()
        for parameter in model.parameters())


def test_xic_v3_all_masked_fragment_set_is_finite_and_invariant(tmp_path):
    source = _write_training_signals(tmp_path)
    record = XICDataset(source, [0])[0]
    record["fragment_mask"][:] = False
    first = collate_xic([record])
    second = deepcopy(first)
    second["fragment"][:] = 1e6
    second["fragment_ion_type"][:] = 2
    second["fragment_charge"][:] = 8
    model = _strong_model().eval()

    with torch.no_grad():
        first_logit, first_attention = model(first, return_attention=True)
        second_logit, second_attention = model(
            second, return_attention=True)

    assert torch.isfinite(first_logit).all()
    assert first_attention.sum().item() == 0.0
    assert second_attention.sum().item() == 0.0
    torch.testing.assert_close(
        first_logit, second_logit, rtol=0.0, atol=1e-7)


def test_xic_v3_reports_attention_head_collapse_diagnostics(tmp_path):
    source = _write_training_signals(tmp_path)
    dataset = XICDataset(source, [0, 1])

    diagnostics = attention_head_diagnostics(
        _strong_model(), dataset, batch_size=2, device="cpu")

    assert diagnostics["n_heads"] == 2
    assert diagnostics["n_samples_with_multiple_fragments"] == 2
    assert 0.0 <= diagnostics["mean_pairwise_cosine_similarity"] <= 1.0
    assert 0.0 <= diagnostics["fraction_pairwise_cosine_ge_0_99"] <= 1.0
    assert diagnostics["mean_effective_fragments_per_head"] >= 1.0


def test_masked_fragment_magnitude_cannot_change_scale_or_model_logit(tmp_path):
    settings = ExtractionSettings(
        xic_cycle_window=1, mass_tol_ppm=10.0, fragment_charges=(1, 2))
    first = extract_signal_sample(
        _psm(), _FakeDia(), settings,
        {"sample_id": "masked-original", "label": 1})
    second = deepcopy(first)
    second.metadata["sample_id"] = "masked-amplified"
    masked = ~(second.fragment_attempted & second.fragment_separable)
    assert masked.any()
    second.fragment_intensity[masked] *= 10000.0
    second.fragment_rt_delta[masked] *= 10000.0
    output = tmp_path / "masked-signals"
    write_signal_dataset(
        [first, second], output, settings, build_metadata={"mode": "test"})
    dataset = XICDataset(open_signal_dataset(output))
    original, amplified = dataset[0], dataset[1]

    assert np.array_equal(
        original["signal_scale"], amplified["signal_scale"])
    assert np.array_equal(
        original["precursor"], amplified["precursor"])
    assert np.array_equal(
        original["fragment"][original["fragment_mask"]],
        amplified["fragment"][amplified["fragment_mask"]])
    model = _model().eval()
    with torch.no_grad():
        original_logit = model(collate_xic([original]))[0]
        amplified_logit = model(collate_xic([amplified]))[0]
    torch.testing.assert_close(
        original_logit, amplified_logit, rtol=0.0, atol=1e-7)


def test_xic_model_rejects_embedding_indices_outside_contract(tmp_path):
    source = _write_training_signals(tmp_path)
    record = XICDataset(source, [0])[0]
    batch = collate_xic([record])
    batch["fragment_charge"][0, 0] = 99

    with np.testing.assert_raises_regex(ValueError, "charge"):
        _model()(batch)


def test_xic_training_and_dataset_bound_checkpoint_roundtrip(
        tmp_path, caplog):
    caplog.set_level(logging.INFO)
    source = _write_training_signals(tmp_path)
    config = {
        "model": {
            "type": "xic_fusion_attention_v2",
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
    assert payload["model_implementation"]["model_type"] \
        == "xic_fusion_attention_v2"
    assert payload["runtime_device"] == fitted.device_trace
    assert fitted.device_trace["device_type"] == "cpu"
    assert sum(
        "training will run without GPU acceleration" in record.message
        for record in caplog.records) == 1
    assert scores.shape == (2,)
    assert np.isfinite(scores).all()

    other = _write_training_signals(
        tmp_path, n_samples=6, name="other-signals")
    with pytest.raises(ValueError, match="different Phase 2 XIC dataset"):
        load_checkpoint(checkpoint, source=other)

    monkeypatch = pytest.MonkeyPatch()
    original = checkpoint_module.input_adapter_contract
    monkeypatch.setattr(
        checkpoint_module, "input_adapter_contract",
        lambda **kwargs: {**original(**kwargs), "schema": "future_adapter"})
    with pytest.raises(ValueError, match="different Phase 2 input adapter"):
        load_checkpoint(checkpoint, source=source)
    monkeypatch.undo()

    monkeypatch = pytest.MonkeyPatch()
    original_model = checkpoint_module.model_implementation_contract
    monkeypatch.setattr(
        checkpoint_module, "model_implementation_contract",
        lambda model_type: {
            **original_model(model_type), "sha256": "future-model",
        })
    with pytest.raises(ValueError, match="different Phase 2 model"):
        load_checkpoint(checkpoint, source=source)
    monkeypatch.undo()


def test_xic_v3_training_and_checkpoint_roundtrip(tmp_path):
    source = _write_training_signals(tmp_path)
    config = {
        "model": _strong_model().architecture(),
        "training": {
            "device": "cpu", "deterministic": True,
            "torch_num_threads": 1, "epochs": 1, "batch_size": 2,
            "inference_batch_size": 2, "learning_rate": 0.001,
            "weight_decay": 0.0, "gradient_clip_norm": 5.0,
            "patience": 1, "min_delta": 0.0,
            "class_weighting": "none", "num_workers": 0,
        },
    }
    fitted = fit_xic_model(
        source, [0, 1, 2, 3], [4, 5, 6, 7], config,
        validation_score=lambda labels, trust: float(
            np.mean(trust[np.asarray(labels) == 1])
            - np.mean(trust[np.asarray(labels) == 0])),
        seed=23)
    checkpoint = tmp_path / "model-v3.pt"
    save_checkpoint(checkpoint, fitted, dataset_identity={
        "signal_checksums_sha256": source.complete["checksums_sha256"],
    }, metadata={
        "metric_semantics": "error_identification_positive_v1",
    })

    restored, payload = load_checkpoint(
        checkpoint, source=source, device="cpu")
    scores = predict_indices(
        checkpoint, source, [0, 3], device="cpu", batch_size=2)

    assert isinstance(restored, XICPairInteractionNetwork)
    assert payload["architecture"] == fitted.model.architecture()
    assert scores.shape == (2,)
    assert np.isfinite(scores).all()


def test_published_v2_checkpoint_implementation_remains_allowlisted(tmp_path):
    source = _write_training_signals(tmp_path)
    checkpoint = tmp_path / "published-v2.pt"
    model = _model()
    with torch.no_grad():
        for index, parameter in enumerate(model.parameters()):
            parameter.fill_((index + 1) * 0.001)
    fitted = SimpleNamespace(
        model=model, history=[], best_epoch=1,
        best_validation_score=0.5,
        device_trace={
            "device": "cpu", "device_type": "cpu",
            "cuda_available": False, "torch_cuda_runtime": None,
        })
    save_checkpoint(checkpoint, fitted, dataset_identity={
        "signal_checksums_sha256": source.complete["checksums_sha256"],
    }, metadata={
        "metric_semantics": "error_identification_positive_v1",
    })
    payload = torch.load(checkpoint, map_location="cpu", weights_only=True)
    payload["model_implementation"] = {
        "schema": "phase2_xic_model_implementation_v1",
        "sha256": (
            "ce4814ddead969e87da4b3996a19474bb5b85d1a405575d441f649b7bb197aca"
        ),
    }
    torch.save(payload, checkpoint)

    restored, _ = load_checkpoint(
        checkpoint, source=source, device="cpu")
    trust = predict_indices(
        checkpoint, source, [0, 3], device="cpu", batch_size=2)

    assert isinstance(restored, XICFusionNetwork)
    np.testing.assert_allclose(
        trust, [0.5065788626670837, 0.5065788626670837],
        rtol=0.0, atol=1e-8)


def test_model_implementation_hash_is_isolated_by_architecture(monkeypatch):
    before = model_implementation_sha256("xic_fusion_attention_v2")
    v3_factory, _v3_parts = xic_model_module._MODEL_REGISTRY[
        "xic_pair_interaction_v3"]
    monkeypatch.setitem(
        xic_model_module._MODEL_REGISTRY,
        "xic_pair_interaction_v3", (v3_factory, (v3_factory,)))

    after = model_implementation_sha256("xic_fusion_attention_v2")

    assert after == before


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


def test_phase2_preflight_rejects_model_adapter_channel_mismatch(tmp_path):
    source = _write_training_signals(tmp_path, n_samples=2)
    protocol = SimpleNamespace(source=source)
    config = {
        "model": {
            "precursor_channels": 19,
            "fragment_channels": 10,
            "include_predicted_intensity": False,
        },
    }

    with pytest.raises(ValueError, match="differ from the XIC adapter"):
        experiment_module._validate_model_input_shape(protocol, config)


def test_strong_xic_config_is_a_separate_supported_model_arm():
    path = Path(__file__).parents[1] / (
        "tools/deep_trainer/phase2/config/xic_pair_interaction.yaml")
    config = yaml.safe_load(path.read_text(encoding="utf-8"))

    experiment_module._validate_config(config)
    model = model_from_architecture(config["model"])

    assert isinstance(model, XICPairInteractionNetwork)
    assert model.architecture() == config["model"]
    assert 150_000 <= n_trainable_parameters(model) <= 250_000
    assert config["training"]["device"] == "cuda"
    assert config["training"]["class_weighting"] == "none"

    invalid = deepcopy(config)
    invalid["model"]["fragment_channels"] = 9
    with pytest.raises(ValueError, match="paired five-feature fragment"):
        experiment_module._validate_config(invalid)


def test_phase2_preflight_requires_all_reported_working_points():
    protocol = SimpleNamespace(prepared=SimpleNamespace(
        target_fprs=[0.01, 0.05]))

    with pytest.raises(ValueError, match="lacks required FPR working points"):
        experiment_module._validate_required_working_points(protocol)

    protocol.prepared.target_fprs = [0.05, 0.10]
    contract = experiment_module._validate_required_working_points(protocol)
    assert contract["required_target_fprs"] == [0.05, 0.10]


def test_phase2_preflight_accepts_make_generated_working_points():
    from tools.spec_trainer.gen_cv_configs import to_cv_config

    project_root = Path(__file__).resolve().parents[1]
    source = yaml.safe_load((
        project_root / "tools" / "spec_trainer" / "config"
        / "in_2da_neg20.yaml").read_text(encoding="utf-8"))
    generated = to_cv_config(source, "in_2da_neg20")
    protocol = SimpleNamespace(prepared=SimpleNamespace(
        target_fprs=generated["operating_point"]["target_fprs"]))

    contract = experiment_module._validate_required_working_points(protocol)
    assert contract["available_target_fprs"] == [0.05, 0.10]


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
            "type": "xic_fusion_attention_v2",
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
    fixed_predictions = pd.read_csv(
        staging / "predictions" / "fixed_test_predictions.csv")
    oof_predictions = pd.read_csv(
        staging / "predictions" / "xic_m20_seed42_train_oof.csv")
    assert summary["metric_semantics"] \
        == "error_identification_positive_v1"
    assert summary["positive_class"] == "incorrect_identification"
    assert set(table["model"]) == {
        "LightGBM_M20", "XIC_M20_seed42", "XIC_M20_ensemble"}
    assert {"roc_auc", "error_pr_auc", "fnr_at_fpr5",
            "error_recall_at_fpr10"} <= set(table.columns)
    for predictions in (fixed_predictions, oof_predictions):
        assert set(predictions["metric_semantics"]) == {
            "error_identification_positive_v1"}
        assert set(predictions["positive_class"]) == {
            "incorrect_identification"}
    assert (staging / "models" / "xic_m20_seed42.fold0.pt").is_file()
    assert (staging / "models" / "xic_m20_seed42.fold1.pt").is_file()
    assert (staging / "COMPLETE").is_file()
    experiment_module._verify_complete_bundle(staging)
    logging.info("logging after result finalization must not mutate train.log")
    experiment_module._verify_complete_bundle(staging)

    published = tmp_path / "published"
    backup = tmp_path / ".published.backup.interrupted"
    shutil.copytree(staging, backup)
    experiment_module._recover_publish(published, cleanup_stale=False)
    assert published.is_dir() and not backup.exists()
    experiment_module._verify_complete_bundle(published)

    legacy = tmp_path / "legacy-result"
    legacy.mkdir()
    (legacy / "old-summary.json").write_text("{}\n", encoding="utf-8")
    experiment_module._recover_publish(legacy, cleanup_stale=True)
    assert (legacy / "old-summary.json").is_file()
    with pytest.raises(ValueError, match="incomplete Phase 2 result"):
        experiment_module._recover_publish(legacy, cleanup_stale=False)

    with (staging / "fixed_test_summary.csv").open("a", encoding="utf-8") \
            as handle:
        handle.write("tampered\n")
    with pytest.raises(ValueError, match="artifact changed"):
        experiment_module._verify_complete_bundle(staging)
