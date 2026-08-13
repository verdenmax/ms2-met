import numpy as np
import torch

from spectrum.dia_data import XIC_DTYPE
from spectrum.psm_info import PSMInfo
from tools.deep_trainer.phase2.data import XICDataset, collate_xic
from tools.deep_trainer.phase2.extraction import extract_signal_sample
from tools.deep_trainer.phase2.model import XICFusionNetwork
from tools.deep_trainer.phase2.schema import ExtractionSettings
from tools.deep_trainer.phase2.store import (
    open_signal_dataset, write_signal_dataset,
)



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


def _write_training_signals(tmp_path, n_samples=8):
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
    output = tmp_path / "signals"
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
