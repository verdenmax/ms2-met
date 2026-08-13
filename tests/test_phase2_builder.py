from pathlib import Path
from types import SimpleNamespace

import numpy as np
import pandas as pd
import pytest
import yaml

from spectrum.dia_data import XIC_DTYPE
from spectrum.labeling import HeavyType
from spectrum.psm_info import PSMInfo
from tools.deep_trainer.phase2 import builder
from tools.deep_trainer.phase2.store import open_signal_dataset


class _FakeDia:
    ms1_indexs = np.arange(10, dtype=np.int32)

    @staticmethod
    def find_near_ms1_idx(_rt):
        return 4

    @staticmethod
    def _xic(scale):
        return np.asarray([
            (9.9, -1.0, 1.0 * scale, 3),
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
        values = {
            charge: self._xic(float(charge)) for charge in fragment_charges
        }
        return values, 1000.0


def _psm(rt, precursor_mz):
    return PSMInfo(
        sequence="AK", charge=2, modify=[], rt=rt,
        precursor_mz=precursor_mz, raw_title="raw-a",
        protein_names="HUMAN_P", q_value=0.005,
        label_type="positive")


def test_builder_validates_saved_shards_and_preserves_frozen_folds(
        tmp_path, monkeypatch):
    feature_root = tmp_path / "features"
    feature_root.mkdir()
    raw = tmp_path / "raw-a.mzML"
    raw.write_bytes(b"raw")
    psm_path = tmp_path / "psms.json"
    psm_path.write_text("[]\n", encoding="utf-8")
    ini = feature_root / "extract.ini"
    ini.write_text(
        "[input]\n"
        "search_engine_type = 0\n"
        "raw_num = 1\n"
        f"raw_path_1 = {raw}\n"
        f"light_result_file = {psm_path}\n"
        "[general]\n"
        "feature_type = 0\n"
        "xic_cycle_window = 1\n"
        "mass_tol_ppm = 10.0\n"
        "labeling = silac\n",
        encoding="utf-8")
    build_config = tmp_path / "build.yaml"
    build_config.write_text(yaml.safe_dump({
        "schema": "phase2_xic_pilot_config_v1",
        "extraction": {
            "xic_cycle_window": 1,
            "mass_tol_ppm": 10.0,
            "fragment_charges": [1, 2],
        },
        "pilot": {
            "correct_per_dataset": 1,
            "error_per_dataset": 1,
            "seed": 7,
        },
        "storage": {"shard_size": 1},
        "prediction": {"include": False},
        "datasets": {"2da": {"extraction_config": "extract.ini"}},
    }), encoding="utf-8")
    split_config = tmp_path / "split.yaml"
    split_config.write_text("data: {}\n", encoding="utf-8")
    protocol_root = tmp_path / "protocol"
    protocol_root.mkdir()
    (protocol_root / "summary.json").write_text(
        '{"schema":"test_protocol"}\n', encoding="utf-8")

    rows = []
    psms = []
    for index, label in enumerate((1, 0)):
        rt = 10.0 + index
        precursor_mz = 500.0 + index
        psms.append(_psm(rt, precursor_mz))
        rows.append({
            "sample_id": f"sample-{index}", "dataset": "2da",
            "label": label, "negative_tier": (
                "correct" if label else "tier_1"),
            "fixed_split": "train", "outer_fold": index,
            "inner_valid_for_fold_0": index == 0,
            "inner_valid_for_fold_1": index == 1,
            "sequence": "AK", "charge": 2,
            "precursor_mz": precursor_mz, "rt": rt,
            "raw_title1": "raw-a", "label_type": "positive",
        })
    frame = pd.DataFrame(rows)
    protocol = SimpleNamespace(
        frame=frame, dataset_col="dataset", sample_id_col="sample_id",
        group_col="sequence", target_fprs=[0.01, 0.05, 0.10],
        identity_cols=[
            "sequence", "charge", "precursor_mz", "rt", "raw_title1",
            "label_type",
        ],
        validation={
            "frozen_protocol": {
                "contract": "lightgbm_fixed_negpool_manifest_v1",
                "manifest_sha256": {"membership": "test-digest"},
            },
        },
    )
    monkeypatch.setattr(builder, "prepare_protocol", lambda *_: protocol)
    monkeypatch.setattr(
        builder, "_load_psms",
        lambda *_args, **_kwargs: (psms, psm_path, HeavyType.SILAC))
    fake_cache = tmp_path / "cache" / "2da" / "raw-a.dia.npz"
    fake_cache.parent.mkdir(parents=True)
    fake_cache.write_bytes(b"cache")
    monkeypatch.setattr(
        builder, "resolve_dia_cache",
        lambda *_args, **_kwargs: (fake_cache, {
            "dataset": "2da", "kind": "dia_cache",
            "cache": {"path": str(fake_cache), "sha256": "test"},
        }))
    monkeypatch.setattr(
        builder, "resolve_mmap_dia_cache",
        lambda *_args, **_kwargs: (tmp_path / "mmap", {
            "kind": "dia_mmap_cache", "path": str(tmp_path / "mmap"),
            "manifest_sha256": "test-mmap",
        }))
    monkeypatch.setattr(
        builder, "load_mmap_dia_cache", lambda *_args, **_kwargs: _FakeDia())

    parity_calls = []

    def compare_serialized(sample, _row, _settings):
        # Arrays loaded from saved .npy shards are read-only mmap views. This
        # fails if the builder regresses to validating in-memory samples.
        assert sample.precursor_intensity.flags.writeable is False
        parity_calls.append(str(sample.metadata["sample_id"]))
        return [{
            "sample_id": str(sample.metadata["sample_id"]),
            "feature": "test_feature", "expected": 1.0,
            "reconstructed": 1.0, "absolute_difference": 0.0,
            "passed": True,
        }]

    monkeypatch.setattr(builder, "compare_to_feature_row", compare_serialized)
    output = tmp_path / "signals"
    report = builder.build_signal_dataset(
        str(build_config), str(split_config), str(feature_root),
        str(protocol_root), str(output), cache_root=str(tmp_path / "cache"))

    assert sorted(parity_calls) == ["sample-0", "sample-1"]
    assert report["staged_validation"] == {
        "serialized_shards_validated": True,
        "frozen_membership_exact_match": True,
        "required_feature_parity_all_passed": True,
        "feature_parity_comparisons": 2,
        "required_feature_parity_comparisons": 2,
        "legacy_isotope_audit_comparisons": 0,
        "legacy_isotope_audit_mismatches": 0,
    }
    dataset = open_signal_dataset(output)
    restored = {
        str(dataset.sample(index).metadata["sample_id"]): dataset.sample(index)
        for index in range(len(dataset))
    }
    assert restored["sample-0"].metadata["inner_valid_for_fold_0"] is True
    assert restored["sample-1"].metadata["inner_valid_for_fold_1"] is True
    assert (output / "audit" / "feature_parity.csv").is_file()
    assert dataset.schema["build"]["frozen_protocol_contract"][
        "manifest_sha256"]["membership"] == "test-digest"


def test_feature_snapshot_isotope_model_preflight_rejects_unknown():
    frame = pd.DataFrame({
        "isotope_model": [
            "ideal_full_label_exact_mass_v2", "mistyped_model",
        ],
    })

    with pytest.raises(ValueError, match="unknown isotope_model"):
        builder._feature_snapshot_isotope_models(frame)


def test_feature_snapshot_isotope_model_preflight_allows_known_migration():
    assert builder._feature_snapshot_isotope_models(pd.DataFrame({
        "isotope_model": [
            "ideal_full_label_exact_mass_v2", "ideal_full_label_v1", None,
        ],
    })) == [
        "ideal_full_label_exact_mass_v2", "ideal_full_label_v1",
        "undeclared_legacy",
    ]


def test_full_selection_keeps_every_frozen_row_in_stable_order():
    frame = pd.DataFrame({
        "sample_id": ["z", "a", "m"],
        "dataset": ["normal", "2da", "5da"],
        "label": [1, 0, 1],
    })
    protocol = SimpleNamespace(
        frame=frame, sample_id_col="sample_id", dataset_col="dataset")

    selected, contract = builder._select_rows(protocol, {
        "schema": "phase2_xic_dataset_config_v2",
        "selection": {"mode": "full"},
    })

    assert selected["sample_id"].tolist() == ["a", "m", "z"]
    assert contract == {
        "mode": "full_frozen_protocol",
        "sample_ids_exactly_equal_frozen_protocol": True,
    }


def test_extractor_implementation_contract_content_binds_tensor_modules():
    contract = builder._extractor_implementation_contract()

    assert contract["schema"] == "phase2_extractor_implementation_v1"
    assert set(builder._EXTRACTOR_IMPLEMENTATION_FILES) <= set(
        contract["files_sha256"])
    assert {
        "spectrum/psm_info.py", "workflows/q1a_helpers.py",
        "workflows/pred_store.py", "workflows/modified_psm_policy.py",
    } <= set(contract["files_sha256"])
    assert all(
        len(value) == 64 for value in contract["files_sha256"].values())
