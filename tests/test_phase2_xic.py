import json
import shutil

import numpy as np
import pandas as pd
import pytest

from spectrum.dia_data import XIC_DTYPE
from spectrum.psm_info import PSMInfo
from tools.deep_trainer.phase2.extraction import extract_signal_sample
from tools.deep_trainer.phase2.builder import _source_row_metadata
from tools.deep_trainer.phase2.data import (
    ShardBatchSampler, XICDataset, collate_xic,
)
from tools.deep_trainer.phase2.matching import (
    match_psms_to_protocol, select_pilot_rows,
)
from tools.deep_trainer.phase2.parity import (
    compare_to_feature_row, reconstruct_legacy_features,
)
from tools.deep_trainer.phase2.schema import (
    FRAGMENT_STATUS_TO_CODE, ExtractionSettings,
)
from tools.deep_trainer.phase2.store import (
    StagedValidation, open_signal_dataset, recover_interrupted_publish,
    write_signal_dataset,
)


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
        assert mass_tol_ppm > 0
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


def _sample(sample_id="sample-a"):
    settings = ExtractionSettings(
        xic_cycle_window=1, mass_tol_ppm=10.0, fragment_charges=(1, 2))
    sample = extract_signal_sample(
        _psm(), _FakeDia(), settings, {
            "sample_id": sample_id, "dataset": "2da", "label": 1,
            "negative_tier": "correct", "fixed_split": "train",
            "outer_fold": 0, "sequence": "AK", "charge": 2,
            "precursor_mz": 500.0, "rt": 10.0,
            "raw_title1": "raw-a", "label_type": "positive",
        })
    return sample, settings


def test_extract_signal_sample_preserves_masks_charge_and_skip_status():
    sample, settings = _sample()
    assert sample.precursor_intensity.shape == (4, settings.trace_length)
    assert sample.precursor_scan_mask.all()
    assert sample.precursor_peak_mask[:, -1].tolist() == [False] * 4
    # AK has b1 (no K/R label, co-isolated) and y1 (K label); each has z1/z2.
    assert sample.fragment_intensity.shape == (4, 2, settings.trace_length)
    assert set(sample.fragment_charge.tolist()) == {1, 2}
    assert np.count_nonzero(
        sample.fragment_status
        == FRAGMENT_STATUS_TO_CODE["coisolated_same_mz"]) == 2
    assert np.count_nonzero(
        sample.fragment_status == FRAGMENT_STATUS_TO_CODE["valid"]) == 2
    assert sample.fragment_attempted.tolist() == [False, False, True, True]
    assert not sample.fragment_prediction_present.any()
    assert np.isnan(sample.fragment_predicted_intensity).all()


def test_fragment_peak_status_is_charge_resolved():
    class ChargeTwoMissingDia(_FakeDia):
        def xic_ms2_charge_resolved_extract(
                self, _rt, _window, precursor_mz, ions_mass, mass_tol_ppm,
                fragment_charges):
            values, total = super().xic_ms2_charge_resolved_extract(
                _rt, _window, precursor_mz, ions_mass, mass_tol_ppm,
                fragment_charges)
            values[2] = values[2].copy()
            values[2]["intensity"] = 0.0
            values[2]["ppm_error"] = np.nan
            return values, total

    sample, settings = _sample()
    sample = extract_signal_sample(
        _psm(), ChargeTwoMissingDia(), settings, sample.metadata)
    attempted = np.flatnonzero(sample.fragment_attempted)
    statuses = {
        int(sample.fragment_charge[index]): int(sample.fragment_status[index])
        for index in attempted
    }
    assert statuses[1] == FRAGMENT_STATUS_TO_CODE["valid"]
    assert statuses[2] == FRAGMENT_STATUS_TO_CODE[
        "no_light_or_heavy_peak"]


def test_source_metadata_keeps_frozen_inner_fold_assignments():
    row = pd.Series({
        "sample_id": "s", "dataset": "2da", "label": 1,
        "sequence": "AK", "charge": 2, "precursor_mz": 500.0,
        "rt": 10.0, "raw_title1": "raw-a", "label_type": "positive",
        "inner_valid_for_fold_0": True,
        "inner_valid_for_fold_1": False,
    })
    metadata = _source_row_metadata(row, _psm())
    assert metadata["inner_valid_for_fold_0"] is True
    assert metadata["inner_valid_for_fold_1"] is False


def test_reconstructed_features_compare_value_by_value():
    sample, settings = _sample()
    reconstructed = reconstruct_legacy_features(sample, settings)
    rows = compare_to_feature_row(
        sample, pd.Series(reconstructed), settings)
    assert rows
    assert all(row["passed"] for row in rows)
    assert reconstructed["valid_fragment_ions_num"] == 1
    assert reconstructed["fragment_same_mass_count"] == 1


def test_legacy_isotope_parity_is_audited_but_not_publish_blocking():
    sample, settings = _sample()
    reconstructed = reconstruct_legacy_features(sample, settings)
    legacy = dict(reconstructed)
    legacy["isotope_model"] = "ideal_full_label_v1"
    legacy["isotope_correlation"] += 0.2

    rows = compare_to_feature_row(sample, pd.Series(legacy), settings)
    by_feature = {row["feature"]: row for row in rows}

    isotope = by_feature["isotope_correlation"]
    assert isotope["passed"] is False
    assert isotope["required_for_publish"] is False
    assert isotope["parity_policy"] == "legacy_isotope_model_audit_only"
    assert by_feature["precursor_pearson"]["required_for_publish"] is True


def test_current_isotope_model_parity_remains_publish_blocking():
    sample, settings = _sample()
    reconstructed = reconstruct_legacy_features(sample, settings)
    current = dict(reconstructed)
    current["isotope_model"] = "ideal_full_label_exact_mass_v2"
    current["isotope_correlation"] += 0.2

    rows = compare_to_feature_row(sample, pd.Series(current), settings)
    isotope = next(
        row for row in rows if row["feature"] == "isotope_correlation")

    assert isotope["passed"] is False
    assert isotope["required_for_publish"] is True
    assert isotope["parity_policy"] == "required"


def test_unknown_isotope_model_cannot_bypass_publish_parity():
    sample, settings = _sample()
    reconstructed = reconstruct_legacy_features(sample, settings)
    unknown = dict(reconstructed)
    unknown["isotope_model"] = "mistyped_or_future_model"
    unknown["isotope_correlation"] += 0.2

    rows = compare_to_feature_row(sample, pd.Series(unknown), settings)
    isotope = next(
        row for row in rows if row["feature"] == "isotope_correlation")

    assert isotope["passed"] is False
    assert isotope["required_for_publish"] is True
    assert isotope["parity_policy"] == "required"


def test_sharded_store_roundtrip_and_checksum_validation(tmp_path):
    first, settings = _sample("sample-a")
    second, _ = _sample("sample-b")
    second.metadata["label"] = 0
    output = tmp_path / "signals"
    report = write_signal_dataset(
        [first, second], output, settings,
        build_metadata={"mode": "test"}, shard_size=1,
        audit_tables={"parity.csv": pd.DataFrame([{"passed": True}])},
        audit_documents={"chemistry.json": {"labeling": "silac"}},
    )
    assert report["n_samples"] == 2
    assert (output / "COMPLETE").is_file()
    assert (output / "audit" / "parity.csv").is_file()
    dataset = open_signal_dataset(output, verify_checksums=True)
    assert len(dataset) == 2
    restored = dataset[1]
    assert restored["metadata"]["sample_id"] == "sample-b"
    assert np.array_equal(
        restored["precursor_intensity"], second.precursor_intensity)
    assert np.array_equal(
        restored["fragment_status"], second.fragment_status)
    schema = json.loads((output / "schema.json").read_text())
    assert schema["positive_class"] == "incorrect_identification"
    assert schema["schema"] == "phase2_raw_xic_v2"
    assert schema["isotope_model"] == "ideal_full_label_exact_mass_v2"


def test_xic_torch_adapter_uses_bounded_signal_inputs_and_ragged_padding(
        tmp_path):
    first, settings = _sample("torch-a")
    second, _ = _sample("torch-b")
    # Exercise ragged padding and an all-ineligible fragment collection.
    for name in (
        "fragment_intensity", "fragment_ppm_error", "fragment_rt_delta",
        "fragment_scan_mask", "fragment_peak_mask",
    ):
        setattr(second, name, getattr(second, name)[:2])
    for name in (
        "fragment_ion_type", "fragment_ordinal", "fragment_charge",
        "fragment_light_mz", "fragment_heavy_mz",
        "fragment_predicted_intensity", "fragment_prediction_present",
        "fragment_separable", "fragment_attempted", "fragment_status",
    ):
        setattr(second, name, getattr(second, name)[:2])
    second.fragment_scan_mask[:] = False
    second.fragment_peak_mask[:] = False
    output = tmp_path / "signals"
    write_signal_dataset(
        [first, second], output, settings,
        build_metadata={"mode": "test"}, shard_size=2)

    source = open_signal_dataset(output)
    dataset = XICDataset(source)
    batch = collate_xic([dataset[0], dataset[1]])

    assert batch["precursor"].shape == (2, 20, settings.trace_length)
    assert batch["fragment"].shape == (2, 4, 10, settings.trace_length)
    assert batch["fragment_mask"][1].sum().item() == 0
    assert batch["fragment_mask"][0].tolist() == [False, False, True, True]
    assert batch["fragment_ion_type"][0, -1].item() in {1, 2}
    assert batch["fragment_ion_type"][1, -1].item() == 0
    assert batch["label"].tolist() == [1.0, 1.0]
    assert batch["sample_id"] == ["torch-a", "torch-b"]
    assert np.isfinite(batch["precursor"].numpy()).all()
    assert batch["precursor"].abs().max().item() <= 1.0
    assert "fragment_status" not in batch
    assert "fragment_ordinal" not in batch
    assert "negative_tier" not in batch


def test_xic_prediction_arm_is_explicit_and_preserves_missingness(tmp_path):
    sample, settings = _sample("prediction-a")
    sample.fragment_prediction_present[0] = True
    sample.fragment_predicted_intensity[0] = 0.75
    output = tmp_path / "signals"
    write_signal_dataset(
        [sample], output, settings, build_metadata={"mode": "test"})
    source = open_signal_dataset(output)

    assert "fragment_prediction" not in XICDataset(source)[0]
    record = XICDataset(source, include_predicted_intensity=True)[0]
    assert record["fragment_prediction"][0].tolist() == [0.75, 1.0]
    assert record["fragment_prediction"][1].tolist() == [0.0, 0.0]


def test_shard_batch_sampler_is_complete_deterministic_and_shard_local(
        tmp_path):
    samples = [_sample(f"shard-{index}")[0] for index in range(5)]
    settings = _sample()[1]
    output = tmp_path / "signals"
    write_signal_dataset(
        samples, output, settings, build_metadata={"mode": "test"},
        shard_size=2)
    dataset = XICDataset(open_signal_dataset(output))
    sampler = ShardBatchSampler(dataset, 2, seed=17)

    first = list(iter(sampler))
    second = list(iter(sampler))
    assert first == second
    assert sorted(index for batch in first for index in batch) == list(range(5))
    batch_shards = []
    for batch in first:
        shards = {
            dataset.source.manifest.iloc[int(dataset.indices[index])]["shard"]
            for index in batch
        }
        assert len(shards) == 1
        batch_shards.append(next(iter(shards)))
    # All batches for one mmap shard stay adjacent, so each array set is loaded
    # only once per epoch despite randomized shard order.
    shard_runs = [
        value for index, value in enumerate(batch_shards)
        if index == 0 or value != batch_shards[index - 1]
    ]
    assert len(shard_runs) == len(set(batch_shards))
    sampler.set_epoch(1)
    assert sorted(
        index for batch in sampler for index in batch) == list(range(5))


def test_staged_validator_reads_serialized_samples_and_is_checksummed(tmp_path):
    sample, settings = _sample("serialized-a")
    output = tmp_path / "signals"

    def validate(dataset):
        restored = dataset.sample(0)
        assert restored.metadata["sample_id"] == "serialized-a"
        assert np.array_equal(
            restored.precursor_intensity, sample.precursor_intensity)
        return StagedValidation(
            audit_tables={
                "serialized_parity.csv": pd.DataFrame([{"passed": True}]),
            },
            summary={"serialized_shards_validated": True},
        )

    report = write_signal_dataset(
        [sample], output, settings, build_metadata={"mode": "test"},
        staged_validator=validate)
    assert report["staged_validation"]["serialized_shards_validated"] is True
    checksums = json.loads((output / "checksums.json").read_text())
    assert "audit/serialized_parity.csv" in checksums
    open_signal_dataset(output)


def test_default_open_rejects_tampered_artifact(tmp_path):
    sample, settings = _sample()
    output = tmp_path / "signals"
    write_signal_dataset(
        [sample], output, settings, build_metadata={"mode": "test"})
    shard = next((output / "shards").rglob("precursor_intensity.npy"))
    with shard.open("ab") as handle:
        handle.write(b"tampered")
    with pytest.raises(ValueError, match="artifact changed"):
        open_signal_dataset(output)


def test_complete_marker_anchors_checksum_manifest(tmp_path):
    sample, settings = _sample()
    output = tmp_path / "signals"
    write_signal_dataset(
        [sample], output, settings, build_metadata={"mode": "test"})
    with (output / "checksums.json").open("a", encoding="utf-8") as handle:
        handle.write("\n")
    with pytest.raises(ValueError, match="differs from COMPLETE"):
        open_signal_dataset(output)


def test_interrupted_overwrite_restores_unique_backup(tmp_path):
    sample, settings = _sample()
    output = tmp_path / "signals"
    write_signal_dataset(
        [sample], output, settings, build_metadata={"mode": "test"})
    backup = output.with_name(f".{output.name}.backup.interrupted")
    output.rename(backup)
    recover_interrupted_publish(output)
    assert output.is_dir()
    assert not backup.exists()
    assert open_signal_dataset(output).sample(0).metadata["sample_id"] \
        == "sample-a"


def test_overwrite_removes_stale_backups_after_verifying_current_output(
        tmp_path):
    first, settings = _sample("first")
    output = tmp_path / "signals"
    write_signal_dataset(
        [first], output, settings, build_metadata={"mode": "test"})
    for suffix in ("old-a", "old-b"):
        shutil.copytree(
            output, output.with_name(f".{output.name}.backup.{suffix}"))

    replacement, _ = _sample("replacement")
    write_signal_dataset(
        [replacement], output, settings, build_metadata={"mode": "test"},
        overwrite=True)

    assert not list(tmp_path.glob(".signals.backup.*"))
    assert open_signal_dataset(output).sample(0).metadata["sample_id"] \
        == "replacement"


def test_overwrite_preserves_backups_when_current_output_is_corrupt(tmp_path):
    first, settings = _sample("first")
    output = tmp_path / "signals"
    write_signal_dataset(
        [first], output, settings, build_metadata={"mode": "test"})
    backup = output.with_name(f".{output.name}.backup.old")
    shutil.copytree(output, backup)
    shard = next((output / "shards").rglob("precursor_intensity.npy"))
    with shard.open("ab") as handle:
        handle.write(b"tampered")
    replacement, _ = _sample("replacement")

    with pytest.raises(ValueError, match="artifact changed"):
        write_signal_dataset(
            [replacement], output, settings,
            build_metadata={"mode": "test"}, overwrite=True)

    assert backup.is_dir()
    assert output.is_dir()


def test_store_refuses_duplicate_ids_without_publishing(tmp_path):
    first, settings = _sample("duplicate")
    second, _ = _sample("duplicate")
    output = tmp_path / "signals"
    with pytest.raises(ValueError, match="duplicate signal sample_id"):
        write_signal_dataset(
            [first, second], output, settings,
            build_metadata={"mode": "test"})
    assert not output.exists()


def test_resumable_store_reuses_only_committed_shards(tmp_path):
    first, settings = _sample("resume-a")
    second, _ = _sample("resume-b")
    third, _ = _sample("resume-c")
    output = tmp_path / "signals"
    build_metadata = {"mode": "test", "source_fingerprints": []}

    def interrupted(completed, checkpoint):
        assert completed == frozenset()
        assert checkpoint == {}
        build_metadata["source_fingerprints"].append({"sha256": "source-a"})
        yield first
        yield second
        raise RuntimeError("simulated interruption")

    with pytest.raises(RuntimeError, match="simulated interruption"):
        write_signal_dataset(
            interrupted, output, settings, build_metadata=build_metadata,
            shard_size=2, resume=True,
            resume_identity={"snapshot": "fixed"})
    building = tmp_path / ".signals.building"
    assert building.is_dir()
    assert (building / "shards" / "shard_00000").is_dir()

    resumed_metadata = {"mode": "test", "source_fingerprints": []}

    def resumed(completed, checkpoint):
        assert completed == frozenset({"resume-a", "resume-b"})
        assert checkpoint["source_fingerprints"] == [{
            "sha256": "source-a",
        }]
        resumed_metadata["source_fingerprints"].extend(
            checkpoint["source_fingerprints"])
        yield third

    report = write_signal_dataset(
        resumed, output, settings, build_metadata=resumed_metadata,
        shard_size=2, resume=True,
        resume_identity={"snapshot": "fixed"})

    assert report["n_samples"] == 3
    assert report["resumable_build"] is True
    assert not building.exists()
    dataset = open_signal_dataset(output)
    assert set(dataset.manifest["sample_id"]) == {
        "resume-a", "resume-b", "resume-c",
    }
    assert (output / "RESUME_STATE.json").is_file()


def test_resumable_store_rejects_different_input_identity(tmp_path):
    first, settings = _sample("resume-a")
    output = tmp_path / "signals"

    def interrupted(_completed, _checkpoint):
        yield first
        raise RuntimeError("stop")

    with pytest.raises(RuntimeError, match="stop"):
        write_signal_dataset(
            interrupted, output, settings, build_metadata={"mode": "test"},
            shard_size=1, resume=True,
            resume_identity={"snapshot": "first"})

    with pytest.raises(ValueError, match="different inputs"):
        write_signal_dataset(
            lambda *_: (), output, settings,
            build_metadata={"mode": "test"}, shard_size=1, resume=True,
            resume_identity={"snapshot": "second"})


def test_pilot_selection_is_balanced_and_row_order_independent():
    rows = []
    for dataset in ("2da", "5da", "normal"):
        for label in (0, 1):
            for index in range(4):
                rows.append({
                    "sample_id": f"{dataset}-{label}-{index}",
                    "dataset": dataset, "label": label,
                })
    frame = pd.DataFrame(rows)
    first = select_pilot_rows(
        frame, correct_per_dataset=2, error_per_dataset=2, seed=7)
    second = select_pilot_rows(
        frame.sample(frac=1, random_state=8),
        correct_per_dataset=2, error_per_dataset=2, seed=7)
    assert first["sample_id"].tolist() == second["sample_id"].tolist()
    assert first.groupby(["dataset", "label"]).size().eq(2).all()


def test_legacy_psm_matcher_tolerates_float32_csv_roundtrip():
    psm = _psm()
    protocol = pd.DataFrame([{
        "sample_id": "frozen-a", "dataset": "2da", "sequence": "AK",
        "charge": 2, "precursor_mz": 500.00001, "rt": 10.00001,
        "raw_title1": "raw-a", "label_type": "positive",
    }])
    identity = [
        "sequence", "charge", "precursor_mz", "rt", "raw_title1",
        "label_type",
    ]
    matched, audit = match_psms_to_protocol(protocol, [psm], identity)
    assert matched["frozen-a"] is psm
    assert audit.loc[0, "status"] == "matched"
