import json

import numpy as np
import pandas as pd
import pytest

from spectrum.dia_data import XIC_DTYPE
from spectrum.psm_info import PSMInfo
from tools.deep_trainer.phase2.extraction import extract_signal_sample
from tools.deep_trainer.phase2.builder import _source_row_metadata
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


def test_store_refuses_duplicate_ids_without_publishing(tmp_path):
    first, settings = _sample("duplicate")
    second, _ = _sample("duplicate")
    output = tmp_path / "signals"
    with pytest.raises(ValueError, match="duplicate signal sample_id"):
        write_signal_dataset(
            [first, second], output, settings,
            build_metadata={"mode": "test"})
    assert not output.exists()


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
