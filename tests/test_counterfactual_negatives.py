import json

import numpy as np
import pandas as pd
import pytest

from spectrum.entrapment_classifier import TargetIndex, classify_peptide
from spectrum.labeling import HeavyType, get_heavy_increase_mass
from spectrum.psm_dataset_manifest import validate_manifest
from spectrum.psm_dataset_manifest import write_manifest
from spectrum.psm_identity import peptide_group_id
from spectrum.psm_info import PSMInfo
from tools.counterfactual_negatives import (
    BUILD_SCHEMA,
    LOCAL_PROPOSAL_SCHEMA,
    PARENT_SAMPLING_SCHEMA,
    SOURCE_COMPOSITION_SHUFFLE,
    SOURCE_KR_POSITION_SHUFFLE,
    SOURCE_LOCAL_MASS_GAP,
    CounterfactualConfig,
    CounterfactualJob,
    _precursor_mz,
    _sample_parents,
    build_counterfactual_negatives,
    run_job,
)


def _target(sequence):
    return TargetIndex(
        raw_text=sequence,
        li_normalized_text=sequence.replace("I", "L"),
        n_proteins=1,
    )


def _parent(sequence="MGTPEPTK", *, label_type="positive", modify=None,
            raw="raw1", prepared=True, dataset_split="label_dev_train"):
    return PSMInfo(
        sequence=sequence,
        charge=2,
        modify=modify or [],
        rt=np.float32(12.5),
        precursor_mz=np.float32(_precursor_mz(sequence, 2)),
        raw_title=raw,
        protein_names="TARGET",
        label_type=label_type,
        heavy_confirmed=True if prepared else None,
        dataset_split=dataset_split if prepared else None,
        peptide_group_id=peptide_group_id(sequence) if prepared else None,
    )


def _all_source_config(**overrides):
    values = dict(
        dataset_split="label_dev_train",
        seed=7,
        composition_shuffle_per_parent=1,
        kr_position_shuffle_per_parent=1,
        local_denovo_per_parent=1,
        max_attempts_per_source=1000,
        local_min_segment_length=2,
        local_max_segment_length=3,
    )
    values.update(overrides)
    return CounterfactualConfig(**values)


def test_builds_deterministic_grouped_hypotheses_from_three_sources():
    parent = _parent()
    cfg = _all_source_config()

    first = build_counterfactual_negatives(
        [parent], _target(parent._sequence), cfg)
    second = build_counterfactual_negatives(
        [parent], _target(parent._sequence), cfg)

    assert first.audit["schema"] == BUILD_SCHEMA
    assert first.audit["counts"]["eligible_parents"] == 1
    assert first.audit["counts"]["negative_children"] == 3
    assert set(first.manifest["negative_source"]) == {
        SOURCE_COMPOSITION_SHUFFLE,
        SOURCE_KR_POSITION_SHUFFLE,
        SOURCE_LOCAL_MASS_GAP,
    }
    assert list(first.manifest["query_id"]) == list(
        second.manifest["query_id"])
    assert list(first.manifest["sequence"]) == list(
        second.manifest["sequence"])

    parent_row, *children = first.psms
    assert parent_row._label_type == "positive"
    assert all(child._label_type == "negative" for child in children)
    assert all(child._group_id == parent_row._group_id for child in children)
    assert all(child._parent_id == parent_row._parent_id for child in children)
    assert all(child._peptide_group_id == parent_row._peptide_group_id
               for child in children)
    assert all(child._dataset_split == "label_dev_train"
               for child in children)
    assert all(child._precursor_mz == parent_row._precursor_mz
               for child in children)
    assert all(child._rt == parent_row._rt for child in children)

    target = _target(parent._sequence)
    assert all(classify_peptide(child._sequence, target) == "L4"
               for child in children)
    assert all(child._sequence.replace("I", "L")
               != parent._sequence.replace("I", "L")
               for child in children)
    assert (first.manifest["n_changed_fragment_positions"] >= 2).all()
    assert (first.manifest["candidate_observed_mass_error_ppm"].abs()
            <= cfg.precursor_mass_tolerance_ppm).all()


def test_parallel_generation_exactly_matches_serial_rows_and_order():
    parents = [_parent(raw=f"raw{index}") for index in range(4)]
    cfg = _all_source_config()
    target = _target(parents[0]._sequence)

    serial = build_counterfactual_negatives(
        parents, target, cfg, workers=1, worker_chunk_size=1)
    parallel = build_counterfactual_negatives(
        parents, target, cfg, workers=2, worker_chunk_size=1)

    assert [psm.to_dict() for psm in parallel.psms] == [
        psm.to_dict() for psm in serial.psms
    ]
    pd.testing.assert_frame_equal(parallel.manifest, serial.manifest)
    assert parallel.audit["counts"] == serial.audit["counts"]
    assert parallel.audit["failures"] == serial.audit["failures"]
    assert serial.audit["execution"]["effective_workers"] == 1
    assert parallel.audit["execution"]["effective_workers"] == 2


def test_pilot_parent_sampling_is_stable_across_input_order():
    parents = [_parent(raw=f"raw{index}") for index in range(10)]

    selected, audit = _sample_parents(parents, max_parents=4, seed=17)
    reversed_selected, _ = _sample_parents(
        list(reversed(parents)), max_parents=4, seed=17)

    assert {psm._raw_title for psm in selected} == {
        psm._raw_title for psm in reversed_selected
    }
    assert [psm._raw_title for psm in selected] == sorted(
        (psm._raw_title for psm in selected),
        key=lambda raw: int(raw.removeprefix("raw")),
    )
    assert audit == {
        "schema": PARENT_SAMPLING_SCHEMA,
        "available_parents": 10,
        "selected_parents": 4,
        "max_parents": 4,
        "seed": 17,
        "applied": True,
    }


def test_local_mass_gap_is_high_overlap_but_has_distinguishing_fragments():
    parent = _parent()
    result = build_counterfactual_negatives(
        [parent], _target(parent._sequence),
        _all_source_config(
            composition_shuffle_per_parent=0,
            kr_position_shuffle_per_parent=0,
            local_min_segment_length=2,
            local_max_segment_length=2,
        ),
    )

    row = result.manifest.iloc[0]
    assert row["negative_source"] == SOURCE_LOCAL_MASS_GAP
    assert row["local_proposal_schema"] == LOCAL_PROPOSAL_SCHEMA
    assert not bool(row["local_uses_observed_fragment_anchors"])
    assert sorted(row["local_original"]) != sorted(row["local_replacement"])
    assert row["shared_theoretical_fragment_fraction"] >= 0.5
    assert row["n_changed_fragment_positions"] >= 2
    assert abs(row["candidate_parent_neutral_mass_delta_da"]) < 0.01


def test_child_heavy_precursor_uses_child_sequence_shift():
    parent = _parent("MAQTPEPK")
    result = build_counterfactual_negatives(
        [parent], _target(parent._sequence),
        CounterfactualConfig(
            dataset_split="label_dev_train",
            seed=12,
            composition_shuffle_per_parent=0,
            kr_position_shuffle_per_parent=0,
            local_denovo_per_parent=1,
            local_min_segment_length=3,
            local_max_segment_length=3,
            local_min_shared_fragment_fraction=0.4,
            max_attempts_per_source=100,
        ),
    )

    row = result.manifest.iloc[0]
    child = result.psms[-1]
    assert row["kr_count_delta"] != 0
    assert row["label_shift_delta"] != 0
    heavy_precursor, _ = child.get_heavy_info(HeavyType.SILAC)
    expected_shift = get_heavy_increase_mass(
        child._sequence, HeavyType.SILAC)
    assert heavy_precursor - child._precursor_mz == pytest.approx(
        expected_shift / child._charge, abs=5e-5)
    assert expected_shift == pytest.approx(row["candidate_label_shift"])
    assert expected_shift != pytest.approx(row["parent_label_shift"])


def test_parent_filtering_is_audited_without_relabeling_inputs():
    eligible = _parent()
    negative = _parent(label_type="negative", raw="raw2")
    modified = _parent(modify=[(2, 35)], raw="raw3")
    result = build_counterfactual_negatives(
        [eligible, negative, modified], _target(eligible._sequence),
        _all_source_config(
            composition_shuffle_per_parent=1,
            kr_position_shuffle_per_parent=0,
            local_denovo_per_parent=0,
        ),
    )

    assert result.audit["counts"]["input_parents"] == 3
    assert result.audit["counts"]["eligible_parents"] == 1
    assert result.audit["failures"]["parent:not_positive"] == 1
    assert result.audit["failures"]["parent:modified"] == 1
    assert negative._label_type == "negative"
    assert modified._modify == [(2, 35)]


def test_rejects_missing_split_and_non_silac():
    parent = _parent()
    with pytest.raises(ValueError, match="dataset_split"):
        build_counterfactual_negatives(
            [parent], _target(parent._sequence),
            CounterfactualConfig(dataset_split=""),
        )
    with pytest.raises(ValueError, match="supports SILAC only"):
        build_counterfactual_negatives(
            [parent], _target(parent._sequence),
            CounterfactualConfig(
                dataset_split="train", labeling=HeavyType.C13),
        )


def test_rejects_unprepared_parent_and_split_mismatch():
    target = _target("MGTPEPTK")
    cfg = _all_source_config(
        composition_shuffle_per_parent=1,
        kr_position_shuffle_per_parent=0,
        local_denovo_per_parent=0,
    )
    with pytest.raises(ValueError, match="no eligible positive parents"):
        build_counterfactual_negatives(
            [_parent(prepared=False)], target, cfg)
    with pytest.raises(ValueError, match="no eligible positive parents"):
        build_counterfactual_negatives(
            [_parent(dataset_split="immutable_test")], target, cfg)


def test_run_job_writes_extractable_psm_json_manifest_and_audit(tmp_path):
    parent = _parent()
    parent_path = tmp_path / "parents.json"
    parent_path.write_text(
        json.dumps([parent.to_dict()]), encoding="utf-8")
    write_manifest(str(parent_path), [parent], HeavyType.SILAC)
    fasta = tmp_path / "target.fasta"
    fasta.write_text(f">p\n{parent._sequence}\n", encoding="utf-8")
    config_path = tmp_path / "counterfactual.ini"
    config_path.write_text(
        "[counterfactual]\ndataset_split=label_dev_train\n",
        encoding="utf-8")
    output_psms = tmp_path / "counterfactual.json"
    output_manifest = tmp_path / "counterfactual.tsv"
    output_audit = tmp_path / "counterfactual.audit.json"
    job = CounterfactualJob(
        parents=str(parent_path),
        target_fasta=str(fasta),
        output_psms=str(output_psms),
        output_manifest=str(output_manifest),
        output_audit=str(output_audit),
        build=_all_source_config(
            composition_shuffle_per_parent=1,
            kr_position_shuffle_per_parent=0,
            local_denovo_per_parent=0,
        ),
    )

    audit = run_job(job, source_config_path=str(config_path))

    payload = json.loads(output_psms.read_text())
    manifest = pd.read_csv(output_manifest, sep="\t")
    on_disk_audit = json.loads(output_audit.read_text())
    assert len(payload) == 2
    assert payload[0]["label_type"] == "positive"
    assert payload[1]["label_type"] == "negative"
    assert manifest.iloc[0]["query_id"] == payload[1]["query_id"]
    assert audit["counts"]["negative_children"] == 1
    assert audit["parent_sampling"]["selected_parents"] == 1
    assert audit["execution"]["effective_workers"] == 1
    assert on_disk_audit["outputs"]["psms"] == str(output_psms)
    sidecar = validate_manifest(str(output_psms), "silac", require=True)
    assert sidecar["dataset"]["counts_by_label_type"] == {
        "negative": 1, "positive": 1,
    }
