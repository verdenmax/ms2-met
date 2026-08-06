import json

import pandas as pd
import pytest
from pyteomics import mass

from spectrum.entrapment_classifier import classify_peptide, load_target_fasta
from spectrum.psm_info import HeavyType
from tools.training_set_builder import (
    AssemblyConfig,
    QueryBuildConfig,
    SOURCE_GOLD,
    SOURCE_MARKOV,
    SOURCE_POSITIVE,
    SOURCE_SHUFFLE,
    assemble_training_set,
    generate_queries,
)


def _write_fasta(path):
    path.write_text(
        ">p1\nMPEPTIDEKACDEFGHIKLMNPQRSTVWYR\n"
        ">p2\nGAVLIPFYWSTCMNQHDEKRAGTLMNPQR\n",
        encoding="utf-8",
    )


def test_generate_queries_writes_two_generator_manifest_and_fasta(tmp_path):
    fasta = tmp_path / "target.fasta"
    _write_fasta(fasta)
    positives = tmp_path / "positive.csv"
    pd.DataFrame([{
        "sequence": "ACDEFGHIK",
        "charge": 2,
        "precursor_mz": 510.0,
        "label": 1,
        "heavy_confirmed": 1,
        "modification_count": 0,
    }]).to_csv(positives, index=False)
    manifest = tmp_path / "queries.tsv"
    query_fasta = tmp_path / "queries.fasta"

    summary = generate_queries(QueryBuildConfig(
        positives=str(positives),
        target_fasta=str(fasta),
        output_manifest=str(manifest),
        output_fasta=str(query_fasta),
        labeling=HeavyType.SILAC,
        shuffle_per_parent=1,
        markov_per_parent=1,
        max_attempts=2000,
        mz_bin_width=1000.0,
        shift_bin_width=1000.0,
        seed=7,
    ))

    out = pd.read_csv(manifest, sep="\t")
    assert summary["n_queries"] == 2
    assert set(out["generator"]) == {SOURCE_SHUFFLE, SOURCE_MARKOV}
    assert set(out["labeling"]) == {"silac"}
    assert out["sequence"].is_unique
    assert (out["sequence"].str.endswith(("K", "R"))).all()
    assert (out["negative_confidence"] == "silver").all()
    target = load_target_fasta(str(fasta))
    assert all(classify_peptide(seq, target) == "L4"
               for seq in out["sequence"])
    fasta_text = query_fasta.read_text(encoding="utf-8")
    assert fasta_text.count(">SYNTH_") == 2
    assert (tmp_path / "queries.tsv.summary.json").exists()


def test_generate_queries_requires_independent_heavy_confirmation(tmp_path):
    fasta = tmp_path / "target.fasta"
    _write_fasta(fasta)
    positives = tmp_path / "positive.csv"
    pd.DataFrame([{
        "sequence": "ACDEFGHIK", "charge": 2, "label": 1,
    }]).to_csv(positives, index=False)
    with pytest.raises(ValueError, match="heavy_confirmed"):
        generate_queries(QueryBuildConfig(
            positives=str(positives),
            target_fasta=str(fasta),
            output_manifest=str(tmp_path / "q.tsv"),
            output_fasta=str(tmp_path / "q.fasta"),
        ))


@pytest.mark.parametrize("labeling", [HeavyType.CHEAVY, HeavyType.NHEAVY])
def test_generate_rejects_modified_uniform_label_mode(tmp_path, labeling):
    fasta = tmp_path / "target.fasta"
    _write_fasta(fasta)
    positives = tmp_path / "positive.csv"
    pd.DataFrame([{
        "sequence": "ACDEFGHIK",
        "charge": 2,
        "label": 1,
        "heavy_confirmed": 1,
        "modification_count": 1,
    }]).to_csv(positives, index=False)

    with pytest.raises(ValueError, match="modified.*C13/N15|C13/N15.*modified"):
        generate_queries(QueryBuildConfig(
            positives=str(positives),
            target_fasta=str(fasta),
            output_manifest=str(tmp_path / "q.tsv"),
            output_fasta=str(tmp_path / "q.fasta"),
            labeling=labeling,
            exclude_modified=False,
        ))


@pytest.mark.parametrize("labeling", [HeavyType.CHEAVY, HeavyType.NHEAVY])
def test_generate_uniform_label_retains_no_kr_parent(tmp_path, labeling):
    fasta = tmp_path / "target.fasta"
    _write_fasta(fasta)
    positives = tmp_path / "positive.csv"
    pd.DataFrame([{
        "sequence": "ACDFGHILMNPQSTVWY",
        "charge": 2,
        "label": 1,
        "heavy_confirmed": 1,
        "modification_count": 0,
    }]).to_csv(positives, index=False)
    manifest = tmp_path / "q.tsv"

    generate_queries(QueryBuildConfig(
        positives=str(positives),
        target_fasta=str(fasta),
        output_manifest=str(manifest),
        output_fasta=str(tmp_path / "q.fasta"),
        labeling=labeling,
        shuffle_per_parent=1,
        markov_per_parent=0,
        require_tryptic_c_terminus=False,
        max_attempts=2000,
        seed=11,
    ))

    out = pd.read_csv(manifest, sep="\t")
    assert len(out) == 1
    assert not out.iloc[0]["sequence"].endswith(("K", "R"))
    assert out.iloc[0]["labeling"] == (
        "c13" if labeling is HeavyType.CHEAVY else "n15")


@pytest.mark.parametrize("labeling,element", [
    (HeavyType.CHEAVY, "C"),
    (HeavyType.NHEAVY, "N"),
])
def test_markov_uniform_label_does_not_require_exact_element_count(
        tmp_path, labeling, element):
    fasta = tmp_path / "target.fasta"
    _write_fasta(fasta)
    positives = tmp_path / "positive.csv"
    parent = "ACDEFGHIK"
    pd.DataFrame([{
        "sequence": parent,
        "charge": 2,
        "label": 1,
        "heavy_confirmed": 1,
        "modification_count": 0,
    }]).to_csv(positives, index=False)
    manifest = tmp_path / "q.tsv"

    generate_queries(QueryBuildConfig(
        positives=str(positives),
        target_fasta=str(fasta),
        output_manifest=str(manifest),
        output_fasta=str(tmp_path / "q.fasta"),
        labeling=labeling,
        shuffle_per_parent=0,
        markov_per_parent=1,
        mz_bin_width=10000.0,
        shift_bin_width=10000.0,
        max_attempts=5000,
        seed=19,
    ))

    candidate = pd.read_csv(manifest, sep="\t").iloc[0]["sequence"]
    assert mass.Composition(candidate)[element] != mass.Composition(parent)[element]


def _feature_row(sequence, raw, label, *, light=100.0, heavy=80.0,
                 light_frags=5, heavy_frags=3, confirmed=1):
    return {
        "sequence": sequence,
        "charge": 2,
        "precursor_mz": 500.0,
        "rt": 20.0,
        "raw_title1": raw,
        "protein_names": "sp|P1|X_HUMAN",
        "sequence_len": len(sequence),
        "label": label,
        "label_type": "positive" if label else "negative",
        "heavy_confirmed": confirmed,
        "precursor_light_max_int": light,
        "precursor_heavy_max_int": heavy,
        "precursor_xic_empty": 0,
        "all_count": light_frags,
        "q1a_TP_shifted": heavy_frags,
        "heavy_in_raw": 1,
        "heavy_out_of_range": 0,
        "total_silac_shift": 8.0,
        "psm_is_split_window": 1,
        "precursor_pearson": 0.5,
    }


def test_assemble_filters_signal_matches_distribution_and_tracks_parent(tmp_path):
    fasta = tmp_path / "target.fasta"
    _write_fasta(fasta)
    positives_path = tmp_path / "positive.csv"
    gold_path = tmp_path / "gold.csv"
    silver_path = tmp_path / "silver.csv"
    heldout_path = tmp_path / "heldout.csv"
    manifest_path = tmp_path / "manifest.tsv"

    positives = pd.DataFrame([
        _feature_row("ACDEFGHIK", "train_raw", 1),
        _feature_row("LMNPQRSTVR", "train_raw", 1),
    ])
    positives.to_csv(positives_path, index=False)
    pd.DataFrame([
        _feature_row("TTTTAAAK", "train_raw", 0, confirmed=0),
    ]).to_csv(gold_path, index=False)
    silver = pd.DataFrame([
        _feature_row("VVVVAAAAK", "train_raw", 0, confirmed=0),
        # Has a peak but lacks the required heavy fragment evidence.
        _feature_row("WWWWAAAAK", "train_raw", 0, heavy_frags=1, confirmed=0),
    ])
    silver.to_csv(silver_path, index=False)
    pd.DataFrame([
        _feature_row("ACDEFGHIK", "heldout_raw", 1),
    ]).to_csv(heldout_path, index=False)
    parent_id = "Pparent"
    pd.DataFrame([
        {
            "query_id": "Qgood",
            "parent_id": parent_id,
            "sequence": "VVVVAAAAK",
            "charge": 2,
            "generator": SOURCE_SHUFFLE,
            "negative_source": SOURCE_SHUFFLE,
            "generator_seed": 9,
            "labeling": "silac",
        },
        {
            "query_id": "Qweak",
            "parent_id": "Pother",
            "sequence": "WWWWAAAAK",
            "charge": 2,
            "generator": SOURCE_MARKOV,
            "negative_source": SOURCE_MARKOV,
            "generator_seed": 10,
            "labeling": "silac",
        },
    ]).to_csv(manifest_path, sep="\t", index=False)

    output = tmp_path / "training.csv"
    audit = tmp_path / "training.audit.json"
    summary = assemble_training_set(AssemblyConfig(
        positive_features=(str(positives_path),),
        gold_features=(str(gold_path),),
        silver_features=(str(silver_path),),
        query_manifest=str(manifest_path),
        target_fasta=str(fasta),
        heldout_features=(str(heldout_path),),
        output_features=str(output),
        output_audit=str(audit),
        distribution_columns=("charge", "precursor_mz", "sequence_len",
                              "total_silac_shift",
                              "psm_is_split_window", "rt"),
        seed=3,
    ))

    out = pd.read_csv(output)
    assert summary["silver_signal_filter"]["input"] == 2
    assert summary["silver_signal_filter"]["kept"] == 1
    assert set(out["negative_source"]) == {
        SOURCE_POSITIVE, SOURCE_GOLD, SOURCE_SHUFFLE}
    synthetic = out[out["negative_source"] == SOURCE_SHUFFLE].iloc[0]
    assert synthetic["group_id"] == parent_id
    assert synthetic["label"] == 0
    assert json.loads(audit.read_text())["heldout"]["checked"] is True


def test_assemble_rejects_raw_overlap_with_heldout(tmp_path):
    fasta = tmp_path / "target.fasta"
    _write_fasta(fasta)
    row = _feature_row("ACDEFGHIK", "same_raw", 1)
    positives = tmp_path / "positive.csv"
    heldout = tmp_path / "heldout.csv"
    silver = tmp_path / "silver.csv"
    manifest = tmp_path / "manifest.tsv"
    pd.DataFrame([row]).to_csv(positives, index=False)
    pd.DataFrame([row]).to_csv(heldout, index=False)
    pd.DataFrame([
        _feature_row("VVVVAAAK", "train_other", 0, confirmed=0),
    ]).to_csv(silver, index=False)
    pd.DataFrame([{
        "query_id": "Q1", "parent_id": "P1", "sequence": "VVVVAAAK",
        "charge": 2, "generator": SOURCE_SHUFFLE,
        "negative_source": SOURCE_SHUFFLE,
        "labeling": "silac",
    }]).to_csv(manifest, sep="\t", index=False)

    with pytest.raises(ValueError, match="raw leakage"):
        assemble_training_set(AssemblyConfig(
            positive_features=(str(positives),),
            gold_features=(),
            silver_features=(str(silver),),
            query_manifest=str(manifest),
            target_fasta=str(fasta),
            heldout_features=(str(heldout),),
            output_features=str(tmp_path / "out.csv"),
            output_audit=str(tmp_path / "audit.json"),
        ))


@pytest.mark.parametrize("labeling", [HeavyType.CHEAVY, HeavyType.NHEAVY])
def test_assemble_rejects_modified_uniform_label_rows(tmp_path, labeling):
    fasta = tmp_path / "target.fasta"
    _write_fasta(fasta)
    positives_path = tmp_path / "positive.csv"
    silver_path = tmp_path / "silver.csv"
    heldout_path = tmp_path / "heldout.csv"
    manifest_path = tmp_path / "manifest.tsv"

    positive = _feature_row("ACDEFGHIK", "train_raw", 1)
    positive["modification_count"] = 1
    pd.DataFrame([positive]).to_csv(positives_path, index=False)
    pd.DataFrame([
        _feature_row("VVVVAAAAK", "train_raw", 0, confirmed=0),
    ]).to_csv(silver_path, index=False)
    pd.DataFrame([
        _feature_row("ACDEFGHIK", "heldout_raw", 1),
    ]).to_csv(heldout_path, index=False)
    pd.DataFrame([{
        "query_id": "Q1", "parent_id": "P1",
        "sequence": "VVVVAAAAK", "charge": 2,
        "generator": SOURCE_SHUFFLE,
        "negative_source": SOURCE_SHUFFLE,
        "labeling": "c13" if labeling is HeavyType.CHEAVY else "n15",
    }]).to_csv(manifest_path, sep="\t", index=False)

    with pytest.raises(ValueError, match="modified C13/N15"):
        assemble_training_set(AssemblyConfig(
            positive_features=(str(positives_path),),
            gold_features=(),
            silver_features=(str(silver_path),),
            query_manifest=str(manifest_path),
            target_fasta=str(fasta),
            labeling=labeling,
            heldout_features=(str(heldout_path),),
            output_features=str(tmp_path / "out.csv"),
            output_audit=str(tmp_path / "audit.json"),
        ))


def test_assemble_rejects_manifest_labeling_mismatch(tmp_path):
    fasta = tmp_path / "target.fasta"
    _write_fasta(fasta)
    positives = tmp_path / "positive.csv"
    silver = tmp_path / "silver.csv"
    heldout = tmp_path / "heldout.csv"
    manifest = tmp_path / "manifest.tsv"
    pd.DataFrame([
        _feature_row("ACDEFGHIK", "train_raw", 1),
    ]).to_csv(positives, index=False)
    pd.DataFrame([
        _feature_row("VVVVAAAAK", "train_raw", 0, confirmed=0),
    ]).to_csv(silver, index=False)
    pd.DataFrame([
        _feature_row("ACDEFGHIK", "heldout_raw", 1),
    ]).to_csv(heldout, index=False)
    pd.DataFrame([{
        "query_id": "Q1", "parent_id": "P1",
        "sequence": "VVVVAAAAK", "charge": 2,
        "generator": SOURCE_SHUFFLE,
        "negative_source": SOURCE_SHUFFLE,
        "labeling": "c13",
    }]).to_csv(manifest, sep="\t", index=False)

    with pytest.raises(ValueError, match="manifest labeling.*assembly"):
        assemble_training_set(AssemblyConfig(
            positive_features=(str(positives),),
            gold_features=(),
            silver_features=(str(silver),),
            query_manifest=str(manifest),
            target_fasta=str(fasta),
            labeling=HeavyType.SILAC,
            heldout_features=(str(heldout),),
            output_features=str(tmp_path / "out.csv"),
            output_audit=str(tmp_path / "audit.json"),
        ))


def test_assemble_requires_heavy_acquisition_range_evidence(tmp_path):
    fasta = tmp_path / "target.fasta"
    _write_fasta(fasta)
    positives = tmp_path / "positive.csv"
    silver = tmp_path / "silver.csv"
    heldout = tmp_path / "heldout.csv"
    manifest = tmp_path / "manifest.tsv"
    pd.DataFrame([
        _feature_row("ACDEFGHIK", "train_raw", 1),
    ]).to_csv(positives, index=False)
    silver_row = _feature_row(
        "VVVVAAAAK", "train_raw", 0, confirmed=0)
    silver_row.pop("heavy_in_raw")
    silver_row.pop("heavy_out_of_range")
    pd.DataFrame([silver_row]).to_csv(silver, index=False)
    pd.DataFrame([
        _feature_row("ACDEFGHIK", "heldout_raw", 1),
    ]).to_csv(heldout, index=False)
    pd.DataFrame([{
        "query_id": "Q1", "parent_id": "P1",
        "sequence": "VVVVAAAAK", "charge": 2,
        "generator": SOURCE_SHUFFLE,
        "negative_source": SOURCE_SHUFFLE,
        "labeling": "silac",
    }]).to_csv(manifest, sep="\t", index=False)

    with pytest.raises(ValueError, match="acquisition-range"):
        assemble_training_set(AssemblyConfig(
            positive_features=(str(positives),),
            gold_features=(),
            silver_features=(str(silver),),
            query_manifest=str(manifest),
            target_fasta=str(fasta),
            heldout_features=(str(heldout),),
            output_features=str(tmp_path / "out.csv"),
            output_audit=str(tmp_path / "audit.json"),
        ))


def test_assemble_c13_uses_canonical_shift_and_keeps_no_kr_gold(tmp_path):
    fasta = tmp_path / "target.fasta"
    _write_fasta(fasta)
    paths = {
        name: tmp_path / f"{name}.csv"
        for name in ("positive", "gold", "silver", "heldout")
    }
    positive = _feature_row("ACDFGHILMNPQSTVWY", "train_raw", 1)
    gold = _feature_row("TTTTAAAA", "train_raw", 0, confirmed=0)
    silver = _feature_row("VVVVAAAA", "train_raw", 0, confirmed=0)
    heldout = _feature_row("ACDFGHILMNPQSTVWY", "heldout_raw", 1)
    for row in (positive, gold, silver, heldout):
        row["total_label_shift"] = 40.0
        row.pop("total_silac_shift")
    pd.DataFrame([positive]).to_csv(paths["positive"], index=False)
    pd.DataFrame([gold]).to_csv(paths["gold"], index=False)
    pd.DataFrame([silver]).to_csv(paths["silver"], index=False)
    pd.DataFrame([heldout]).to_csv(paths["heldout"], index=False)
    manifest = tmp_path / "manifest.tsv"
    pd.DataFrame([{
        "query_id": "Q1", "parent_id": "P1",
        "sequence": "VVVVAAAA", "charge": 2,
        "generator": SOURCE_SHUFFLE,
        "negative_source": SOURCE_SHUFFLE,
        "labeling": "c13",
    }]).to_csv(manifest, sep="\t", index=False)

    output = tmp_path / "out.csv"
    summary = assemble_training_set(AssemblyConfig(
        positive_features=(str(paths["positive"]),),
        gold_features=(str(paths["gold"]),),
        silver_features=(str(paths["silver"]),),
        query_manifest=str(manifest),
        target_fasta=str(fasta),
        labeling=HeavyType.CHEAVY,
        heldout_features=(str(paths["heldout"]),),
        output_features=str(output),
        output_audit=str(tmp_path / "audit.json"),
    ))

    out = pd.read_csv(output)
    assert summary["labeling"] == "c13"
    assert summary["distribution_matching"]["shift_column"] == (
        "total_label_shift")
    assert "TTTTAAAA" in set(out["sequence"])
    assert set(out["labeling"]) == {"c13"}


def test_assemble_c13_rejects_legacy_silac_shift(tmp_path):
    fasta = tmp_path / "target.fasta"
    _write_fasta(fasta)
    positive_path = tmp_path / "positive.csv"
    silver_path = tmp_path / "silver.csv"
    heldout_path = tmp_path / "heldout.csv"
    manifest = tmp_path / "manifest.tsv"
    pd.DataFrame([
        _feature_row("ACDEFGHIK", "train_raw", 1),
    ]).to_csv(positive_path, index=False)
    pd.DataFrame([
        _feature_row("VVVVAAAAK", "train_raw", 0, confirmed=0),
    ]).to_csv(silver_path, index=False)
    pd.DataFrame([
        _feature_row("ACDEFGHIK", "heldout_raw", 1),
    ]).to_csv(heldout_path, index=False)
    pd.DataFrame([{
        "query_id": "Q1", "parent_id": "P1",
        "sequence": "VVVVAAAAK", "charge": 2,
        "generator": SOURCE_SHUFFLE,
        "negative_source": SOURCE_SHUFFLE,
        "labeling": "c13",
    }]).to_csv(manifest, sep="\t", index=False)

    with pytest.raises(ValueError, match="requires total_label_shift"):
        assemble_training_set(AssemblyConfig(
            positive_features=(str(positive_path),),
            gold_features=(),
            silver_features=(str(silver_path),),
            query_manifest=str(manifest),
            target_fasta=str(fasta),
            labeling=HeavyType.CHEAVY,
            heldout_features=(str(heldout_path),),
            output_features=str(tmp_path / "out.csv"),
            output_audit=str(tmp_path / "audit.json"),
        ))
