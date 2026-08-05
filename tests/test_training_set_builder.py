import json

import pandas as pd
import pytest

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
        },
        {
            "query_id": "Qweak",
            "parent_id": "Pother",
            "sequence": "WWWWAAAAK",
            "charge": 2,
            "generator": SOURCE_MARKOV,
            "negative_source": SOURCE_MARKOV,
            "generator_seed": 10,
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
