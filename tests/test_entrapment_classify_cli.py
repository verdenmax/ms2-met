"""Tests for tools/entrapment_classify.py CLI."""
import json
import subprocess
import sys
import textwrap

import pandas as pd
import pytest


def _make_negatives_json(tmp_path, psms):
    """Write a negatives PSM JSON in the format produced by
    tools/extract_common.py (each PSM is a dict)."""
    path = tmp_path / "negatives.json"
    with open(path, "w") as f:
        json.dump(psms, f)
    return str(path)


def _make_target_fasta(tmp_path, sequences):
    """sequences: dict[header -> aa_seq]"""
    path = tmp_path / "target.fasta"
    with open(path, "w") as f:
        for header, seq in sequences.items():
            f.write(f">{header}\n{seq}\n")
    return str(path)


def _run_cli(*args):
    """Invoke the CLI as a subprocess so we exercise the full module."""
    cmd = [sys.executable, "-m", "tools.entrapment_classify", *args]
    return subprocess.run(cmd, capture_output=True, text=True)


def test_cli_basic_l0_l1_l4(tmp_path):
    negatives = _make_negatives_json(tmp_path, [
        # All entries are "negative" PSMs from extract_common output.
        # CLI only classifies entries with label_type=="negative".
        {"sequence": "PEPTIDE", "charge": 2, "modify": [],
         "rt": 30.0, "precursor_mz": 400.0,
         "raw_title": "raw1", "protein_names": "X_YEAST",
         "label_type": "negative"},
        {"sequence": "IIIR", "charge": 2, "modify": [],
         "rt": 31.0, "precursor_mz": 400.0,
         "raw_title": "raw1", "protein_names": "X_YEAST",
         "label_type": "negative"},
        {"sequence": "ZZZZZZ", "charge": 2, "modify": [],
         "rt": 32.0, "precursor_mz": 400.0,
         "raw_title": "raw1", "protein_names": "X_YEAST",
         "label_type": "negative"},
    ])
    target = _make_target_fasta(tmp_path, {
        "p1_HUMAN": "MKPEPTIDEKAAARLLLR",
    })
    out_tsv = tmp_path / "classified.tsv"

    result = _run_cli(
        "--negatives", negatives,
        "--target-fasta", target,
        "--output", str(out_tsv),
    )
    assert result.returncode == 0, (
        f"CLI failed: stderr={result.stderr}, stdout={result.stdout}")
    assert out_tsv.exists()

    df = pd.read_csv(out_tsv, sep="\t", dtype=str, keep_default_na=False)
    # Required schema columns for load_entrapment_classifications
    for col in ("peptide", "charge", "spectrum_file", "group", "level"):
        assert col in df.columns

    lookup = {(r.peptide, r.charge, r.spectrum_file): r.level
              for r in df.itertuples(index=False)}
    assert lookup[("PEPTIDE", "2", "raw1")] == "L0"
    assert lookup[("IIIR", "2", "raw1")] == "L1"
    assert lookup[("ZZZZZZ", "2", "raw1")] == "L4"

    # group must be "trap" so it survives extract_common's group filter
    assert (df["group"] == "trap").all()


def test_cli_only_classifies_negatives(tmp_path):
    """positive PSMs in the JSON must NOT appear in the output TSV."""
    negatives = _make_negatives_json(tmp_path, [
        {"sequence": "POSPEP", "charge": 2, "modify": [],
         "rt": 30.0, "precursor_mz": 400.0,
         "raw_title": "raw1", "protein_names": "X_HUMAN",
         "label_type": "positive"},
        {"sequence": "TRAPPEP", "charge": 2, "modify": [],
         "rt": 31.0, "precursor_mz": 400.0,
         "raw_title": "raw1", "protein_names": "X_YEAST",
         "label_type": "negative"},
    ])
    target = _make_target_fasta(tmp_path, {"p1_HUMAN": "POSPEPK"})
    out_tsv = tmp_path / "classified.tsv"

    result = _run_cli(
        "--negatives", negatives,
        "--target-fasta", target,
        "--output", str(out_tsv),
    )
    assert result.returncode == 0, result.stderr

    df = pd.read_csv(out_tsv, sep="\t")
    assert "POSPEP" not in df["peptide"].values
    assert "TRAPPEP" in df["peptide"].values


def test_cli_output_compatible_with_extract_common_loader(tmp_path):
    """Output TSV must round-trip through load_entrapment_classifications."""
    negatives = _make_negatives_json(tmp_path, [
        {"sequence": "PEPTIDE", "charge": 2, "modify": [],
         "rt": 30.0, "precursor_mz": 400.0,
         "raw_title": "raw1", "protein_names": "X_YEAST",
         "label_type": "negative"},
    ])
    target = _make_target_fasta(tmp_path, {"p1": "MKPEPTIDEK"})
    out_tsv = tmp_path / "classified.tsv"

    result = _run_cli(
        "--negatives", negatives,
        "--target-fasta", target,
        "--output", str(out_tsv),
    )
    assert result.returncode == 0, result.stderr

    from tools.extract_common import load_entrapment_classifications
    cls = load_entrapment_classifications(str(out_tsv))
    assert cls[("PEPTIDE", 2, "raw1")] == "L0"


def test_cli_logs_level_distribution(tmp_path):
    """CLI should report L0/L1/L4 counts on stderr or log."""
    negatives = _make_negatives_json(tmp_path, [
        {"sequence": "L0PEP", "charge": 2, "modify": [],
         "rt": 30.0, "precursor_mz": 400.0,
         "raw_title": "r", "protein_names": "X_YEAST",
         "label_type": "negative"},
        {"sequence": "L4PEP", "charge": 2, "modify": [],
         "rt": 31.0, "precursor_mz": 400.0,
         "raw_title": "r", "protein_names": "X_YEAST",
         "label_type": "negative"},
    ])
    target = _make_target_fasta(tmp_path, {"p1": "L0PEPK"})
    out_tsv = tmp_path / "classified.tsv"
    log_path = tmp_path / "run.log"

    result = _run_cli(
        "--negatives", negatives,
        "--target-fasta", target,
        "--output", str(out_tsv),
        "--logpath", str(log_path),
    )
    assert result.returncode == 0, result.stderr

    combined = result.stdout + result.stderr + log_path.read_text()
    assert "L0" in combined
    assert "L4" in combined
    # Should report total + per-level counts
    assert "1" in combined  # count of L0 / count of L4


def test_cli_missing_fasta_fails_cleanly(tmp_path):
    negatives = _make_negatives_json(tmp_path, [
        {"sequence": "X", "charge": 2, "modify": [],
         "rt": 30.0, "precursor_mz": 400.0,
         "raw_title": "r", "protein_names": "X_YEAST",
         "label_type": "negative"},
    ])
    out_tsv = tmp_path / "out.tsv"

    result = _run_cli(
        "--negatives", negatives,
        "--target-fasta", str(tmp_path / "nope.fasta"),
        "--output", str(out_tsv),
    )
    assert result.returncode != 0
    err = result.stdout + result.stderr
    assert "FASTA" in err or "fasta" in err or "nope.fasta" in err


def test_cli_creates_output_parent_directory(tmp_path):
    negatives = _make_negatives_json(tmp_path, [
        {"sequence": "L0PEP", "charge": 2, "modify": [],
         "rt": 30.0, "precursor_mz": 400.0,
         "raw_title": "r", "protein_names": "X_YEAST",
         "label_type": "negative"},
    ])
    target = _make_target_fasta(tmp_path, {"p1": "L0PEPK"})
    # Note: nested directory that doesn't exist yet
    out_tsv = tmp_path / "deep" / "nested" / "classified.tsv"
    result = _run_cli(
        "--negatives", negatives,
        "--target-fasta", target,
        "--output", str(out_tsv),
    )
    assert result.returncode == 0, result.stderr
    assert out_tsv.exists()
