"""tools/entrapment_classify.py — classify negative PSMs as L0/L1/L4.

Given a negatives JSON (from tools/extract_common.py) and a target
proteome FASTA (e.g. HUMAN SwissProt), produce a classified.tsv
compatible with tools/extract_common.load_entrapment_classifications.

This removes the proteinCopilot dependency for L0/L1 filtering:
ms2-met can now build its own clean negative set in a single command.

L0/L1/L4 semantics:
  - L0 razor-error:  trap stripped sequence is a substring of target proteome
  - L1 LI-isomer:    trap (L↔I-normalized) is a substring of target (L↔I-normalized)
  - L4 true-trap:    neither (we don't compute L2/L3 here — Hamming scans omitted)

Usage:
    python -m tools.entrapment_classify \\
        --negatives datasets/hela_2da_pfind_diann.json \\
        --target-fasta /path/to/human_swissprot.fasta \\
        --output datasets/entrapment_classified.tsv

Then add to extract_common config:
    [entrapment]
    classified_tsv = ./datasets/entrapment_classified.tsv
    drop_levels = L0, L1
"""
from __future__ import annotations

import argparse
import csv
import json
import logging
import os
import sys
from collections import Counter

_PROJECT_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
if _PROJECT_ROOT not in sys.path:
    sys.path.insert(0, _PROJECT_ROOT)

from spectrum.entrapment_classifier import (
    classify_peptide,
    load_target_fasta,
)


def classify_negatives_file(
    negatives_path: str,
    target_fasta_path: str,
    output_path: str,
) -> dict:
    """Read negatives JSON, classify each, write TSV.

    Args:
        negatives_path: JSON file produced by tools/extract_common.py
            (a list of PSM dicts; only entries with
            label_type=="negative" are classified)
        target_fasta_path: target proteome FASTA
        output_path: where to write the TSV (schema compatible with
            tools/extract_common.load_entrapment_classifications)

    Returns:
        dict with counters: total_negatives, classified, level
        distribution.
    """
    if not os.path.exists(negatives_path):
        raise FileNotFoundError(
            f"negatives JSON 不存在: '{negatives_path}'")

    target = load_target_fasta(target_fasta_path)

    with open(negatives_path, "r", encoding="utf-8") as f:
        psms = json.load(f)

    n_total = len(psms)
    n_skipped_non_negative = 0
    n_skipped_no_sequence = 0
    n_classified = 0

    output_dir = os.path.dirname(output_path)
    if output_dir:
        os.makedirs(output_dir, exist_ok=True)

    level_counter: Counter = Counter()

    header = [
        "peptide", "charge", "precursor_mz", "retention_time",
        "scan_number", "spectrum_file", "protein_ids", "q_value",
        "group", "level",
    ]
    with open(output_path, "w", encoding="utf-8", newline="") as out_f:
        writer = csv.writer(out_f, delimiter="\t")
        writer.writerow(header)
        for psm in psms:
            if psm.get("label_type") != "negative":
                n_skipped_non_negative += 1
                continue
            seq = psm.get("sequence", "")
            if not seq:
                n_skipped_no_sequence += 1
                continue

            level = classify_peptide(seq, target)
            level_counter[level] += 1
            n_classified += 1

            writer.writerow([
                seq,
                psm.get("charge", ""),
                psm.get("precursor_mz", ""),
                psm.get("rt", ""),
                "",  # scan_number: not used by ms2-met loader
                psm.get("raw_title", ""),
                psm.get("protein_names", ""),
                psm.get("q_value", ""),
                "trap",  # group: required by extract_common loader filter
                level,
            ])

    summary = {
        "total_psms_in_input": n_total,
        "classified_as_negative": n_classified,
        "skipped_non_negative": n_skipped_non_negative,
        "skipped_no_sequence": n_skipped_no_sequence,
        "level_distribution": dict(level_counter),
        "target_proteome": {
            "fasta": target_fasta_path,
            "n_proteins": target.n_proteins,
            "total_aa": len(target.raw_text),
        },
    }
    logging.info(
        f"entrapment classify 完成: 输入={n_total}, "
        f"classified={n_classified}, "
        f"skipped_non_negative={n_skipped_non_negative}, "
        f"skipped_no_sequence={n_skipped_no_sequence}"
    )
    logging.info(
        f"  Level 分布: " + ", ".join(
            f"{lvl}={n}"
            for lvl, n in sorted(level_counter.items())
        )
    )
    return summary


def main():
    parser = argparse.ArgumentParser(
        description=(
            "Classify negative PSMs as L0/L1/L4 against a target FASTA. "
            "Output TSV is compatible with extract_common's "
            "[entrapment] classified_tsv field."
        )
    )
    parser.add_argument(
        "--negatives", required=True,
        help="negatives JSON from tools/extract_common.py"
    )
    parser.add_argument(
        "--target-fasta", required=True,
        help="target proteome FASTA (e.g. HUMAN SwissProt)"
    )
    parser.add_argument(
        "--output", required=True,
        help="output classified.tsv path"
    )
    parser.add_argument(
        "--logpath", default=None,
        help="optional log file path (also written to stderr via RichHandler)"
    )
    args = parser.parse_args()

    handlers: list = [logging.StreamHandler(sys.stderr)]
    if args.logpath:
        log_dir = os.path.dirname(args.logpath)
        if log_dir:
            os.makedirs(log_dir, exist_ok=True)
        handlers.append(logging.FileHandler(args.logpath, encoding="utf-8"))
    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s | %(levelname)s | %(message)s",
        handlers=handlers,
    )

    try:
        summary = classify_negatives_file(
            args.negatives, args.target_fasta, args.output)
    except FileNotFoundError as e:
        logging.error(str(e))
        sys.exit(1)
    except Exception as e:
        logging.exception(f"分类失败: {e}")
        sys.exit(1)

    print(json.dumps(summary, indent=2, ensure_ascii=False))


if __name__ == "__main__":
    main()
