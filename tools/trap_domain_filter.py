"""tools/trap_domain_filter.py — drop trap PSMs outside the SILAC tool's
discrimination limit (spec §12).

Two filters implemented now (contaminant list = TODO, spec §12 class 2):
  - class 1  L0/L1 human-homolog: trap sequence (incl. L<->I isomer) appears
    in the human proteome -> mass-spec-indistinguishable -> drop.
    (reuses spectrum.entrapment_classifier)
  - class 3  heavy-out-of-window: heavy precursor m/z outside this raw's
    acquisition range (`heavy_out_of_range == 1`) -> SILAC channel missing
    -> drop.
  - class 4  no-label-site: peptide has no K/R -> no heavy partner
    (heavy == light) -> light/heavy validation undefined -> drop.

Positives (targets) are never dropped. Only `label_type == "negative"`
rows are evaluated. Usage:

    python -m tools.trap_domain_filter \
        --features runs/pilot_2da_speclib/features.csv \
        --human-fasta /path/to/human_swissprot.fasta \
        --output runs/pilot_2da_speclib/features.clean.csv
"""
from __future__ import annotations

import argparse
import logging
import os
import sys

import pandas as pd

_PROJECT_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
if _PROJECT_ROOT not in sys.path:
    sys.path.insert(0, _PROJECT_ROOT)

from spectrum.entrapment_classifier import classify_peptide, load_target_fasta
from spectrum.psm_info import has_label_site  # noqa: F401  (re-exported for tests)

logger = logging.getLogger(__name__)

# Entrapment levels considered mass-spec-indistinguishable from human (class 1).
HOMOLOG_DROP_LEVELS = frozenset({"L0", "L1"})


def beyond_tool_limit(level: str, heavy_out_of_range,
                      has_kr: bool = True) -> tuple[bool, str | None]:
    """Decide whether a trap PSM is beyond the SILAC tool's limit (spec §12).

    Args:
        level: entrapment level "L0"/"L1"/"L4" from entrapment_classifier.
        heavy_out_of_range: 0/1 (or bool); 1 => heavy precursor not acquired.
        has_kr: whether the peptide carries a label site (K/R). False =>
            no heavy partner => SILAC inapplicable (class 4).

    Returns:
        (drop, reason). reason is "homolog_L0"/"homolog_L1" (class 1),
        "no_label_site" (class 4), or "heavy_out_of_window" (class 3), else
        None. Precedence: class 1 (indistinguishable) > class 4 (no label
        site) > class 3 (out of window) — earlier reasons are more
        fundamental; all three drop the PSM regardless.
    """
    if level in HOMOLOG_DROP_LEVELS:
        return True, f"homolog_{level}"
    if not has_kr:
        return True, "no_label_site"
    if int(heavy_out_of_range) == 1:
        return True, "heavy_out_of_window"
    return False, None


def annotate_traps(df: pd.DataFrame, target_index) -> pd.DataFrame:
    """Add `entrap_level` / `domain_drop` / `domain_reason` columns for trap
    rows (label_type == 'negative'). Positives get level 'target', no drop."""
    out = df.copy()
    levels, drops, reasons = [], [], []
    for _, row in out.iterrows():
        if row.get("label_type") != "negative":
            levels.append("target"); drops.append(False); reasons.append(None)
            continue
        lvl = classify_peptide(str(row["sequence"]), target_index)
        hor = row.get("heavy_out_of_range", 0)
        has_kr = has_label_site(row["sequence"])
        drop, reason = beyond_tool_limit(lvl, hor, has_kr)
        levels.append(lvl); drops.append(drop); reasons.append(reason)
    out["entrap_level"] = levels
    out["domain_drop"] = drops
    out["domain_reason"] = reasons
    return out


def main():
    ap = argparse.ArgumentParser(description="Drop traps beyond the SILAC "
                                 "tool's limit (spec §12 class 1 + 3).")
    ap.add_argument("--features", required=True, help="labeled features.csv")
    ap.add_argument("--human-fasta", required=True,
                    help="human proteome FASTA (target for L0/L1)")
    ap.add_argument("--output", required=True, help="cleaned features.csv")
    args = ap.parse_args()
    logging.basicConfig(level=logging.INFO)

    df = pd.read_csv(args.features)
    target = load_target_fasta(args.human_fasta)
    ann = annotate_traps(df, target)

    traps = ann[ann.label_type == "negative"]
    dropped = traps[traps.domain_drop]
    kept = ann[~ann.domain_drop]
    logger.info("traps total=%d, dropped=%d, kept=%d",
                len(traps), int(traps.domain_drop.sum()),
                len(traps) - int(traps.domain_drop.sum()))
    logger.info("drop reasons: %s",
                dict(dropped.domain_reason.value_counts()))
    logger.info("entrap levels: %s",
                dict(traps.entrap_level.value_counts()))
    kept.to_csv(args.output, index=False)
    logger.info("wrote cleaned features (targets + kept traps): %s (%d rows)",
                args.output, len(kept))


if __name__ == "__main__":
    main()
