"""Post-feature-extraction filters applied to the assembled features table.

These run after the per-PSM XIC features have been computed (so DIA-derived
columns like ``heavy_out_of_range`` already exist), but before the features
CSV is written and consumed by downstream training/eval.

Currently one filter:

  - ``filter_heavy_out_of_range``: drops PSMs whose expected heavy precursor m/z
    fell outside the raw's acquisition range (``heavy_out_of_range == 1``).
    Such PSMs have no acquired heavy channel, so they cannot be validated by
    SILAC fragment evidence. They are dropped for BOTH classes (positive and
    negative) — symmetric, to avoid teaching the model a spurious
    "out-of-range => positive" rule (which would happen if only the trap
    negatives were dropped while the out-of-range positives were kept).
"""
from __future__ import annotations

import logging

import pandas as pd

logger = logging.getLogger(__name__)


def filter_heavy_out_of_range(df: pd.DataFrame):
    """Drop rows with ``heavy_out_of_range == 1`` (both classes).

    Args:
        df: assembled per-PSM features table. May or may not carry the
            ``heavy_out_of_range`` column (e.g. historical feature tables).

    Returns:
        ``(kept_df, n_pos_dropped, n_neg_dropped)``. ``kept_df`` has a reset
        index. If the column is absent the input is returned unchanged with
        zero drop counts. Class is read from ``label_type`` when present,
        else from ``label`` (1 = positive, 0 = negative).
    """
    if "heavy_out_of_range" not in df.columns:
        return df, 0, 0

    # Robust to int/float/bool/str encodings (CSV round-trips may stringify).
    vals = pd.to_numeric(df["heavy_out_of_range"], errors="coerce")
    out_mask = vals == 1

    if "label_type" in df.columns:
        n_pos = int((out_mask & (df["label_type"] == "positive")).sum())
        n_neg = int((out_mask & (df["label_type"] == "negative")).sum())
    elif "label" in df.columns:
        n_pos = int((out_mask & (df["label"] == 1)).sum())
        n_neg = int((out_mask & (df["label"] == 0)).sum())
    else:
        n_pos = int(out_mask.sum())
        n_neg = 0

    kept = df.loc[~out_mask].reset_index(drop=True)
    return kept, n_pos, n_neg


def filter_csv_file(path: str, output: str | None = None,
                    in_place: bool = False, make_backup: bool = True):
    """Apply ``filter_heavy_out_of_range`` to a features CSV on disk.

    Args:
        path: input features.csv.
        output: write result here (mutually exclusive with in_place).
        in_place: overwrite ``path`` (writes ``path + '.prefilter.bak'`` first
            unless make_backup is False).
        make_backup: whether to back up before in-place overwrite.

    Returns:
        ``(n_pos_dropped, n_neg_dropped)``. With neither ``output`` nor
        ``in_place`` it is a dry run (counts only, nothing written).
    """
    import shutil

    df = pd.read_csv(path)
    kept, n_pos, n_neg = filter_heavy_out_of_range(df)
    mode = "dry-run" if not (output or in_place) else "write"
    logger.info(
        "heavy_out_of_range 过滤(%s) %s: 剔除 positive=%d, negative=%d, "
        "输出=%d/%d", mode, path, n_pos, n_neg, len(kept), len(df))
    if output:
        kept.to_csv(output, index=False)
    elif in_place:
        if make_backup:
            shutil.copy2(path, path + ".prefilter.bak")
        kept.to_csv(path, index=False)
    return n_pos, n_neg


def main(argv=None):
    import argparse

    ap = argparse.ArgumentParser(
        description="Drop heavy-out-of-range PSMs (both classes) from one or "
                    "more features.csv. Default is a dry run (counts only).")
    ap.add_argument("features", nargs="+", help="features.csv path(s)")
    ap.add_argument("--output", help="write result here (single input only)")
    ap.add_argument("--in-place", action="store_true",
                    help="overwrite each input (writes a .prefilter.bak first)")
    ap.add_argument("--no-backup", action="store_true",
                    help="skip the .prefilter.bak backup with --in-place")
    args = ap.parse_args(argv)

    logging.basicConfig(level=logging.INFO,
                        format="%(asctime)s | %(levelname)s | %(message)s")
    if args.output and (len(args.features) != 1 or args.in_place):
        ap.error("--output requires exactly one input and excludes --in-place")

    tot_pos = tot_neg = 0
    for path in args.features:
        np_, nn = filter_csv_file(
            path, output=args.output, in_place=args.in_place,
            make_backup=not args.no_backup)
        tot_pos += np_
        tot_neg += nn
    logger.info("总计剔除: positive=%d, negative=%d (文件数=%d)",
                tot_pos, tot_neg, len(args.features))


if __name__ == "__main__":
    main()
