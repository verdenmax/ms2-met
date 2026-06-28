"""Cross-validated training entry for spec_trainer (production LightGBM).

One CV pass yields OOF predictions that drive: honest evaluation, fold
ensemble (saved per-fold models), and label-noise audit. Does NOT touch
main.py (single-holdout flow unchanged). lightgbm is imported lazily inside
assemble_oof so this module (and path helpers) import without it.
"""
import argparse
import json
import logging
import os
import re

import numpy as np
import pandas as pd

# yaml (PyYAML) is imported lazily inside main() where the config is parsed —
# mirroring the lazy lightgbm import in assemble_oof — so this module and its
# path / IO helpers (derive_paths, read_dataframe) import cleanly even in
# minimal environments where PyYAML is not installed.

from cv_core import (average_proba, audit_labels, evaluate_oof, fnr_at_fpr5,
                     make_cv_splits)
from feature_cols import resolve_feature_cols


def read_dataframe(train_files):
    """Concatenate feature CSVs, keeping all columns (sequence/charge needed)."""
    return pd.concat([pd.read_csv(f) for f in train_files], ignore_index=True)


def derive_paths(cfg):
    """(model_prefix, result_path, suspects_path) from output.{model,result}_path."""
    model_path = cfg["output"]["model_path"]
    result_path = cfg["output"]["result_path"]
    model_prefix = re.sub(r"\.txt$", "", model_path)
    suspects_path = re.sub(r"\.json$", ".suspects.csv", result_path)
    return model_prefix, result_path, suspects_path


def _build_parser():
    p = argparse.ArgumentParser(description="CV train + ensemble + label audit")
    p.add_argument("--config", required=True)
    p.add_argument("--name", required=True)
    p.add_argument("--logpath", default="./cv_spec.log")
    return p
