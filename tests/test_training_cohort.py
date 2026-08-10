import sys
from pathlib import Path

import pandas as pd
import pytest


_ROOT = Path(__file__).resolve().parents[1]
_SRC = _ROOT / "tools" / "spec_trainer" / "src"
if str(_SRC) not in sys.path:
    sys.path.insert(0, str(_SRC))

from cohort import apply_training_cohort  # noqa: E402


def _rows():
    base = {
        "heavy_in_raw": 1,
        "heavy_out_of_range": 0,
        "precursor_xic_empty": 0,
        "q1a_valid": 1,
        "has_lib_pred": 1,
        "isotope_model_valid": 1,
    }
    rows = []
    for label in (1, 0):
        row = dict(base, label=label)
        rows.append(row)
    failed = dict(base, label=0, has_lib_pred=0)
    rows.append(failed)
    return pd.DataFrame(rows)


def test_evidence_common_filters_and_reports_class_counts():
    filtered, audit = apply_training_cohort(
        _rows(), "evidence_common", target_col="label")

    assert filtered["label"].tolist() == [1, 0]
    assert audit["name"] == "evidence_common"
    assert audit["before"] == {
        "n_rows": 3, "n_correct": 1, "n_error": 2}
    assert audit["after"] == {
        "n_rows": 2, "n_correct": 1, "n_error": 1}
    assert audit["failed_by_rule"]["has_lib_pred == 1"] == 1


def test_none_cohort_preserves_rows():
    df = _rows()
    filtered, audit = apply_training_cohort(df, None, target_col="label")
    assert filtered.equals(df.reset_index(drop=True))
    assert audit["name"] == "none"


def test_evidence_common_rejects_missing_eligibility_column():
    with pytest.raises(ValueError, match="has_lib_pred"):
        apply_training_cohort(
            _rows().drop(columns="has_lib_pred"),
            "evidence_common", target_col="label")


def test_cohort_rejects_unknown_name():
    with pytest.raises(ValueError, match="unknown cohort"):
        apply_training_cohort(_rows(), "not_a_cohort", target_col="label")
