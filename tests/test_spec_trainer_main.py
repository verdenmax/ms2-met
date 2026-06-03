"""Test spec_trainer feature column resolution (lightgbm-free).

After review finding I-ST1 (2026-06-03 audit), the helper is in
tools/spec_trainer/src/feature_cols.py and these tests import it
directly. Previously they tried to import all of main.py and skipped
on missing lightgbm.
"""
import os
import sys


_PROJECT_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
_SPEC_TRAINER_SRC = os.path.join(_PROJECT_ROOT, "tools", "spec_trainer", "src")

if _SPEC_TRAINER_SRC not in sys.path:
    sys.path.insert(0, _SPEC_TRAINER_SRC)

from feature_cols import resolve_feature_cols  # noqa: E402


def test_resolve_feature_cols_explicit_list_passthrough():
    """When yaml provides explicit feature_cols list, return it unchanged."""
    result = resolve_feature_cols(
        explicit=["a", "b", "c"],
        sample_csv_path="/nonexistent.csv",
        target_col="label",
    )
    assert result == ["a", "b", "c"]


def test_resolve_feature_cols_empty_triggers_auto_detect(tmp_path):
    """Empty feature_cols triggers auto-detection from CSV header."""
    csv = tmp_path / "fake.csv"
    csv.write_text(
        "sequence,charge,protein_names,label,precursor_mz,sequence_len,"
        "raw_title1,raw_title2,label_type,modification_count,"
        "precursor_pearson,b_mean,y_p50\n"
    )
    result = resolve_feature_cols(
        explicit=[],
        sample_csv_path=str(csv),
        target_col="label",
    )
    assert result == ["precursor_pearson", "b_mean", "y_p50"]
    assert "label" not in result
    assert "modification_count" not in result
    assert "precursor_mz" not in result
    assert "sequence_len" not in result
    assert "raw_title1" not in result
    assert "protein_names" not in result


def test_resolve_feature_cols_none_triggers_auto_detect(tmp_path):
    """None feature_cols (yaml missing key) also triggers auto-detection."""
    csv = tmp_path / "fake.csv"
    csv.write_text(
        "label,precursor_pearson,b_mean,modification_count\n"
    )
    result = resolve_feature_cols(
        explicit=None,
        sample_csv_path=str(csv),
        target_col="label",
    )
    assert result == ["precursor_pearson", "b_mean"]
