"""Test spec_trainer feature column resolution (lightgbm-free).

After review finding I-ST1 (2026-06-03 audit), the helper is in
tools/spec_trainer/src/feature_cols.py and these tests import it
directly. Previously they tried to import all of main.py and skipped
on missing lightgbm.
"""
import os
import sys

import pytest


_PROJECT_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
_SPEC_TRAINER_SRC = os.path.join(_PROJECT_ROOT, "tools", "spec_trainer", "src")

if _SPEC_TRAINER_SRC not in sys.path:
    sys.path.insert(0, _SPEC_TRAINER_SRC)

from feature_cols import resolve_feature_cols  # noqa: E402


def test_resolve_feature_cols_explicit_list_passthrough():
    """When yaml provides explicit feature_cols list, return it unchanged."""
    result = resolve_feature_cols(
        explicit=["a", "b", "c"],
        sample_csv_paths="/nonexistent.csv",
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
        sample_csv_paths=str(csv),
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
        sample_csv_paths=str(csv),
        target_col="label",
    )
    assert result == ["precursor_pearson", "b_mean"]


def test_main_creates_model_output_directory():
    """main.py must mkdir -p the model output directory before save().

    Regression for review finding I-ST3 (2026-06-03): model.save() had no
    mkdir, so direct python invocation crashed when runs/spec_trainer/models/
    didn't exist (Makefile pre-created it, masking the bug).
    """
    src_path = os.path.join(_SPEC_TRAINER_SRC, "main.py")
    src = open(src_path).read()
    assert "os.makedirs(os.path.dirname(model_path)" in src, (
        "main.py is missing mkdir guard before model.save (I-ST3)")


def test_resolve_feature_cols_takes_intersection_of_multiple_files(tmp_path):
    """When given multiple sample CSVs, return the column intersection (P1-5, Pipeline-I5)."""
    from feature_cols import resolve_feature_cols
    csv_a = tmp_path / "a.csv"
    csv_a.write_text(
        "label,sequence,feat_common,feat_a_only\n"
    )
    csv_b = tmp_path / "b.csv"
    csv_b.write_text(
        "label,sequence,feat_common,feat_b_only\n"
    )
    result = resolve_feature_cols(
        explicit=None,
        sample_csv_paths=[str(csv_a), str(csv_b)],
        target_col="label",
    )
    # Intersection minus META: only feat_common
    assert result == ["feat_common"]
    assert "feat_a_only" not in result
    assert "feat_b_only" not in result


def test_resolve_feature_cols_single_path_backward_compat(tmp_path):
    """Calling with a single string path still works (P1-5 back-compat)."""
    from feature_cols import resolve_feature_cols
    csv = tmp_path / "x.csv"
    csv.write_text("label,sequence,feat1,feat2\n")
    # New API with list of 1
    r1 = resolve_feature_cols(explicit=None, sample_csv_paths=[str(csv)],
                               target_col="label")
    assert r1 == ["feat1", "feat2"]
    # Back-compat: bare string also accepted
    r2 = resolve_feature_cols(explicit=None, sample_csv_paths=str(csv),
                               target_col="label")
    assert r2 == ["feat1", "feat2"]


def test_resolve_feature_cols_explicit_passthrough_unchanged(tmp_path):
    """Explicit list still passes through unchanged (no regression)."""
    from feature_cols import resolve_feature_cols
    result = resolve_feature_cols(
        explicit=["a", "b"],
        sample_csv_paths="/nonexistent.csv",
        target_col="label",
    )
    assert result == ["a", "b"]


def test_resolve_feature_cols_logs_warning_on_dropped_columns(tmp_path, caplog):
    """When a file has columns absent from intersection, log a warning naming them."""
    import logging
    from feature_cols import resolve_feature_cols
    csv_a = tmp_path / "a.csv"
    csv_a.write_text("label,feat_common,feat_a_only\n")
    csv_b = tmp_path / "b.csv"
    csv_b.write_text("label,feat_common\n")
    caplog.clear()
    with caplog.at_level(logging.WARNING):
        resolve_feature_cols(
            explicit=None,
            sample_csv_paths=[str(csv_a), str(csv_b)],
            target_col="label",
        )
    # Warning message should mention the dropped column
    msgs = [r.message for r in caplog.records if r.levelno >= logging.WARNING]
    assert any("feat_a_only" in m for m in msgs), (
        f"Expected warning about feat_a_only being dropped; got: {msgs}")


def test_resolve_feature_cols_raises_when_result_empty(tmp_path):
    """Empty result must raise ValueError, not silently return []
    (P2-7, Silent-I5)."""
    from feature_cols import resolve_feature_cols
    csv = tmp_path / "empty.csv"
    # All META columns; nothing left after exclusion
    csv.write_text("label,sequence,charge,modification_count\n")
    with pytest.raises(ValueError, match="0 features"):
        resolve_feature_cols(
            explicit=None,
            sample_csv_paths=[str(csv)],
            target_col="label",
        )
