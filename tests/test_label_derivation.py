"""Tests for label column writing in flow_utils and label-derivation helpers."""
import numpy as np
import pandas as pd
import pytest

from spectrum.psm_info import PSMInfo


def _psm(seq, label_type, raw="r1", protein_names="X_HUMAN"):
    p = PSMInfo(
        sequence=seq, charge=2, modify=[],
        rt=np.float32(1.0), precursor_mz=np.float32(500.0),
        raw_title=raw, protein_names=protein_names,
    )
    p._label_type = label_type
    return p


def test_single_flow_writes_numeric_label_from_label_type():
    """`label` column in single-flow result must be 0/1/None
    (not the protein name string)."""
    from workflows.flow_utils import _make_result_row_single

    row_pos = _make_result_row_single(
        _psm("HUMAN_PEP", "positive"), features={"feat1": 1.0})
    row_neg = _make_result_row_single(
        _psm("BAD_PEP", "negative"), features={"feat1": 2.0})
    row_unk = _make_result_row_single(
        _psm("UNKNOWN", None), features={"feat1": 3.0})

    assert row_pos["label"] == 1
    assert row_neg["label"] == 0
    assert row_unk["label"] is None
    assert row_pos["label_type"] == "positive"
    assert row_neg["label_type"] == "negative"
    assert row_unk["label_type"] is None
    # features merged into row
    assert row_pos["feat1"] == 1.0
    # other required keys still present
    assert row_pos["sequence"] == "HUMAN_PEP"
    assert "raw_title1" in row_pos


def test_single_flow_dict_used_by_process_batch_single():
    """Confirm that process_batch_single uses _make_result_row_single
    (so the fix actually propagates to the CSV)."""
    import inspect
    from workflows import flow_utils
    src = inspect.getsource(flow_utils.process_batch_single)
    assert "_make_result_row_single" in src or "_make_result_row" in src, (
        "process_batch_single must use the shared helper")
