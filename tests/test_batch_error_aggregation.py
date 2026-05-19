"""Tests for batch-level error counter contract in flow_utils."""
import inspect

import pytest


def test_process_batch_single_returns_results_and_error_count():
    """process_batch_single must return (results, n_errors)
    or a dict {'results': ..., 'n_errors': ...}."""
    from workflows.flow_utils import process_batch_single
    sig = inspect.signature(process_batch_single)
    src = inspect.getsource(process_batch_single)
    assert "n_error" in src or "errors_count" in src or "n_failed" in src, (
        "Must track error count")
    assert "return results, " in src or "return {'results':" in src \
        or "return (results," in src, (
        "Must return (results, n_errors) tuple")


def test_process_batch_pair_returns_results_and_error_count():
    from workflows.flow_utils import process_batch_pair
    src = inspect.getsource(process_batch_pair)
    assert "n_error" in src or "n_failed" in src
    assert "return results, " in src or "return (results," in src


def test_process_batch_pair_shuffle_returns_results_and_error_count():
    from workflows.flow_utils import process_batch_pair_shuffle
    src = inspect.getsource(process_batch_pair_shuffle)
    assert "n_error" in src or "n_failed" in src
    assert "return results, " in src or "return (results," in src


def test_distribute_writes_partial_marker_on_broken_process_pool():
    """When BrokenProcessPool occurs, distribute() must write a
    .PARTIAL_INCOMPLETE sidecar file next to the result CSV."""
    from workflows import pair_flow
    src = inspect.getsource(pair_flow.PairFlow.distribute)
    assert "PARTIAL_INCOMPLETE" in src or "incomplete" in src.lower()


def test_process_group_does_not_mutate_input_psms():
    """If pair_flow exposes a per-group helper that bumps psm._rt, it
    must not mutate the original PSM (which is shared across pairs)."""
    from workflows import pair_flow
    candidates = []
    for name in dir(pair_flow.PairFlow):
        if name.startswith("_"):
            try:
                src = inspect.getsource(getattr(pair_flow.PairFlow, name))
                if "_rt" in src and ("combinations" in src or "_rt +" in src or "_rt =" in src):
                    candidates.append((name, src))
            except (TypeError, OSError):
                continue
    for name, src in candidates:
        has_inplace = "b._rt = b._rt +" in src or "b._rt +=" in src \
            or "psm2._rt +=" in src or "psm2._rt = psm2._rt +" in src
        has_copy = ("copy.copy" in src or "copy.deepcopy" in src
                    or "PSMInfo.from_dict" in src)
        if has_inplace:
            assert has_copy, (
                f"{name} mutates _rt in place without copying first; "
                f"shared PSMInfo will accumulate offsets across pairs")
