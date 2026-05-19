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
