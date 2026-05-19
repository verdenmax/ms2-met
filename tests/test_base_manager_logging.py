"""Tests for defensive logging in manager/base_manager.py."""
import logging


def test_load_corrupted_pickle_logs_traceback_and_path(tmp_path, caplog):
    """If pickle.load raises, the error log must include path and traceback."""
    from manager.base_manager import BaseManager

    bad_path = tmp_path / "corrupt.pkl"
    bad_path.write_bytes(b"\x00\x01garbage")  # invalid pickle

    # BaseManager.load is called in __init__ when load_from_file=True
    with caplog.at_level(logging.ERROR, logger=""):
        BaseManager(path=str(bad_path), load_from_file=True)

    log_text = "\n".join(r.message for r in caplog.records)
    # Path must appear so users can find the broken file
    assert str(bad_path) in log_text or "corrupt.pkl" in log_text, (
        "Error log must include the path of the broken pickle. "
        f"Got: {log_text!r}")
    # Traceback markers
    has_tb_in_text = "Traceback" in log_text
    has_exc_info = any(r.exc_info is not None for r in caplog.records)
    assert has_tb_in_text or has_exc_info, (
        "Error log must include a traceback (via logging.exception or "
        f"explicit traceback formatting). Got: {log_text!r}")
