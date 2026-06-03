"""Verify main.py reads work_directory from config (not hardcoded).

Regression for review finding I-MK2 (2026-06-03 audit): main.py
hardcoded work_path='./workspace', causing parallel make targets to
collide on the same workspace directory.
"""
import os


def test_main_uses_config_work_directory():
    """main.py source must read work_directory from the [general] config."""
    project_root = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
    src = open(os.path.join(project_root, "main.py")).read()
    # Look for the config-read pattern. Accept either configparser API or
    # the constant K.GENERAL.WORK_DIRECTORY (see constant/keys.py:28).
    assert (
        'work_directory' in src.lower() or 'WORK_DIRECTORY' in src
    ), "main.py does not appear to read work_directory from config (I-MK2)"
    # Specifically reject the old hardcoded literal as the sole source.
    # (The literal may still appear as a default fallback, so we only
    # check that the work_path argument is no longer a string literal.)
    assert 'work_path="./workspace"' not in src, (
        "main.py still uses hardcoded work_path='./workspace' literal "
        "as the sole value (I-MK2). Read from config with this as fallback.")
