import os, sys
sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", "tools", "spec_trainer", "src"))
from feature_cols import EXCLUDED_EXTRA
def test_q_value_excluded():
    assert "q_value" in EXCLUDED_EXTRA
