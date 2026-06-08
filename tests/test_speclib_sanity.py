import subprocess
import sys

from workflows.pred_features import spectral_angle
from workflows.pred_store import frag_key
from tools.speclib_sanity import (
    similarity_distribution, gate_pass, build_pairs_from_maps)


def test_distribution_of_identical_pairs_is_one():
    pairs = [([1.0, 2.0, 3.0], [1.0, 2.0, 3.0]),
             ([0.5, 0.2, 0.9], [5.0, 2.0, 9.0])]
    stats = similarity_distribution(pairs, metric=spectral_angle)
    assert stats["n"] == 2
    assert abs(stats["median"] - 1.0) < 1e-9


def test_distribution_skips_nan_pairs():
    pairs = [([1.0, 2.0], [1.0, 2.0]), ([1.0], [1.0])]
    stats = similarity_distribution(pairs, metric=spectral_angle)
    assert stats["n"] == 1


def test_gate_pass_threshold():
    good = {"n": 100, "median": 0.82, "p25": 0.7, "p75": 0.9}
    bad = {"n": 100, "median": 0.4, "p25": 0.2, "p75": 0.6}
    empty = {"n": 0, "median": float("nan"), "p25": float("nan"), "p75": float("nan")}
    assert gate_pass(good, min_sim=0.7) is True
    assert gate_pass(bad, min_sim=0.7) is False
    assert gate_pass(empty, min_sim=0.7) is False


def test_cli_help_exits_zero():
    r = subprocess.run(
        [sys.executable, "-m", "tools.speclib_sanity", "--help"],
        capture_output=True, text=True)
    assert r.returncode == 0
    assert "--library-dir" in r.stdout
    assert "--min-sim" in r.stdout


def test_build_observed_pred_pairs_aligns_on_common_fragments():
    pred_map = {frag_key("b", 0, 1): 1.0, frag_key("y", 1, 1): 0.4}
    obs_map = {frag_key("y", 1, 1): 50.0, frag_key("y", 2, 1): 10.0}
    pred_vec, obs_vec = build_pairs_from_maps(pred_map, obs_map)
    assert pred_vec == [0.4]
    assert obs_vec == [50.0]
