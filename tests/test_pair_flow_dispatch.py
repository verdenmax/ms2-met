"""PairFlow 任务分发健壮性测试：raw_path 缺失 / raw_title 不匹配。"""
import configparser
import numpy as np
import pytest
from workflows.pair_flow import PairFlow
from spectrum.psm_info import PSMInfo


def test_resolve_raw_paths_missing_raises():
    cfg = configparser.ConfigParser()
    cfg["input"] = {"raw_path_1": "a.mzML"}   # raw_num=2 但只配了 1 个
    with pytest.raises(ValueError, match="raw_path_2"):
        PairFlow._resolve_raw_paths(cfg, 2)


def test_resolve_raw_paths_ok():
    cfg = configparser.ConfigParser()
    cfg["input"] = {"raw_path_1": "a.mzML", "raw_path_2": "b.mzML"}
    assert PairFlow._resolve_raw_paths(cfg, 2) == ["a.mzML", "b.mzML"]


def test_resolve_raw_paths_expands_home(monkeypatch):
    monkeypatch.setenv("HOME", "/home/u")
    cfg = configparser.ConfigParser()
    cfg["input"] = {"raw_path_1": "~/share/x.mzML", "raw_path_2": "/abs/y.mzML"}
    # ~ expands; absolute path is left untouched (expanduser is idempotent).
    assert PairFlow._resolve_raw_paths(cfg, 2) == [
        "/home/u/share/x.mzML", "/abs/y.mzML"]


def _psm(seq, raw):
    return PSMInfo(seq, 2, [], np.float32(30.0), np.float32(500.0), raw, "P")


def test_build_raw_tasks_skips_unknown_raw_title():
    p_ok = _psm("PEPTIDEK", "raw1")
    p_bad = _psm("SAMPLER", "rawX")          # rawX 不在配置里
    psm_groups = {
        ("PEPTIDEK", 2, ()): [p_ok],
        ("SAMPLER", 2, ()): [p_bad],
    }
    name_to_shared = {"raw1": "/tmp/raw1.npz"}
    tasks, n_skipped = PairFlow._build_raw_tasks(psm_groups, name_to_shared, 0)
    assert n_skipped == 1
    assert len(tasks) == 1
    assert tasks[0][1] == "/tmp/raw1.npz"
