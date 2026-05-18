"""extract_common 工具端到端集成测试。"""
import configparser
import json
import os
import pytest

from spectrum.psm_info import PSMInfo
from tools.extract_common import (
    extract_n_engines, write_psms_to_json, load_engine_psms,
)


def test_load_engine_pfind(tmp_path, sample_pfind_file):
    """load_engine_psms 应能加载 pfind 引擎。"""
    config = configparser.ConfigParser()
    config["engine.pfind"] = {
        "path": sample_pfind_file,
        "qvalue_threshold": "0.01",
    }
    psms = load_engine_psms("pfind", config)
    assert len(psms) == 3


def test_extract_single_engine_with_marker(tmp_path, sample_pfind_file):
    """单引擎 + marker 模式：自交集 = 自己，按 marker 分正负。"""
    config = configparser.ConfigParser()
    config["extract"] = {
        "engines": "pfind",
        "positive_species_marker": "HUMAN",
        "result_file": str(tmp_path / "out.json"),
    }
    config["engine.pfind"] = {
        "path": sample_pfind_file,
        "qvalue_threshold": "0.01",
    }
    psms = extract_n_engines(config)
    pos = [p for p in psms if p._label_type == "positive"]
    neg = [p for p in psms if p._label_type == "negative"]
    assert len(pos) == 2
    assert len(neg) == 1


def test_extract_to_json_roundtrip(tmp_path, sample_pfind_file):
    """写出 JSON 后再读回，PSM 数应一致且 label_type 保留。"""
    config = configparser.ConfigParser()
    output = str(tmp_path / "out.json")
    config["extract"] = {
        "engines": "pfind",
        "positive_species_marker": "HUMAN",
        "result_file": output,
    }
    config["engine.pfind"] = {
        "path": sample_pfind_file,
        "qvalue_threshold": "0.01",
    }
    psms = extract_n_engines(config)
    write_psms_to_json(psms, output)

    with open(output) as f:
        data = json.load(f)

    reconstructed = [PSMInfo.from_dict(d) for d in data]
    assert len(reconstructed) == len(psms)
    for p in reconstructed:
        assert p._label_type in ("positive", "negative")
        assert p._q_value is not None
