"""测试 LightResult 对 pfind 输入的支持。"""
import configparser
import pytest
from spectrum.light_result import LightResult


def test_load_from_pfind_input_file(sample_pfind_file):
    """LightResult._load_from_pfind_input 加载单文件。"""
    lr = LightResult()
    lr._load_from_pfind_input(sample_pfind_file, qvalue_threshold=0.01)
    assert lr.peptide_len == 3
    assert len(lr.psm_info) == 3


def test_load_from_pfind_input_directory(sample_pfind_dir):
    """LightResult._load_from_pfind_input 加载目录。"""
    lr = LightResult()
    lr._load_from_pfind_input(sample_pfind_dir, qvalue_threshold=0.01)
    assert lr.peptide_len == 3


def test_load_from_pfind_input_psm_has_q_value(sample_pfind_file):
    """加载后 PSM 应携带 q_value 字段。"""
    lr = LightResult()
    lr._load_from_pfind_input(sample_pfind_file, qvalue_threshold=0.01)
    for psm in lr.psm_info:
        assert psm._q_value is not None


def test_light_result_manager_dispatch_pfind(sample_pfind_file):
    """LightResultManager 应能根据 search_engine_type=3 分发到 pfind loader。"""
    from manager.light_result_manager import LightResultManager
    from constant.keys import ConfigKeys

    config = configparser.ConfigParser()
    config[ConfigKeys.INPUT] = {
        ConfigKeys.LIGHT_RESULT_PATH: sample_pfind_file,
        ConfigKeys.SEARCH_ENGINE_TYPE: "3",
        ConfigKeys.PFIND_QVALUE_THRESHOLD: "0.01",
    }

    mgr = LightResultManager(config=config, path=None, load_from_file=False)
    lr = mgr.get_light_result_object(sample_pfind_file)
    assert len(lr.psm_info) == 3
