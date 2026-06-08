"""LightResultManager 的搜索引擎类型分派测试。"""
import configparser
import pytest
from manager.light_result_manager import LightResultManager


def test_invalid_search_engine_type_raises():
    """非法 search_engine_type 应抛 ValueError，而非静默返回空结果导致空 CSV。"""
    cfg = configparser.ConfigParser()
    cfg["input"] = {"search_engine_type": "99"}
    mgr = LightResultManager(config=cfg)
    with pytest.raises(ValueError, match="search_engine_type"):
        mgr.get_light_result_object("dummy_path")
