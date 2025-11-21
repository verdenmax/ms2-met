
import configparser
import logging

from manager.base_manager import BaseManager
from spectrum.light_result import LightResult

from constant.keys import ConfigKeys


class LightResultManager(BaseManager):

    def __init__(
        self,
        config: None | configparser.ConfigParser = None,
        path: None | str = None,
        load_from_file: bool = False,
        figure_path: None | str = None,
    ):
        """从 raw 文件中加载信息"""
        self.stats = {}  # needs to be before super().__init__

        super().__init__(
            path=path, load_from_file=load_from_file, figure_path=figure_path)

        self._config: configparser.ConfigParser = config

        # 之后决定是否要存储 DIA-NN 搜索结果

        logging.info(f"初始化： {self.__class__.__name__}")

    def get_light_result_object(
        self,
        light_result_path: None | str = None,
    ) -> LightResult:
        """ 从路径中读取 dia-nn 搜索结果 """

        light_result = LightResult()

        search_engine_type = self._config[ConfigKeys.INPUT].getint(
            ConfigKeys.SEARCH_ENGINE_TYPE, fallback=1)

        if search_engine_type == 1:
            light_result._load_from_dia_nn_input(light_result_path)
        elif search_engine_type == 2:
            light_result._load_from_alphadia_input(light_result_path)
        else:
            logging.error("错误搜索引擎类型")

        return light_result
