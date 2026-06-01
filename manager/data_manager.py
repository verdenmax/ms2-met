
import configparser
import logging

import manager.base_manager as base_manager
from spectrum.dia_data import DIAData
from constant.keys import ConfigKeys


class DataManager(base_manager.BaseManager):

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

        # 之后决定是否要存储 DIA 文件实例，如果存储就放在 self 下

        logging.info(f"初始化： {self.__class__.__name__}")

    def get_dia_data_object(self, tot_raw_path: None | str = None) -> DIAData:
        """ 从路径中读取 dia 数据 """

        dia_data = DIAData()

        # 注入 centroid 配置（spec 2026-06-01-mzml-centroiding-on-load §5.3）
        # 不存在时退回 DIAData 默认值 (True / 1e-3)
        if self._config is not None and self._config.has_section(
                ConfigKeys.GENERAL):
            dia_data._centroid_enabled = self._config.getboolean(
                ConfigKeys.GENERAL, ConfigKeys.CENTROID_ENABLED,
                fallback=dia_data._centroid_enabled)
            dia_data._centroid_rel_threshold = self._config.getfloat(
                ConfigKeys.GENERAL, ConfigKeys.CENTROID_REL_THRESHOLD,
                fallback=dia_data._centroid_rel_threshold)

        dia_data._load_from_mzml(tot_raw_path)

        return dia_data
