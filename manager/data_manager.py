
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

    def get_centroid_params(self) -> tuple[bool, float]:
        """Resolve centroid params from config, falling back to DIAData
        module-level defaults.

        Single source of truth used by both get_dia_data_object (which
        injects them into DIAData before loading mzML) and
        workflows.flow_utils.data_to_npz (which uses them to validate
        the npz cache). Eliminates the duplication noted by the P0-3
        code review (2026-06-03).
        """
        from spectrum.dia_data import (
            DEFAULT_CENTROID_ENABLED, DEFAULT_CENTROID_REL_THRESHOLD,
        )
        if self._config is None or not self._config.has_section(ConfigKeys.GENERAL):
            return DEFAULT_CENTROID_ENABLED, DEFAULT_CENTROID_REL_THRESHOLD
        enabled = self._config.getboolean(
            ConfigKeys.GENERAL, ConfigKeys.CENTROID_ENABLED,
            fallback=DEFAULT_CENTROID_ENABLED)
        threshold = self._config.getfloat(
            ConfigKeys.GENERAL, ConfigKeys.CENTROID_REL_THRESHOLD,
            fallback=DEFAULT_CENTROID_REL_THRESHOLD)
        return enabled, threshold

    def get_dia_data_object(self, tot_raw_path: None | str = None) -> DIAData:
        """ 从路径中读取 dia 数据 """
        dia_data = DIAData()
        # P0-3 (2026-06-03 audit): single source of truth via get_centroid_params.
        dia_data._centroid_enabled, dia_data._centroid_rel_threshold = \
            self.get_centroid_params()
        dia_data._load_from_mzml(tot_raw_path)
        return dia_data
