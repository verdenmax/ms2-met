
import configparser
import logging
import manager.base_manager as base_manager


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
