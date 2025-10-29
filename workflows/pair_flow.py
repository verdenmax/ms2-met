import configparser
import logging
import os

import manager.data_manager as data_manager

from constant.keys import ConfigKeys


class PairFlow:
    """
    轻重标匹配的工作流
    这个工作流将会完成
    1. 质谱数据文件读取，通过 data_manager
    TODO: 
    2. 搜索结果文件读取，需要支持多种格式
    3. 取出每一条搜索结果，将其在谱图中找到对应的轻重标配对
    4. 得到通过检验的结果，存储到文件中
    """

    RAW_DATA_MANAGER_PICKLE = "raw_data_manager.pkl"

    def __init__(
        self,
        workname: str,
        config: None | configparser.ConfigParser = None,
        work_path: str = "./Pairworkspace",
    ) -> None:
        """
        初始化工作流
        """
        self.workname: str = workname

        self._config: configparser.ConfigParser = config

        self._workpath: str = work_path

        # 创建不存在的目录
        for path in [self._workpath]:
            if path and not os.path.exists(path):
                logging.info(f"Creating folder {path}")

                os.makedirs(
                    path,
                    exist_ok=True,
                )

    def load(
        self
    ) -> None:
        # 从配置文件中加载所需信息
        raw_file_manager = data_manager.DataManager(
            self._config,
            path=os.path.join(self._workpath, self.RAW_DATA_MANAGER_PICKLE),
        )

        # 处理每一个任务
        # 对于每一个文件，需要传递给他一个质谱数据、一个输入数据、config
        raw_file_nums = self._config.getint(
            ConfigKeys.INPUT, ConfigKeys.RAW_NUM, fallback=1)

        for i in range(raw_file_nums):
            tot_raw_path_key = f"{ConfigKeys.RAW_PATH}_{i + 1}"

            # 读取配置文件中的 RAW PATH
            tot_raw_path = self._config[ConfigKeys.INPUT][tot_raw_path_key]

            self._dia_data = raw_file_manager.get_dia_data_object(tot_raw_path)

            logging.info(tot_raw_path)

        # TODO: 这里还需
        # self._dia_data = raw_file_manager.get_dia_data_object(dia_data_path)

        raw_file_manager.save()

    def run(self) -> None:
        logging.info(f"运行任务 {self.workname}")

        self.load()

        # TODO: 根据不同的谱图标题，分配到不同的任务，多进程执行

        # TODO: self.save()
