import configparser
import logging
import os
import numpy as np
import pandas as pd

from concurrent.futures import ProcessPoolExecutor, as_completed
from rich.progress import Progress

import workflows.flow_utils as flow_utils
import manager.data_manager as data_manager

from workflows.single_work import single_pair_work
from manager.light_result_manager import LightResultManager
from spectrum.psm_info import PSMInfo

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
    LIGHT_RESULT_MANAGER_PUCKEL = "light_result_manager.pkl"

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
        # 读取light result
        self._light_result_manager = LightResultManager(
            self._config,
            path=os.path.join(
                self._workpath, self.LIGHT_RESULT_MANAGER_PUCKEL),
        )

        light_result_path = (
            self._config[ConfigKeys.INPUT][ConfigKeys.LIGHT_RESULT_PATH])
        self._light_result = self._light_result_manager.get_light_result_object(
            light_result_path)

        # 从配置文件中加载所需信息
        self._raw_file_manager = data_manager.DataManager(
            self._config,
            path=os.path.join(self._workpath, self.RAW_DATA_MANAGER_PICKLE),
        )
        self._raw_file_manager.save()

    def multi_handle(
        self,
        raw_file_path: str,
    ):
        """ 进行多线程地处理 """
        # 获取当前 file name
        raw_file_name = flow_utils.get_filename_stem(raw_file_path)

        # 从 self._light_result 中获得所有该文件数据
        light_PSM_infos: np.ndarray[tuple(int), PSMInfo] = (
            self._light_result.filtered_by_raw_title(raw_file_name))

        # 获取 dia 数据，当之后想要多进程读数据时，可以直接将 multi_handle 多进程即可_
        dia_data = self._raw_file_manager.get_dia_data_object(raw_file_path)

        with Progress() as progress:
            rich_task_progress = progress.add_task(
                f"[cyan] 处理文件{raw_file_name} ...", total=len(light_PSM_infos))

            ans = []

            # 遍历每一个psm 信息
            for psminfo in light_PSM_infos:
                # TODO: 计算出信息
                psm, ms2_count = single_pair_work(
                    psm=psminfo,
                    dia_data=dia_data,
                    config=self._config,
                )

                ans.append({
                    "sequence": psm._sequence,
                    "charge": psm._charge,
                    "precursor_mz": psm._precursor_mz,
                    "raw_title": psm._raw_title,
                    "ms2_count": ms2_count
                })

                progress.update(rich_task_progress, advance=1)

        return ans

    def distribute(self):
        # 处理每一个任务
        # 对于每一个文件，需要传递给他一个质谱数据、一个输入数据、config
        raw_file_nums = self._config.getint(
            ConfigKeys.INPUT, ConfigKeys.RAW_NUM, fallback=1)

        ans = []

        # 进行多进程
        with ProcessPoolExecutor(max_workers=25) as executor:
            futures = []

            for i in range(raw_file_nums):
                tot_raw_path_key = f"{ConfigKeys.RAW_PATH}_{i + 1}"

                # 读取配置文件中的 RAW PATH
                tot_raw_path = self._config[ConfigKeys.INPUT][tot_raw_path_key]

                futures.append(executor.submit(
                    self.multi_handle, tot_raw_path))

            for future in as_completed(futures):
                res = future.result()

                # 接收 light_PSM_infos 和 rawfiledata，进行处理
                ans.extend(res)

        # NOTE: 保存结果

        ans_df = pd.DataFrame(ans)

        ans_df.to_csv("result.csv", sep=',', index=False)

    def run(self) -> None:
        logging.info(f"运行任务 {self.workname}")

        # 加载DIA-NN 结果，加载 Data manager
        self.load()

        # TODO: 根据不同的谱图标题，分配到不同的任务，多进程执行
        # 分配不同的进程运行
        self.distribute()

        # TODO: self.save()
