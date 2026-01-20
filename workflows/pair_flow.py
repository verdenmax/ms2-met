import configparser
import logging
import os
import numpy as np
import pandas as pd
from collections import defaultdict
from itertools import combinations


from concurrent.futures import ProcessPoolExecutor, as_completed
from rich.progress import Progress

import workflows.flow_utils as flow_utils
import manager.data_manager as data_manager

from workflows.flow_utils import data_to_npz, process_psm_pair_shared, process_psm_single
from workflows.single_work import multi_batch_work
from manager.light_result_manager import LightResultManager
from spectrum.psm_info import PSMInfo

from constant.keys import ConfigKeys


class PairFlow:
    """
    轻重标匹配的工作流
    这个工作流将会完成
    1. 质谱数据文件读取，通过 data_manager
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

        self._nametoDATA = {}

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
        psm1: PSMInfo,
        psm2: PSMInfo,
        label: int,
    ):
        """ 进行多线程地处理 """

        # 获取 dia 数据，当之后想要多进程读数据时，可以直接将 multi_handle 多进程即可_
        # dia_data = self._raw_file_manager.get_dia_data_object(raw_file_path)
        light_dia_data = self._nametoDATA[psm1._raw_title]
        heavy_dia_data = self._nametoDATA[psm2._raw_title]

        # TODO: 计算出信息
        tot_features = multi_batch_work(
            psm1=psm1,
            dia_data1=light_dia_data,
            psm2=psm2,
            dia_data2=heavy_dia_data,
            config=self._config,
        )

        return {
            "sequence": psm1._sequence,
            "charge": psm1._charge,
            "precursor_mz": psm1._precursor_mz,
            "raw_title1": psm1._raw_title,
            "raw_title2": psm2._raw_title,
            "protein_names": psm1._protein_names,
            "sequence_len": len(psm1._sequence),
            "label": label,
            ** tot_features
        }

    def pharse_data(self, tot_raw_path: str):
        """ 从 manager 中解析出数据"""
        tot_raw_file_name = flow_utils.get_filename_stem(tot_raw_path)

        tot_dia_data = self._raw_file_manager.get_dia_data_object(
            tot_raw_path)

        return tot_raw_file_name, tot_dia_data

    def _process_group(self, group):
        ans = []

        # 这个其实是在重复样本中找到重复出现的
        for a, b in combinations(group, 2):
            # label 设置为 1,记为正样本
            res = self.multi_handle(a, b, 1)
            ans.append(res)

        # 在重复样本中，找到负样本
        for a, b in combinations(group, 2):
            # label 设置为 0,记为负样本
            # 这里对 b 进行一个小处理就行 ，其实就是对b 的 时间稍微做一个偏移

            b._rt = b._rt + 10
            res = self.multi_handle(a, b, 0)
            ans.append(res)

        return ans

    def distribute(self):
        # 处理每一个任务
        # 对于每一个文件，需要传递给他一个质谱数据、一个输入数据、config
        raw_file_nums = self._config.getint(
            ConfigKeys.INPUT, ConfigKeys.RAW_NUM, fallback=1)

        # 进行多进程
        with ProcessPoolExecutor(
                max_workers=min(raw_file_nums, 25)) as executor:

            futures = []

            # 记录所有 shared 文件对应信息，就是 name 对应的 shared_path
            shared_files = []
            for i in range(raw_file_nums):
                tot_raw_path_key = f"{ConfigKeys.RAW_PATH}_{i + 1}"

                # 读取配置文件中的 RAW PATH
                tot_raw_path = self._config[ConfigKeys.INPUT][tot_raw_path_key]

                # 分发任务，去分配不同的进程进行读取 mz 数据
                futures.append(executor.submit(
                    data_to_npz,
                    self._raw_file_manager, tot_raw_path, self._workpath))

            for future in as_completed(futures):
                tot_raw_file_name, shared_path = future.result()
                shared_files.append((tot_raw_file_name, shared_path))

            name_to_shared = {name: path for name, path in shared_files}

        logging.info("开始psm 任务分配")
        # NOTE: 处理信息，让 psm 信息两两分组
        psm_groups = defaultdict(list)

        # 对每一个 psm ，都映射他们的 key 到一个相同的 groups
        for psm in self._light_result.psm_info:
            key = PSMInfo.get_key(psm)
            psm_groups[key].append(psm)

        tasks = []

        # 不同的类型
        if self._config.getint(
                ConfigKeys.GENERAL,
                ConfigKeys.FEATURE_TYPE, fallback=0) == 0:

            for group in psm_groups.values():
                for a in group:
                    tasks.append(
                        (a.to_dict(),
                         name_to_shared[a._raw_title]))

        else:
            # 两两之间进行处理任务
            for group in psm_groups.values():
                for a, b in combinations(group, 2):
                    # 添加正样本
                    tasks.append(
                        (a.to_dict(), b.to_dict(),
                         name_to_shared[a._raw_title],
                         name_to_shared[b._raw_title],
                         1))

                    # 添加负样本
                    tasks.append(
                        (a.to_dict(), b.to_dict(),
                         name_to_shared[a._raw_title],
                         name_to_shared[b._raw_title],
                         0))

        # 进行计算
        ans = []
        with ProcessPoolExecutor(max_workers=25) as executor:

            if self._config.getint(
                    ConfigKeys.GENERAL,
                    ConfigKeys.FEATURE_TYPE, fallback=0) == 0:
                multi_sample_futures = [
                    executor.submit(
                        process_psm_single,
                        psm1_dict, shared1, self._config
                    )
                    for (psm1_dict, shared1) in tasks
                ]
            else:
                multi_sample_futures = [
                    executor.submit(
                        process_psm_pair_shared,
                        psm1_dict, psm2_dict, shared1, shared2, self._config, label
                    )
                    for (psm1_dict, psm2_dict, shared1, shared2, label) in tasks
                ]

            with Progress() as progress:
                rich_task_progress = progress.add_task(
                    "[cyan] 处理进度 ...", total=len(multi_sample_futures))

                # 收集结果（as_completed 可尽早获取已完成任务）
                for future in as_completed(multi_sample_futures):
                    progress.update(rich_task_progress, advance=1)
                    try:
                        tot_results = future.result()
                        ans.append(tot_results)  # 主线程合并，线程安全 ✅
                    except Exception as e:
                        # 建议记录日志，不要静默失败
                        logging.info(f"Error in group processing: {e}")

        # NOTE: 保存结果
        ans_df = pd.DataFrame(ans)

        result_file = self._config.get(
            ConfigKeys.GENERAL, ConfigKeys.RESULT_FILE,
            fallback="result.csv")

        logging.info(f"保存结果文件 {result_file}")
        ans_df.to_csv(result_file, sep=',', index=False)

    def run(self) -> None:
        logging.info(f"运行任务 {self.workname}")

        # 加载DIA-NN 结果，加载 Data manager
        self.load()

        # TODO: 根据不同的谱图标题，分配到不同的任务，多进程执行
        # 分配不同的进程运行
        self.distribute()

        # TODO: self.save()
