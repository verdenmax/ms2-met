
import os
import numpy as np
import pandas as pd
import logging

from spectrum.psm_info import PSMInfo


class LightResult:
    """ 存储各种搜索引擎搜索得到的轻标结果 """

    def __init__(self):
        """ 初始化 """

        self.peptide_len: np.int64 = 0

        self.psm_info: np.ndarray[tuple[int], PSMInfo] = []

    def _load_from_dia_nn_input(self, light_result_path: str):
        """ 输入diann 的搜索结果 """

        if light_result_path is None or not os.path.exists(light_result_path):
            logging.error("dia_nn 搜索结果 report.parquet 不存在")

        # 正在加载文件
        logging.info(f"正在加载 DIA-NN report: {light_result_path}")

        light_input = pd.read_parquet(light_result_path)

        for row in light_input.itertuples():

            tot_psminfo = PSMInfo(
                sequence=row._6,
                charge=row._7,
                modify=None,
                rt=row.RT,
                precursor_mz=row._11,
                raw_title=row.Run,
            )

            self.psm_info.append(tot_psminfo)

        self.peptide_len = len(self.psm_info)
