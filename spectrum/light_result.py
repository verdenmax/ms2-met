
import re
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

            modifications = parse_diann_peptide_modify(row._5)

            tot_psminfo = PSMInfo(
                sequence=row._6,
                charge=row._7,
                modify=modifications,
                rt=row.RT,
                rt_start=row._40,
                rt_stop=row._41,
                precursor_mz=row._11,
                raw_title=row.Run,
            )

            self.psm_info.append(tot_psminfo)

        self.peptide_len = len(self.psm_info)

    def filtered_by_raw_title(
            self, raw_title: str
    ) -> np.ndarray[tuple[int], PSMInfo]:
        """ 过滤出不同的 raw_title """
        return np.array(
            [psm
             for psm in self.psm_info
                if psm._raw_title == raw_title])


def parse_diann_peptide_modify(sequence: str):
    """ 从DIA-NN 给出的肽段结果中解析出修饰 """

    # 修饰的结果，代表(该修饰位置，unimod id)
    modifications: [(int, int)] = []

    index = 0
    count_index = 0
    slen = len(sequence)
    while index < slen:
        if sequence[index] == '(':
            rindex = index
            while sequence[rindex] != ')':
                rindex += 1

            # 解析出unimod id
            match = re.search(r'UniMod:(\d+)', sequence[index:rindex])
            unimod_id = int(match.group(1))

            # 记录到结果中
            modifications.append((count_index, unimod_id))

            index = rindex
        else:
            count_index += 1

        index += 1

    return modifications
