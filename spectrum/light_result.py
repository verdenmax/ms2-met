
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

    def _load_from_alphadia_input(self, light_result_path: str):
        """ 输入 alphadia 的搜索结果 """

        if light_result_path is None or not os.path.exists(light_result_path):
            logging.error("Alphadia 搜索结果 report.parquet 不存在")

        # 正在加载文件
        logging.info(f"正在加载 Alphadia report: {light_result_path}")

        light_input = pd.read_parquet(light_result_path)

        for idx, row in light_input.iterrows():

            modifications = parse_alphadia_peptide_modify(
                row["precursor.mods"], row["precursor.mod_sites"])

            rt = rt_sec_to_min(row["precursor.rt.observed"])

            tot_psminfo = PSMInfo(
                sequence=row["precursor.sequence"],
                charge=row["precursor.charge"],
                modify=modifications,
                rt=rt,
                precursor_mz=row["precursor.mz.observed"],
                raw_title=row["raw.name"],
                protein_names=row["pg.genes"],
            )

            self.psm_info.append(tot_psminfo)

        self.peptide_len = len(self.psm_info)

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
                precursor_mz=row._11,
                raw_title=row.Run,
                protein_names=row._14,
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


def parse_alphadia_peptide_modify(modify_str: str, site_str: str):
    """ 从 Alphadia 给出结果中解析出修饰 """

    # 修饰的结果，代表(该修饰位置，unimod id)
    modifications: [(int, int)] = []

    if modify_str == "" or site_str == "":
        return modifications

    mods_list = modify_str.split(';')
    mod_sites_list = site_str.split(';')

    if len(mods_list) != len(mod_sites_list):
        logging.warning("修饰数量不匹配")
        return modifications

    mod_to_unimod = {
        "Carbamidomethyl": 4,      # UniMod:4 - Carbamidomethyl
        "Oxidation": 35,           # UniMod:35 - Oxidation
        "Phospho": 21,             # UniMod:21 - Phospho
        "Acetyl": 1,               # UniMod:1 - Acetyl
        "Methyl": 34,              # UniMod:34 - Methyl
        "Dimethyl": 36,            # UniMod:36 - Dimethyl
        "Trimethyl": 37,           # UniMod:37 - Trimethyl
        "Deamidated": 7,           # UniMod:7 - Deamidated
        "Pyro-carbamidomethyl": 26,  # UniMod:26 - Pyro-carbamidomethyl
        "Gln->pyro-Glu": 28,       # UniMod:28 - Gln->pyro-Glu
        "Glu->pyro-Glu": 27,       # UniMod:27 - Glu->pyro-Glu
    }

    for mod_type, site_str in zip(mods_list, mod_sites_list):
        if '@' in mod_type:
            mod_name, target_aa = mod_type.split('@')
        else:
            mod_name = mod_type
            target_aa = None

        unimod_id = mod_to_unimod.get(mod_name)

        site_index = int(site_str) - 1

        modifications.append((site_index, unimod_id))

    return modifications


def rt_sec_to_min(rt: float):
    return rt / 60
