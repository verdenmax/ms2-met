
import re
import os
import json
import numpy as np
import pandas as pd
import logging

from spectrum.psm_info import PSMInfo
from spectrum.pfind_parser import load_pfind_path


class LightResult:
    """ 存储各种搜索引擎搜索得到的轻标结果 """

    def __init__(self):
        """ 初始化 """

        self.peptide_len: np.int64 = 0

        self.psm_info: np.ndarray[tuple[int], PSMInfo] = []

    def _load_from_pkl(self, light_result_path: str):
        """ 从自定义的 json 文件读取 """

        if light_result_path is None or not os.path.exists(light_result_path):
            logging.error(f"json 结果 {light_result_path} 不存在")

        # 正在加载文件
        logging.info(f"正在加载 json 结果: {light_result_path}")

        with open(light_result_path, 'rb') as f:
            loaded_data = json.load(f)

        # 重建 PSMInfo 对象列表
        self.psm_info = [PSMInfo.from_dict(item) for item in loaded_data]

        self.peptide_len = len(self.psm_info)

    def _load_from_alphadia_input(
        self,
        light_result_path: str,
        qvalue_threshold: float = 0.01,
    ):
        """加载 alphadia precursors.parquet，应用 FDR + decoy 过滤。

        过滤顺序：
          1. precursor.qval > qvalue_threshold → 丢弃（FDR）
          2. precursor.decoy != 0 → 丢弃（decoy）
          3. PSMInfo.valid() == False → 丢弃
        """
        if light_result_path is None or not os.path.exists(light_result_path):
            logging.error("Alphadia 搜索结果 report.parquet 不存在")
            return

        logging.info(f"正在加载 Alphadia report: {light_result_path}")

        df = pd.read_parquet(light_result_path)

        required = {
            "precursor.sequence", "precursor.charge", "precursor.rt.observed",
            "precursor.mz.observed", "pg.genes", "precursor.qval",
            "precursor.decoy", "precursor.mods", "precursor.mod_sites",
            "raw.name",
        }
        missing = required - set(df.columns)
        if missing:
            raise ValueError(
                f"Alphadia parquet 缺少列 {sorted(missing)} "
                f"（已有 {sorted(df.columns)}）")

        # 用合法 Python identifier 替换列名，便于 itertuples 属性访问
        col_map = {
            "precursor.sequence": "Sequence",
            "precursor.charge": "Charge",
            "precursor.rt.observed": "RtObserved",
            "precursor.mz.observed": "MzObserved",
            "pg.genes": "Proteins",
            "precursor.qval": "QVal",
            "precursor.decoy": "DecoyFlag",
            "precursor.mods": "Mods",
            "precursor.mod_sites": "ModSites",
            "raw.name": "RawName",
        }
        df = df.rename(columns=col_map)

        n_total = len(df)
        n_fdr = n_decoy = n_invalid = n_parse_err = 0

        for row in df.itertuples(index=False):
            try:
                qvalue = float(row.QVal)
            except (ValueError, TypeError):
                n_parse_err += 1
                continue
            if not np.isfinite(qvalue) or qvalue > qvalue_threshold:
                n_fdr += 1
                continue

            try:
                if int(row.DecoyFlag) != 0:
                    n_decoy += 1
                    continue
            except (ValueError, TypeError):
                n_parse_err += 1
                continue

            try:
                modifications = parse_alphadia_peptide_modify(
                    row.Mods, row.ModSites)
                charge = int(row.Charge)
                # AlphaDIA report 的 RtObserved 是秒 → 转管线规范的分钟
                rt_val = rt_sec_to_min(float(row.RtObserved))
                if not np.isfinite(rt_val):
                    n_parse_err += 1
                    continue
                psm = PSMInfo(
                    sequence=str(row.Sequence),
                    charge=charge,
                    modify=modifications,
                    rt=np.float32(rt_val),
                    precursor_mz=np.float32(row.MzObserved),
                    raw_title=str(row.RawName),
                    protein_names=str(row.Proteins or ""),
                    q_value=qvalue,
                )
            except (AttributeError, ValueError, TypeError) as e:
                n_parse_err += 1
                logging.warning(f"alphadia 行解析失败: {e}")
                continue

            if not psm.valid():
                n_invalid += 1
                continue
            self.psm_info.append(psm)

        self.peptide_len = len(self.psm_info)
        logging.info(
            f"alphadia 加载完成 {light_result_path}: total={n_total}, "
            f"kept={len(self.psm_info)}, fdr_filtered={n_fdr}, "
            f"decoy_filtered={n_decoy}, invalid={n_invalid}, "
            f"parse_error={n_parse_err}"
        )

    def _load_from_dia_nn_input(
        self,
        light_result_path: str,
        qvalue_threshold: float = 0.01,
    ):
        """加载 DIA-NN report.parquet，应用 FDR + decoy 过滤。

        过滤顺序：
          1. Q.Value > qvalue_threshold → 丢弃（FDR）
          2. Decoy == 1 OR Protein.Names 以 REV_/_REV/DECOY_ 开头 → 丢弃
          3. PSMInfo.valid() == False → 丢弃
        """
        if light_result_path is None or not os.path.exists(light_result_path):
            logging.error("dia_nn 搜索结果 report.parquet 不存在")
            return

        logging.info(f"正在加载 DIA-NN report: {light_result_path}")

        df = pd.read_parquet(light_result_path)

        required = {
            "Modified.Sequence", "Precursor.Charge", "RT", "Precursor.Mz",
            "Protein.Names", "Q.Value", "Run",
        }
        missing = required - set(df.columns)
        if missing:
            raise ValueError(
                f"DIA-NN parquet 缺少列 {sorted(missing)} "
                f"（已有 {sorted(df.columns)}）")

        col_map = {
            "Modified.Sequence": "ModifiedSequence",
            "Stripped.Sequence": "StrippedSequence",
            "Precursor.Charge": "PrecursorCharge",
            "Precursor.Mz": "PrecursorMz",
            "Protein.Names": "ProteinNames",
            "Q.Value": "QValue",
            "Decoy": "DecoyFlag",
        }
        df = df.rename(columns=col_map)

        has_decoy_col = "DecoyFlag" in df.columns
        has_stripped = "StrippedSequence" in df.columns

        n_total = len(df)
        n_fdr = n_decoy = n_invalid = n_parse_err = 0

        for row in df.itertuples(index=False):
            try:
                qvalue = float(row.QValue)
            except (ValueError, TypeError):
                n_parse_err += 1
                continue
            if not np.isfinite(qvalue) or qvalue > qvalue_threshold:
                n_fdr += 1
                continue

            if has_decoy_col:
                try:
                    if int(getattr(row, "DecoyFlag")) != 0:
                        n_decoy += 1
                        continue
                except (ValueError, TypeError):
                    pass

            proteins = str(row.ProteinNames or "")
            # 与 pFind 一致：仅当所有蛋白 token 都是 decoy 才丢弃（decoy-led 但
            # 含 target 的组应保留）。DIA-NN ProteinNames 以 ';' 分隔。
            protein_tokens = [t.strip() for t in proteins.split(";") if t.strip()]
            if protein_tokens and all(
                    t.startswith(("REV_", "_REV", "DECOY_"))
                    for t in protein_tokens):
                n_decoy += 1
                continue

            try:
                charge = int(float(row.PrecursorCharge))
                mod_str = str(row.ModifiedSequence)
                modifications = parse_diann_peptide_modify(mod_str)
                if has_stripped and row.StrippedSequence:
                    sequence = str(row.StrippedSequence)
                else:
                    sequence = re.sub(r"\(.*?\)", "", mod_str)
                # DIA-NN report 的 RT 已是分钟（管线规范单位），直读不换算
                rt_val = float(row.RT)
                if not np.isfinite(rt_val):
                    n_parse_err += 1
                    continue
                psm = PSMInfo(
                    sequence=sequence,
                    charge=charge,
                    modify=modifications,
                    rt=np.float32(rt_val),
                    precursor_mz=np.float32(row.PrecursorMz),
                    raw_title=str(row.Run),
                    protein_names=proteins,
                    q_value=qvalue,
                )
            except (AttributeError, ValueError, TypeError) as e:
                n_parse_err += 1
                logging.warning(f"DIA-NN 行解析失败: {e}")
                continue

            if not psm.valid():
                n_invalid += 1
                continue
            self.psm_info.append(psm)

        self.peptide_len = len(self.psm_info)
        logging.info(
            f"DIA-NN 加载完成 {light_result_path}: total={n_total}, "
            f"kept={len(self.psm_info)}, fdr_filtered={n_fdr}, "
            f"decoy_filtered={n_decoy}, invalid={n_invalid}, "
            f"parse_error={n_parse_err}"
        )

    def _load_from_pfind_input(
        self,
        light_result_path: str,
        qvalue_threshold: float = 0.01,
    ):
        """加载 pfind 搜索结果（.qry.res 单文件或目录）。"""
        self.psm_info = load_pfind_path(
            light_result_path, qvalue_threshold=qvalue_threshold)
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

            # 解析出unimod id（容错：缺失 UniMod:\d+ 时跳过，不抛异常）
            group = sequence[index + 1:rindex]
            match = re.search(r'UniMod:(\d+)', group)
            if match is None:
                logging.warning(
                    f"DIA-NN 非 UniMod 修饰，跳过: '{group}' (in {sequence!r})")
            else:
                unimod_id = int(match.group(1))
                # count_index 在遇到 '(' 前已把被修饰残基计入，故残基修饰需 -1
                # 得到 0-based 残基下标；N 端修饰（前导括号，count_index=0）保持 0。
                modifications.append((max(count_index - 1, 0), unimod_id))

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
        if unimod_id is None:
            logging.warning(
                f"alphadia 修饰未知，跳过: '{mod_name}' (site={site_str})")
            continue

        try:
            site_index = int(site_str) - 1
        except (ValueError, TypeError):
            logging.warning(
                f"alphadia 修饰位置无法解析: '{site_str}' (mod={mod_name})")
            continue

        modifications.append((site_index, unimod_id))

    return modifications


def rt_sec_to_min(rt: float):
    """秒 → 分钟。管线 RT 规范单位是分钟（见 DIAData._get_retention_time）。"""
    return rt / 60
