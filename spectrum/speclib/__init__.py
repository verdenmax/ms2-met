"""pFind 谱库（spectral library）二进制读取模块。"""
from .config_io import (Protein, ModEntry, parse_fasta, parse_modifications,
                        parse_element_masses, parse_residue_masses, water_mass)
from .pepdata import LibPeptide, ModSite, iter_pepdata, read_pepdata
from .predictions import FragIon, read_rt_pred, iter_ms2_records, read_chg_max_from_trailer
from .speclib import SpecLib, MassValidationReport

__all__ = [
    # 顶层
    "SpecLib", "MassValidationReport",
    # 数据类型
    "Protein", "ModEntry", "LibPeptide", "ModSite", "FragIon",
    # 配置解析
    "parse_fasta", "parse_modifications", "parse_element_masses",
    "parse_residue_masses", "water_mass",
    # 二进制读取
    "iter_pepdata", "read_pepdata", "read_rt_pred", "iter_ms2_records",
    "read_chg_max_from_trailer",
]
