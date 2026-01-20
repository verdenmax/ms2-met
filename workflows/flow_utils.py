
import os
import configparser
import logging

from spectrum.dia_data import DIAData
from spectrum.psm_info import PSMInfo
import manager.data_manager as data_manager
from workflows.single_work import multi_batch_work, single_pair_work


def get_filename_stem(filepath: str) -> str:
    """ 从路径中获取这个文件的文件名，去除扩展 """
    filename = os.path.basename(filepath)
    stem, _ = os.path.splitext(filename)
    return stem


def data_to_npz(
        raw_file_manager: data_manager.DataManager,
        filepath: str, _workpath: str = "."):
    """ 将一个dia_data 数据保存为 npz 文件，用于内存映射 mmap"""
    # 从路径中获得这个文件的纯文件名
    name = get_filename_stem(filepath)

    shared_path = os.path.join(_workpath, f"{name}.dia.npz")
    if not os.path.exists(shared_path):
        dia_data = raw_file_manager.get_dia_data_object(filepath)
        dia_data.save_to_file(shared_path)

    logging.info(f"生成 DIA data {shared_path} 完成")
    return name, shared_path


def process_psm_pair_shared(
        psm1_dict: dict, psm2_dict: dict,
        shared1_file: str, shared2_file: str,
        config: configparser.ConfigParser, label):
    """ 处理轻重标两个肽段的,获取他们的特征，其中 label 是最后的正样本还是负样本 """

    # 子进程：mmap 加载（物理内存共享）
    dia1 = DIAData.load_from_file(shared1_file, use_mmap=True)
    dia2 = DIAData.load_from_file(shared2_file, use_mmap=True)

    psm1 = PSMInfo.from_dict(psm1_dict)
    psm2 = PSMInfo.from_dict(psm2_dict)

    if label == 0:
        psm2._rt += 10

    # TODO: 计算出信息
    tot_features = multi_batch_work(
        psm1=psm1,
        dia_data1=dia1,
        psm2=psm2,
        dia_data2=dia2,
        config=config,
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


def process_psm_single(
        psm1_dict: dict,
        shared1_file: str,
        config: configparser.ConfigParser):
    """ 处理轻重标两个肽段的,获取他们的特征，其中 label 是最后的正样本还是负样本 """

    # 子进程：mmap 加载（物理内存共享）
    dia1 = DIAData.load_from_file(shared1_file, use_mmap=True)

    psm1 = PSMInfo.from_dict(psm1_dict)

    # TODO: 计算出信息
    tot_features = single_pair_work(
        psm=psm1,
        dia_data=dia1,
        config=config,
    )

    return {
        "sequence": psm1._sequence,
        "charge": psm1._charge,
        "precursor_mz": psm1._precursor_mz,
        "raw_title1": psm1._raw_title,
        "protein_names": psm1._protein_names,
        "sequence_len": len(psm1._sequence),
        "label": psm1._protein_names,
        ** tot_features
    }
