
import os
import configparser
import logging

from spectrum.dia_data import DIAData
from spectrum.psm_info import PSMInfo
from spectrum.psm_info import sequence_controlled_shuffle
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


def process_batch_single(shared_path: str, batch_psm_dicts: list, config):
    """ 批量处理单文件任务 """
    dia_data = DIAData.load_from_file(shared_path, use_mmap=True)
    results = []
    for (psm_dict,) in batch_psm_dicts:
        psm = PSMInfo.from_dict(psm_dict)
        features = single_pair_work(psm=psm, dia_data=dia_data, config=config)
        results.append({
            "sequence": psm._sequence,
            "charge": psm._charge,
            "precursor_mz": psm._precursor_mz,
            "raw_title1": psm._raw_title,
            "protein_names": psm._protein_names,
            "sequence_len": len(psm._sequence),
            "label": psm._protein_names,
            **features
        })
    return results


def process_batch_pair(shared1: str, shared2: str, batch_items: list, config):
    """ 批量处理双文件任务 """
    dia1 = DIAData.load_from_file(shared1, use_mmap=True)
    dia2 = DIAData.load_from_file(shared2, use_mmap=True)
    results = []
    for psm1_dict, psm2_dict, label in batch_items:
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

        results.append({
            "sequence": psm1._sequence,
            "charge": psm1._charge,
            "precursor_mz": psm1._precursor_mz,
            "raw_title1": psm1._raw_title,
            "raw_title2": psm2._raw_title,
            "protein_names": psm1._protein_names,
            "sequence_len": len(psm1._sequence),
            "label": label,
            ** tot_features
        })
    return results


def process_batch_pair_shuffle(shared1: str, shared2: str, batch_items: list, config):
    """ 使用shuffle 的模式处理所有负例 """
    dia1 = DIAData.load_from_file(shared1, use_mmap=True)
    dia2 = DIAData.load_from_file(shared2, use_mmap=True)
    results = []
    for psm1_dict, psm2_dict, label in batch_items:
        psm1 = PSMInfo.from_dict(psm1_dict)
        psm2 = PSMInfo.from_dict(psm2_dict)
        if label == 0:
            new_sequence = sequence_controlled_shuffle(
                psm1._sequence,
                anchor_len=2, shuffle_ratio=0.5
            )
            psm1._sequence = new_sequence
            psm2._sequence = new_sequence

        # TODO: 计算出信息
        tot_features = multi_batch_work(
            psm1=psm1,
            dia_data1=dia1,
            psm2=psm2,
            dia_data2=dia2,
            config=config,
        )

        results.append({
            "sequence": psm1._sequence,
            "charge": psm1._charge,
            "precursor_mz": psm1._precursor_mz,
            "raw_title1": psm1._raw_title,
            "raw_title2": psm2._raw_title,
            "protein_names": psm1._protein_names,
            "sequence_len": len(psm1._sequence),
            "label": label,
            ** tot_features
        })
    return results
