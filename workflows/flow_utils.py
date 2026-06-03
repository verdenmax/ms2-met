
import os
import configparser
import logging
import traceback

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

    # P0-3, Silent-C3 (2026-06-03 audit): read currently-configured
    # centroid params from raw_file_manager's config; validate any existing
    # cache against them. Mismatch -> delete cache -> rebuild with current
    # params. Without this, changing centroid params has no effect.
    # P0-3 (2026-06-03 audit + review I1/I4):
    # 1) single source of truth for current params via DataManager helper
    # 2) lightweight validate_cache_params (mmap'd scalar reads, no array load)
    expected_enabled, expected_thresh = raw_file_manager.get_centroid_params()

    if os.path.exists(shared_path):
        try:
            DIAData.validate_cache_params(
                shared_path,
                expected_centroid_enabled=expected_enabled,
                expected_centroid_rel_threshold=expected_thresh,
            )
            logging.info(f"DIA cache {shared_path} 命中（params 匹配）")
        except ValueError as e:
            logging.warning(
                f"DIA cache {shared_path} 失效（{e}）；删除并重建")
            os.remove(shared_path)

    if not os.path.exists(shared_path):
        dia_data = raw_file_manager.get_dia_data_object(filepath)
        dia_data.save_to_file(shared_path)

    logging.info(f"生成 DIA data {shared_path} 完成")
    return name, shared_path


# (deleted — moved to DataManager.get_centroid_params, P0-3 review I4)


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
        "label_type": "positive" if label == 1 else "negative",
        ** tot_features
    }


def _make_result_row_single(psm, features: dict) -> dict:
    """Build the result dict for a single-flow PSM.

    Maps psm._label_type ("positive"/"negative"/None) to label int (1/0/None)
    so the CSV's `label` column is numeric — matching the pair-flow convention.
    """
    label_type = psm._label_type
    if label_type == "positive":
        label = 1
    elif label_type == "negative":
        label = 0
    else:
        label = None
    return {
        "sequence": psm._sequence,
        "charge": psm._charge,
        "precursor_mz": psm._precursor_mz,
        "raw_title1": psm._raw_title,
        "protein_names": psm._protein_names,
        "sequence_len": len(psm._sequence),
        "label": label,
        "label_type": label_type,
        **features,
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

    return _make_result_row_single(psm1, tot_features)


def process_batch_single(shared_path: str, batch_psm_dicts: list, config):
    """ 批量处理单文件任务

    Returns:
        tuple: (results, n_errors) — results is the list of successfully
        processed PSM rows; n_errors counts per-PSM exceptions that were
        caught and logged (so callers can detect silent failures).
    """
    dia_data = DIAData.load_from_file(shared_path, use_mmap=True)
    results = []
    n_errors = 0
    for (psm_dict,) in batch_psm_dicts:
        try:
            psm = PSMInfo.from_dict(psm_dict)
            features = single_pair_work(psm=psm, dia_data=dia_data, config=config)
            results.append(_make_result_row_single(psm, features))
        except Exception:
            logging.error(f"PSM处理失败 seq={psm_dict.get('sequence','?')} "
                          f"charge={psm_dict.get('charge','?')}: "
                          f"{traceback.format_exc()}")
            n_errors += 1
    return results, n_errors


def process_batch_pair(shared1: str, shared2: str, batch_items: list, config):
    """ 批量处理双文件任务

    Returns:
        tuple: (results, n_errors).
    """
    dia1 = DIAData.load_from_file(shared1, use_mmap=True)
    dia2 = DIAData.load_from_file(shared2, use_mmap=True)
    results = []
    n_errors = 0
    for psm1_dict, psm2_dict, label in batch_items:
        try:
            psm1 = PSMInfo.from_dict(psm1_dict)
            psm2 = PSMInfo.from_dict(psm2_dict)
            if label == 0:
                psm2._rt += 10

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
                "label_type": "positive" if label == 1 else "negative",
                ** tot_features
            })
        except Exception:
            logging.error(f"PSM处理失败 seq={psm1_dict.get('sequence','?')} "
                          f"charge={psm1_dict.get('charge','?')}: "
                          f"{traceback.format_exc()}")
            n_errors += 1
    return results, n_errors


def process_batch_pair_shuffle(shared1: str, shared2: str, batch_items: list, config):
    """ 使用shuffle 的模式处理所有负例

    Returns:
        tuple: (results, n_errors).
    """
    dia1 = DIAData.load_from_file(shared1, use_mmap=True)
    dia2 = DIAData.load_from_file(shared2, use_mmap=True)
    results = []
    n_errors = 0
    for psm1_dict, psm2_dict, label in batch_items:
        try:
            psm1 = PSMInfo.from_dict(psm1_dict)
            psm2 = PSMInfo.from_dict(psm2_dict)
            if label == 0:
                new_sequence = sequence_controlled_shuffle(
                    psm1._sequence,
                    anchor_len=2, shuffle_ratio=0.5
                )
                psm1._sequence = new_sequence
                psm2._sequence = new_sequence

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
                "label_type": "positive" if label == 1 else "negative",
                ** tot_features
            })
        except Exception:
            logging.error(f"PSM处理失败 seq={psm1_dict.get('sequence','?')} "
                          f"charge={psm1_dict.get('charge','?')}: "
                          f"{traceback.format_exc()}")
            n_errors += 1
    return results, n_errors
