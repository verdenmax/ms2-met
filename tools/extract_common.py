"""extract_common：通用 N 引擎交并集工具。

从多个搜索引擎的结果中构造正负例数据集：
- 正例：所有引擎都识别为目标物种的 PSM（key 交集 + species marker 匹配）
- 负例：任一引擎识别为非目标物种的 PSM（key 并集 + species marker 不匹配）

支持的引擎：pfind, diann, alphadia
"""
import argparse
import configparser
import json
import logging
import os
import sys
from typing import Optional

from rich.logging import RichHandler

_PROJECT_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
if _PROJECT_ROOT not in sys.path:
    sys.path.insert(0, _PROJECT_ROOT)

from spectrum.light_result import LightResult
from spectrum.psm_info import PSMInfo


SUPPORTED_ENGINES = {"pfind", "diann", "alphadia"}


def load_engine_psms(engine_name: str, config: configparser.ConfigParser) -> list:
    """根据引擎名加载对应 PSM 列表。"""
    section = f"engine.{engine_name}"
    if section not in config:
        raise ValueError(f"配置中缺少 [{section}] 段")
    path = config[section].get("path")
    if not path:
        raise ValueError(f"[{section}] 缺少 path 配置")

    lr = LightResult()
    if engine_name == "pfind":
        qvalue = config[section].getfloat("qvalue_threshold", fallback=0.01)
        lr._load_from_pfind_input(path, qvalue_threshold=qvalue)
    elif engine_name == "diann":
        lr._load_from_dia_nn_input(path)
    elif engine_name == "alphadia":
        lr._load_from_alphadia_input(path)
    else:
        raise ValueError(
            f"不支持的引擎: {engine_name}（支持 {SUPPORTED_ENGINES}）")

    return lr.psm_info


def extract_n_engines_from_psms(
    engine_psms: dict,
    engine_order: list,
    positive_marker: Optional[str] = None,
) -> list:
    """从多引擎的 PSM 列表构造正负例数据集（核心算法）。

    Args:
        engine_psms: dict[engine_name -> list[PSMInfo]]
        engine_order: list[engine_name]，引擎列表（指定参与交并集的引擎集合）
        positive_marker: 物种 marker 字符串；为 None 则仅取交集，不打 label

    Note:
        权威 PSM 选择规则：
          - 若 'diann' 在 engine_order 中 → diann 的 PSM 作为权威源（优先）
          - 否则按 engine_order 顺序，先到先得
        marker 检查（positive_marker in protein_names）只看权威 PSM。
        极少数多引擎对同一肽段蛋白归属不一致的情况，以权威引擎为准。

    Returns:
        list[PSMInfo]，每条 PSM 的 label_type 字段已被设置（或保持 None）
    """
    # 防止 stale state：清空所有输入 PSM 的 label_type，避免重复调用时残留
    for psms in engine_psms.values():
        for psm in psms:
            psm._label_type = None

    key_sets = {name: {p.get_key_with_raw() for p in psms}
                for name, psms in engine_psms.items()}

    intersection_keys = set.intersection(*key_sets.values()) if key_sets else set()
    union_keys = set.union(*key_sets.values()) if key_sets else set()

    # 2. 构建 key → PSM 映射
    #
    # 选择"权威 PSM"的优先级：
    #   1. 如果 "diann" 在 engine_order 中，diann 的 PSM 优先（DIANN 蛋白归属更可靠）
    #   2. 否则按 engine_order 顺序，先到先得
    #
    # 注：此函数的 marker 检查（positive_marker in protein_names）只看权威 PSM。
    # 极少数情况下多引擎对同一肽段（同 sequence+charge+modify）的蛋白归属可能
    # 不一致；此简化以权威引擎为准。
    if "diann" in engine_order:
        authoritative_order = ["diann"] + [e for e in engine_order if e != "diann"]
    else:
        authoritative_order = list(engine_order)

    key_to_psm = {}
    for engine_name in authoritative_order:
        for psm in engine_psms.get(engine_name, []):
            key = psm.get_key_with_raw()
            if key not in key_to_psm:
                key_to_psm[key] = psm

    result = []

    if not positive_marker:
        for key in intersection_keys:
            psm = key_to_psm.get(key)
            if psm is not None:
                psm._label_type = None
                result.append(psm)
        logging.info(
            f"无 marker 模式：intersection size={len(result)}")
        return result

    pos_count = 0
    neg_count = 0
    positive_keys = set()
    for key in intersection_keys:
        psm = key_to_psm.get(key)
        if psm is None:
            continue
        if positive_marker in psm._protein_names:
            psm._label_type = "positive"
            result.append(psm)
            positive_keys.add(key)
            pos_count += 1

    for key in union_keys:
        if key in positive_keys:
            continue
        psm = key_to_psm.get(key)
        if psm is None:
            continue
        if positive_marker not in psm._protein_names:
            psm._label_type = "negative"
            result.append(psm)
            neg_count += 1

    logging.info(
        f"marker='{positive_marker}': positive={pos_count}, negative={neg_count}, "
        f"total={len(result)}"
    )
    return result


def extract_n_engines(config: configparser.ConfigParser) -> list:
    """根据 config 加载各引擎并构造正负例。"""
    engines_str = config["extract"]["engines"]
    engine_order = [e.strip() for e in engines_str.split(",") if e.strip()]

    invalid = [e for e in engine_order if e not in SUPPORTED_ENGINES]
    if invalid:
        raise ValueError(
            f"未知引擎: {invalid}（支持 {SUPPORTED_ENGINES}）")

    positive_marker = config["extract"].get("positive_species_marker", "").strip()
    if not positive_marker:
        positive_marker = None

    engine_psms = {}
    for name in engine_order:
        logging.info(f"加载引擎: {name}")
        engine_psms[name] = load_engine_psms(name, config)
        logging.info(f"  → {name} 共 {len(engine_psms[name])} 条 PSM")

    return extract_n_engines_from_psms(
        engine_psms, engine_order, positive_marker)


def write_psms_to_json(psms: list, output_path: str):
    """把 PSMInfo 列表序列化到 JSON。"""
    output_dir = os.path.dirname(output_path)
    if output_dir and not os.path.exists(output_dir):
        os.makedirs(output_dir, exist_ok=True)
    with open(output_path, "w") as f:
        json.dump([p.to_dict() for p in psms], f, indent=2)
    logging.info(f"已写入 {len(psms)} 条 PSM 到 {output_path}")


def main():
    parser = argparse.ArgumentParser(
        description="extract_common: 通用 N 引擎交并集数据集构造工具")
    parser.add_argument(
        "--configpath", default="./extract_common_config.ini",
        help="配置文件路径")
    parser.add_argument(
        "--logpath", default="./extract_common.log", help="日志文件路径")
    args = parser.parse_args()

    # 确保 log 父目录存在（避免 FileHandler 因目录不存在而失败）
    log_dir = os.path.dirname(args.logpath)
    if log_dir and not os.path.exists(log_dir):
        os.makedirs(log_dir, exist_ok=True)

    file_handler = logging.FileHandler(args.logpath, encoding="utf-8")
    file_handler.setFormatter(logging.Formatter(
        "%(asctime)s | %(levelname)s | %(name)s | %(message)s"))
    logging.basicConfig(
        level=logging.INFO,
        handlers=[RichHandler(), file_handler],
    )

    config = configparser.ConfigParser()
    config.read(args.configpath)

    if "extract" not in config:
        logging.error(f"配置文件 {args.configpath} 缺少 [extract] 段")
        sys.exit(1)

    psms = extract_n_engines(config)
    result_file = config["extract"]["result_file"]
    write_psms_to_json(psms, result_file)


if __name__ == "__main__":
    main()
