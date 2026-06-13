"""extract_common：通用 N 引擎交并集工具。

从多个搜索引擎的结果中构造正负例数据集：
- 正例：所有引擎都识别为目标物种的 PSM（key 交集 + species marker 匹配）
- 负例：任一引擎识别为非目标物种的 PSM（key 并集 + species marker 不匹配）

支持的引擎：pfind, diann, alphadia

可选后处理：用 proteinCopilot 的 entrapment 分析结果（classified.tsv）剔除
L0（razor-error）/ L1（L↔I 异构体）级别的"伪负例"——它们物理上与正例无法分辨，
不应作为 SILAC 配对验证的负样本。
"""
import argparse
import configparser
import json
import logging
import os
import sys
from typing import Optional

import pandas as pd
from rich.logging import RichHandler

_PROJECT_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
if _PROJECT_ROOT not in sys.path:
    sys.path.insert(0, _PROJECT_ROOT)

from spectrum.light_result import LightResult
from spectrum.psm_info import PSMInfo, HeavyType, has_label_site
from spectrum.species_marker import matches_species_marker


SUPPORTED_ENGINES = {"pfind", "diann", "alphadia"}

# entrapment 默认剔除的级别：质谱不可分级
DEFAULT_DROP_LEVELS = frozenset({"L0", "L1"})

# proteinCopilot entrapment 分级合法值
VALID_ENTRAPMENT_LEVELS = frozenset({"L0", "L1", "L2", "L3", "L4"})

# 严重程度优先级：collision 时保留更严重的（数字越小越严重）
_LEVEL_SEVERITY = {"L0": 0, "L1": 1, "L2": 2, "L3": 3, "L4": 4}


def _load_engine(engine_name: str, path: str, qvalue_threshold: float) -> list:
    """Internal dispatch: load PSMs from one engine with a single FDR threshold.

    Returns the list of PSMInfo from the chosen engine's LightResult loader.
    Raises ValueError on unknown engine name.
    """
    lr = LightResult()
    if engine_name == "pfind":
        lr._load_from_pfind_input(path, qvalue_threshold=qvalue_threshold)
    elif engine_name == "diann":
        lr._load_from_dia_nn_input(path, qvalue_threshold=qvalue_threshold)
    elif engine_name == "alphadia":
        lr._load_from_alphadia_input(path, qvalue_threshold=qvalue_threshold)
    else:
        raise ValueError(
            f"不支持的引擎: {engine_name}（支持 {SUPPORTED_ENGINES}）")
    return lr.psm_info


def load_engine_psms(engine_name: str, config: configparser.ConfigParser) -> list:
    """根据引擎名加载对应 PSM 列表（单 FDR 阈值，向后兼容入口）。

    Note: New code should use load_engine_psms_dual (Task 2) which
    supports separate tight/loose FDR thresholds for positive/negative
    candidate pools. This single-threshold variant is retained as a
    thin wrapper for any external callers.
    """
    section = f"engine.{engine_name}"
    if section not in config:
        raise ValueError(f"配置中缺少 [{section}] 段")
    path = config[section].get("path")
    if not path:
        raise ValueError(f"[{section}] 缺少 path 配置")
    qvalue = config[section].getfloat("qvalue_threshold", fallback=0.01)
    return _load_engine(engine_name, path, qvalue)


def load_engine_psms_dual(
    engine_name: str,
    config: configparser.ConfigParser,
) -> dict:
    """Load engine PSMs with optional dual FDR (tight for positives,
    loose for negatives).

    Reads two thresholds from [engine.<name>]:
      - qvalue_threshold          (tight, gates positive candidates)
      - negative_qvalue_threshold (loose, gates negative candidates)

    When negative_qvalue_threshold is absent, defaults to qvalue_threshold
    (single-threshold behavior, backward compatible).

    Returns:
        dict {"tight": [PSMInfo], "loose": [PSMInfo]}
        When the two thresholds are equal, both keys point to the SAME
        list (no redundant I/O).

    Raises:
        ValueError: if [engine.<name>] is missing, path is missing,
                    or negative_qvalue_threshold < qvalue_threshold.

    See docs/specs/2026-06-03-dual-fdr-threshold-design.md.
    """
    section = f"engine.{engine_name}"
    if section not in config:
        raise ValueError(f"配置中缺少 [{section}] 段")
    path = config[section].get("path")
    if not path:
        raise ValueError(f"[{section}] 缺少 path 配置")

    tight = config[section].getfloat("qvalue_threshold", fallback=0.01)
    loose = config[section].getfloat(
        "negative_qvalue_threshold", fallback=tight)

    if loose < tight:
        raise ValueError(
            f"[{section}] negative_qvalue_threshold={loose} 不能小于 "
            f"qvalue_threshold={tight} (negative pool must be ⊇ positive pool)"
        )

    tight_psms = _load_engine(engine_name, path, tight)
    if loose == tight:
        loose_psms = tight_psms
    else:
        loose_psms = _load_engine(engine_name, path, loose)

    return {"tight": tight_psms, "loose": loose_psms}


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
        if matches_species_marker(psm._protein_names, positive_marker):
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
        if not matches_species_marker(psm._protein_names, positive_marker):
            psm._label_type = "negative"
            result.append(psm)
            neg_count += 1

    logging.info(
        f"marker='{positive_marker}': positive={pos_count}, negative={neg_count}, "
        f"total={len(result)}"
    )
    return result


def extract_n_engines_from_psms_dual(
    engine_psms_dual: dict,
    engine_order: list,
    positive_marker: Optional[str] = None,
) -> list:
    """Dual-pool variant of extract_n_engines_from_psms.

    Args:
        engine_psms_dual: dict[engine_name -> {"tight": [PSMInfo], "loose": [PSMInfo]}]
        engine_order: list[engine_name]
        positive_marker: species marker string; None ⇒ intersection only,
                         no label assignment.

    Algorithm:
      - Positives: intersection of TIGHT key sets across engines, then
        species marker match against authoritative PSM (looked up in
        LOOSE pool to ensure we always find a PSM — loose ⊇ tight).
      - Negatives: union of LOOSE key sets across engines, then species
        marker mismatch. Same authoritative-PSM rule as positives.

    Returns:
        list[PSMInfo] with _label_type set to "positive"/"negative" (or
        None when positive_marker is None).

    See docs/specs/2026-06-03-dual-fdr-threshold-design.md.
    """
    for pools in engine_psms_dual.values():
        for pool_psms in pools.values():
            for psm in pool_psms:
                psm._label_type = None

    tight_keys = {
        name: {p.get_key_with_raw() for p in pools["tight"]}
        for name, pools in engine_psms_dual.items()
    }
    intersection_keys = (set.intersection(*tight_keys.values())
                         if tight_keys else set())

    loose_keys = {
        name: {p.get_key_with_raw() for p in pools["loose"]}
        for name, pools in engine_psms_dual.items()
    }
    union_keys = set.union(*loose_keys.values()) if loose_keys else set()

    if "diann" in engine_order:
        authoritative_order = ["diann"] + [e for e in engine_order if e != "diann"]
    else:
        authoritative_order = list(engine_order)

    key_to_psm = {}
    for engine_name in authoritative_order:
        for psm in engine_psms_dual.get(engine_name, {}).get("loose", []):
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
        if matches_species_marker(psm._protein_names, positive_marker):
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
        if not matches_species_marker(psm._protein_names, positive_marker):
            psm._label_type = "negative"
            result.append(psm)
            neg_count += 1

    logging.info(
        f"marker='{positive_marker}' (dual-FDR): "
        f"positive={pos_count}, negative={neg_count}, total={len(result)}"
    )
    return result


def load_entrapment_classifications(tsv_path: str) -> dict:
    """从 proteinCopilot 输出的 classified.tsv 加载 PSM 级别分类。

    classified.tsv 由 proteinCopilot 的 entrapment 分析产出，含 L0-L4 五级:
      - L0: razor-error  (序列在 target 库中精确存在 → 质谱不可分)
      - L1: LI-isomer    (仅 L↔I 互换 → 质谱不可分)
      - L2: near-identical (1 处差异 + Δm < 阈值 → 弱可分)
      - L3: homolog      (1-2 处差异 + Δm 充足 → 理论可分)
      - L4: true-trap    (无近邻 target 肽段 → 理想负样本)

    Args:
        tsv_path: classified.tsv 的路径

    Returns:
        dict[(sequence, charge, raw_title) -> level]。
        - 缺少 level 字段的行会被跳过
        - group != "trap" 的行（如 target 行）会被跳过
        - 非法 level（不在 L0-L4 中）的行会被跳过并 warn
        - charge 写成 "2.0" 浮点字符串可正确解析
        - 同 key 多行（不同 modify variant）合并：
          * 若 level 一致 → 静默合并
          * 若 level 不一致 → warn，保留严重性最高的（L0 > L1 > L2 > L3 > L4）

    Note:
        匹配键忽略 modify：L0/L1 是序列层判定（stripped sequence），
        同 (seq, charge, raw) 不同修饰的 PSM 共享 level。
    """
    # 不存在路径 → 抛带路径上下文的异常
    if not os.path.exists(tsv_path):
        raise FileNotFoundError(
            f"classified.tsv 路径不存在: '{tsv_path}'")

    # keep_default_na=False 重要：避免肽段 "NA"、"NaN" 被 pandas 解析为 NaN
    df = pd.read_csv(tsv_path, sep="\t", dtype=str, keep_default_na=False)

    required = {"peptide", "charge", "spectrum_file", "level"}
    missing = required - set(df.columns)
    if missing:
        raise ValueError(
            f"classified.tsv 缺少列 {sorted(missing)}（已有 {sorted(df.columns)}）")

    has_group_col = "group" in df.columns

    total_rows = len(df)
    skipped_empty_level = 0
    skipped_bad_charge = 0
    skipped_non_trap = 0
    skipped_empty_field = 0
    skipped_invalid_level = 0
    collisions = 0
    conflicts = 0

    result: dict = {}
    for row in df.itertuples(index=False):
        if has_group_col:
            group_val = (getattr(row, "group", "") or "").strip().lower()
            if group_val and group_val != "trap":
                skipped_non_trap += 1
                continue

        level = (row.level or "").strip()
        if not level:
            skipped_empty_level += 1
            continue

        # 标准化 + 白名单
        level_upper = level.upper()
        if level_upper not in VALID_ENTRAPMENT_LEVELS:
            logging.warning(
                f"classified.tsv 含非法 level={level!r}（合法: "
                f"{sorted(VALID_ENTRAPMENT_LEVELS)}），跳过该行")
            skipped_invalid_level += 1
            continue
        level = level_upper

        # charge 解析：兼容 "2", "2.0", "  3 " 等
        charge_raw = (row.charge or "").strip()
        try:
            charge = int(float(charge_raw))
        except (ValueError, TypeError):
            skipped_bad_charge += 1
            continue

        peptide = (row.peptide or "").strip()
        raw_title = (row.spectrum_file or "").strip()
        if not peptide or not raw_title:
            skipped_empty_field += 1
            continue

        key = (peptide, charge, raw_title)
        if key in result:
            collisions += 1
            existing = result[key]
            if existing != level:
                conflicts += 1
                # 取更严重的 level（数值小者）
                new_level = (existing if _LEVEL_SEVERITY[existing]
                             <= _LEVEL_SEVERITY[level] else level)
                logging.warning(
                    f"classified.tsv 同 key={key} 出现冲突 level: "
                    f"existing={existing}, new={level} → 保留更严重的 {new_level}")
                result[key] = new_level
            # else: 同 level 静默合并
        else:
            result[key] = level

    logging.info(
        f"加载 entrapment 分类: total_rows={total_rows}, loaded={len(result)}, "
        f"skipped_empty_level={skipped_empty_level}, "
        f"skipped_bad_charge={skipped_bad_charge}, "
        f"skipped_non_trap={skipped_non_trap}, "
        f"skipped_invalid_level={skipped_invalid_level}, "
        f"skipped_empty_field={skipped_empty_field}, "
        f"collisions={collisions} (conflicts={conflicts}) "
        f"来自 {tsv_path}"
    )
    return result


def filter_by_entrapment(
    psms: list,
    classifications: dict,
    drop_levels=DEFAULT_DROP_LEVELS,
) -> list:
    """剔除 classified 中处于 drop_levels 的 negative PSM。

    Args:
        psms: PSMInfo 列表（已设置 label_type）
        classifications: load_entrapment_classifications 的输出
        drop_levels: 要剔除的 level 集合，默认 {"L0", "L1"}（质谱不可分）。
            支持小写输入（内部 normalize 为大写）。

    Returns:
        过滤后的 PSMInfo 列表。规则：
          - label_type == "negative" 且 (seq, charge, raw_title) 命中 drop_levels → 剔除
          - 其他 negative（不在 classifications 中或 level 不在 drop_levels） → 保留
          - positive PSM 一律不动（即使误进 classified.tsv）
          - label_type 为其他值（None 等）也一律不动

        匹配只看 (sequence, charge, raw_title)，忽略 modify；被保留的 PSM 自身
        modify 字段不变。
    """
    drop_levels = {lvl.strip().upper() for lvl in drop_levels}
    if not classifications or not drop_levels:
        return list(psms)

    kept = []
    neg_total = 0
    neg_dropped = 0
    neg_dropped_by_level: dict = {}
    neg_unknown = 0

    for psm in psms:
        if psm._label_type != "negative":
            kept.append(psm)
            continue

        neg_total += 1
        key = (psm._sequence, psm._charge, psm._raw_title)
        level = classifications.get(key)

        if level is None:
            neg_unknown += 1
            kept.append(psm)
            continue

        if level in drop_levels:
            neg_dropped += 1
            neg_dropped_by_level[level] = neg_dropped_by_level.get(level, 0) + 1
            continue

        kept.append(psm)

    by_level_str = ", ".join(
        f"{lvl}={n}" for lvl, n in sorted(neg_dropped_by_level.items()))
    # 这些被剔的 L0/L1 negative，其序列本质上**同源于 target(人)蛋白组**
    # （L0=精确子串、L1=L↔I 异构后子串），质谱不可分——很可能是搜索引擎把
    # 物种分错了（实为 target 肽而非陷阱）。故日志命名为「同源肽段过滤」。
    logging.info(
        f"同源肽段过滤: negative 输入={neg_total}, 剔除={neg_dropped} "
        f"({by_level_str if by_level_str else 'none'}), "
        f"unknown(保留)={neg_unknown}, negative 输出={neg_total - neg_dropped}"
    )
    return kept


_LABELING_ALIASES = {
    "silac": HeavyType.SILAC,
    "c13": HeavyType.CHEAVY, "13c": HeavyType.CHEAVY, "cheavy": HeavyType.CHEAVY,
    "n15": HeavyType.NHEAVY, "15n": HeavyType.NHEAVY, "nheavy": HeavyType.NHEAVY,
}


def _parse_labeling(config: configparser.ConfigParser) -> HeavyType:
    """Read [extract] labeling (default 'silac'); map to HeavyType.

    Accepts case-insensitive aliases: silac; c13/13c/cheavy; n15/15n/nheavy.
    Raises ValueError on an unknown value.
    """
    raw = "silac"
    if config.has_section("extract"):
        raw = config["extract"].get("labeling", "silac")
    key = str(raw).strip().lower()
    if key not in _LABELING_ALIASES:
        raise ValueError(
            f"非法 [extract] labeling={raw!r}（合法: {sorted(_LABELING_ALIASES)}）")
    return _LABELING_ALIASES[key]


def filter_by_label_site(psms: list, heavy_type: HeavyType) -> list:
    """Drop PSMs (both target and trap) with no metabolic-label site under
    heavy_type — they cannot be light/heavy validated (spec §12 class 4).

    Under SILAC this drops no-K/R peptides; under CHEAVY/NHEAVY every peptide
    is labeled so nothing is dropped.
    """
    kept = []
    dropped_pos = 0
    dropped_neg = 0
    for psm in psms:
        if has_label_site(psm._sequence, heavy_type):
            kept.append(psm)
            continue
        if psm._label_type == "negative":
            dropped_neg += 1
        else:
            dropped_pos += 1
    logging.info(
        f"label-site 过滤({heavy_type.name}): 剔除 positive={dropped_pos}, "
        f"negative={dropped_neg}, 输出={len(kept)}")
    return kept


def filter_by_contaminant(psms: list, contaminant_index,
                          match_li: bool = True) -> list:
    """Drop PSMs (both target and trap) whose stripped sequence maps to a
    contaminant protein (cRAP-style 污染库).

    复用 entrapment_classifier 的子串索引：肽段序列若是污染蛋白的精确子串
    （L0），或（当 match_li=True）L↔I 归一化后子串（L1），即判为污染并剔除。
    正负例都过滤。
    """
    from spectrum.entrapment_classifier import classify_peptide

    drop_levels = {"L0", "L1"} if match_li else {"L0"}
    kept = []
    dropped_pos = 0
    dropped_neg = 0
    for psm in psms:
        level = classify_peptide(psm._sequence, contaminant_index)
        if level in drop_levels:
            if psm._label_type == "negative":
                dropped_neg += 1
            else:
                dropped_pos += 1
            continue
        kept.append(psm)
    logging.info(
        f"污染库过滤(contaminant, match_li={match_li}): 剔除 "
        f"positive={dropped_pos}, negative={dropped_neg}, 输出={len(kept)}")
    return kept


def extract_n_engines(config: configparser.ConfigParser) -> list:
    """根据 config 加载各引擎并构造正负例（含无标记位点过滤 + 可选污染库 + entrapment 过滤）。"""
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
        engine_psms[name] = load_engine_psms_dual(name, config)
        n_tight = len(engine_psms[name]["tight"])
        n_loose = len(engine_psms[name]["loose"])
        if engine_psms[name]["tight"] is engine_psms[name]["loose"]:
            logging.info(f"  → {name} 共 {n_tight} 条 PSM (单 FDR 池)")
        else:
            logging.info(
                f"  → {name} tight={n_tight}, loose={n_loose} (双 FDR 池)")

    psms = extract_n_engines_from_psms_dual(
        engine_psms, engine_order, positive_marker)

    # Domain-of-applicability filter: drop peptides with no metabolic-label
    # site (spec §12 class 4). Runs unconditionally, both classes.
    psms = filter_by_label_site(psms, _parse_labeling(config))

    # Contaminant-library filter (污染库过滤): drop peptides (both classes)
    # mapping to a contaminant protein. Optional [contaminant] section.
    if "contaminant" in config:
        cont_fasta = os.path.expanduser(
            config["contaminant"].get("fasta", "").strip())
        if cont_fasta:
            from spectrum.entrapment_classifier import load_target_fasta
            match_li = config["contaminant"].getboolean(
                "match_li", fallback=True)
            # 子串+L↔I 是"宁可多删"的保守方向，对清洗 negative 是安全的；这里
            # 也作用于 positive（用户选择 B），即与污染蛋白质谱不可分的肽段一律剔。
            cont_index = load_target_fasta(cont_fasta, log_label="污染库")
            psms = filter_by_contaminant(psms, cont_index, match_li=match_li)
        else:
            logging.info(
                "[contaminant] 段已配置但 fasta 为空，跳过污染库过滤")
    else:
        logging.info("未配置 [contaminant] 段，跳过污染库过滤")

    if "entrapment" in config:
        classified_tsv = os.path.expanduser(
            config["entrapment"].get("classified_tsv", "").strip())
        target_fasta = os.path.expanduser(
            config["entrapment"].get("target_fasta", "").strip())
        drop_levels_str = config["entrapment"].get(
            "drop_levels", "L0,L1").strip()
        drop_levels = {
            lvl.strip().upper() for lvl in drop_levels_str.split(",")
            if lvl.strip()
        }

        classifications = None
        if classified_tsv:
            # Explicit TSV wins (precomputed by proteinCopilot or earlier run)
            classifications = load_entrapment_classifications(classified_tsv)
            if target_fasta:
                logging.info(
                    "[entrapment] 同时指定了 classified_tsv 和 target_fasta；"
                    "使用 classified_tsv（显式 > 派生）")
        elif target_fasta:
            # In-memory classification: run classifier on the negatives we
            # just constructed. Saves a round-trip through disk TSV.
            from spectrum.entrapment_classifier import (
                classify_peptide, load_target_fasta,
            )
            target = load_target_fasta(target_fasta)
            classifications = {}
            n_negative_classified = 0
            for psm in psms:
                if psm._label_type != "negative":
                    continue
                level = classify_peptide(psm._sequence, target)
                key = (psm._sequence, psm._charge, psm._raw_title)
                classifications[key] = level
                n_negative_classified += 1
            logging.info(
                f"[entrapment] 内联分类完成: target={target_fasta}, "
                f"n_proteins={target.n_proteins}, "
                f"classified={n_negative_classified} negatives")

        if classifications is not None:
            psms = filter_by_entrapment(
                psms, classifications, drop_levels=drop_levels)

    return psms


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
    read_files = config.read(args.configpath)
    if not read_files:
        logging.error(
            f"配置文件不存在或无法读取: '{args.configpath}' "
            f"(configparser.read() 返回空列表)")
        sys.exit(1)
    if not config.sections():
        logging.error(
            f"配置文件 '{args.configpath}' 内容为空或无 [section]")
        sys.exit(1)

    if "extract" not in config:
        logging.error(f"配置文件 {args.configpath} 缺少 [extract] 段")
        sys.exit(1)

    psms = extract_n_engines(config)
    result_file = config["extract"]["result_file"]
    write_psms_to_json(psms, result_file)


if __name__ == "__main__":
    main()
