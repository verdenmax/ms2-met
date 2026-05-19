"""pfind 搜索引擎结果的解析工具。

提供修饰名称解析、MH+ → m/z 转换、raw_title 提取等工具。
"""
import os
import re
import glob
import logging
from functools import lru_cache

from pyteomics import mass


# 质子质量
PROTON_MASS = 1.00727646677


# pfind 修饰名 → UniMod ID 硬编码字典（覆盖最常见的修饰，
# 避免每次都查 unimod.xml）。键含 pfind 风格的氨基酸标注。
PFIND_MOD_TO_UNIMOD: dict[str, int] = {
    # Carbamidomethyl
    "Carbamidomethyl[C]": 4,
    "Carbamidomethyl[AnyN-term]": 4,
    # Oxidation
    "Oxidation[M]": 35,
    "Oxidation[W]": 35,
    "Oxidation[H]": 35,
    # Phospho
    "Phospho[S]": 21,
    "Phospho[T]": 21,
    "Phospho[Y]": 21,
    # Acetyl
    "Acetyl[K]": 1,
    "Acetyl[ProteinN-term]": 1,
    "Acetyl[AnyN-term]": 1,
    # Methyl / Dimethyl / Trimethyl
    "Methyl[K]": 34,
    "Methyl[R]": 34,
    "Dimethyl[K]": 36,
    "Dimethyl[R]": 36,
    "Trimethyl[K]": 37,
    # Deamidated
    "Deamidated[N]": 7,
    "Deamidated[Q]": 7,
    # N 端 pyro 转换
    "Pyro-carbamidomethyl[AnyN-term]": 26,
    "Gln->pyro-Glu[AnyN-termQ]": 28,
    "Glu->pyro-Glu[AnyN-termE]": 27,
}


# 单例 UniMod 数据库（lazy）
_UNIMOD_DB = None


def _get_unimod_db():
    """Lazy 加载 UniMod 数据库（pyteomics）。"""
    global _UNIMOD_DB
    if _UNIMOD_DB is None:
        unimod_xml_path = os.path.join(
            os.path.dirname(os.path.dirname(os.path.abspath(__file__))),
            "unimod.xml",
        )
        if os.path.exists(unimod_xml_path):
            with open(unimod_xml_path, "rb") as f:
                _UNIMOD_DB = mass.Unimod(source=f)
        else:
            _UNIMOD_DB = mass.Unimod()
    return _UNIMOD_DB


@lru_cache(maxsize=1024)
def resolve_pfind_mod_name(name: str) -> int | None:
    """从 pfind 修饰名称解析出 UniMod ID。

    解析顺序：
      1. 先查硬编码字典（性能优先，覆盖常见修饰）
      2. 兑底用 UniMod 数据库按 base name 查询（取 "[" 之前的部分）
      3. 未命中则返回 None（调用方应 log warning 并跳过该修饰）

    Returns:
        UniMod ID（整数），或 None 表示无法解析。
    """
    if name in PFIND_MOD_TO_UNIMOD:
        return PFIND_MOD_TO_UNIMOD[name]

    # 兑底：取基础名（如 "Carbamidomethyl[C]" → "Carbamidomethyl"）
    base_name = name.split("[")[0] if "[" in name else name
    try:
        db = _get_unimod_db()
        record = db.by_title(base_name)
        if record is not None:
            return int(record.get("record_id"))
    except Exception as e:
        logging.debug(f"UniMod 查询失败 name={name} base={base_name}: {e}")
    return None


def parse_pfind_modify(modify_str) -> list[tuple[int, int]]:
    """解析 pfind Modifications 字段。

    输入格式（pfind 输出）："3,Carbamidomethyl[C];10,Carbamidomethyl[C];"
      - 位置是 1-based
      - 多个修饰用 ";" 分隔，末尾可能有 ";"

    输入容错：
      - None / 非字符串（包括 pandas 的 float NaN）→ 返回 []
      - 空字符串 / 纯空白 → 返回 []

    输出：list of (0-based position, unimod_id)。

    未知修饰会被跳过并 log warning。
    """
    if not isinstance(modify_str, str):
        return []
    if not modify_str.strip():
        return []

    modifications: list[tuple[int, int]] = []
    for entry in modify_str.rstrip(";").split(";"):
        entry = entry.strip()
        if not entry:
            continue

        # 每个修饰应为 "位置,名称"
        try:
            pos_str, name = entry.split(",", 1)
            pos = int(pos_str.strip()) - 1  # 1-based → 0-based
            name = name.strip()
        except (ValueError, IndexError):
            logging.warning(f"pfind 修饰格式无法解析: '{entry}'")
            continue

        unimod_id = resolve_pfind_mod_name(name)
        if unimod_id is None:
            logging.warning(f"pfind 修饰未知，跳过: '{name}'")
            continue

        modifications.append((pos, unimod_id))

    return modifications


def mhp_to_mz(mhp: float, charge: int) -> float:
    """pfind MH+ → 带 charge 的 m/z。

    MH+ 表示 1+ 离子质量 = 中性质量 + 1 × proton_mass。
    m/z(z) = (中性质量 + z × proton_mass) / z
           = (MH+ + (z-1) × proton_mass) / z
    """
    if charge <= 0:
        raise ValueError(f"charge 必须 > 0，得到 {charge}")
    return (mhp + (charge - 1) * PROTON_MASS) / charge


def extract_raw_title_from_pfind_path(path: str) -> str:
    """从 pfind .qry.res 文件路径提取 raw_title（去掉目录和 .qry.res 后缀）。"""
    basename = os.path.basename(path)
    if basename.endswith(".qry.res"):
        return basename[: -len(".qry.res")]
    return basename


import numpy as np
import pandas as pd

from spectrum.psm_info import PSMInfo


# pfind .qry.res 文件中标识 decoy 的蛋白前缀
PFIND_DECOY_PREFIX = "REV_"


def load_pfind_file(
    file_path: str,
    qvalue_threshold: float = 0.01,
) -> list:
    """加载单个 pfind .qry.res 文件并应用过滤。

    过滤顺序：
      1. QValue > qvalue_threshold → 丢弃（FDR 过滤）
      2. Proteins 以 "REV_" 开头 → 丢弃（decoy 过滤）
      3. PSMInfo.valid() 为 False → 丢弃（含 X 等）

    Returns:
        list[PSMInfo]
    """
    if not os.path.exists(file_path):
        logging.error(f"pfind 文件不存在: {file_path}")
        return []

    logging.info(f"正在加载 pfind 文件: {file_path}")
    df = pd.read_csv(file_path, sep="\t")

    if 'Modifications' in df.columns:
        df['Modifications'] = df['Modifications'].fillna('')

    # 重命名特殊字符列名为合法 Python identifier，以便 itertuples 直接属性访问
    # MH+ → MHPlus, DeltaRT(Min) → DeltaRTMin
    df = df.rename(columns={
        'MH+': 'MHPlus',
        'DeltaRT(Min)': 'DeltaRTMin',
    })

    raw_title = extract_raw_title_from_pfind_path(file_path)
    psms: list[PSMInfo] = []

    n_total = len(df)
    n_filtered_fdr = 0
    n_filtered_decoy = 0
    n_filtered_invalid = 0
    n_parse_error = 0

    # 用 itertuples（比 to_dict(orient='records') 快 3-5x、省内存 3-5x）
    for row in df.itertuples(index=False):
        try:
            qvalue = float(row.QValue)
        except (ValueError, TypeError):
            n_parse_error += 1
            continue

        if qvalue > qvalue_threshold:
            n_filtered_fdr += 1
            continue

        proteins = str(row.Proteins)
        # pfind 多蛋白用 "/" 或 ";" 分隔；当且仅当所有 token 都是 decoy
        # 前缀时才视为 decoy（避免把混合命中的真 target 也丢掉）
        protein_tokens = [t.strip() for t in re.split(r"[;/]", proteins)
                          if t.strip()]
        if protein_tokens and all(
                t.startswith(PFIND_DECOY_PREFIX) for t in protein_tokens):
            n_filtered_decoy += 1
            continue

        try:
            modifications = parse_pfind_modify(row.Modifications)
            charge = int(row.Charge)
            mhp_value = float(row.MHPlus)
            precursor_mz = mhp_to_mz(mhp_value, charge)
            pred_rt = float(row.PredRT)
            delta_rt = float(row.DeltaRTMin)
            rt = pred_rt + delta_rt
            if not np.isfinite(rt):
                n_parse_error += 1
                continue
            score = float(row.FinalScore)
            sequence = str(row.PeptideSequence)
        except (AttributeError, ValueError, TypeError) as e:
            n_parse_error += 1
            logging.warning(f"pfind 行解析失败 file={raw_title}: {e}")
            continue

        psm = PSMInfo(
            sequence=sequence,
            charge=charge,
            modify=modifications,
            rt=np.float32(rt),
            precursor_mz=np.float32(precursor_mz),
            raw_title=raw_title,
            protein_names=proteins,
            q_value=qvalue,
            score=score,
        )

        if not psm.valid():
            n_filtered_invalid += 1
            continue

        psms.append(psm)

    logging.info(
        f"pfind 加载完成 {raw_title}: total={n_total}, "
        f"kept={len(psms)}, fdr_filtered={n_filtered_fdr}, "
        f"decoy_filtered={n_filtered_decoy}, "
        f"invalid={n_filtered_invalid}, parse_error={n_parse_error}"
    )
    return psms


def load_pfind_path(
    path: str,
    qvalue_threshold: float = 0.01,
) -> list:
    """加载 pfind 路径——目录则扫描所有 .qry.res 文件，单文件则只加载该文件。

    Args:
        path: 目录或单个 .qry.res 文件路径
        qvalue_threshold: FDR 阈值

    Returns:
        list[PSMInfo]
    """
    if not os.path.exists(path):
        logging.error(f"pfind 路径不存在: {path}")
        return []

    if os.path.isdir(path):
        files = sorted(glob.glob(os.path.join(path, "*.qry.res")))
        logging.info(f"pfind 目录扫描: {path}，找到 {len(files)} 个 .qry.res 文件")
    else:
        files = [path]

    all_psms = []
    for file_path in files:
        all_psms.extend(load_pfind_file(file_path, qvalue_threshold))

    logging.info(f"pfind 路径加载完毕: {path}，共 {len(all_psms)} 条 PSM")
    return all_psms
