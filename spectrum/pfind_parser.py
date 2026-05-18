"""pfind 搜索引擎结果的解析工具。

提供修饰名称解析、MH+ → m/z 转换、raw_title 提取等工具。
"""
import os
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
    except (KeyError, Exception) as e:
        logging.debug(f"UniMod 查询失败 name={name} base={base_name}: {e}")
    return None


def parse_pfind_modify(modify_str: str) -> list[tuple[int, int]]:
    """解析 pfind Modifications 字段。

    输入格式（pfind 输出）："3,Carbamidomethyl[C];10,Carbamidomethyl[C];"
      - 位置是 1-based
      - 多个修饰用 ";" 分隔，末尾可能有 ";"

    输出：list of (0-based position, unimod_id)。

    未知修饰会被跳过并 log warning。
    """
    if not modify_str or not modify_str.strip():
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
