"""流式读取 pepdata.pdb（二进制肽段库），复刻 CReader::ReadPepData。

需要：proteins（parse_fasta 结果，按 pro_id 索引）、mods_by_id（mod_id->ModEntry）。
体量大（~312 万肽段），核心用生成器 iter_pepdata 逐条产出；read_pepdata 为 list 包装。
"""
import mmap
import os
import struct
from dataclasses import dataclass, field

from .config_io import Protein, ModEntry

_HEADER = struct.Struct("<IIbbbbIQ")  # pro_id,pep_start,pep_len,pro_nc,enz,miss,mod_pep_num,mod_pep_bytes
_VAR = struct.Struct("<db")           # mass(double), mod_cnt(char)
_MOD = struct.Struct("<bi")           # pos(char), mod_id(int)


@dataclass(slots=True)
class ModSite:
    pos: int
    mod_id: int
    name: str = ""
    mono_mass: float = 0.0


@dataclass(slots=True)
class LibPeptide:
    sequence: str
    mods: list[ModSite]
    neutral_mass: float
    protein: str
    is_decoy: bool
    pro_nc: int = 0
    enz: int = 0
    miss: int = 0
    charge_mask: int = 0  # pdb 读取恒为 0（对应 C++ PepInfo.chg，预测库不限电荷）
    pred_rt: float | None = None
    pred_ms2: dict = field(default_factory=dict)  # charge -> list[FragIon] | np.ndarray


def iter_pepdata(path: str, proteins: list[Protein],
                 mods_by_id: dict[int, ModEntry],
                 validate_bytes: bool = True):
    """逐肽段变体 yield LibPeptide（内存 O(1)，不含预测值）。

    用 mmap：早退的调用方（如只取前 N 个样例）无需读完整 91MB pdb。
    """
    size = os.path.getsize(path)
    if size == 0:
        return
    with open(path, "rb") as fh:
        data = mmap.mmap(fh.fileno(), 0, access=mmap.ACCESS_READ)
        try:
            try:
                data.madvise(mmap.MADV_SEQUENTIAL)
            except (AttributeError, OSError):
                pass
            off = 0
            n = size
            while off < n:
                (pro_id, pep_start, pep_len, pro_nc, enz, miss,
                 mod_pep_num, mod_pep_bytes) = _HEADER.unpack_from(data, off)
                off += _HEADER.size
                protein = proteins[pro_id]
                seq = protein.sequence[pep_start:pep_start + pep_len]
                consumed = 0
                for _ in range(mod_pep_num):
                    mass, mod_cnt = _VAR.unpack_from(data, off)
                    off += _VAR.size
                    consumed += _VAR.size
                    sites: list[ModSite] = []
                    for _ in range(mod_cnt):
                        mpos, mid = _MOD.unpack_from(data, off)
                        off += _MOD.size
                        consumed += _MOD.size
                        entry = mods_by_id.get(mid)
                        sites.append(ModSite(
                            pos=mpos, mod_id=mid,
                            name=entry.name if entry else "",
                            mono_mass=entry.mono_mass if entry else 0.0))
                    yield LibPeptide(
                        sequence=seq, mods=sites, neutral_mass=mass,
                        protein=protein.ac, is_decoy=protein.is_decoy,
                        pro_nc=pro_nc, enz=enz, miss=miss)
                if validate_bytes and consumed != mod_pep_bytes:
                    raise ValueError(
                        f"mod_pep_bytes mismatch at pro_id={pro_id}: "
                        f"consumed {consumed} != declared {mod_pep_bytes}")
        finally:
            data.close()


def read_pepdata(path: str, proteins: list[Protein],
                 mods_by_id: dict[int, ModEntry],
                 validate_bytes: bool = True) -> list[LibPeptide]:
    """list 包装（小数据 / 测试用）。"""
    return list(iter_pepdata(path, proteins, mods_by_id, validate_bytes))
