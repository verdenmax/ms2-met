"""读取 pepdata.rt.predb（M 个 float）与 pepdata.ms2.predb（二进制记录 + 文本尾巴）。

复刻 pPredRT.cpp / pPredMS2.cpp 写出格式。MS2 文件结构为
`[M×chg_max 二进制记录][M 行文本尾巴]`，读取遇尾巴即停。
"""
import mmap
import os
import struct
import sys
from array import array
from dataclasses import dataclass

_MS2_HEAD = struct.Struct("<h")   # n_size(short)
_MS2_ION = struct.Struct("<bbf")  # pos(char), iontype(char), inten(float)
MAX_ION_OUTPUT = 1000


@dataclass
class FragIon:
    ion_type: str       # 'b' | 'y'
    frag_pos: int       # 0-indexed 切割位
    frag_charge: int    # 1..6
    intensity: float


def read_rt_pred(path: str) -> "array":
    """全量读取（~12MB），返回 array('f')。"""
    with open(path, "rb") as fh:
        data = fh.read()
    a = array("f")
    a.frombytes(data[:len(data) // 4 * 4])
    if sys.byteorder != "little":   # array('f') 用本机字节序；文件是小端
        a.byteswap()
    return a


def iter_ms2_records(path: str, max_ions: int = MAX_ION_OUTPUT):
    """逐记录 yield list[FragIon]；遇 n_size<0 或 >max_ions（文本尾巴）即停。

    用 mmap 而非 fh.read()：真实 ms2.predb ~4.4GB，必须保持 O(1) 常驻内存。
    """
    size = os.path.getsize(path)
    if size == 0:
        return
    with open(path, "rb") as fh:
        mm = mmap.mmap(fh.fileno(), 0, access=mmap.ACCESS_READ)
        try:
            off = 0
            n = size
            ion_size = _MS2_ION.size
            while off + 2 <= n:
                (n_size,) = _MS2_HEAD.unpack_from(mm, off)
                if n_size < 0 or n_size > max_ions:
                    break               # 进入文本尾巴
                off += 2
                if off + n_size * ion_size > n:
                    break               # 截断/损坏：干净停止
                ions: list[FragIon] = []
                for _ in range(n_size):
                    pos, iontype, inten = _MS2_ION.unpack_from(mm, off)
                    off += ion_size
                    ions.append(FragIon(
                        ion_type="b" if iontype % 2 == 0 else "y",
                        frag_pos=pos,
                        frag_charge=iontype // 2 + 1,
                        intensity=inten))
                yield ions
        finally:
            mm.close()


def read_chg_max_from_trailer(path: str, tail_bytes: int = 8192) -> int:
    """从文件末尾文本尾巴解析 chg_max。尾巴行形如 '1\\t0\\t2\\t0\\t...\\tC\\t0\\t'。"""
    size = os.path.getsize(path)
    with open(path, "rb") as fh:
        fh.seek(max(0, size - tail_bytes))
        tail = fh.read().decode("latin-1", errors="replace")
    for line in reversed(tail.split("\n")):
        toks = line.split("\t")
        charges = toks[0::2]            # 偶数下标为电荷
        if charges and charges[-1] == "":
            charges = charges[:-1]
        if charges and all(c.isdigit() for c in charges):
            vals = [int(c) for c in charges]
            if vals == list(range(1, len(vals) + 1)):
                return len(vals)
    raise ValueError(f"cannot parse chg_max from trailer of {path}")
