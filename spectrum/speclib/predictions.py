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

import numpy as np

_MS2_HEAD = struct.Struct("<h")   # n_size(short)
_MS2_ION = struct.Struct("<bbf")  # pos(char), iontype(char), inten(float)
# 与 _MS2_ION 同布局的 numpy 结构化 dtype（itemsize=6），用于批量解码
_ION_DTYPE = np.dtype([("pos", "i1"), ("iontype", "i1"), ("inten", "<f4")])
MAX_ION_OUTPUT = 1000


@dataclass(slots=True)
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
    a.frombytes(data[:len(data) // 4 * 4])   # 丢弃不足 4 字节的尾部
    if sys.byteorder != "little":   # array('f') 用本机字节序；文件是小端
        a.byteswap()
    return a


def iter_ms2_arrays(path: str, max_ions: int = MAX_ION_OUTPUT,
                    copy: bool = True):
    """逐记录 yield 一个 numpy 结构化数组（字段 pos:i1, iontype:i1, inten:f4）。

    比 iter_ms2_records（list[FragIon]）快约 8×、省约 8× 内存，适合数值处理。
    用 mmap，O(1) 常驻内存（真实 ms2.predb ~4.4GB）。遇 n_size<0 或 >max_ions
    （文本尾巴）即停。
    - copy=True（默认）：每条记录是独立副本，可安全长期保留（如挂到 pred_ms2）。
    - copy=False：零拷贝 mmap 视图，仅在"取出即用、不跨迭代保留"时使用（保留视图
      会令文件无法干净关闭 / 悬垂）。
    """
    size = os.path.getsize(path)
    if size == 0:
        return
    with open(path, "rb") as fh:
        mm = mmap.mmap(fh.fileno(), 0, access=mmap.ACCESS_READ)
        try:
            try:
                mm.madvise(mmap.MADV_SEQUENTIAL)   # 提示内核顺序预读
            except (AttributeError, OSError):
                pass
            off = 0
            n = size
            while off + 2 <= n:
                (n_size,) = _MS2_HEAD.unpack_from(mm, off)
                if n_size < 0 or n_size > max_ions:
                    break               # 进入文本尾巴
                off += 2
                end = off + n_size * 6
                if end > n:
                    break               # 截断/损坏：干净停止
                view = np.frombuffer(mm, dtype=_ION_DTYPE, count=n_size, offset=off)
                off = end
                if copy:
                    rec = view.copy()
                    del view            # 释放对 mmap 的视图，保证 finally 能干净 close
                    yield rec
                else:
                    yield view          # 零拷贝视图：不可跨迭代保留
        finally:
            mm.close()


def iter_ms2_records(path: str, max_ions: int = MAX_ION_OUTPUT):
    """逐记录 yield list[FragIon]（便捷对象 API）。

    建立在 iter_ms2_arrays 之上；数值/性能敏感场景请直接用 iter_ms2_arrays。
    遇 n_size<0 或 >max_ions（文本尾巴）即停。
    """
    for arr in iter_ms2_arrays(path, max_ions=max_ions, copy=True):
        yield [FragIon("b" if it % 2 == 0 else "y", pos, it // 2 + 1, inten)
               for pos, it, inten in arr.tolist()]


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
