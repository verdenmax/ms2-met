# predictions — API 参考（`spectrum/speclib/predictions.py`）

## `FragIon` (dataclass, slots)

字段：`ion_type: str`（'b'|'y'）、`frag_pos: int`（0-indexed 切割位）、`frag_charge: int`（1..6）、`intensity: float`。

## `MAX_ION_OUTPUT = 1000`

单记录离子上限，同时作为二进制/文本尾巴的分界判据。

## `read_rt_pred(path: str) -> array('f')`

全量读取 `pepdata.rt.predb`，返回 `array('f')`（M 个 float32，分钟）。按下标随机访问。
注：`array('f')` 用本机字节序，文件为小端；非小端主机会 `byteswap()` 纠正。

## `iter_ms2_arrays(path: str, max_ions: int = MAX_ION_OUTPUT, copy: bool = True)`

- **产出**：生成器，每次一个 numpy 结构化数组（dtype 字段 `pos:i1, iontype:i1, inten:f4`；可能为空）。
- **快路径**：比 `iter_ms2_records` 快约 5–8×、省约 8× 内存；数值/性能敏感场景首选。
- **copy**：`True`（默认）每记录独立副本，可安全保留；`False` 零拷贝视图，**只可取出即用、不可跨迭代保留**（否则 mmap 无法干净关闭）。
- **停止**：遇 `n_size<0` 或 `>max_ions`（文本尾巴）/截断即结束。
- iontype→(b/y, 电荷)：`'b' if it%2==0 else 'y'`、`it//2+1`，按需在数组上向量化派生。

## `iter_ms2_records(path: str, max_ions: int = MAX_ION_OUTPUT)`

- **产出**：生成器，每次 `list[FragIon]`（一条记录；可能为空 `[]`）。便捷对象 API，建立在 `iter_ms2_arrays(copy=True)` 之上。
- **流式**：用 `mmap` 读，常驻内存 O(1)（真实文件 ~4.4GB 不 OOM）。
- **不做**分组；分组到肽段由 `speclib` 按 `chg_max` 锁步完成。

## `read_chg_max_from_trailer(path: str, tail_bytes: int = 8192) -> int`

- 从文件末尾文本尾巴解析 `chg_max`。
- **异常**：`ValueError`（无法解析出形如 `1..C` 的尾巴行）。
- 前提：库至少 2 个肽段（保证末尾有干净尾巴行）。

## 示例

```python
from spectrum.speclib.predictions import (
    read_rt_pred, iter_ms2_arrays, iter_ms2_records, read_chg_max_from_trailer)

rt = read_rt_pred("pepdata.rt.predb")           # array('f')
chg_max = read_chg_max_from_trailer("pepdata.ms2.predb")   # 4

# 快路径（numpy 数组）
for arr in iter_ms2_arrays("pepdata.ms2.predb"):
    ...  # arr["pos"], arr["iontype"], arr["inten"]

# 便捷对象路径
for ions in iter_ms2_records("pepdata.ms2.predb"):
    ...  # ions: list[FragIon]
```
