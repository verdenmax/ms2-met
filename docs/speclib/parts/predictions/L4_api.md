# predictions — API 参考（`spectrum/speclib/predictions.py`）

## `FragIon` (dataclass)

字段：`ion_type: str`（'b'|'y'）、`frag_pos: int`（0-indexed 切割位）、`frag_charge: int`（1..6）、`intensity: float`。

## `MAX_ION_OUTPUT = 1000`

单记录离子上限，同时作为二进制/文本尾巴的分界判据。

## `read_rt_pred(path: str) -> array('f')`

全量读取 `pepdata.rt.predb`，返回 `array('f')`（M 个 float32，分钟）。按下标随机访问。

## `iter_ms2_records(path: str, max_ions: int = MAX_ION_OUTPUT)`

- **产出**：生成器，每次 `list[FragIon]`（一条记录；可能为空 `[]`）。
- **停止**：遇 `n_size < 0` 或 `n_size > max_ions`（文本尾巴）即结束。
- **不做**分组；分组到肽段由 `speclib` 按 `chg_max` 锁步完成。

## `read_chg_max_from_trailer(path: str, tail_bytes: int = 8192) -> int`

- 从文件末尾文本尾巴解析 `chg_max`。
- **异常**：`ValueError`（无法解析出形如 `1..C` 的尾巴行）。
- 前提：库至少 2 个肽段（保证末尾有干净尾巴行）。

## 示例

```python
from spectrum.speclib.predictions import (
    read_rt_pred, iter_ms2_records, read_chg_max_from_trailer)

rt = read_rt_pred("pepdata.rt.predb")           # array('f')
chg_max = read_chg_max_from_trailer("pepdata.ms2.predb")   # 4
for ions in iter_ms2_records("pepdata.ms2.predb"):
    ...  # ions: list[FragIon]
```
