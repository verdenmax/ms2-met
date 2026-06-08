# speclib — API 参考（`spectrum/speclib/speclib.py`）

## `MassValidationReport` (dataclass)

字段：`total: int`、`passed: int`、`failed: int`、`max_abs_error: float`、`failures: list`（元素 = `(index, seq, computed, stored, err)`，最多 20 条）。

## `class SpecLib`

构造一般经 `open` / `open_dir`，不直接 `__init__`。

### `SpecLib.open(*, pepdata_path, rt_path, ms2_path, fasta_path, mod_path) -> SpecLib`
显式三路径打开。open 时加载 proteins/mods/RT/chg_max；pdb、ms2 延迟到流式读。
**异常**：`ValueError`（chg_max 不在 [1,6]）。

### `SpecLib.open_dir(library_dir, *, fasta_path, mod_path) -> SpecLib`
目录内固定 `pepdata.pdb`/`pepdata.rt.predb`/`pepdata.ms2.predb`。

### `SpecLib.num_peptides -> int`
= `len(self.rt)` = M。

### `SpecLib.chg_max -> int`
从 MS2 尾巴解析（实测 4）。

### `SpecLib.iter_peptides()`
锁步流式生成器，逐 `LibPeptide`（已填 `pred_rt: float`、`pred_ms2: dict[int, list[FragIon]]`）。
**异常**：`ValueError`（肽段数与 RT 数不符 / MS2 记录耗尽）。

### `SpecLib.validate_masses(element_path, aa_path, tol=0.01, limit=None) -> MassValidationReport`
流式质量交叉校验。`limit` 限制校验条数。

## 示例

```python
from spectrum.speclib import SpecLib

lib = SpecLib.open_dir("lib-2th",
                       fasta_path="merge_human_ecoli_yeast.fasta",
                       mod_path="modification.ini")
print(lib.num_peptides, lib.chg_max)
for pep in lib.iter_peptides():
    print(pep.sequence, pep.pred_rt, pep.pred_ms2[2][:3])
    break

rep = lib.validate_masses("element.ini", "aa.ini", tol=0.01, limit=100000)
print(rep.passed, "/", rep.total, "max_err", rep.max_abs_error)
```
