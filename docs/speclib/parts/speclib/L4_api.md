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

### `SpecLib.iter_peptides(decode_ms2="objects")`
锁步流式生成器，逐 `LibPeptide`（已填 `pred_rt: float`）。`decode_ms2` 控制 MS2：
- `"objects"`（默认）：`pred_ms2 = {charge: list[FragIon]}`。
- `"arrays"`：`pred_ms2 = {charge: np.ndarray}`（~5-8× 快、~8× 省内存）。
- `"none"`：跳过 MS2（不读 4.4GB 文件），`pred_ms2 = {}`；只需 RT/身份时最快。
**异常**：`ValueError`（decode_ms2 非法 / 肽段数与 RT 数不符 / MS2 记录耗尽或过供）。

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
