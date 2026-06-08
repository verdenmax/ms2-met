# pepdata — API 参考（`spectrum/speclib/pepdata.py`）

## `ModSite` (dataclass, slots)

字段：`pos: int`、`mod_id: int`、`name: str = ""`、`mono_mass: float = 0.0`。

## `LibPeptide` (dataclass, slots)

| 字段 | 类型 | 说明 |
|---|---|---|
| `sequence` | str | 肽段序列 |
| `mods` | list[ModSite] | 修饰位点 |
| `neutral_mass` | float | pdb 存储的中性质量 |
| `protein` | str | 首个蛋白 AC |
| `is_decoy` | bool | AC 以 `REV_` 开头 |
| `pro_nc` / `enz` / `miss` | int | 头中的 N/C 端属性 / 酶切特异性 / 漏切 |
| `charge_mask` | int | pdb 读取恒 0 |
| `pred_rt` | float \| None | 预测 RT（speclib 锁步填充）|
| `pred_ms2` | dict[int, list[FragIon] \| np.ndarray] | 电荷→离子（speclib 锁步填充；`decode_ms2="arrays"` 时为 numpy 结构化数组，`"none"` 时为 `{}`）|

## `iter_pepdata(path, proteins, mods_by_id, validate_bytes=True)`

- **参数**：`path` pdb 路径；`proteins: list[Protein]`（下标=pro_id）；`mods_by_id: dict[int, ModEntry]`；`validate_bytes` 是否做 mod_pep_bytes 校验。
- **产出**：生成器，逐 `LibPeptide`。
- **异常**：`ValueError("mod_pep_bytes mismatch ...")`（校验开启且字节不符）；`IndexError`（pro_id 越界，即 fasta 不匹配）。

## `read_pepdata(...) -> list[LibPeptide]`

`iter_pepdata` 的 list 包装；仅用于小数据 / 测试（真实库请用 `iter_pepdata` 流式）。

## 示例

```python
from spectrum.speclib.pepdata import iter_pepdata
for pep in iter_pepdata("pepdata.pdb", proteins, mods_by_id):
    ...  # pep.sequence, pep.mods, pep.neutral_mass
```
