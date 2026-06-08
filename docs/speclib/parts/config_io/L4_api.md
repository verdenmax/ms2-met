# config_io — API 参考（`spectrum/speclib/config_io.py`）

## `Protein` (dataclass)

字段：`ac: str`、`description: str`、`sequence: str`。
属性：`is_decoy -> bool`（`ac.startswith("REV_")`）。

## `ModEntry` (dataclass)

字段：`mod_id: int`、`name: str`、`mono_mass: float`、`sites: str`、`mod_type: str`。

## `parse_fasta(path: str) -> list[Protein]`

按文件顺序解析所有蛋白条目。返回 list，下标即 `pro_id`。

## `parse_modifications(path: str) -> list[ModEntry]`

解析 modification.ini，返回过滤后按 read-order 赋 1-based `mod_id` 的列表。
常用：`{m.mod_id: m for m in parse_modifications(path)}` 建 `mods_by_id`。

## `parse_element_masses(path: str) -> dict[str, float]`

元素符号 → 最高丰度同位素质量。

## `parse_residue_masses(path: str, element_masses: dict[str, float]) -> dict[str, float]`

氨基酸单字母 → 残基质量（不含水）。需先有 `element_masses`。

## `water_mass(element_masses: dict[str, float]) -> float`

返回 `2*H + O`。

## 示例

```python
from spectrum.speclib.config_io import (
    parse_fasta, parse_modifications, parse_element_masses,
    parse_residue_masses, water_mass)

proteins = parse_fasta("merge_human_ecoli_yeast.fasta")
mods_by_id = {m.mod_id: m for m in parse_modifications("modification.ini")}
em = parse_element_masses("element.ini")
res = parse_residue_masses("aa.ini", em)
neutral = water_mass(em) + sum(res[a] for a in "PEPTIDEK")
```
