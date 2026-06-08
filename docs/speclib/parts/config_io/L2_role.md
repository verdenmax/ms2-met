# config_io — 职责与接口

## 一句话职责

解析谱库解码所需的全部**文本配置**，输出内存结构供二进制读取与质量校验使用。

## 对外接口

| 符号 | 签名 | 简述 |
|---|---|---|
| `Protein` | dataclass(`ac`, `description`, `sequence`) | FASTA 一条蛋白；`.is_decoy` = AC 以 `REV_` 开头 |
| `ModEntry` | dataclass(`mod_id`, `name`, `mono_mass`, `sites`, `mod_type`) | modification.ini 一条修饰 |
| `parse_fasta(path)` | → `list[Protein]` | 按文件顺序，list 下标 = pdb 的 `pro_id` |
| `parse_modifications(path)` | → `list[ModEntry]` | 过滤后 read-order 赋 1-based id |
| `parse_element_masses(path)` | → `dict[str,float]` | 元素 → 最高丰度同位素质量 |
| `parse_residue_masses(path, element_masses)` | → `dict[str,float]` | 残基 → 质量（不含水）|
| `water_mass(element_masses)` | → `float` | H₂O = 2·H + O |

## 依赖

- 依赖：仅标准库（纯文本解析）。
- 被依赖：`pepdata`（需 `Protein`/`ModEntry`）、`speclib`（质量校验需元素/残基质量）。

## 输入 / 输出

- 输入：`merge_human_ecoli_yeast.fasta`、`modification.ini`、`element.ini`、`aa.ini`（latin-1 读取）。
- 输出：上表的 dataclass 列表 / 质量字典。
