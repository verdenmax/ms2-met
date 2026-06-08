# pepdata — 职责与接口

## 一句话职责

流式解析二进制肽段库 `pepdata.pdb`，逐肽段变体产出 `LibPeptide`（序列 / 修饰 / 中性质量 / 蛋白 / decoy），不含预测值。

## 对外接口

| 符号 | 签名 | 简述 |
|---|---|---|
| `iter_pepdata(path, proteins, mods_by_id, validate_bytes=True)` | → 生成器[`LibPeptide`] | 逐条 yield，内存 O(1) |
| `read_pepdata(path, proteins, mods_by_id, validate_bytes=True)` | → `list[LibPeptide]` | `list(iter_pepdata(...))`，小数据/测试用 |
| `LibPeptide` | dataclass | 见 L4 字段表；`pred_rt`/`pred_ms2` 由 speclib 锁步填充 |
| `ModSite` | dataclass(`pos`, `mod_id`, `name`, `mono_mass`) | 一个修饰位点 |

## 依赖

- 依赖：`config_io`（`Protein`/`ModEntry`）、标准库 `struct`。
- 被依赖：`speclib`（`iter_peptides` / `validate_masses` 流式消费 `iter_pepdata`）。

## 输入 / 输出

- 输入：`pepdata.pdb` 路径 + `proteins`（下标=pro_id）+ `mods_by_id`。
- 输出：`LibPeptide` 流 / 列表。`validate_bytes=True` 时每条目做 `mod_pep_bytes` 自校验。
