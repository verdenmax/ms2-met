# speclib — 职责与接口

## 一句话职责

顶层 loader：把 `config_io` + `pepdata` + `predictions` 拼成**锁步流式**接口，逐肽段产出预测，并提供质量交叉校验。

## 对外接口

| 符号 | 签名 | 简述 |
|---|---|---|
| `SpecLib.open(*, pepdata_path, rt_path, ms2_path, fasta_path, mod_path)` | → `SpecLib` | 显式三路径打开 |
| `SpecLib.open_dir(library_dir, *, fasta_path, mod_path)` | → `SpecLib` | 目录内三文件名固定 |
| `SpecLib.num_peptides` | property → `int` | M（= RT 数）|
| `SpecLib.chg_max` | 属性 | 从尾巴解析（约束 [1,6]）|
| `SpecLib.iter_peptides()` | → 生成器[`LibPeptide`] | 锁步逐肽段，已填 `pred_rt`/`pred_ms2` |
| `SpecLib.validate_masses(element_path, aa_path, tol=0.01, limit=None)` | → `MassValidationReport` | 流式质量交叉校验 |
| `MassValidationReport` | dataclass | `total/passed/failed/max_abs_error/failures` |

## 依赖

- 依赖：`config_io` + `pepdata` + `predictions`。
- 被依赖：`tools/speclib_inspect`（CLI）；未来 SILAC pipeline 接入。

## 输入 / 输出

- 输入：谱库目录（或三显式路径）+ FASTA + modification.ini；质量校验另需 element.ini + aa.ini。
- 输出：`LibPeptide` 流（含预测）/ 质量校验报告。
