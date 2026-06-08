# speclib — pFind 谱库读取模块

> L1 概览。本文档随各组件实现逐步补全；详见各组件 `parts/<组件>/`。

## 目标

独立、自校验、**内存安全（流式）**地读取 pFind 经 pPred 生成的谱库：解码出 M 个肽段，并为每个肽段提供预测 RT 与预测 MS2。本步不接入 SILAC 特征提取 pipeline。

## 架构

```
config_io   ──┐
              ├─> speclib (SpecLib) ──> tools/speclib_inspect (CLI)
pepdata     ──┤
predictions ──┘
```

- `config_io`：解析 FASTA / modification.ini / element.ini / aa.ini。
- `pepdata`：流式解析 `pepdata.pdb` → `LibPeptide`。
- `predictions`：读 `pepdata.rt.predb`（RT 数组）+ 流式读 `pepdata.ms2.predb`（跳过文本尾巴）。
- `speclib`：顶层锁步流式 loader + 质量自校验。
- `speclib_inspect`：真实库验证 CLI。

## 数据流

谱库目录（`pepdata.pdb` / `pepdata.rt.predb` / `pepdata.ms2.predb`）+ FASTA + modification.ini
→ 解析配置 → 锁步流式逐肽段产出 `LibPeptide(序列, 修饰, 质量, pred_rt, pred_ms2)`。

## 快速上手

```python
from spectrum.speclib import SpecLib

lib = SpecLib.open_dir(
    "lib-2th",
    fasta_path="merge_human_ecoli_yeast.fasta",
    mod_path="modification.ini")
print(lib.num_peptides, lib.chg_max)          # 3124520 4

for pep in lib.iter_peptides():               # 锁步流式，内存 O(1)
    print(pep.sequence, pep.mods, pep.pred_rt, pep.pred_ms2[2][:3])
    break

# 质量交叉校验（流式；--limit 控制条数）
rep = lib.validate_masses("element.ini", "aa.ini", tol=0.01, limit=300000)
print(rep.passed, "/", rep.total, "max_err", rep.max_abs_error)
```

CLI（真实库验证，不 OOM）：

```bash
python -m tools.speclib_inspect \
  --library-dir <谱库目录> \
  --fasta merge_human_ecoli_yeast.fasta --mod modification.ini \
  --element element.ini --aa aa.ini \
  --n-samples 10 --tol 0.01 --mass-limit 300000
```

## 组件索引

- [config_io](parts/config_io/L2_role.md) — 文本配置解析
- [pepdata](parts/pepdata/L2_role.md) — 流式读 pepdata.pdb
- [predictions](parts/predictions/L2_role.md) — RT 数组 + 流式 MS2（跳尾巴）
- [speclib](parts/speclib/L2_role.md) — 锁步流式 loader + 质量校验
- [speclib_inspect](parts/speclib_inspect/L2_role.md) — 真实库验证 CLI

## 关键事实（真实库 `lib-2th` 实测）

- 库为**目录**：`pepdata.pdb` / `pepdata.rt.predb` / `pepdata.ms2.predb`（+ `model_*.pt` 忽略）。
- M（肽段变体数）= 3,124,520；FASTA 59,490 蛋白。
- RT = M 个 f32（分钟）。
- MS2 = `[M×chg_max 二进制记录][M 行文本尾巴]`，**chg_max = 4**；读取须在 `n_size>1000` 处停止跳过尾巴；`iter_ms2_records` 用 `mmap` 流式（4.4GB 峰值 RSS ~184MB）。
- 中性质量交叉校验 **100%**（首 30 万条 max_err = 0）。
