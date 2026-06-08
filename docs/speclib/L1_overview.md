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

> Task 5 回填（`SpecLib.open_dir` + `iter_peptides` 示例、`python -m tools.speclib_inspect` 命令）。

## 组件索引

- [config_io](parts/config_io/L2_role.md)
- pepdata（待实现）
- predictions（待实现）
- speclib（待实现）
- speclib_inspect（待实现）

## 关键事实（真实库 `lib-2th` 实测）

- 库为**目录**：`pepdata.pdb` / `pepdata.rt.predb` / `pepdata.ms2.predb`（+ `model_*.pt` 忽略）。
- M（肽段变体数）= 3,124,520；FASTA 59,490 蛋白。
- RT = M 个 f32（分钟）。
- MS2 = `[M×chg_max 二进制记录][M 行文本尾巴]`，**chg_max = 4**；读取须在 `n_size>1000` 处停止跳过尾巴。
- 中性质量交叉校验 **100%**（max_err = 0）。
