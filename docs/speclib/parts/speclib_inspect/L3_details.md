# speclib_inspect — 细节

## 工作流程

1. `SpecLib.open_dir` 打开库（加载 proteins/mods/RT/chg_max）。
2. 打印 `peptides`（=M）、`chg_max`、RT 范围（`min/max` 数组）。
3. **流式**取前 `n_samples` 个肽段（`iter_peptides` + 提前 break），打印序列、修饰、质量、RT、top-3 MS2 离子（按强度跨电荷桶取）。
4. 若给 `--element`+`--aa`，调 `validate_masses(limit=mass_limit)` 做质量交叉校验，打印 `mass pass: passed/total (max_abs_err=...)` 及最多 5 条失败样例。

## 为何不 OOM

- 样例只取前 N 个肽段就 break，不遍历全库。
- `validate_masses` 流式遍历 `iter_pepdata`（只读 91MB pdb，不碰 4.4GB ms2），`--mass-limit` 进一步限条数。
- `iter_peptides` 的 MS2 走 `mmap`，常驻内存 O(1)。

## 真实库实测（lib-2th，2026-06-08）

```
peptides: 3124520
chg_max: 4
rt range (min): -2.722 .. 147.668
  MKIPEAVNHINVQNNIDLVDGK mods=[0:Acetyl[ProteinN-term]] mass=2502.2901 rt=71.46 ...
  ...
mass pass: 300000/300000 (max_abs_err=0.00000, tol=0.01)
```

- `--mass-limit 300000` 约 10s；不带 limit 跑全量 312 万约 30s。
- 一个肽段头条目可有多个修饰变体（如 Acetyl[ProteinN-term] / Oxidation[M] / 无修饰），各自独立 RT/MS2，验证了锁步对齐。
- top_ms2 同一标签（如 `b2^1`）可能出现多次：来自不同前体电荷桶（charge-c 桶含 frag_charge≤c 的离子），非重复 bug。

## 运行命令

```bash
python -m tools.speclib_inspect \
  --library-dir ~/share/2026_06_07_kongweisa_guangshan_puku/lib-2th \
  --fasta ../puku/merge_human_ecoli_yeast.fasta \
  --mod ../puku/modification.ini \
  --element ../puku/element.ini --aa ../puku/aa.ini \
  --n-samples 6 --tol 0.01 --mass-limit 300000
```
