# speclib — 细节

## open / open_dir

- `open_dir` 固定三文件名：`pepdata.pdb` / `pepdata.rt.predb` / `pepdata.ms2.predb`。
- `open` 时即加载：`parse_fasta` → proteins；`parse_modifications` → `mods_by_id`；`read_rt_pred` → RT 数组（全量，~12MB）；`read_chg_max_from_trailer` → chg_max，并硬校验 `1 ≤ chg_max ≤ 6`。
- pdb / ms2 **不在 open 时读取**，留给 `iter_peptides` / `validate_masses` 流式处理（体量大）。

## iter_peptides（锁步流式，核心）

```
ms2 = iter_ms2_records(ms2_path)          # 流式记录
for i, pep in enumerate(iter_pepdata(...)):  # pdb 流
    pep.pred_rt = rt[i]                    # RT 按下标
    pep.pred_ms2 = {chg: next(ms2) for chg in 1..chg_max}
    yield pep
```

- **同序保证**：pdb 变体顺序 == RT 顺序 == MS2 记录"肽段外层"顺序，故三者按下标/顺序对齐，无需额外 key。
- 每肽段消费 `chg_max` 条 MS2 记录（实测 4）。
- **对齐校验**：`i >= n_rt` → 肽段多于 RT 报错；`next(ms2)` StopIteration → MS2 记录耗尽报错；遍历结束 `i+1 != n_rt` → 肽段少于 RT 报错。
- 内存 O(1)：调用方即用即弃；适合批处理/导出/验证。随机按肽段查 MS2 需缓存偏移索引（本步不做）。

## validate_masses（质量交叉校验）

- 流式遍历 `iter_pepdata`，对每肽段独立重算中性质量 `= water + Σ残基 + Σ修饰单同位素质量`，与存储 `neutral_mass` 比较。
- `tol` 容差（默认 0.01 Da）；`limit` 只校验前 N 条（加速）；最多记录 20 条失败样例。
- **质量公式经 C++ 验证**：`_m2mz`（sdk.h:881）证明存储质量不含质子；y 离子用 `MOLECULE_MASS_H2O`（Instrument.cpp）证明含水；残基质量来自 aa.ini（不含水）。真实 `lib-2th` 实测 100% 通过、max_err=0。

## 设计取舍

- 为何流式而非全量：ms2 ~4.4GB、pdb ~312 万肽段，全量物化成对象会吃几十 GB；锁步流式把三路文件按同序拉链，内存 O(1)。
- RT 例外全量加载：仅 ~12MB，且需要按下标随机取（`rt[i]`）。
