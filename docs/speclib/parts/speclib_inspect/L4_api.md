# speclib_inspect — API 参考（`tools/speclib_inspect.py`）

## `summarize(*, library_dir, fasta_path, mod_path, element_path=None, aa_path=None, n_samples=5, tol=0.01, mass_limit=None) -> str`

- **参数**：
  - `library_dir`：谱库目录（含 `pepdata.pdb`/`pepdata.rt.predb`/`pepdata.ms2.predb`）。
  - `fasta_path` / `mod_path`：解码用 FASTA / modification.ini。
  - `element_path` / `aa_path`：可选；提供则做质量交叉校验，否则跳过。
  - `n_samples`：打印样例肽段数。
  - `tol`：质量容差（Da）。
  - `mass_limit`：质量校验条数上限（None=全量）。
- **返回**：多行摘要字符串（peptides / chg_max / rt range / 样例 / mass pass 或 skip）。
- **异常**：透传 `SpecLib` 的异常（如 chg_max 越界、文件缺失、对齐错误）。

## `main()`

argparse CLI 入口；将命令行参数转交 `summarize` 并 `print`。

## 运行示例

```bash
python -m tools.speclib_inspect \
  --library-dir <谱库目录> \
  --fasta merge_human_ecoli_yeast.fasta --mod modification.ini \
  --element element.ini --aa aa.ini \
  --n-samples 10 --tol 0.01 --mass-limit 200000
```
