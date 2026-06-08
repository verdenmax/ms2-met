# speclib_inspect — 职责与接口

## 一句话职责

命令行工具：流式加载真实谱库（~4.4GB 不 OOM），打印摘要 + 样例肽段 + 质量交叉校验，用于人工验证读取正确性。

## 对外接口

| 符号 | 签名 | 简述 |
|---|---|---|
| `summarize(*, library_dir, fasta_path, mod_path, element_path=None, aa_path=None, n_samples=5, tol=0.01, mass_limit=None)` | → `str` | 生成多行摘要文本 |
| `main()` | — | argparse CLI 入口 |

## CLI 参数

| 参数 | 必填 | 说明 |
|---|---|---|
| `--library-dir` | 是 | 谱库目录 |
| `--fasta` | 是 | 数据库 FASTA |
| `--mod` | 是 | modification.ini |
| `--element` / `--aa` | 否 | 提供则做质量交叉校验 |
| `--n-samples` | 否 | 打印样例肽段数（默认 5）|
| `--tol` | 否 | 质量容差（默认 0.01 Da）|
| `--mass-limit` | 否 | 只校验前 N 条质量（加速）|

## 依赖

- 依赖：`spectrum.speclib.SpecLib`。
- 被依赖：无（终端工具）。
