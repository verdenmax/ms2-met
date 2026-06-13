# tools — 职责与接口

## 一句话职责

顶层 CLI/工具脚本集合：从多搜索引擎结果构造 SILAC 正负例数据集（`extract_common`，含标记位点 + entrapment 过滤），对负例做 entrapment 分级（`entrapment_classify`），对提取后特征做适用域清洗（`trap_domain_filter`），跑模型评估与特征分组对照实验（`eval_baseline` / `eval_feature_ablation`），并提供谱库检视与预测/观测 sanity gate（`speclib_inspect` / `speclib_sanity`）。

## 对外接口

| 脚本 | 用途 | 主要函数 | CLI 入口 |
|---|---|---|---|
| `extract_common.py` | N 引擎交并集构造正负例数据集，含无标记位点过滤 + 可选污染库过滤 + 可选 entrapment 过滤 | `extract_n_engines`、`extract_n_engines_from_psms_dual`、`load_engine_psms_dual`、`filter_by_label_site`、`filter_by_contaminant`、`_parse_labeling`、`load_entrapment_classifications`、`filter_by_entrapment`、`write_psms_to_json` | `--configpath` / `--logpath` |
| `eval_baseline.py` | 当前全特征集跑 5-fold CV，输出 AUC/AUPRC/MCC + 工作点 + 特征重要性 | `load_features`、`derive_binary_label`、`cv_evaluate`、`compute_working_points`、`compute_feature_importance` | `--features --output [--skip-importance --positive-marker]` |
| `eval_feature_ablation.py` | 4 组特征集对照 CV，衡量 SILAC 配对信号贡献 | `split_features`、`cv_one`（复用 `derive_binary_label`） | `--features --output` |
| `entrapment_classify.py` | 把负例 PSM 按 L0/L1/L4 分级，产出 classified.tsv | `classify_negatives_file` | `--negatives --target-fasta --output [--logpath]` |
| `trap_domain_filter.py` | 对提取后 features.csv 的 trap 行做适用域过滤（类1 L0/L1 同源、类3 heavy 出窗、类4 无标记位点），只删 negative | `beyond_tool_limit`、`annotate_traps` | `--features --human-fasta --output` |
| `speclib_inspect.py` | 流式检视 pFind 谱库 + 可选质量交叉校验 | `summarize` | `--library-dir --fasta --mod [--element --aa ...]` |
| `speclib_sanity.py` | 预测碎片强度 vs 观测 light 谱相似度 sanity gate（构建预测强度特征前 go/no-go）| `similarity_distribution`、`gate_pass`、`build_pairs_from_maps`、`filter_psms_by_raw` | `--library-dir --fasta --mod --psm-file [--raw/--dia-npz ...]` |

## 依赖

- 依赖：`spectrum.light_result`（多引擎 PSM 加载）、`spectrum.psm_info`、`spectrum.species_marker`（物种 marker 匹配）、`spectrum.entrapment_classifier`（`classify_peptide`/`load_target_fasta`）；`pandas`、`numpy`、`scikit-learn`、`rich`。
- 被依赖：`eval_feature_ablation` 复用 `eval_baseline.derive_binary_label`；`entrapment_classify` 与 `extract_common.load_entrapment_classifications` 共享 TSV schema。
- 排除：`tools/spec_trainer/`（在别处文档）。

## 输入 / 输出

- `extract_common`：输入 INI 配置（各引擎结果路径 + FDR 阈值 + 可选 entrapment 段）→ 输出 PSM 列表 JSON（`result_file`）。
- `entrapment_classify`：输入 negatives JSON + target FASTA → 输出 classified.tsv，stdout 打印 JSON 摘要。
- `eval_baseline` / `eval_feature_ablation`：输入 `features.csv` → 输出指标 JSON。
- `trap_domain_filter`：输入 labeled `features.csv` + human FASTA → 输出 cleaned `features.csv`（targets + 保留的 trap）。
- `speclib_inspect`：输入谱库目录 + FASTA + mod（可选 element/aa）→ stdout 摘要文本。
- `speclib_sanity`：输入谱库 + PSM + DIA（`--raw`/`--dia-npz`）→ 日志相似度统计；退出码 0=PASS / 2=FAIL。
