# tools — API 参考

## tools/extract_common.py

通用 N 引擎交并集数据集构造工具。

### 常量

- `SUPPORTED_ENGINES = {"pfind", "diann", "alphadia"}`
- `DEFAULT_DROP_LEVELS = {"L0", "L1"}`、`VALID_ENTRAPMENT_LEVELS = {"L0".."L4"}`
- `_LEVEL_SEVERITY`：L0=0 … L4=4（数字越小越严重）。

### 主要函数

- `load_engine_psms(engine_name, config) -> list[PSMInfo]`：单 FDR 阈值加载（向后兼容薄封装）。
- `load_engine_psms_dual(engine_name, config) -> dict`：返回 `{"tight": [...], "loose": [...]}`；读 `qvalue_threshold` / `negative_qvalue_threshold`（缺省=tight），`loose < tight` 抛 `ValueError`，相等时两 key 共享同一 list。
- `extract_n_engines_from_psms(engine_psms, engine_order, positive_marker=None) -> list[PSMInfo]`：单池正负例构造。
- `extract_n_engines_from_psms_dual(engine_psms_dual, engine_order, positive_marker=None) -> list[PSMInfo]`：正例用 tight 交集、负例用 loose 并集。
- `load_entrapment_classifications(tsv_path) -> dict[(seq,charge,raw)->level]`：加载 classified.tsv（含校验/跳过/冲突合并），路径不存在抛 `FileNotFoundError`，缺列抛 `ValueError`。
- `filter_by_entrapment(psms, classifications, drop_levels=DEFAULT_DROP_LEVELS) -> list[PSMInfo]`：剔除命中 drop_levels 的 negative。
- `extract_n_engines(config) -> list[PSMInfo]`：顶层装配（加载引擎 + 构造 + 可选 entrapment 过滤）。
- `write_psms_to_json(psms, output_path)`：序列化 `PSMInfo.to_dict()`，自动建目录。

### 配置（INI）

```ini
[extract]
engines = pfind, diann
positive_species_marker = HUMAN
result_file = ./datasets/hela_2da.json

[engine.pfind]
path = .../pfind_result.txt
qvalue_threshold = 0.01
negative_qvalue_threshold = 0.05   ; 可选，loose ≥ tight

[entrapment]                        ; 可选
classified_tsv = ./datasets/entrapment_classified.tsv
; 或 target_fasta = /path/human_swissprot.fasta（内联分类）
drop_levels = L0, L1
```

### CLI / 运行示例

| 参数 | 默认 | 说明 |
|---|---|---|
| `--configpath` | `./extract_common_config.ini` | 配置文件 |
| `--logpath` | `./extract_common.log` | 日志文件 |

```bash
python -m tools.extract_common \
  --configpath ./extract_common_config.ini \
  --logpath ./runs/extract.log
```

---

## tools/eval_baseline.py

### 主要函数

- `derive_binary_label(df, marker="HUMAN") -> pd.Series`：三级优先派生 0/1 标签（label_type → 数值 label → protein_names），皆无抛 `ValueError`。
- `load_features(path, marker="HUMAN") -> (X, y, feature_cols)`：读 CSV，剔除 `META_COLUMNS`，±inf→NaN，过滤非 0/1 行。
- `compute_working_points(y_true, y_score) -> dict`：neg_recall 95/90/80 工作点。
- `cv_evaluate(X, y, n_splits=5, random_state=42) -> dict`：5-fold CV，含 fold 指标、mean±std、working_points。
- `compute_feature_importance(X, y, feature_cols, random_state=42, n_repeats=5) -> list[dict]`：permutation importance（AUPRC drop）降序。

### CLI / 运行示例

| 参数 | 必填 | 说明 |
|---|---|---|
| `--features` | 是 | features.csv |
| `--output` | 是 | 指标 JSON 输出 |
| `--skip-importance` | 否 | 跳过特征重要性 |
| `--positive-marker` | 否 | 默认 `HUMAN`，仅 protein_names 回退层用 |

输出 JSON：`n_samples / n_positive / n_negative / n_features / cv_summary / feature_importance`。

```bash
python tools/eval_baseline.py \
  --features runs/baseline_2da/features.csv \
  --output runs/baseline_2da/baseline_metrics.json
```

---

## tools/eval_feature_ablation.py

### 主要函数

- `split_features(all_features) -> dict[str, list[str]]`：分出 `sequence_only / silac_only / silac_minus_intensity / all`。
- `cv_one(X, y, name, n_splits=5) -> dict`：单组 5-fold CV，返回 `name / n_features / auc_mean / auc_std / auprc_mean / working_points`。
- 复用 `eval_baseline.derive_binary_label`。

### CLI / 运行示例

| 参数 | 必填 | 说明 |
|---|---|---|
| `--features` | 是 | features.csv |
| `--output` | 是 | 各组结果 list 的 JSON |

```bash
python tools/eval_feature_ablation.py \
  --features runs/baseline_2da/features.csv \
  --output runs/baseline_2da/ablation.json
```

---

## tools/entrapment_classify.py

### 主要函数

- `classify_negatives_file(negatives_path, target_fasta_path, output_path) -> dict`：分类 negatives JSON 中的 negative PSM，写 classified.tsv，返回计数摘要（total_psms_in_input / classified_as_negative / skipped_* / level_distribution / target_proteome）。negatives 不存在抛 `FileNotFoundError`。
- 依赖 `spectrum.entrapment_classifier.classify_peptide / load_target_fasta`。

### CLI / 运行示例

| 参数 | 必填 | 说明 |
|---|---|---|
| `--negatives` | 是 | extract_common 产出的 negatives JSON |
| `--target-fasta` | 是 | target proteome FASTA（如 HUMAN SwissProt）|
| `--output` | 是 | 输出 classified.tsv 路径 |
| `--logpath` | 否 | 额外日志文件（同时写 stderr）|

输出 TSV schema 与 `extract_common.load_entrapment_classifications` 兼容；stdout 打印摘要 JSON。

```bash
python -m tools.entrapment_classify \
  --negatives datasets/hela_2da_pfind_diann.json \
  --target-fasta /path/to/human_swissprot.fasta \
  --output datasets/entrapment_classified.tsv
```
