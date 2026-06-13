# tools — API 参考

## tools/extract_common.py

通用 N 引擎交并集数据集构造工具。

### 常量

- `SUPPORTED_ENGINES = {"pfind", "diann", "alphadia"}`
- `DEFAULT_DROP_LEVELS = {"L0", "L1"}`、`VALID_ENTRAPMENT_LEVELS = {"L0".."L4"}`
- `_LEVEL_SEVERITY`：L0=0 … L4=4（数字越小越严重）。
- `_LABELING_ALIASES`：`silac`→SILAC；`c13`/`13c`/`cheavy`→CHEAVY；`n15`/`15n`/`nheavy`→NHEAVY。

### 主要函数

- `load_engine_psms(engine_name, config) -> list[PSMInfo]`：单 FDR 阈值加载（向后兼容薄封装）。
- `load_engine_psms_dual(engine_name, config) -> dict`：返回 `{"tight": [...], "loose": [...]}`；读 `qvalue_threshold` / `negative_qvalue_threshold`（缺省=tight），`loose < tight` 抛 `ValueError`，相等时两 key 共享同一 list。
- `extract_n_engines_from_psms(engine_psms, engine_order, positive_marker=None) -> list[PSMInfo]`：单池正负例构造。
- `extract_n_engines_from_psms_dual(engine_psms_dual, engine_order, positive_marker=None) -> list[PSMInfo]`：正例用 tight 交集、负例用 loose 并集。
- `load_entrapment_classifications(tsv_path) -> dict[(seq,charge,raw)->level]`：加载 classified.tsv（含校验/跳过/冲突合并），路径不存在抛 `FileNotFoundError`，缺列抛 `ValueError`。
- `filter_by_entrapment(psms, classifications, drop_levels=DEFAULT_DROP_LEVELS) -> list[PSMInfo]`：剔除命中 drop_levels 的 negative。
- `_parse_labeling(config) -> HeavyType`：读 `[extract] labeling`（缺省 `silac`），大小写不敏感别名映射；非法值抛 `ValueError`。
- `filter_by_label_site(psms, heavy_type) -> list[PSMInfo]`：剔除在 `heavy_type` 下无标记位点的 PSM（正负例都剔）。SILAC 剔无 K/R；CHEAVY/NHEAVY 为 no-op。委托 `spectrum.psm_info.has_label_site`。
- `filter_by_contaminant(psms, contaminant_index, match_li=True) -> list[PSMInfo]`：剔除映射到污染蛋白(cRAP)的 PSM（正负例都剔）。复用 `spectrum.entrapment_classifier.load_target_fasta`+`classify_peptide`：精确子串(L0)，或 `match_li` 时 L↔I 子串(L1)，即判为污染。
- `extract_n_engines(config) -> list[PSMInfo]`：顶层装配（加载引擎 + 构造 + **无条件 no-label-site 过滤** + 可选污染库过滤 + 可选 entrapment 过滤）。
- `write_psms_to_json(psms, output_path)`：序列化 `PSMInfo.to_dict()`，自动建目录。

### 配置（INI）

```ini
[extract]
engines = pfind, diann
positive_species_marker = HUMAN
labeling = silac                     ; 可选，缺省 silac，选择标记方案（silac/c13/n15）
result_file = ./datasets/hela_2da.json

[engine.pfind]
path = .../pfind_result.txt
qvalue_threshold = 0.01
negative_qvalue_threshold = 0.05   ; 可选，loose ≥ tight

[entrapment]                        ; 可选
classified_tsv = ./datasets/entrapment_classified.tsv
; 或 target_fasta = /path/human_swissprot.fasta（内联分类）
drop_levels = L0, L1

[contaminant]                       ; 可选（污染库过滤，正负例都剔）
fasta = /path/contaminant.fasta
; match_li = true                   ; 缺省 true（L0+L↔I）；false 仅精确子串 L0
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

---

## tools/trap_domain_filter.py

- 常量 `HOMOLOG_DROP_LEVELS = {"L0", "L1"}`（类1 质谱不可分）。
- `beyond_tool_limit(level, heavy_out_of_range, has_kr=True) -> (bool, str|None)`：判定 trap 是否超出 SILAC 工具适用域。优先级 类1(`homolog_L0/L1`) > 类4(`no_label_site`) > 类3(`heavy_out_of_window`)。
- `annotate_traps(df, target_index) -> pd.DataFrame`：为 negative 行加 `entrap_level/domain_drop/domain_reason`；positive 标 `target`、不丢。
- `has_label_site`：从 `spectrum.psm_info` 重导出。
- CLI：`--features`（labeled features.csv，必填）/ `--human-fasta`（human proteome FASTA，必填）/ `--output`（cleaned features.csv = targets + 保留 trap，必填）。

```bash
python -m tools.trap_domain_filter --features runs/X/features.csv \
  --human-fasta human_swissprot.fasta --output runs/X/features.clean.csv
```

---

## tools/speclib_inspect.py

- `summarize(*, library_dir, fasta_path, mod_path, element_path=None, aa_path=None, n_samples=5, tol=0.01, mass_limit=None) -> str`：返回摘要文本；给了 `--element`+`--aa` 才做质量校验。
- CLI：`--library-dir --fasta --mod`（必填）；`--element --aa`（启用质量校验）；`--n-samples`(5) `--tol`(0.01) `--mass-limit`(None)。

```bash
python -m tools.speclib_inspect --library-dir DIR --fasta merge.fasta \
  --mod modification.ini --element element.ini --aa aa.ini --n-samples 5 --tol 0.01
```

---

## tools/speclib_sanity.py

- `similarity_distribution(pairs, metric=spectral_angle) -> dict`（`{n,median,p25,p75}`）；`gate_pass(stats, min_sim) -> bool`；`build_pairs_from_maps(pred_map, obs_map) -> (pred_vec,obs_vec)`；`filter_psms_by_raw(psms, raw_title) -> list`。
- CLI：`--library-dir --fasta --mod --psm-file`（必填）；`--raw` / `--dia-npz`（二选一，npz mmap 优先）；`--search-engine-type`(3) `--raw-title`(None) `--metric`(spectral_angle|spearman) `--min-sim`(0.7) `--mass-tol-ppm`(10.0) `--xic-cycle-window`(6) `--limit`(2000)。退出码 0=PASS / 2=FAIL。

```bash
python -m tools.speclib_sanity --library-dir DIR --fasta f.fasta \
  --mod modification.ini --psm-file psms.json --dia-npz raw.dia.npz --min-sim 0.7
```
