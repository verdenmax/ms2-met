# tools — 细节

## tools/extract_common.py — N 引擎交并集

### 核心算法（`extract_n_engines_from_psms` / `_dual`）

- 支持引擎：`pfind`、`diann`、`alphadia`（`SUPPORTED_ENGINES`）。
- key 用 `PSMInfo.get_key_with_raw()`（sequence + charge + modify + raw_title）。
- **正例**：所有引擎 key 集合的**交集**，且权威 PSM 的 `protein_names` 命中 `positive_marker`（`matches_species_marker`，suffix + decoy-aware）→ `label_type="positive"`。
- **负例**：所有引擎 key 集合的**并集**，去掉正例 key，且权威 PSM 不命中 marker → `label_type="negative"`。
- `positive_marker=None` → 仅取交集，不打 label。
- **权威 PSM 选择**：若 `diann` 在引擎列表中则 diann 优先（蛋白归属更可靠），否则按 `engine_order` 先到先得。marker 检查只看权威 PSM。
- 进入算法前先把所有输入 PSM 的 `_label_type` 清空，防止重复调用残留 stale state。

### 双 FDR 阈值（`load_engine_psms_dual` / `extract_..._dual`）

- 每个引擎读 `qvalue_threshold`（tight，默认 0.01）与 `negative_qvalue_threshold`（loose，缺省=tight）。
- 正例用 **tight** 池交集；负例用 **loose** 池并集。要求 loose ⊇ tight，故 `loose < tight` 直接报错。
- 两阈值相等时 tight/loose 指向同一 list（不重复 I/O）。
- 权威 PSM 在 **loose** 池查找，保证 tight 交集 key 也一定能找到 PSM。
- `load_engine_psms`（单阈值）保留为向后兼容薄封装。

### no-label-site 过滤（标记位点感知，`_parse_labeling` / `filter_by_label_site`）

- 目的：SILAC 轻重标验证只能作用在**有重标位点**的肽上。无标记位点的肽其"重标版"≡轻标，本工具原理上无法验证，落在能力边界外（speclib spec §12 类4）。
- **正负例都剔**（标记位点与标签无关）；**默认开、无条件运行**，不依赖 `[entrapment]` 段。
- 判据来自 `spectrum.psm_info.has_label_site(seq, heavy_type)`：SILAC → 序列含 **K 或 R**；CHEAVY(¹³C)/NHEAVY(¹⁵N) → 全原子代谢标记，任何肽必含 C 和 N → 恒 True → **no-op（一条不剔）**；空序列 → False（剔）。
- 标记方案由 `[extract] labeling` 决定（缺省 `silac`，向后兼容），大小写不敏感，别名：`silac`；`c13/13c/cheavy`；`n15/15n/nheavy`。非法值 → `ValueError`（fail-fast）。
- 调用点：`extract_n_engines` 中、`label_type` 已设置之后、**先于** entrapment L0/L1 过滤（均为删行）。日志打印剔除的 positive / negative 数。

### entrapment 过滤

- 目的：剔除质谱不可分的"伪负例"，避免污染 SILAC 配对验证负样本。
- 两条来源（`[entrapment]` 段，**显式 TSV > 派生**）：
  1. `classified_tsv`：加载 proteinCopilot/`entrapment_classify` 产出的 classified.tsv。
  2. 否则 `target_fasta`：对刚构造出的 negative 内联调 `classify_peptide`，省去落盘往返。
  - 两者同时给出时用 `classified_tsv` 并 warn。
- `drop_levels` 默认 `L0,L1`（`DEFAULT_DROP_LEVELS`），大小写不敏感。
- L0-L4 语义：L0 razor-error / L1 LI-isomer（均不可分）/ L2 near-identical / L3 homolog / L4 true-trap（理想负样本）。

### classified.tsv 加载坑（`load_entrapment_classifications`）

- 匹配键 `(sequence, charge, raw_title)`，**忽略 modify**（L0/L1 是 stripped sequence 层判定）。
- `keep_default_na=False`：防止肽段 "NA"/"NaN" 被 pandas 当成缺失。
- charge 兼容 `"2"`/`"2.0"`/`" 3 "`（`int(float(...))`）。
- 必需列：`peptide, charge, spectrum_file, level`；缺列报错。
- 跳过：空 level、非法 level（非 L0-L4，warn）、`group != "trap"`、空 peptide/raw。
- 同 key 多行（不同 modify variant）合并：level 一致静默；不一致 warn 并保留**更严重者**（`_LEVEL_SEVERITY` L0<L1<...<L4，数字越小越严重）。

### `filter_by_entrapment` 规则

- 只动 `label_type=="negative"` 的 PSM；positive 与其它 label（None 等）一律保留。
- negative 命中 `drop_levels` → 剔除；不在 classifications 或 level 不在 drop_levels → 保留（unknown 计数）。
- 匹配只看 `(_sequence, _charge, _raw_title)`，忽略 modify。

### CLI / 边界

- 配置文件读不到或无 section、缺 `[extract]` 段 → 报错 `sys.exit(1)`。
- 输出经 `write_psms_to_json`（`PSMInfo.to_dict()`），自动建父目录。日志同时走 `RichHandler` + 文件。

---

## tools/eval_baseline.py — Baseline 评估

### 标签派生（`derive_binary_label`，三级优先）

1. `label_type` 列（`positive`→1 / `negative`→0），未映射置 -1。
2. `label` 列已是 0/1（兼容 pair flow）。
3. 从 `protein_names` 经 `matches_species_marker(marker)` 派生（与 extract_common 共享规则）。
- 三者皆无 → `ValueError`。

### 特征与训练

- 特征列 = 全列 − `META_COLUMNS`（sequence/charge/raw_title*/protein_names/label*/precursor_mz/sequence_len）。
- **不 fillna(0)**：`HistGradientBoostingClassifier` 原生支持 NaN，仅把 ±inf 替换为 NaN（避免把"无数据"与"值=0"混淆）。
- label 既非 0 也非 1 的行被过滤并 warn。
- 模型固定超参：`max_iter=300, lr=0.05, max_depth=6, l2=1.0, class_weight="balanced", random_state=42`。
- `StratifiedKFold(5, shuffle, seed=42)`，每折算 AUC/AUPRC/MCC（阈值 0.5），汇总 mean±std。

### 工作点（`compute_working_points`）

- 固定 negative recall = 0.95/0.90/0.80（即 fpr 0.05/0.10/0.20）。
- 阈值 = 负例分数的 `quantile(1-fpr)`，再看正例 recall。

### 特征重要性

- 全数据训练后 `permutation_importance`（`scoring="average_precision"`, `n_repeats=5`, `n_jobs=-1`），按均值降序排名。
- `--skip-importance` 可跳过（耗时）。

---

## tools/eval_feature_ablation.py — 特征分组对照

### 4 组特征（`split_features`）

| 组 | 含义 |
|---|---|
| `sequence_only` | 仅肽段序列属性（`SEQUENCE_FEATURES`：modification_count、kr_count、sequence_len、valid_fragment_ions_num、total_silac_shift、window_width、precursor_centering、heavy_in_raw）|
| `silac_only` | 全特征 − sequence 特征 |
| `silac_minus_intensity` | silac_only 再去掉绝对强度类（`INTENSITY_FEATURES`：precursor_*_max_int、*_snr* 等），避免肽段丰度泄漏 |
| `all` | 全部 |

- 全特征 = 全列 − `ID_COLUMNS`。
- 标签复用 `eval_baseline.derive_binary_label`，仅保留 0/1。
- 每组跑 5-fold CV（同 baseline 超参），输出 AUC mean±std、AUPRC、`pos_recall@neg_recall=95/90/80`。空组跳过。

### 设计取舍

- 与 `eval_baseline` 故意分离的轻量 CV（`cv_one` 只算 AUC/AUPRC + 工作点，不算特征重要性），便于快速对照"SILAC 配对信号 vs 序列先验"贡献。

---

## tools/entrapment_classify.py — 负例分级

### 流程（`classify_negatives_file`）

1. `load_target_fasta(target_fasta)` 载入 target proteome。
2. 读 negatives JSON（extract_common 产物），**仅** `label_type=="negative"` 的条目参与；非 negative / 无 sequence 跳过计数。
3. 每条 `classify_peptide(seq, target)` 得 level，写入 TSV。
4. 返回含计数 + level 分布 + target 信息的摘要 dict，`main` 打印 JSON。

### L0/L1/L4 语义（此处不算 L2/L3）

- L0 razor-error：trap stripped sequence 是 target proteome 子串。
- L1 LI-isomer：L↔I 归一化后是 target（归一化）子串。
- L4 true-trap：两者都不是（**省略 L2/L3 的 Hamming 扫描**）。

### 输出 schema / 坑

- TSV 列：`peptide, charge, precursor_mz, retention_time, scan_number, spectrum_file, protein_ids, q_value, group, level`。
- `group` 固定写 `"trap"`（extract_common loader 的过滤要求）；`scan_number` 留空（loader 不用）。
- 用途：去掉 proteinCopilot 依赖，ms2-met 单条命令即可自建干净负例集；产出可直接填进 extract_common `[entrapment] classified_tsv` + `drop_levels = L0, L1`。
- 错误处理：negatives 不存在 → `FileNotFoundError` → `exit(1)`；其它异常 `logging.exception` + `exit(1)`。

---

## tools/trap_domain_filter.py — 超出工具适用域的 trap 过滤（spec §12）

对**提取后**的 `features.csv` 做"删行"清洗：只评估 `label_type == "negative"`（trap）行，positive（target）永不删。

`beyond_tool_limit(level, heavy_out_of_range, has_kr=True) -> (drop, reason)` 三类丢弃：类1 同源 `level ∈ {L0,L1}`（`HOMOLOG_DROP_LEVELS`，L↔I 异构体在 human proteome，质谱不可分）→ `homolog_L0/L1`；类4 无标记位点 `has_kr == False`（无 K/R，heavy≡light）→ `no_label_site`；类3 heavy 出窗 `int(heavy_out_of_range)==1` → `heavy_out_of_window`。优先级 类1 > 类4 > 类3（三者都丢）。类2（污染物）尚未实现（TODO）。

`annotate_traps(df, target_index) -> df`：给每行加 `entrap_level` / `domain_drop` / `domain_reason`；positive 行记 `target`、不丢；trap 行 `classify_peptide` 得 level、`has_kr = has_label_site(seq)`（SILAC），再调 `beyond_tool_limit`。

CLI：`--features`（labeled features.csv）/ `--human-fasta`（target，L0/L1 用）/ `--output`（cleaned features.csv = targets + 保留 trap）。日志打印 dropped/kept、drop reason 与 entrap level 分布。

---

## tools/speclib_inspect.py — 谱库流式检视

流式加载 pFind 谱库，打印摘要（肽段数 / chg_max / RT 范围 / 前 N 条肽段的 mods+mass+RT+top3 MS2），可选做质量交叉校验。

`summarize(*, library_dir, fasta_path, mod_path, element_path=None, aa_path=None, n_samples=5, tol=0.01, mass_limit=None) -> str`：`SpecLib.open_dir(...)` + `iter_peptides()` 流式取前 `n_samples` 条（不全载）。给了 `--element` 与 `--aa` 才跑 `validate_masses(..., tol, limit=mass_limit)`，打印 `pass/total`、`max_abs_err` 与前 5 条失败；否则 "mass validation skipped"。依赖 `spectrum.speclib.SpecLib`。

---

## tools/speclib_sanity.py — 预测/观测相似度 sanity gate（spec §4.0）

构建任何"预测强度"特征**之前**的 go/no-go：confident light PSM 上，谱库预测碎片强度与观测 light 谱是否一致。

纯核心（单测覆盖）：`similarity_distribution(pairs, metric=spectral_angle) -> {n,median,p25,p75}`；`gate_pass(stats, min_sim) -> bool`（`n>0` 且 median 有限且 `>min_sim`）；`build_pairs_from_maps(pred_map, obs_map) -> (pred_vec,obs_vec)`（共同碎片稳定排序对齐）；`filter_psms_by_raw(psms, raw_title)`（`None` 不过滤）。

main：`SpecLib.open_dir` + `LightResultManager` 载 PSM；DIA 走 `--dia-npz`（mmap，优先）或 `--raw`；`build_pred_store` 算覆盖率；每 PSM 取 pred/obs 共同碎片（≥2）算相似度 + gate。**退出码 0=PASS / 2=FAIL**。依赖 `spectrum.speclib`、`spectrum.dia_data`、`manager.*`、`workflows.pred_*`。
