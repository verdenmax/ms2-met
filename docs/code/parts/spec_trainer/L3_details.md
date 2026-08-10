# spec_trainer — 细节

## 统一的评价口径

- 为兼容已有 CSV 和模型，存储标签仍为 `label=1` 表示正确鉴定、`label=0`
  表示错误鉴定，模型输出的 `trust_score=P(正确鉴定)` 越大越可信。
- 计算评价指标时统一转换为 `error_truth=1-label`、
  `error_score=1-trust_score`，即**错误鉴定是实际阳性，正确鉴定是实际阴性**。
- FP 是正确鉴定被误报为错误，FN 是错误鉴定被漏判为正确。因此 FPR 是
  正确鉴定的误报率，FNR 是错误鉴定的漏检率，error recall 为错误鉴定检出率。
- 新结果带有 `metric_semantics=error_identification_positive_v1`。没有该字段的
  历史 JSON 使用旧口径，不能只改字段名后与新结果混用。

## 训练流程（`src/main.py`）

1. 把 `src/` 插入 `sys.path`，使脚本可从任意工作目录调用。
2. 读 YAML；`target_col = cfg['data']['target_col']`。
3. `resolve_feature_cols(...)`：当 `feature_cols: []` 时自动检测特征列（见下）。
4. `load_data(train_files, feature_cols, target_col)`：逐文件 `read_csv` 后 `concat`，缺文件抛 `FileNotFoundError`。
5. `resolve_holdout(...)`：决定测试集来源（见下）。
6. 可选验证集：`training.valid_size > 0` 时再 `train_test_split`（`stratify=y`, `random_state=42`）。
7. `ModelManager.create(cfg, feature_names=feature_cols)` 建模 → `model.fit(...)`。
8. `model.save(model_path)`（自动 `makedirs`）。
9. `predict_proba` / `predict` → `evaluate_and_report(...)` 写 JSON + 画图。

## 模型抽象（`models/base_model.py` + `model_manager.py`）

- `BaseModel(ABC)` 持有 `feature_names`，强制子类实现 `fit / predict_proba / predict / save / _raw_feature_importance`。
- `feature_importance()` 在基类统一对齐：
  - 子类返回 `dict{name: imp}`（XGB/sklearn）→ 按 `feature_names` 顺序取值，缺失填 `0.0`。
  - 子类返回 array（LightGBM）→ 要求长度等于 `feature_names`，否则报错。
  - `model` 未训练时返回全零数组。
- `ModelManager.create` 按 `model.type`（默认 `lightgbm`）分派到 4 个后端；未知类型抛 `ValueError`。

| 后端 | type | 早停 | 概率 | 保存 | 重要性来源 |
|---|---|---|---|---|---|
| LightGBM | `lightgbm` | `early_stopping_rounds` | `predict` | `save_model`（.txt） | `feature_importance(gain)` array |
| XGBoost | `xgboost` | `early_stopping_rounds` | `predict(DMatrix)` | `joblib.dump` | `get_score` dict |
| 随机森林 | `sklearn_rf` | 不支持（忽略 val） | `predict_proba[:,1]` | `joblib.dump` | `feature_importances_` dict |
| 逻辑回归 | `sklearn_lr` | 不支持（忽略 val） | `predict_proba[:,1]` | `joblib.dump` | `|coef_|` dict |

- 阈值固定 0.5（`predict = proba > 0.5`）；sklearn 后端设了 `valid_size > 0` 会告警但不报错。
- 关键回归修复：`feature_cols: []` 时必须把解析后的 `feature_names` 显式传给模型，否则 `lgb.Dataset(feature_name=[])` 触发 “Length of feature_name(0) and num_feature(N) don't match”。

## 特征列解析（`feature_cols.py`，重点：排除泄露列）

- 正式 `train-cv-*` 配置使用注册表中的 `feature_arm: evidence_all`，即
  MS1 observed + MS2 observed + MS2 predicted；context 与 eligibility flags
  都不进入模型。`evidence_common` 在选特征前用 eligibility flags 过滤为统一
  可评估队列。
- 正式 CV 统一删除 `spec_pattern_spearman_b` 与 `spec_pattern_SA_b`；当前
  schema 最终使用 152 个模型特征。
- `evidence_core` 是可选的 35 特征紧凑臂，避免绝对丰度与 evidence-opportunity
  计数；`evidence_all` 仍是默认完整基线。正式配置开启
  `require_complete_arm`，缺少任一预期列会失败，不再取交集后静默缩小。
- 只有没有 `feature_arm` 的历史配置才使用自动检测：`explicit` 非空则直接
  使用，否则取所有输入表头的交集并剔除 META/EXCLUDED。
- 剔除三类：
  - `META_COLUMNS`：标识/标签列（`sequence/charge/raw_title*/protein_names/label/label_type/precursor_mz/sequence_len`），与 `tools/eval_baseline.py` 一致。
  - `EXCLUDED_EXTRA`：非物理过拟合或跨数据集泄露列——`modification_count`（修饰→负例的伪规则）、`window_width`（=数据集 ID 代理）、`fragment_xic_empty_count`（常量 0）、`fragment_same_mass_count` / `fragment_heavy_absent_count`（被 DIA 窗宽决定，cross_test 下成 dataset 代理）。
  - `target_col`。
- 结果为空抛 `ValueError`（P2-7）。

## 配置 YAML（`config/*.yaml`）

四段结构 `data / model / training / output`。命名约定：

| 前缀 | 含义 | held-out 方式 |
|---|---|---|
| `in_<ds>_<fdr>` | 单数据集内训练 | `test_files: []` + `test_size: 0.2`（同集切分）|
| `cross_test_<held>_<fdr>` | 留一数据集外推 | 另两集训练，`held` 作 `test_files`，`test_size: 0.0` |
| `combined_<fdr>` | 三集合并 | `test_size: 0.2` |
| `exp1/exp2` | 早期示例 | — |

`fdr` ∈ `clean / neg05 / neg10 / neg15 / neg20`；`ds` ∈
`2da / 5da / normal`。`cv_*.yaml` 由 `gen_cv_configs.py` 生成并固定为上述
evidence-only 设置，不能把 `feature_cols: []` 误解为自动使用全部列。

`make train-cv-all FEATURE_ROOT=<snapshot> CV_OUTPUT_ROOT=<output>` 会根据指定
快照生成运行时配置，不复制或修改外部特征文件。默认仍为仓库内 `runs/` 与
`runs/spec_trainer/`。六个 eligibility 字段会记录在 cohort audit 中，但不会
成为模型特征。

正式 LightGBM 配置使用 2000 轮上限、AUC-first early stopping、有效的
`bagging_fraction=0.8 / bagging_freq=1` 和固定随机种子；类别权重保持关闭。
cross-test 将测试标签导出的 working point 标记为 retrospective oracle，锁定
判定则使用各成员 outer-OOF 阈值后的多数投票。结果 bundle 还包含 OOF/test
逐行分数、每折 best iteration、按类别/来源的缺失率、sequence 重叠和环境
provenance；默认拒绝覆盖已完成 bundle。

## 固定测试集的嵌套负样本池实验

`make train-fixed-test-negpool-2da`（或 `...-all`）以 neg20 特征表为唯一主表，
将错误样本划分为 `T5=E5`、`T5_10=E10-E5`、`T10_20=E20-E10`。程序先按
sequence 冻结一次完整 E20 测试集，再训练：M5 只含 T5，M10 含 T5+T5_10，
M20 含全部三层；正确训练行、特征、cohort、外层 fold 及折内 early-stopping
分组在三个模型间完全相同。

所有模型都在同一测试行上报告 ROC-AUC、error PR-AUC 和锁定的
FNR@FPR5/Recall@FPR10，并额外在三个错误层分别评估。差值使用固定测试集上
按 sequence 的配对 cluster bootstrap，不能把嵌套的 5%/10%/20% 池当作三次
独立重复。测试标签不参与阈值选择；阈值仍来自每个成员自己的 outer OOF，
外部判定仍为成员多数投票。

## rescore 多阈值评估（`rescore.py`）

- 仅针对 LightGBM `.txt` 模型；`discover_models` 扫 `--models-dir/*.txt`，可用 `--models` 过滤。
- `infer_data_source(basename, template)`：按模型名前缀映射回 `features.csv` 与模式——`in_*`（3 段）→ `in_sample`（同集 `train_test_split` 取 20% 测试），`cross_test_*`（4 段）→ `cross_test`（整表为测试）。
- `score_model` 复用 `resolve_feature_cols` 保证特征一致，输出 `(y_true, y_proba)`。
- `compute_metrics(y_true, y_proba, threshold)`：将传入阈值解释为
  `error_score` 阈值，并按错误鉴定为阳性计算 TP/FP/FN/TN、FPR、FNR、
  error/correct recall、error precision、ROC-AUC 和 error PR-AUC；写 CSV 并用
  `rich.Table` 分组打印。
- 阈值经 `_threshold_arg` 校验，须落在开区间 (0,1)。

## 边界与设计取舍

- `resolve_holdout` 拒绝隐式 in-sample 评估：既无独立 `test_files` 又 `test_size<=0` 时直接报错（I-ST2）。
- 特征解析、held-out 逻辑抽到独立模块，便于在不引入 lightgbm 的前提下单测（I-ST1/I-ST2）。
- `classification_report` 用 `zero_division=0`：SILAC 极不均衡 + `is_unbalance` + 阈值 0.5 下负类可能无预测样本，避免告警。
- ROC 图按错误鉴定为阳性绘制并标注约登点（`TPR-FPR` 最大），但 JSON 报告
  不落该阈值（代码中被注释）。
- `train.py` / `train2.py` 是早期单后端实现，要求显式 `feature_cols` 且无泄露列剔除，仅作历史参考。
