# spec_trainer — 细节

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

- `explicit` 非空 → 直接用。否则取所有 `sample_csv_paths` 列名的**交集**（schema drift 会告警），顺序按第一个文件。
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

`fdr` ∈ `clean / neg05 / neg10 / neg15 / neg20`（`combined_*` 仅有 `clean / neg05 / neg10` 三档）；`ds` ∈ `2da / 5da / normal`。`feature_cols: []` 一律自动检测。当前 `config/` 共 33 个实验配置（`in_*` 15 + `cross_test_*` 15 + `combined_*` 3）外加 `exp1`/`exp2` 两个早期示例。

## rescore 多阈值评估（`rescore.py`）

- 仅针对 LightGBM `.txt` 模型；`discover_models` 扫 `--models-dir/*.txt`，可用 `--models` 过滤。
- `infer_data_source(basename, template)`：按模型名前缀映射回 `features.csv` 与模式——`in_*`（3 段）→ `in_sample`（同集 `train_test_split` 取 20% 测试），`cross_test_*`（4 段）→ `cross_test`（整表为测试）。
- `score_model` 复用 `resolve_feature_cols` 保证特征一致，输出 `(y_true, y_proba)`。
- `compute_metrics(y_true, y_proba, threshold)`：每阈值算 TN/FP/FN/TP、正/负类 recall+precision、`f1_neg`、AUC；写 CSV 并用 `rich.Table` 分组打印。
- 阈值经 `_threshold_arg` 校验，须落在开区间 (0,1)。

## 边界与设计取舍

- `resolve_holdout` 拒绝隐式 in-sample 评估：既无独立 `test_files` 又 `test_size<=0` 时直接报错（I-ST2）。
- 特征解析、held-out 逻辑抽到独立模块，便于在不引入 lightgbm 的前提下单测（I-ST1/I-ST2）。
- `classification_report` 用 `zero_division=0`：SILAC 极不均衡 + `is_unbalance` + 阈值 0.5 下负类可能无预测样本，避免告警。
- ROC 图标注约登点（`TPR-FPR` 最大），但 JSON 报告不落该阈值（代码中被注释）。
- `train.py` / `train2.py` 是早期单后端实现，要求显式 `feature_cols` 且无泄露列剔除，仅作历史参考。
