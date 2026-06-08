# spec_trainer — API 参考

逐文件列主要 class / 函数签名。路径以仓库根为基准。

## tools/spec_trainer/src/main.py

当前训练入口。

- `load_data(file_paths, feature_cols, target_col) -> (X, y)`：逐文件 `read_csv` 后 `concat`；文件不存在抛 `FileNotFoundError`。返回 `X=df[feature_cols]`、`y=df[target_col]`。
- `save_feature_importance(model, feature_names, output_path)`：取 `model.feature_importance('gain')`，按重要性降序画水平条形图存 PNG。
- `save_roc_figure(y_true, y_proba, output_path)`：画 ROC 曲线并标注约登最优点（`argmax(tpr-fpr)`）。
- `evaluate_and_report(y_true, y_pred, y_proba, feature_names=None, model=None, report_path=None, fig_path=None, roc_path=None) -> dict`：算 accuracy / auc / confusion_matrix / classification_report（`zero_division=0`），按需写 JSON、画重要性图与 ROC 图，返回 metrics dict。
- `main()`：解析 `--config`（必填）、`--name`（必填）、`--logpath`（默认 `./spec.log`）；配置 rich+文件日志；执行完整训练评估流程。

## tools/spec_trainer/src/feature_cols.py

- `META_COLUMNS: set[str]`：非特征的标识/标签列。
- `EXCLUDED_EXTRA: set[str]`：额外排除的过拟合 / 跨数据集泄露列。
- `resolve_feature_cols(explicit, sample_csv_paths, target_col) -> list[str]`：`explicit` 非空则直接返回；否则取各 CSV 列名交集（仅读 header，`nrows=0`），剔除 `META_COLUMNS ∪ EXCLUDED_EXTRA ∪ {target_col}`，顺序随第一个文件。drift 时告警；结果为空抛 `ValueError`。`sample_csv_paths` 可为单字符串（向后兼容）。

## tools/spec_trainer/src/holdout.py

- `resolve_holdout(X_train, y_train, train_files, test_files, test_size, feature_cols, target_col, loader) -> (X_train, X_test, y_train, y_test)`：优先级——① `test_files` 非空且异于 `train_files` → 用 `loader(test_files, feature_cols, target_col)` 加载；② `test_size>0` → `train_test_split(stratify=y, random_state=42)`；③ 否则抛 `ValueError`（拒绝 in-sample 评估）。

## tools/spec_trainer/src/models/base_model.py

- `class BaseModel(ABC)`
  - `__init__(self, feature_names=None)`：记录特征名以对齐重要性。
  - 抽象方法：`fit(X_train, y_train, X_val=None, y_val=None)`、`predict_proba(X) -> np.ndarray`、`predict(X) -> np.ndarray`、`save(path)`、`_raw_feature_importance(importance_type='gain')`。
  - `feature_importance(importance_type='gain') -> np.ndarray`：把子类原始重要性（dict 或 array）对齐到 `feature_names` 顺序，长度一致；`feature_names` 为 None 时报错。

## tools/spec_trainer/src/models/model_manager.py

- `class ModelManager`
  - `@staticmethod create(config: dict, feature_names: list=None) -> BaseModel`：按 `config['model']['type']`（默认 `lightgbm`，小写）返回 `LGBModel` / `XGBModel` / `SklearnRFModel` / `SklearnLRModel`；未知类型抛 `ValueError`。`feature_names` 缺省回退到 `config['data']['feature_cols']`。

## tools/spec_trainer/src/models/lgb_model.py

- `class LGBModel(BaseModel)`
  - `__init__(self, model_params, training_params, feature_names)`。
  - `fit(...)`：构 `lgb.Dataset(feature_name=feature_names)`，可选验证集，回调 `early_stopping(training_params['early_stopping_rounds'])` + `log_evaluation(100)`，`num_boost_round=training_params['num_boost_round']`。
  - `predict_proba(X)` → `model.predict(X)`；`predict(X)` → `(proba>0.5).astype(int)`。
  - `save(path)`：`save_model`（文本 .txt）。
  - `_raw_feature_importance(...)`：返回 gain array（顺序同 `feature_name`）。

## tools/spec_trainer/src/models/xgb_model.py

- `class XGBModel(BaseModel)`
  - `fit(...)`：`xgb.DMatrix(feature_names=...)`，`evals` 含 train/valid，`early_stopping_rounds`、`verbose_eval=100`。
  - `predict_proba(X)`：`model.predict(DMatrix(X))`；`predict(X)`：`>0.5` 取整。
  - `save(path)`：`joblib.dump`。
  - `_raw_feature_importance(...)`：`model.get_score(...)` 返回 dict。

## tools/spec_trainer/src/models/sklearn_rf_model.py

- `class SklearnRFModel(BaseModel)`：随机森林。`valid_size>0` 时告警（不支持早停）。`fit` 用 `RandomForestClassifier(**model_params)`；`predict_proba` 取正类列；`save` 用 `joblib.dump`；重要性取 `feature_importances_`（dict）。

## tools/spec_trainer/src/models/sklearn_lr_model.py

- `class SklearnLRModel(BaseModel)`：逻辑回归。同样对 `valid_size>0` 告警。`fit` 用 `LogisticRegression(**model_params)`；`predict_proba` 取正类列；重要性取 `|coef_[0]|`（dict）。

## tools/spec_trainer/rescore.py

对已训练 LightGBM 模型多阈值重评估。

- `compute_metrics(y_true, y_proba, threshold) -> dict`：算 n_pos/n_neg、TN/FP/FN/TP、正负类 recall+precision、`f1_neg`、`auc`。
- `infer_data_source(model_basename, features_root_template) -> (Path, mode)`：`in_*`（3 段）→ `in_sample`；`cross_test_*`（4 段）→ `cross_test`；其它抛 `ValueError`。
- `discover_models(models_dir, filter_names) -> list[Path]`：列 `*.txt`，按 stem 过滤，缺失告警。
- `score_model(model_path, csv_path, mode, target_col='label') -> (y_true, y_proba)`：载 `lgb.Booster`，用 `resolve_feature_cols` 取特征；`in_sample` 切 20% 测试，`cross_test` 用整表。
- `_threshold_arg(s) -> float`：argparse 校验，须 (0,1)。
- `_build_parser()`：参数 `--thresholds`（必填，可多个）、`--models`、`--output`、`--models-dir`、`--features-root`。
- `_print_console_table(rows)` / `_write_csv(rows, output_path)`：rich 表格输出 / 写汇总 CSV。
- `main(argv=None) -> int`：装配上述流程，返回退出码。

## tools/spec_trainer/src/train.py、train2.py

早期单后端（仅 LightGBM）脚本，已被 `main.py` + `ModelManager` 取代。`train.py:main(config_path)` 与 `train2.py:main()` 直接读 `cfg['data']['feature_cols']`（需显式列出），无泄露列剔除与多后端抽象，仅作历史参考。
