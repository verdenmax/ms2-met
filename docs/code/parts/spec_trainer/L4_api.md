# spec_trainer — API 参考

逐文件列主要 class / 函数签名。路径以仓库根为基准。

## tools/spec_trainer/src/main.py

当前训练入口。

- `load_data(file_paths, feature_cols, target_col) -> (X, y)`：逐文件 `read_csv` 后 `concat`；文件不存在抛 `FileNotFoundError`。返回 `X=df[feature_cols]`、`y=df[target_col]`。
- `save_feature_importance(model, feature_names, output_path)`：取 `model.feature_importance('gain')`，按重要性降序画水平条形图存 PNG。
- `save_roc_figure(y_true, y_proba, output_path)`：先用
  `error_truth=1-y_true`、`error_score=1-y_proba` 转换，再画错误鉴定为阳性的
  ROC 曲线并标注约登最优点（`argmax(tpr-fpr)`）。
- `evaluate_and_report(y_true, y_pred, y_proba, feature_names=None, model=None, report_path=None, fig_path=None, roc_path=None) -> dict`：按错误鉴定为阳性计算 accuracy / `roc_auc` / confusion_matrix / classification_report（`zero_division=0`），写入 `metric_semantics` 与 `positive_class`，并按需写 JSON、画重要性图与 ROC 图。
- `main()`：解析 `--config`（必填）、`--name`（必填）、`--logpath`（默认 `./spec.log`）；配置 rich+文件日志；执行完整训练评估流程。

## tools/spec_trainer/src/feature_cols.py

- `META_COLUMNS: set[str]`：非特征的标识/标签列。
- `EXCLUDED_EXTRA: set[str]`：额外排除的过拟合 / 跨数据集泄露列。
- `resolve_feature_cols(explicit, sample_csv_paths, target_col) -> list[str]`：`explicit` 非空则直接返回；否则取各 CSV 列名交集（仅读 header，`nrows=0`），剔除 `META_COLUMNS ∪ EXCLUDED_EXTRA ∪ {target_col}`，顺序随第一个文件。drift 时告警；结果为空抛 `ValueError`。`sample_csv_paths` 可为单字符串（向后兼容）。

## tools/spec_trainer/src/holdout.py

- `resolve_holdout(X_train, y_train, train_files, test_files, test_size, feature_cols, target_col, loader) -> (X_train, X_test, y_train, y_test)`：优先级——① `test_files` 非空且异于 `train_files` → 用 `loader(test_files, feature_cols, target_col)` 加载；② `test_size>0` → `train_test_split(stratify=y, random_state=42)`；③ 否则抛 `ValueError`（拒绝 in-sample 评估）。

## tools/spec_trainer/src/cv_train.py

- `read_dataframe(paths) -> DataFrame`：合并特征表，并增加只用于审计的
  `__source_file/__source_row`，不作为模型输入。
- `assemble_oof(..., return_fold_ids=False, predefined_fold_ids=None,
  predefined_inner_valid=None)`：按 sequence 分组完成外层 CV 和
  分组早停；保存每折模型、OOF 分数、best iteration，以及每个成员模型在自身
  outer-OOF 上得到的 FPR5/FPR10 校准阈值。两个 predefined 参数用于严格配对
  实验，传入时会校验折号完整性和 group 不跨折。
- `predict_ensemble(model_paths, X)`：返回成员 trust score 均值；DataFrame
  输入会严格验证列名及顺序。
- `evaluate_cross_test(..., fold_metrics=None, return_details=False)`：连续
  ensemble 分数用于 ROC-AUC/error PR-AUC；外部标签重选工作点明确标记为
  retrospective oracle。传入 fold calibration 时，锁定判定采用逐成员阈值和
  多数投票，不把单模型 OOF 阈值应用到平均分数。
- `main(argv=None)`：默认拒绝覆盖已有 bundle；`--overwrite` 显式允许重跑。
  原子写入 JSON、suspects、OOF/test 逐样本分数；JSON 还记录缺失模式、来源
  分层指标、sequence overlap、每折迭代数、配置/Git/依赖/input fingerprint。

## tools/spec_trainer/src/fixed_negpool.py

- `prepare_fixed_negpool(paths, cfg, ...) -> PreparedFixedNegpool`：验证三档样本
  身份、嵌套、正确样本一致性及共享特征一致性；从 neg20 主表构造共同 cohort、
  固定 sequence 测试集及可复用 outer/inner fold map。
- `run_fixed_negpool(config_path, feature_root, dataset, output_root, ...)`：训练
  M5/M10/M20，在相同 E20 测试行及三个错误层上评估，并输出 sequence-cluster
  paired bootstrap、逐样本分数、manifest、模型和 provenance。

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
  - `fit(...)`：构 `lgb.Dataset(feature_name=feature_names)`，可选验证集，回调 `early_stopping(training_params['early_stopping_rounds'], first_metric_only=...)` + `log_evaluation(100)`，`num_boost_round=training_params['num_boost_round']`；正式配置仅由第一项 AUC 控制早停。
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

- `compute_metrics(y_true, y_proba, threshold) -> dict`：`threshold` 是
  `error_score=1-y_proba` 的阈值；按错误鉴定为阳性返回
  `n_actual_error/n_actual_correct`、TP/FP/FN/TN、FPR/FNR、
  `error_recall/correct_recall/error_precision`、`roc_auc/error_pr_auc`。
- `infer_data_source(model_basename, features_root_template) -> (Path, mode)`：`in_*`（3 段）→ `in_sample`；`cross_test_*`（4 段）→ `cross_test`；其它抛 `ValueError`。
- `discover_models(models_dir, filter_names) -> list[Path]`：列 `*.txt`，按 stem 过滤，缺失告警。
- `score_model(model_path, csv_path, mode, target_col='label') -> (y_true, y_proba)`：载 `lgb.Booster`，用 `resolve_feature_cols` 取特征；`in_sample` 切 20% 测试，`cross_test` 用整表。
- `_threshold_arg(s) -> float`：argparse 校验，须 (0,1)。
- `_build_parser()`：参数 `--thresholds`（必填，可多个）、`--models`、`--output`、`--models-dir`、`--features-root`。
- `_print_console_table(rows)` / `_write_csv(rows, output_path)`：rich 表格输出 / 写汇总 CSV。
- `main(argv=None) -> int`：装配上述流程，返回退出码。

## tools/spec_trainer/src/train.py、train2.py

早期单后端（仅 LightGBM）脚本，已被 `main.py` + `ModelManager` 取代。`train.py:main(config_path)` 与 `train2.py:main()` 直接读 `cfg['data']['feature_cols']`（需显式列出），无泄露列剔除与多后端抽象，仅作历史参考。
