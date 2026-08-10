# spec_trainer — 职责与接口

## 一句话职责

训练 / 评估二分类器，对每条 PSM 的 SILAC 验证特征（`features.csv`）打可信度
分数并区分正确鉴定与错误鉴定。存储/训练保持 `label=1` 为正确鉴定；对外评价
统一以错误鉴定为阳性，并产出模型、JSON 报告与图。

## 对外接口

| 符号 | 签名 | 简述 |
|---|---|---|
| `src/main.py:main()` | CLI `--config <yaml> --name <exp> [--logpath]` | 当前训练入口：解析特征 → 划分 held-out → 建模 → 训练 → 评估 |
| `resolve_feature_cols(explicit, sample_csv_paths, target_col)` | → `list[str]` | 自动检测特征列：取多文件列名**交集**，剔除元数据/泄露列 |
| `resolve_holdout(...)` | → `(X_train, X_test, y_train, y_test)` | 决定测试集来源：独立 `test_files` 或 `train_test_split`，否则报错 |
| `ModelManager.create(config, feature_names=None)` | → `BaseModel` | 工厂：按 `model.type` 返回对应后端模型 |
| `BaseModel` | ABC | 统一接口 `fit/predict/predict_proba/save/feature_importance` |
| `LGBModel` / `XGBModel` / `SklearnRFModel` / `SklearnLRModel` | `BaseModel` 子类 | LightGBM / XGBoost / 随机森林 / 逻辑回归后端 |
| `rescore.py:main(argv)` | CLI `--thresholds … [--models] [--models-dir] …` | 对已训练 LightGBM 模型在多阈值下重评估，汇总 CSV + 控制台表 |
| `src/train.py`、`src/train2.py` | `main()` | 早期单后端（仅 LightGBM）脚本，已被 `main.py` + `ModelManager` 取代 |

## 依赖

- 依赖：`lightgbm`、`xgboost`、`scikit-learn`、`pandas`、`numpy`、`pyyaml`、`matplotlib`、`rich`、`joblib`。
- 被依赖：无（独立工具）；消费上游 `runs/baseline_*/features.csv`（由 baseline pipeline 产出）。

## 输入 / 输出

- 输入：YAML 配置（`config/*.yaml`）、特征表 `runs/baseline_{ds}_{fdr}/features.csv`（含 `label` 列）。
- 输出：
  - 模型 `runs/spec_trainer/models/<name>.txt`（LightGBM 文本；其余后端用 joblib）。
  - 报告 `runs/spec_trainer/results/<name>.json`（带语义版本的 error-positive
    ROC-AUC、error PR-AUC、FNR@FPR5、工作点 / 混淆矩阵等）。
  - 图 `runs/spec_trainer/figures/<name>_importance.png`、`<name>_roc_curve.png`。
  - rescore 汇总 `runs/spec_trainer/rescore_summary.csv`。
