# Rescore Tool — 多阈值后评估工具

**Date**: 2026-06-03
**Scope**: 训练后对比工具
**Status**: Approved

## 问题

`tools/spec_trainer/src/main.py` 训练后只用阈值 0.5 输出 confusion matrix 和精确率/召回率。在 SILAC 极度不均衡的测试集（99%+ 正例）上，is_unbalance=True 配合单棵树早停时，几乎所有样本都被预测为正类，让混淆矩阵几乎无意义。AUC 本身没问题，但用户希望快速对比多个阈值下的指标，**无需重新训练**。

## 决策摘要

| 决策 | 选择 |
|---|---|
| 实现方式 | 独立 CLI 脚本 `tools/spec_trainer/rescore.py`，不修改 main.py |
| 模型默认范围 | 扫描 `runs/spec_trainer/models/*.txt` |
| 模型选择 | `--models` 名字白名单可过滤子集（按 basename，无 `.txt`） |
| 阈值参数 | `--thresholds 0.5 0.7 0.9 0.99`（必填，可多值） |
| 数据源推断 | 模型名前缀决定：`in_<ds>_<fdr>` → 对应 features.csv 20% split (random_state=42, stratify=y)；`cross_test_<held>_<fdr>` → 整个 held-out features.csv |
| 特征列 | 调 `feature_cols.resolve_feature_cols`，与训练时严格一致（已排除 5 列泄露列） |
| 输出 | CSV 表 + `rich.table.Table` 控制台同时输出 |
| 输出路径 | `--output runs/spec_trainer/rescore_summary.csv`（默认） |

## 行为契约

### 输入
- `--thresholds T1 T2 ...`：浮点列表，0 < t < 1
- `--models name1 name2 ...`：可选，模型 basename（如 `in_2da_clean`），默认全部 18 个
- `--output PATH`：CSV 输出路径

### 输出 CSV schema
| 列 | 含义 |
|---|---|
| experiment | 模型 basename（如 `in_2da_clean`）|
| threshold | 应用的阈值 |
| n_pos | 测试集真实正例数 |
| n_neg | 测试集真实负例数 |
| tn / fp / fn / tp | 混淆矩阵 |
| pos_recall | TP / (TP + FN) |
| neg_recall | TN / (TN + FP) |
| pos_precision | TP / (TP + FP)；分母为 0 时返回 0.0 |
| neg_precision | TN / (TN + FN)；分母为 0 时返回 0.0 |
| f1_neg | 2 × neg_precision × neg_recall / (neg_precision + neg_recall) |
| auc | sklearn roc_auc_score(y_te, proba)，与 threshold 无关 |

每个 (experiment, threshold) 一行。18 个 × 4 阈值 = 72 行。

### 验证
- **Sanity check**：对 `in_2da_clean` 在阈值 0.5 的输出，应与 `runs/spec_trainer/results/in_2da_clean.json` 中的 confusion_matrix 完全一致（同一 random_state + 同一阈值）。这是测试要覆盖的核心场景。

### 错误处理
- 模型文件不存在 → 跳过 + warn 日志
- features.csv 不存在 → 跳过 + warn 日志
- 阈值 ≤ 0 或 ≥ 1 → CLI argparse 拒绝（type=float + 自定义 validator）

## 实现要点

- 同一份 X 在多个阈值下共用，不重复 `predict()`：先调一次 `model.predict(X)` 拿到 `proba`，再循环 `(proba > t).astype(int)` 拿到不同阈值的 y_pred
- 不重新 fit，不修改 model.txt
- 不写新 JSON，保留原 results/ 不变
- 用 `lightgbm.Booster(model_file=...)` 加载（与原 `model.save()` 配对）

## 测试

放在 `tests/test_rescore_tool.py`：

1. **test_rescore_in_sample_split_matches_training** — `in_2da_clean` 在 0.5 阈值 → confusion matrix 等于现有 JSON
2. **test_rescore_cross_test_uses_full_held_file** — `cross_test_2da_clean` 行数等于整个 `runs/baseline_2da_clean/features.csv`
3. **test_rescore_threshold_monotonicity** — 阈值递增时 neg_recall 单调不减、pos_recall 单调不增
4. **test_rescore_models_filter** — `--models in_2da_clean` 只输出该一行 × N 阈值
5. **test_rescore_invalid_threshold_rejected** — `--thresholds 1.5` 被 argparse 拒绝

测试可选择跳过 sanity check（如 features.csv 不存在 → skip 而非 fail），用 `pytest.mark.skipif`。

## 不做的事

- 不画图（用 figures/ 已有）
- 不算 AUPRC / MCC
- 不修改 main.py 训练流程
- 不写新 JSON
- 不重训
