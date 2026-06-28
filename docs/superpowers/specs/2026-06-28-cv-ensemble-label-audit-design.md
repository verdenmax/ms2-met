# 5 折 CV + 折间 ensemble + 标签噪声审计 设计

**日期**：2026-06-28
**背景**：SILAC MS2 验证是二分类，但 `runs/baseline_*_clean` 极度不平衡（2da 实测正例 104445 / 负例 870 = 99.2% 正例）。真正难分的是约 **91 个（占负例 10.5%）"有碎片信号但是假"的干扰型 hard 负例**（在 `all_p75`/`y_mean`/`all_cosine_mean` 等最强碎片特征上与正例重叠）。现行 `tools/spec_trainer/src/main.py` 用**单次 0.2 stratified holdout**（`holdout.py:resolve_holdout`），在这种少负例场景下有两个硬伤：

1. **浪费负例**：0.2 holdout 把 ~174 个负例丢进测试、只剩 ~696 进训练；其中 hard 负例测试侧仅 ~18 个 → FNR@FPR 估计极不稳。
2. **评估方差大**：单一划分的指标受随机种子摆布，无法判断改动（如 shape-defect 特征）是真提升还是噪声。

**目标**：用 **5 折交叉验证**让每个负例都既参与训练（4 折）又参与测试（1 折），并顺带产出**折间 ensemble** 与**标签噪声审计名单**。三者共用同一个产物——**out-of-fold（OOF）预测**——因此是一条流水线，而非三套独立逻辑。

**关键代码（复用，不重写）**：`tools/spec_trainer/src/models/model_manager.py:ModelManager.create`、`models/lgb_model.py:LGBModel.{fit,predict_proba,save}`、`feature_cols.py:resolve_feature_cols`。新增独立入口 `tools/spec_trainer/src/cv_train.py`，**不改 `main.py`**（现行单 holdout 流程保持不变）。

---

## 1. 核心思想：OOF 是三件事的共同产物

```
5 折 CV 跑一遍 ──► 每个样本一个 OOF 预测（由"没训过它"的那折模型给出）
                              │
        ┌──────────────────────┼──────────────────────┐
        ▼                      ▼                      ▼
  ① 诚实评估             ② 折间 ensemble         ③ 标签噪声审计
  全量 OOF 上算          保存 5 个折模型,         负例按 OOF"像正例"
  AUC/FNR@FPR            新数据 = 5 模型均值       程度降序 → 人工复核名单
  (870 负例全用上)
```

OOF 的定义：样本 i 的 OOF 预测 = 把 i 划入测试折的那一次、由其余 4 折训练出的模型对 i 的 `predict_proba`。**每个样本恰好被预测一次，且预测它的模型从未见过它** → 无泄漏、可作诚实的 in-sample 评估。

### 1.1 与现有 `eval_baseline.cv_evaluate` 的关系（复用而非重造）

`tools/eval_baseline.py` 已有 `cv_evaluate`（`StratifiedKFold` 5 折 + OOF 拼接 + `compute_working_points`）。**但它面向"快速基线筛查"**：用 `sklearn.HistGradientBoostingClassifier(class_weight="balanced")`、不分组、不存模型、无审计。本期 `cv_train.py` 与之**目的不同、互补**：

| | `eval_baseline.cv_evaluate`（已有） | `cv_train.py`（本期） |
|---|---|---|
| 模型 | sklearn HGB（基线代用） | **生产 LightGBM**（`LGBModel`，与上线一致） |
| 分组 | 普通 `StratifiedKFold` | `StratifiedGroupKFold(by sequence)` 防泄漏 |
| 模型持久化 | 不存 | 存 5 个 → **折间 ensemble** |
| 标签审计 | 无 | **有**（OOF 排序） |
| FNR@FPR 约定 | `compute_working_points` | **复用同一函数**（见 §4） |

即：本期把 `cv_evaluate` 的 CV+OOF 哲学搬到**生产 LightGBM** 上，并补齐 ensemble 持久化与标签审计。

---

## 2. 架构与数据流

```
cv_train.py
  ├─ 读 yaml（沿用现 schema + §5 新增键）
  ├─ pd.read_csv(train_files) → df            # 保留 sequence/charge 供分组+审计
  ├─ resolve_feature_cols(...) → feature_cols # 复用；新列(含 shape-defect)自动纳入
  ├─ X = df[feature_cols]; y = df[target_col]; groups = df[group_col]
  ├─ splitter = StratifiedGroupKFold(n_splits=cv_folds, shuffle=True,
  │                                  random_state=cv_seed)
  ├─ oof_proba = full(len(df), nan)
  ├─ for k, (tr_idx, te_idx) in enumerate(splitter.split(X, y, groups)):
  │     # 折内再切早停验证集（按 sequence 分组，避免污染）
  │     tr2_idx, val_idx = GroupShuffleSplit(1, test_size=valid_size,
  │                          random_state=cv_seed).split(X[tr_idx], groups=groups[tr_idx])
  │     model_k = ModelManager.create(cfg, feature_names=feature_cols)
  │     model_k.fit(X[tr2_idx], y[tr2_idx], X[val_idx], y[val_idx])
  │     oof_proba[te_idx] = model_k.predict_proba(X[te_idx])
  │     model_k.save(f"{model_prefix}.fold{k}.txt")
  │     fold_metrics[k] = eval(y[te_idx], oof_proba[te_idx])      # 每折单独指标
  ├─ assert not isnan(oof_proba).any()         # 每样本恰好预测一次
  ├─ ① 评估：metrics = eval(y, oof_proba) + fold mean±std → cv_result_path(.json)
  └─ ③ 审计：suspects = audit(df, oof_proba) → suspects_path(.csv)
```

设计为三个**可独立测试**的纯函数 + 一个编排 `main()`：

- `make_cv_splits(X, y, groups, n_folds, seed)` → 折索引列表（无模型依赖，可单测）。
- `assemble_oof(df, cfg, feature_cols, splits)` → `(oof_proba, fold_models, fold_metrics)`（依赖 LGB）。
- `evaluate_oof(y, oof_proba)` → 指标 dict（无模型依赖，可单测）。
- `audit_labels(df, oof_proba, threshold, top_n)` → suspects DataFrame（无模型依赖，可单测）。
- `predict_ensemble(model_paths, X)` → `mean_k predict_proba`（折间 ensemble 的消费侧，可单测）。

---

## 3. 三个组件细节

### 3.1 ① 5 折 CV（每个负例都进过训练和测试）

- **分组**：`StratifiedGroupKFold(by=group_col, 默认 sequence)`。同一条肽段的所有 PSM 行不会一半在训、一半在测 → **防同肽段泄漏**（比单 holdout 更诚实）。`stratify=label` 保证每折类比例稳定。
- **折内早停**：每折用其余 4 折训练时，再用 `GroupShuffleSplit(test_size=valid_size, by=sequence)` 切出早停验证集；**早停验证集不进 OOF 折**，避免污染。
- **结果**：870 负例全部既参与训练（4 折）又参与测试（1 折）；全量 OOF 上的指标比"174 个测试负例"稳得多。

### 3.2 ② 折间 ensemble（bagging）

- **两种预测模式**（必须区分清楚）：
  - **OOF 预测** —— 只用于本数据集的诚实 in-sample 评估（每折模型只预测自己的测试折）。
  - **Ensemble 预测** —— 用于**新/外部数据**：`mean_k model_k.predict_proba(X_new)`。本期产出 `predict_ensemble` helper（供上线打分/后续 cross_test 复用）；**把它接进 cross_test make 管线属后续**（见 §10）。
- **落地**：保存 5 个 `{model_prefix}.fold{k}.txt`；`predict_ensemble(model_paths, X)` 加载 5 个 booster 取均值。`rescore.py` 后续只需把"加载单模型"换成"加载 5 模型取均值"。
- **收益**：信号来自 ~91 个 hard 负例时，单模型方差大，5 折平均显著更稳。

### 3.3 ③ 标签噪声审计（从 OOF 免费掉出来，零新依赖）

- **原理**：对每个**负例**看其 OOF 概率。`oof_proba` 很高 = "没训过它的模型都觉得它像正例"。两种可能：(a) 真·hard 负例（干扰型）；(b) **标错了**（人源同源/共享肽、FDR 误判、entrapment 误纳）。
- **产出** `suspects.csv`：负例按 `oof_proba` 降序，列含 `sequence, charge, label_type, oof_proba` + 诊断特征（`all_p75, precursor_pearson, all_cosine_mean`，若存在则加 `all_heavy_shape_irregularity_max`）。筛选规则：`oof_proba >= suspect_threshold`（默认 0.9），至多 `suspect_top_n`（默认 200）行。
- **⚠️ 定位 = triage（排序出嫌疑名单供人看），不是自动改标签**。高 OOF 分既可能是标错、也可能是真难例 —— 由人（用户）判断。
- **对称（可选，次要）**：OOF 极低的**正例**也可同法列出（疑似漏报/弱信号真例），默认只输出负例侧。

---

## 4. 评估指标（复用 `eval_baseline.compute_working_points` 既有约定）

⚠️ **复用 `tools/eval_baseline.py:119 compute_working_points`，不另造 roc_curve 版**，保证与既有评估口径一致。其约定：按**负例分位数**定阈值 `thr = quantile(neg_scores, 1-fpr)`（FPR∈{5,10,20}%），报该阈值下的 `pos_recall(=TPR=1-FNR)` 与 `neg_recall`。

在**全量 OOF**（`y, oof_proba`）上：

- `working_points = compute_working_points(y, oof_proba)` → 含 `neg_recall_95/90/80`。
- **FNR@FPR≤5%** = `1 - working_points["neg_recall_95"]["pos_recall"]`（你关注的边界指标）。
- `auc = roc_auc_score(y, oof_proba)`。
- `per_fold`：每折单独 `auc` + working_points → 报 `mean ± std`（暴露方差，判断改动是否真提升）。

**实现复用**：`compute_working_points` 现位于 `tools/eval_baseline.py`，`cv_train.py` 直接 import 复用（该函数仅依赖 numpy，导入轻量）；实施计划中若两处共用，可抽到共享 `metrics.py`。输出 JSON 对齐 `eval_baseline.cv_evaluate` 的 summary 结构（`auc_mean/std + fold_metrics + working_points`）+ 增补 `cv_folds / suspects_path` 等本期字段。

---

## 5. 配置（最小新增，向后兼容）

`cv_train.py` 沿用现 yaml，新增/复用以下键（缺省即退化为合理默认；`main.py` 不读这些键，故其行为不变）：

```yaml
data:
  group_col: sequence        # 分组列；缺失 → 退化为 StratifiedKFold(不分组) + 警告
training:
  cv_folds: 5                # 折数
  cv_seed: 42                # 复现实验
  valid_size: 0.15           # 复用现有键：折内早停验证集比例
audit:
  suspect_threshold: 0.9     # 负例 OOF 概率 ≥ 此值进嫌疑名单
  suspect_top_n: 200         # 名单上限
```

路径从现有 `output.model_path` / `output.result_path` **派生**（不新增路径键）：
`model_prefix` = `model_path` 去掉 `.txt`（→ `.fold{k}.txt`）；`cv_result_path` = `result_path` 改后缀 `.cv.json`；`suspects_path` = 同目录 `<name>.suspects.csv`。

---

## 6. 产出物

```
runs/spec_trainer/models/in_2da_clean.fold0..4.txt   # 5 模型 = ensemble
runs/spec_trainer/results/in_2da_clean.cv.json       # OOF 指标 + per-fold mean±std
runs/spec_trainer/results/in_2da_clean.suspects.csv  # 负例嫌疑名单(按像正例程度降序)
```

可选新增 Makefile 目标 `train-cv-2da`（调 `cv_train.py --config in_2da_clean.yaml`），起始只接 `in_2da_clean`。

---

## 7. 边界 / 错误处理

- `group_col` 缺失 → 退化 `StratifiedKFold`（不分组）+ 记一条 warning（仍能跑，但提示可能有同肽段泄漏）。
- 某折测试侧零负例（负例集中在少数 group 时可能）→ `StratifiedGroupKFold` 已尽量均衡；该折 `auc/fnr` 记 `NaN` 并在汇总时跳过，不崩。
- OOF 残留 `NaN`（理论不应发生）→ `assert` 失败并报错（每样本必须恰好预测一次）。
- 类样本数不足以分 5 折 → 显式报错提示降低 `cv_folds`。
- 复现：`cv_seed` 固定；`StratifiedGroupKFold(shuffle=True, random_state=cv_seed)`。

---

## 8. 测试

**纯函数（不需 lightgbm，合成数据断言）**：

- `make_cv_splits`：5 折测试索引两两不相交且并集 = 全体；**同一 group 不同时出现在某折的 train 与 test**；类比例近似均衡。
- `evaluate_oof`：在构造的 `(y, oof_proba)` 上，`auc` 与 `fnr_at_fpr5` 数值正确（与手算/已知值一致）。
- `audit_labels`：负例按 `oof_proba` 降序；`>= threshold` 过滤与 `top_n` 截断正确；只含负例。
- `predict_ensemble`：对 mock booster 列表，输出 = 各 `predict_proba` 的均值。

**集成（需 lightgbm）**：

- 在小型合成 features.csv 上跑 `cv_train.py`：产出 3 个文件、`oof_proba` 无 `NaN`、5 个折模型文件存在、`cv.json` 含 `per_fold` 与 `fnr_at_fpr5`。
- 回归：`main.py` 单 holdout 流程不受影响（现有 spec_trainer 测试全绿）。

---

## 9. 兼容性

- **不动 `main.py` / `holdout.py`**：单 holdout 流程与 `make train-clean-all` 现状不变。
- `feature_cols.py` 不改：`feature_cols: []` 自动纳入新列（与 shape-defect 同机制），CV 与单 holdout 看到的特征集一致。
- 新文件均在 `tools/spec_trainer/src/`，与现有结构一致。

---

## 10. 不在范围

- **cross_test 的 CV 重构**：cross_test 已有外部测试集，CV 只会重构其训练侧（在数据集 A 上做 CV 训 5 模型、ensemble 预测数据集 B）。本期先做 `in_*`，cross_test 作后续扩展。
- **confident-learning / `cleanlab` 置信学习**：本期标签审计用免依赖的 OOF 排序；cleanlab 原则性阈值作后续可选增强。
- **自动改标签**：审计永远只产出嫌疑名单供人复核，不自动 relabel。
- **超参搜索 / 单调约束 / 自定义 FNR@FPR eval**：是后续独立优化项（见会话讨论），不在本期。

---

## 11. 环境前提

运行需 `lightgbm` + `scikit-learn`（`StratifiedGroupKFold`/`GroupShuffleSplit`）。当前 `jianyan` 环境**未安装**，需先：

```
conda install -n jianyan -c conda-forge lightgbm scikit-learn
```
（纯函数的单测可在无 lightgbm 下运行；集成测试与实跑需上述依赖。）
