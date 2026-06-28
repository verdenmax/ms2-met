# CV 扩展到完整 30 矩阵 + cross_test 模式 设计

**日期**：2026-06-28
**背景**：上一里程碑（`2026-06-28-cv-ensemble-label-audit-design.md`）为 spec_trainer 增加了 5 折分组 CV + 折间 ensemble + 标签审计，但**只接了一个实验** `cv_in_2da_clean`，且 CV 仅 in-sample（spec §10 明确把 cross_test 列为不在范围）。本设计把 CV 推广到**完整 30 实验矩阵**（镜像现有 `make train-all`），并新增 **cross_test CV 模式**。

**为什么 cross_test 现在能做**：OOF（out-of-fold）是 in-sample 评估机制——靠"同一数据集内轮换折"让每个样本被没训过它的模型预测；cross_test 的测试集模型从头没训过，对它谈不上 OOF。但 CV 的**折间 ensemble** 完全能给外部数据打分：在训练侧（如 5da+normal）CV 出 5 个折模型，对外部测试集（2da）取均值预测。`predict_ensemble` 正是为此而建。所以 cross_test CV = "训练侧 CV 出 5 模型 → ensemble 预测外部测试集 → 在外部测试集上评估/审计"。

**关键代码**：唯一的"硬"改动是 `tools/spec_trainer/src/cv_train.py:main()` 加一个 cross_test 分支 + 一个 per-fold 外部评估 helper；其余是配置生成（30 个 cv_ yaml）+ Makefile 目标。**不改 `main.py` / `holdout.py` / 既有 30 个 in_/cross_test 配置 / cv_core.py**。

---

## 1. 矩阵（30 = 镜像 train-all）

源配置已有 30 个：`{in_,cross_test_}{2da,5da,normal}_{clean,neg05,neg10,neg15,neg20}.yaml`。本设计为每个生成一个 `cv_` 变体：

| FDR | in_（in-sample CV） | cross_test_（ensemble CV） |
|---|---|---|
| clean / neg05 / neg10 / neg15 / neg20 | `cv_in_{2da,5da,normal}_{fdr}`（15） | `cv_cross_test_{2da,5da,normal}_{fdr}`（15） |

- `in_*`：CV 在该数据集自身上做 → OOF → 在 OOF 上评估/审计（与现状同模式）。
- `cross_test_*`：CV 在 `train_files`（另两个数据集）上做 → 5 折模型 → ensemble 预测 `test_files`（held-out 数据集）→ 在外部测试集上评估/审计。

源 cross_test 配置结构（参考 `cross_test_2da_clean.yaml`）：`train_files=[5da,normal]`、`test_files=[2da]`、`test_size=0.0`。所有 in_/cross_test 配置的 `model.params` 统一（num_leaves 15 / lr 0.02），故 cv 变体只需加 CV 键 + 改输出路径。

---

## 2. `cv_train.py` 的 cross_test 分支（唯一代码改动）

`main()` 按 `test_files` 是否存在且与 `train_files` 不同来分流（镜像 `holdout.py:resolve_holdout` 的判定）：

```
# 两模式都先做：在 train_files 上 CV，得 5 折模型 + 训练侧 OOF
oof, train_fold_metrics, model_paths = assemble_oof(df, X, y, groups, cfg, feature_cols, model_prefix)

test_files = cfg["data"].get("test_files")
if test_files and set(test_files) != set(train_files):     # cross_test 模式
    test_df = read_dataframe(test_files)
    X_test = test_df[feature_cols];  y_test = test_df[target_col]
    ens_proba, test_per_fold, test_agg = evaluate_cross_test(model_paths, X_test, y_test)
    eval_df, eval_y, eval_proba, mode = test_df, y_test, ens_proba, "cross_test"
else:                                                      # in-sample 模式（现状）
    eval_df, eval_y, eval_proba, mode = df, y, oof, "in_sample"

summary = evaluate_oof(eval_y, eval_proba)        # headline: auc/fnr_at_fpr5/working_points
summary["mode"] = mode
...（模式相关字段，见 §3）...
susp = audit_labels(eval_df, eval_proba, label_col=target_col, ...)   # 审计评估目标集
```

**全程复用已有函数**：`assemble_oof`（5 模型 + 训练 OOF）、`predict_ensemble`/`average_proba`（外部打分）、`evaluate_oof`（指标）、`audit_labels`（审计）。新增的只有 `evaluate_cross_test` helper（见 §2.1）+ `main()` 的分支。

`feature_cols` 仍由 `resolve_feature_cols(cfg["data"].get("feature_cols"), train_files, target_col)` 从 **train_files** 解析（与现有 `main.py` cross_test 行为一致）；`X_test = test_df[feature_cols]`（测试集须含这些列，否则 KeyError 显式失败）。

### 2.1 新 helper `evaluate_cross_test(model_paths, X, y)`（cv_train.py，lazy lightgbm）

```
probas = [lgb.Booster(model_file=p).predict(X.values) for p in model_paths]   # 每折预测外部
ens = average_proba(probas)                                                   # ensemble
per_fold = [{"fold": k, **_auc_fnr(y, probas[k])} for k in range(len(probas))]  # 各折在外部
agg = mean/std of per_fold auc & fnr_at_fpr5
return ens, per_fold, agg
```
`_auc_fnr(y, p)` 用 `roc_auc_score` + `fnr_at_fpr5`，单类时记 NaN（与 assemble_oof 守卫一致）。返回 `ens` 供 headline 评估 + audit。

---

## 3. cv.json 报告结构（两模式）

**in_sample 模式**（与现状一致，不变）：
`mode="in_sample"`、`auc/fnr_at_fpr5/working_points`（OOF）、`fold_metrics`（训练 OOF 各折）、`auc_mean/std`、`fnr_at_fpr5_mean/std`、`cv_folds`、`model_paths`、`n_pos/n_neg`（训练集）、`name`。

**cross_test 模式**（新增）：
- `mode="cross_test"`
- headline：`auc`/`fnr_at_fpr5`/`working_points` = **ensemble 在外部测试集**的指标（5 模型均值，比单模型稳）
- 方差：`test_per_fold`（5 折各自在外部测试集的 auc/fnr）+ `test_auc_mean/std`、`test_fnr_at_fpr5_mean/std`（跨数据集泛化稳不稳）
- 参考：`train_oof_auc`、`train_oof_fnr_at_fpr5`（训练侧 OOF，= `evaluate_oof(y_train, oof)`）+ `train_fold_metrics`（训练 OOF 各折）
- `cv_folds`、`model_paths`、`n_pos/n_neg`（**外部测试集**计数）、`name`

`suspects.csv`：in_sample = 训练集 OOF 里的可疑负例；cross_test = **外部测试集**里 ensemble 打分"像正例"的负例（跨数据集可疑 ID）。

---

## 4. 配置生成 `tools/spec_trainer/gen_cv_configs.py`

避免手写 30 个近重复文件：读现有每个 `in_*`/`cross_test_*` yaml，写 `cv_<name>` 变体。变换规则：

1. `data.group_col: sequence`（CV 分组防泄漏；源配置无此键）。
2. `training.cv_folds: 5`、`training.cv_seed: 42`；`training.valid_size: 0.15`（折内早停；覆盖源的 0.2）。保留 `num_boost_round`/`early_stopping_rounds`。
3. `audit: {suspect_threshold: 0.9, suspect_top_n: 200}`。
4. 输出路径改 cv 专用，**防与单 holdout 结果碰撞**：
   - `output.model_path`: `runs/spec_trainer/models/cv_<name>.txt`（→ `.fold0..4.txt`）
   - `output.result_path`: `runs/spec_trainer/results/cv_<name>.cv.json`
   - 删去 `figures_dir`（CV 不画 ROC 图）。
5. 其余（`train_files`/`test_files`/`feature_cols`/`target_col`/`model`/`num_boost_round`/`early_stopping_rounds`）原样保留。

生成的 **30 个 `cv_*.yaml` 也 commit**（符合项目静态配置惯例，`make` 无需先跑生成器）；生成器供源配置变动后同步重生成。生成器幂等（重跑结果一致）。

---

## 5. Makefile（镜像 train-all）

```makefile
CV_CLEAN_YAMLS  := cv_in_2da_clean cv_in_5da_clean cv_in_normal_clean \
                   cv_cross_test_2da_clean cv_cross_test_5da_clean cv_cross_test_normal_clean
# ...neg05/10/15/20 同构...

train-cv-clean-all:  $(CV_CLEAN_FEATURES)          # 6 实验：3 in + 3 cross_test
	@for y in $(CV_CLEAN_YAMLS); do $(PY) tools/spec_trainer/src/cv_train.py \
	    --config tools/spec_trainer/config/$$y.yaml --name $$y ... ; done
# train-cv-neg05-all / ...neg10/15/20-all 同构
train-cv-all:                                      # 串跑 5 组 = 30 实验
	$(MAKE) train-cv-clean-all
	$(MAKE) train-cv-neg05-all
	... neg10/15/20 ...
```

- 现有 `train-cv-2da`（单实验）保留。
- 各 `train-cv-*-all` 的 features.csv 前置依赖复用现有 `CLEAN_FEATURES`/`NEG05_FEATURES`/... 变量（缺则触发提取，与 `train-*-all` 同）。
- `cross_test` CV 的前置：需 train 侧两个数据集 + test 侧数据集的 features.csv 都在。

---

## 6. 边界 / 错误处理

- cross_test 的 `test_files` 缺列（feature_cols 不全）→ `test_df[feature_cols]` KeyError 显式失败（不静默）。
- cross_test 外部测试集单类（理论不会）→ `evaluate_cross_test` 的 per-fold auc 记 NaN，headline `evaluate_oof` 仍算（外部测试集含正负例）。
- `derive_paths` 的 `.json` 守卫沿用（result_path 必须 `.json`）。
- 生成器：源 yaml 缺失/字段异常 → 报错并指明文件；不覆盖手工编辑（重生成幂等）。

---

## 7. 测试

- **gen_cv_configs 单测**（无 lightgbm）：对一个样例 `in_*` 与一个 `cross_test_*` 源 dict，断言生成的 cv dict 含 `group_col: sequence`、`cv_folds/cv_seed`、`audit`，输出路径为 `cv_*.cv.json`/`.txt` 且无 `figures_dir`，`train_files`/`test_files` 原样。幂等性断言。
- **cross_test 模式集成测试**（`@requires_lgb`）：toy 训练集（数据集 A）+ toy 测试集（数据集 B，分布不同），跑 `cv_train.main`，断言 cv.json `mode=="cross_test"`、含 `test_per_fold`(5)/`test_auc_mean`/`train_oof_auc`，suspects.csv 行来自**测试集 B**，headline `n_pos/n_neg` 为 B 的计数。
- **evaluate_cross_test 单测**（`@requires_lgb`）：给定 model_paths + X/y，断言 ensemble = per-fold 均值、per_fold 长度 = 折数、agg mean/std 正确。
- **in_sample 回归**：现有 `test_main_writes_outputs` 仍绿（`mode=="in_sample"` 路径不变）。
- **Makefile**：`make -n train-cv-all PY="conda run -n jianyan python"` 展开为 30 次 cv_train.py 调用（顺序 clean→neg05→...→neg20）。
- 全套 `pytest tests/` 绿。

---

## 8. 兼容性

- **不改** `main.py` / `holdout.py` / `cv_core.py` / 既有 30 个 in_/cross_test 配置：旧的单 holdout `make train-all` 流程零改动。
- `cv_train.py` 仅在 `main()` 加分支 + 新 helper；现有 in_sample 路径（含 `test_main_writes_outputs`）行为不变。
- cv 输出路径（`cv_*.cv.json`/`cv_*.txt`）与单 holdout 输出（`*.json`/`*.txt`）不碰撞。

---

## 9. 不在范围

- 不改负样本生成、不改特征、不改 cv_core.py 既有 6 个函数。
- 不做 cross_test 的"在外部测试集上 OOF"（无意义——测试集模型没训过，只 ensemble 预测）。
- 不引入 cleanlab / 自动改标签 / 超参搜索 / 单调约束（独立后续项）。
- 不为 cv 配置做动态参数化运行（沿用项目静态 yaml 惯例，用生成器维护）。
