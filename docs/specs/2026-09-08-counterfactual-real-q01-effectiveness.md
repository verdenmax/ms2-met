# Counterfactual 负样本在真实 q≤1% entrapment 上的增益实验

## 要回答的问题

本实验检验一个预先限定的命题：在真实 q≤1% 正确鉴定和 entrapment
错误上训练的基线模型之外，加入 counterfactual 负样本后，模型能否在
**训练中从未出现的真实 peptide/candidate family** 上检出更多 entrapment
错误，并且不明显增加对正确鉴定的误报。

此前的单次 group holdout 可以比较 C、K、L 三种生成来源，但没有包含
“只用真实样本训练”的 M-Real 对照，也没有让每个真实样本都得到一次严格
外层留出预测。新实验使用固定的五折外层评估，每折约 80% 的真实连通组用于
训练，约 20% 只用于最终测试。五折结果合并后，每个真实行恰好被测试一次。

实验给出的结论是这些合成负样本对当前 2Da、当前 q≤1% 数据域是否有
**增量训练价值**。它不能单独证明生成机制与真实错误机制完全相同，也不能
替代另一个批次、实验室或标准合成肽上的外部复现。

## 标签和评估语义

输入 CSV 保持仓库存储约定：

- `label=1`、`label_type=positive`：操作性正确鉴定；
- `label=0`、`label_type=negative`：来自当前 entrapment 流程的错误鉴定；
- 模型输出 `trust_score=P(correct identification)`。

构建器要求 real 表每一行 `q_value <= 0.01`，并检查 `label_type` 与 `label`
一一对应。这里的“真实”表示这些行来自实际搜索和谱图特征，而不是人工生成；
“正确”和“错误”仍是当前 target/entrapment 数据生产规则下的操作性标签，
不是逐谱人工确认。

统计评估以错误鉴定为正类：

```text
error_truth = 1 - stored_label
error_score = 1 - trust_score
```

FPR 是正确鉴定中被错误标记为可疑的比例，error recall 是 entrapment 错误中
被成功检出的比例。所有 JSON/CSV 结果写入
`metric_semantics=error_identification_positive_v1` 和
`positive_class=incorrect_identification`。

## 防泄漏外层划分

构建器先合并 counterfactual 表和 real q≤1% 表，再连接以下关系：

- L/I 等价的 `peptide_group_id`；
- `parent_id`、`group_id`、`candidate_family_id`；
- parent 与它生成的所有候选；
- real 表中相同 peptide family 的不同 Rep、raw 和电荷观测。

包括 orphan synthetic 在内的全部输入行都会参与关系图；orphan 随后从模型
候选中删除，但仍保留其已经形成的连通关系。real 表已有的 family 字段也在
图构建时保留。完整关系图建立后才应用 `evidence_observed` cohort。
counterfactual parent 正例只充当图中的桥，不进入任何模型。外层折按最终连通分量划分，因此一个测试
正样本的相同 peptide、不同 Rep、parent 以及生成候选都不会进入训练侧。
这个约束比仅检查行 ID 不重叠更严格。

每个外层折执行以下步骤：

1. 取其余四折的 real 正确和 real entrapment 行作为公共训练底表；
2. 只允许连接到训练侧 real correct family 的 synthetic 候选进入增强模型；
3. 每个 `parent_id × source` 以稳定哈希选择一个候选；
4. 测试表只含当前折的 real correct 和 real entrapment，不含任何 synthetic 行；
5. 五个模型使用完全相同的测试成员、特征臂和 LightGBM 参数。

实验不按 Rep 划分。Rep 之间可能含相同 peptide，按 Rep 留出会把同一正样本
家族同时放进训练和测试。当前设计测量同一 2Da 数据域内的未见 family 泛化，
不把它描述成跨数据集泛化。

## 五个模型

| 模型 | 训练错误样本 |
|---|---|
| M-Real | 训练侧 real q≤1% entrapment |
| M-Real+C | M-Real + composition shuffle |
| M-Real+K | M-Real + KR-position shuffle |
| M-Real+L | M-Real + local mass-gap |
| M-Real+All | M-Real + C、K、L |

M-Real 是回答“加入人工构造负样本是否真的改善真实错误检测”的关键对照。
C、K、L 都按每个 parent 一个候选选择，使三个单来源模型的 synthetic 数量
近似相等，可以比较哪个来源更有效。M-Real+All 的 synthetic 数量约为单来源
的三倍，因此它表示实际合并训练收益，不能解释为严格等行数的来源消融。

这个实验直接检验的是“把该来源的样本加入训练是否有用”。如果需要进一步
区分收益来自新错误分布还是单纯来自更大的负类权重，应在结果有增益后增加
重复 real entrapment 或显式 sample-weight 的等负类权重对照。

## 内层训练和锁定阈值

每个外层训练集交给现有 `tools/spec_trainer/src/cv_train.py`：

- 使用 `ms1_ms2_no_prediction` 的 133 个注册特征；
- 使用 connected `leakage_group_id` 做五折内层 CV；五个模型共享预先冻结的
  real-group 内层 OOF 折和早停验证组，synthetic 继承其 parent family 分区；
- 训练器把 `leakage_group_id` 明确视为由完整输入关系图冻结的分组，验证当前
  训练子集中仍可见的关系没有跨组后直接使用；外层留出导致 parent 不在当前
  子集时，不会把该组误报为关系保护不完整；
- 每个成员模型只用自己的 inner OOF 分数校准 FPR 1%、5%、10% 的
  `error_threshold`；
- 对外层测试集逐成员判定，然后以五成员多数投票得到正式结果。

外层测试标签不会参与正式阈值选择。测试标签只用于计算已经锁定的多数投票
在测试集上实际产生的 FP、FN、FPR、FNR 和 error recall。连续 ensemble
分数上的 pooled ROC-AUC 和 error PR-AUC 是补充排序指标。

## 预先声明的主比较

主工作点是 FPR 5%。四个增强模型分别与 M-Real 在完全相同的外层测试行上
配对比较：

```text
ΔRecall = error_recall(augmented) - error_recall(M-Real)
ΔFPR    = observed_fpr(augmented) - observed_fpr(M-Real)
```

置信区间使用按 `leakage_group_id` 的分层 paired cluster bootstrap：correct
组和 error 组分别有放回抽样，同一抽样权重同时应用于基线和增强模型。默认
1000 次，不把五个 CV 折当作五个独立实验。四个模型比较各包含 recall 增益
和 FPR 非劣效两个单侧判断；成功判定使用覆盖八个单侧边界的 Bonferroni
family-wise α=0.05 校正，同时保留未校正的 95% CI 供描述。

某一增强模型只有同时满足以下预声明条件，才写为
`effectiveness_supported=true`：

- 观察到的 FPR 5% error-recall 增益至少 3 个百分点；
- recall 增益的 family-wise bootstrap 下界大于 0；
- 实际 FPR 增量的 family-wise bootstrap 上界不超过 1 个百分点。

如果不满足，只能报告当前数据没有支持预声明的实用增益；不能用测试结果重新
选择阈值、改变成功线或筛选 generator 参数后再把同一测试称为最终验证。

## 运行方式

当前数据可用一个 Make target 完成 bundle 构建、25 次训练和汇总：

```bash
make counterfactual-2da-effectiveness \
  PY=/home/verden/.conda/envs/jianyan/bin/python \
  COUNTERFACTUAL_2DA_FEATURES=/home/verden/share/2026_04_27_kongweisa_diann_ZHOUHUdataset/feature_result/ms2-met-runs-09-07/counterfactual_2da_label_dev_train/features.csv \
  COUNTERFACTUAL_2DA_REAL_Q01_FEATURES=/home/verden/share/2026_04_27_kongweisa_diann_ZHOUHUdataset/feature_result/ms2-met-runs-08-20/baseline_2da_clean/features.csv \
  COUNTERFACTUAL_2DA_EFFECTIVENESS_ROOT=/home/verden/share/2026_04_27_kongweisa_diann_ZHOUHUdataset/feature_result/ms2-met-runs-09-08/counterfactual_2da_real_q01_effectiveness
```

默认输入分别是 `runs/counterfactual_2da_label_dev_train/features.csv` 和
`runs/baseline_2da_clean/features.csv`；默认输出位于
`runs/spec_trainer/counterfactual-2da-real-q01-effectiveness/`。

也可以按构建、训练、汇总三个阶段运行：

```bash
make counterfactual-2da-effectiveness-build ...
make counterfactual-2da-effectiveness-train ...
make counterfactual-2da-effectiveness-summarize ...
```

构建器原子写入并拒绝覆盖已有 bundle。训练默认也拒绝覆盖已存在结果；确实
需要在同一冻结 bundle 上重训时，运行 `counterfactual-2da-effectiveness-train`
并传入 `CV_OVERWRITE=1`，随后单独运行 summarize。完整 target 会先执行 build，
因此不能用于覆盖已有 bundle。当前实际 bundle
约 1.5 GB，完整实验顺序运行 5 个外层折 × 5 个模型。

## 输出

| 路径 | 内容 |
|---|---|
| `split_audit.json` | 输入 provenance、cohort、图分组、各折计数和零泄漏断言 |
| `outer_fold_manifest.csv` | 每个 real/synthetic 行的稳定 ID、连通组和外层折 |
| `synthetic_diagnostics.csv` | 所有符合条件的 synthetic 候选及入选标志 |
| `folds/fold_k/train_real_q01.csv` | 第 k 折公共 real 训练底表 |
| `folds/fold_k/train_synthetic_*.csv` | 第 k 折训练侧各来源候选 |
| `folds/fold_k/test_real_q01.csv` | 第 k 折只含 real 的测试表 |
| `configs/fold_k/*.yaml` | 该折五个冻结训练配置 |
| `training/fold_k/<model>/training.cv.json` | 内层 OOF、锁定阈值和外层测试结果 |
| `pooled_real_test_predictions.csv` | 每个 real 行恰好一次的五模型外层预测 |
| `effectiveness_summary.csv/json` | 五模型 pooled 指标和标准工作点 |
| `paired_group_bootstrap.csv` | 四个增强模型相对 M-Real 的主比较和 CI |

首先检查 `bundle_status.json` 为 `complete`，再确认：

- `every_real_row_is_tested_once=true`；
- `every_outer_fold_has_zero_train_test_group_overlap=true`；
- `test_contains_no_synthetic_rows=true`；
- 训练开始及汇总开始前的 frozen artifact SHA-256 检查通过；
- pooled 的 `n_actual_correct + n_actual_error` 等于审计中的 real cohort 行数；
- 主结论读取 `paired_group_bootstrap.csv`，阈值工作点读取 JSON 中
  `models.<model>.operating_points.fpr_5.external_ensemble.test_metrics`。

用当前 09-07 counterfactual 和 08-20 real 输入做构建审计时，real 输入
114117 行，经共同 cohort 后为 106666 个 correct 行和 561 个 entrapment
error 行；每折测试约 21333 个 correct 和 112–113 个 error。可用 synthetic
候选中有 9 个 orphan local mass-gap 行仅作为完整关系图的桥，其中 4 行通过
共同 cohort 后仍从训练候选中排除；另有 4 个非 orphan local mass-gap 行没有
连接到可评估 real correct family，也从训练候选中删除。上述计数属于这次输入
快照，正式运行应以输出审计文件为准。
