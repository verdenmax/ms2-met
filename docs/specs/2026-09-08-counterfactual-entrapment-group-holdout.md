# Counterfactual 负样本与真实 entrapment 的统一分组留出实验

## 实验目的

当前 counterfactual 训练结果中的错误样本全部来自人工生成。它可以回答模型是否能识别这些生成方式，却不能单独证明模型能识别真实错误。第一阶段实验要回答一个更窄、可复现的问题：在同一 2Da 数据域内，分别用 composition shuffle、KR-position shuffle、local mass-gap 训练后，对从目标 FASTA 筛出的真实 entrapment 错误能检出多少；同时比较三种来源合并训练是否有增益。

该实验不按 Rep 或 raw 文件划分。相同正样本可能出现在多个 Rep，按 Rep 留出会造成序列泄漏。划分单位沿用仓库现有的 connected grouping：L/I 归一化序列、`peptide_group_id`、`parent_id`、`group_id`、`candidate_family_id` 和候选自身序列共同形成一个连通分量。一个连通分量只能属于训练集或测试集。

## 输入及真实错误的定义

实验读取两张已经完成特征提取的表：

1. counterfactual 表：`gold_positive` 正样本和三种 synthetic 错误；
2. 2Da clean 表：只取存储标签 `label=0` 的行，并在实验表中标记为 `gold_entrapment`。

这里的 `gold_entrapment` 是现有 clean 数据生产规则下的操作性名称：它来自配置的 target FASTA、物种标记和已有过滤规则，不表示人工逐条确认的绝对真值。后续报告必须同时保留这一定义和源文件路径。

存储和评估语义保持仓库约定：CSV 中 `label=1` 表示正确鉴定，`label=0` 表示错误鉴定；模型输出 `trust_score=P(correct)`。评估边界使用 `error_truth=1-label` 和 `error_score=1-trust_score`，错误鉴定是统计正类。

## 冻结划分过程

构建器按以下顺序工作：

1. 检查 counterfactual 来源与存储标签是否一致；
2. 删除 `parent_id` 无法解析到 `gold_positive` 的孤儿 synthetic 行，并记录分来源计数；
3. 从 clean 表只保留 `label=0`，标记为 `gold_entrapment`，为其按 L/I 归一化序列生成 `peptide_group_id`；
4. 合并两张表，然后调用 `tools/spec_trainer/src/sample_groups.py` 计算完整连通分组；
5. 分组完成后才应用共同的 `evidence_observed` cohort；
6. 删除 cohort 过滤后失去可评估 parent 的 synthetic 行；
7. 所有包含 `gold_entrapment` 的连通分量强制进入测试集；
8. 对其余含正样本的连通分量，以 seed 42 的 SHA-256 稳定排序选择 20% 进入测试集；
9. 测试集中的 `gold_positive` 与全部通过 cohort 的 `gold_entrapment` 组成唯一的主测试集；测试分组中的 synthetic 行单独写入诊断文件，不参与真实错误主指标。

这个设计允许同一个 raw/Rep 同时出现在训练与测试中，因为目标是“同域、未见 peptide/family”的泛化。`split_audit.json` 会显式报告 raw 重叠数量以及 train/test 连通分组重叠数量。后者必须为 0。

## 四个训练模型

四个模型使用完全相同的训练正样本、分组划分、真实测试集、特征组和 LightGBM 参数：

| 模型 | 每个训练 `parent_id` 使用的 synthetic 错误 |
|---|---|
| M-C | 一个稳定选择的 `synthetic_composition_shuffle` |
| M-K | 一个稳定选择的 `synthetic_kr_position_shuffle` |
| M-L | 一个稳定选择的 `synthetic_local_mass_gap` |
| M-All | 上述每种来源各一个，最多三个 |

M-C、M-K、M-L 通过“每个 parent、每个来源一个候选”实现近似等量比较。M-All 的错误样本数约为单源模型的三倍，因此它是合并训练效果，不是严格控制训练行数后的来源消融。若初版结果有价值，下一阶段再增加一个总错误数匹配的 M-All-balanced。

所有模型使用 `ms1_ms2_no_prediction` 特征臂与 `evidence_observed` cohort。`cv_train.py` 在训练组内部继续使用 connected group CV。FPR 1%、5%、10% 的正式外部测试决策使用每个 fold 成员各自的 outer-OOF `error_threshold`，在外部测试上多数投票；外部标签得到的阈值只出现在 `retrospective_test_working_points`，属于 oracle 分析。

## 运行方式

当前两份实际输入位于不同结果快照，因此建议把路径都写全。以下命令一次完成冻结数据和四个模型训练：

```bash
make counterfactual-2da-group-holdout \
  PY=/home/verden/.conda/envs/jianyan/bin/python \
  COUNTERFACTUAL_2DA_FEATURES=/home/verden/share/2026_04_27_kongweisa_diann_ZHOUHUdataset/feature_result/ms2-met-runs-09-07/counterfactual_2da_label_dev_train/features.csv \
  COUNTERFACTUAL_2DA_ENTRAPMENT_FEATURES=/home/verden/share/2026_04_27_kongweisa_diann_ZHOUHUdataset/feature_result/ms2-met-runs-08-20/baseline_2da_clean/features.csv \
  COUNTERFACTUAL_2DA_GROUP_HOLDOUT_ROOT=/home/verden/share/2026_04_27_kongweisa_diann_ZHOUHUdataset/feature_result/ms2-met-runs-09-08/counterfactual_2da_group_holdout
```

如果希望先审计冻结数据，再决定是否训练，可以分两步运行：

```bash
make counterfactual-2da-group-holdout-build \
  PY=/home/verden/.conda/envs/jianyan/bin/python \
  COUNTERFACTUAL_2DA_FEATURES=/path/to/counterfactual/features.csv \
  COUNTERFACTUAL_2DA_ENTRAPMENT_FEATURES=/path/to/baseline_2da_clean/features.csv \
  COUNTERFACTUAL_2DA_GROUP_HOLDOUT_ROOT=/path/to/output

make counterfactual-2da-group-holdout-train \
  PY=/home/verden/.conda/envs/jianyan/bin/python \
  COUNTERFACTUAL_2DA_GROUP_HOLDOUT_ROOT=/path/to/output
```

构建器默认拒绝覆盖已有 bundle。训练结果也默认拒绝覆盖；只有明确设置 `CV_OVERWRITE=1` 才会替换已有训练结果。

## 输出和检查顺序

输出根目录包含：

| 文件 | 含义 |
|---|---|
| `train_m_c.csv`、`train_m_k.csv`、`train_m_l.csv`、`train_m_all.csv` | 四套冻结训练表 |
| `test_gold_entrapment.csv` | 四个模型共用的正样本加真实 entrapment 主测试集 |
| `synthetic_diagnostics.csv` | 测试分组内的 synthetic 行，只用于生成器诊断 |
| `split_manifest.csv` | 每行的稳定 ID、连通分组、split 和四个模型成员关系 |
| `split_audit.json` | 孤儿、cohort、分组、来源、raw 重叠和零泄漏审计 |
| `artifact_checksums.json`、`bundle_status.json` | 冻结文件 SHA-256 与完成标记 |
| `configs/{m_c,m_k,m_l,m_all}.yaml` | 指向同一个测试集的训练配置 |
| `training/<model>/training.cv.json` | 每个模型的正式结果 |
| `training/<model>/training.cv.test_scores.csv` | 外部测试分数与 fold 投票比例 |

首先检查 `bundle_status.json` 的 `status=complete`，然后确认 `split_audit.json` 中：

- `metric_semantics=error_identification_positive_v1`；
- `validation.n_train_primary_test_overlapping_groups=0`；
- `validation.all_entrapment_groups_in_test=true`；
- `inputs.n_orphan_synthetic_rows_dropped` 与分来源计数符合预期；
- 四个训练表的 `gold_positive` 计数一致；
- 主测试错误来源只有 `gold_entrapment`。

结果比较优先读取各模型 `operating_points.fpr_5.external_ensemble.test_metrics` 的 `fpr`、`fnr` 和 `error_recall`。同时报告 `n_actual_correct`、`n_actual_error`、ROC-AUC 和 `error_pr_auc`。`retrospective_test_working_points` 只能作为区分能力的 oracle 参考，不能当作可部署的锁定阈值结果。

## 初版实验的边界

这是同一个 2Da 数据域中的未见 peptide/family 留出，并非跨数据集或跨实验室验证。entrapment 标签仍由现有目标 FASTA 规则产生，其纯度需要后续抽样核验。由于四个模型共享同一个测试集，结果差异是配对比较；fold 间标准差只是模型成员的描述性离散程度，不是独立重复或置信区间。
