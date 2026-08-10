
# 使用说明

直接使用 `make exp1` 就是按照 配置进行训练。


# 参数说明

exp1 中 model

``` lightgbm
model:
  type: lightgbm  # 选择是什么模型
  params:         # 这个模型的参数
    boosting_type: gbdt
    objective: binary
    metric: [auc, binary_logloss]
    num_leaves: 31
    learning_rate: 0.05
    feature_fraction: 0.9
    bagging_fraction: 0.8
    verbose: -1
```


``` xgboost
model:
  type: xgboost
  params:
    objective: binary:logistic          # 二分类逻辑回归（输出概率）
    eval_metric: [auc, logloss]         # 对应 LightGBM 的 metric: [auc, binary_logloss]
    max_depth: 5                        # LightGBM 的 num_leaves=31 ≈ 完全二叉树深度 5（2^5=32）
    learning_rate: 0.05                 # 同 LightGBM
    subsample: 0.8                      # 对应 bagging_fraction（行采样）
    colsample_bytree: 0.9               # 对应 feature_fraction（列采样）
    verbosity: 0                        # 类似 verbose=-1，静默模式
    random_state: 42                    # 建议固定随机种子以保证可复现性
```

注意：

num_leaves: 31  max_depth: 5    LightGBM 用叶子数控制复杂度，XGBoost 用最大深度。31 个叶子 ≈ 深度 5（因为 25=3225=32）

learning_rate: 0.05 learning_rate: 0.05 两个意义相同
bagging_fraction: 0.8   subsample: 0.8  行采样比例
feature_fraction: 0.9   colsample_bytree: 0.9 每棵树使用的特征比例
metric: [auc, ...]   eval_metric: [auc, ...]  评估指标（XGBoost 不支持列表写法时可只写一个，但新版支持）
verbose: -1   verbosity: 0  控制日志输出（0=静默，1=警告，2=信息，3=调试）
tree_method: hist XGBoost 默认使用 exact 树构建方法，而 LightGBM 使用直方图近似。若追求速度，可加 tree_method: hist

# 输出结果


`results/exp1_report.json`

## 正式 CV 矩阵

`train-cv-*` 固定使用 `evidence_all` 特征组，只包含 MS1 observed、MS2
observed 和 MS2 predicted。Context 仅用于消融对照，不进入正式模型；六个
eligibility flags 只用于 `evidence_common` 队列过滤，也不进入模型。两个质量
审计未通过的预测特征 `spec_pattern_spearman_b` 和 `spec_pattern_SA_b` 同样
排除。在当前特征 schema 下最终为 152 个模型输入。

```bash
make train-cv-all \
  FEATURE_ROOT=/path/to/feature-snapshot \
  CV_OUTPUT_ROOT=/path/to/cv-output
```

`FEATURE_ROOT` 控制 15 个输入目录；`CV_OUTPUT_ROOT` 控制运行时配置、模型、
JSON、suspects 和日志的输出位置。默认值分别为 `runs` 和
`runs/spec_trainer`。

## 固定 E20 测试集比较负样本池

```bash
make train-fixed-test-negpool-2da \
  FEATURE_ROOT=/path/to/feature-snapshot \
  FIXED_NEGPOOL_OUTPUT_ROOT=/path/to/fixed-negpool-output
```

该入口严格验证 E5/E10/E20 的嵌套关系和共享特征值，用 neg20 主表冻结一个
sequence-held-out 测试集，并以同一正确样本、同一 outer/inner fold map 训练
M5/M10/M20。结果包含相同 E20 测试集的主表、分层错误集结果及按 sequence 的
1000 次配对 bootstrap；可用 `FIXED_NEGPOOL_BOOTSTRAPS=<N>` 调整次数。

## CV 决策阈值

为兼容现有数据和模型，CSV 仍保存 `label=1`（正确鉴定），模型仍输出
`trust_score=P(正确鉴定)`。对外评估统一使用错误鉴定为阳性：

```text
error_truth = 1 - label
error_score = 1 - trust_score
```

因此 FP 是正确鉴定被误报为错误，FN 是错误鉴定被漏判为正确。`train-cv-*`
默认使用训练数据的 OOF 预测选择 error-score 阈值，目标为 `FPR <= 10%`：

```yaml
operating_point:
  target_fpr: 0.10
```

规则为 `error_score >= error_threshold => 错误鉴定`（等价于
`trust_score <= trust_threshold`）。阈值只根据训练 OOF 中实际正确鉴定的
error score 确定；`cross_test` 会锁定该阈值后再评估外部测试集，不会根据
测试标签重新选择。结果写入 CV JSON 的 `operating_point`，其中
`train_oof_metrics` 是阈值选择侧指标，`test_metrics` 是外部测试集上的实际
FPR、FNR 与 error recall。发生域偏移时，外部测试 FPR 可能高于 10%，此时
应报告实际值，不能在测试集上重新调阈值。

内层 early-stopping 验证集按 `sequence` 分组抽取，同一 sequence 不跨
train/valid；允许一个 sequence 同时包含正确样本及其构造错误样本。每折 JSON
的 `fold_metrics[].split_counts` 会记录 train/valid/OOF 的正确/错误行数、
相关 sequence 数和 mixed group 数。默认要求每块的少数类至少有 5 个
sequence，否则训练直接
失败并提示调整折数、验证比例或补充负例：

```yaml
training:
  min_class_groups_per_split: 5
```

新结果的核心字段示例：

```json
{
  "metric_semantics": "error_identification_positive_v1",
  "positive_class": "incorrect_identification",
  "roc_auc": 0.90,
  "error_pr_auc": 0.62,
  "fnr_at_fpr5": 0.28,
  "error_recall_at_fpr10": 0.80,
  "n_actual_correct": 1000,
  "n_actual_error": 200
}
```

没有 `metric_semantics` 的历史结果使用旧口径，不应与新结果直接混合。

图

特征重要性


图 

Youden 点：TPR - FPR 最大化的点。 （TPR 就是召回率， FPR： 假阳率）
