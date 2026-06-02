
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

示例结果: 

```
"accuracy": 0.867053755800786,  准确率：(TP+TN) / ALL
"auc": 0.8965088797455758,      模型区分正负样本的能力，不受数据分布影响
"confusion_matrix": [
    [
        89943,
        131
    ],
    [
        23819,
        66255
    ]
],
"classification_report": {
    "0": {
        "precision": 0.790624285789631,
        "recall": 0.9985456402513488,
        "f1-score": 0.8825035813104652,
        "support": 90074.0
    },
    "1": {
        "precision": 0.9980266923748983,
        "recall": 0.7355618713502231,
        "f1-score": 0.8469257318164387,
        "support": 90074.0
    },
    "accuracy": 0.867053755800786,
    "macro avg": {
        "precision": 0.8943254890822646,
        "recall": 0.867053755800786,
        "f1-score": 0.864714656563452,
        "support": 180148.0
    },
    "weighted avg": {
        "precision": 0.8943254890822646,
        "recall": 0.867053755800786,
        "f1-score": 0.8647146565634519,
        "support": 180148.0
    }
}
```

图

特征重要性


图 

Youden 点：TPR - FPR 最大化的点。 （TPR 就是召回率， FPR： 假阳率）
