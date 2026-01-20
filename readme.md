
# 使用说明

直接使用 `make exp1` 就是按照 配置进行训练。



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
