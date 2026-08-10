import pandas as pd
import lightgbm as lgb
import yaml
import json
import argparse
import os
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split  # 划分数据集
from sklearn.metrics import (  # 评估模型效果的各种指标
    accuracy_score, average_precision_score, roc_auc_score,
    classification_report, confusion_matrix
)
from sklearn.metrics import roc_curve, auc
import numpy as np
from cv_core import METRIC_SEMANTICS_VERSION, as_error_detection


def load_data(file_paths, feature_cols, target_col):
    dfs = []

    # 遍历每个文件路径
    # 使用 panda 进行 read csv
    for f in file_paths:
        if not os.path.exists(f):
            raise FileNotFoundError(f"Data file not found: {f}")
        dfs.append(pd.read_csv(f))

    # 将所有文件 concat 到一个 df 中
    # 获得 feature 列 和 target 列
    df = pd.concat(dfs, ignore_index=True)
    X = df[feature_cols]
    y = df[target_col]
    return X, y


def save_feature_importance(model, feature_names, output_path):
    """ 画出 不同特征重要性条形图 """
    # 获取特征重要性
    importance = model.feature_importance(importance_type='gain')

    # 按重要性从大到小排序
    indices = np.argsort(importance)[::-1]

    # 画条形图，显示不同特征的重要性
    plt.figure(figsize=(10, 6))
    plt.title("Feature Importance (Gain)")
    plt.barh(range(len(indices)), importance[indices], align="center")
    plt.yticks(range(len(indices)), [feature_names[i] for i in indices])
    plt.gca().invert_yaxis()
    plt.tight_layout()
    plt.savefig(output_path, dpi=150)
    plt.close()


def save_roc_figure(y_true, y_proba, output_path):
    """Plot ROC with incorrect identifications as the positive class."""
    error_truth, error_score = as_error_detection(y_true, y_proba)
    fpr, tpr, thresholds = roc_curve(error_truth, error_score)

    roc_auc = auc(fpr, tpr)
    # 计算约登指数
    youden_idx = tpr - fpr
    best_idx = np.argmax(youden_idx)
    best_threshold = thresholds[best_idx]
    best_fpr = fpr[best_idx]
    best_tpr = tpr[best_idx]

    # metrics['best_threshold'] = float(best_threshold)
    # metrics['best_fpr'] = float(best_fpr)
    # metrics['best_tpr'] = float(best_tpr)
    plt.figure(figsize=(8, 6))
    plt.plot(fpr, tpr, color='darkorange', lw=2,
             label=f'ROC curve (area = {roc_auc:.2f})')
    plt.plot([0, 1], [0, 1], color='navy', lw=2, linestyle='--')
    plt.scatter(best_fpr, best_tpr, color='red', s=50, zorder=5,
                label=f'Best threshold (Youden) = {best_threshold:.2f}')
    plt.xlim([0.0, 1.0])
    plt.ylim([0.0, 1.05])
    plt.xlabel('False Positive Rate')
    plt.ylabel('True Positive Rate')
    plt.title('Receiver Operating Characteristic (ROC) Curve')
    plt.legend(loc="lower right")
    plt.tight_layout()
    plt.savefig(output_path, dpi=150)
    plt.close()


def evaluate_and_report(
        y_true, y_pred, y_proba,
        feature_names=None, model=None,
        report_path=None, fig_path=None,
        roc_path=None):
    error_truth = 1 - np.asarray(y_true, dtype=int)
    predicted_error = 1 - np.asarray(y_pred, dtype=int)
    error_score = None if y_proba is None else 1.0 - np.asarray(y_proba)
    metrics = {
        "metric_semantics": METRIC_SEMANTICS_VERSION,
        "positive_class": "incorrect_identification",
        "accuracy": float(accuracy_score(error_truth, predicted_error)),
        "roc_auc": float(roc_auc_score(error_truth, error_score))
        if error_score is not None else None,
        "error_pr_auc": float(
            average_precision_score(error_truth, error_score))
        if error_score is not None else None,
        "error_threshold": 0.5,
        "trust_threshold": 0.5,
        "confusion_matrix_labels": [
            "correct_identification", "incorrect_identification"],
        "confusion_matrix": confusion_matrix(
            error_truth, predicted_error, labels=[0, 1]).tolist(),
        "classification_report": classification_report(
            error_truth, predicted_error, labels=[0, 1],
            output_dict=True, zero_division=0,
            target_names=["correct_identification",
                          "incorrect_identification"]),
    }

    # 计算ROC曲线和最佳阈值（约登指数）
    # 画出 roc figure
    if y_proba is not None and roc_path:
        os.makedirs(os.path.dirname(roc_path), exist_ok=True)
        save_roc_figure(y_true, y_proba, roc_path)

    if report_path:
        os.makedirs(os.path.dirname(report_path), exist_ok=True)
        with open(report_path, 'w') as f:
            json.dump(metrics, f, indent=4)

    if model is not None and fig_path and feature_names:
        os.makedirs(os.path.dirname(fig_path), exist_ok=True)
        save_feature_importance(model, feature_names, fig_path)

    return metrics


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", type=str, required=True)
    parser.add_argument("--name", type=str, required=True)  # e.g., "exp1"
    args = parser.parse_args()

    # 加载配置文件
    with open(args.config) as f:
        cfg = yaml.safe_load(f)

    # Load data
    X_train, y_train = load_data(
        cfg['data']['train_files'],
        cfg['data']['feature_cols'],
        cfg['data']['target_col']
    )
    X_test, y_test = load_data(
        cfg['data']['test_files'],
        cfg['data']['feature_cols'],
        cfg['data']['target_col']
    )

    # 划分验证集
    # Optional validation split
    valid_size = cfg['training'].get('valid_size', 0.0)
    if valid_size > 0:
        X_train, X_val, y_train, y_val = train_test_split(
            X_train, y_train, test_size=valid_size, random_state=42, stratify=y_train
        )
        val_data = lgb.Dataset(X_val, label=y_val)
        valid_sets = [val_data]
        valid_names = ['valid']
    else:
        valid_sets = None
        valid_names = None

    # 准备训练数据
    train_data = lgb.Dataset(X_train, label=y_train)

    # 开始训练模型
    # Train
    model = lgb.train(
        params=cfg['model'],  # 模型参数
        train_set=train_data,
        num_boost_round=cfg['training']['num_boost_round'],  # 训练轮数
        valid_sets=valid_sets,  # 验证数据（如果有）
        valid_names=valid_names,  # 验证集名称
        callbacks=[  # 训练过程中的回调函数
            # 早停机制
            lgb.early_stopping(
                stopping_rounds=cfg['training']['early_stopping_rounds']),

            # 每 100 轮输出一次日志
            lgb.log_evaluation(100)
        ]
    )

    # Save model
    # 模型保存路径
    model_path = f"models/{args.name}.txt"
    os.makedirs("models", exist_ok=True)
    model.save_model(model_path)

    # Predict
    # 用模型预测测试数据
    y_proba = model.predict(X_test)
    y_pred = (y_proba > 0.5).astype(int)

    # Save report and figure
    # 保存评估报告和特征重要性图
    report_path = f"results/{args.name}_report.json"
    fig_path = f"figures/{args.name}_importance.png"
    roc_path = f"figures/{args.name}_roc_curve.png"

    evaluate_and_report(
        y_test, y_pred, y_proba,
        feature_names=cfg['data']['feature_cols'],
        model=model,
        report_path=report_path,
        fig_path=fig_path,
        roc_path=roc_path
    )

    print(f"✅ Experiment '{args.name}' completed.")
    print(f"   Model: {model_path}")
    print(f"   Report: {report_path}")
    print(f"   Feature importance: {fig_path}")
    print(f"   ROC curve: {roc_path}")


if __name__ == "__main__":
    main()
