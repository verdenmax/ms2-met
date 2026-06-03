import os
import sys

# 让 src/ 进入 sys.path，使得 main.py 可以从任何工作目录被调用
# 仿照 tools/eval_baseline.py 的 pattern
_SRC_ROOT = os.path.dirname(os.path.abspath(__file__))
if _SRC_ROOT not in sys.path:
    sys.path.insert(0, _SRC_ROOT)

from models.model_manager import ModelManager
import pandas as pd
import yaml
import json
import argparse
import os
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split  # 划分数据集
from sklearn.metrics import (  # 评估模型效果的各种指标
    accuracy_score, roc_auc_score, classification_report, confusion_matrix
)
from sklearn.metrics import roc_curve, auc
import numpy as np

import logging
from rich.logging import RichHandler


# Feature column resolution moved to feature_cols.py for testability
# without lightgbm dependency (review finding I-ST1, 2026-06-03 audit).
from feature_cols import (
    META_COLUMNS,
    EXCLUDED_EXTRA,
    resolve_feature_cols as _resolve_feature_cols,
)


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
    """ 画出 roc 曲线图 """
    fpr, tpr, thresholds = roc_curve(y_true, y_proba)

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
    # 计算各种评估指标
    metrics = {
        "accuracy": float(accuracy_score(y_true, y_pred)),  # 准确率
        "auc": float(roc_auc_score(y_true, y_proba)) if y_proba is not None else None,
        "confusion_matrix": confusion_matrix(y_true, y_pred).tolist(),  # 混淆矩阵
        # 详细分类报告
        "classification_report": classification_report(y_true, y_pred, output_dict=True),
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
    parser.add_argument("--name", type=str, required=True)
    parser.add_argument(
        '--logpath', help='日志文件路径，默认为 ./spec.log',
        default="./spec.log")

    args = parser.parse_args()

    # 设置日志文件handle
    file_handler = logging.FileHandler(args.logpath, encoding="utf-8")
    file_formatter = logging.Formatter(
        "%(asctime)s | %(levelname)s | %(name)s | %(message)s"
    )
    file_handler.setFormatter(file_formatter)

    # 注册日志
    logging.basicConfig(level=logging.INFO, handlers=[
                        RichHandler(), file_handler])

    # 加载配置文件
    with open(args.config) as f:
        cfg = yaml.safe_load(f)

    # Load data
    target_col = cfg['data']['target_col']
    feature_cols = _resolve_feature_cols(
        explicit=cfg['data'].get('feature_cols'),
        sample_csv_paths=cfg['data']['train_files'],
        target_col=target_col,
    )
    logging.info(f"using {len(feature_cols)} feature columns")

    X_train, y_train = load_data(
        cfg['data']['train_files'],
        feature_cols,
        target_col,
    )

    # Held-out set resolution (review finding I-ST2, 2026-06-03 audit).
    # Helper extracted to holdout.py for testability (rubber-duck N4).
    from holdout import resolve_holdout
    X_train, X_test, y_train, y_test = resolve_holdout(
        X_train, y_train,
        train_files=cfg['data']['train_files'],
        test_files=cfg['data'].get('test_files') or [],
        test_size=cfg['data'].get('test_size', 0.0),
        feature_cols=feature_cols,
        target_col=target_col,
        loader=load_data,
    )

    # 划分验证集
    # Optional validation split
    X_val, y_val = None, None
    valid_size = cfg['training'].get('valid_size', 0.0)
    if valid_size > 0:
        X_train, X_val, y_train, y_val = train_test_split(
            X_train, y_train, test_size=valid_size, random_state=42, stratify=y_train
        )

    # 创建模型（关键改动）
    model = ModelManager.create(cfg)

    # Train
    model.fit(X_train, y_train, X_val, y_val)

    # Save model (ensure parent dir exists; direct python invocation must
    # not rely on Makefile pre-creating runs/spec_trainer/models/).
    model_path = cfg['output']['model_path']
    os.makedirs(os.path.dirname(model_path) or ".", exist_ok=True)
    model.save(model_path)

    # Predict
    y_proba = model.predict_proba(X_test)
    y_pred = model.predict(X_test)

    # Evaluate
    report_path = cfg['output']['result_path']
    fig_path = f"runs/spec_trainer/figures/{args.name}_importance.png"
    roc_path = f"runs/spec_trainer/figures/{args.name}_roc_curve.png"

    evaluate_and_report(
        y_test, y_pred, y_proba,
        feature_names=feature_cols,
        model=model,  # 传入 model 对象，内部会调用 feature_importance()
        report_path=report_path,
        fig_path=fig_path,
        roc_path=roc_path
    )

    logging.info(f"✅ Experiment '{args.name}' completed.")
    logging.info(f"   Model: {model_path}")
    logging.info(f"   Report: {report_path}")
    logging.info(f"   Feature importance: {fig_path}")
    logging.info(f"   ROC curve: {roc_path}")


if __name__ == "__main__":
    main()
