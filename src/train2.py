import pandas as pd
import lightgbm as lgb
import yaml
import json
import argparse
import os
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split
from sklearn.metrics import (
    accuracy_score, roc_auc_score, classification_report, confusion_matrix
)
import numpy as np


def load_data(file_paths, feature_cols, target_col):
    dfs = []
    for f in file_paths:
        if not os.path.exists(f):
            raise FileNotFoundError(f"Data file not found: {f}")
        dfs.append(pd.read_csv(f))
    df = pd.concat(dfs, ignore_index=True)
    X = df[feature_cols]
    y = df[target_col]
    return X, y


def save_feature_importance(model, feature_names, output_path):
    importance = model.feature_importance(importance_type='gain')
    indices = np.argsort(importance)[::-1]

    plt.figure(figsize=(10, 6))
    plt.title("Feature Importance (Gain)")
    plt.barh(range(len(indices)), importance[indices], align="center")
    plt.yticks(range(len(indices)), [feature_names[i] for i in indices])
    plt.gca().invert_yaxis()
    plt.tight_layout()
    plt.savefig(output_path, dpi=150)
    plt.close()


def evaluate_and_report(y_true, y_pred, y_proba, feature_names=None, model=None, report_path=None, fig_path=None):
    metrics = {
        "accuracy": float(accuracy_score(y_true, y_pred)),
        "auc": float(roc_auc_score(y_true, y_proba)) if y_proba is not None else None,
        "confusion_matrix": confusion_matrix(y_true, y_pred).tolist(),
        "classification_report": classification_report(y_true, y_pred, output_dict=True)
    }

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

    train_data = lgb.Dataset(X_train, label=y_train)

    # Train
    model = lgb.train(
        params=cfg['model'],
        train_set=train_data,
        num_boost_round=cfg['training']['num_boost_round'],
        valid_sets=valid_sets,
        valid_names=valid_names,
        callbacks=[
            lgb.early_stopping(
                stopping_rounds=cfg['training']['early_stopping_rounds']),
            lgb.log_evaluation(100)
        ]
    )

    # Save model
    model_path = f"models/{args.name}.txt"
    os.makedirs("models", exist_ok=True)
    model.save_model(model_path)

    # Predict
    y_proba = model.predict(X_test)
    y_pred = (y_proba > 0.5).astype(int)

    # Save report and figure
    report_path = f"results/{args.name}_report.json"
    fig_path = f"figures/{args.name}_importance.png"
    evaluate_and_report(
        y_test, y_pred, y_proba,
        feature_names=cfg['data']['feature_cols'],
        model=model,
        report_path=report_path,
        fig_path=fig_path
    )

    print(f"✅ Experiment '{args.name}' completed.")
    print(f"   Model: {model_path}")
    print(f"   Report: {report_path}")
    print(f"   Feature importance: {fig_path}")


if __name__ == "__main__":
    main()
