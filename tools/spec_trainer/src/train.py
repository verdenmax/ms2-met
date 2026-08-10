import pandas as pd
import lightgbm as lgb
import yaml
import json
import argparse
from sklearn.model_selection import train_test_split
from sklearn.metrics import (accuracy_score, average_precision_score,
                             roc_auc_score)
import os
from cv_core import METRIC_SEMANTICS_VERSION, as_error_detection


def load_data(file_paths, feature_cols, target_col):
    dfs = [pd.read_csv(f) for f in file_paths]
    df = pd.concat(dfs, ignore_index=True)
    X = df[feature_cols]
    y = df[target_col]
    return X, y


def evaluate(y_true, y_pred, y_proba=None):
    error_truth = 1 - y_true.astype(int)
    predicted_error = 1 - y_pred.astype(int)
    metrics = {
        "metric_semantics": METRIC_SEMANTICS_VERSION,
        "positive_class": "incorrect_identification",
        "accuracy": accuracy_score(error_truth, predicted_error),
    }
    if y_proba is not None:
        converted_truth, error_score = as_error_detection(y_true, y_proba)
        metrics["roc_auc"] = roc_auc_score(converted_truth, error_score)
        metrics["error_pr_auc"] = average_precision_score(
            converted_truth, error_score)
        metrics["error_threshold"] = 0.5
        metrics["trust_threshold"] = 0.5
    return metrics


def main(config_path):
    with open(config_path) as f:
        cfg = yaml.safe_load(f)

    # Load data
    X_train, y_train = load_data(cfg['data']['train_files'],
                                 cfg['data']['feature_cols'],
                                 cfg['data']['target_col'])

    # Optional: split validation set
    valid_size = cfg['training'].get('valid_size', 0.0)
    if valid_size > 0:
        X_train, X_val, y_train, y_val = train_test_split(
            X_train, y_train, test_size=valid_size, random_state=42, stratify=y_train
        )
        val_set = (X_val, y_val)
    else:
        val_set = None

    # Load test set
    X_test, y_test = load_data(cfg['data']['test_files'],
                               cfg['data']['feature_cols'],
                               cfg['data']['target_col'])

    # Create LightGBM datasets
    train_data = lgb.Dataset(X_train, label=y_train)
    valid_sets = [train_data]
    valid_names = ['train']
    if val_set:
        val_data = lgb.Dataset(
            val_set[0], label=val_set[1], reference=train_data)
        valid_sets.append(val_data)
        valid_names.append('valid')

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
    os.makedirs(os.path.dirname(cfg['output']['model_path']), exist_ok=True)
    model.save_model(cfg['output']['model_path'])

    # Predict & Evaluate
    y_pred = (model.predict(X_test) > 0.5).astype(int)
    y_proba = model.predict(X_test)
    metrics = evaluate(y_test, y_pred, y_proba)

    # Save results
    os.makedirs(os.path.dirname(cfg['output']['result_path']), exist_ok=True)
    with open(cfg['output']['result_path'], 'w') as f:
        json.dump(metrics, f, indent=4)

    print("Evaluation metrics:", metrics)


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", type=str, default="config/exp1.yaml")
    args = parser.parse_args()
    main(args.config)
